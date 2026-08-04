import random
import threading
import time

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql, sqlite

from mlflow.genai.alerts import histogram as hist
from mlflow.genai.alerts.entities import (
    BUCKET_MS,
    AlertInstance,
    AlertRule,
    SeriesKey,
    derive_evaluation_interval_seconds,
)
from mlflow.genai.alerts.evaluator import (
    AlertEvaluator,
    Decision,
    IncrementalMergeCache,
    WindowAccumulator,
    build_due_rule_query,
    decide,
    evaluation_window,
    group_rules,
    is_orphaned,
    may_claim,
    next_due_ms,
    read_signature,
    sticky_worker_index,
)
from mlflow.genai.alerts.rollup_reader import Bucket, FakeRollupReader, aggregate_buckets
from mlflow.genai.alerts.sketch import LOG_SKETCH
from mlflow.genai.alerts.state_machine import OPEN_STATES

MINUTE = 60_000
HOUR = 3_600_000
T0 = (1_700_000_000_000 // BUCKET_MS) * BUCKET_MS

SERIES = SeriesKey(dimension_key="TRACES", experiment_id=7, metric_key="latency")


class InMemoryAlertStore:
    """Everything :class:`AlertEvaluationStore` requires and nothing else.

    Enforces the one-open-instance-per-series rule that the partial unique index
    ``index_alert_instances_open`` enforces on Postgres and SQLite, so a test that
    made the evaluator page twice would fail loudly rather than silently.

    "Open" is ``OPEN_STATES`` -- PENDING and FIRED -- and deliberately not "every
    undismissed state". An INACTIVE instance has recovered and is still waiting to
    be acknowledged, but it must not be handed back or block an insert, or a rule
    that breached, recovered and breached again could never report the second
    episode. Sharing the constant with the state machine is what keeps this fake,
    the real store and the index from drifting apart.

    Locked because the evaluator may now drive it from several threads, and a
    fake whose own dicts raced would produce failures that look like evaluator
    bugs.
    """

    def __init__(self, rules: list[AlertRule] | None = None):
        self.rules = {r.alert_rule_id: r for r in (rules or [])}
        self.instances: dict[str, AlertInstance] = {}
        self.evaluated: list[tuple[str, int, int, int | None]] = []
        self._lock = threading.RLock()

    def lease_due_alert_rules(self, worker_id, now_ms, worker_count, worker_index, limit):
        with self._lock:
            due = [
                r
                for r in self.rules.values()
                if r.enabled
                and r.deleted_at_ms is None
                and (r.next_evaluation_at_ms or 0) <= now_ms
                and may_claim(r, now_ms, worker_count, worker_index)
                and (r.lease_expires_ms is None or r.lease_expires_ms < now_ms)
            ]
            due.sort(key=lambda r: r.next_evaluation_at_ms or 0)
            for rule in due[:limit]:
                rule.lease_owner = worker_id
                rule.lease_expires_ms = now_ms + 60_000
            return due[:limit]

    def get_open_alert_instance(self, alert_rule_id):
        with self._lock:
            return next(
                (
                    i
                    for i in self.instances.values()
                    if i.alert_rule_id == alert_rule_id and i.state in OPEN_STATES
                ),
                None,
            )

    def save_alert_instance(self, instance):
        with self._lock:
            if instance.state in OPEN_STATES:
                clash = next(
                    (
                        i
                        for i in self.instances.values()
                        if i.alert_rule_id == instance.alert_rule_id
                        and i.state in OPEN_STATES
                        and i.alert_instance_id != instance.alert_instance_id
                    ),
                    None,
                )
                if clash is not None:
                    raise AssertionError(f"second open instance for rule {instance.alert_rule_id}")
            self.instances[instance.alert_instance_id] = instance
            return instance

    def record_alert_rule_evaluated(
        self,
        alert_rule_id,
        last_evaluated_ms,
        next_evaluation_at_ms,
        last_sample_count,
        lease_owner=None,
    ):
        with self._lock:
            rule = self.rules.get(alert_rule_id)
            # Same guard as the real store: don't write through someone else's
            # lease, but an unleased rule still records. A test that double-claims
            # then shows up as a missing write rather than a silent double one.
            held_by_other = (
                rule is not None
                and lease_owner is not None
                and rule.lease_owner is not None
                and rule.lease_owner != lease_owner
            )
            if rule is not None and not held_by_other:
                rule.last_evaluated_ms = last_evaluated_ms
                rule.next_evaluation_at_ms = next_evaluation_at_ms
                rule.last_sample_count = last_sample_count
                rule.lease_owner = None
                rule.lease_expires_ms = None
            self.evaluated.append((
                alert_rule_id,
                last_evaluated_ms,
                next_evaluation_at_ms,
                last_sample_count,
            ))

    @property
    def open_instances(self) -> list[AlertInstance]:
        return [i for i in self.instances.values() if i.state in OPEN_STATES]


class CountingReader:
    """Spy that counts reads so grouping can be asserted rather than assumed."""

    def __init__(self, inner: FakeRollupReader):
        self.inner = inner
        self.reads: list[tuple[SeriesKey, int, int]] = []

    def read_buckets(self, series, start_ms, end_ms):
        self.reads.append((series, start_ms, end_ms))
        return self.inner.read_buckets(series, start_ms, end_ms)

    def read_window(self, series, start_ms, end_ms):
        # Delegated rather than forwarded to the inner reader, so a full read still
        # shows up in `reads` and the subclasses below that override `read_buckets`
        # keep working on both paths.
        return self.read_buckets(series, start_ms, end_ms)

    def latest_sealed_bucket_ms(self, series=None):
        return self.inner.latest_sealed_bucket_ms(series)

    def coverage_start_ms(self, series):
        return self.inner.coverage_start_ms(series)

    def sketch_for(self, series):
        return self.inner.sketch_for(series)


class SpyVerifier:
    def __init__(self, exact_count: int):
        self.exact_count = exact_count
        self.calls: list[tuple[SeriesKey, int, int, float]] = []

    def count_above(self, series, start_ms, end_ms, threshold):
        self.calls.append((series, start_ms, end_ms, threshold))
        return self.exact_count


def make_rule(**overrides) -> AlertRule:
    window_seconds = overrides.pop("window_seconds", 3600)
    kwargs = {
        "alert_rule_id": "rule-1",
        "experiment_id": SERIES.experiment_id,
        "name": "Slow checkout responses",
        "metric_key": "latency",
        "dimension_key": "TRACES",
        "aggregation": "PERCENTILE",
        "comparator": "GT",
        "threshold": 45 * MINUTE,
        "window_seconds": window_seconds,
        "evaluation_interval_seconds": derive_evaluation_interval_seconds(window_seconds),
        "percentile_value": 95.0,
        "min_sample_count": 200,
        "next_evaluation_at_ms": 0,
    }
    kwargs.update(overrides)
    return AlertRule(**kwargs)


def seed_range(
    reader: FakeRollupReader,
    start_ms: int,
    end_ms: int,
    count: int,
    value: float,
    series: SeriesKey = SERIES,
):
    reader.seed_constant(series, start_ms, end_ms, count, value)


###############################################################################
# Windows
###############################################################################


@pytest.mark.parametrize("window_seconds", [300, 600, 3600, 86_400, 259_200])
def test_window_derives_from_the_watermark_and_is_bucket_aligned(window_seconds):
    watermark = T0 - BUCKET_MS
    window = evaluation_window(watermark, window_seconds)

    assert window.end_ms == watermark + BUCKET_MS
    assert window.end_ms % BUCKET_MS == 0
    assert window.start_ms % BUCKET_MS == 0
    assert window.span_ms % BUCKET_MS == 0
    assert window.span_ms == window_seconds * 1000


def test_cycle_window_end_is_the_watermark_plus_one_bucket():
    reader = FakeRollupReader()
    seed_range(reader, T0, T0 + 60 * BUCKET_MS, count=5, value=10_000)
    rule = make_rule()
    store = InMemoryAlertStore([rule])

    cycle = AlertEvaluator(reader, store).run_once(now_ms=T0 + 60 * BUCKET_MS)

    assert cycle.window_end_ms == reader.latest_sealed_bucket_ms() + BUCKET_MS
    assert cycle.window_end_ms == T0 + 60 * BUCKET_MS
    observation = cycle.evaluations[0].observation
    assert observation.window_end_ms - observation.window_start_ms == 3600 * 1000


###############################################################################
# Read grouping
###############################################################################


def test_rules_sharing_a_signature_group_together():
    rules = [make_rule(alert_rule_id=f"rule-{i}", threshold=t) for i, t in enumerate([1, 2, 3])]
    rules.append(make_rule(alert_rule_id="other", window_seconds=600))
    rules.append(make_rule(alert_rule_id="other-dim", dimension_value="ERROR"))

    grouped = group_rules(rules)

    assert len(grouped) == 3
    assert len(grouped[read_signature(rules[0])]) == 3


def test_ten_rules_with_different_thresholds_are_one_read():
    reader = FakeRollupReader()
    seed_range(reader, T0, T0 + 60 * BUCKET_MS, count=5, value=10_000)
    spy = CountingReader(reader)
    rules = [make_rule(alert_rule_id=f"rule-{i}", threshold=float(i) * MINUTE) for i in range(10)]
    store = InMemoryAlertStore(rules)

    cycle = AlertEvaluator(spy, store).run_once(now_ms=T0 + 60 * BUCKET_MS)

    assert len(cycle.evaluations) == 10
    assert cycle.groups_read == 1
    assert len(spy.reads) == 1


def test_different_aggregations_over_one_series_still_share_a_read():
    reader = FakeRollupReader()
    seed_range(reader, T0, T0 + 60 * BUCKET_MS, count=5, value=10_000)
    spy = CountingReader(reader)
    rules = [
        make_rule(alert_rule_id="p95", aggregation="PERCENTILE", percentile_value=95.0),
        make_rule(alert_rule_id="avg", aggregation="AVG", percentile_value=None),
        make_rule(alert_rule_id="count", aggregation="COUNT", percentile_value=None),
    ]
    store = InMemoryAlertStore(rules)

    cycle = AlertEvaluator(spy, store).run_once(now_ms=T0 + 60 * BUCKET_MS)

    assert len(spy.reads) == 1
    values = {e.rule.alert_rule_id: e.observation.observed_value for e in cycle.evaluations}
    assert values["count"] == 300.0
    assert values["avg"] == 10_000.0


###############################################################################
# Transitions end to end
###############################################################################


def test_healthy_buckets_produce_no_instance():
    reader = FakeRollupReader()
    seed_range(reader, T0, T0 + 60 * BUCKET_MS, count=5, value=10_000)
    rule = make_rule(sustain_seconds=0)
    store = InMemoryAlertStore([rule])

    cycle = AlertEvaluator(reader, store).run_once(now_ms=T0 + 60 * BUCKET_MS)

    assert store.instances == {}
    assert cycle.notifications == []
    assert cycle.evaluations[0].observation.sample_count == 300
    assert store.rules["rule-1"].last_sample_count == 300


RATE_SERIES = SeriesKey(dimension_key="SPAN_NAME", experiment_id=7, metric_key="error_rate")


def _rate_rule(**overrides) -> AlertRule:
    """A rule over a series whose `sum` is failures and `count` is attempts."""
    kwargs = {
        "metric_key": "error_rate",
        "dimension_key": "SPAN_NAME",
        "aggregation": "AVG",
        "percentile_value": None,
        "comparator": "GT",
        "threshold": 0.10,
        "min_sample_count": 0,
        "sustain_seconds": 0,
        "window_seconds": 3600,
    }
    kwargs.update(overrides)
    return make_rule(**kwargs)


def test_an_error_rate_rule_fires_on_the_ratio_not_the_count():
    """The read path for a rate, end to end.

    Everything else about rates is tested at the aggregation layer. This is the
    part that matters to a user: AVG over (attempts, failures) crosses a threshold
    expressed as a fraction, and opens an instance.
    """
    reader = FakeRollupReader()
    # 20 attempts a bucket, 5 of them failures -- a 25% error rate, over a threshold
    # of 10%. `seed_constant` stores sum = value * count, which is the 5.
    seed_range(reader, T0, T0 + 60 * BUCKET_MS, count=20, value=0.25, series=RATE_SERIES)
    rule = _rate_rule()
    store = InMemoryAlertStore([rule])

    cycle = AlertEvaluator(reader, store).evaluate_rules([rule], now_ms=T0 + 61 * BUCKET_MS)

    (evaluation,) = cycle.evaluations
    assert evaluation.observation.observed_value == pytest.approx(0.25)
    # The sample count is attempts, which is what `min_sample_count` gates on -- not
    # the number of failures.
    assert evaluation.observation.sample_count == 20 * 60
    assert evaluation.transition.event == "FIRED"


def test_an_error_rate_below_its_threshold_does_not_fire():
    reader = FakeRollupReader()
    # 1 failure in 20 attempts: 5%, under the 10% threshold.
    seed_range(reader, T0, T0 + 60 * BUCKET_MS, count=20, value=0.05, series=RATE_SERIES)
    rule = _rate_rule()
    store = InMemoryAlertStore([rule])

    cycle = AlertEvaluator(reader, store).evaluate_rules([rule], now_ms=T0 + 61 * BUCKET_MS)

    assert cycle.evaluations[0].observation.observed_value == pytest.approx(0.05)
    assert store.open_instances == []


def test_a_rate_over_an_empty_window_is_no_data_not_zero_percent():
    """0/0 is undefined, and reporting it as 0% would be a lie in the safe direction.

    It also means a rate rule cannot fire during a total outage -- nothing was
    called, so nothing failed. Catching that is an absence rule's job, which is
    exactly why COUNT and AVG diverge on an empty window.
    """
    reader = FakeRollupReader(latest_sealed_bucket_ms=T0 + 60 * BUCKET_MS)
    reader.set_coverage_start(RATE_SERIES, 0)
    rule = _rate_rule(comparator="LT", threshold=0.99)
    store = InMemoryAlertStore([rule])

    cycle = AlertEvaluator(reader, store).evaluate_rules([rule], now_ms=T0 + 61 * BUCKET_MS)

    (evaluation,) = cycle.evaluations
    assert evaluation.observation.observed_value is None
    # A "rate below 99%" rule would fire on a 0% reading; no-data cannot open an
    # instance, which is the behaviour that keeps an idle window quiet.
    assert evaluation.transition.event == "NONE"
    assert store.open_instances == []


def test_breaching_buckets_go_pending_then_fired_and_notify_once():
    """The headline claim: these buckets, this transition, exactly these events."""
    reader = FakeRollupReader()
    seed_range(reader, T0, T0 + 60 * BUCKET_MS, count=5, value=90 * MINUTE)
    rule = make_rule(sustain_seconds=300, window_seconds=3600)
    store = InMemoryAlertStore([rule])
    evaluator = AlertEvaluator(reader, store)

    events = []
    notifications = 0
    for step in range(6):
        now = T0 + (60 + step) * BUCKET_MS
        seed_range(reader, T0 + (60 + step) * BUCKET_MS, now + BUCKET_MS, 5, 90 * MINUTE)
        rule.next_evaluation_at_ms = now
        cycle = evaluator.run_once(now_ms=now)
        events.append(cycle.evaluations[0].transition.event)
        notifications += len(cycle.notifications)

    assert events == ["OPENED", "UPDATED", "UPDATED", "UPDATED", "UPDATED", "FIRED"]
    assert notifications == 1
    instance = store.open_instances[0]
    assert instance.state == "FIRED"
    assert instance.fired_at_ms == T0 + 65 * BUCKET_MS
    assert instance.threshold == 45 * MINUTE
    # The sketch reports the value itself, within 2%, rather than the containing
    # bucket's upper edge -- which used to turn a 90-minute p95 into "2 hours".
    assert instance.peak_value == pytest.approx(90 * MINUTE, rel=0.02)


def test_instance_stays_open_after_one_healthy_evaluation():
    """One healthy window is a blip. Recovery has to be sustained to close it."""
    reader = FakeRollupReader()
    seed_range(reader, T0, T0 + 60 * BUCKET_MS, count=5, value=90 * MINUTE)
    rule = make_rule(sustain_seconds=0)
    store = InMemoryAlertStore([rule])
    evaluator = AlertEvaluator(reader, store)

    fired = evaluator.run_once(now_ms=T0 + 60 * BUCKET_MS)
    assert fired.evaluations[0].transition.event == "FIRED"

    # A whole fresh window of healthy traffic.
    seed_range(reader, T0 + 60 * BUCKET_MS, T0 + 120 * BUCKET_MS, count=5, value=10_000)
    rule.next_evaluation_at_ms = T0 + 120 * BUCKET_MS
    recovered = evaluator.run_once(now_ms=T0 + 120 * BUCKET_MS)

    assert recovered.evaluations[0].transition.event == "UPDATED"
    assert store.open_instances[0].state == "FIRED"
    assert store.open_instances[0].peak_value == pytest.approx(90 * MINUTE, rel=0.02)
    assert recovered.notifications == []


def test_a_sustained_recovery_frees_the_rule_to_report_the_next_incident():
    """Breach, sustained recovery, breach again: two instances, two notifications.

    The end-to-end version of the INACTIVE contract. The first episode stays on
    record with its peak intact, and the second one pages on its own.
    """
    reader = FakeRollupReader()
    seed_range(reader, T0, T0 + 60 * BUCKET_MS, count=5, value=90 * MINUTE)
    rule = make_rule(sustain_seconds=0)
    store = InMemoryAlertStore([rule])
    evaluator = AlertEvaluator(reader, store)
    notifications = []

    def evaluate(at_bucket: int) -> str:
        rule.next_evaluation_at_ms = T0 + at_bucket * BUCKET_MS
        cycle = evaluator.run_once(now_ms=T0 + at_bucket * BUCKET_MS)
        notifications.extend(cycle.notifications)
        return cycle.evaluations[0].transition.event

    assert evaluate(60) == "FIRED"

    # Two healthy windows in a row: the first arms the recovery, the second closes it.
    seed_range(reader, T0 + 60 * BUCKET_MS, T0 + 180 * BUCKET_MS, count=5, value=10_000)
    assert evaluate(120) == "UPDATED"
    assert evaluate(180) == "RECOVERED"

    # Breaching again. The INACTIVE instance neither suppresses nor absorbs it.
    seed_range(reader, T0 + 180 * BUCKET_MS, T0 + 240 * BUCKET_MS, count=5, value=90 * MINUTE)
    assert evaluate(240) == "FIRED"

    first, second = sorted(store.instances.values(), key=lambda i: i.started_at_ms)
    assert first.state == "INACTIVE"
    assert first.peak_value == pytest.approx(90 * MINUTE, rel=0.02)
    assert first.dismissed_at_ms is None
    assert second.state == "FIRED"
    assert second.alert_instance_id != first.alert_instance_id
    assert [n.alert_instance_id for n in notifications] == [
        first.alert_instance_id,
        second.alert_instance_id,
    ]


def test_a_gap_bucket_does_not_read_as_healthy():
    reader = FakeRollupReader()
    seed_range(reader, T0, T0 + 60 * BUCKET_MS, count=5, value=90 * MINUTE)
    rule = make_rule(sustain_seconds=0)
    store = InMemoryAlertStore([rule])
    evaluator = AlertEvaluator(reader, store)
    evaluator.run_once(now_ms=T0 + 60 * BUCKET_MS)

    seed_range(reader, T0 + 60 * BUCKET_MS, T0 + 119 * BUCKET_MS, count=5, value=10_000)
    reader.seed(SERIES, Bucket(bucket_start_ms=T0 + 119 * BUCKET_MS, count=0, is_gap=True))
    rule.next_evaluation_at_ms = T0 + 120 * BUCKET_MS
    cycle = evaluator.run_once(now_ms=T0 + 120 * BUCKET_MS)

    assert cycle.evaluations[0].observation.observed_value is None
    assert cycle.evaluations[0].transition.event == "NONE"
    assert store.open_instances[0].state == "FIRED"


def test_too_few_samples_leaves_the_rule_without_an_instance():
    reader = FakeRollupReader()
    seed_range(reader, T0, T0 + 60 * BUCKET_MS, count=1, value=90 * MINUTE)
    rule = make_rule(sustain_seconds=0, min_sample_count=200)
    store = InMemoryAlertStore([rule])

    cycle = AlertEvaluator(reader, store).run_once(now_ms=T0 + 60 * BUCKET_MS)

    assert cycle.evaluations[0].observation.sample_count == 60
    assert cycle.evaluations[0].transition.event == "NONE"
    assert store.instances == {}
    assert store.rules["rule-1"].last_sample_count == 60


def test_notifier_runs_after_the_instance_is_committed():
    reader = FakeRollupReader()
    seed_range(reader, T0, T0 + 60 * BUCKET_MS, count=5, value=90 * MINUTE)
    rule = make_rule(sustain_seconds=0)
    store = InMemoryAlertStore([rule])
    seen: list[str] = []

    def notifier(notified_rule, instance):
        # Durability comes from ordering: the row is already committed here.
        assert store.instances[instance.alert_instance_id].state == "FIRED"
        seen.append(notified_rule.alert_rule_id)

    AlertEvaluator(reader, store, notifier=notifier).run_once(now_ms=T0 + 60 * BUCKET_MS)

    assert seen == ["rule-1"]


###############################################################################
# Bounds, then verify
###############################################################################


def _hist_of(values: list[float]) -> list[int]:
    """Stored form: interleaved (bucket index, count) pairs."""
    h = hist.empty()
    for v in values:
        hist.observe(h, v, LOG_SKETCH)
    return hist.to_pairs(h)


def _threshold_inside_bucket_of(value: float) -> float:
    """A threshold in the same sketch bucket as ``value`` -- the ambiguous case.

    Computed rather than hardcoded because the sketch's buckets are ~4% wide: two
    numbers a human would consider "close" (35 and 40 minutes) land in different
    buckets and are answered exactly. Only a threshold within a couple of percent
    of the observations is genuinely undecidable, which is the point.
    """
    return LOG_SKETCH.estimate(LOG_SKETCH.index(value))


def test_on_boundary_threshold_decides_with_no_raw_query():
    reader = FakeRollupReader()
    seed_range(reader, T0, T0 + 60 * BUCKET_MS, count=5, value=90 * MINUTE)
    rule = make_rule(sustain_seconds=0, threshold=45 * MINUTE)
    store = InMemoryAlertStore([rule])
    verifier = SpyVerifier(exact_count=0)

    cycle = AlertEvaluator(reader, store, verifier=verifier).run_once(now_ms=T0 + 60 * BUCKET_MS)
    decision = cycle.evaluations[0].decision

    assert decision.breaching is True
    assert decision.exact is True
    assert decision.verified is False
    assert verifier.calls == []


def test_mid_bucket_threshold_falls_back_to_the_raw_query():
    reader = FakeRollupReader()
    # Every sample lands in the bucket the threshold cuts through, so the bounds
    # are [0, 300] and cannot decide "more than 5% above it" on their own.
    threshold = _threshold_inside_bucket_of(35 * MINUTE)
    seed_range(reader, T0, T0 + 60 * BUCKET_MS, count=5, value=35 * MINUTE)
    rule = make_rule(sustain_seconds=0, threshold=threshold)
    store = InMemoryAlertStore([rule])
    verifier = SpyVerifier(exact_count=0)

    cycle = AlertEvaluator(reader, store, verifier=verifier).run_once(now_ms=T0 + 60 * BUCKET_MS)
    decision = cycle.evaluations[0].decision

    assert decision.verified is True
    assert decision.exact is True
    assert decision.breaching is False
    assert len(verifier.calls) == 1
    _, start_ms, end_ms, verified_threshold = verifier.calls[0]
    assert (start_ms, end_ms) == (T0, T0 + 60 * BUCKET_MS)
    assert verified_threshold == threshold
    assert store.instances == {}


def test_raw_verification_can_flip_an_ambiguous_case_into_a_breach():
    reader = FakeRollupReader()
    seed_range(reader, T0, T0 + 60 * BUCKET_MS, count=5, value=35 * MINUTE)
    rule = make_rule(sustain_seconds=0, threshold=_threshold_inside_bucket_of(35 * MINUTE))
    store = InMemoryAlertStore([rule])
    verifier = SpyVerifier(exact_count=100)

    cycle = AlertEvaluator(reader, store, verifier=verifier).run_once(now_ms=T0 + 60 * BUCKET_MS)

    assert cycle.evaluations[0].decision.breaching is True
    assert len(verifier.calls) == 1
    assert store.open_instances[0].state == "FIRED"


def test_ambiguous_bounds_without_a_verifier_are_marked_inexact():
    accumulator = WindowAccumulator().add([
        Bucket(
            bucket_start_ms=T0,
            count=300,
            sum=0.0,
            histogram=_hist_of([35 * MINUTE] * 300),
            boundaries_version=hist.SKETCH_VERSION,
        )
    ])
    window = evaluation_window(T0, 3600)
    rule = make_rule(threshold=_threshold_inside_bucket_of(35 * MINUTE))
    observation = accumulator.to_observation("PERCENTILE", window, 95.0, LOG_SKETCH)

    decision = decide(rule, observation, accumulator, spec=LOG_SKETCH)

    assert decision.exact is False
    assert decision.verified is False


def test_non_percentile_rules_never_consult_the_verifier():
    reader = FakeRollupReader()
    seed_range(reader, T0, T0 + 60 * BUCKET_MS, count=5, value=10_000)
    rule = make_rule(aggregation="COUNT", percentile_value=None, comparator="GT", threshold=100)
    store = InMemoryAlertStore([rule])
    verifier = SpyVerifier(exact_count=0)

    cycle = AlertEvaluator(reader, store, verifier=verifier).run_once(now_ms=T0 + 60 * BUCKET_MS)

    assert cycle.evaluations[0].decision == Decision(breaching=True, exact=True, verified=False)
    assert verifier.calls == []


###############################################################################
# Incremental merge
###############################################################################


def _seed_random_bucket(reader: FakeRollupReader, bucket_start_ms: int, rng: random.Random):
    if rng.random() < 0.08:
        reader.seed(SERIES, Bucket(bucket_start_ms=bucket_start_ms, count=0, is_gap=True))
        return
    count = rng.randint(0, 6)
    h = hist.empty()
    total = 0.0
    for _ in range(count):
        value = float(rng.choice([10, 100, 1_000, 30_000, 300_000, 2_700_000, 7_200_000]))
        hist.observe(h, value, LOG_SKETCH)
        total += value
    reader.seed(
        SERIES,
        Bucket(
            bucket_start_ms=bucket_start_ms,
            count=count,
            sum=total,
            histogram=hist.to_pairs(h),
            boundaries_version=hist.SKETCH_VERSION,
        ),
    )


def _assert_same_observation(a, b):
    assert a.sample_count == b.sample_count
    assert a.window_start_ms == b.window_start_ms
    assert a.window_end_ms == b.window_end_ms
    if a.observed_value is None or b.observed_value is None:
        assert a.observed_value is None
        assert b.observed_value is None
    else:
        assert a.observed_value == pytest.approx(b.observed_value, rel=1e-12, abs=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("window_seconds", [600, 3600])
@pytest.mark.parametrize("interval_seconds", [60, 300])
@pytest.mark.parametrize("aggregation", ["COUNT", "SUM", "AVG", "PERCENTILE"])
def test_incremental_merge_matches_full_recompute(
    seed, window_seconds, interval_seconds, aggregation
):
    """The test that matters most: the 288x optimization must be a no-op on results.

    Slides a window across a randomly seeded series -- including gap buckets, which
    have to *leave* the window as well as enter it -- and asserts the merged
    accumulator agrees with both a cold full read and the independent
    ``aggregate_buckets`` implementation at every step.
    """
    rng = random.Random(seed)
    reader = FakeRollupReader()
    window_buckets = window_seconds // 60
    step_buckets = interval_seconds // 60

    for i in range(window_buckets * 2):
        _seed_random_bucket(reader, T0 + i * BUCKET_MS, rng)

    rule = make_rule(
        window_seconds=window_seconds,
        evaluation_interval_seconds=interval_seconds,
        aggregation=aggregation,
        percentile_value=95.0 if aggregation == "PERCENTILE" else None,
    )
    signature = read_signature(rule)
    warm = IncrementalMergeCache()
    cold = IncrementalMergeCache(enabled=False)

    next_bucket = window_buckets * 2
    for _ in range(40):
        for _ in range(step_buckets):
            _seed_random_bucket(reader, T0 + next_bucket * BUCKET_MS, rng)
            next_bucket += 1
        now_ms = reader.latest_sealed_bucket_ms() + BUCKET_MS
        window = evaluation_window(reader.latest_sealed_bucket_ms(), window_seconds)

        merged = warm.accumulator_for(signature, window, reader, now_ms).to_observation(
            aggregation, window, rule.percentile_value, LOG_SKETCH
        )
        full = cold.accumulator_for(signature, window, reader, now_ms).to_observation(
            aggregation, window, rule.percentile_value, LOG_SKETCH
        )
        independent = aggregate_buckets(
            reader.read_buckets(SERIES, window.start_ms, window.end_ms),
            aggregation,
            window.start_ms,
            window.end_ms,
            rule.percentile_value,
            LOG_SKETCH,
        )

        _assert_same_observation(merged, full)
        _assert_same_observation(merged, independent)


def test_incremental_merge_reads_only_the_delta():
    reader = FakeRollupReader()
    seed_range(reader, T0, T0 + 1440 * BUCKET_MS, count=1, value=10_000)
    spy = CountingReader(reader)
    rule = make_rule(window_seconds=86_400, sustain_seconds=0, threshold=HOUR)
    store = InMemoryAlertStore([rule])
    evaluator = AlertEvaluator(spy, store)

    evaluator.run_once(now_ms=T0 + 1440 * BUCKET_MS)
    first_read = spy.reads[0]
    assert first_read[2] - first_read[1] == 86_400 * 1000

    spy.reads.clear()
    seed_range(reader, T0 + 1440 * BUCKET_MS, T0 + 1445 * BUCKET_MS, count=1, value=10_000)
    rule.next_evaluation_at_ms = T0 + 1445 * BUCKET_MS
    evaluator.run_once(now_ms=T0 + 1445 * BUCKET_MS)

    # Five buckets in, five buckets out -- 1,435 of 1,440 were never re-read.
    assert sorted(end - start for _, start, end in spy.reads) == [
        5 * BUCKET_MS,
        5 * BUCKET_MS,
    ]


def test_an_unchanged_window_costs_no_read_at_all():
    reader = FakeRollupReader()
    seed_range(reader, T0, T0 + 60 * BUCKET_MS, count=5, value=10_000)
    spy = CountingReader(reader)
    rule = make_rule(sustain_seconds=0)
    store = InMemoryAlertStore([rule])
    evaluator = AlertEvaluator(spy, store)

    evaluator.run_once(now_ms=T0 + 60 * BUCKET_MS)
    spy.reads.clear()
    rule.next_evaluation_at_ms = T0 + 60 * BUCKET_MS
    evaluator.run_once(now_ms=T0 + 60 * BUCKET_MS + 1)

    assert spy.reads == []


def test_hourly_full_recompute_rebuilds_the_window():
    reader = FakeRollupReader()
    seed_range(reader, T0, T0 + 60 * BUCKET_MS, count=5, value=10_000)
    spy = CountingReader(reader)
    cache = IncrementalMergeCache(full_recompute_interval_ms=HOUR)
    signature = read_signature(make_rule())

    window = evaluation_window(T0 + 59 * BUCKET_MS, 3600)
    cache.accumulator_for(signature, window, spy, T0)
    spy.reads.clear()

    later = evaluation_window(T0 + 60 * BUCKET_MS, 3600)
    cache.accumulator_for(signature, later, spy, T0 + HOUR)

    assert len(spy.reads) == 1
    _, start_ms, end_ms = spy.reads[0]
    assert end_ms - start_ms == 3600 * 1000


def test_a_window_that_moved_past_itself_falls_back_to_a_full_read():
    reader = FakeRollupReader()
    seed_range(reader, T0, T0 + 200 * BUCKET_MS, count=1, value=10_000)
    spy = CountingReader(reader)
    cache = IncrementalMergeCache()
    signature = read_signature(make_rule(window_seconds=600))

    cache.accumulator_for(signature, evaluation_window(T0 + 10 * BUCKET_MS, 600), spy, T0)
    spy.reads.clear()
    cache.accumulator_for(signature, evaluation_window(T0 + 100 * BUCKET_MS, 600), spy, T0)

    assert len(spy.reads) == 1
    _, start_ms, end_ms = spy.reads[0]
    assert end_ms - start_ms == 600 * 1000


def test_editing_the_window_changes_the_cache_key_and_forces_a_reread():
    reader = FakeRollupReader()
    seed_range(reader, T0, T0 + 200 * BUCKET_MS, count=1, value=10_000)
    spy = CountingReader(reader)
    cache = IncrementalMergeCache()

    window_a = evaluation_window(T0 + 100 * BUCKET_MS, 600)
    cache.accumulator_for(read_signature(make_rule(window_seconds=600)), window_a, spy, T0)
    spy.reads.clear()

    window_b = evaluation_window(T0 + 100 * BUCKET_MS, 1200)
    cache.accumulator_for(read_signature(make_rule(window_seconds=1200)), window_b, spy, T0)

    assert len(spy.reads) == 1
    _, start_ms, end_ms = spy.reads[0]
    assert end_ms - start_ms == 1200 * 1000


def test_accumulator_add_and_remove_are_inverses():
    buckets = [
        Bucket(
            bucket_start_ms=T0 + i * BUCKET_MS,
            count=3,
            sum=300.0,
            histogram=_hist_of([100.0, 100.0, 100.0]),
        )
        for i in range(5)
    ]
    accumulator = WindowAccumulator().add(buckets)
    accumulator.remove(buckets)

    assert accumulator.count == 0
    assert accumulator.sum_value == 0.0
    assert accumulator.histogram == hist.empty()


@pytest.mark.parametrize("aggregation", ["COUNT", "SUM", "AVG", "PERCENTILE"])
@pytest.mark.parametrize("has_gap", [False, True])
def test_the_two_read_paths_agree_on_an_empty_window(aggregation: str, has_gap: bool):
    """`aggregate_buckets` and `WindowAccumulator` must project identically.

    They were separate implementations of one contract and they disagreed: the
    accumulator -- the path the evaluator actually uses -- returned no-data on
    `count == 0` for every aggregation, while `aggregate_buckets`, which nothing
    in production called, correctly returned 0 for COUNT.

    The consequence was that a rule whose signal is *absence* ("requests < 10",
    "traffic dropped") went silent at exactly the moment it should fire. Confirmed
    against a live stack before this was fixed.
    """
    window = evaluation_window(T0 + BUCKET_MS, 120)
    buckets = [Bucket(bucket_start_ms=T0, count=0, is_gap=True)] if has_gap else []

    from_accumulator = (
        WindowAccumulator()
        .add(buckets)
        .to_observation(aggregation, window, percentile_value=95, spec=LOG_SKETCH)
    )
    from_buckets = aggregate_buckets(
        buckets, aggregation, window.start_ms, window.end_ms, percentile_value=95, spec=LOG_SKETCH
    )

    assert from_accumulator == from_buckets

    if aggregation == "COUNT" and not has_gap:
        # An empty window genuinely counted zero things.
        assert from_accumulator.observed_value == 0.0
        assert from_accumulator.has_data
    else:
        # The average or p95 of nothing is not a number, and a recorded gap means
        # aggregation did not run -- neither is an answer.
        assert from_accumulator.observed_value is None
        assert not from_accumulator.has_data


def test_an_absence_rule_fires_when_traffic_stops():
    """The end-to-end shape of the bug above, at the evaluator level."""
    reader = FakeRollupReader()
    reader.seed_constant(SERIES, T0, T0 + 10 * MINUTE, count=50, value=1_000)
    rule = make_rule(
        alert_rule_id="traffic-dropped",
        aggregation="COUNT",
        percentile_value=None,
        comparator="LT",
        threshold=5,
        window_seconds=600,
        min_sample_count=0,
        sustain_seconds=0,
    )
    store = InMemoryAlertStore([rule])
    evaluator = AlertEvaluator(reader, store)

    # Healthy: 500 requests in the window is not below 5.
    evaluator.evaluate_rules([rule], now_ms=T0 + 10 * MINUTE)
    assert store.instances == {}

    # Traffic stops. Buckets simply stop being written -- no gap markers, because
    # the aggregator is keeping up; it is a genuinely quiet period.
    reader._latest_sealed_bucket_ms = T0 + 20 * MINUTE
    cycle = evaluator.evaluate_rules([rule], now_ms=T0 + 21 * MINUTE)

    (evaluation,) = cycle.evaluations
    assert evaluation.observation.observed_value == 0.0
    assert evaluation.transition.event == "FIRED"


def test_an_absence_rule_does_not_fire_over_a_window_that_predates_aggregation():
    """A fresh install must not report "traffic dropped" before it started looking.

    The window of a 10-minute absence rule on a server that booted two minutes ago
    is mostly a period nothing was aggregating. An empty result there is not
    evidence of anything, so it must read as no-data -- exactly like a gap -- rather
    than as a genuine zero.
    """
    reader = FakeRollupReader(latest_sealed_bucket_ms=T0 + 10 * MINUTE)
    # Aggregation only began 3 minutes ago; the 10-minute window reaches back well
    # before that.
    reader.set_coverage_start(SERIES, T0 + 8 * MINUTE)
    rule = make_rule(
        alert_rule_id="traffic-dropped",
        aggregation="COUNT",
        percentile_value=None,
        comparator="LT",
        threshold=5,
        window_seconds=600,
        min_sample_count=0,
        sustain_seconds=0,
    )
    store = InMemoryAlertStore([rule])

    cycle = AlertEvaluator(reader, store).evaluate_rules([rule], now_ms=T0 + 11 * MINUTE)

    (evaluation,) = cycle.evaluations
    assert evaluation.observation.observed_value is None
    assert evaluation.transition.event == "NONE"
    assert store.instances == {}
    # The rule is still being evaluated -- it must not look stalled, which is
    # indistinguishable from a dead scheduler.
    assert store.evaluated


def test_an_absence_rule_fires_once_the_window_is_fully_covered():
    """The other half: coverage must not suppress a real drop forever."""
    reader = FakeRollupReader(latest_sealed_bucket_ms=T0 + 20 * MINUTE)
    reader.set_coverage_start(SERIES, T0 + 8 * MINUTE)
    rule = make_rule(
        alert_rule_id="traffic-dropped",
        aggregation="COUNT",
        percentile_value=None,
        comparator="LT",
        threshold=5,
        window_seconds=600,
        min_sample_count=0,
        sustain_seconds=0,
    )
    store = InMemoryAlertStore([rule])

    # Window is now [11min, 21min), entirely after coverage began.
    cycle = AlertEvaluator(reader, store).evaluate_rules([rule], now_ms=T0 + 21 * MINUTE)

    (evaluation,) = cycle.evaluations
    assert evaluation.observation.observed_value == 0.0
    assert evaluation.transition.event == "FIRED"


def test_a_gap_still_reads_as_no_data_not_as_zero():
    """The distinction the gap marker exists for.

    Zero traffic is an answer; a gap means aggregation did not run, and treating
    it as zero would let an outage fire an absence rule spuriously.
    """
    reader = FakeRollupReader()
    reader.seed(SERIES, Bucket(bucket_start_ms=T0, count=0, is_gap=True))
    rule = make_rule(
        alert_rule_id="traffic-dropped",
        aggregation="COUNT",
        percentile_value=None,
        comparator="LT",
        threshold=5,
        window_seconds=300,
        min_sample_count=0,
        sustain_seconds=0,
    )
    store = InMemoryAlertStore([rule])
    cycle = AlertEvaluator(reader, store).evaluate_rules([rule], now_ms=T0 + BUCKET_MS)

    (evaluation,) = cycle.evaluations
    assert evaluation.observation.observed_value is None
    assert evaluation.transition.event == "NONE"
    assert store.instances == {}


def test_a_gap_leaves_the_window_when_it_expires():
    accumulator = WindowAccumulator()
    gap = Bucket(bucket_start_ms=T0, count=0, is_gap=True)
    accumulator.add([gap, Bucket(bucket_start_ms=T0 + BUCKET_MS, count=4, sum=400.0)])
    window = evaluation_window(T0 + BUCKET_MS, 120)

    assert accumulator.to_observation("COUNT", window).observed_value is None

    accumulator.remove([gap])
    assert accumulator.to_observation("COUNT", window).observed_value == 4.0


###############################################################################
# Concurrency
#
# None of this was covered before: `worker_count > 1` was tested only as
# arithmetic, and no test had ever started a thread.
###############################################################################


def _many_group_rules(count: int) -> list[AlertRule]:
    """Rules that land in distinct read groups, so they fan out across threads."""
    return [
        make_rule(
            alert_rule_id=f"rule-{i}",
            dimension_key="SPAN_NAME",
            dimension_value=f"tool-{i}",
            next_evaluation_at_ms=T0,
        )
        for i in range(count)
    ]


def _seed_all(reader: FakeRollupReader, rules: list[AlertRule], **kwargs) -> None:
    for rule in rules:
        seed_range(reader, T0, T0 + 60 * BUCKET_MS, series=read_signature(rule).series, **kwargs)


@pytest.mark.parametrize("max_workers", [1, 4])
def test_threads_produce_the_same_cycle_as_one_thread(max_workers):
    """Fan-out must change only how long it takes, never what it decides."""
    rules = _many_group_rules(8)
    reader = FakeRollupReader(latest_sealed_bucket_ms=T0 + 60 * BUCKET_MS)
    _seed_all(reader, rules, count=5, value=50 * MINUTE)
    store = InMemoryAlertStore(rules)

    cycle = AlertEvaluator(reader, store, max_workers=max_workers).evaluate_rules(
        rules, now_ms=T0 + 61 * BUCKET_MS
    )

    # Submission order, not completion order: the cycle must not depend on which
    # thread happened to finish first.
    assert [e.rule.alert_rule_id for e in cycle.evaluations] == [r.alert_rule_id for r in rules]
    assert cycle.groups_read == 8
    assert len(store.open_instances) == 8


def test_one_failing_group_does_not_cost_the_others_their_evaluation():
    """A single bad rule used to abort the cycle and leave every other rule unread."""
    rules = _many_group_rules(4)
    reader = FakeRollupReader(latest_sealed_bucket_ms=T0 + 60 * BUCKET_MS)
    _seed_all(reader, rules, count=5, value=50 * MINUTE)

    class ExplodingStore(InMemoryAlertStore):
        def get_open_alert_instance(self, alert_rule_id):
            if alert_rule_id == "rule-2":
                raise RuntimeError("transient database failure")
            return super().get_open_alert_instance(alert_rule_id)

    store = ExplodingStore(rules)
    cycle = AlertEvaluator(reader, store, max_workers=4).evaluate_rules(
        rules, now_ms=T0 + 61 * BUCKET_MS
    )

    assert [e.rule.alert_rule_id for e in cycle.evaluations] == ["rule-0", "rule-1", "rule-3"]
    assert len(store.open_instances) == 3


def test_a_failing_read_drops_only_its_own_group_from_the_window():
    rules = _many_group_rules(3)
    reader = FakeRollupReader(latest_sealed_bucket_ms=T0 + 60 * BUCKET_MS)
    _seed_all(reader, rules, count=5, value=50 * MINUTE)

    class UnreadableSeries(CountingReader):
        def read_buckets(self, series, start_ms, end_ms):
            if series.dimension_value == "tool-1":
                raise RuntimeError("rollup read failed")
            return super().read_buckets(series, start_ms, end_ms)

    store = InMemoryAlertStore(rules)
    cycle = AlertEvaluator(UnreadableSeries(reader), store, max_workers=3).evaluate_rules(
        rules, now_ms=T0 + 61 * BUCKET_MS
    )

    assert [e.rule.alert_rule_id for e in cycle.evaluations] == ["rule-0", "rule-2"]
    # The failed group contributes no watermark rather than dragging the cycle's
    # window back to zero.
    assert cycle.window_end_ms == T0 + 61 * BUCKET_MS


def test_a_cold_signature_is_read_once_however_many_threads_want_it():
    """The cache is one object shared by every thread.

    `accumulator_for` is a read-modify-write around the reads it caches, so
    without per-entry locking every thread that arrives while the entry is cold
    sees `None` and performs its own full window read. Serializing on the entry
    turns that into one read plus N cache hits -- which for a 24h rule is the
    difference between 1,440 rows and 1,440 x N.
    """
    cache = IncrementalMergeCache()
    inner = FakeRollupReader(latest_sealed_bucket_ms=T0 + 600 * BUCKET_MS)
    seed_range(inner, T0, T0 + 600 * BUCKET_MS, count=1, value=10 * MINUTE)

    class SlowReader(CountingReader):
        """Slow enough that every thread would reach the cold path unserialized.

        Without the delay the first thread finishes before the rest are scheduled,
        and the test passes whether or not the lock is there.
        """

        def read_buckets(self, series, start_ms, end_ms):
            time.sleep(0.05)
            return super().read_buckets(series, start_ms, end_ms)

    reader = SlowReader(inner)
    signature = read_signature(make_rule(window_seconds=3600))
    window = evaluation_window(T0 + 600 * BUCKET_MS, 3600)

    counts = []
    lock = threading.Lock()
    barrier = threading.Barrier(8)

    def run():
        barrier.wait()
        accumulator = cache.accumulator_for(signature, window, reader, now_ms=T0)
        with lock:
            counts.append(accumulator.count)

    threads = [threading.Thread(target=run) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(reader.reads) == 1
    # And every thread got the same answer out of it.
    assert len(set(counts)) == 1


def test_one_rule_is_leased_once_even_when_several_workers_race():
    """The claim is the atomic step, so a rule is evaluated by exactly one caller."""
    rules = _many_group_rules(6)
    store = InMemoryAlertStore(rules)
    now = T0 + 61 * BUCKET_MS
    claimed: list[list[AlertRule]] = []
    lock = threading.Lock()
    barrier = threading.Barrier(4)

    def claim(worker_index):
        barrier.wait()
        got = store.lease_due_alert_rules(
            worker_id=f"worker-{worker_index}",
            now_ms=now,
            worker_count=1,
            worker_index=0,
            limit=50,
        )
        with lock:
            claimed.append(got)

    threads = [threading.Thread(target=claim, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    all_ids = [r.alert_rule_id for batch in claimed for r in batch]
    assert sorted(all_ids) == sorted(r.alert_rule_id for r in rules)


###############################################################################
# Leasing
###############################################################################


def test_sticky_assignment_is_stable_and_spreads_rules():
    ids = [f"rule-{i}" for i in range(200)]
    first = [sticky_worker_index(i, 4) for i in ids]
    second = [sticky_worker_index(i, 4) for i in ids]

    assert first == second
    assert set(first) == {0, 1, 2, 3}
    assert all(sticky_worker_index(i, 1) == 0 for i in ids)


def test_a_rule_two_intervals_overdue_may_be_claimed_by_anyone():
    rule = make_rule(evaluation_interval_seconds=300)
    now = T0

    rule.next_evaluation_at_ms = now - 300_000
    assert is_orphaned(rule, now) is False

    rule.next_evaluation_at_ms = now - 601_000
    assert is_orphaned(rule, now) is True
    assert all(may_claim(rule, now, 4, index) for index in range(4))


def test_a_rule_on_time_is_only_claimed_by_its_owner():
    rule = make_rule(evaluation_interval_seconds=300, next_evaluation_at_ms=T0)
    owner = sticky_worker_index(rule.alert_rule_id, 4)

    claims = [may_claim(rule, T0, 4, index) for index in range(4)]

    assert claims.count(True) == 1
    assert claims[owner] is True


def test_postgres_lease_query_pushes_stickiness_and_skip_locked_into_sql():
    query = build_due_rule_query("postgresql", T0, worker_count=4, worker_index=2, limit=50)
    compiled = str(query.compile(dialect=postgresql.dialect()))

    assert "hashtext" in compiled
    # Cast before abs(): abs(int4 -2147483648) overflows and would abort the scan.
    assert "abs(CAST(hashtext" in compiled
    assert "FOR UPDATE" in compiled
    assert "SKIP LOCKED" in compiled
    assert "next_evaluation_at_ms" in compiled
    assert "deleted_at_ms IS NULL" in compiled


def test_sqlite_lease_query_omits_the_postgres_only_constructs():
    query = build_due_rule_query("sqlite", T0, worker_count=4, worker_index=2, limit=50)
    compiled = str(query.compile(dialect=sqlite.dialect()))

    assert "hashtext" not in compiled
    assert "FOR UPDATE" not in compiled
    assert "ORDER BY" in compiled


def test_lease_query_orders_most_overdue_first_so_nothing_starves():
    query = build_due_rule_query("sqlite", T0, worker_count=1, worker_index=0, limit=50)
    order_by = [str(c) for c in query._order_by_clauses]

    assert order_by == ["alert_rules.next_evaluation_at_ms"]
    assert isinstance(query, sa.sql.Select)


def test_only_due_undeleted_enabled_rules_are_leased():
    reader = FakeRollupReader()
    seed_range(reader, T0, T0 + 60 * BUCKET_MS, count=5, value=10_000)
    now = T0 + 60 * BUCKET_MS
    rules = [
        make_rule(alert_rule_id="due", next_evaluation_at_ms=now - 1),
        make_rule(alert_rule_id="not-yet", next_evaluation_at_ms=now + 60_000),
        make_rule(alert_rule_id="disabled", next_evaluation_at_ms=now - 1, enabled=False),
        make_rule(alert_rule_id="deleted", next_evaluation_at_ms=now - 1, deleted_at_ms=now),
    ]
    store = InMemoryAlertStore(rules)

    cycle = AlertEvaluator(reader, store).run_once(now_ms=now)

    assert [e.rule.alert_rule_id for e in cycle.evaluations] == ["due"]
    # Claimed, then released on the way out: a rule whose interval is shorter than
    # the lease duration must not be blocked by its own previous claim.
    assert store.rules["due"].lease_owner is None
    # Advances from the rule's previous due time (``now - 1``), not from when it
    # actually ran -- scheduling off execution time halves the effective rate.
    assert store.rules["due"].next_evaluation_at_ms == (now - 1) + 300_000


class TestNextDueScheduling:
    """Cadence must not depend on how long an evaluation takes.

    The scheduler fires on a fixed grid. Deriving the next due time from execution
    time instead of the previous due time halves the real rate, because
    ``now + interval`` always lands just after the next tick.
    """

    def _rule(self, interval_seconds=60, next_evaluation_at_ms=None):
        return AlertRule(
            alert_rule_id="r1",
            experiment_id=1,
            name="r",
            metric_key="latency",
            dimension_key="TRACES",
            aggregation="AVG",
            comparator="GT",
            threshold=1.0,
            window_seconds=600,
            evaluation_interval_seconds=interval_seconds,
            next_evaluation_at_ms=next_evaluation_at_ms,
        )

    def test_first_schedule_uses_now(self):
        assert next_due_ms(self._rule(), now_ms=1_000_000) == 1_060_000

    def test_due_time_advances_from_the_previous_due_not_from_execution_time(self):
        # Ran 7s late; the next slot must stay on the grid, not inherit the delay.
        rule = self._rule(next_evaluation_at_ms=1_000_000)
        assert next_due_ms(rule, now_ms=1_007_000) == 1_060_000

    def test_cadence_does_not_drift_across_many_ticks(self):
        """The regression itself: a fixed-grid scheduler plus per-run delay.

        Every tick is offered to the rule, and one is claimed whenever it is due.
        With execution-time scheduling this yields half the ticks; the assertion
        below is what fails in that case.
        """
        interval_ms = 60_000
        rule = self._rule(next_evaluation_at_ms=0)
        evaluated_at = []
        for tick_ms in range(0, 60 * interval_ms, interval_ms):
            if rule.next_evaluation_at_ms <= tick_ms:
                ran_at = tick_ms + 900  # dispatch + lease + query
                evaluated_at.append(ran_at)
                rule.next_evaluation_at_ms = next_due_ms(rule, ran_at)

        assert len(evaluated_at) == 60
        gaps = {b - a for a, b in zip(evaluated_at, evaluated_at[1:])}
        assert gaps == {interval_ms}

    def test_a_stalled_rule_skips_missed_intervals_rather_than_replaying_them(self):
        # Ten intervals behind. Every evaluation reads the same trailing window, so
        # replaying them would only burn reads.
        rule = self._rule(next_evaluation_at_ms=1_000_000)
        assert next_due_ms(rule, now_ms=1_000_000 + 10 * 60_000 + 5_000) == 1_660_000

    def test_result_is_always_in_the_future(self):
        rule = self._rule(next_evaluation_at_ms=500_000)
        for now_ms in (500_000, 560_000, 1_234_567, 9_999_999):
            assert next_due_ms(rule, now_ms) > now_ms
