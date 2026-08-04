"""Phase 2 integration: the four streams joined together.

Each stream proved its own piece against a fake or a hand-seeded table. These
tests exercise the seams between them -- the places where two independently
correct components can still disagree, which is where the bugs actually live.
"""

import json
import uuid
from pathlib import Path

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource as OTelResource
from opentelemetry.sdk.trace import Event as OTelEvent
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

from mlflow.entities.span import create_mlflow_span
from mlflow.genai.alerts.aggregator import RollupAggregator
from mlflow.genai.alerts.entities import (
    AlertRule,
    SeriesKey,
    derive_evaluation_interval_seconds,
)
from mlflow.genai.alerts.evaluator import AlertEvaluator
from mlflow.genai.alerts.sql_rollup_reader import SqlRollupReader
from mlflow.store.tracking.dbmodels.models import SqlSpanError, SqlTraceInfo
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracing.utils import TraceJSONEncoder

pytestmark = pytest.mark.notrackingurimock

BUCKET_MS = 60_000
MINUTE_MS = 60_000


@pytest.fixture
def store(tmp_path: Path, db_uri: str) -> SqlAlchemyStore:
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir(exist_ok=True)
    return SqlAlchemyStore(db_uri, artifact_uri.as_uri())


@pytest.fixture
def experiment_id(store: SqlAlchemyStore) -> str:
    return store.create_experiment("alerts-e2e")


def make_span(name, span_id, trace_id, parent_id=None, span_type="TOOL", events=None):
    def context(id_num):
        return trace_api.SpanContext(
            trace_id=4242,
            span_id=id_num,
            is_remote=False,
            trace_flags=trace_api.TraceFlags(1),
        )

    otel_span = OTelReadableSpan(
        name=name,
        context=context(span_id),
        parent=context(parent_id) if parent_id is not None else None,
        attributes={
            "mlflow.traceRequestId": json.dumps(trace_id),
            "mlflow.spanType": json.dumps(span_type, cls=TraceJSONEncoder),
        },
        events=events or [],
        start_time=500_000_000,
        end_time=2_000_000_000,
        status=trace_api.Status(trace_api.StatusCode.UNSET),
        resource=OTelResource.get_empty(),
    )
    return create_mlflow_span(otel_span, trace_id, span_type)


def exception_event(exception_type: str) -> OTelEvent:
    return OTelEvent(
        name="exception",
        attributes={
            "exception.message": "boom",
            "exception.type": exception_type,
            "exception.stacktrace": f"Traceback\n{exception_type}: boom",
        },
        timestamp=1_500_000_000,
    )


def seed_traces(store, experiment_id, *, bucket_ms, count, latency_ms, status="OK"):
    """Insert completed traces landing in one bucket.

    Rows are written directly rather than through ``log_spans`` so completion
    times can be placed precisely; the ingest path itself is covered by
    ``test_log_spans_alone_writes_span_errors``.
    """
    with store.ManagedSessionMaker(read_only=False) as session:
        for _ in range(count):
            end_ms = bucket_ms + 1_000
            session.add(
                SqlTraceInfo(
                    request_id=f"tr-{uuid.uuid4().hex}",
                    experiment_id=int(experiment_id),
                    timestamp_ms=end_ms - latency_ms,
                    execution_time_ms=latency_ms,
                    end_time_ms=end_ms,
                    status=status,
                )
            )
        session.commit()


def prime(aggregator, now_ms):
    """Establish a watermark, the way a continuously running server has one.

    A fresh install deliberately starts its watermark at "now" rather than
    scanning all history, so an aggregator that has never run will not seal
    buckets that are already in the past. Tests must therefore prime before
    seeding, or they measure the wrong thing.
    """
    aggregator.run_once(now_ms=now_ms)


def make_rule(store, experiment_id, **overrides) -> AlertRule:
    """Persist a rule through the store.

    Going through ``create_alert_rule`` rather than constructing one in memory is
    deliberate: an instance carries a foreign key to its rule, so an unpersisted
    rule makes the whole evaluation path unexercisable.
    """
    window_seconds = overrides.pop("window_seconds", 3600)
    rule = AlertRule(
        alert_rule_id=str(uuid.uuid4()),
        experiment_id=int(experiment_id),
        name=overrides.pop("name", "slow responses"),
        metric_key="latency",
        dimension_key="TRACES",
        dimension_value=None,
        aggregation="AVG",
        comparator="GT",
        threshold=45 * MINUTE_MS,
        window_seconds=window_seconds,
        evaluation_interval_seconds=derive_evaluation_interval_seconds(window_seconds),
        sustain_seconds=0,
        min_sample_count=0,
    )
    for key, value in overrides.items():
        setattr(rule, key, value)
    return store.create_alert_rule(rule)


def test_log_spans_alone_writes_span_errors(store, experiment_id):
    """The ingest hook is wired.

    Stream A's own tests called ``extract_span_errors`` directly because the call
    site did not exist yet. This asserts the wiring itself: logging spans is now
    sufficient, with no second call.
    """
    trace_id = f"tr-{uuid.uuid4().hex}"
    spans = [
        make_span(
            "search_docs", 1, trace_id, parent_id=2, events=[exception_event("TimeoutError")]
        ),
        make_span("retrieve", 2, trace_id, parent_id=3, events=[exception_event("TimeoutError")]),
        make_span(
            "agent_run", 3, trace_id, span_type="AGENT", events=[exception_event("TimeoutError")]
        ),
    ]
    store.log_spans(experiment_id, spans)

    with store.ManagedSessionMaker() as session:
        rows = session.query(SqlSpanError).filter(SqlSpanError.trace_id == trace_id).all()
        origins = [r.span_name for r in rows if r.is_origin]

    assert len(rows) == 3
    # Propagation is deduped: the deepest span is the only origin, so the error
    # count does not depend on how deeply the agent was nested.
    assert origins == ["search_docs"]


def test_aggregator_and_evaluator_agree_on_window_boundaries(store, experiment_id):
    """The predicted integration bug, asserted directly.

    The aggregator's bucketing and the evaluator's ``window_end = watermark +
    60_000`` must agree, or windows half-cover buckets and silently under-count.
    """
    now_ms = 400 * BUCKET_MS
    aggregator = RollupAggregator(store)
    prime(aggregator, 340 * BUCKET_MS)
    for i in range(10):
        seed_traces(
            store,
            experiment_id,
            bucket_ms=(360 + i) * BUCKET_MS,
            count=3,
            latency_ms=60_000,
        )
    aggregator.run_once(now_ms=now_ms)

    reader = SqlRollupReader(store)
    watermark = reader.latest_sealed_bucket_ms()
    assert watermark % BUCKET_MS == 0

    window_end = watermark + BUCKET_MS
    window_start = window_end - 3600 * 1000
    assert window_start % BUCKET_MS == 0

    buckets = reader.read_buckets(
        SeriesKey("TRACES", int(experiment_id), "latency", ""), window_start, window_end
    )
    # Every bucket read is fully inside the window, and none is the still-open one.
    assert all(window_start <= b.bucket_start_ms < window_end for b in buckets)
    assert all(b.bucket_start_ms <= watermark for b in buckets)


def test_pipeline_reaches_fired_and_notifies_once(store, experiment_id):
    """Raw traces -> rollups -> a fired alert, with exactly one notification."""
    now_ms = 400 * BUCKET_MS
    aggregator = RollupAggregator(store)
    prime(aggregator, 340 * BUCKET_MS)

    # The rule comes first. Aggregation is demand-driven, so a series nothing
    # subscribes to is never written -- which means a rule created *after* the
    # traffic it wants to measure has nothing to read until fresh buckets seal.
    # A sustain longer than one interval is what separates PENDING from FIRED, so
    # without it the rule fires on its first breach and the path under test never
    # runs.
    rule = make_rule(store, experiment_id, sustain_seconds=300)

    for i in range(30):
        seed_traces(
            store,
            experiment_id,
            bucket_ms=(360 + i) * BUCKET_MS,
            count=5,
            latency_ms=90 * MINUTE_MS,
        )
    aggregator.run_once(now_ms=now_ms)

    notified = []
    evaluator = AlertEvaluator(
        SqlRollupReader(store),
        store,
        notifier=lambda r, i: notified.append(i),
    )

    events = []
    for tick in range(4):
        cycle = evaluator.evaluate_rules([rule], now_ms + tick * 5 * MINUTE_MS)
        events.append(cycle.evaluations[0].transition.event)

    assert events[0] == "OPENED"
    assert "FIRED" in events
    # A sustained incident notifies once, not once per evaluation. This single
    # behavior decides whether the feature survives contact with users.
    assert len(notified) == 1
    assert notified[0].observed_value > 45 * MINUTE_MS


def test_a_healthy_pipeline_produces_no_instance(store, experiment_id):
    """The negative control: the same wiring must stay silent on good traffic."""
    now_ms = 400 * BUCKET_MS
    aggregator = RollupAggregator(store)
    prime(aggregator, 340 * BUCKET_MS)
    for i in range(30):
        seed_traces(
            store,
            experiment_id,
            bucket_ms=(360 + i) * BUCKET_MS,
            count=5,
            latency_ms=2_000,
        )
    aggregator.run_once(now_ms=now_ms)

    notified = []
    evaluator = AlertEvaluator(
        SqlRollupReader(store), store, notifier=lambda r, i: notified.append(i)
    )
    cycle = evaluator.evaluate_rules([make_rule(store, experiment_id)], now_ms)

    assert cycle.evaluations[0].transition.event == "NONE"
    assert notified == []


def test_a_first_run_does_not_roll_up_pre_existing_history(store, experiment_id):
    """Pins a behavior that is easy to mistake for a bug, and that operators
    upgrading an existing MLflow install will hit.

    The watermark starts at "now", so traces already in the database when the
    aggregator first runs are never sealed. That is the right default -- the
    alternative is scanning all history on first boot -- but it means a rule
    created immediately after an upgrade has no rollups to read until fresh
    traffic accumulates.
    """
    for i in range(10):
        seed_traces(
            store,
            experiment_id,
            bucket_ms=(360 + i) * BUCKET_MS,
            count=5,
            latency_ms=90 * MINUTE_MS,
        )

    RollupAggregator(store).run_once(now_ms=400 * BUCKET_MS)

    reader = SqlRollupReader(store)
    buckets = reader.read_buckets(
        SeriesKey("TRACES", int(experiment_id), "latency", ""), 0, 400 * BUCKET_MS
    )
    assert buckets == []


def test_a_second_incident_is_reported_after_the_first_recovers(store, experiment_id):
    """Breach, sustained recovery, breach again -- through the real store.

    The user-visible payoff of INACTIVE: two rows, not one. The first episode
    keeps its peak and is still waiting to be acknowledged; the second is a new
    instance that notified on its own. Before INACTIVE the single row stayed FIRED
    throughout and the second incident was invisible.

    A ten-minute window so the window can move fully past the breaching buckets --
    an AVG over a window straddling both phases is neither.
    """
    aggregator = RollupAggregator(store)
    prime(aggregator, 340 * BUCKET_MS)
    rule = make_rule(store, experiment_id, window_seconds=600)
    notified = []
    evaluator = AlertEvaluator(
        SqlRollupReader(store), store, notifier=lambda r, i: notified.append(i)
    )

    def seed(first_bucket, last_bucket, latency_ms):
        for bucket in range(first_bucket, last_bucket):
            seed_traces(
                store,
                experiment_id,
                bucket_ms=bucket * BUCKET_MS,
                count=5,
                latency_ms=latency_ms,
            )

    seed(360, 380, 90 * MINUTE_MS)
    aggregator.run_once(now_ms=382 * BUCKET_MS)
    assert evaluator.evaluate_rules([rule], 382 * BUCKET_MS).evaluations[0].transition.event == (
        "FIRED"
    )

    seed(381, 400, 2_000)
    aggregator.run_once(now_ms=402 * BUCKET_MS)
    # Two healthy evaluations of the same recovered window: the first arms the
    # recovery, the second is what closes it.
    assert evaluator.evaluate_rules([rule], 402 * BUCKET_MS).evaluations[0].transition.event == (
        "UPDATED"
    )
    assert evaluator.evaluate_rules([rule], 403 * BUCKET_MS).evaluations[0].transition.event == (
        "RECOVERED"
    )

    seed(401, 420, 90 * MINUTE_MS)
    aggregator.run_once(now_ms=422 * BUCKET_MS)
    assert evaluator.evaluate_rules([rule], 422 * BUCKET_MS).evaluations[0].transition.event == (
        "FIRED"
    )

    active = sorted(store.list_alert_instances(experiment_id), key=lambda i: i.started_at_ms)
    assert [i.state for i in active] == ["INACTIVE", "FIRED"]
    # INACTIVE is still in the active view and still unacknowledged.
    assert active[0].dismissed_by is None
    assert active[0].peak_value > 45 * MINUTE_MS
    assert active[0].healthy_since_ms is not None
    # Recovering paged nobody; the second incident did.
    assert len(notified) == 2
    assert {n.alert_instance_id for n in notified} == {i.alert_instance_id for i in active}

    # ...and it is dismissible from INACTIVE, which is the whole reason it stays.
    dismissed = store.dismiss_alert_instance(active[0].alert_instance_id, "alice@example.com")
    assert dismissed.state == "DISMISSED"
    assert dismissed.dismissed_by == "alice@example.com"
    assert dismissed.peak_value == active[0].peak_value


def test_deleting_a_rule_closes_its_open_instances_but_keeps_them(store, experiment_id):
    """Deleting a rule is an explicit "stop telling me about this".

    An unacknowledged red row for a rule that no longer exists is noise the user
    cannot act on, and nothing would ever update it again since the rule is no
    longer evaluated. Closing is not forgetting: the row survives as DISMISSED
    with a system actor, so a postmortem can still see what fired.
    """
    now_ms = 400 * BUCKET_MS
    aggregator = RollupAggregator(store)
    prime(aggregator, 340 * BUCKET_MS)
    # Before the traffic: aggregation is demand-driven, so the subscription has to
    # exist while the buckets are being sealed.
    rule = make_rule(store, experiment_id)
    for i in range(30):
        seed_traces(
            store,
            experiment_id,
            bucket_ms=(360 + i) * BUCKET_MS,
            count=5,
            latency_ms=90 * MINUTE_MS,
        )
    aggregator.run_once(now_ms=now_ms)

    AlertEvaluator(SqlRollupReader(store), store).evaluate_rules([rule], now_ms)
    assert len(store.list_alert_instances(experiment_id)) == 1

    store.delete_alert_rule(rule.alert_rule_id)

    assert store.list_alert_instances(experiment_id) == []
    history = store.list_alert_instances(experiment_id, states=["DISMISSED"])
    assert len(history) == 1
    assert history[0].dismissed_by == "system:rule_deleted"
    assert history[0].observed_value > 45 * MINUTE_MS
