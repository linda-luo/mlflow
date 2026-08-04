"""Evaluator workers — claim due rules, read the window, compare, transition.

A plain callable, driveable from a script, so it is testable against
:class:`~mlflow.genai.alerts.rollup_reader.FakeRollupReader` without Timescale or
a scheduler.

Three things here are easy to get subtly wrong and are therefore stated once:

* **Windows derive from the watermark, never from the clock.** They are then
  bucket-aligned and always fully sealed. If the aggregator's bucketing and
  ``window_end_ms`` disagree on boundaries, windows half-cover buckets and
  silently under-count.
* **Rules are grouped by read signature before querying.** Ten latency rules with
  different thresholds are one read. Rules in a group share a window, therefore
  an interval, therefore a due time — the group is the natural schedulable unit.
* **Decisions come from exact bounds, never estimates.** Histogram bucket counts
  are exact, so a histogram yields exact bounds; only when those bounds straddle
  the threshold does anything touch raw rows.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Callable, Protocol

from mlflow.genai.alerts import histogram as hist
from mlflow.genai.alerts.entities import (
    BUCKET_MS,
    AlertInstance,
    AlertRule,
    Observation,
    SeriesKey,
    Transition,
)
from mlflow.genai.alerts.exemplars import collect_exemplar_trace_ids
from mlflow.genai.alerts.rollup_reader import Bucket, RollupReader, project_aggregate
from mlflow.genai.alerts.sketch import SketchSpec
from mlflow.genai.alerts.state_machine import evaluate_transition
from mlflow.utils.time import get_current_time_millis

_logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 50
"""Rules evaluated in one cycle. A bound on the cycle, not a claim."""
DEFAULT_MAX_WORKERS = 1
"""Evaluator threads per process. One unless a deployment opts in.

The work is almost entirely waiting on the database, so threads overlap it and
the GIL is not the constraint. The connection pool is: its default is 15 for the
whole process, shared with the periodic-task consumer and every rollup task, so
this cannot be raised freely. See ``MLFLOW_ALERT_EVALUATOR_THREADS``.
"""
DEFAULT_FULL_RECOMPUTE_INTERVAL_MS = 3_600_000
"""Drift guard for the incremental merge cache.

A cached running total can drift — a bucket sealed late is added after the window
that should have contained it already moved on — so a periodic full read is the
backstop that lets the cache stay in memory rather than being persisted.
"""


###############################################################################
# Read grouping
###############################################################################


@dataclass(frozen=True)
class ReadSignature:
    """What makes two rules readable in a single query.

    Deliberately excludes ``aggregation``, ``percentile_value``, ``comparator``
    and ``threshold``: those select a column and a predicate over an already-read
    window, so ten rules differing only in threshold still cost one read.
    """

    experiment_id: int
    dimension_key: str
    metric_key: str
    dimension_value: str
    window_seconds: int

    @property
    def series(self) -> SeriesKey:
        return SeriesKey(
            dimension_key=self.dimension_key,
            experiment_id=self.experiment_id,
            metric_key=self.metric_key,
            dimension_value=self.dimension_value,
        )


def read_signature(rule: AlertRule) -> ReadSignature:
    return ReadSignature(
        experiment_id=rule.experiment_id,
        dimension_key=rule.dimension_key,
        metric_key=rule.metric_key,
        dimension_value=rule.dimension_value or "",
        window_seconds=rule.window_seconds,
    )


def group_rules(rules: list[AlertRule]) -> dict[ReadSignature, list[AlertRule]]:
    grouped: dict[ReadSignature, list[AlertRule]] = {}
    for rule in rules:
        grouped.setdefault(read_signature(rule), []).append(rule)
    return grouped


###############################################################################
# Windows
###############################################################################


@dataclass(frozen=True)
class Window:
    start_ms: int
    end_ms: int

    @property
    def span_ms(self) -> int:
        return self.end_ms - self.start_ms


def evaluation_window(latest_sealed_bucket_ms: int, window_seconds: int) -> Window:
    """The window to read, derived from the watermark rather than the clock.

    ``latest_sealed_bucket_ms`` is the *start* of the last fully sealed bucket, so
    the exclusive end of the readable range is one bucket later. Both edges are
    multiples of ``BUCKET_MS``, which is the invariant the aggregator's
    ``time_bucket`` must agree with.
    """
    end_ms = latest_sealed_bucket_ms + BUCKET_MS
    return Window(start_ms=end_ms - window_seconds * 1000, end_ms=end_ms)


###############################################################################
# Window accumulation and incremental merge
###############################################################################


@dataclass
class WindowAccumulator:
    """Every aggregate the rollup stores, folded over a window.

    One accumulator serves every rule in a read group: ``aggregation`` selects
    which field is projected, so a COUNT rule and a PERCENTILE rule over the same
    series share a single read.

    Every field is invertible. That is not incidental — it is why ``min``/``max``
    are not stored, and it is what makes :meth:`remove` (and therefore incremental
    merge) possible.
    """

    count: int = 0
    sum_value: float = 0.0
    histogram: hist.Histogram = field(default_factory=hist.empty)
    gap_buckets: set[int] = field(default_factory=set)
    """Bucket starts explicitly marked as gaps.

    Tracked by start rather than as a flag because a gap has to *leave* the window
    when it expires; a boolean could never be un-set.
    """

    histogram_bucket_count: int = 0
    """How much of the window carried a histogram, in minute-equivalents.

    A counter rather than a flag, for the same reason ``gap_buckets`` is a set: a
    histogram-bearing bucket eventually *leaves* the window, and a boolean could
    never be un-set. Distinguishes "no bucket stored a histogram" -- which must
    fail loudly for a PERCENTILE rule -- from "the histogram is genuinely all
    zeros", which an accumulated vector alone cannot express.

    Counted in minutes rather than rows so it survives a window stitched from more
    than one tier: an hourly row is worth the sixty minute rows that will later
    expire out of the window in its place. Counting rows made the two directions
    disagree by a factor of sixty, and the counter went negative.
    """

    def copy(self) -> "WindowAccumulator":
        return WindowAccumulator(
            count=self.count,
            sum_value=self.sum_value,
            histogram=dict(self.histogram),
            gap_buckets=set(self.gap_buckets),
            histogram_bucket_count=self.histogram_bucket_count,
        )

    def _counts_toward_histogram(self, bucket: Bucket) -> bool:
        # A bucket sealed under a different sketch version describes different value
        # ranges. There are none in normal operation -- the grid is a constant -- so
        # this guards only a future change to GAMMA.
        return bucket.histogram is not None and bucket.boundaries_version == hist.SKETCH_VERSION

    def add(self, buckets: list[Bucket]) -> "WindowAccumulator":
        for b in buckets:
            if b.is_gap:
                self.gap_buckets.add(b.bucket_start_ms)
                continue
            self.count += b.count
            self.sum_value += b.sum or 0.0
            if self._counts_toward_histogram(b):
                self.histogram = hist.merge(self.histogram, hist.from_pairs(b.histogram))
                self.histogram_bucket_count += b.width_ms // BUCKET_MS
        return self

    def remove(self, buckets: list[Bucket]) -> "WindowAccumulator":
        for b in buckets:
            if b.is_gap:
                self.gap_buckets.discard(b.bucket_start_ms)
                continue
            self.count -= b.count
            self.sum_value -= b.sum or 0.0
            if self._counts_toward_histogram(b):
                self.histogram = hist.subtract(self.histogram, hist.from_pairs(b.histogram))
                self.histogram_bucket_count -= b.width_ms // BUCKET_MS
        return self

    def to_observation(
        self,
        aggregation: str,
        window: Window,
        percentile_value: float | None = None,
        spec: SketchSpec | None = None,
    ) -> Observation:
        """Project one rule's number.

        Delegates to :func:`~mlflow.genai.alerts.rollup_reader.project_aggregate`
        rather than reimplementing it. The two used to be separate, and they
        disagreed: this path treated an empty window as no-data for *every*
        aggregation, so an absence-signal rule could never fire.
        """
        return project_aggregate(
            aggregation,
            count=self.count,
            sum_value=self.sum_value,
            histogram=self.histogram if self.histogram_bucket_count > 0 else None,
            has_gap=bool(self.gap_buckets),
            window_start_ms=window.start_ms,
            window_end_ms=window.end_ms,
            percentile_value=percentile_value,
            spec=spec,
        )


@dataclass
class _CacheEntry:
    window: Window
    accumulator: WindowAccumulator
    built_at_ms: int


class IncrementalMergeCache:
    """Read amplification is ``window / interval`` — 864x for a 3-day rule at 5 min.

    A 3-day window shares 4,315 of its 4,320 buckets with the previous evaluation,
    so each cycle adds the buckets that entered and subtracts the ones that
    expired. The delta is ``interval / 60`` buckets regardless of window size —
    ``window`` cancels out, which is why the longest window costs no more per cycle
    than the shortest.

    The expiring edge is read at one-minute granularity, so it is only readable
    while the raw tier still holds it. That is the constraint behind
    ``MAX_WINDOW_SECONDS`` being capped at raw retention: past it, the read returns
    nothing, ``remove`` subtracts nothing, and the accumulator silently stops
    shedding.

    In memory and process-local. It is shared by every evaluator thread, which is
    the reason threads are the right shape for this: separate processes would each
    hold a cold copy, and a rule bouncing between them would silently go back to a
    full read every cycle. Sticky assignment therefore now separates *replicas*
    rather than threads. A restart costs one cold read; an hourly full recompute
    guards against drift.

    Keyed by :class:`ReadSignature` rather than ``(rule_id, series_id)``: the
    signature already contains everything that changes what is read, so an edit to
    window, metric or dimension moves the rule to a different key and invalidates
    itself, while a threshold edit correctly keeps the warm entry.

    Thread safety is per *entry*, not global. ``accumulator_for`` does a
    read-modify-write around two DB reads, so a single lock held across it would
    serialize every thread and undo the fan-out entirely; a lock released before
    the reads would let a slow thread overwrite a newer window with an older one,
    which silently degrades the next cycle to a full read. A lock per signature
    gives both: distinct read groups never contend, and the one case that must be
    ordered -- two threads on the same signature -- is.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        full_recompute_interval_ms: int = DEFAULT_FULL_RECOMPUTE_INTERVAL_MS,
    ):
        self.enabled = enabled
        self.full_recompute_interval_ms = full_recompute_interval_ms
        self._entries: dict[ReadSignature, _CacheEntry] = {}
        self._entry_locks: dict[ReadSignature, threading.Lock] = {}
        self._map_lock = threading.Lock()
        """Guards the two dicts themselves, never held across a read."""

    def _lock_for(self, signature: ReadSignature) -> threading.Lock:
        with self._map_lock:
            return self._entry_locks.setdefault(signature, threading.Lock())

    def drop(self, signature: ReadSignature) -> None:
        with self._lock_for(signature), self._map_lock:
            self._entries.pop(signature, None)

    def clear(self) -> None:
        with self._map_lock:
            self._entries.clear()
            self._entry_locks.clear()

    def accumulator_for(
        self,
        signature: ReadSignature,
        window: Window,
        reader: RollupReader,
        now_ms: int,
    ) -> WindowAccumulator:
        with self._lock_for(signature):
            with self._map_lock:
                entry = self._entries.get(signature)
            accumulator = self._merged(signature, window, reader, now_ms, entry)
            if accumulator is None:
                # `read_window`, not `read_buckets`: this is the one path that reads
                # a window end to end, so it is the one that can be served from the
                # hourly tier. A 24h rule goes from 1,440 rows to about 83.
                accumulator = WindowAccumulator().add(
                    reader.read_window(signature.series, window.start_ms, window.end_ms)
                )
                built_at_ms = now_ms
            else:
                # From the entry seen under this lock, so a racing writer cannot
                # reset the hourly drift guard by carrying its own timestamp in.
                built_at_ms = entry.built_at_ms
            with self._map_lock:
                self._entries[signature] = _CacheEntry(
                    window=window, accumulator=accumulator, built_at_ms=built_at_ms
                )
            return accumulator.copy()

    def _merged(
        self,
        signature: ReadSignature,
        window: Window,
        reader: RollupReader,
        now_ms: int,
        entry: _CacheEntry | None,
    ) -> WindowAccumulator | None:
        """The incremental path, or None when a full read is required."""
        if not self.enabled or entry is None:
            return None
        if now_ms - entry.built_at_ms >= self.full_recompute_interval_ms:
            return None
        if entry.window == window:
            return entry.accumulator.copy()
        shift = window.end_ms - entry.window.end_ms
        if shift <= 0 or shift >= window.span_ms or window.span_ms != entry.window.span_ms:
            # Backwards, or so far ahead that nothing overlaps: a full read is both
            # cheaper and the only correct option.
            return None
        accumulator = entry.accumulator.copy()
        accumulator.add(reader.read_buckets(signature.series, entry.window.end_ms, window.end_ms))
        accumulator.remove(
            reader.read_buckets(signature.series, entry.window.start_ms, window.start_ms)
        )
        return accumulator


###############################################################################
# Deciding a breach
###############################################################################


class RawValueVerifier(Protocol):
    """The escape hatch for the rare ambiguous case.

    Only consulted when exact histogram bounds straddle the threshold, which
    thresholds snapped to a boundary at rule-creation time make uncommon.
    """

    def count_above(self, series: SeriesKey, start_ms: int, end_ms: int, threshold: float) -> int:
        """Exact count of raw observations strictly above ``threshold``."""
        ...


@dataclass(frozen=True)
class Decision:
    breaching: bool
    exact: bool
    """False only when bounds were ambiguous and no verifier was available."""
    verified: bool
    """True when raw rows were consulted."""


def percentile_count_threshold(rule: AlertRule, sample_count: int) -> float:
    """ "p95 > T" is "more than 5% of samples exceed T" — a count predicate.

    Restating a percentile comparison as a count above a boundary is what lets an
    exact bucket count decide it, since the histogram knows counts exactly and
    knows the percentile only to within a bucket.
    """
    if rule.percentile_value is None:
        raise ValueError("PERCENTILE aggregation requires percentile_value")
    return (1.0 - rule.percentile_value / 100.0) * sample_count


def decide(
    rule: AlertRule,
    observation: Observation,
    accumulator: WindowAccumulator,
    *,
    verifier: RawValueVerifier | None = None,
    series: SeriesKey | None = None,
    spec: SketchSpec | None = None,
) -> Decision:
    """Bounds first, raw rows only when the bounds disagree with each other.

    The bounds are *exact counts* either way -- what varies is how far apart they
    are, and on the sketch's grid they differ by one bucket spanning barely 4% of
    its own value. So the ambiguous branch below is reached only when the metric is
    sitting within that band of the threshold, and no threshold is snapped to make
    it so.
    """
    if observation.observed_value is None:
        return Decision(breaching=False, exact=True, verified=False)

    if rule.aggregation != "PERCENTILE":
        # count / sum / avg come from exact columns; there is nothing to bound.
        breaching = hist.compare(observation.observed_value, rule.comparator, rule.threshold)
        return Decision(breaching=breaching, exact=True, verified=False)

    if spec is None:
        return Decision(breaching=False, exact=True, verified=False)
    count_threshold = percentile_count_threshold(rule, observation.sample_count)
    verdict = hist.bounds_decide(
        accumulator.histogram, rule.threshold, rule.comparator, count_threshold, spec
    )
    if verdict is not None:
        return Decision(breaching=verdict, exact=True, verified=False)

    if verifier is not None and series is not None:
        exact_count = verifier.count_above(
            series, observation.window_start_ms, observation.window_end_ms, rule.threshold
        )
        breaching = hist.compare(exact_count, rule.comparator, count_threshold)
        return Decision(breaching=breaching, exact=True, verified=True)

    # No verifier wired up: fall back to the histogram's own estimate, which is an
    # over-estimate by less than one bucket width, and say so.
    _logger.debug(
        "Alert rule %s straddles a histogram boundary and no raw verifier is "
        "configured; deciding from the bucketed estimate",
        rule.alert_rule_id,
    )
    breaching = hist.compare(observation.observed_value, rule.comparator, rule.threshold)
    return Decision(breaching=breaching, exact=False, verified=False)


###############################################################################
# Leasing
###############################################################################


def next_due_ms(rule: AlertRule, now_ms: int) -> int:
    """When this rule should next be evaluated.

    Advances from the rule's *previous due time*, not from when it actually ran.
    Scheduling from execution time looks equivalent but halves the real rate: the
    scheduler fires on a fixed grid, so ``now + interval`` always lands just after
    the next tick and that tick finds nothing due.

        tick at T     -> runs at T+d,  next = T+d+interval
        tick at T+i   -> T+d+i <= T+i is false for any d > 0  -> skipped
        tick at T+2i  -> due

    For any non-zero execution delay the rule permanently evaluates on every second
    tick. That is not merely a late alert: consecutive windows are supposed to
    overlap by ``window - interval``, so a doubled interval can reach ``window`` and
    leave spans that no evaluation ever inspects.

    When a rule has fallen behind -- a stalled worker, a paused host -- whole
    intervals are skipped rather than replayed. Catching up serves no purpose when
    every evaluation reads the same trailing window anyway, and it would otherwise
    fire a burst of redundant reads.
    """
    interval_ms = rule.evaluation_interval_seconds * 1000
    previous_due_ms = rule.next_evaluation_at_ms
    if previous_due_ms is None:
        return now_ms + interval_ms

    next_ms = previous_due_ms + interval_ms
    if next_ms <= now_ms:
        missed = (now_ms - previous_due_ms) // interval_ms + 1
        next_ms = previous_due_ms + missed * interval_ms
    return next_ms


def build_due_rule_query(now_ms: int, limit: int = DEFAULT_BATCH_SIZE):
    """SELECT of due rule ids, most-overdue-first.

    One query, one dialect. This used to fan rules across replicas -- ``hashtext``
    sticky assignment and ``FOR UPDATE SKIP LOCKED`` on Postgres, an over-fetching
    select filtered in Python everywhere else, and a lease column marking rules
    in flight. Nothing else in MLflow coordinates background work that way: every
    other periodic task settles for ``huey.lock_task`` and one consumer, which is
    the deployment shape the product actually supports. Alerting now matches.

    ``next_evaluation_at_ms`` is materialized rather than computed:
    ``last_evaluated_ms + interval * 1000 <= now`` is arithmetic across two columns
    and no index can seek it.
    """
    # Imported lazily: sqlalchemy is a core, not a skinny, dependency and this
    # module must stay importable by clients that only ever call the state machine.
    import sqlalchemy as sa

    from mlflow.store.tracking.dbmodels.models import SqlAlertRule

    table = SqlAlertRule.__table__
    return (
        sa
        .select(table.c.alert_rule_id)
        .where(
            table.c.enabled.is_(True),
            table.c.deleted_at_ms.is_(None),
            table.c.next_evaluation_at_ms <= now_ms,
        )
        # Most overdue first, so a rule that fell behind is not starved by the
        # batch limit on every subsequent cycle.
        .order_by(table.c.next_evaluation_at_ms)
        .limit(limit)
    )


def due_rule_ids(session, now_ms: int, limit: int = DEFAULT_BATCH_SIZE) -> list[str]:
    """Ids of the rules due now.

    No claim step: ``alert-evaluator-lock`` serializes cycles within the process,
    and a rule belongs to exactly one read group, so no two threads reach the same
    rule. Concurrency beyond that is out of scope -- see :func:`build_due_rule_query`.
    """
    return list(session.execute(build_due_rule_query(now_ms, limit)).scalars().all())


###############################################################################
# The evaluator
###############################################################################


class AlertEvaluationStore(Protocol):
    """The persistence the evaluator needs, and nothing else.

    Narrow on purpose: it lets the evaluator be driven from a script against an
    in-memory implementation, and it is the exact set of methods the store layer
    has to grow.
    """

    def due_alert_rules(self, now_ms: int, limit: int) -> list[AlertRule]: ...

    def get_open_alert_instance(self, alert_rule_id: str) -> AlertInstance | None:
        """The rule's PENDING or FIRED instance, if it has one.

        Not merely "undismissed": an INACTIVE instance has recovered and is still
        waiting to be acknowledged, but it must not be returned here or the rule
        could never report the next episode.
        """
        ...

    def save_alert_instance(self, instance: AlertInstance) -> AlertInstance:
        """Insert or update. Must uphold the one-open-instance-per-series rule."""
        ...

    def record_alert_rule_evaluated(
        self,
        alert_rule_id: str,
        last_evaluated_ms: int,
        next_evaluation_at_ms: int,
        last_sample_count: int | None,
    ) -> None:
        """Persist the outcome: when it ran, when it next runs, what it saw."""
        ...


@dataclass(frozen=True)
class RuleEvaluation:
    rule: AlertRule
    observation: Observation
    transition: Transition
    decision: Decision


@dataclass(frozen=True)
class EvaluationCycle:
    now_ms: int
    window_end_ms: int
    evaluations: list[RuleEvaluation]
    groups_read: int
    notifications: list[AlertInstance]


@dataclass(frozen=True)
class _GroupResult:
    """One read group's outcome, returned rather than raised.

    ``watermark_ms`` is ``None`` when the group's read failed, so a group that
    could not be read is excluded from the cycle's window rather than dragging it
    back to zero.
    """

    watermark_ms: int | None
    evaluations: list[RuleEvaluation]
    notifications: list[AlertInstance]


class AlertEvaluator:
    """The evaluation loop, as a plain callable.

    Args:
        reader: rollup source. ``FakeRollupReader`` in tests, the Timescale-backed
            reader in production; the evaluator cannot tell the difference.
        store: see :class:`AlertEvaluationStore`.
        verifier: consulted only when histogram bounds are ambiguous.
        cache: incremental merge cache; pass ``IncrementalMergeCache(enabled=False)``
            to force a full read every cycle.
        notifier: called after the instance is committed. Durability comes from
            that ordering, not from a delivery table.
        batch_size: rules evaluated in one cycle.
        max_workers: threads to spread this cycle's read groups over. One means
            no pool is created at all. This is *within* the process -- see
            ``MLFLOW_ALERT_EVALUATOR_THREADS`` -- and is unrelated to how rules
            are selected.
    """

    def __init__(
        self,
        reader: RollupReader,
        store: AlertEvaluationStore,
        *,
        verifier: RawValueVerifier | None = None,
        cache: IncrementalMergeCache | None = None,
        notifier: Callable[[AlertRule, AlertInstance], None] | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_workers: int = DEFAULT_MAX_WORKERS,
    ):
        self.reader = reader
        self.store = store
        self.verifier = verifier
        self.cache = cache if cache is not None else IncrementalMergeCache()
        self.notifier = notifier
        self.batch_size = batch_size
        self.max_workers = max(1, max_workers)

    def run_once(self, now_ms: int | None = None) -> EvaluationCycle:
        now_ms = get_current_time_millis() if now_ms is None else now_ms
        rules = self.store.due_alert_rules(now_ms=now_ms, limit=self.batch_size)
        return self.evaluate_rules(rules, now_ms)

    def evaluate_rules(self, rules: list[AlertRule], now_ms: int) -> EvaluationCycle:
        # One watermark *per read group*, not one for the whole cycle. Groups read
        # different series, and a series is only as fresh as the family that seals
        # it -- assessments land minutes after the traces they score, so a single
        # cycle-wide watermark let a lagging quality rollup hold back latency rules
        # that shared nothing with it.
        #
        # Rules within a group still share a watermark, which is what matters: they
        # read the same series, so their windows remain directly comparable.
        groups = group_rules(rules)
        items = list(groups.items())

        # The group, not the rule, is the unit of work: a group is one read shared
        # by every rule in it, and it is also exactly one cache entry, so handing
        # whole groups out means two threads never contend for the same entry.
        if self.max_workers > 1 and len(items) > 1:
            with ThreadPoolExecutor(
                max_workers=min(self.max_workers, len(items)),
                thread_name_prefix="MlflowAlertEval",
            ) as executor:
                futures = [
                    executor.submit(self._evaluate_group, signature, group, now_ms)
                    for signature, group in items
                ]
                # Submission order, not completion order: the cycle's contents must
                # not depend on which thread happened to finish first.
                results = [future.result() for future in futures]
        else:
            results = [self._evaluate_group(signature, group, now_ms) for signature, group in items]

        evaluations = [e for result in results for e in result.evaluations]
        notifications = [n for result in results for n in result.notifications]
        watermarks = [r.watermark_ms for r in results if r.watermark_ms is not None]

        return EvaluationCycle(
            now_ms=now_ms,
            # The most conservative window this cycle covered; reporting per-group
            # ends would make this field meaningless as a single number.
            window_end_ms=(min(watermarks) if watermarks else 0) + BUCKET_MS,
            evaluations=evaluations,
            groups_read=len(groups),
            notifications=notifications,
        )

    def _evaluate_group(
        self, signature: ReadSignature, group: list[AlertRule], now_ms: int
    ) -> "_GroupResult":
        """One read group: the shared read, then every rule that reads it.

        Returns its outcome rather than raising, and owns its own notification
        list rather than appending to a shared one. Both are what make the group
        safe to run on its own thread -- and the first is worth having even at one
        thread, since a single failing group used to abort the whole cycle and
        leave every remaining rule unevaluated.
        """
        try:
            watermark_ms = self.reader.latest_sealed_bucket_ms(signature.series)
            window = evaluation_window(watermark_ms, signature.window_seconds)
            accumulator = self.cache.accumulator_for(signature, window, self.reader, now_ms)
        except Exception:
            _logger.exception(
                "Failed to read rollups for %s/%s; skipping its %d rule(s) this cycle",
                signature.series.dimension_key,
                signature.series.metric_key,
                len(group),
            )
            return _GroupResult(watermark_ms=None, evaluations=[], notifications=[])

        notifications: list[AlertInstance] = []
        evaluations: list[RuleEvaluation] = []
        for rule in group:
            try:
                evaluations.append(
                    self._evaluate_one(rule, signature, window, accumulator, now_ms, notifications)
                )
            except Exception:
                # Scoped to the rule for the same reason the group boundary exists:
                # a transient write failure on one rule must not cost every other
                # rule its evaluation.
                _logger.exception(
                    "Alert rule %s failed to evaluate; skipping it this cycle",
                    rule.alert_rule_id,
                )
        return _GroupResult(
            watermark_ms=watermark_ms, evaluations=evaluations, notifications=notifications
        )

    def _evaluate_one(
        self,
        rule: AlertRule,
        signature: ReadSignature,
        window: Window,
        accumulator: WindowAccumulator,
        now_ms: int,
        notifications: list[AlertInstance],
    ) -> RuleEvaluation:
        if window.start_ms < self.reader.coverage_start_ms(signature.series):
            # The window reaches back before aggregation began for this series, so
            # an empty result over it is not evidence of anything. Treated exactly
            # like a recorded gap -- "nobody was looking" -- which cannot open or
            # close an instance.
            #
            # Only an absence rule can tell the difference, and for that rule the
            # difference is a false alarm every time a rule or a deployment is new:
            # a 10-minute "traffic dropped" window on a server that started two
            # minutes ago is mostly a period nothing was watching.
            return self._finish(
                rule,
                Observation(None, 0, window.start_ms, window.end_ms),
                Decision(breaching=False, exact=True, verified=False),
                now_ms,
                notifications,
            )

        spec = self.reader.sketch_for(signature.series)
        try:
            observation = accumulator.to_observation(
                rule.aggregation, window, rule.percentile_value, spec
            )
        except ValueError:
            # An unanswerable rule -- a PERCENTILE over a metric with no histogram
            # boundaries. `validate_metric_triple` rejects these at creation, so
            # reaching here means a rule predates that check.
            #
            # Loud, but scoped to the rule: letting this propagate would abort the
            # whole cycle, so one misconfigured rule would stop every *other* rule
            # from being evaluated. The log is the loud part; the rule itself
            # reports no data, which cannot close an open instance.
            _logger.exception(
                "Alert rule %s (%s %s) cannot be evaluated; skipping it this cycle",
                rule.alert_rule_id,
                rule.aggregation,
                rule.metric_key,
            )
            observation = Observation(None, 0, window.start_ms, window.end_ms)
        decision = decide(
            rule,
            observation,
            accumulator,
            verifier=self.verifier,
            series=signature.series,
            spec=spec,
        )
        return self._finish(rule, observation, decision, now_ms, notifications)

    def _finish(
        self,
        rule: AlertRule,
        observation: Observation,
        decision: Decision,
        now_ms: int,
        notifications: list[AlertInstance],
    ) -> RuleEvaluation:
        """Transition, persist, notify, and record that the rule was evaluated.

        Shared by every exit from :meth:`_evaluate_one` so that a rule which could
        not be read still has its ``last_evaluated_ms`` advanced -- otherwise it
        would look stalled, which is indistinguishable from the scheduler being dead.
        """
        open_instance = self.store.get_open_alert_instance(rule.alert_rule_id)
        transition = evaluate_transition(
            rule,
            open_instance,
            observation,
            now_ms,
            breaching=decision.breaching,
        )

        if transition.instance is not None and transition.event != "NONE":
            if transition.should_notify:
                # Gathered here rather than when someone opens the alert: by then
                # the window has rolled past, and traces are archived on their own
                # schedule. `should_notify` is true on exactly the transition into
                # FIRED, so this runs once per episode, not once per evaluation.
                transition = replace(
                    transition,
                    instance=replace(
                        transition.instance,
                        exemplar_trace_ids=collect_exemplar_trace_ids(
                            self.store,
                            rule,
                            observation.window_start_ms,
                            observation.window_end_ms,
                        ),
                    ),
                )
            saved = self.store.save_alert_instance(transition.instance)
            transition = replace(transition, instance=saved)
            # Commit first, then dispatch: ordering is the durability story.
            #
            # `should_notify` is true on exactly the transition into FIRED. A
            # RECOVERED transition is a de-escalation and pages nobody; the next
            # breach opens a new instance and that one notifies normally.
            if transition.should_notify:
                notifications.append(saved)
                if self.notifier is not None:
                    self.notifier(rule, saved)

        self.store.record_alert_rule_evaluated(
            alert_rule_id=rule.alert_rule_id,
            last_evaluated_ms=now_ms,
            next_evaluation_at_ms=next_due_ms(rule, now_ms),
            last_sample_count=observation.sample_count,
        )
        return RuleEvaluation(
            rule=rule, observation=observation, transition=transition, decision=decision
        )
