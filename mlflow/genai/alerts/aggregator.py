"""The 1-minute rollup aggregator.

Every minute this seals the bucket that just closed by scanning six sources and
upserting ``metric_series`` / ``metric_rollups`` rows. This is the one aggregation
Timescale cannot express: a continuous aggregate covers a single hypertable and
cannot join, so the fan-out from raw trace tables to series has to live here.

Each source is one single-table scan covering every experiment, grouped by
experiment plus one or two dimensions. Two sources are scanned once and grouped
twice (``spans``, ``span_errors``, ``span_metrics``), which is why the fan-out is
expressed as a list of groupings over a shared result set rather than as extra
queries.

Buckets are keyed by *completion* time. Start-time bucketing would hold a bucket
open until the slowest trace in it finished, which is unbounded for a long-running
agent; completion is a point event, so buckets seal on a fixed schedule. The
interval is half-open, ``[T, T + 60000)``, which is the entire no-double-count
story: ``floor(t / 60000) * 60000`` maps every row to exactly one bucket.
"""

import logging
import math
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from mlflow.genai.alerts import histogram as hist
from mlflow.genai.alerts.entities import (
    BUCKET_MS,
    MAX_DIMENSION_VALUE_LENGTH,
    MAX_WINDOW_SECONDS,
    METRIC_CATALOGUE,
    SeriesKey,
    is_sliceable,
)
from mlflow.genai.alerts.sketch import GAMMA, ZERO_INDEX, SketchSpec, spec_for
from mlflow.genai.alerts.subscriptions import (
    SeriesFamily,
    Subscription,
    is_subscribed,
    load_active_subscriptions,
)
from mlflow.store.db import db_types
from mlflow.store.tracking.dbmodels.models import (
    SqlAssessments,
    SqlMetricRollup,
    SqlMetricSeries,
    SqlRollupState,
    SqlSpan,
    SqlSpanError,
    SqlSpanMetrics,
    SqlTraceInfo,
    SqlTraceMetrics,
)
from mlflow.store.tracking.utils.sql_trace_metrics_utils import (
    _get_assessment_numeric_value_column,
    _get_json_dimension_column,
)
from mlflow.tracing.constant import SpanAttributeKey, SpanMetricKey, TraceMetricKey

_logger = logging.getLogger(__name__)

LAG_MS = 60_000
"""How long to wait past a bucket's end before sealing it.

Covers out-of-order arrival: the async span exporter runs a queue, so a row whose
completion timestamp falls in bucket T can be committed slightly after T ends.
Timescale's ``end_offset`` is measured against this (see ``timescale.py``).
"""

MAX_BACKFILL_BUCKETS = 60
"""How far behind the watermark may be before the shortfall becomes a gap.

Beyond an hour, catching up minute-by-minute would take longer than the outage.
"""

MAX_GAP_BUCKETS = MAX_WINDOW_SECONDS // (BUCKET_MS // 1000)
"""Cap on how many skipped buckets get an explicit gap marker.

No alert window is longer than ``MAX_WINDOW_SECONDS``, so a bucket further back
than this can never overlap a live evaluation and marking it buys nothing. The cap
is what keeps a multi-day outage from writing days x series gap rows.
"""

_IN_CLAUSE_CHUNK = 500
"""SQLite caps a statement at 999 bound parameters."""

SOURCE_NAMES = (
    "trace_info",
    "trace_error_rate",
    "trace_metrics",
    "spans",
    "span_error_rate",
    "span_metrics",
    "span_errors",
    "assessments",
)
"""Every scan, each with its own watermark row in ``rollup_state``.

Separate watermarks because the sources arrive on different cadences -- an
assessment lands minutes after the trace it scores, and one slow source must not
hold back the rest.

The two ``*_error_rate`` entries re-scan tables another source already reads. A
``_Source`` carries one ``value_column``, and a rate needs a different one from the
latency it sits beside -- so it is a second scan rather than a second aggregate on
the first. Measured at ~0.138s for 300k spans against a 60s budget, and the
established bottleneck is series count rather than scan time.
"""

_LOG_GAMMA_SQL = math.log(GAMMA)
"""``ln(gamma)`` as a literal for the SQL bucket-index expression.

Computed here rather than emitted as ``ln(1.0408)`` so the Python and SQL bucketings
cannot drift by a rounding digit -- a value landing one bucket either side of the
boundary between them would make the stored sketch disagree with the one tests
build in memory.
"""

_MAX_METRIC_KEY_LEN = 250
_MAX_DIMENSION_VALUE_LEN = MAX_DIMENSION_VALUE_LENGTH


@dataclass(frozen=True)
class _Grouping:
    """One series family produced from a scan's result set.

    ``dim_index`` selects which of the scan's dimension columns supplies
    ``dimension_value``; ``None`` means the series is not sliced at all.
    """

    dimension_key: str
    dim_index: int | None = None
    include_total: bool = False
    """Also emit a ``dimension_value=""`` series summing every value of the dimension.

    Needed because the UI maps "number of requests" onto ``latency`` + COUNT with no
    dimension value, while the scan itself is grouped by trace status.
    """

    only_when_sql: Callable[[], sa.ColumnElement] | None = None
    """Restricts this grouping to a subset of the source's rows.

    Pushed into the query rather than applied to fetched rows. That is only possible
    because each grouping is scanned as its own unit of work: when one result set
    was fanned out into several groupings, a predicate excluding rows from *one* of
    them had to be applied in Python afterwards, and the same rule ended up written
    twice -- once in SQL for the rule editor and once in Python for the scan.

    Pushing it down also makes the extra scan cheaper than the one it replaces:
    ``SPAN_NAME``/``latency`` now reads only ``TOOL`` spans instead of every span.
    """


@dataclass(frozen=True)
class _Source:
    """One single-table scan and the series it fans out into."""

    name: str
    time_column: sa.Column
    experiment_column: sa.Column
    dimension_columns: tuple[sa.ColumnElement, ...]
    groupings: tuple[_Grouping, ...]
    series_pairs: tuple[tuple[str, str], ...]
    """``(dimension_key, metric_key)`` pairs this source owns.

    Only used to find the series a gap marker has to be written against. The pairs
    are disjoint across sources, so no two watermarks ever contend for a row.
    """

    metric_key: str | None = None
    metric_key_column: sa.ColumnElement | None = None
    value_column: sa.ColumnElement | None = None
    """``None`` for count-only sources (``span_errors`` has nothing to average)."""

    trace_id_column: sa.ColumnElement | None = None
    """Where a row's trace id lives, for pulling exemplars when a rule fires.

    Not used by aggregation, which only ever writes counts and sums -- this is the
    one thing an alert needs that the rollups deliberately throw away, so it has to
    come from the source table."""

    extra_filters: tuple[sa.ColumnElement, ...] = ()
    join: Callable[[sa.orm.Query], sa.orm.Query] | None = None
    time_scale: int = 1
    """Divisor turning the time column into milliseconds (spans store nanoseconds)."""


@dataclass
class _Accumulator:
    count: int = 0
    total: float | None = None
    """``None`` for count-only sources, so ``error_count`` stores a NULL sum rather
    than a zero that AVG would happily divide."""

    histogram: hist.Histogram | None = None
    """Sparse bucket index -> count. ``None`` for a metric with no sketch spec.

    No width to carry: a sketch bucket is identified by its index, not by a position
    in a fixed-length array, so two accumulators covering different value ranges
    still combine correctly.
    """

    def add(self, count: int, total: float | None, index: int, with_histogram: bool) -> None:
        self.count += count
        if total is not None:
            self.total = (self.total or 0.0) + total
        if with_histogram:
            if self.histogram is None:
                self.histogram = hist.empty()
            self.histogram[index] = self.histogram.get(index, 0) + count


@dataclass(frozen=True)
class WorkUnit:
    """One ``(source, dimension_key, metric_key)`` family -- the unit of sealing.

    Units write disjoint series, so nothing orders them and each carries its own
    watermark row. That is what makes them independently schedulable, and it is also
    what lets a rule be evaluated against the freshness of the one family it reads
    rather than against the slowest family in the deployment.
    """

    source: "_Source"
    grouping: _Grouping
    metric_key: str

    @property
    def name(self) -> str:
        return f"{self.source.name}/{self.grouping.dimension_key}/{self.metric_key}"

    @property
    def key(self) -> tuple[str, str, str]:
        """Primary key of this unit's ``rollup_state`` row."""
        return (self.source.name, self.grouping.dimension_key, self.metric_key)


def rollup_unit_names() -> tuple[str, ...]:
    """Every unit's name, without needing a database connection.

    Unit identity comes from ``series_pairs`` and the grouping's ``dimension_key``,
    both of which are dialect-independent -- only the column *expressions* inside a
    source vary by dialect. So the scheduler can enumerate units at registration
    time, before a tracking store exists.
    """
    return tuple(unit.name for unit in build_work_units(db_types.SQLITE))


def build_work_units(db_type: str, sources: Sequence[str] | None = None) -> tuple[WorkUnit, ...]:
    """Every family the aggregator seals, in a stable order.

    Derived from each source's ``series_pairs``, so the units and the series they own
    cannot disagree.
    """
    units = []
    for source in _build_sources(db_type):
        if sources is not None and source.name not in sources:
            continue
        by_dimension = {g.dimension_key: g for g in source.groupings}
        for dimension_key, metric_key in source.series_pairs:
            units.append(WorkUnit(source, by_dimension[dimension_key], metric_key))
    return tuple(units)


@dataclass(frozen=True)
class UnitResult:
    unit: str
    """``source/dimension_key/metric_key`` -- see :class:`WorkUnit`."""

    watermark_ms: int
    sealed_buckets: list[int] = field(default_factory=list)
    gap_buckets: list[int] = field(default_factory=list)
    rows_written: int = 0

    scan_seconds: float = 0.0
    """Wall time spent scanning raw tables for this unit.

    Split from ``upsert_seconds`` because the two scale with different things and
    have different fixes: the scan grows with traffic, while the upsert grows with
    series count. Measured rather than assumed -- the scan turned out to use well
    under 1% of the per-bucket budget at the design envelope, while the upsert was
    the real constraint.
    """

    upsert_seconds: float = 0.0

    @property
    def source(self) -> str:
        return self.unit.split("/", 1)[0]

    @property
    def total_seconds(self) -> float:
        return self.scan_seconds + self.upsert_seconds


@dataclass(frozen=True)
class AggregationRun:
    now_ms: int
    sealable_max_ms: int
    units: dict[str, UnitResult]

    @property
    def sealed_bucket_count(self) -> int:
        return sum(len(r.sealed_buckets) for r in self.units.values())

    @property
    def total_seconds(self) -> float:
        return sum(r.total_seconds for r in self.units.values())

    def by_source(self) -> dict[str, list[UnitResult]]:
        grouped: dict[str, list[UnitResult]] = {}
        for result in self.units.values():
            grouped.setdefault(result.source, []).append(result)
        return grouped

    def timing_summary(self) -> str:
        """Slowest unit first -- the one that bounds this run."""
        parts = [
            f"{r.unit}={r.scan_seconds:.3f}s scan/{r.upsert_seconds:.3f}s upsert "
            f"({len(r.sealed_buckets)} buckets, {r.rows_written} rows)"
            for r in sorted(self.units.values(), key=lambda r: -r.total_seconds)
            if r.sealed_buckets or r.gap_buckets
        ]
        return "; ".join(parts)


def _span_latency_ms_column() -> sa.ColumnElement:
    """Mirrors ``_get_column_to_aggregate(SPANS, LATENCY)``.

    Kept identical -- including the floor division -- because a rule's observed
    value has to equal the Overview chart's value for the same window.
    """
    return (SqlSpan.end_time_unix_nano - SqlSpan.start_time_unix_nano) // 1_000_000


def _error_indicator(status_column: sa.ColumnElement) -> sa.ColumnElement:
    """1 for a failed row, 0 for a healthy one -- a rate's numerator.

    ``SUM`` of this over a bucket is the failure count while ``COUNT(*)`` is the
    attempt count, so one scan yields both halves of the ratio and ``AVG`` divides
    them. That is what keeps the rate correct across a window: both halves are
    additive, so they merge across buckets and up into the coarse tiers, and the
    incremental cache can subtract them.

    ``else_=0`` and never NULL. Both folds coerce a null sum with ``or 0.0``, so a
    NULL numerator would read as a 0% error rate rather than as missing data --
    a bucket with no failures has to say zero, not nothing.

    Reads ``status`` rather than the exception events behind ``span_errors``.
    ``status`` is the broader signal: several integrations mark a span ERROR without
    recording an event, and for "did this call fail" those are failures. The
    consequence, which is deliberate, is that ``error_rate`` and ``error_count``
    measure different populations -- the count is origin-deduplicated and typed,
    this is per-row and untyped -- and they will not reconcile.
    """
    return sa.case((status_column == "ERROR", 1), else_=0)


def _build_sources(db_type: str) -> tuple[_Source, ...]:
    span_model = _get_json_dimension_column(db_type, SpanAttributeKey.MODEL, "span_model_name")
    return (
        # trace_info: latency by trace status. Its `count` doubles as the request
        # count, which is why there is no separate request_count metric.
        _Source(
            name="trace_info",
            trace_id_column=SqlTraceInfo.request_id,
            time_column=SqlTraceInfo.end_time_ms,
            experiment_column=SqlTraceInfo.experiment_id,
            dimension_columns=(SqlTraceInfo.status,),
            metric_key="latency",
            value_column=SqlTraceInfo.execution_time_ms,
            extra_filters=(SqlTraceInfo.execution_time_ms.isnot(None),),
            groupings=(_Grouping("TRACES", dim_index=0, include_total=True),),
            series_pairs=(("TRACES", "latency"),),
        ),
        # trace_error_rate: the share of requests that failed.
        #
        # Stored as two additive numbers rather than as a rate: `count` is every
        # completed request and `sum` is the failing ones, so AVG divides them at read
        # time. A precomputed per-minute rate could not be merged across buckets --
        # averaging rates weights a quiet minute with one failed request the same as a
        # busy one with a thousand successes, which is the opposite of what an alert
        # should do.
        #
        # Unsliced on purpose. The `trace_info` source above groups by status; slicing a
        # rate by the very column that defines its numerator is meaningless, since the
        # error rate of ERROR traces is 100%.
        _Source(
            name="trace_error_rate",
            trace_id_column=SqlTraceInfo.request_id,
            time_column=SqlTraceInfo.end_time_ms,
            experiment_column=SqlTraceInfo.experiment_id,
            dimension_columns=(),
            metric_key="error_rate",
            value_column=_error_indicator(SqlTraceInfo.status),
            extra_filters=(SqlTraceInfo.execution_time_ms.isnot(None),),
            groupings=(_Grouping("TRACES"),),
            series_pairs=(("TRACES", "error_rate"),),
        ),
        # trace_metrics: token counts. `experiment_id`/`timestamp_ms` are denormalized
        # onto this table precisely so the token rollup needs no EAV join.
        _Source(
            name="trace_metrics",
            trace_id_column=SqlTraceMetrics.request_id,
            time_column=SqlTraceMetrics.timestamp_ms,
            experiment_column=SqlTraceMetrics.experiment_id,
            dimension_columns=(),
            metric_key_column=SqlTraceMetrics.key,
            value_column=SqlTraceMetrics.value,
            extra_filters=(
                SqlTraceMetrics.key.in_(TraceMetricKey.token_usage_keys()),
                SqlTraceMetrics.value.isnot(None),
                SqlTraceMetrics.experiment_id.isnot(None),
            ),
            groupings=(_Grouping("TRACES"),),
            series_pairs=tuple(("TRACES", k) for k in TraceMetricKey.token_usage_keys()),
        ),
        # spans: one scan, two groupings. SPAN_NAME series are written only for TOOL
        # spans -- cardinality control lives here rather than in the schema, which is
        # what bounds them to registered tools instead of every autologged span name.
        _Source(
            name="spans",
            trace_id_column=SqlSpan.trace_id,
            time_column=SqlSpan.end_time_unix_nano,
            time_scale=1_000_000,
            experiment_column=SqlSpan.experiment_id,
            dimension_columns=(SqlSpan.type, SqlSpan.name),
            metric_key="latency",
            value_column=_span_latency_ms_column(),
            extra_filters=(SqlSpan.end_time_unix_nano.isnot(None),),
            groupings=(
                _Grouping("SPAN_TYPE", dim_index=0, include_total=True),
                # The two totals are not the same number, and both are wanted: this
                # one is "every tool span", the one above is "every span".
                _Grouping(
                    "SPAN_NAME",
                    dim_index=1,
                    include_total=True,
                    only_when_sql=lambda: SqlSpan.type == "TOOL",
                ),
            ),
            series_pairs=(("SPAN_TYPE", "latency"), ("SPAN_NAME", "latency")),
        ),
        # span_error_rate: the share of calls that failed, per tool and per span type.
        #
        # The groupings mirror the `spans` source exactly -- including the TOOL
        # restriction on SPAN_NAME -- because numerator and denominator have to come
        # from the same population. Restricting one and not the other would divide tool
        # failures by every span in the trace.
        _Source(
            name="span_error_rate",
            trace_id_column=SqlSpan.trace_id,
            time_column=SqlSpan.end_time_unix_nano,
            time_scale=1_000_000,
            experiment_column=SqlSpan.experiment_id,
            dimension_columns=(SqlSpan.type, SqlSpan.name),
            metric_key="error_rate",
            value_column=_error_indicator(SqlSpan.status),
            # Excludes in-flight spans, not failed ones -- and excludes them from both
            # halves of the ratio.
            extra_filters=(SqlSpan.end_time_unix_nano.isnot(None),),
            groupings=(
                _Grouping("SPAN_TYPE", dim_index=0, include_total=True),
                _Grouping(
                    "SPAN_NAME",
                    dim_index=1,
                    include_total=True,
                    only_when_sql=lambda: SqlSpan.type == "TOOL",
                ),
            ),
            series_pairs=(("SPAN_TYPE", "error_rate"), ("SPAN_NAME", "error_rate")),
        ),
        # span_metrics: cost by model and by span type. The model lives in
        # `spans.dimension_attributes`, so this is the one scan that still joins --
        # see the module note in the stream report.
        _Source(
            name="span_metrics",
            trace_id_column=SqlSpanMetrics.trace_id,
            time_column=SqlSpanMetrics.timestamp_ms,
            experiment_column=SqlSpanMetrics.experiment_id,
            dimension_columns=(span_model, SqlSpan.type),
            metric_key_column=SqlSpanMetrics.key,
            value_column=SqlSpanMetrics.value,
            extra_filters=(
                SqlSpanMetrics.key.in_(SpanMetricKey.cost_keys()),
                SqlSpanMetrics.value.isnot(None),
                SqlSpanMetrics.experiment_id.isnot(None),
            ),
            join=lambda q: q.join(
                SqlSpan,
                sa.and_(
                    SqlSpan.trace_id == SqlSpanMetrics.trace_id,
                    SqlSpan.span_id == SqlSpanMetrics.span_id,
                ),
            ),
            groupings=(
                _Grouping("SPAN_MODEL", dim_index=0, include_total=True),
                _Grouping("SPAN_TYPE", dim_index=1, include_total=True),
            ),
            series_pairs=tuple(
                (dim, k) for dim in ("SPAN_MODEL", "SPAN_TYPE") for k in SpanMetricKey.cost_keys()
            ),
        ),
        # span_errors: one scan, two groupings. "TimeoutErrors spiked" and
        # "search_docs is breaking" are independent questions off the same rows.
        _Source(
            name="span_errors",
            trace_id_column=SqlSpanError.trace_id,
            time_column=SqlSpanError.timestamp_ms,
            experiment_column=SqlSpanError.experiment_id,
            dimension_columns=(SqlSpanError.exception_type, SqlSpanError.span_name),
            metric_key="error_count",
            value_column=None,
            extra_filters=(SqlSpanError.is_origin.is_(True),),
            groupings=(
                _Grouping("ERROR", dim_index=0, include_total=True),
                _Grouping("SPAN_NAME", dim_index=1, include_total=True),
            ),
            series_pairs=(("ERROR", "error_count"), ("SPAN_NAME", "error_count")),
        ),
        # assessments: judge scores, bucketed by when the judge ran rather than when
        # the trace ran. Only `valid` rows count, matching the Overview dashboard, so
        # an overridden verdict is not double-counted.
        _Source(
            name="assessments",
            trace_id_column=SqlAssessments.trace_id,
            time_column=SqlAssessments.created_timestamp,
            experiment_column=SqlAssessments.experiment_id,
            dimension_columns=(SqlAssessments.name,),
            metric_key="assessment_value",
            value_column=_get_assessment_numeric_value_column(SqlAssessments.value),
            extra_filters=(
                SqlAssessments.valid.is_(True),
                SqlAssessments.experiment_id.isnot(None),
            ),
            groupings=(_Grouping("ASSESSMENTS", dim_index=0, include_total=True),),
            series_pairs=(("ASSESSMENTS", "assessment_value"),),
        ),
    )


DIMENSION_VALUE_LOOKBACK_MS = 24 * 60 * 60 * 1000
"""How far back the rule editor looks for dimension values to offer."""


def distinct_dimension_values(
    session,
    db_type: str,
    experiment_id: int,
    metric_key: str,
    dimension_key: str,
    now_ms: int,
    lookback_ms: int = DIMENSION_VALUE_LOOKBACK_MS,
    limit: int = 1000,
) -> list[str]:
    """Values a rule could be scoped to, read from the raw tables.

    Deliberately *not* read from ``metric_series``. That table is written by
    aggregation, so sourcing the rule editor from it makes the feature bootstrap
    into a state it cannot leave: nothing aggregated means an empty dropdown means
    the user cannot create the rule that would have caused the aggregation. The
    deadlock is latent today (everything is aggregated unconditionally) and becomes
    real the moment aggregation is demand-driven.

    Reads the same source table, the same filters and the same join the aggregator
    would use for this ``(metric_key, dimension_key)``, so the values offered are
    exactly the ones that would produce a series -- see :func:`_build_sources`.

    Judge names, tool names and exception types are open vocabularies
    (``make_judge()`` lets a user name a judge anything), so an observed-values
    query is the only honest source; there is no list to hardcode.

    Raises ``ValueError`` for a pair that could never have values, so the caller
    can answer 400. An empty list then means exactly one thing -- nothing has been
    observed yet -- rather than doubling as the answer for a nonsense request.
    """
    if metric_key not in METRIC_CATALOGUE:
        raise ValueError(
            f"Unknown metric {metric_key!r}. Expected one of {sorted(METRIC_CATALOGUE)}."
        )
    if dimension_key not in METRIC_CATALOGUE[metric_key].dimension_keys:
        raise ValueError(
            f"Metric {metric_key!r} cannot be sliced by {dimension_key!r}. Expected one of "
            f"{sorted(METRIC_CATALOGUE[metric_key].dimension_keys)}."
        )
    if not is_sliceable(metric_key, dimension_key):
        raise ValueError(
            f"Metric {metric_key!r} is aggregated across all of {dimension_key!r} rather than "
            "per value, so there are no values to scope it to."
        )

    for source in _build_sources(db_type):
        for grouping in source.groupings:
            if grouping.dimension_key != dimension_key or grouping.dim_index is None:
                continue
            if source.metric_key is not None and source.metric_key != metric_key:
                continue

            column = source.dimension_columns[grouping.dim_index]
            # Drive from the source's own table, not from whatever table the
            # selected column happens to belong to. The cost source selects a model
            # name out of `spans` but must still be driven by `span_metrics`, or the
            # join has nothing to attach to.
            query = session.query(column).select_from(source.experiment_column.class_)
            if source.join is not None:
                query = source.join(query)
            query = query.filter(
                source.experiment_column == int(experiment_id),
                source.time_column >= (now_ms - lookback_ms) * source.time_scale,
                source.time_column < (now_ms + BUCKET_MS) * source.time_scale,
                column.isnot(None),
                *source.extra_filters,
            )
            if source.metric_key_column is not None:
                # EAV sources (tokens, cost) carry the metric in a column rather
                # than in the source itself.
                query = query.filter(source.metric_key_column == metric_key)
            if grouping.only_when_sql is not None:
                query = query.filter(grouping.only_when_sql())

            rows = query.distinct().limit(limit).all()
            return sorted({str(row[0]) for row in rows if row[0] is not None})
    return []


def _index_expression(value_column: sa.ColumnElement, spec: SketchSpec) -> sa.ColumnElement:
    """SQL for the sketch bucket index of ``value_column``.

    Arithmetic rather than a chain of comparisons -- simpler than the 21-branch
    ``CASE`` it replaces, and it grows not at all as accuracy improves. Both
    backends compute it natively: Postgres always has ``ln``, and SQLite has had it
    since 3.35.

    Non-positive values map to the zero bucket, matching :meth:`SketchSpec.index`
    exactly. A skewed span with a negative duration must not abort the seal.
    """
    if not spec.log_scale:
        index = sa.func.ceil(value_column / spec.linear_step)
    else:
        index = sa.func.ceil(sa.func.ln(value_column) / _LOG_GAMMA_SQL)
    return sa.case((value_column <= 0, ZERO_INDEX), else_=sa.cast(index, sa.BigInteger))


def _chunks(items: Sequence, size: int = _IN_CLAUSE_CHUNK) -> Iterator[Sequence]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def floor_bucket(timestamp_ms: int) -> int:
    return (timestamp_ms // BUCKET_MS) * BUCKET_MS


def sealable_max_bucket_ms(now_ms: int, lag_ms: int = LAG_MS) -> int:
    """Start of the newest bucket that has fully ended and is safe to seal.

    ``floor((now - lag) / BUCKET_MS) * BUCKET_MS`` alone returns the bucket being
    stood inside; the extra ``- BUCKET_MS`` takes the last one that fully ended.
    """
    return floor_bucket(now_ms - lag_ms) - BUCKET_MS


class RollupAggregator:
    """Seals 1-minute buckets from raw trace tables into ``metric_rollups``.

    Plain callable with no scheduler of its own, so it can be driven from a
    periodic task, a script, or a test.
    """

    def __init__(
        self,
        store,
        lag_ms: int = LAG_MS,
        max_backfill_buckets: int = MAX_BACKFILL_BUCKETS,
        max_gap_buckets: int = MAX_GAP_BUCKETS,
        only_sources: Sequence[str] | None = None,
        only_units: Sequence[str] | None = None,
    ):
        """
        Args:
            only_sources: restrict to a subset of :data:`SOURCE_NAMES`.
            only_units: restrict to specific ``source/dimension_key/metric_key``
                families. Units are independent -- separate watermark rows and
                disjoint series, so no two can ever contend on the same rollup row --
                which is what lets each be sealed by its own worker rather than all
                of them serially on one connection.
        """
        self._session_maker = store.ManagedSessionMaker
        self._lag_ms = lag_ms
        self._max_backfill_buckets = max_backfill_buckets
        self._max_gap_buckets = max_gap_buckets
        if only_sources is not None and (unknown := sorted(set(only_sources) - set(SOURCE_NAMES))):
            raise ValueError(f"Unknown rollup source(s): {unknown}. Expected {SOURCE_NAMES}.")
        self._units = build_work_units(store.db_type, only_sources)
        if only_units is not None:
            wanted = set(only_units)
            if unknown := sorted(wanted - {u.name for u in build_work_units(store.db_type)}):
                raise ValueError(f"Unknown rollup unit(s): {unknown}.")
            self._units = tuple(u for u in self._units if u.name in wanted)
        self._series_cache: dict[SeriesKey, int] = {}
        self._subscriptions: list[Subscription] = []

    @property
    def units(self) -> tuple[WorkUnit, ...]:
        return self._units

    def run_once(self, now_ms: int | None = None) -> AggregationRun:
        now_ms = int(time.time() * 1000) if now_ms is None else now_ms
        sealable_max = sealable_max_bucket_ms(now_ms, self._lag_ms)
        results: dict[str, UnitResult] = {}
        with self._session_maker(read_only=False) as session:
            # Loaded once per run: what anything actually reads, and therefore what
            # is worth writing. A unit nothing subscribes to still advances its
            # watermark -- so it stays ready the moment a rule does subscribe --
            # but writes no rows.
            self._subscriptions = load_active_subscriptions(session)
            for unit in self._units:
                results[unit.name] = self._run_unit(session, unit, sealable_max)
        return AggregationRun(now_ms=now_ms, sealable_max_ms=sealable_max, units=results)

    def _run_unit(self, session, unit: WorkUnit, sealable_max: int) -> UnitResult:
        watermark = self._read_watermark(session, unit, sealable_max)
        sealed: list[int] = []
        gaps: list[int] = []
        rows_written = 0

        behind = (sealable_max - watermark) // BUCKET_MS
        if behind > self._max_backfill_buckets:
            # ``- 1`` so exactly ``max_backfill_buckets`` buckets get sealed: the loop
            # below is inclusive of ``sealable_max``.
            resume = sealable_max - (self._max_backfill_buckets - 1) * BUCKET_MS
            gaps = self._gap_range(watermark + BUCKET_MS, resume)
            rows_written += self._mark_gaps(session, unit, gaps)
            watermark = resume - BUCKET_MS
            self._write_watermark(session, unit, watermark)
            session.commit()
            _logger.warning(
                "Rollup unit %s was %d buckets behind; marked %d gap buckets and resumed at %d",
                unit.name,
                behind,
                len(gaps),
                resume,
            )

        bucket = watermark + BUCKET_MS
        scan_seconds = 0.0
        upsert_seconds = 0.0
        while bucket <= sealable_max:
            scan_start = time.perf_counter()
            accumulators = self._scan(session, unit, bucket)
            upsert_start = time.perf_counter()
            rows_written += self._upsert(session, accumulators, bucket)
            upsert_end = time.perf_counter()
            scan_seconds += upsert_start - scan_start
            upsert_seconds += upsert_end - upsert_start

            watermark = bucket
            self._write_watermark(session, unit, watermark)
            # Committing per bucket is what makes the loop crash-safe: a restart
            # resumes at the last fully written bucket, never mid-bucket.
            session.commit()
            sealed.append(bucket)
            bucket += BUCKET_MS

        return UnitResult(
            unit=unit.name,
            watermark_ms=watermark,
            sealed_buckets=sealed,
            gap_buckets=gaps,
            rows_written=rows_written,
            scan_seconds=scan_seconds,
            upsert_seconds=upsert_seconds,
        )

    def _gap_range(self, start_ms: int, end_ms: int) -> list[int]:
        buckets = list(range(start_ms, end_ms, BUCKET_MS))
        return buckets[-self._max_gap_buckets :]

    def _read_watermark(self, session, unit: WorkUnit, sealable_max: int) -> int:
        row = session.get(SqlRollupState, unit.key)
        if row is not None:
            return row.watermark_ms
        # A fresh install must not try to backfill from the epoch. Starting one
        # bucket back means the first run seals exactly the bucket that just closed.
        watermark = sealable_max - BUCKET_MS
        source, dimension_key, metric_key = unit.key
        session.add(
            SqlRollupState(
                source=source,
                dimension_key=dimension_key,
                metric_key=metric_key,
                watermark_ms=watermark,
                # The first bucket this unit will seal. Everything earlier predates
                # aggregation, so an empty window reaching back past it means "we
                # were not looking yet" rather than "nothing happened".
                coverage_start_ms=watermark + BUCKET_MS,
                last_updated_ms=int(time.time() * 1000),
            )
        )
        session.flush()
        return watermark

    def _write_watermark(self, session, unit: WorkUnit, watermark_ms: int) -> None:
        row = session.get(SqlRollupState, unit.key)
        row.watermark_ms = watermark_ms
        row.last_updated_ms = int(time.time() * 1000)

    def _scan(self, session, unit: WorkUnit, bucket_start_ms: int) -> dict[SeriesKey, _Accumulator]:
        source = unit.source
        grouping = unit.grouping
        lo = bucket_start_ms * source.time_scale
        hi = (bucket_start_ms + BUCKET_MS) * source.time_scale

        dimension_column = (
            source.dimension_columns[grouping.dim_index] if grouping.dim_index is not None else None
        )
        # Only compute the index when a sketch is actually stored: a metric with no
        # spec (`error_count`) has nothing meaningful to bucket, and grouping by the
        # expression anyway would multiply the result set for nothing.
        spec = spec_for(unit.metric_key)
        with_histogram = spec is not None and source.value_column is not None
        index_column = _index_expression(source.value_column, spec) if with_histogram else None

        group_columns: list[sa.ColumnElement] = [source.experiment_column]
        if dimension_column is not None:
            group_columns.append(dimension_column)
        if index_column is not None:
            group_columns.append(index_column)

        selected = [*group_columns, sa.func.count()]
        if source.value_column is not None:
            selected.append(sa.func.sum(source.value_column))

        query = session.query(*selected).select_from(source.experiment_column.class_)
        if source.join is not None:
            query = source.join(query)
        filters = [
            source.time_column >= lo,
            source.time_column < hi,
            *source.extra_filters,
        ]
        if source.metric_key_column is not None:
            # EAV sources carry several metrics in one table; a unit owns exactly one.
            filters.append(source.metric_key_column == unit.metric_key)
        if grouping.only_when_sql is not None:
            # Pushed down rather than filtered in Python -- possible only because this
            # grouping is scanned on its own. Also narrows the scan.
            filters.append(grouping.only_when_sql())
        query = query.filter(*filters).group_by(*group_columns)

        metric_key = unit.metric_key[:_MAX_METRIC_KEY_LEN]
        accumulators: dict[SeriesKey, _Accumulator] = {}
        for row in query.all():
            experiment_id = row[0]
            if experiment_id is None:
                continue
            cursor = 1
            if dimension_column is not None:
                raw_dimension = row[cursor]
                cursor += 1
            else:
                raw_dimension = None
            index = int(row[cursor]) if index_column is not None else 0
            cursor += 1 if index_column is not None else 0
            count = int(row[cursor])
            # Postgres returns NUMERIC (-> decimal.Decimal) for SUM() over an integer
            # column; SQLite returns a float. Coerce at the boundary, or the first
            # accumulation does `float + Decimal` and raises. Nothing downstream
            # should have to know which dialect produced the row.
            raw_total = row[cursor + 1] if source.value_column is not None else None
            total = None if raw_total is None else float(raw_total)

            for key in self._fan_out(unit, int(experiment_id), metric_key, raw_dimension):
                accumulators.setdefault(key, _Accumulator()).add(
                    count, total, index, with_histogram
                )
        return {
            key: acc
            for key, acc in accumulators.items()
            if is_subscribed(
                self._subscriptions,
                SeriesFamily(unit.grouping.dimension_key, unit.metric_key),
                key.experiment_id,
                key.dimension_value,
            )
        }

    def _fan_out(
        self,
        unit: WorkUnit,
        experiment_id: int,
        metric_key: str,
        raw_dimension: object,
    ) -> Iterator[SeriesKey]:
        grouping = unit.grouping
        if grouping.dim_index is None:
            dimension_value = ""
        else:
            # A NULL dimension cannot name a series; the dashboard drops these data
            # points too, so dropping them keeps the two consistent.
            if raw_dimension is None:
                return
            dimension_value = str(raw_dimension)[:_MAX_DIMENSION_VALUE_LEN]
        yield SeriesKey(
            dimension_key=grouping.dimension_key,
            experiment_id=experiment_id,
            metric_key=metric_key,
            dimension_value=dimension_value,
        )
        if grouping.include_total and dimension_value != "":
            yield SeriesKey(
                dimension_key=grouping.dimension_key,
                experiment_id=experiment_id,
                metric_key=metric_key,
                dimension_value="",
            )

    def _upsert(
        self, session, accumulators: dict[SeriesKey, _Accumulator], bucket_start_ms: int
    ) -> int:
        if not accumulators:
            return 0
        keys = list(accumulators)
        series_ids = self._series_ids(session, keys)
        by_series_id = {series_ids[key]: accumulators[key] for key in keys}
        existing = self._existing_rollup_ids(session, list(by_series_id), bucket_start_ms)

        table = SqlMetricRollup.__table__
        inserts = []
        updates = []
        for series_id, acc in by_series_id.items():
            histogram = acc.histogram
            row = {
                "series_id": series_id,
                "bucket_start_ms": bucket_start_ms,
                "count": acc.count,
                "sum": acc.total,
                # Flattened to interleaved (index, count) pairs -- the shape the
                # Postgres merge aggregate also operates on.
                "histogram": hist.to_pairs(histogram) if histogram else None,
                # Which grid these indices mean. A constant in normal operation --
                # the grid does not move when rules change, which is the point --
                # but stored so a future change to GAMMA stays detectable.
                "boundaries_version": hist.SKETCH_VERSION if histogram is not None else None,
                "is_gap": False,
            }
            # Re-sealing has to be idempotent: a crash between the row write and the
            # watermark write replays the same bucket on restart.
            (updates if series_id in existing else inserts).append(row)

        if inserts:
            session.execute(sa.insert(table), inserts)
        if updates:
            # executemany against the primary key. `bindparam` renames the key
            # columns so they can be used in the WHERE clause without colliding with
            # the SET values.
            session.execute(
                sa
                .update(table)
                .where(
                    table.c.series_id == sa.bindparam("b_series_id"),
                    table.c.bucket_start_ms == sa.bindparam("b_bucket_start_ms"),
                )
                .values(
                    count=sa.bindparam("count"),
                    sum=sa.bindparam("sum"),
                    histogram=sa.bindparam("histogram"),
                    boundaries_version=sa.bindparam("boundaries_version"),
                    is_gap=sa.bindparam("is_gap"),
                ),
                [
                    {
                        **{
                            k: v
                            for k, v in row.items()
                            if k not in ("series_id", "bucket_start_ms")
                        },
                        "b_series_id": row["series_id"],
                        "b_bucket_start_ms": row["bucket_start_ms"],
                    }
                    for row in updates
                ],
            )
        session.flush()
        return len(by_series_id)

    def _existing_rollup_ids(
        self, session, series_ids: list[int], bucket_start_ms: int
    ) -> set[int]:
        """Which of these series already have a row in this bucket.

        Only the ids are needed -- the row is fully overwritten either way -- so
        this deliberately does not materialize ORM objects.
        """
        found: set[int] = set()
        table = SqlMetricRollup.__table__
        for chunk in _chunks(series_ids):
            rows = session.execute(
                sa.select(table.c.series_id).where(
                    table.c.series_id.in_(chunk),
                    table.c.bucket_start_ms == bucket_start_ms,
                )
            ).all()
            found.update(row[0] for row in rows)
        return found

    def _mark_gaps(self, session, unit: WorkUnit, buckets: list[int]) -> int:
        """Record the buckets an outage skipped instead of leaving them absent.

        A missing row is indistinguishable from a genuinely quiet minute, so an
        overlapping window would silently under-count. The marker makes it report
        NO_DATA instead.

        Markers can only be written against series that already exist, since a
        series is created by the scan that first populates it. A unit that has never
        produced a series has nothing an alert rule could be reading, so there is
        nothing to mark.

        Scoped to *this unit's* series. Marking every series a source owns would let
        one unit falling behind report NO_DATA for series that a different, healthy
        unit is still sealing normally.
        """
        if not buckets:
            return 0
        series_ids = [
            row[0]
            for row in session
            .query(SqlMetricSeries.series_id)
            .filter(
                SqlMetricSeries.dimension_key == unit.grouping.dimension_key,
                SqlMetricSeries.metric_key == unit.metric_key,
            )
            .all()
        ]
        if not series_ids:
            return 0
        written = 0
        for bucket in buckets:
            existing = self._existing_rollup_ids(session, series_ids, bucket)
            rows = [
                {
                    "series_id": series_id,
                    "bucket_start_ms": bucket,
                    "count": 0,
                    "sum": None,
                    "histogram": None,
                    "boundaries_version": None,
                    "is_gap": True,
                }
                for series_id in series_ids
                if series_id not in existing
            ]
            if rows:
                session.execute(sa.insert(SqlMetricRollup.__table__), rows)
                written += len(rows)
        session.flush()
        return written

    def _series_ids(self, session, keys: Sequence[SeriesKey]) -> dict[SeriesKey, int]:
        """Resolve every series in one bucket, in a bounded number of round trips.

        Resolving one series at a time cost a SELECT, an INSERT and a SAVEPOINT
        *each*, which is what made sealing dominated by series count rather than by
        traffic: at 5,000 series one bucket of one source took 18.9s of its 60s
        budget, against 0.15s to scan 300,000 spans. This resolves the whole bucket
        in a handful of statements instead.
        """
        resolved: dict[SeriesKey, int] = {}
        missing = []
        for key in keys:
            if (series_id := self._series_cache.get(key)) is not None:
                resolved[key] = series_id
            else:
                missing.append(key)
        if not missing:
            return resolved

        found = self._lookup_series(session, missing)
        resolved.update(found)
        to_create = [key for key in missing if key not in found]
        if not to_create:
            self._series_cache.update(found)
            return resolved

        # A savepoint so losing the race to a concurrent worker does not discard the
        # buckets already written in this transaction. The re-lookup below is what
        # actually resolves the contested rows, so the insert only has to not abort.
        try:
            with session.begin_nested():
                session.execute(
                    sa.insert(SqlMetricSeries.__table__),
                    [
                        {
                            "dimension_key": key.dimension_key,
                            "experiment_id": key.experiment_id,
                            "metric_key": key.metric_key,
                            "dimension_value": key.dimension_value,
                        }
                        for key in to_create
                    ],
                )
        except IntegrityError:
            _logger.debug("Lost a series-creation race for %d series; re-resolving", len(to_create))

        created = self._lookup_series(session, to_create)
        resolved.update(created)
        self._series_cache.update(resolved)
        if len(resolved) != len(keys):
            unresolved = [key for key in keys if key not in resolved]
            raise RuntimeError(f"Could not resolve series ids for {unresolved!r}")
        return resolved

    def _lookup_series(self, session, keys: Sequence[SeriesKey]) -> dict[SeriesKey, int]:
        table = SqlMetricSeries.__table__
        found: dict[SeriesKey, int] = {}
        # Four bind parameters per key, against SQLite's 999-parameter statement cap.
        for chunk in _chunks(keys, size=_IN_CLAUSE_CHUNK // 4):
            rows = session.execute(
                sa.select(
                    table.c.series_id,
                    table.c.dimension_key,
                    table.c.experiment_id,
                    table.c.metric_key,
                    table.c.dimension_value,
                ).where(
                    sa.tuple_(
                        table.c.dimension_key,
                        table.c.experiment_id,
                        table.c.metric_key,
                        table.c.dimension_value,
                    ).in_([
                        (k.dimension_key, k.experiment_id, k.metric_key, k.dimension_value)
                        for k in chunk
                    ])
                )
            ).all()
            for series_id, dimension_key, experiment_id, metric_key, dimension_value in rows:
                found[
                    SeriesKey(
                        dimension_key=dimension_key,
                        experiment_id=experiment_id,
                        metric_key=metric_key,
                        dimension_value=dimension_value,
                    )
                ] = series_id
        return found


def run_rollup_aggregation(
    now_ms: int | None = None,
    sources: Sequence[str] | None = None,
    units: Sequence[str] | None = None,
) -> AggregationRun:
    """Entry point for the every-minute periodic task.

    Registered alongside the other schedulers in
    ``mlflow.server.jobs.utils.register_periodic_tasks``, once per source, so the
    six scans run concurrently rather than serially on a single connection. Safe to
    run more often than once a minute -- it seals whatever buckets have closed since
    the watermark and does nothing when none have.

    Args:
        now_ms: evaluation time; defaults to the wall clock.
        sources: restrict to a subset of :data:`SOURCE_NAMES`.
        units: restrict to specific unit names (see :func:`rollup_unit_names`).
            ``None`` for both runs every unit in one process, which is what a script
            or a test wants.
    """
    # Imported lazily so importing the alerting package does not drag the Flask
    # server stack into a client-only install.
    from mlflow.server.handlers import _get_tracking_store

    return RollupAggregator(_get_tracking_store(), only_sources=sources, only_units=units).run_once(
        now_ms=now_ms
    )
