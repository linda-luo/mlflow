"""Entities and contracts shared by every alerting component.

This module is the seam between the ingest, aggregation, evaluation and API
layers. Nothing here touches a database or a clock, so it can be imported from
any of them without creating a cycle.
"""

from dataclasses import dataclass, field
from typing import Literal

DimensionKey = Literal[
    "TRACES",
    "SPAN_TYPE",
    "SPAN_NAME",
    "SPAN_MODEL",
    "ASSESSMENTS",
    "ERROR",
]

Aggregation = Literal["COUNT", "SUM", "AVG", "PERCENTILE"]

Comparator = Literal["GT", "GTE", "LT", "LTE"]

Severity = Literal["LOW", "MEDIUM", "HIGH"]

AlertState = Literal["PENDING", "FIRED", "INACTIVE", "DISMISSED"]
"""The lifecycle of one firing episode.

``INACTIVE`` is "fired, has since recovered, nobody has acknowledged it yet". It
is still on screen and still dismissible -- the distinction it preserves is
between *a human reviewed this* (DISMISSED) and *it stopped on its own*.
"""

# Reserved ``dismissed_by`` values. An instance closed by the system rather than
# by a person reuses DISMISSED, distinguished only by the actor string.
SYSTEM_DISMISS_RULE_EDITED = "system:rule_edited"
SYSTEM_DISMISS_RULE_DISABLED = "system:rule_disabled"
SYSTEM_DISMISS_RULE_DELETED = "system:rule_deleted"
SYSTEM_DISMISS_NOT_SUSTAINED = "system:not_sustained"

BUCKET_MS = 60_000
"""Width of the finest rollup bucket. Every window boundary is a multiple of this."""

MIN_WINDOW_SECONDS = 300
MAX_WINDOW_SECONDS = 259_200
"""Five minutes to three days.

The floor is five buckets, so a window always spans several and one late arrival
cannot dominate it. Below it the derived interval hits its own floor and the overlap
between consecutive evaluations collapses, leaving stretches nothing inspects.

**The ceiling is raw retention, and that is not a coincidence.** It equals
``timescale.RAW_RETENTION_MS`` exactly. A window can only be answered from data that
still exists, and the incremental merge path expires buckets by re-reading them at
one-minute granularity -- so a window reaching past retention reads an empty range,
subtracts nothing, and silently stops shedding. Nothing checks that the expiring
range returned the rows it should, so the symptom is an observation that drifts up
for an hour and snaps back at the full recompute.

Timescale drops whole chunks, and only once a chunk's entire range is older than the
cutoff, so the row at exactly ``now - 3d`` is always inside a surviving chunk. That
holds for any chunk width; day-wide chunks and the retention job's cadence only add
margin on top.

Raising this past ``RAW_RETENTION_MS`` requires raising retention with it. A test
pins the two together.
"""

MIN_EVALUATION_INTERVAL_SECONDS = 60
MAX_EVALUATION_INTERVAL_SECONDS = 300

MAX_DIMENSION_VALUE_LENGTH = 250
"""Width of ``metric_series.dimension_value``, and of the rule column naming it.

Observed values are truncated to this on the way in, so a rule naming a longer
value could never match the series it means.
"""


@dataclass(frozen=True)
class MetricSpec:
    dimension_keys: frozenset[str]
    aggregations: frozenset[str]
    unsliceable_dimension_keys: frozenset[str] = frozenset()
    """Dimensions this metric is grouped *under* but not sliced *by*.

    A metric can legally name a dimension while being aggregated with no
    per-value breakdown -- whole-trace token counts are grouped under ``TRACES``
    but written with an empty ``dimension_value``, because there is no second
    column to group on. A rule naming a value for one of these addresses a series
    that can never exist, so it would be accepted and then silently never fire.
    """


METRIC_CATALOGUE: dict[str, MetricSpec] = {
    "latency": MetricSpec(
        dimension_keys=frozenset({"TRACES", "SPAN_TYPE", "SPAN_NAME"}),
        aggregations=frozenset({"COUNT", "AVG", "PERCENTILE"}),
    ),
    "total_tokens": MetricSpec(
        dimension_keys=frozenset({"TRACES"}),
        aggregations=frozenset({"SUM", "AVG", "PERCENTILE"}),
        # Trace latency is sliced by trace status; trace token counts are not
        # sliced at all. The `trace_metrics` source has no `dim_index`.
        unsliceable_dimension_keys=frozenset({"TRACES"}),
    ),
    # The components of `total_tokens`. The aggregator has always written units for
    # these and the sketch layer has always had grids for them -- only the catalogue
    # entry was missing, so `validate_metric_triple` rejected every rule naming one
    # and the work unit scanned every minute and discarded the result.
    #
    # They are not redundant with the total. A prompt-bloat regression moves
    # `input_tokens` while `output_tokens` holds; a verbosity regression does the
    # reverse; and a prompt-cache regression shows up in `cache_read_input_tokens`
    # while the total barely moves. The three have different fixes.
    "input_tokens": MetricSpec(
        dimension_keys=frozenset({"TRACES"}),
        aggregations=frozenset({"SUM", "AVG", "PERCENTILE"}),
        unsliceable_dimension_keys=frozenset({"TRACES"}),
    ),
    "output_tokens": MetricSpec(
        dimension_keys=frozenset({"TRACES"}),
        aggregations=frozenset({"SUM", "AVG", "PERCENTILE"}),
        unsliceable_dimension_keys=frozenset({"TRACES"}),
    ),
    "cache_read_input_tokens": MetricSpec(
        dimension_keys=frozenset({"TRACES"}),
        aggregations=frozenset({"SUM", "AVG", "PERCENTILE"}),
        unsliceable_dimension_keys=frozenset({"TRACES"}),
    ),
    "cache_creation_input_tokens": MetricSpec(
        dimension_keys=frozenset({"TRACES"}),
        aggregations=frozenset({"SUM", "AVG", "PERCENTILE"}),
        unsliceable_dimension_keys=frozenset({"TRACES"}),
    ),
    "total_cost": MetricSpec(
        dimension_keys=frozenset({"SPAN_MODEL", "SPAN_TYPE"}),
        aggregations=frozenset({"SUM", "AVG", "PERCENTILE"}),
    ),
    # Same story as the token components, and sliced the same way as their total:
    # `span_metrics` carries a model and a span type, so both are per-value.
    "input_cost": MetricSpec(
        dimension_keys=frozenset({"SPAN_MODEL", "SPAN_TYPE"}),
        aggregations=frozenset({"SUM", "AVG", "PERCENTILE"}),
    ),
    "output_cost": MetricSpec(
        dimension_keys=frozenset({"SPAN_MODEL", "SPAN_TYPE"}),
        aggregations=frozenset({"SUM", "AVG", "PERCENTILE"}),
    ),
    "error_count": MetricSpec(
        dimension_keys=frozenset({"ERROR", "SPAN_NAME"}),
        aggregations=frozenset({"COUNT"}),
    ),
    # The share of calls that failed, as opposed to how many did.
    #
    # A count is a spike detector: it rises with traffic and falls silent when
    # traffic collapses. A rate is what catches a regression that scales -- "this
    # tool fails 8% of the time" holds at any volume. They answer different
    # questions, so both exist.
    #
    # AVG only, and that is load-bearing. The series stores the denominator in
    # `count` and the numerator in `sum`, so AVG is the ratio. SUM and COUNT would
    # read back as "failures including propagated" and "invocations", which
    # disagree with `error_count` and with `latency` + COUNT respectively.
    # PERCENTILE is excluded by having no sketch spec at all: a ratio has nothing
    # to bucket, and `project_aggregate` raises rather than reporting no-data.
    #
    # TRACES is unsliceable here because the trace rate is not broken down by
    # status -- slicing a rate by the column that defines its numerator would make
    # the error rate of ERROR traces 100% by construction.
    "error_rate": MetricSpec(
        dimension_keys=frozenset({"TRACES", "SPAN_TYPE", "SPAN_NAME"}),
        aggregations=frozenset({"AVG"}),
        unsliceable_dimension_keys=frozenset({"TRACES"}),
    ),
    "assessment_value": MetricSpec(
        dimension_keys=frozenset({"ASSESSMENTS"}),
        aggregations=frozenset({"AVG", "PERCENTILE", "COUNT"}),
    ),
}
"""Legal ``(metric_key, dimension_key, aggregation)`` triples.

Single source of truth for both the UI dropdowns and :func:`validate_metric_triple`,
so the form cannot offer a combination the evaluator would reject.
"""


def percentile_supported_metrics() -> frozenset[str]:
    """Metrics that can answer a PERCENTILE query.

    A percentile needs a sketch, and a sketch needs a grid whose buckets mean
    something in that metric's own unit -- a token count bucketed on a duration
    grid would read back as milliseconds. So this is exactly the set of metrics
    with a sketch spec.

    Imported lazily to keep ``entities`` free of a dependency on ``sketch``, which
    is the direction the rest of the package imports in.
    """
    from mlflow.genai.alerts.sketch import SPECS

    return frozenset(SPECS)


def validate_metric_triple(metric_key: str, dimension_key: str, aggregation: str) -> None:
    """Reject a rule the evaluator could not answer. Raises ``ValueError``.

    Called on create and update so an unanswerable rule can never be persisted --
    catching it here rather than at evaluation time is the difference between a
    form error and a rule that silently never fires.
    """
    spec = METRIC_CATALOGUE.get(metric_key)
    if spec is None:
        raise ValueError(
            f"Unknown metric {metric_key!r}. Expected one of {sorted(METRIC_CATALOGUE)}."
        )
    if dimension_key not in spec.dimension_keys:
        raise ValueError(
            f"Metric {metric_key!r} cannot be sliced by {dimension_key!r}. "
            f"Expected one of {sorted(spec.dimension_keys)}."
        )
    if aggregation not in spec.aggregations:
        raise ValueError(
            f"Metric {metric_key!r} does not support {aggregation!r}. "
            f"Expected one of {sorted(spec.aggregations)}."
        )
    if aggregation == "PERCENTILE" and metric_key not in (
        supported := percentile_supported_metrics()
    ):
        raise ValueError(
            f"PERCENTILE on {metric_key!r} is not supported: percentiles are served from a "
            f"sketch, and sketch grids are only defined for {sorted(supported)}. "
            "Use AVG or SUM instead."
        )


def is_sliceable(metric_key: str, dimension_key: str) -> bool:
    """Whether a rule on this pair may name a ``dimension_value``.

    Shared with the UI, which uses it to decide whether to render a value control
    at all -- the alternative is a form that offers a value the aggregator never
    writes.
    """
    spec = METRIC_CATALOGUE.get(metric_key)
    return spec is not None and dimension_key not in spec.unsliceable_dimension_keys


def validate_dimension_value(metric_key: str, dimension_key: str, dimension_value) -> None:
    """Reject a ``dimension_value`` that could never match a series. Raises ``ValueError``.

    Separate from :func:`validate_metric_triple` because it is about the *value*,
    not the triple, and because an empty value is always legal.
    """
    if not dimension_value:
        return
    if not is_sliceable(metric_key, dimension_key):
        raise ValueError(
            f"Metric {metric_key!r} is aggregated across all of {dimension_key!r} rather than "
            f"per value, so it cannot be scoped to {dimension_value!r}. Leave the value empty."
        )
    if len(dimension_value) > MAX_DIMENSION_VALUE_LENGTH:
        # The aggregator truncates observed values to the column width, so a
        # longer value could never match the series it names.
        raise ValueError(
            f"`dimension_value` must be at most {MAX_DIMENSION_VALUE_LENGTH} characters, "
            f"got {len(dimension_value)}."
        )


@dataclass(frozen=True)
class SeriesKey:
    """Identifies one rollup series. Maps 1:1 onto a ``metric_series`` row."""

    dimension_key: str
    experiment_id: int
    metric_key: str
    dimension_value: str = ""


@dataclass(frozen=True)
class Observation:
    """What the evaluator read for one window. ``None`` value means no data."""

    observed_value: float | None
    sample_count: int
    window_start_ms: int
    window_end_ms: int

    @property
    def has_data(self) -> bool:
        """Whether this window produced an answer.

        Deliberately not ``sample_count > 0``: a COUNT over an empty window has an
        answer (zero) despite having no samples, and that is the case an
        absence-signal rule needs. Windows with no answer report
        ``observed_value is None``.
        """
        return self.observed_value is not None


@dataclass
class AlertRule:
    alert_rule_id: str
    experiment_id: int
    name: str
    metric_key: str
    dimension_key: str
    aggregation: str
    comparator: str
    threshold: float
    window_seconds: int
    evaluation_interval_seconds: int
    dimension_value: str | None = None
    percentile_value: float | None = None
    sustain_seconds: int = 0
    min_sample_count: int = 0
    """Samples the window must hold before the rule may fire at all.

    Zero -- no floor -- rather than one. A floor of one makes "traffic dropped to
    zero" unfireable, which is the one rule whose whole signal is an empty window.
    The store used to override this default on every create, so it only ever bit a
    rule constructed directly.
    """
    severity: str = "MEDIUM"
    enabled: bool = True
    last_evaluated_ms: int | None = None
    next_evaluation_at_ms: int | None = None
    last_sample_count: int | None = None
    deleted_at_ms: int | None = None
    channels: list[dict] = field(default_factory=list)
    created_by: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None

    @property
    def series_key(self) -> SeriesKey:
        return SeriesKey(
            dimension_key=self.dimension_key,
            experiment_id=self.experiment_id,
            metric_key=self.metric_key,
            dimension_value=self.dimension_value or "",
        )

    @property
    def effective_sustain_seconds(self) -> int:
        """Sustain rounded up to the evaluation interval.

        A sustain shorter than the interval cannot be observed, so surfacing the
        rounded value keeps the UI honest about what the rule actually does.
        """
        if self.sustain_seconds <= 0:
            return 0
        interval = self.evaluation_interval_seconds
        return -(-self.sustain_seconds // interval) * interval


@dataclass
class AlertInstance:
    """One firing episode. A rule that fires twice has two instances."""

    alert_instance_id: str
    alert_rule_id: str
    experiment_id: int
    state: str
    started_at_ms: int
    window_start_ms: int
    window_end_ms: int
    observed_value: float | None = None
    peak_value: float | None = None
    threshold: float | None = None
    sample_count: int = 0
    fired_at_ms: int | None = None
    dismissed_at_ms: int | None = None
    dismissed_by: str | None = None
    healthy_since_ms: int | None = None
    """Start of the current unbroken run of healthy evaluations, or None.

    Set on the first healthy evaluation after a breach and cleared by the next
    breaching one, so it is both the "has this been healthy before?" flag that
    :func:`~mlflow.genai.alerts.state_machine.evaluate_transition` needs and,
    once the instance reaches INACTIVE, the time it actually recovered.

    Deliberately not a counter of healthy evaluations: an instance needs to know
    *when* the recovery began to compare it against ``sustain_seconds``, and a
    count cannot answer that.
    """

    exemplar_trace_ids: list[str] = field(default_factory=list)


TransitionEvent = Literal[
    "NONE",
    "OPENED",
    "FIRED",
    "UPDATED",
    "RECOVERED",
    "CLOSED_NOT_SUSTAINED",
]


@dataclass(frozen=True)
class Transition:
    """The state machine's entire output. Pure data — the caller does the I/O."""

    event: str
    state: str | None
    instance: AlertInstance | None
    should_notify: bool = False


def derive_evaluation_interval_seconds(window_seconds: int) -> int:
    """Ten evaluations per window, clamped.

    Consecutive evaluations must overlap by ``window - interval`` so no breach
    can slip between them; the clamp keeps that ratio sane at both extremes.
    """
    return max(
        MIN_EVALUATION_INTERVAL_SECONDS,
        min(MAX_EVALUATION_INTERVAL_SECONDS, window_seconds // 10),
    )


def derive_min_sample_count(aggregation: str, percentile_value: float | None) -> int:
    """How many samples a percentile needs before it means anything.

    Roughly ten observations in the smaller tail: p50 needs 20, p95 needs 200,
    p99 needs 1,000. Agent traffic is low-volume, so this is the number that
    decides whether a rule can ever fire.

    A *suggestion*, not a policy. The form seeds its input from this and the store
    falls back to it when a client sends nothing, but whatever the user types wins
    -- including zero. It was previously imposed and never shown, so a p99 rule on
    a workload that produces 500 samples a window sat permanently in
    INSUFFICIENT_DATA with nothing on screen saying why, unable to open *or close*
    an instance.
    """
    if aggregation != "PERCENTILE" or percentile_value is None:
        # Zero, not one: the guard exists to stop a percentile being computed from
        # too few points. A COUNT rule has no such floor, and requiring one sample
        # would make "requests dropped to zero" unfireable.
        return 0
    tail = min(percentile_value, 100.0 - percentile_value)
    if tail <= 0:
        raise ValueError(
            f"percentile_value must be strictly between 0 and 100, got {percentile_value}"
        )
    return int(10 / (tail / 100.0))
