"""The metric behind a rule, as a line the alert detail view can draw.

**The chart has to agree with the alert**, and that is not automatic. Buckets are
one minute wide but a rule fires on a *windowed* value -- a 10-minute p95, not a
1-minute one -- and the two cross a threshold at different moments. Plotting
per-bucket values would draw a curve that contradicts the alert printed beside it.

So each point here is the same rolling window the evaluator saw: for a step at
``t``, the projection over ``[t - window, t)``. Stepping by the rule's own
evaluation interval means the points land where evaluations landed.

Projection goes through :func:`~mlflow.genai.alerts.rollup_reader.aggregate_buckets`
-- and therefore ``project_aggregate``, which its docstring calls "the one place
this projection is implemented". Two copies of it once disagreed about the empty
window and the evaluator used the wrong one; a chart with its own arithmetic would
be a third copy, free to disagree with both.
"""

from dataclasses import dataclass

from mlflow.genai.alerts.entities import BUCKET_MS, AlertRule, SeriesKey
from mlflow.genai.alerts.rollup_reader import Bucket, RollupReader, aggregate_buckets

MAX_SERIES_POINTS = 500
"""Ceiling on points returned for one range.

Three days at a 5-minute step is 864 windows, and each is a fold over its buckets.
Widening the step instead of refusing keeps a long range cheap and still legible --
a chart cannot show 864 points on a drawer-width canvas anyway.
"""


@dataclass(frozen=True)
class SeriesPoint:
    timestamp_ms: int
    """End of the window this point summarises, which is where evaluation landed."""

    value: float | None
    """``None`` when the window has a gap or no data.

    The two are different upstream -- a gap means aggregation did not run, an
    empty COUNT window is a true zero -- and both arrive here as ``None`` only
    when the projection itself declines to answer. The caller must break the line
    rather than draw through it: joining across a gap draws a slope nobody
    measured.
    """

    sample_count: int
    is_gap: bool


@dataclass(frozen=True)
class RuleSeries:
    points: list[SeriesPoint]
    threshold: float
    window_seconds: int
    step_seconds: int


def series_key_for(rule: AlertRule) -> SeriesKey:
    return SeriesKey(
        dimension_key=rule.dimension_key,
        experiment_id=rule.experiment_id,
        metric_key=rule.metric_key,
        dimension_value=rule.dimension_value or "",
    )


def choose_step_ms(start_ms: int, end_ms: int, interval_seconds: int) -> int:
    """The rule's own interval, widened until the range fits in the point cap.

    Rounded up to a whole bucket: a step finer than one minute would produce
    consecutive points over identical bucket sets, which costs a fold each and
    draws a flat segment that looks like real data.
    """
    step_ms = max(BUCKET_MS, interval_seconds * 1000)
    span_ms = max(0, end_ms - start_ms)
    if span_ms // step_ms > MAX_SERIES_POINTS:
        needed = -(-span_ms // MAX_SERIES_POINTS)  # ceil
        step_ms = -(-needed // BUCKET_MS) * BUCKET_MS
    return step_ms


def compute_rule_series(
    reader: RollupReader,
    rule: AlertRule,
    start_ms: int,
    end_ms: int,
) -> RuleSeries:
    """Rolling-window values for ``rule`` across ``[start_ms, end_ms]``.

    One read covers every window: the earliest one reaches back a full
    ``window_seconds`` before ``start_ms``, so the range fetched is wider than the
    range plotted. Folding in memory keeps this to a single query rather than one
    per point.
    """
    window_ms = rule.window_seconds * 1000
    series = series_key_for(rule)
    spec = reader.sketch_for(series)

    # Never plot past the watermark. Windows reaching into unsealed time contain
    # fewer and fewer sealed buckets, and a COUNT over a partly-empty window is a
    # real, shrinking number -- so the line ramps smoothly to zero at the right
    # edge, which for an absence rule is the exact shape of a genuine outage.
    #
    # This is the same bound `evaluator.evaluation_window` derives, and the reason
    # the evaluator cannot make this mistake: it takes its window *from* the
    # watermark rather than from the clock.
    end_ms = min(end_ms, reader.latest_sealed_bucket_ms(series) + BUCKET_MS)
    step_ms = choose_step_ms(start_ms, end_ms, rule.evaluation_interval_seconds)

    buckets = reader.read_buckets(series, start_ms - window_ms, end_ms)
    by_start: dict[int, Bucket] = {b.bucket_start_ms: b for b in buckets}

    points: list[SeriesPoint] = []
    # Aligned to the step grid so the same range always yields the same
    # timestamps; an unaligned start would shift every point when the caller
    # scrolls by a few seconds.
    first = -(-start_ms // step_ms) * step_ms
    for end in range(first, end_ms + 1, step_ms):
        window_start = end - window_ms
        in_window = [
            by_start[start]
            for start in range(-(-window_start // BUCKET_MS) * BUCKET_MS, end, BUCKET_MS)
            if start in by_start
        ]
        observation = aggregate_buckets(
            in_window,
            rule.aggregation,
            window_start,
            end,
            rule.percentile_value,
            spec,
        )
        points.append(
            SeriesPoint(
                timestamp_ms=end,
                value=observation.observed_value,
                sample_count=observation.sample_count,
                is_gap=any(b.is_gap for b in in_window),
            )
        )

    return RuleSeries(
        points=points,
        threshold=rule.threshold,
        window_seconds=rule.window_seconds,
        step_seconds=step_ms // 1000,
    )
