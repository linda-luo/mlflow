import pytest

from mlflow.genai.alerts.entities import (
    BUCKET_MS,
    AlertRule,
    SeriesKey,
    derive_evaluation_interval_seconds,
)
from mlflow.genai.alerts.evaluator import AlertEvaluator
from mlflow.genai.alerts.rollup_reader import Bucket, FakeRollupReader
from mlflow.genai.alerts.series import (
    MAX_SERIES_POINTS,
    choose_step_ms,
    compute_rule_series,
)

MINUTE = 60_000
T0 = (1_700_000_000_000 // BUCKET_MS) * BUCKET_MS
SERIES = SeriesKey(dimension_key="TRACES", experiment_id=7, metric_key="latency")


def make_rule(**overrides) -> AlertRule:
    window_seconds = overrides.pop("window_seconds", 600)
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
        "min_sample_count": 0,
    }
    kwargs.update(overrides)
    return AlertRule(**kwargs)


def seed(reader: FakeRollupReader, buckets: int, value_ms: float, count: int = 20) -> None:
    """Constant traffic, via the reader's own seeder so buckets match production shape."""
    reader.seed_constant(SERIES, T0, T0 + buckets * BUCKET_MS, count, value_ms)


class InMemoryStore:
    """Just enough store for `AlertEvaluator` to record against."""

    def __init__(self, rules):
        self.rules = rules

    def lease_due_alert_rules(self, *args, **kwargs):
        return list(self.rules)

    def get_open_alert_instance(self, *args, **kwargs):
        return None

    def save_alert_instance(self, instance):
        return instance

    def record_alert_rule_evaluated(self, **kwargs):
        pass


###############################################################################
# The chart has to agree with the alert
###############################################################################


@pytest.mark.parametrize("aggregation", ["COUNT", "AVG", "PERCENTILE"])
def test_a_series_point_equals_what_the_evaluator_read_for_that_window(aggregation):
    """The property the whole feature rests on.

    A chart that projects differently from the evaluator draws a line crossing the
    threshold somewhere the alert did not, which makes the alert look wrong. Both
    sides go through ``project_aggregate`` precisely so this cannot drift.
    """
    reader = FakeRollupReader(latest_sealed_bucket_ms=T0 + 29 * BUCKET_MS)
    seed(reader, buckets=30, value_ms=50 * MINUTE)
    rule = make_rule(aggregation=aggregation, window_seconds=600)

    cycle = AlertEvaluator(reader, InMemoryStore([rule])).evaluate_rules(
        [rule], now_ms=T0 + 30 * BUCKET_MS
    )
    (evaluation,) = cycle.evaluations
    window = evaluation.observation

    series = compute_rule_series(reader, rule, window.window_end_ms, window.window_end_ms)
    (point,) = series.points

    assert point.timestamp_ms == window.window_end_ms
    assert point.value == evaluation.observation.observed_value
    assert point.sample_count == evaluation.observation.sample_count


def test_the_threshold_travels_with_the_series():
    reader = FakeRollupReader(latest_sealed_bucket_ms=T0 + 29 * BUCKET_MS)
    seed(reader, buckets=30, value_ms=50 * MINUTE)
    rule = make_rule(threshold=30 * MINUTE)

    series = compute_rule_series(reader, rule, T0 + 20 * BUCKET_MS, T0 + 25 * BUCKET_MS)

    assert series.threshold == 30 * MINUTE
    assert series.window_seconds == rule.window_seconds


###############################################################################
# Gaps
###############################################################################


def test_a_gapped_window_is_blank_rather_than_zero():
    """Drawing through a gap invents a slope nobody measured.

    A gap means aggregation did not run. Rendering it as ``0`` would show a crash
    to zero -- which for an absence rule is exactly the shape of a real incident.
    """
    reader = FakeRollupReader(latest_sealed_bucket_ms=T0 + 29 * BUCKET_MS)
    seed(reader, buckets=30, value_ms=50 * MINUTE)
    reader.seed(SERIES, Bucket(bucket_start_ms=T0 + 15 * BUCKET_MS, count=0, is_gap=True))
    rule = make_rule(aggregation="COUNT", window_seconds=600)

    series = compute_rule_series(reader, rule, T0 + 10 * BUCKET_MS, T0 + 29 * BUCKET_MS)

    blanked = [p for p in series.points if p.value is None]
    assert blanked
    assert all(p.is_gap for p in blanked)
    # Exactly the windows containing the gap: a 10-minute window at a 1-minute
    # step means ten of them, and it leaves once it expires.
    assert len(blanked) == 10


###############################################################################
# The point cap
###############################################################################


def test_a_three_day_range_stays_under_the_point_cap():
    """864 windows at the ceiling, each a fold. Widen the step instead."""
    three_days_ms = 3 * 24 * 60 * 60 * 1000
    step_ms = choose_step_ms(0, three_days_ms, interval_seconds=300)

    assert three_days_ms // step_ms <= MAX_SERIES_POINTS
    assert step_ms % BUCKET_MS == 0


def test_a_short_range_keeps_the_rule_own_interval():
    step_ms = choose_step_ms(0, 60 * MINUTE, interval_seconds=300)
    assert step_ms == 300 * 1000


def test_the_step_never_goes_below_one_bucket():
    """Sub-bucket steps fold identical bucket sets and draw a flat lie."""
    assert choose_step_ms(0, 10 * MINUTE, interval_seconds=1) == BUCKET_MS


def test_points_are_aligned_to_the_step_grid():
    """Otherwise scrolling by a few seconds shifts every point on the chart."""
    reader = FakeRollupReader(latest_sealed_bucket_ms=T0 + 29 * BUCKET_MS)
    seed(reader, buckets=30, value_ms=50 * MINUTE)
    rule = make_rule(window_seconds=600)

    series = compute_rule_series(reader, rule, T0 + 10 * BUCKET_MS + 137, T0 + 20 * BUCKET_MS)

    step_ms = series.step_seconds * 1000
    assert all(p.timestamp_ms % step_ms == 0 for p in series.points)


def test_an_empty_range_yields_no_points():
    reader = FakeRollupReader(latest_sealed_bucket_ms=T0)
    rule = make_rule()
    series = compute_rule_series(reader, rule, T0 + BUCKET_MS, T0)
    assert series.points == []


###############################################################################
# The right edge
###############################################################################


def test_the_line_stops_at_the_watermark_rather_than_ramping_to_zero():
    """Windows past the watermark are not "no traffic", they are "not sealed yet".

    Each successive window past it holds fewer sealed buckets, so a COUNT over one
    is a real, shrinking number -- the chart drew a smooth collapse to zero at its
    right edge that never happened, which for an absence rule is exactly the shape
    of a genuine outage. Observed live: a server at 08:39 returning points out to
    08:49, decaying 272 -> 0.
    """
    watermark = T0 + 29 * BUCKET_MS
    reader = FakeRollupReader(latest_sealed_bucket_ms=watermark)
    seed(reader, buckets=30, value_ms=50 * MINUTE)
    rule = make_rule(aggregation="COUNT", window_seconds=600)

    # Ask for ten minutes past the last sealed bucket, as the drawer does.
    series = compute_rule_series(reader, rule, T0 + 20 * BUCKET_MS, watermark + 10 * BUCKET_MS)

    assert series.points
    assert max(p.timestamp_ms for p in series.points) <= watermark + BUCKET_MS
    # And nothing decays: every window plotted is fully sealed.
    values = [p.value for p in series.points]
    assert all(v == values[0] for v in values)


def test_a_range_entirely_past_the_watermark_plots_nothing():
    """Better an empty chart than an invented one."""
    watermark = T0 + 29 * BUCKET_MS
    reader = FakeRollupReader(latest_sealed_bucket_ms=watermark)
    seed(reader, buckets=30, value_ms=50 * MINUTE)
    rule = make_rule(aggregation="COUNT", window_seconds=600)

    series = compute_rule_series(
        reader, rule, watermark + 5 * BUCKET_MS, watermark + 20 * BUCKET_MS
    )

    assert series.points == []
