"""The sketch's accuracy guarantee, asserted rather than trusted.

Everything else in the alerting stack rests on one claim: a quantile read back from
the sketch is within ``ALPHA`` *relative* error of the true one, at every magnitude.
If that does not hold, no threshold comparison built on it means anything -- so it
is asserted directly against brute-force quantiles over the raw values, not inferred
from the fact that gamma was computed correctly.
"""

import random

import pytest

from mlflow.genai.alerts import histogram as hist
from mlflow.genai.alerts.rollup_reader import Bucket, aggregate_buckets
from mlflow.genai.alerts.sketch import (
    ALPHA,
    LOG_SKETCH,
    SCORE_SKETCH,
    ZERO_INDEX,
    SketchSpec,
    spec_for,
)

MINUTE = 60_000


def true_quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def build(values, spec: SketchSpec = LOG_SKETCH) -> hist.Histogram:
    h = hist.empty()
    for value in values:
        hist.observe(h, value, spec)
    return h


###############################################################################
# The guarantee
###############################################################################


@pytest.mark.parametrize(
    ("name", "generate"),
    [
        ("uniform", lambda r: r.uniform(1, 10_000)),
        ("lognormal", lambda r: r.lognormvariate(8, 2)),
        ("heavy tail", lambda r: r.paretovariate(1.2) * 100),
        ("tight cluster", lambda r: r.uniform(999, 1001)),
    ],
)
@pytest.mark.parametrize("q", [0.5, 0.95, 0.99])
def test_quantiles_are_within_the_relative_error_guarantee(name, generate, q):
    rng = random.Random(11)
    values = [generate(rng) for _ in range(20_000)]

    estimate = hist.quantile(build(values), q, LOG_SKETCH)

    truth = true_quantile(values, q)
    assert abs(estimate - truth) / truth <= ALPHA


@pytest.mark.parametrize(
    "magnitude",
    [1e-3, 1e-1, 1e0, 1e2, 1e4, 1e6, 1e8],
)
def test_accuracy_does_not_degrade_with_magnitude(magnitude):
    """Relative error is flat across the range; only absolute error scales.

    This is the property the design is chosen for and the one a reader would most
    reasonably doubt, so it is swept across seven orders of magnitude rather than
    checked at one convenient point.
    """
    rng = random.Random(3)
    values = [rng.uniform(magnitude, magnitude * 10) for _ in range(5_000)]

    estimate = hist.quantile(build(values), 0.95, LOG_SKETCH)

    truth = true_quantile(values, 0.95)
    assert abs(estimate - truth) / truth <= ALPHA


def test_one_sketch_spans_a_microsecond_and_a_thirty_hour_run():
    """The unbounded range, which a fixed-width array could not have represented.

    The boundaries this replaces stopped at 4 hours and put everything beyond into
    a single bucket whose error was unbounded.
    """
    values = [0.001] * 500 + [30 * 3600 * 1000.0] * 500

    h = build(values)

    assert hist.quantile(h, 0.10, LOG_SKETCH) == pytest.approx(0.001, rel=ALPHA)
    assert hist.quantile(h, 0.90, LOG_SKETCH) == pytest.approx(30 * 3600 * 1000, rel=ALPHA)


def test_the_estimate_is_not_the_bucket_upper_edge():
    """The regression behind `design-critique.md` section 4.

    Returning the containing bucket's upper edge made a true p95 of ~90 minutes
    display as 120. The representative value is within alpha in both directions.
    """
    values = [90 * MINUTE] * 1000

    estimate = hist.quantile(build(values), 0.95, LOG_SKETCH)

    assert estimate == pytest.approx(90 * MINUTE, rel=ALPHA)
    assert estimate < 91 * MINUTE


###############################################################################
# Thresholds, and the absence of snapping
###############################################################################


@pytest.mark.parametrize("threshold", [37.0, 700.0, 1234.5, 43 * MINUTE, 90 * MINUTE])
def test_no_threshold_is_special(threshold):
    """Any threshold is answered from exact counts, wherever it falls.

    Nothing is snapped onto a grid: the counts below are the true number of values
    above the threshold, and the sketch's bounds must contain it while differing by
    at most the one bucket the threshold lands in.
    """
    rng = random.Random(5)
    values = [rng.uniform(1, 10 * MINUTE) for _ in range(20_000)]
    truth = sum(1 for v in values if v > threshold)

    lower, upper = hist.count_above(build(values), threshold, LOG_SKETCH)

    assert lower <= truth <= upper
    # The ambiguity is one bucket wide, not one arbitrary hand-picked gap wide.
    assert (upper - lower) <= max(1, int(len(values) * 0.05))


def test_bounds_are_exact_counts_not_estimates():
    values = [100.0] * 7 + [10_000.0] * 3

    lower, upper = hist.count_above(build(values), 1_000.0, LOG_SKETCH)

    # 1000 falls in no occupied bucket, so both bounds agree exactly.
    assert (lower, upper) == (3, 3)


def test_a_threshold_inside_an_occupied_bucket_is_the_only_ambiguous_case():
    """And when it is ambiguous, the metric is sitting on the threshold.

    Bucket 116 spans (99.55, 103.61]. A threshold of 101 falls inside it, so the
    sketch genuinely cannot say how many of the values at 100 exceed it -- but the
    whole bucket is within 2% of the threshold, so the question is academic.
    """
    values = [100.0] * 10

    lower, upper = hist.count_above(build(values), 101.0, LOG_SKETCH)

    assert (lower, upper) == (0, 10)
    assert 101.0 == pytest.approx(100.0, rel=2 * ALPHA)


def test_a_threshold_on_a_bucket_edge_is_answered_exactly():
    """Not because it was snapped there -- nothing is snapped -- but because the
    grid is fine enough that most thresholds land clear of the occupied buckets.
    """
    values = [100.0] * 10

    # 99.5 falls in bucket 115; the values are all in 116.
    assert hist.count_above(build(values), 99.5, LOG_SKETCH) == (10, 10)


###############################################################################
# Merge, subtract, serialization
###############################################################################


def test_merge_is_a_union_not_an_element_wise_add():
    """Two sketches over disjoint ranges combine correctly.

    The fixed-width form this replaces required equal widths, which is what made
    two differently-bucketed histograms unmergeable.
    """
    fast = build([1.0, 2.0, 3.0])
    slow = build([1e7, 2e7])

    merged = hist.merge(fast, slow)

    assert hist.total(merged) == 5
    assert set(merged) == set(fast) | set(slow)


def test_subtract_inverts_merge():
    """Invertibility is what incremental merge depends on."""
    rng = random.Random(9)
    a = build([rng.uniform(1, 1e6) for _ in range(500)])
    b = build([rng.uniform(1, 1e6) for _ in range(500)])

    assert hist.subtract(hist.merge(a, b), b) == a


def test_subtract_to_empty_leaves_no_dead_buckets():
    """A window that empties must return to {}, not accumulate zero entries."""
    a = build([1.0, 100.0, 10_000.0])

    assert hist.subtract(a, a) == {}


def test_pairs_round_trip_and_are_index_sorted():
    h = build([5.0, 5.0, 1e6, 0.001])

    pairs = hist.to_pairs(h)

    assert hist.from_pairs(pairs) == h
    indices = pairs[0::2]
    assert indices == sorted(indices)


def test_odd_length_pairs_are_rejected():
    with pytest.raises(ValueError, match="even length"):
        hist.from_pairs([1, 2, 3])


###############################################################################
# Zero, negatives, and metrics without a sketch
###############################################################################


def test_zero_gets_its_own_bucket_and_does_not_break_a_quantile():
    h = build([0.0] * 5 + [100.0] * 5)

    assert h[ZERO_INDEX] == 5
    assert hist.quantile(h, 0.1, LOG_SKETCH) == 0.0
    assert hist.quantile(h, 0.9, LOG_SKETCH) == pytest.approx(100.0, rel=ALPHA)


def test_a_negative_value_is_counted_as_zero_rather_than_aborting():
    """Clock skew produces negative durations; one must not fail a whole seal."""
    assert LOG_SKETCH.index(-5.0) == ZERO_INDEX


def test_count_above_zero_excludes_the_zero_bucket():
    h = build([0.0] * 3 + [10.0] * 4)

    assert hist.count_above(h, 0.0, LOG_SKETCH) == (4, 4)


def test_scores_use_a_linear_grid():
    """Relative error is the wrong guarantee for a 0..1 pass rate.

    2% of 0.95 is +/-0.019, which straddles the thresholds people actually set, and
    log spacing puts its coarsest buckets exactly where pass rates cluster.
    """
    assert spec_for("assessment_value") is SCORE_SKETCH
    assert not SCORE_SKETCH.log_scale

    h = build([0.94, 0.95, 0.96], SCORE_SKETCH)

    assert hist.quantile(h, 0.5, SCORE_SKETCH) == pytest.approx(0.95, abs=0.01)


def test_a_metric_with_no_spec_stores_no_sketch():
    """`error_count` counts events; a percentile over it has nothing to bucket."""
    assert spec_for("error_count") is None


def test_percentile_over_buckets_without_a_sketch_raises_rather_than_going_quiet():
    buckets = [Bucket(bucket_start_ms=0, count=10, sum=500.0, histogram=None)]

    with pytest.raises(ValueError, match="no bucket stored a sketch"):
        aggregate_buckets(buckets, "PERCENTILE", 0, 60_000, percentile_value=95, spec=LOG_SKETCH)
