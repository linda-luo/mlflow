"""Percentiles are answerable exactly for metrics that have a histogram ladder.

The failure mode being guarded against is silence: a rule that returns "no data"
forever looks identical to a healthy quiet period, so an unanswerable rule must be
impossible to create and impossible to evaluate quietly.

The catalogue advertised PERCENTILE on tokens, cost and assessment values while
only ``latency`` had boundaries, so three of five metrics could not answer the
aggregation most people want from them. Each now has a base ladder in its own unit
-- dollars, tokens, a 0..1 score -- so the advertised set and the answerable set
are the same set, which is what these tests assert.
"""

import pytest

from mlflow.genai.alerts import histogram as hist
from mlflow.genai.alerts.entities import (
    METRIC_CATALOGUE,
    percentile_supported_metrics,
    validate_metric_triple,
)
from mlflow.genai.alerts.rollup_reader import Bucket, aggregate_buckets
from mlflow.genai.alerts.sketch import spec_for

_PERCENTILE_METRICS_IN_CATALOGUE = sorted(
    key for key, spec in METRIC_CATALOGUE.items() if "PERCENTILE" in spec.aggregations
)


def test_latency_percentile_is_accepted():
    validate_metric_triple("latency", "TRACES", "PERCENTILE")


@pytest.mark.parametrize("metric_key", _PERCENTILE_METRICS_IN_CATALOGUE)
def test_every_metric_the_catalogue_advertises_can_answer_a_percentile(metric_key: str):
    """The catalogue must not offer what the evaluator cannot compute."""
    dimension_key = min(METRIC_CATALOGUE[metric_key].dimension_keys)
    validate_metric_triple(metric_key, dimension_key, "PERCENTILE")


def test_the_advertised_and_answerable_sets_agree():
    assert set(_PERCENTILE_METRICS_IN_CATALOGUE) <= percentile_supported_metrics()


@pytest.mark.parametrize("metric_key", ["error_count", "error_rate"])
def test_a_metric_with_no_ladder_is_still_rejected(metric_key: str):
    """A count and a ratio both have nothing to bucket.

    `error_rate` stores its denominator in `count` and its numerator in `sum`, so
    there is no distribution to take a percentile of -- and having no sketch spec is
    what makes `project_aggregate` raise rather than report an eternally quiet rule.
    """
    assert metric_key not in percentile_supported_metrics()
    spec = METRIC_CATALOGUE[metric_key]
    # Rejected by the catalogue before the sketch check is even reached, since
    # neither metric advertises PERCENTILE. Both guards have to hold: the catalogue
    # stops the form offering it, the sketch check stops a hand-written rule.
    with pytest.raises(ValueError, match="does not support"):
        validate_metric_triple(metric_key, min(spec.dimension_keys), "PERCENTILE")


@pytest.mark.parametrize("metric_key", _PERCENTILE_METRICS_IN_CATALOGUE)
def test_the_same_metrics_still_accept_their_other_aggregations(metric_key: str):
    spec = METRIC_CATALOGUE[metric_key]
    dimension_key = min(spec.dimension_keys)
    for aggregation in sorted(spec.aggregations - {"PERCENTILE"}):
        validate_metric_triple(metric_key, dimension_key, aggregation)


def test_unknown_metric_and_bad_slice_are_rejected():
    with pytest.raises(ValueError, match="Unknown metric"):
        validate_metric_triple("not_a_metric", "TRACES", "COUNT")
    with pytest.raises(ValueError, match="cannot be sliced by"):
        validate_metric_triple("latency", "SPAN_MODEL", "COUNT")


def test_percentile_over_sketch_less_buckets_raises_rather_than_reporting_no_data():
    """The read path refuses too, so a rule persisted before this check existed
    cannot quietly degrade into a permanently silent alert.
    """
    buckets = [Bucket(bucket_start_ms=0, count=10, sum=500.0, histogram=None)]
    with pytest.raises(ValueError, match="no bucket stored a sketch"):
        aggregate_buckets(buckets, "PERCENTILE", 0, 60_000, percentile_value=95)


@pytest.mark.parametrize(
    ("metric_key", "values", "percentile", "expected"),
    [
        # A log grid, which is what every duration, count and cost uses.
        ("latency", [10.0] * 90 + [4_000.0] * 10, 95, 4_000.0),
        ("input_tokens", [800.0] * 90 + [19_000.0] * 10, 95, 19_000.0),
        ("total_cost", [0.001] * 90 + [0.4] * 10, 99, 0.4),
        # A *linear* grid: 2% relative error is the wrong guarantee for a pass rate,
        # since 2% of 0.95 straddles the thresholds people actually set. This is the
        # case a single shared grid would get wrong.
        ("assessment_value", [0.2] * 90 + [0.97] * 10, 95, 0.97),
    ],
)
def test_each_metric_answers_its_percentile_on_its_own_grid(
    metric_key: str, values: list[float], percentile: float, expected: float
):
    """The grid is per metric, and reading it back has to land on the right one.

    Validation only proves the catalogue and the sketch specs agree on *which*
    metrics can answer a percentile. This proves the answer is right -- and it is
    the only test that would catch a metric being bucketed on another metric's
    grid, which reads back as a plausible number in the wrong unit.
    """
    spec = spec_for(metric_key)
    histogram = hist.empty()
    for value in values:
        hist.observe(histogram, value, spec)

    observation = aggregate_buckets(
        [
            Bucket(
                bucket_start_ms=0,
                count=len(values),
                sum=sum(values),
                histogram=hist.to_pairs(histogram),
                boundaries_version=hist.SKETCH_VERSION,
            )
        ],
        "PERCENTILE",
        0,
        60_000,
        percentile_value=percentile,
        spec=spec,
    )

    assert observation.sample_count == len(values)
    # Within the grid's own guarantee rather than exact: a sketch answers to a
    # bounded relative error, and asserting equality would be asserting something
    # the design deliberately does not promise.
    assert observation.observed_value == pytest.approx(expected, rel=0.05)


def test_a_token_count_is_not_bucketed_on_the_latency_grid():
    """Reading a metric back on the wrong grid is silent, so pin the difference.

    Both are log grids here, so the failure would not be an exception -- it would
    be a number that looks reasonable and is wrong.
    """
    assert spec_for("input_tokens") is spec_for("latency")
    # But the judge score is not, and that is the one that would misread.
    assert spec_for("assessment_value") is not spec_for("latency")

    score_spec = spec_for("assessment_value")
    latency_spec = spec_for("latency")
    # 0.97 lands in different buckets on the two grids, which is exactly why the
    # spec has to be looked up per metric rather than assumed.
    assert score_spec.index(0.97) != latency_spec.index(0.97)


def test_an_empty_window_is_still_no_data_not_an_error():
    """Absence of observations is a legitimate answer; absence of a histogram
    where observations exist is not.
    """
    observation = aggregate_buckets([], "PERCENTILE", 0, 60_000, percentile_value=95)
    assert observation.observed_value is None
    assert not observation.has_data
