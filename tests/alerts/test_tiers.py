"""Covering a window from the coarsest tier that fits.

The property under test throughout is that stitching changes *how many rows are
read*, never *what the window folds to*. Every test that reads asserts against the
unstitched read rather than against a hand-written expectation.
"""

import pytest

from mlflow.alerts import histogram as hist
from mlflow.alerts.entities import BUCKET_MS, MAX_WINDOW_SECONDS, SeriesKey
from mlflow.alerts.evaluator import (
    IncrementalMergeCache,
    WindowAccumulator,
    evaluation_window,
    read_signature,
)
from mlflow.alerts.rollup_reader import Bucket, FakeRollupReader
from mlflow.alerts.sketch import LOG_SKETCH
from mlflow.alerts.tiers import HOUR_MS, plan_cover, rows_read
from mlflow.alerts.timescale import RAW_RETENTION_MS

SERIES = SeriesKey(dimension_key="TRACES", experiment_id=7, metric_key="latency")

# A round hour, so offsets in the tests mean what they say.
H0 = (1_700_000_000_000 // HOUR_MS) * HOUR_MS
FOREVER = H0 + 10_000 * HOUR_MS


def _spans(start_ms, end_ms, usable_until_ms=FOREVER):
    return [
        (s.start_ms, s.end_ms, s.width_ms) for s in plan_cover(start_ms, end_ms, usable_until_ms)
    ]


###############################################################################
# Planning
###############################################################################


def test_a_window_ending_mid_hour_is_not_snapped():
    """The reason the tiers went unread: a window ends at the watermark.

    A 2.5h window running 12:07 to 14:37 contains exactly one whole hour, 13:00 to
    14:00. Both edges stay minute-grained; only what fits entirely inside goes
    coarse.
    """
    start = H0 + 12 * HOUR_MS + 7 * BUCKET_MS
    end = start + 150 * BUCKET_MS

    assert _spans(start, end) == [
        (start, H0 + 13 * HOUR_MS, BUCKET_MS),
        (H0 + 13 * HOUR_MS, H0 + 14 * HOUR_MS, HOUR_MS),
        (H0 + 14 * HOUR_MS, end, BUCKET_MS),
    ]


def test_an_hour_aligned_window_is_entirely_coarse():
    assert _spans(H0, H0 + 3 * HOUR_MS) == [(H0, H0 + 3 * HOUR_MS, HOUR_MS)]


def test_a_window_too_small_to_contain_whole_hours_stays_fine():
    # Nothing to gain, and the second query plus its freshness check would cost
    # more than the rows saved.
    assert _spans(H0 + 30 * BUCKET_MS, H0 + 90 * BUCKET_MS) == [
        (H0 + 30 * BUCKET_MS, H0 + 90 * BUCKET_MS, BUCKET_MS)
    ]


def test_an_empty_or_inverted_window_reads_nothing():
    assert plan_cover(H0, H0, FOREVER) == []
    assert plan_cover(H0 + HOUR_MS, H0, FOREVER) == []


def test_the_coarse_tier_is_not_trusted_past_its_watermark():
    """An hour materialized while the aggregator was behind is permanently short.

    Timescale never revisits a materialized bucket, so the under-count is silent.
    Everything past the trustworthy bound falls back to minutes.
    """
    start, end = H0, H0 + 6 * HOUR_MS

    assert _spans(start, end, usable_until_ms=H0 + 3 * HOUR_MS) == [
        (H0, H0 + 3 * HOUR_MS, HOUR_MS),
        (H0 + 3 * HOUR_MS, end, BUCKET_MS),
    ]


def test_a_stale_coarse_tier_degrades_to_a_plain_minute_read():
    assert _spans(H0, H0 + 6 * HOUR_MS, usable_until_ms=H0) == [(H0, H0 + 6 * HOUR_MS, BUCKET_MS)]


@pytest.mark.parametrize(
    ("hours", "offset_buckets", "expected_rows"),
    [
        (24, 37, 83),  # the headline case: 23 hourly + 23 + 37 minutes
        (24, 0, 24),  # hour-aligned, all coarse
        (2.5, 7, 91),  # 1 hourly + 53 + 37 minutes, against 150
        (2.5, 0, 32),  # starts on the hour, still ends mid-hour: 2 hourly + 30
    ],
)
def test_rows_read_collapses_for_long_windows(hours, offset_buckets, expected_rows):
    start = H0 + offset_buckets * BUCKET_MS
    end = start + int(hours * HOUR_MS)

    assert rows_read(plan_cover(start, end, FOREVER)) == expected_rows


def test_the_longest_window_stays_on_the_hourly_tier():
    """Three days costs 131 reads against 4,320 raw, with no daily tier involved.

    Pins the two-tier design. A day+hour+minute cover would take this to 85 -- 46
    rows saved once an hour per read group, against a freshness gate, a third bucket
    width, and a gap fallback where one bad minute invalidates a whole day.
    """
    start = H0 + 37 * BUCKET_MS
    end = start + 72 * HOUR_MS

    spans = plan_cover(start, end, FOREVER)

    assert [s.width_ms for s in spans] == [BUCKET_MS, HOUR_MS, BUCKET_MS]
    # 23 leading minutes + 71 whole hours + 37 trailing minutes.
    assert rows_read(spans) == 131
    assert (end - start) // BUCKET_MS == 4_320


def test_the_ragged_edges_are_bounded_however_long_the_window():
    """Cost is O(window/hour + 120): at most two partial hours of minutes."""
    for hours in (3, 6, 12, 24):
        start = H0 + 37 * BUCKET_MS
        spans = plan_cover(start, start + hours * HOUR_MS, FOREVER)
        fine_rows = sum((s.end_ms - s.start_ms) // BUCKET_MS for s in spans if not s.is_coarse)
        assert fine_rows <= 120


###############################################################################
# Reading
###############################################################################


def _seed(reader, start_ms, end_ms, count=3, value=90_000.0):
    reader.seed_constant(SERIES, start_ms, end_ms, count, value)


def _fold(buckets):
    return WindowAccumulator().add(buckets)


@pytest.mark.parametrize("offset_buckets", [0, 7, 37, 59])
def test_a_stitched_window_folds_to_exactly_the_unstitched_window(offset_buckets):
    """The gate. Identical, not merely close -- an hourly row is the sum of its
    minutes, so any divergence is a bug rather than a rounding difference.
    """
    start = H0 + offset_buckets * BUCKET_MS
    end = start + 6 * HOUR_MS
    reader = FakeRollupReader(latest_sealed_bucket_ms=end)
    _seed(reader, H0, end + HOUR_MS)

    plain = _fold(reader.read_buckets(SERIES, start, end))
    reader.enable_coarse_tier(FOREVER)
    stitched = _fold(reader.read_window(SERIES, start, end))

    assert stitched.count == plain.count
    assert stitched.sum_value == plain.sum_value
    assert stitched.histogram == plain.histogram
    # Minute-equivalents, so the two tiers agree on what the counter means.
    assert stitched.histogram_bucket_count == plain.histogram_bucket_count


def test_an_hour_containing_a_gap_falls_back_to_its_minutes():
    """`bool_or(is_gap)` would mark the whole hour, and a gap suppresses the
    entire window -- so a stitched read would go silent where the minute read
    reports a real number.
    """
    start, end = H0, H0 + 4 * HOUR_MS
    reader = FakeRollupReader(latest_sealed_bucket_ms=end)
    _seed(reader, start, end)
    gap_at = H0 + 2 * HOUR_MS + 5 * BUCKET_MS
    reader.seed(SERIES, Bucket(bucket_start_ms=gap_at, count=0, is_gap=True))

    plain = _fold(reader.read_buckets(SERIES, start, end))
    reader.enable_coarse_tier(FOREVER)
    stitched = _fold(reader.read_window(SERIES, start, end))

    assert stitched.count == plain.count
    # Keyed at minute granularity, so it can still leave the window when it
    # expires; an hour-aligned key never would.
    assert stitched.gap_buckets == {gap_at}


def test_a_stitched_read_still_expires_correctly_on_the_next_cycle():
    """The reason `width_ms` exists.

    The full read seeds the cache, and the next cycle subtracts *minutes* from it.
    Counting hourly rows as one bucket each made the two directions disagree by a
    factor of sixty, so the counter went negative and every percentile rule on the
    series silently reported no data.
    """
    cache = IncrementalMergeCache()
    watermark = H0 + 24 * HOUR_MS + 37 * BUCKET_MS
    reader = FakeRollupReader(latest_sealed_bucket_ms=watermark)
    _seed(reader, H0, watermark + 10 * BUCKET_MS)
    reader.enable_coarse_tier(FOREVER)
    signature = read_signature(_rule())

    first = cache.accumulator_for(
        signature, evaluation_window(watermark, 86_400), reader, now_ms=watermark
    )
    assert first.histogram_bucket_count == 1440

    # Five minutes later: five buckets enter, five expire.
    later = watermark + 5 * BUCKET_MS
    second = cache.accumulator_for(
        signature, evaluation_window(later, 86_400), reader, now_ms=later
    )

    assert second.histogram_bucket_count == 1440
    assert second.count == first.count
    # And it still answers a percentile, which is what the counter gates.
    assert (
        second.to_observation(
            "PERCENTILE", evaluation_window(later, 86_400), 95.0, LOG_SKETCH
        ).observed_value
        is not None
    )


def test_the_full_read_uses_the_window_path_and_the_incremental_read_does_not():
    cache = IncrementalMergeCache()
    watermark = H0 + 24 * HOUR_MS
    reader = FakeRollupReader(latest_sealed_bucket_ms=watermark)
    _seed(reader, H0, watermark + 10 * BUCKET_MS)
    signature = read_signature(_rule())

    cache.accumulator_for(signature, evaluation_window(watermark, 86_400), reader, now_ms=watermark)
    assert len(reader.window_reads) == 1

    later = watermark + 5 * BUCKET_MS
    cache.accumulator_for(signature, evaluation_window(later, 86_400), reader, now_ms=later)
    # Still one: the incremental path reads the few minutes either side directly,
    # where a coarse tier has nothing to offer.
    assert len(reader.window_reads) == 1


def test_the_incremental_path_still_sheds_at_the_longest_window():
    """The regression a seven-day ceiling introduced, pinned at three days.

    The expiring edge is read at one-minute granularity. Past raw retention that read
    returns nothing, ``remove`` subtracts nothing, and the accumulator keeps every
    bucket it ever added -- drifting upward until the hourly recompute hides it.

    Here the reader refuses ranges older than retention, exactly as a Timescale
    deployment would. At ``MAX_WINDOW_SECONDS`` the expiring edge is still inside
    retention, so the count must hold steady; at seven days it would climb every
    cycle.
    """
    window_seconds = MAX_WINDOW_SECONDS
    watermark = H0 + 200 * HOUR_MS

    class RetainedOnlyReader(FakeRollupReader):
        """Returns nothing for ranges the raw tier would no longer hold."""

        def read_buckets(self, series, start_ms, end_ms):
            if start_ms < watermark - RAW_RETENTION_MS:
                return []
            return super().read_buckets(series, start_ms, end_ms)

    reader = RetainedOnlyReader(latest_sealed_bucket_ms=watermark)
    # Seeded past the last cycle's leading edge, so a falling count can only mean the
    # expiring side outran the entering one -- not that the data simply ran out.
    _seed(reader, H0, watermark + 40 * BUCKET_MS, count=2)
    cache = IncrementalMergeCache()
    signature = read_signature(_rule(window_seconds=window_seconds))

    counts = []
    for step in range(6):
        now = watermark + step * 5 * BUCKET_MS
        accumulator = cache.accumulator_for(
            signature, evaluation_window(now, window_seconds), reader, now_ms=now
        )
        counts.append(accumulator.count)

    # Every cycle adds five minutes and expires five, so the total holds steady. A
    # window past retention would show a strictly increasing series here.
    assert len(set(counts)) == 1, counts


def test_a_reader_without_a_coarse_tier_answers_the_window_from_minutes():
    """The default. `setup_timescale` runs only from the dev bootstrap, so a real
    Postgres deployment may have no continuous aggregates at all.
    """
    start, end = H0, H0 + 4 * HOUR_MS
    reader = FakeRollupReader(latest_sealed_bucket_ms=end)
    _seed(reader, start, end)

    assert reader.read_window(SERIES, start, end) == reader.read_buckets(SERIES, start, end)


def _rule(window_seconds: int = 86_400):
    from mlflow.alerts.entities import AlertRule

    return AlertRule(
        alert_rule_id="rule-1",
        experiment_id=SERIES.experiment_id,
        name="Slow",
        metric_key="latency",
        dimension_key="TRACES",
        aggregation="PERCENTILE",
        percentile_value=95.0,
        comparator="GT",
        threshold=45 * 60_000,
        window_seconds=window_seconds,
        evaluation_interval_seconds=300,
    )


def test_hourly_rows_carry_their_width_and_minute_rows_keep_the_default():
    reader = FakeRollupReader(latest_sealed_bucket_ms=H0 + 4 * HOUR_MS)
    _seed(reader, H0, H0 + 4 * HOUR_MS)
    reader.enable_coarse_tier(FOREVER)

    buckets = reader.read_window(SERIES, H0 + 30 * BUCKET_MS, H0 + 4 * HOUR_MS)

    assert {b.width_ms for b in buckets} == {BUCKET_MS, HOUR_MS}
    assert all(b.width_ms == HOUR_MS for b in buckets if b.bucket_start_ms % HOUR_MS == 0)


def test_an_hourly_sketch_is_the_merge_of_its_minutes():
    """The tiering invariant, asserted directly on the sketch rather than assumed."""
    reader = FakeRollupReader(latest_sealed_bucket_ms=H0 + 2 * HOUR_MS)
    _seed(reader, H0, H0 + 2 * HOUR_MS, count=4, value=1234.0)
    reader.enable_coarse_tier(FOREVER)

    (first_hour, _) = reader.read_window(SERIES, H0, H0 + 2 * HOUR_MS)
    expected = hist.empty()
    for b in reader.read_buckets(SERIES, H0, H0 + HOUR_MS):
        expected = hist.merge(expected, hist.from_pairs(b.histogram))

    assert hist.from_pairs(first_hour.histogram) == expected
    assert first_hour.count == 60 * 4
