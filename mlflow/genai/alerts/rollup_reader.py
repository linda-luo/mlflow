"""The seam between aggregation (stream B) and evaluation (stream C).

The evaluator's correctness is about merging, comparison and state transitions —
none of which need a real aggregator. :class:`FakeRollupReader` lets stream C be
built and tested against hand-seeded buckets while stream B is still being
written, and swapping in the real reader is the first integration test.
"""

from dataclasses import dataclass
from typing import Protocol

from mlflow.genai.alerts import histogram as hist
from mlflow.genai.alerts.entities import BUCKET_MS, Observation, SeriesKey
from mlflow.genai.alerts.sketch import LOG_SKETCH, SketchSpec


@dataclass(frozen=True)
class Bucket:
    """One sealed rollup bucket. Maps 1:1 onto a ``metric_rollups`` row.

    Or onto an hourly row from the continuous aggregate, which is the same shape
    over a wider span -- see ``width_ms``.
    """

    bucket_start_ms: int
    count: int
    sum: float | None = None
    histogram: list[int] | None = None
    boundaries_version: int | None = None
    is_gap: bool = False
    """Explicit marker that aggregation did not run for this bucket.

    A gap must be recorded rather than skipped: a missing row is indistinguishable
    from a quiet minute, so silently treating it as zero would let an outage read
    as healthy.
    """

    width_ms: int = BUCKET_MS
    """How much time this row covers. One minute unless it came from a coarse tier.

    Carried because a window can be stitched from more than one tier, and the
    accumulator's bucket counter has to stay symmetric under ``add``/``remove``.
    An hourly row added as *one* bucket and later expired as *sixty* minute rows
    drives that counter negative, which reads as "no bucket carried a histogram"
    and silently stops every percentile rule on the series. ``count``, ``sum`` and
    the sketch are all exactly additive and need no such care.
    """


class RollupReader(Protocol):
    def read_buckets(self, series: SeriesKey, start_ms: int, end_ms: int) -> list[Bucket]:
        """Sealed buckets whose start falls in ``[start_ms, end_ms)``."""
        ...

    def latest_sealed_bucket_ms(self, series: SeriesKey | None = None) -> int:
        """Start of the most recent fully sealed bucket.

        Evaluation windows are derived from this, never from the clock, so they
        are always bucket-aligned and never half-cover a bucket that is still
        being written.

        ``series`` scopes the answer to the families that feed it, so a rule is
        not held back by an unrelated signal running behind. ``None`` asks for the
        deployment-wide watermark.
        """
        ...

    def coverage_start_ms(self, series: SeriesKey) -> int:
        """Earliest bucket for which this series has trustworthy history.

        A window starting before this predates aggregation, so an empty result
        over it means "nothing was watching", not "nothing happened". Only a rule
        whose signal is *absence* can tell the difference matters, and for that
        rule the difference is a false alarm on every fresh install.
        """
        ...

    def sketch_for(self, series: SeriesKey) -> "SketchSpec | None":
        """How this series' histograms are bucketed.

        Fixed per metric, so it cannot move because a rule was created. ``None``
        means the metric stores no histogram.
        """
        ...

    def read_window(self, series: SeriesKey, start_ms: int, end_ms: int) -> list[Bucket]:
        """A whole window at once, for the full-read path.

        Separate from :meth:`read_buckets` so a reader may answer it from a coarser
        tier: the ranges are the same, but only this one is known to span an entire
        window rather than the few minutes that entered or left it. Defaults to the
        1-minute read, which is always correct.
        """
        return self.read_buckets(series, start_ms, end_ms)


def project_aggregate(
    aggregation: str,
    *,
    count: int,
    sum_value: float,
    histogram: "hist.Histogram | None",
    has_gap: bool,
    window_start_ms: int,
    window_end_ms: int,
    percentile_value: float | None = None,
    spec: SketchSpec | None = None,
) -> Observation:
    """Turn one window's folded aggregates into the single number a rule compares.

    **The one place this projection is implemented.** Both readers reach a window
    the same way -- :func:`aggregate_buckets` folds a list of buckets, the
    evaluator's ``WindowAccumulator`` folds incrementally -- but what the folded
    totals *mean* must not depend on which of them got there. Two copies of this
    logic previously disagreed on the empty window, and the copy the evaluator
    actually used was the wrong one: an absence-signal rule ("requests < 10",
    "traffic dropped") read as insufficient-data at exactly the moment it should
    fire, so it could never fire at all.

    Args:
        aggregation: COUNT, SUM, AVG or PERCENTILE. Selects which fold is read,
            which is why one series answers several questions and there is no
            separate ``request_count`` metric -- ``latency`` with COUNT already is
            one.
        count: total observations in the window.
        sum_value: total of the observed values; ignored unless SUM or AVG.
        histogram: merged histogram, or ``None`` when no bucket stored one.
        has_gap: whether any bucket in the window is an explicit gap marker.
        percentile_value: required for PERCENTILE, ignored otherwise.
    """
    if has_gap:
        # A gap outranks everything: aggregation did not run, so no fold over this
        # window means anything.
        return Observation(None, 0, window_start_ms, window_end_ms)

    if count == 0:
        # Counting nothing is genuinely zero; the average or p95 of nothing is not
        # a number. This split is what makes a rule whose signal is *absence*
        # expressible at all.
        #
        # A recorded gap never reaches here -- it returned above -- which is what
        # makes "nobody called us" distinguishable from "we stopped looking", and
        # is the whole reason gap markers are stored separately from a zero count.
        if aggregation == "COUNT":
            return Observation(0.0, 0, window_start_ms, window_end_ms)
        return Observation(None, 0, window_start_ms, window_end_ms)

    if aggregation == "COUNT":
        value = float(count)
    elif aggregation == "SUM":
        value = sum_value
    elif aggregation == "AVG":
        value = sum_value / count
    elif aggregation == "PERCENTILE":
        if percentile_value is None:
            raise ValueError("PERCENTILE aggregation requires percentile_value")
        if not histogram or spec is None:
            # Observations exist but no bucket stored a sketch -- the metric has no
            # spec. Returning "no data" would make the rule look merely quiet
            # forever, which is precisely the silent failure alerting exists to
            # prevent, so refuse loudly instead.
            raise ValueError(
                f"PERCENTILE requested over {count} observations but no bucket "
                "stored a sketch. This metric has no sketch spec defined, so "
                "percentiles cannot be computed for it."
            )
        value = hist.quantile(histogram, percentile_value / 100.0, spec)
        if value is None:
            return Observation(None, count, window_start_ms, window_end_ms)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    return Observation(value, count, window_start_ms, window_end_ms)


def aggregate_buckets(
    buckets: list[Bucket],
    aggregation: str,
    window_start_ms: int,
    window_end_ms: int,
    percentile_value: float | None = None,
    spec: SketchSpec | None = None,
) -> Observation:
    """Fold a list of sealed buckets, then project. See :func:`project_aggregate`.

    Buckets sealed under a different sketch version are skipped. In normal operation
    there are none -- the grid is a constant and does not move when rules change,
    which is the property this design exists for -- so this guards only a future
    change to GAMMA, where old indices would mean different value ranges.
    """
    merged: hist.Histogram | None = None
    for b in buckets:
        if b.histogram is None or b.boundaries_version != hist.SKETCH_VERSION:
            continue
        pairs = hist.from_pairs(b.histogram)
        merged = pairs if merged is None else hist.merge(merged, pairs)

    return project_aggregate(
        aggregation,
        count=sum(b.count for b in buckets),
        sum_value=sum(b.sum or 0.0 for b in buckets),
        histogram=merged,
        has_gap=any(b.is_gap for b in buckets),
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        percentile_value=percentile_value,
        spec=spec,
    )


class FakeRollupReader(RollupReader):
    """In-memory reader for tests. No database, no Timescale, no aggregator."""

    def __init__(self, latest_sealed_bucket_ms: int = 0):
        self._buckets: dict[SeriesKey, dict[int, Bucket]] = {}
        self._latest_sealed_bucket_ms = latest_sealed_bucket_ms
        self._series_watermarks: dict[SeriesKey, int] = {}
        self._coverage_starts: dict[SeriesKey, int] = {}
        self._sketches: dict[SeriesKey, SketchSpec | None] = {}
        self._coarse_usable_until_ms: int | None = None
        self.window_reads: list[tuple[SeriesKey, int, int]] = []

    def enable_coarse_tier(self, usable_until_ms: int) -> None:
        """Serve full-window reads from a synthesized hourly tier.

        The hourly rows are folded from this reader's own minute buckets exactly as
        the continuous aggregate folds them, so a stitched read is only correct here
        if it would be correct against Timescale.
        """
        self._coarse_usable_until_ms = usable_until_ms

    def seed(self, series: SeriesKey, bucket: Bucket) -> None:
        self._buckets.setdefault(series, {})[bucket.bucket_start_ms] = bucket
        self._latest_sealed_bucket_ms = max(self._latest_sealed_bucket_ms, bucket.bucket_start_ms)

    def seed_constant(
        self,
        series: SeriesKey,
        start_ms: int,
        end_ms: int,
        count: int,
        value: float,
    ) -> None:
        """Fill every bucket in a range with the same observation."""
        spec = self.sketch_for(series) or LOG_SKETCH
        for bucket_start in range(start_ms, end_ms, BUCKET_MS):
            h = hist.empty()
            for _ in range(count):
                hist.observe(h, value, spec)
            self.seed(
                series,
                Bucket(
                    bucket_start_ms=bucket_start,
                    count=count,
                    sum=value * count,
                    histogram=hist.to_pairs(h),
                    boundaries_version=hist.SKETCH_VERSION,
                ),
            )

    def read_buckets(self, series: SeriesKey, start_ms: int, end_ms: int) -> list[Bucket]:
        by_start = self._buckets.get(series, {})
        return [by_start[b] for b in sorted(by_start) if start_ms <= b < end_ms]

    def read_window(self, series: SeriesKey, start_ms: int, end_ms: int) -> list[Bucket]:
        self.window_reads.append((series, start_ms, end_ms))
        if self._coarse_usable_until_ms is None:
            return self.read_buckets(series, start_ms, end_ms)

        from mlflow.genai.alerts.tiers import HOUR_MS, plan_cover

        buckets: list[Bucket] = []
        for span in plan_cover(start_ms, end_ms, self._coarse_usable_until_ms):
            if not span.is_coarse:
                buckets.extend(self.read_buckets(series, span.start_ms, span.end_ms))
                continue
            for hour_start in range(span.start_ms, span.end_ms, HOUR_MS):
                minutes = self.read_buckets(series, hour_start, hour_start + HOUR_MS)
                if any(b.is_gap for b in minutes):
                    # Same fallback as the real reader: `bool_or` would mark the
                    # whole hour, and a gap suppresses the entire window.
                    buckets.extend(minutes)
                    continue
                merged = hist.empty()
                for b in minutes:
                    if b.histogram is not None:
                        merged = hist.merge(merged, hist.from_pairs(b.histogram))
                buckets.append(
                    Bucket(
                        bucket_start_ms=hour_start,
                        count=sum(b.count for b in minutes),
                        sum=sum(b.sum or 0.0 for b in minutes),
                        histogram=hist.to_pairs(merged) if merged else None,
                        boundaries_version=hist.SKETCH_VERSION if merged else None,
                        width_ms=HOUR_MS,
                    )
                )
        return buckets

    def latest_sealed_bucket_ms(self, series: SeriesKey | None = None) -> int:
        # Per-series watermarks can be seeded to exercise the isolation the real
        # reader provides; absent one, every series shares the global value.
        if series is not None and series in self._series_watermarks:
            return self._series_watermarks[series]
        return self._latest_sealed_bucket_ms

    def set_series_watermark(self, series: SeriesKey, watermark_ms: int) -> None:
        """Make one series lag the rest, as a slow source would."""
        self._series_watermarks[series] = watermark_ms

    def coverage_start_ms(self, series: SeriesKey) -> int:
        # Hand-seeded buckets are the whole history by construction, so everything
        # is covered unless a test says otherwise.
        return self._coverage_starts.get(series, 0)

    def set_coverage_start(self, series: SeriesKey, coverage_start_ms: int) -> None:
        """Pretend aggregation for this series only began at ``coverage_start_ms``."""
        self._coverage_starts[series] = coverage_start_ms

    def sketch_for(self, series: SeriesKey) -> SketchSpec | None:
        return self._sketches.get(series, LOG_SKETCH)

    def set_sketch(self, series: SeriesKey, spec: SketchSpec | None) -> None:
        self._sketches[series] = spec
