"""Sparse sketch histograms — the shared contract between aggregation and evaluation.

A histogram is a map from bucket index to count, holding only *occupied* buckets.
The bucketing itself lives in :mod:`mlflow.alerts.sketch`; this module is the
arithmetic over it.

Sparse rather than a fixed-width vector because the sketch's index range is
unbounded — a dense array would have to reserve slots for every value anyone might
ever record, which is precisely the clamp that made the previous implementation
blind above 4 hours. A minute of one series' traffic occupies tens of buckets, so
sparse stores what was actually seen rather than what could conceivably be seen.

Bucket counts are *exact*. That is what lets a threshold be answered with certainty
even though the underlying values were discarded: the sketch cannot say what the
true p95 is, but it can say exactly how many observations fell above a given bucket.
Only when the threshold lands *inside* an occupied bucket does the answer need the
sketch's +/-alpha estimate.

On the wire a histogram is a flat ``BIGINT[]`` of interleaved ``(index, count)``
pairs sorted by index — see :func:`to_pairs` — so the Postgres merge aggregate and
the Python one operate on the same shape.
"""

from mlflow.alerts.entities import Comparator
from mlflow.alerts.sketch import SKETCH_VERSION, SketchSpec, spec_for

__all__ = [
    "SKETCH_VERSION",
    "SketchSpec",
    "bounds_decide",
    "compare",
    "count_above",
    "empty",
    "from_pairs",
    "merge",
    "observe",
    "quantile",
    "spec_for",
    "subtract",
    "to_pairs",
    "total",
]

Histogram = dict[int, int]
"""Bucket index -> count. Absent index means zero, which is why it is sparse."""


def empty() -> Histogram:
    return {}


def observe(hist: Histogram, value: float, spec: SketchSpec) -> Histogram:
    index = spec.index(value)
    hist[index] = hist.get(index, 0) + 1
    return hist


def merge(a: Histogram, b: Histogram) -> Histogram:
    """Union, summing counts. This is what makes tiered rollups legal.

    No width to agree on, unlike the fixed-vector form this replaces: two sketches
    covering completely different value ranges merge correctly because a bucket is
    identified by its index, not by its position in an array.
    """
    merged = dict(a)
    for index, count in b.items():
        merged[index] = merged.get(index, 0) + count
    return merged


def subtract(a: Histogram, b: Histogram) -> Histogram:
    """Inverse of :func:`merge` — what makes incremental merge possible.

    Every stored aggregate must support this, which is why min/max are not stored:
    they cannot be un-merged when a bucket leaves the window. Buckets that reach
    zero are dropped, so a window that empties returns to ``{}`` rather than
    accumulating dead indices forever.
    """
    result = dict(a)
    for index, count in b.items():
        remaining = result.get(index, 0) - count
        if remaining:
            result[index] = remaining
        else:
            result.pop(index, None)
    return result


def total(hist: Histogram) -> int:
    return sum(hist.values())


def to_pairs(hist: Histogram) -> list[int]:
    """Flatten to the stored form: ``[index, count, index, count, ...]``, index-sorted.

    Sorted so the Postgres merge can rely on ordering and so two equal sketches
    serialize identically, which keeps re-sealing a bucket genuinely idempotent.
    """
    pairs: list[int] = []
    for index in sorted(hist):
        count = hist[index]
        if count:
            pairs.extend((index, count))
    return pairs


def from_pairs(pairs: list[int] | None) -> Histogram:
    if not pairs:
        return {}
    if len(pairs) % 2:
        raise ValueError(f"Sketch pairs must have even length, got {len(pairs)}")
    return {pairs[i]: pairs[i + 1] for i in range(0, len(pairs), 2) if pairs[i + 1]}


def quantile(hist: Histogram, q: float, spec: SketchSpec) -> float | None:
    """The ``q``th quantile (0 < q < 1), within ``ALPHA`` relative error.

    Returns the containing bucket's *representative* value rather than its upper
    edge. The upper edge is systematically high — it is what made a true p95 of
    ~90 minutes report as 120 — whereas the representative value is within alpha in
    both directions.
    """
    n = total(hist)
    if n == 0:
        return None
    target = q * n
    seen = 0
    for index in sorted(hist):
        seen += hist[index]
        if seen >= target:
            return spec.estimate(index)
    return spec.estimate(max(hist))


def count_above(hist: Histogram, threshold: float, spec: SketchSpec) -> tuple[int, int]:
    """Exact ``[lower, upper]`` bounds on how many observations exceed ``threshold``.

    Both numbers are exact counts, not estimates: ``lower`` is everything in buckets
    entirely above the threshold, and ``upper`` adds the one bucket the threshold
    falls inside. They differ only by that bucket's population, and the bucket spans
    barely 4% of its own value.

    No threshold is special here. The previous implementation collapsed these bounds
    only when the threshold happened to equal a boundary, which is why thresholds
    were snapped onto boundaries — moving the user's number to make the arithmetic
    convenient. Nothing is moved now; the bounds are simply tight enough.
    """
    index = spec.index(threshold)
    lower = sum(count for i, count in hist.items() if i > index)
    if threshold == spec.upper_edge(index):
        # Exactly on a bucket edge: nothing in that bucket exceeds it.
        return lower, lower
    return lower, lower + hist.get(index, 0)


def bounds_decide(
    hist: Histogram,
    threshold: float,
    count_comparator: Comparator,
    count_threshold: float,
    spec: SketchSpec,
) -> bool | None:
    """Decide "N observations beyond ``threshold``" from exact bounds alone.

    ``True``/``False`` when the bounds agree — the overwhelmingly common case, since
    they differ only by one narrow bucket. ``None`` when they disagree, which means
    the metric is sitting within alpha of the threshold and the caller must either
    consult raw rows or accept the sketch's estimate.
    """
    lower, upper = count_above(hist, threshold, spec)
    low_verdict = compare(lower, count_comparator, count_threshold)
    high_verdict = compare(upper, count_comparator, count_threshold)
    return low_verdict if low_verdict == high_verdict else None


def compare(value: float, comparator: Comparator, threshold: float) -> bool:
    if comparator == "GT":
        return value > threshold
    if comparator == "GTE":
        return value >= threshold
    if comparator == "LT":
        return value < threshold
    if comparator == "LTE":
        return value <= threshold
    raise ValueError(f"Unknown comparator: {comparator}")
