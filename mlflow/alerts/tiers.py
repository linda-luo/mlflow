"""Covering a window with the coarsest rollup tier that fits.

The 1h and 1d continuous aggregates were written every minute and read by nothing:
alert windows end at the watermark, so a 24h window runs 14:37 to 14:37 and is
never an hourly bucket. Reading only whole hours that fall *inside* such a window
still replaces most of it -- 23 hourly rows and two ragged minute-grained edges
cover 24 hours in about 83 reads instead of 1,440.

Only the full-read path uses this. The incremental path adds and removes a few
minutes per cycle and has nothing to gain, while mixing tiers across an
``add``/``remove`` pair is what :attr:`Bucket.width_ms` exists to survive.

Buckets are aligned to absolute multiples of their width from the epoch --
``time_bucket`` is called with no origin, and 3,600,000 divides the epoch evenly --
so an hour boundary is ``ms // _HOUR_MS`` with no round-trip to the database.
"""

from dataclasses import dataclass

from mlflow.alerts.entities import BUCKET_MS

HOUR_MS = 3_600_000

MIN_COARSE_HOURS = 1
"""Whole hours a window must contain before stitching is worth it.

One is enough: a single hourly row replaces sixty, against the cost of one extra
query and a freshness check. A 2.5h window ending mid-hour contains exactly one
whole hour and drops from 150 reads to 91, so requiring two would skip the
smallest window that still gains materially.
"""


@dataclass(frozen=True)
class CoverSpan:
    """A half-open range to read at one granularity."""

    start_ms: int
    end_ms: int
    width_ms: int

    @property
    def is_coarse(self) -> bool:
        return self.width_ms != BUCKET_MS


def _ceil_hour(ms: int) -> int:
    return -(-ms // HOUR_MS) * HOUR_MS


def _floor_hour(ms: int) -> int:
    return (ms // HOUR_MS) * HOUR_MS


def plan_cover(start_ms: int, end_ms: int, coarse_usable_until_ms: int) -> list[CoverSpan]:
    """Greedy cover of ``[start_ms, end_ms)``, coarsest-first.

    The window is never snapped. An hourly row is used only where it lies wholly
    inside the window *and* wholly inside the range the coarse tier can be trusted
    for; everything else is read at one minute, so the answer is identical to the
    unstitched read rather than merely close to it.

    Args:
        coarse_usable_until_ms: exclusive bound past which the hourly tier must not
            be trusted. Its rows are materialized on a schedule and never revisited,
            so an hour built while the 1-minute aggregator was behind is permanently
            short -- reading it would silently under-count.
    """
    if end_ms <= start_ms:
        return []

    hour_start = _ceil_hour(start_ms)
    hour_end = min(_floor_hour(end_ms), _floor_hour(coarse_usable_until_ms))
    if hour_end - hour_start < MIN_COARSE_HOURS * HOUR_MS:
        return [CoverSpan(start_ms, end_ms, BUCKET_MS)]

    spans = []
    if start_ms < hour_start:
        spans.append(CoverSpan(start_ms, hour_start, BUCKET_MS))
    spans.append(CoverSpan(hour_start, hour_end, HOUR_MS))
    if hour_end < end_ms:
        spans.append(CoverSpan(hour_end, end_ms, BUCKET_MS))
    return spans


def rows_read(spans: list[CoverSpan]) -> int:
    """Upper bound on rows a cover touches. For tests and for logging."""
    return sum((s.end_ms - s.start_ms) // s.width_ms for s in spans)
