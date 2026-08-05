"""Log-scale sketch bucketing, modelled on DDSketch.

DDSketch (Masson, Rim & Lee, VLDB 2019) buckets on a geometric grid: bucket ``i``
covers ``(gamma^(i-1), gamma^i]`` with ``gamma = (1 + alpha) / (1 - alpha)``. Any quantile read back
from it is within ``alpha`` *relative* error of the true one, at every magnitude.

That single property removes an entire class of problem the hand-written boundary
list created:

* **Nothing has to be snapped.** A threshold used to be moved onto the nearest
  boundary so a comparison could be answered exactly -- a rule asking for
  ``p95 > 700ms`` was stored, and alerted, at 1000ms. Here the grid is fine enough
  everywhere that the user's number is kept verbatim.
* **Nothing has to be refined.** Boundaries do not depend on which rules exist, so
  creating a rule cannot re-bucket a series, cannot invalidate history another rule
  was reading, and cannot leak across experiments.
* **Nothing overflows.** ``ceil(log_gamma(v))`` is defined for every positive value and
  goes negative below 1, so a 3-millisecond span and a 30-hour agent run share one
  sketch at the same accuracy. The previous boundaries stopped at 4 hours and put
  everything beyond into a single bucket of unbounded error.

Relative error is flat; *absolute* error scales with magnitude -- +/-2ms at 100ms,
+/-4.8min at 4h. That is the intended trade. Uniform absolute error is unachievable
(one fixed width is either useless at the low end or needs millions of buckets at
the high end), and the boundaries this replaces were worse at every magnitude: a
200% gap between 10s and 30s, and unbounded above 4h.

We implement this rather than using ``timescaledb_toolkit``'s ``uddsketch`` because
the toolkit is a second Postgres extension that the deployment does not carry, and
because the bucket index is arithmetic that both Postgres and SQLite compute
natively.
"""

import math
from dataclasses import dataclass

ALPHA = 0.02
"""Relative accuracy. A quantile read back is within 2% of the true value.

Chosen against how precisely anyone actually states a latency objective: a 30-minute
threshold resolves to +/-36 seconds, a 700ms threshold to +/-14ms. Halving it to 1%
doubles the occupied buckets and buys precision nobody expressed.
"""

GAMMA = (1 + ALPHA) / (1 - ALPHA)
_LOG_GAMMA = math.log(GAMMA)

SKETCH_VERSION = 1
"""Bump only if GAMMA or a metric's scale changes.

Stored on every rollup row so buckets written under different grids are detectable.
Unlike the boundary version it replaces, this does not change when rules change --
which is the entire point, and why it is a constant rather than derived state.
"""

ZERO_INDEX = -(2**31)
"""Bucket for exactly zero, whose logarithm is undefined.

Deliberately far below any real index (``log_gamma`` of a nanosecond is about -519), so
ordering by index still puts zeros first and quantile walks in value order without
a special case.
"""


@dataclass(frozen=True)
class SketchSpec:
    """How one metric's values map to bucket indices.

    A property of the *metric*, fixed in code. Not of an experiment, a rule, or a
    threshold -- that independence is what makes the grid stable.
    """

    log_scale: bool = True
    linear_step: float | None = None

    def index(self, value: float) -> int:
        """Bucket index for ``value``. Unbounded; negative below 1.

        Non-positive values go to :data:`ZERO_INDEX`. Zero because its logarithm is
        undefined; negative because a span whose end precedes its start (clock skew
        across hosts) is a real thing that reaches this code, and aborting the seal
        of a whole bucket over one skewed row would cost far more than counting it
        as zero. The SQL bucketing expression makes the same choice, so the two
        paths cannot disagree.
        """
        if value <= 0:
            return ZERO_INDEX
        if not self.log_scale:
            return math.ceil(value / self.linear_step)
        return math.ceil(math.log(value) / _LOG_GAMMA)

    def estimate(self, index: int) -> float:
        """Representative value of a bucket, within ``ALPHA`` of anything in it.

        ``2gamma^i / (gamma + 1)`` rather than the bucket's upper edge ``gamma^i``. The upper
        edge is what the previous implementation returned, and it is systematically
        high -- a true p95 of ~90 minutes displayed as 120. This sits in the middle
        of the bucket in relative terms, so the error is bounded by alpha in *both*
        directions instead of by the full bucket width in one.
        """
        if index == ZERO_INDEX:
            return 0.0
        if not self.log_scale:
            return (index - 0.5) * self.linear_step
        return 2 * GAMMA**index / (GAMMA + 1)

    def upper_edge(self, index: int) -> float:
        """Largest value that falls in this bucket. Used for exact bounds."""
        if index == ZERO_INDEX:
            return 0.0
        if not self.log_scale:
            return index * self.linear_step
        return GAMMA**index


LOG_SKETCH = SketchSpec(log_scale=True)

SCORE_SKETCH = SketchSpec(log_scale=False, linear_step=0.01)
"""Judge scores, on 0..1 in hundredths.

Linear on purpose. Relative error is the wrong guarantee for a pass rate: 2% of
0.95 is +/-0.019, which straddles the thresholds people actually set. And log spacing
puts its *coarsest* buckets near 1.0, which is exactly where pass rates cluster.
"""

SPECS: dict[str, SketchSpec] = {
    "latency": LOG_SKETCH,
    "input_cost": LOG_SKETCH,
    "output_cost": LOG_SKETCH,
    "total_cost": LOG_SKETCH,
    "input_tokens": LOG_SKETCH,
    "output_tokens": LOG_SKETCH,
    "total_tokens": LOG_SKETCH,
    "cache_read_input_tokens": LOG_SKETCH,
    "cache_creation_input_tokens": LOG_SKETCH,
    "assessment_value": SCORE_SKETCH,
}
"""Metrics that store a sketch.

A metric absent here stores no histogram at all, which is the honest answer for
``error_count``: a percentile over a count of events has nothing to bucket.
"""


def spec_for(metric_key: str) -> SketchSpec | None:
    return SPECS.get(metric_key)
