"""Database-backed :class:`RollupReader` -- the real thing behind the seam.

Drop-in replacement for :class:`~mlflow.genai.alerts.rollup_reader.FakeRollupReader`:
same two Protocol methods, same :class:`Bucket` return shape, so the evaluator does
not change when it is swapped in.
"""

import logging

import sqlalchemy as sa

from mlflow.genai.alerts.aggregator import build_work_units
from mlflow.genai.alerts.entities import BUCKET_MS, SeriesKey
from mlflow.genai.alerts.rollup_reader import Bucket, RollupReader
from mlflow.genai.alerts.sketch import SketchSpec, spec_for
from mlflow.genai.alerts.tiers import HOUR_MS, CoverSpan, plan_cover
from mlflow.store.db import db_types
from mlflow.store.tracking.dbmodels.models import (
    SqlMetricRollup,
    SqlMetricSeries,
    SqlRollupState,
)

_logger = logging.getLogger(__name__)

_NOTHING_IS_COVERED = 1 << 62
"""Sentinel coverage start: later than any real window, so nothing looks covered.

Deliberately not ``0``, which would make an un-aggregated family look like it had
complete history back to the epoch -- the exact false-confidence this guards.
"""

_HOURLY_VIEW = "metric_rollups_1h"

_HOURLY_ROWS_SQL = sa.text(
    f"""
    SELECT bucket_start_ms, count, sum, histogram, boundaries_version, is_gap
    FROM {_HOURLY_VIEW}
    WHERE series_id = :series_id
      AND bucket_start_ms >= :start_ms
      AND bucket_start_ms < :end_ms
    ORDER BY bucket_start_ms
    """
)

_CAGG_WATERMARK_SQL = sa.text(
    """
    SELECT _timescaledb_functions.cagg_watermark(mat_hypertable_id)
    FROM _timescaledb_catalog.continuous_agg
    WHERE user_view_name = :view_name
    """
)


class SqlRollupReader(RollupReader):
    def __init__(self, store):
        self._session_maker = store.ManagedSessionMaker
        self._series_cache: dict[SeriesKey, int] = {}
        self._units = build_work_units(store.db_type)
        self._db_type = store.db_type
        self._hourly_tier_available: bool | None = None
        """Probed once, lazily. ``setup_timescale`` runs only from the dev
        bootstrap, so a real Postgres deployment may have a plain table and no
        continuous aggregates at all -- which makes this a capability question,
        not a dialect one."""

    def read_buckets(self, series: SeriesKey, start_ms: int, end_ms: int) -> list[Bucket]:
        with self._session_maker() as session:
            series_id = self._series_id(session, series)
            if series_id is None:
                return []
            return self._read_minutes(session, series_id, start_ms, end_ms)

    def read_window(self, series: SeriesKey, start_ms: int, end_ms: int) -> list[Bucket]:
        """A whole window, stitched from the coarsest tier that fits.

        Only for the full-read path -- a cold cache or the hourly recompute -- where
        the window is read end to end. The incremental path reads a few minutes
        either side and has nothing to gain here.

        Falls back to the 1-minute read on anything unexpected: the tier may not
        exist, and being slower is always better than being wrong.
        """
        # Read outside the session below rather than inside it: this opens a
        # session of its own, and nesting one inside another would hold two
        # connections per thread at once against a pool of fifteen shared with
        # every other periodic task.
        watermark_ms = self.latest_sealed_bucket_ms(series)

        with self._session_maker() as session:
            series_id = self._series_id(session, series)
            if series_id is None:
                return []
            if not self._hourly_available(session):
                return self._read_minutes(session, series_id, start_ms, end_ms)

            usable_until_ms = self._hourly_usable_until(session, watermark_ms)
            spans = plan_cover(start_ms, end_ms, usable_until_ms)
            if not any(span.is_coarse for span in spans):
                return self._read_minutes(session, series_id, start_ms, end_ms)

            buckets: list[Bucket] = []
            for span in spans:
                if span.is_coarse:
                    buckets.extend(self._read_hours(session, series_id, span))
                else:
                    buckets.extend(
                        self._read_minutes(session, series_id, span.start_ms, span.end_ms)
                    )
            return buckets

    def _read_minutes(self, session, series_id: int, start_ms: int, end_ms: int) -> list[Bucket]:
        rows = (
            session
            .query(SqlMetricRollup)
            .filter(
                SqlMetricRollup.series_id == series_id,
                SqlMetricRollup.bucket_start_ms >= start_ms,
                SqlMetricRollup.bucket_start_ms < end_ms,
            )
            .order_by(SqlMetricRollup.bucket_start_ms)
            .all()
        )
        return [
            Bucket(
                bucket_start_ms=row.bucket_start_ms,
                count=row.count,
                sum=row.sum,
                histogram=list(row.histogram) if row.histogram is not None else None,
                boundaries_version=row.boundaries_version,
                is_gap=row.is_gap,
            )
            for row in rows
        ]

    def _read_hours(self, session, series_id: int, span: CoverSpan) -> list[Bucket]:
        """Hourly rows for one coarse span, minus any hour that contains a gap.

        The views propagate ``bool_or(is_gap)``, so one bad minute marks the whole
        hour -- and a gap suppresses the entire window's observation. Falling back
        to that hour's sixty minutes keeps the answer identical to the unstitched
        read, and keeps ``gap_buckets`` keyed at minute granularity so the gap can
        still leave the window when it expires. Gaps are rare, so this normally
        costs nothing.
        """
        rows = session.execute(
            _HOURLY_ROWS_SQL,
            {"series_id": series_id, "start_ms": span.start_ms, "end_ms": span.end_ms},
        ).all()

        buckets: list[Bucket] = []
        for row in rows:
            if row.is_gap:
                buckets.extend(
                    self._read_minutes(
                        session, series_id, row.bucket_start_ms, row.bucket_start_ms + HOUR_MS
                    )
                )
                continue
            buckets.append(
                Bucket(
                    bucket_start_ms=row.bucket_start_ms,
                    # `int()`, because the two tiers disagree on type: the raw column
                    # is BIGINT and arrives as `int`, but the view's `sum(count)` is
                    # Postgres `numeric` and arrives as `decimal.Decimal`. Decimal
                    # refuses to multiply by float, so a stitched read reached
                    # `percentile_count_threshold` and raised `TypeError` -- every
                    # PERCENTILE rule whose window spanned a whole hour failed to
                    # evaluate, forever, visible only in the log.
                    count=int(row.count),
                    sum=row.sum,
                    histogram=list(row.histogram) if row.histogram is not None else None,
                    boundaries_version=row.boundaries_version,
                    is_gap=False,
                    width_ms=HOUR_MS,
                )
            )
        return buckets

    def _hourly_available(self, session) -> bool:
        if self._hourly_tier_available is not None:
            return self._hourly_tier_available
        if self._db_type != db_types.POSTGRES:
            self._hourly_tier_available = False
            return False
        try:
            self._hourly_tier_available = (
                session.execute(_CAGG_WATERMARK_SQL, {"view_name": _HOURLY_VIEW}).scalar()
                is not None
            )
        except Exception:
            # No Timescale, or a catalog laid out differently by another version.
            # Either way the 1-minute tier is always there.
            _logger.debug("Hourly rollup tier unavailable; reading minutes", exc_info=True)
            self._hourly_tier_available = False
        return self._hourly_tier_available

    def _hourly_usable_until(self, session, watermark_ms: int) -> int:
        """Exclusive bound on hourly rows this read may trust.

        Two conditions, and the second is the one that is easy to miss. Timescale's
        own watermark says what it has materialized. But ``CA_END_OFFSET_MS`` is
        measured against *our* watermark, not the clock: if the 1-minute aggregator
        falls behind, Timescale still refreshes on schedule, materializes an hour
        from partially written minutes, and never revisits it. So an hour is only
        trustworthy once our sealing has also passed it by that offset.
        """
        from mlflow.genai.alerts.timescale import CA_END_OFFSET_MS

        try:
            cagg_watermark = session.execute(
                _CAGG_WATERMARK_SQL, {"view_name": _HOURLY_VIEW}
            ).scalar()
        except Exception:
            _logger.debug("Could not read the continuous-aggregate watermark", exc_info=True)
            return 0
        if cagg_watermark is None:
            return 0
        return min(int(cagg_watermark), watermark_ms + BUCKET_MS - CA_END_OFFSET_MS)

    def latest_sealed_bucket_ms(self, series: SeriesKey | None = None) -> int:
        """Watermark of the families a rule actually depends on.

        With ``series``, this is the minimum over the units that write that series
        -- normally exactly one. Without it, the minimum over every unit, which is
        the deployment-wide answer and the right one for a caller that has no
        particular series in mind.

        The distinction matters more than it looks. A single global watermark meant
        one slow family froze the evaluation window for *every* rule: assessments
        land minutes after the traces they score, so a quality rollup running behind
        would stall latency alerting that had nothing to do with it. Scoping the
        watermark to the family a rule reads is what stops one slow signal degrading
        every other one.

        Still a minimum, never a maximum: a window is only sealed once every unit
        feeding it has sealed it, and taking the maximum would hand the evaluator a
        window that is still being written.
        """
        with self._session_maker() as session:
            rows = session.query(SqlRollupState).all()
            watermarks = {(r.source, r.dimension_key, r.metric_key): r.watermark_ms for r in rows}

        if series is not None:
            relevant = [
                watermarks[unit.key]
                for unit in self._units
                if unit.grouping.dimension_key == series.dimension_key
                and unit.metric_key == series.metric_key
                and unit.key in watermarks
            ]
            # An unknown family has never been sealed, so nothing about it is
            # readable yet -- the same answer a fresh install gives.
            return min(relevant) if relevant else 0

        if any(unit.key not in watermarks for unit in self._units):
            return 0
        return min(watermarks[unit.key] for unit in self._units)

    def sketch_for(self, series: SeriesKey) -> SketchSpec | None:
        """How this series' histograms are bucketed.

        A pure lookup on the metric -- no database read, and no dependence on which
        rules exist. That independence is the whole point of the sketch: the grid
        cannot move because someone created an alert.
        """
        return spec_for(series.metric_key)

    def coverage_start_ms(self, series: SeriesKey) -> int:
        """Earliest bucket for which this series has trustworthy history.

        A window reaching back before this is not evidence of anything, so an empty
        result over it means "we were not looking" rather than "nothing happened".
        Only a rule whose signal is *absence* can tell the difference, and for that
        rule the difference is a false alarm.

        Two starts, and the answer is the later of them:

        * when the **family** was first sealed, from ``rollup_state``. Guards the
          fresh install, where aggregation began after the window opens.
        Per *family*, deliberately, not per series. Narrowing it to the series' own
        earliest bucket looks like an improvement and is not: an empty bucket writes
        no row whether the series was being watched or simply had no traffic, so the
        first stored bucket cannot distinguish "we only started writing this series
        recently" from "this series was quiet until now". Keying off it suppresses
        the second case, which is exactly the window an absence rule exists to
        measure -- a rule whose traffic starts mid-window would read as no-data
        rather than as the genuine zero it is.
        """
        with self._session_maker() as session:
            # Values extracted inside the session: ORM instances detach when it
            # closes, and reading an attribute afterwards raises.
            starts = [
                row.coverage_start_ms
                for row in session.query(SqlRollupState).all()
                if row.dimension_key == series.dimension_key
                and row.metric_key == series.metric_key
                and row.coverage_start_ms is not None
            ]
        # No row yet means nothing has ever been sealed for this family, so no
        # window is covered.
        return max(starts) if starts else _NOTHING_IS_COVERED

    def _series_id(self, session, series: SeriesKey) -> int | None:
        if series in self._series_cache:
            return self._series_cache[series]
        row = (
            session
            .query(SqlMetricSeries.series_id)
            .filter(
                SqlMetricSeries.dimension_key == series.dimension_key,
                SqlMetricSeries.experiment_id == series.experiment_id,
                SqlMetricSeries.metric_key == series.metric_key,
                SqlMetricSeries.dimension_value == series.dimension_value,
            )
            .one_or_none()
        )
        series_id = row[0] if row is not None else None
        # Only a hit is worth caching: a series that does not exist yet is exactly
        # the one about to be created by the next sealing pass.
        if series_id is not None:
            self._series_cache[series] = series_id
        return series_id
