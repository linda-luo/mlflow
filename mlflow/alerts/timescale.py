"""Postgres/TimescaleDB setup for the 1-hour rollup tier.

Everything here is a no-op on any other dialect. MLflow's default backend is
SQLite, and the 1-minute aggregator has to work there without Timescale -- the
continuous aggregate is the only Postgres-specific piece of the rollup layer.

Two tiers, deliberately. The 1h aggregate serves history *and* the evaluator's
full-read path -- a cold cache or the hourly recompute -- via
:mod:`mlflow.alerts.tiers`. A window ending at 10:37 is never itself an hourly
bucket, but the whole hours *inside* it are, which covers a 24h window in about 83
reads instead of 1,440 and a 3-day window in about 131 instead of 4,320.

There was a 1d aggregate. It was built and refreshed and read by nothing, so it was
removed rather than given a consumer: the hourly tier is retained ninety days, which
already covers the longest window by a wide margin, and a daily tier would only have
taken a 3-day cold read from 131 rows to 85 -- while adding a freshness gate, a third
bucket width, and a gap fallback where one bad minute invalidates a whole day.
``HOURLY_RETENTION_MS`` is therefore the hard history horizon.

That makes ``CA_END_OFFSET_MS`` below load-bearing for live alerting rather than
only for history: an hourly bucket materialized while the 1-minute aggregator was
behind is permanently short, so the reader refuses any hour our own watermark has
not passed by that offset.
"""

import logging
from dataclasses import dataclass, field

import sqlalchemy as sa

from mlflow.alerts.aggregator import LAG_MS
from mlflow.store.db import db_types

_logger = logging.getLogger(__name__)

_HOUR_MS = 3_600_000
_DAY_MS = 86_400_000

CHUNK_TIME_INTERVAL_MS = _DAY_MS

CA_END_OFFSET_MS = 3 * LAG_MS
"""How far behind the refresh boundary the 1h continuous aggregate stops.

Measured against **our watermark, not wall clock**. The continuous aggregate reads
``metric_rollups``, which the 1-minute aggregator populates, so the true lag is
``our sealing lag + this offset``. If the aggregator is running behind, Timescale
still refreshes on schedule and materializes an hourly bucket from partially
written minutes -- and a materialized continuous-aggregate bucket is never
revisited, so the under-count is silent and permanent.

Three times ``LAG_MS`` is the starting point; treat "watermark fell further behind
than this" as an operational alarm on the alerting system itself, and raise the
offset rather than letting it be exceeded.
"""

CA_REFRESH_WINDOW_BUCKETS = 2
"""Minimum refresh window, in bucket widths.

Timescale enforces this: a window narrower than two buckets is rejected outright
with "policy refresh window too small". One bucket width can never fully contain a
bucket that began before the window opened, so the boundary period would stay
perpetually half-materialized.

The start offsets are derived from ``end_offset`` inside :func:`setup_timescale`
rather than fixed here -- both offsets are measured back from now, so the window is
their difference, and a hardcoded start offset silently violates the rule whenever
``end_offset`` changes.
"""

RAW_RETENTION_MS = 3 * _DAY_MS
HOURLY_RETENTION_MS = 90 * _DAY_MS

_CURRENT_EPOCH_MS_SQL = """
CREATE OR REPLACE FUNCTION current_epoch_ms() RETURNS BIGINT
LANGUAGE SQL STABLE AS $$
  SELECT (EXTRACT(EPOCH FROM clock_timestamp()) * 1000)::BIGINT;
$$;
"""

# Index-union merge over sparse sketches.
#
# A sketch is a flat array of interleaved (bucket_index, count) pairs sorted by
# index, so merging is "union the indices, sum the counts" rather than the
# element-wise addition a fixed-width vector allowed. Sparse is what lets the
# bucket range be unbounded: a dense array would have to reserve a slot for every
# value anyone might ever record, which is exactly the clamp that made the previous
# implementation blind above 4 hours.
#
# Declared STRICT on purpose: Postgres seeds a strict transition function with the
# first non-null input, so the aggregate needs no initcond.
_HIST_ADD_SQL = """
CREATE OR REPLACE FUNCTION hist_add(a BIGINT[], b BIGINT[]) RETURNS BIGINT[]
LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE AS $$
  WITH pairs AS (
    SELECT arr[i] AS idx, arr[i + 1] AS cnt
    FROM (SELECT a AS arr UNION ALL SELECT b) AS t,
         LATERAL generate_series(1, COALESCE(array_length(arr, 1), 0), 2) AS i
  ), summed AS (
    SELECT idx, sum(cnt) AS cnt FROM pairs GROUP BY idx HAVING sum(cnt) <> 0
  )
  SELECT COALESCE(array_agg(v ORDER BY idx, ord), ARRAY[]::BIGINT[])
  FROM summed, LATERAL (VALUES (1, idx), (2, cnt)) AS x(ord, v);
$$;
"""

# `percentile_cont` is an ordered-set aggregate and is illegal inside a continuous
# aggregate, which is the whole reason this exists. The `combinefunc` is what makes
# it CA-legal; without one, Timescale refuses to materialize the view.
_HIST_MERGE_SQL = """
CREATE AGGREGATE hist_merge(BIGINT[]) (
    SFUNC = hist_add,
    STYPE = BIGINT[],
    COMBINEFUNC = hist_add,
    PARALLEL = SAFE
);
"""

_ROLLUP_1H_SQL = f"""
CREATE MATERIALIZED VIEW IF NOT EXISTS metric_rollups_1h
WITH (timescaledb.continuous) AS
SELECT
    series_id,
    time_bucket({_HOUR_MS}, bucket_start_ms) AS bucket_start_ms,
    sum(count)              AS count,
    sum(sum)                AS sum,
    hist_merge(histogram)   AS histogram,
    max(boundaries_version) AS boundaries_version,
    bool_or(is_gap)         AS is_gap
FROM metric_rollups
GROUP BY 1, 2
WITH NO DATA;
"""


@dataclass
class TimescaleSetupResult:
    applied: bool
    reason: str | None = None
    statements_run: list[str] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)


def is_postgres(engine: sa.engine.Engine) -> bool:
    return engine.dialect.name == db_types.POSTGRES


def is_timescale_installed(engine: sa.engine.Engine) -> bool:
    if not is_postgres(engine):
        return False
    with engine.connect() as conn:
        row = conn.execute(
            sa.text("SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'")
        ).first()
    return row is not None


def setup_timescale(
    engine: sa.engine.Engine,
    end_offset_ms: int = CA_END_OFFSET_MS,
    create_extension: bool = True,
) -> TimescaleSetupResult:
    """Turn ``metric_rollups`` into a hypertable and build the 1h tier.

    Idempotent, and a clean no-op on any dialect other than Postgres. Individual
    statements that fail because the object already exists are recorded rather than
    raised, so this can run on every server start.

    Args:
        engine: SQLAlchemy engine for the tracking database.
        end_offset_ms: How far behind the refresh boundary the 1h aggregate stops.
            See :data:`CA_END_OFFSET_MS` -- this is measured against the aggregator's
            watermark, not the wall clock.
        create_extension: Whether to attempt ``CREATE EXTENSION timescaledb``, which
            needs superuser. Set False when the extension is provisioned out of band.
    """
    if not is_postgres(engine):
        return TimescaleSetupResult(
            applied=False, reason=f"dialect {engine.dialect.name!r} is not PostgreSQL"
        )

    # Timescale rejects a policy whose refresh window is narrower than two buckets:
    # `start_offset` and `end_offset` are both measured back from now, so the window
    # is their *difference*, not `start_offset` itself. Hardcoding start_offset to
    # two bucket widths therefore fails as soon as end_offset is non-zero -- it lands
    # just under the limit, and Timescale reports only "policy refresh window too
    # small". Deriving it makes the invariant unbreakable.
    start_offset_1h = end_offset_ms + 2 * _HOUR_MS

    result = TimescaleSetupResult(applied=True)
    statements: list[tuple[str, str]] = []
    if create_extension:
        statements.append((
            "create_extension",
            "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;",
        ))
    statements.extend([
        ("current_epoch_ms", _CURRENT_EPOCH_MS_SQL),
        (
            "create_hypertable",
            "SELECT create_hypertable('metric_rollups', 'bucket_start_ms', "
            f"chunk_time_interval => {CHUNK_TIME_INTERVAL_MS}, "
            "migrate_data => TRUE, if_not_exists => TRUE);",
        ),
        (
            "set_integer_now_func",
            "SELECT set_integer_now_func('metric_rollups', 'current_epoch_ms', "
            "replace_if_exists => TRUE);",
        ),
        ("hist_add", _HIST_ADD_SQL),
        ("hist_merge", _HIST_MERGE_SQL),
        ("metric_rollups_1h", _ROLLUP_1H_SQL),
        (
            "policy_1h",
            "SELECT add_continuous_aggregate_policy('metric_rollups_1h', "
            f"start_offset => BIGINT '{start_offset_1h}', "
            f"end_offset => BIGINT '{end_offset_ms}', "
            "schedule_interval => INTERVAL '1 minute', if_not_exists => TRUE);",
        ),
        (
            "retention_raw",
            "SELECT add_retention_policy('metric_rollups', "
            f"BIGINT '{RAW_RETENTION_MS}', if_not_exists => TRUE);",
        ),
        (
            "retention_1h",
            "SELECT add_retention_policy('metric_rollups_1h', "
            f"BIGINT '{HOURLY_RETENTION_MS}', if_not_exists => TRUE);",
        ),
    ])

    for name, sql in statements:
        # Each statement gets its own transaction: `CREATE AGGREGATE` has no
        # IF NOT EXISTS, so a rerun must be able to fail without aborting the rest.
        with engine.connect() as conn:
            try:
                conn.execute(sa.text(sql))
                conn.commit()
            except sa.exc.SQLAlchemyError as e:
                conn.rollback()
                result.failures[name] = str(e)
                _logger.debug("Timescale setup step %s did not apply: %s", name, e)
            else:
                result.statements_run.append(name)
    return result
