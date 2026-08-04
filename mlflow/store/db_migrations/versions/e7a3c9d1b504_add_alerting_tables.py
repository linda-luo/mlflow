"""Add alerting tables and denormalized rollup columns.

Creates the alerting schema in a single revision:

* ``span_errors``   -- deduped exception rows written in the ingest path
* ``metric_series`` -- rollup series identity, factored out of the bucket rows
* ``metric_rollups``-- sealed 1-minute buckets (a Timescale hypertable on Postgres)
* ``rollup_state``  -- aggregation watermark per (source, dimension_key, metric_key)
* ``alert_rules``   -- a saved metric query plus a predicate
* ``alert_instances`` -- one firing episode each

Also denormalizes five columns onto existing tables so the aggregator's scans need
no joins. All five are write-once, so there is no update-anomaly risk.

Create Date: 2026-07-31 10:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "e7a3c9d1b504"
down_revision = "6f8d9c3b2a1e"
branch_labels = None
depends_on = None


def _histogram_type():
    return sa.JSON().with_variant(postgresql.ARRAY(sa.BigInteger), "postgresql")


def upgrade():
    bind = op.get_bind()
    dialect = bind.dialect.name

    op.create_table(
        "span_errors",
        sa.Column("trace_id", sa.String(50), nullable=False),
        sa.Column("span_id", sa.String(50), nullable=False),
        sa.Column("exception_type", sa.String(250), nullable=False),
        sa.Column("parent_span_id", sa.String(50), nullable=True),
        sa.Column("is_origin", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("span_name", sa.String(500), nullable=False),
        sa.Column("span_type", sa.String(50), nullable=True),
        sa.Column("exception_message", sa.String(1000), nullable=True),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("timestamp_ms", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("trace_id", "span_id", "exception_type", name="span_errors_pk"),
        sa.ForeignKeyConstraint(
            ["trace_id", "span_id"],
            ["spans.trace_id", "spans.span_id"],
            name="fk_span_errors_span",
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "index_span_errors_dedup",
        "span_errors",
        ["trace_id", "exception_type", "parent_span_id"],
    )
    op.create_index(
        "index_span_errors_experiment_time",
        "span_errors",
        ["experiment_id", "timestamp_ms", "exception_type"],
    )
    op.create_index("index_span_errors_timestamp", "span_errors", ["timestamp_ms"])

    op.create_table(
        "metric_series",
        # The SQLite variant is load-bearing: SQLite only auto-assigns a primary key
        # when the column is declared exactly INTEGER (aliasing the rowid). A BIGINT
        # primary key there never autoincrements, so inserts fail NOT NULL.
        sa.Column(
            "series_id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("dimension_key", sa.String(20), nullable=False),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("metric_key", sa.String(250), nullable=False),
        sa.Column("dimension_value", sa.String(250), nullable=False, server_default=""),
        sa.PrimaryKeyConstraint("series_id", name="metric_series_pk"),
        sa.UniqueConstraint(
            "dimension_key",
            "experiment_id",
            "metric_key",
            "dimension_value",
            name="metric_series_identity",
        ),
    )
    op.create_index(
        "index_metric_series_experiment", "metric_series", ["experiment_id", "metric_key"]
    )

    op.create_table(
        "metric_rollups",
        sa.Column("series_id", sa.BigInteger(), nullable=False),
        sa.Column("bucket_start_ms", sa.BigInteger(), nullable=False),
        sa.Column("count", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("sum", sa.Float(precision=53), nullable=True),
        sa.Column("histogram", _histogram_type(), nullable=True),
        sa.Column("boundaries_version", sa.SmallInteger(), nullable=True),
        sa.Column("is_gap", sa.Boolean(), nullable=False, server_default=sa.false()),
        # Equality column first, range column last, so a single-series window read is
        # one contiguous scan rather than a strided one.
        sa.PrimaryKeyConstraint("series_id", "bucket_start_ms", name="metric_rollups_pk"),
        sa.ForeignKeyConstraint(
            ["series_id"],
            ["metric_series.series_id"],
            name="fk_metric_rollups_series",
            ondelete="CASCADE",
        ),
    )
    op.create_index("index_metric_rollups_bucket", "metric_rollups", ["bucket_start_ms"])

    # Keyed by series family -- one (source, dimension_key, metric_key) -- not by
    # source. Families write disjoint series, so nothing orders them: each can be
    # sealed by its own worker, and a rule can be evaluated against the freshness of
    # the one family it reads rather than the slowest family in the deployment.
    op.create_table(
        "rollup_state",
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("dimension_key", sa.String(20), nullable=False, server_default=""),
        sa.Column("metric_key", sa.String(250), nullable=False, server_default=""),
        sa.Column("watermark_ms", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("coverage_start_ms", sa.BigInteger(), nullable=True),
        sa.Column("last_updated_ms", sa.BigInteger(), nullable=True),
        sa.PrimaryKeyConstraint("source", "dimension_key", "metric_key", name="rollup_state_pk"),
    )

    op.create_table(
        "alert_rules",
        sa.Column("alert_rule_id", sa.String(36), nullable=False),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("description", sa.String(1000), nullable=True),
        sa.Column("severity", sa.String(10), nullable=False, server_default="MEDIUM"),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("metric_key", sa.String(250), nullable=False),
        sa.Column("dimension_key", sa.String(20), nullable=False),
        sa.Column("dimension_value", sa.String(250), nullable=True),
        sa.Column("aggregation", sa.String(20), nullable=False),
        sa.Column("percentile_value", sa.Float(precision=53), nullable=True),
        sa.Column("comparator", sa.String(4), nullable=False),
        sa.Column("threshold", sa.Float(precision=53), nullable=False),
        sa.Column("window_seconds", sa.Integer(), nullable=False),
        sa.Column("evaluation_interval_seconds", sa.Integer(), nullable=False),
        sa.Column("sustain_seconds", sa.Integer(), nullable=False, server_default="0"),
        # Zero, not one: a floor of one sample makes "traffic dropped to zero"
        # unfireable, which is the one rule whose signal is an empty window.
        sa.Column("min_sample_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_evaluated_ms", sa.BigInteger(), nullable=True),
        sa.Column("next_evaluation_at_ms", sa.BigInteger(), nullable=True),
        sa.Column("last_sample_count", sa.Integer(), nullable=True),
        sa.Column("deleted_at_ms", sa.BigInteger(), nullable=True),
        sa.Column("channels", sa.Text(), nullable=True),
        sa.Column("created_by", sa.String(255), nullable=True),
        sa.Column("creation_timestamp", sa.BigInteger(), nullable=True),
        sa.Column("last_updated_timestamp", sa.BigInteger(), nullable=True),
        sa.PrimaryKeyConstraint("alert_rule_id", name="alert_rules_pk"),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            name="fk_alert_rules_experiment",
            ondelete="CASCADE",
        ),
    )
    op.create_index("index_alert_rules_due", "alert_rules", ["enabled", "next_evaluation_at_ms"])
    op.create_index("index_alert_rules_experiment", "alert_rules", ["experiment_id"])

    # Names are unique among live rules only. Rules soft-delete so their instance
    # history survives, and an unconditional unique constraint therefore made a
    # name unusable forever once a rule wearing it had been deleted -- so a typo
    # permanently occupied the name it was created under. Same shape, and same
    # MySQL caveat, as `index_alert_instances_open` below.
    if dialect in ("postgresql", "sqlite"):
        op.create_index(
            "index_alert_rules_experiment_name",
            "alert_rules",
            ["experiment_id", "name"],
            unique=True,
            postgresql_where=sa.text("deleted_at_ms IS NULL"),
            sqlite_where=sa.text("deleted_at_ms IS NULL"),
        )

    op.create_table(
        "alert_instances",
        sa.Column("alert_instance_id", sa.String(36), nullable=False),
        sa.Column("alert_rule_id", sa.String(36), nullable=False),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("state", sa.String(20), nullable=False),
        sa.Column("started_at_ms", sa.BigInteger(), nullable=False),
        sa.Column("fired_at_ms", sa.BigInteger(), nullable=True),
        sa.Column("dismissed_at_ms", sa.BigInteger(), nullable=True),
        sa.Column("dismissed_by", sa.String(255), nullable=True),
        # Start of the current unbroken run of healthy evaluations. Cleared by the
        # next breaching one, so it is both the "one healthy blip is not a
        # recovery" flag and, for an INACTIVE instance, the time it recovered.
        sa.Column("healthy_since_ms", sa.BigInteger(), nullable=True),
        sa.Column("observed_value", sa.Float(precision=53), nullable=True),
        sa.Column("peak_value", sa.Float(precision=53), nullable=True),
        sa.Column("threshold", sa.Float(precision=53), nullable=True),
        sa.Column("sample_count", sa.Integer(), nullable=True),
        sa.Column("window_start_ms", sa.BigInteger(), nullable=True),
        sa.Column("window_end_ms", sa.BigInteger(), nullable=True),
        sa.Column("exemplar_trace_ids", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("alert_instance_id", name="alert_instances_pk"),
        # Deliberately no ON DELETE CASCADE: rules soft-delete so that the record of
        # what fired survives the cleanup action most likely to precede a postmortem.
        sa.ForeignKeyConstraint(
            ["alert_rule_id"],
            ["alert_rules.alert_rule_id"],
            name="fk_alert_instances_rule",
        ),
    )
    op.create_index(
        "index_alert_instances_inbox",
        "alert_instances",
        ["experiment_id", "state", "started_at_ms"],
    )
    op.create_index(
        "index_alert_instances_rule", "alert_instances", ["alert_rule_id", "started_at_ms"]
    )

    # At most one open instance per rule. MySQL has no partial indexes, so there the
    # invariant is enforced in the store layer instead -- a plain unique index would
    # wrongly prevent a rule from ever firing again after being dismissed.
    #
    # The predicate names PENDING and FIRED rather than "not DISMISSED", which is what
    # lets a recovered (INACTIVE) instance stay on screen while the rule opens the next
    # episode. `_assert_no_open_alert_instance` in the store must agree exactly.
    if dialect in ("postgresql", "sqlite"):
        op.create_index(
            "index_alert_instances_open",
            "alert_instances",
            ["alert_rule_id"],
            unique=True,
            postgresql_where=sa.text("state IN ('PENDING', 'FIRED')"),
            sqlite_where=sa.text("state IN ('PENDING', 'FIRED')"),
        )

    # Denormalized columns: five columns across four tables remove every join from the
    # aggregator, which also lets each source keep its own watermark and cadence.
    op.add_column("trace_info", sa.Column("end_time_ms", sa.BigInteger(), nullable=True))
    op.create_index("index_trace_info_end_time_ms", "trace_info", ["end_time_ms"])

    op.add_column("trace_metrics", sa.Column("experiment_id", sa.Integer(), nullable=True))
    op.add_column("trace_metrics", sa.Column("timestamp_ms", sa.BigInteger(), nullable=True))
    op.create_index("index_trace_metrics_rollup", "trace_metrics", ["timestamp_ms", "key"])

    op.add_column("span_metrics", sa.Column("experiment_id", sa.Integer(), nullable=True))
    op.add_column("span_metrics", sa.Column("timestamp_ms", sa.BigInteger(), nullable=True))
    op.create_index("index_span_metrics_rollup", "span_metrics", ["timestamp_ms", "key"])

    op.add_column("assessments", sa.Column("experiment_id", sa.Integer(), nullable=True))
    op.create_index("index_assessments_rollup", "assessments", ["created_timestamp", "name"])

    # Every existing spans index is experiment_id-leading, so none can serve the
    # rollup's global time-ordered scan.
    op.create_index("index_spans_end_time_unix_nano", "spans", ["end_time_unix_nano"])

    # Backfill completion time for existing traces so newly created rules can preview
    # against history rather than starting blind.
    op.execute(
        "UPDATE trace_info SET end_time_ms = timestamp_ms + execution_time_ms "
        "WHERE execution_time_ms IS NOT NULL"
    )


def downgrade():
    op.drop_index("index_spans_end_time_unix_nano", table_name="spans")

    op.drop_index("index_assessments_rollup", table_name="assessments")
    op.drop_column("assessments", "experiment_id")

    op.drop_index("index_span_metrics_rollup", table_name="span_metrics")
    op.drop_column("span_metrics", "timestamp_ms")
    op.drop_column("span_metrics", "experiment_id")

    op.drop_index("index_trace_metrics_rollup", table_name="trace_metrics")
    op.drop_column("trace_metrics", "timestamp_ms")
    op.drop_column("trace_metrics", "experiment_id")

    op.drop_index("index_trace_info_end_time_ms", table_name="trace_info")
    op.drop_column("trace_info", "end_time_ms")

    op.drop_table("alert_instances")
    op.drop_table("alert_rules")
    op.drop_index("index_rollup_state_claim", table_name="rollup_state")
    op.drop_table("rollup_state")
    op.drop_table("metric_rollups")
    op.drop_table("metric_series")
    op.drop_table("span_errors")
