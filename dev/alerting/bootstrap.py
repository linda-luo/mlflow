"""Prepare the demo database: migrations, Timescale tiers, and the alert rules.

Idempotent, so `docker compose up` can be run repeatedly.
"""

import os
import uuid

import sqlalchemy as sa

from mlflow.genai.alerts.entities import AlertRule, derive_evaluation_interval_seconds
from mlflow.genai.alerts.timescale import is_timescale_installed, setup_timescale
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

MINUTE_MS = 60_000
EXPERIMENT_NAME = "checkout-agent"


def rule(experiment_id, name, **kwargs):
    window_seconds = kwargs.pop("window_seconds", 900)
    return AlertRule(
        alert_rule_id=str(uuid.uuid4()),
        experiment_id=int(experiment_id),
        name=name,
        window_seconds=window_seconds,
        evaluation_interval_seconds=derive_evaluation_interval_seconds(window_seconds),
        **kwargs,
    )


def main():
    uri = os.environ["MLFLOW_BACKEND_STORE_URI"]
    # Constructing the store runs the alembic migrations.
    store = SqlAlchemyStore(uri, "file:///tmp/mlflow-artifacts")

    engine = sa.create_engine(uri)
    result = setup_timescale(engine)
    print(f"[bootstrap] timescale extension present: {is_timescale_installed(engine)}")
    print(f"[bootstrap] timescale setup applied: {result.applied} ({result.reason or 'ok'})")
    for name in result.statements_run:
        print(f"[bootstrap]   ran {name}")
    for name, error in result.failures.items():
        print(f"[bootstrap]   SKIPPED {name}: {error}")

    existing = store.search_experiments(filter_string=f"name = '{EXPERIMENT_NAME}'")
    if existing:
        experiment_id = existing[0].experiment_id
        print(f"[bootstrap] experiment {EXPERIMENT_NAME} already exists (id={experiment_id})")
    else:
        experiment_id = store.create_experiment(EXPERIMENT_NAME)
        print(f"[bootstrap] created experiment {EXPERIMENT_NAME} (id={experiment_id})")

    if store.list_alert_rules(experiment_id):
        print("[bootstrap] alert rules already present")
        return

    # The experiment is always created -- `traffic.py` polls for it, and a rule has
    # to live somewhere. Only the demo rules are opt-out, for evaluating the rule
    # editor from an empty page.
    #
    # Note the consequence: aggregation is demand-driven, so with no rules nothing
    # subscribes to a series and the aggregator writes no rollups at all. Buckets
    # start being sealed for a family when the first rule reading it is created.
    if os.environ.get("DEMO_SEED_RULES", "1") == "0":
        print("[bootstrap] DEMO_SEED_RULES=0 -- experiment ready, seeding no demo rules")
        return

    # 300s windows: MIN_WINDOW_SECONDS, the shortest the schema permits. A rule
    # waits one whole window before its first evaluation, so this is also what sets
    # how long the demo takes to say anything.
    rules = [
        rule(
            experiment_id, "Checkout p95 latency",
            metric_key="latency", dimension_key="TRACES", aggregation="PERCENTILE",
            percentile_value=95, comparator="GT", threshold=30 * MINUTE_MS,
            severity="HIGH", sustain_seconds=120, window_seconds=300,
        ),
        rule(
            experiment_id, "Average latency regression",
            metric_key="latency", dimension_key="TRACES", aggregation="AVG",
            comparator="GT", threshold=10 * MINUTE_MS, severity="MEDIUM",
            window_seconds=300,
        ),
        rule(
            experiment_id, "search_docs failures",
            metric_key="error_count", dimension_key="SPAN_NAME",
            dimension_value="search_docs", aggregation="COUNT",
            comparator="GTE", threshold=5, severity="HIGH", window_seconds=300,
        ),
        rule(
            experiment_id, "Safety judge pass rate",
            metric_key="assessment_value", dimension_key="ASSESSMENTS",
            dimension_value="safety", aggregation="AVG",
            comparator="LT", threshold=0.9, severity="HIGH", window_seconds=300,
        ),
        rule(
            experiment_id, "Traffic dropped",
            metric_key="latency", dimension_key="TRACES", aggregation="COUNT",
            comparator="LT", threshold=5, severity="LOW", window_seconds=300,
        ),
    ]
    for r in rules:
        created = store.create_alert_rule(r)
        print(
            f"[bootstrap] rule {created.name!r} "
            f"(every {created.evaluation_interval_seconds}s, "
            f"needs {created.min_sample_count} samples)"
        )


if __name__ == "__main__":
    main()
