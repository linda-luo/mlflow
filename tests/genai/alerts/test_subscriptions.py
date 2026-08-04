"""Aggregate only what some rule reads.

Cardinality used to be bounded by whatever agents emitted; it is now bounded by what
someone actually asked for. Subscriptions are a ``SELECT DISTINCT`` over
``alert_rules`` rather than a table of their own, so there is no reconciliation to
test and no drift to guard against -- a rule and its subscription are the same fact.

Bucketing is deliberately *not* driven from here; see ``test_sketch.py`` for why the
grid is a constant.
"""

import uuid
from pathlib import Path

import pytest

from mlflow.genai.alerts.aggregator import LAG_MS, RollupAggregator, build_work_units
from mlflow.genai.alerts.entities import BUCKET_MS, AlertRule
from mlflow.genai.alerts.subscriptions import load_active_subscriptions
from mlflow.store.tracking.dbmodels.models import (
    SqlMetricSeries,
    SqlRollupState,
    SqlSpan,
    SqlTraceInfo,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

T = 1_700_000_040_000
MINUTE = 60_000


@pytest.fixture
def store(tmp_path: Path, db_uri: str) -> SqlAlchemyStore:
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir()
    return SqlAlchemyStore(db_uri, artifact_uri.as_uri())


@pytest.fixture
def experiment_id(store: SqlAlchemyStore) -> int:
    return int(store.create_experiment("subscriptions"))


def _rule(experiment_id: int, **overrides) -> AlertRule:
    kwargs = {
        "alert_rule_id": "",
        "experiment_id": int(experiment_id),
        "name": f"rule-{uuid.uuid4().hex[:8]}",
        "metric_key": "latency",
        "dimension_key": "TRACES",
        "aggregation": "PERCENTILE",
        "percentile_value": 95.0,
        "comparator": "GT",
        "threshold": 45 * MINUTE,
        "window_seconds": 600,
        "evaluation_interval_seconds": 60,
    }
    kwargs.update(overrides)
    return AlertRule(**kwargs)


def _set_watermark(store: SqlAlchemyStore, watermark_ms: int) -> None:
    with store.ManagedSessionMaker(read_only=False) as session:
        for unit in build_work_units(store.db_type):
            source, dimension_key = unit.key
            row = session.get(SqlRollupState, unit.key)
            if row is None:
                session.add(
                    SqlRollupState(
                        source=source,
                        dimension_key=dimension_key,
                        watermark_ms=watermark_ms,
                    )
                )
            else:
                row.watermark_ms = watermark_ms


def _seed_trace(store, experiment_id, trace_id, *, end_ms, duration_ms):
    with store.ManagedSessionMaker(read_only=False) as session:
        session.add(
            SqlTraceInfo(
                request_id=trace_id,
                experiment_id=int(experiment_id),
                timestamp_ms=end_ms - duration_ms,
                execution_time_ms=duration_ms,
                end_time_ms=end_ms,
                status="OK",
            )
        )


def _seed_tool_span(store, experiment_id, trace_id, span_id, name, end_ms):
    with store.ManagedSessionMaker(read_only=False) as session:
        session.add(
            SqlSpan(
                trace_id=trace_id,
                experiment_id=int(experiment_id),
                span_id=span_id,
                name=name,
                type="TOOL",
                status="OK",
                start_time_unix_nano=(end_ms - 400) * 1_000_000,
                end_time_unix_nano=end_ms * 1_000_000,
                content="{}",
                dimension_attributes={},
            )
        )


def _seal(store, experiment_id=None):
    _set_watermark(store, T - BUCKET_MS)
    RollupAggregator(store).run_once(now_ms=T + BUCKET_MS + LAG_MS)


def _written_series(store) -> set[tuple[str, str]]:
    with store.ManagedSessionMaker() as session:
        return {(row.dimension_key, row.metric_key) for row in session.query(SqlMetricSeries).all()}


###############################################################################
# Subscriptions come from rules
###############################################################################


def test_subscriptions_are_derived_from_the_rules(store, experiment_id):
    """No table of its own -- the rules *are* the subscription set."""
    with store.ManagedSessionMaker() as session:
        assert load_active_subscriptions(session) == []

    store.create_alert_rule(_rule(experiment_id))

    with store.ManagedSessionMaker() as session:
        (subscription,) = load_active_subscriptions(session)
    assert subscription.experiment_id == int(experiment_id)
    assert subscription.dimension_key == "TRACES"
    assert subscription.metric_key == "latency"


def test_rules_reading_one_family_collapse_to_one_subscription(store, experiment_id):
    """Four rules on TRACES/latency is one thing to aggregate, not four."""
    for _ in range(4):
        store.create_alert_rule(_rule(experiment_id))

    with store.ManagedSessionMaker() as session:
        assert len(load_active_subscriptions(session)) == 1


###############################################################################
# What that means for what gets written
###############################################################################


def test_nothing_is_aggregated_without_a_rule(store, experiment_id):
    """The headline behaviour: no reader, no rows.

    Series used to be created by observation, so cardinality was bounded by what
    agents happened to emit rather than by what anyone asked for.
    """
    _seed_trace(store, experiment_id, "t-1", end_ms=T + 1_000, duration_ms=1_500)
    _seal(store)

    assert _written_series(store) == set()


def test_creating_a_rule_is_what_makes_its_series_aggregated(store, experiment_id):
    store.create_alert_rule(_rule(experiment_id))
    _seed_trace(store, experiment_id, "t-1", end_ms=T + 1_000, duration_ms=1_500)
    _seal(store)

    assert ("TRACES", "latency") in _written_series(store)


def test_only_the_subscribed_family_is_written(store, experiment_id):
    """A latency rule must not drag every other family along with it."""
    store.create_alert_rule(_rule(experiment_id))
    _seed_trace(store, experiment_id, "t-1", end_ms=T + 1_000, duration_ms=1_500)
    _seed_tool_span(store, experiment_id, "t-1", "s-1", "search_docs", T + 2_000)
    _seal(store)

    written = _written_series(store)
    assert ("TRACES", "latency") in written
    # The spans were there and were scanned; no rule reads them.
    assert ("SPAN_TYPE", "latency") not in written
    assert ("SPAN_NAME", "latency") not in written


def test_another_experiments_data_is_not_written_by_this_ones_rule(store, experiment_id):
    other = int(store.create_experiment("unsubscribed-experiment"))
    store.create_alert_rule(_rule(experiment_id))
    _seed_trace(store, experiment_id, "t-mine", end_ms=T + 1_000, duration_ms=1_500)
    _seed_trace(store, other, "t-theirs", end_ms=T + 1_000, duration_ms=1_500)
    _seal(store)

    with store.ManagedSessionMaker() as session:
        experiments = {row.experiment_id for row in session.query(SqlMetricSeries).all()}
    assert experiments == {int(experiment_id)}


def test_a_metric_no_rule_can_express_is_never_aggregated(store, experiment_id):
    """`input_tokens` is not in METRIC_CATALOGUE, so nothing can ever read it.

    Falls out of deriving subscriptions from rules: a family with no expressible
    rule has no subscriber by construction, and the aggregator skips it without
    anyone having to maintain an exclusion list.
    """
    store.create_alert_rule(
        _rule(experiment_id, metric_key="total_tokens", aggregation="SUM", percentile_value=None)
    )

    with store.ManagedSessionMaker() as session:
        families = {(s.dimension_key, s.metric_key) for s in load_active_subscriptions(session)}
    assert ("TRACES", "total_tokens") in families
    assert ("TRACES", "input_tokens") not in families


###############################################################################
# Rule lifecycle
###############################################################################


def test_disabling_a_rule_stops_the_aggregation_it_justified(store, experiment_id):
    rule = store.create_alert_rule(_rule(experiment_id))
    store.update_alert_rule(rule.alert_rule_id, enabled=False)

    with store.ManagedSessionMaker() as session:
        assert load_active_subscriptions(session) == []


def test_deleting_a_rule_stops_the_aggregation_it_justified(store, experiment_id):
    rule = store.create_alert_rule(_rule(experiment_id))
    store.delete_alert_rule(rule.alert_rule_id)

    with store.ManagedSessionMaker() as session:
        assert load_active_subscriptions(session) == []


def test_rollups_already_written_survive_the_rule_being_deleted(store, experiment_id):
    """History outlives the cleanup action most likely to precede a postmortem."""
    rule = store.create_alert_rule(_rule(experiment_id))
    _seed_trace(store, experiment_id, "t-1", end_ms=T + 1_000, duration_ms=1_500)
    _seal(store)
    assert ("TRACES", "latency") in _written_series(store)

    store.delete_alert_rule(rule.alert_rule_id)

    assert ("TRACES", "latency") in _written_series(store)


def test_one_rule_of_several_going_away_keeps_the_family_subscribed(store, experiment_id):
    keep = store.create_alert_rule(_rule(experiment_id))
    drop = store.create_alert_rule(_rule(experiment_id))

    store.delete_alert_rule(drop.alert_rule_id)

    with store.ManagedSessionMaker() as session:
        (subscription,) = load_active_subscriptions(session)
    assert subscription.metric_key == "latency"
    assert store.get_alert_rule(keep.alert_rule_id).enabled
