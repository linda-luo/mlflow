from pathlib import Path
from unittest import mock

import pytest

from mlflow.alerts.entities import BUCKET_MS, AlertRule
from mlflow.alerts.exemplars import MAX_EXEMPLARS, collect_exemplar_trace_ids
from mlflow.store.tracking.dbmodels.models import SqlTraceInfo
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

T0 = (1_700_000_000_000 // BUCKET_MS) * BUCKET_MS
MINUTE = 60_000


@pytest.fixture
def store(tmp_path: Path, db_uri: str) -> SqlAlchemyStore:
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir()
    return SqlAlchemyStore(db_uri, artifact_uri.as_uri())


@pytest.fixture
def experiment_id(store: SqlAlchemyStore) -> str:
    return store.create_experiment("alerting")


def _rule(experiment_id: str, **overrides) -> AlertRule:
    kwargs = {
        "alert_rule_id": "rule-1",
        "experiment_id": int(experiment_id),
        "name": "Slow checkout responses",
        "metric_key": "latency",
        "dimension_key": "TRACES",
        "aggregation": "PERCENTILE",
        "percentile_value": 95.0,
        "comparator": "GT",
        "threshold": 30 * MINUTE,
        "window_seconds": 600,
        "evaluation_interval_seconds": 60,
    }
    kwargs.update(overrides)
    return AlertRule(**kwargs)


def _seed_traces(store, experiment_id, durations_ms, status="OK", at_ms=T0):
    with store.ManagedSessionMaker(read_only=False) as session:
        for i, duration in enumerate(durations_ms):
            session.add(
                SqlTraceInfo(
                    request_id=f"tr-{duration}",
                    experiment_id=int(experiment_id),
                    timestamp_ms=at_ms + i,
                    execution_time_ms=duration,
                    end_time_ms=at_ms + i,
                    status=status,
                )
            )


def test_the_worst_traces_come_back_worst_first(store, experiment_id):
    """The point is evidence, so the slowest requests lead -- not an arbitrary page."""
    _seed_traces(store, experiment_id, [1_000, 90_000, 5_000, 60_000, 2_000, 45_000])

    ids = collect_exemplar_trace_ids(store, _rule(experiment_id), T0 - BUCKET_MS, T0 + BUCKET_MS)

    assert ids == ["tr-90000", "tr-60000", "tr-45000", "tr-5000", "tr-2000"]


def test_a_less_than_rule_takes_the_lowest_values(store, experiment_id):
    """Worst follows the comparator.

    For a pass-rate rule ("< 0.9") the damning traces are the *lowest*; sorting
    descending would list the healthiest ones as evidence of an outage.
    """
    _seed_traces(store, experiment_id, [1_000, 90_000, 5_000, 60_000])

    ids = collect_exemplar_trace_ids(
        store,
        _rule(experiment_id, comparator="LT", threshold=10_000),
        T0 - BUCKET_MS,
        T0 + BUCKET_MS,
    )

    assert ids == ["tr-1000", "tr-5000", "tr-60000", "tr-90000"]


def test_no_more_than_the_cap(store, experiment_id):
    _seed_traces(store, experiment_id, [1_000 * i for i in range(1, 30)])

    ids = collect_exemplar_trace_ids(store, _rule(experiment_id), T0 - BUCKET_MS, T0 + BUCKET_MS)

    assert len(ids) == MAX_EXEMPLARS


def test_only_traces_inside_the_window_count(store, experiment_id):
    """The window is the evidence's scope; a slow trace an hour later did not cause it."""
    _seed_traces(store, experiment_id, [90_000], at_ms=T0)
    _seed_traces(store, experiment_id, [99_000], at_ms=T0 + 60 * BUCKET_MS)

    ids = collect_exemplar_trace_ids(store, _rule(experiment_id), T0 - BUCKET_MS, T0 + BUCKET_MS)

    assert ids == ["tr-90000"]


def test_a_scoped_rule_only_collects_its_own_slice(store, experiment_id):
    _seed_traces(store, experiment_id, [90_000], status="ERROR")
    _seed_traces(store, experiment_id, [80_000], status="OK")

    ids = collect_exemplar_trace_ids(
        store,
        _rule(experiment_id, dimension_value="ERROR"),
        T0 - BUCKET_MS,
        T0 + BUCKET_MS,
    )

    assert ids == ["tr-90000"]


def test_an_experiment_only_sees_its_own_traces(store, experiment_id):
    other = store.create_experiment("other")
    _seed_traces(store, other, [90_000])
    _seed_traces(store, experiment_id, [80_000])

    ids = collect_exemplar_trace_ids(store, _rule(experiment_id), T0 - BUCKET_MS, T0 + BUCKET_MS)

    assert ids == ["tr-80000"]


def test_a_failing_query_yields_no_exemplars_rather_than_raising(store, experiment_id):
    """Evidence is worth less than the alert.

    This runs on the fire path, so an exception here would stop the instance being
    saved and the notification being sent -- turning a missing nicety into the
    silent failure alerting exists to prevent.
    """
    _seed_traces(store, experiment_id, [90_000])

    with mock.patch(
        "mlflow.alerts.exemplars._collect",
        side_effect=RuntimeError("database is on fire"),
    ) as failing:
        ids = collect_exemplar_trace_ids(
            store, _rule(experiment_id), T0 - BUCKET_MS, T0 + BUCKET_MS
        )

    failing.assert_called_once()
    assert ids == []


def test_a_metric_with_no_matching_source_yields_nothing(store, experiment_id):
    ids = collect_exemplar_trace_ids(
        store,
        _rule(experiment_id, metric_key="not_a_metric"),
        T0 - BUCKET_MS,
        T0 + BUCKET_MS,
    )

    assert ids == []
