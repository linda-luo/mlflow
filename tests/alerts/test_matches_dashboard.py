"""The sharing invariant: a rule must agree with the Overview dashboard.

Alerting is a second consumer of numbers the dashboard already computes. If the
two disagree, someone gets paged for a spike the chart does not show -- which
destroys trust in both.

They cannot agree unconditionally, and pretending otherwise would make this test
a lie. ``query_trace_metrics`` buckets by trace **start** time; rollups bucket by
**completion**, which is what makes a bucket sealable at all. So the two include
different traces exactly when a trace straddles a bucket boundary.

The invariant asserted here is therefore sharper than plain equality: the time
column is the *only* permitted source of difference. When no trace straddles a
boundary the two must match exactly, and any divergence in that case is a real
bug in the filters, the aggregation, or the dimension handling.
"""

import uuid
from pathlib import Path

import pytest

from mlflow.alerts.aggregator import RollupAggregator
from mlflow.alerts.entities import AlertRule, SeriesKey
from mlflow.alerts.rollup_reader import aggregate_buckets
from mlflow.alerts.sql_rollup_reader import SqlRollupReader
from mlflow.entities.trace_metrics import AggregationType, MetricAggregation, MetricViewType
from mlflow.store.tracking.dbmodels.models import SqlTraceInfo
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

pytestmark = pytest.mark.notrackingurimock

BUCKET_MS = 60_000
WINDOW_BUCKETS = 30
PRIME_BUCKET = 340
FIRST_DATA_BUCKET = 360
NOW_BUCKET = 400


@pytest.fixture
def store(tmp_path: Path, db_uri: str) -> SqlAlchemyStore:
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir(exist_ok=True)
    return SqlAlchemyStore(db_uri, artifact_uri.as_uri())


@pytest.fixture
def experiment_id(store: SqlAlchemyStore) -> str:
    experiment_id = store.create_experiment("sharing-invariant")
    # The aggregator writes only what something reads, so the invariant is checked
    # against a series a rule actually subscribes to -- which is also the only kind
    # a user could ever compare against the dashboard.
    store.create_alert_rule(
        AlertRule(
            alert_rule_id="",
            experiment_id=int(experiment_id),
            name="sharing-invariant-subscription",
            metric_key="latency",
            dimension_key="TRACES",
            aggregation="AVG",
            comparator="GT",
            threshold=1_000_000,
            window_seconds=600,
            evaluation_interval_seconds=60,
        )
    )
    return experiment_id


def seed(store, experiment_id, *, end_ms, latency_ms, count):
    with store.ManagedSessionMaker(read_only=False) as session:
        for _ in range(count):
            session.add(
                SqlTraceInfo(
                    request_id=f"tr-{uuid.uuid4().hex}",
                    experiment_id=int(experiment_id),
                    timestamp_ms=end_ms - latency_ms,
                    execution_time_ms=latency_ms,
                    end_time_ms=end_ms,
                    status="OK",
                )
            )
        session.commit()


def rollup_average(store, experiment_id, window_start_ms, window_end_ms):
    reader = SqlRollupReader(store)
    buckets = reader.read_buckets(
        SeriesKey("TRACES", int(experiment_id), "latency", ""),
        window_start_ms,
        window_end_ms,
    )
    return aggregate_buckets(buckets, "AVG", window_start_ms, window_end_ms)


def dashboard_average(store, experiment_id, window_start_ms, window_end_ms):
    """What the Overview chart shows for the same window."""
    pages = store.query_trace_metrics(
        experiment_ids=[str(experiment_id)],
        view_type=MetricViewType.TRACES,
        metric_name="latency",
        aggregations=[MetricAggregation(aggregation_type=AggregationType.AVG)],
        time_interval_seconds=(window_end_ms - window_start_ms) // 1000,
        start_time_ms=window_start_ms,
        end_time_ms=window_end_ms,
    )
    # Despite the ``PagedList[list[MetricDataPoint]]`` annotation, this is a flat
    # PagedList of data points.
    points = list(pages)
    if not points:
        return None
    return sum(p.values["AVG"] for p in points) / len(points)


def build_window(store, experiment_id, latency_ms):
    """Seed a window of traces at a fixed latency and seal it.

    Priming first is required: a fresh install starts its watermark at "now" and
    will not seal buckets that are already in the past.
    """
    aggregator = RollupAggregator(store)
    aggregator.run_once(now_ms=PRIME_BUCKET * BUCKET_MS)
    for i in range(WINDOW_BUCKETS):
        # 30s into the bucket, so a short trace starts and ends inside it.
        seed(
            store,
            experiment_id,
            end_ms=(FIRST_DATA_BUCKET + i) * BUCKET_MS + 30_000,
            latency_ms=latency_ms,
            count=5,
        )
    aggregator.run_once(now_ms=NOW_BUCKET * BUCKET_MS)

    watermark = SqlRollupReader(store).latest_sealed_bucket_ms()
    window_end_ms = watermark + BUCKET_MS
    return window_end_ms - WINDOW_BUCKETS * 2 * BUCKET_MS, window_end_ms


def test_matches_the_dashboard_when_no_trace_straddles_a_bucket(store, experiment_id):
    """Exact equality. Any difference here is a real disagreement.

    A 2s trace starting 30s into its bucket also ends inside it, so start-time and
    completion-time bucketing select identical traces and the only remaining
    variables are the filters and the aggregation math.
    """
    window_start_ms, window_end_ms = build_window(store, experiment_id, latency_ms=2_000)

    ours = rollup_average(store, experiment_id, window_start_ms, window_end_ms)
    theirs = dashboard_average(store, experiment_id, window_start_ms, window_end_ms)

    assert theirs is not None
    assert ours.observed_value == pytest.approx(theirs)
    assert ours.observed_value == pytest.approx(2_000)


def test_sample_counts_also_match_when_no_trace_straddles(store, experiment_id):
    """The denominator has to agree too, not just the ratio.

    An average can match by coincidence while both engines are counting different
    numbers of traces.
    """
    window_start_ms, window_end_ms = build_window(store, experiment_id, latency_ms=2_000)

    ours = rollup_average(store, experiment_id, window_start_ms, window_end_ms)
    pages = store.query_trace_metrics(
        experiment_ids=[str(experiment_id)],
        view_type=MetricViewType.TRACES,
        metric_name="trace_count",
        aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        time_interval_seconds=(window_end_ms - window_start_ms) // 1000,
        start_time_ms=window_start_ms,
        end_time_ms=window_end_ms,
    )
    dashboard_count = sum(p.values["COUNT"] for p in pages)

    assert ours.sample_count == dashboard_count == WINDOW_BUCKETS * 5


def test_a_straddling_trace_is_the_only_thing_that_may_diverge(store, experiment_id):
    """The documented limitation, pinned so it cannot widen silently.

    A 90-minute trace starts 90 buckets before it completes, so the two engines
    place it in windows an hour and a half apart. This asserts the divergence is
    real -- and equally that it comes from the time column rather than from the
    two engines counting different things, since the rollup still sees every
    trace it should.
    """
    window_start_ms, window_end_ms = build_window(store, experiment_id, latency_ms=90 * BUCKET_MS)

    ours = rollup_average(store, experiment_id, window_start_ms, window_end_ms)

    # Completion-time bucketing sees all of them: they finished in this window.
    assert ours.sample_count == WINDOW_BUCKETS * 5
    assert ours.observed_value == pytest.approx(90 * BUCKET_MS)

    # Start-time bucketing does not, because they began before the window opened.
    theirs = dashboard_average(store, experiment_id, window_start_ms, window_end_ms)
    assert theirs is None or theirs != pytest.approx(ours.observed_value)
