from pathlib import Path

import pytest

from mlflow.genai.alerts import histogram as hist
from mlflow.genai.alerts.aggregator import (
    LAG_MS,
    RollupAggregator,
    build_work_units,
    sealable_max_bucket_ms,
)
from mlflow.genai.alerts.entities import BUCKET_MS, AlertRule, SeriesKey
from mlflow.genai.alerts.evaluator import percentile_count_threshold
from mlflow.genai.alerts.rollup_reader import (
    Bucket,
    FakeRollupReader,
    RollupReader,
    aggregate_buckets,
)
from mlflow.genai.alerts.sketch import ALPHA, LOG_SKETCH
from mlflow.genai.alerts.sql_rollup_reader import SqlRollupReader
from mlflow.store.tracking.dbmodels.models import (
    SqlRollupState,
    SqlTraceInfo,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

T = 1_700_000_040_000


@pytest.fixture
def store(tmp_path: Path, db_uri: str) -> SqlAlchemyStore:
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir()
    return SqlAlchemyStore(db_uri, artifact_uri.as_uri())


@pytest.fixture
def experiment_id(store: SqlAlchemyStore) -> int:
    experiment_id = int(store.create_experiment("alerts-reader"))
    # Aggregation is demand-driven: a series nothing subscribes to is never
    # written, so there would be nothing for the reader to read.
    _subscribe_latency(store, experiment_id)
    return experiment_id


def _subscribe_latency(store: SqlAlchemyStore, experiment_id: int):
    """Subscriptions come from rules, so subscribing means creating one."""
    store.create_alert_rule(
        AlertRule(
            alert_rule_id="",
            experiment_id=int(experiment_id),
            name=f"reader-latency-{experiment_id}",
            metric_key="latency",
            dimension_key="TRACES",
            aggregation="PERCENTILE",
            percentile_value=95.0,
            comparator="GT",
            threshold=1.0,
            window_seconds=600,
            evaluation_interval_seconds=60,
        )
    )


def _set_watermark(store: SqlAlchemyStore, watermark_ms: int, source: str | None = None) -> None:
    """Seed the watermark for every work unit, or for one source's units."""
    with store.ManagedSessionMaker(read_only=False) as session:
        for unit in build_work_units(store.db_type):
            if source is not None and unit.source.name != source:
                continue
            source_name, dimension_key, metric_key = unit.key
            row = session.get(SqlRollupState, unit.key)
            if row is None:
                session.add(
                    SqlRollupState(
                        source=source_name,
                        dimension_key=dimension_key,
                        metric_key=metric_key,
                        watermark_ms=watermark_ms,
                    )
                )
            else:
                row.watermark_ms = watermark_ms


def test_a_lagging_family_does_not_hold_back_an_unrelated_rule(store: SqlAlchemyStore):
    """The isolation that keying watermarks by metric exists to provide.

    Assessments land minutes after the traces they score. With a single global
    watermark, a quality rollup running behind froze the evaluation window for
    *every* rule in the deployment -- including latency rules that share no data
    with it. Scoped watermarks make each rule wait only for what it reads.
    """
    _set_watermark(store, T)
    _set_watermark(store, T - 10 * BUCKET_MS, source="assessments")
    reader = SqlRollupReader(store)

    latency = SeriesKey("TRACES", 1, "latency")
    quality = SeriesKey("ASSESSMENTS", 1, "assessment_value", "safety")

    assert reader.latest_sealed_bucket_ms(latency) == T
    assert reader.latest_sealed_bucket_ms(quality) == T - 10 * BUCKET_MS
    # Unscoped stays conservative: a caller with no series in mind gets the
    # deployment-wide answer.
    assert reader.latest_sealed_bucket_ms() == T - 10 * BUCKET_MS


def test_coverage_starts_where_aggregation_started(store: SqlAlchemyStore):
    """A fresh install has no trustworthy history before its first sealed bucket."""
    reader = SqlRollupReader(store)
    latency = SeriesKey("TRACES", 1, "latency")

    # Nothing sealed yet: no window at all is covered, so an empty result cannot be
    # read as a genuine zero.
    assert reader.coverage_start_ms(latency) > T

    RollupAggregator(store).run_once(now_ms=T + BUCKET_MS + LAG_MS)

    first_sealed = sealable_max_bucket_ms(T + BUCKET_MS + LAG_MS)
    assert reader.coverage_start_ms(latency) == first_sealed
    # History before the aggregator ever ran stays untrusted, which is what stops a
    # "traffic dropped" rule firing on a server that just booted.
    assert reader.coverage_start_ms(latency) > T - 10 * BUCKET_MS


def test_a_series_from_an_unsealed_family_reads_as_not_ready(store: SqlAlchemyStore):
    _set_watermark(store, T, source="trace_info")

    reader = SqlRollupReader(store)
    # trace_info has been sealed, so a TRACES/latency rule can evaluate...
    assert reader.latest_sealed_bucket_ms(SeriesKey("TRACES", 1, "latency")) == T
    # ...while a family nothing has sealed yet is simply not readable.
    assert reader.latest_sealed_bucket_ms(SeriesKey("ERROR", 1, "error_count")) == 0


def _seed_traces(
    store: SqlAlchemyStore,
    experiment_id: int,
    bucket_start_ms: int,
    durations_ms: list[int],
    prefix: str,
) -> None:
    """Traces whose *completion* lands in ``bucket_start_ms``, whatever their duration."""
    with store.ManagedSessionMaker(read_only=False) as session:
        for i, duration in enumerate(durations_ms):
            end_time_ms = bucket_start_ms + 100 + i
            session.add(
                SqlTraceInfo(
                    request_id=f"{prefix}-{i}",
                    experiment_id=experiment_id,
                    timestamp_ms=end_time_ms - duration,
                    execution_time_ms=duration,
                    end_time_ms=end_time_ms,
                    status="OK",
                )
            )


def _seal(store: SqlAlchemyStore, through_bucket_ms: int, from_bucket_ms: int) -> None:
    _set_watermark(store, from_bucket_ms - BUCKET_MS)
    RollupAggregator(store).run_once(now_ms=through_bucket_ms + BUCKET_MS + LAG_MS)


def _latency_series(experiment_id: int) -> SeriesKey:
    return SeriesKey("TRACES", experiment_id, "latency", "OK")


def test_sql_reader_is_a_drop_in_for_the_fake(store: SqlAlchemyStore):
    # `RollupReader` is not runtime_checkable, so the substitutability claim is
    # checked against the surface the evaluator actually calls.
    protocol_methods = {"read_buckets", "latest_sealed_bucket_ms"}

    assert protocol_methods <= set(dir(SqlRollupReader))
    assert protocol_methods <= set(dir(FakeRollupReader))
    assert RollupReader in SqlRollupReader.__mro__
    assert isinstance(SqlRollupReader(store).latest_sealed_bucket_ms(), int)


def test_read_buckets_returns_sealed_buckets(store: SqlAlchemyStore, experiment_id: int):
    _seed_traces(store, experiment_id, T, [1_000, 2_000], "a")
    _seed_traces(store, experiment_id, T + BUCKET_MS, [3_000], "b")
    _seal(store, through_bucket_ms=T + BUCKET_MS, from_bucket_ms=T)

    buckets = SqlRollupReader(store).read_buckets(
        _latency_series(experiment_id), T, T + 2 * BUCKET_MS
    )

    assert [b.bucket_start_ms for b in buckets] == [T, T + BUCKET_MS]
    assert [b.count for b in buckets] == [2, 1]
    assert buckets[0].sum == pytest.approx(3_000.0)
    assert buckets[0].boundaries_version == hist.SKETCH_VERSION
    assert all(b.is_gap is False for b in buckets)
    assert all(isinstance(b, Bucket) for b in buckets)


def test_read_buckets_is_half_open(store: SqlAlchemyStore, experiment_id: int):
    for i in range(3):
        _seed_traces(store, experiment_id, T + i * BUCKET_MS, [500], f"t{i}")
    _seal(store, through_bucket_ms=T + 2 * BUCKET_MS, from_bucket_ms=T)

    buckets = SqlRollupReader(store).read_buckets(
        _latency_series(experiment_id), T, T + 2 * BUCKET_MS
    )

    assert [b.bucket_start_ms for b in buckets] == [T, T + BUCKET_MS]


def test_histogram_round_trips(store: SqlAlchemyStore, experiment_id: int):
    durations = [5, 300, 3_000, 45_000]
    _seed_traces(store, experiment_id, T, durations, "h")
    _seal(store, through_bucket_ms=T, from_bucket_ms=T)

    (bucket,) = SqlRollupReader(store).read_buckets(
        _latency_series(experiment_id), T, T + BUCKET_MS
    )

    expected = hist.empty()
    for value in durations:
        hist.observe(expected, value, LOG_SKETCH)
    assert bucket.histogram == hist.to_pairs(expected)
    assert hist.total(hist.from_pairs(bucket.histogram)) == bucket.count


def test_unknown_series_reads_empty(store: SqlAlchemyStore, experiment_id: int):
    _seed_traces(store, experiment_id, T, [100], "u")
    _seal(store, through_bucket_ms=T, from_bucket_ms=T)

    assert (
        SqlRollupReader(store).read_buckets(
            SeriesKey("SPAN_MODEL", experiment_id, "total_cost", "gpt-5"), T, T + BUCKET_MS
        )
        == []
    )


@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [
        # One series answers several questions depending on which column is read,
        # which is why there is no separate request_count metric.
        ("COUNT", 4.0),
        ("SUM", 4_400.0),
        ("AVG", 1_100.0),
    ],
)
def test_aggregate_buckets_folds_sealed_rows(
    store: SqlAlchemyStore, experiment_id: int, aggregation: str, expected: float
):
    _seed_traces(store, experiment_id, T, [1_000, 1_200], "a")
    _seed_traces(store, experiment_id, T + BUCKET_MS, [1_100, 1_100], "b")
    _seal(store, through_bucket_ms=T + BUCKET_MS, from_bucket_ms=T)
    window_end = T + 2 * BUCKET_MS

    buckets = SqlRollupReader(store).read_buckets(_latency_series(experiment_id), T, window_end)
    observation = aggregate_buckets(buckets, aggregation, T, window_end)

    assert observation.observed_value == pytest.approx(expected)
    assert observation.sample_count == 4
    assert observation.window_start_ms == T
    assert observation.window_end_ms == window_end


def test_aggregate_buckets_percentile_from_merged_histograms(
    store: SqlAlchemyStore, experiment_id: int
):
    _seed_traces(store, experiment_id, T, [100] * 19, "fast")
    _seed_traces(store, experiment_id, T + BUCKET_MS, [45 * 60_000], "slow")
    _seal(store, through_bucket_ms=T + BUCKET_MS, from_bucket_ms=T)
    window_end = T + 2 * BUCKET_MS

    buckets = SqlRollupReader(store).read_buckets(_latency_series(experiment_id), T, window_end)
    observation = aggregate_buckets(
        buckets, "PERCENTILE", T, window_end, percentile_value=99, spec=LOG_SKETCH
    )

    # Sketches from two separate sealed buckets merge by index union, and the answer
    # is the value itself within the relative-error guarantee -- not the containing
    # bucket's upper edge, which used to over-report systematically.
    assert observation.observed_value == pytest.approx(45 * 60_000, rel=ALPHA)
    assert observation.sample_count == 20


def test_a_gap_in_the_window_reports_no_data(store: SqlAlchemyStore, experiment_id: int):
    _seed_traces(store, experiment_id, T, [1_000], "a")
    now = T + BUCKET_MS + LAG_MS
    aggregator = RollupAggregator(store, max_backfill_buckets=2, max_gap_buckets=5)
    _set_watermark(store, T - BUCKET_MS)
    aggregator.run_once(now_ms=now)
    _set_watermark(store, T - 100 * BUCKET_MS)
    run = aggregator.run_once(now_ms=now)

    gap_bucket = run.units["trace_info/TRACES/latency"].gap_buckets[-1]
    reader = SqlRollupReader(store)
    buckets = reader.read_buckets(
        _latency_series(experiment_id), gap_bucket, gap_bucket + BUCKET_MS
    )
    observation = aggregate_buckets(buckets, "COUNT", gap_bucket, gap_bucket + BUCKET_MS)

    assert [b.is_gap for b in buckets] == [True]
    # A missing row would have read as a quiet minute; the marker makes the window
    # report no data instead of a silently under-counted one.
    assert observation.observed_value is None
    assert observation.has_data is False


def test_latest_sealed_bucket_is_the_slowest_source(store: SqlAlchemyStore):
    _set_watermark(store, T)
    _set_watermark(store, T - 5 * BUCKET_MS, source="assessments")

    assert SqlRollupReader(store).latest_sealed_bucket_ms() == T - 5 * BUCKET_MS


def test_latest_sealed_bucket_is_zero_until_every_source_has_run(store: SqlAlchemyStore):
    _set_watermark(store, T, source="trace_info")

    assert SqlRollupReader(store).latest_sealed_bucket_ms() == 0


def test_latest_sealed_bucket_tracks_the_aggregator(store: SqlAlchemyStore, experiment_id: int):
    _seed_traces(store, experiment_id, T, [1_000], "a")
    _set_watermark(store, T - BUCKET_MS)
    run = RollupAggregator(store).run_once(now_ms=T + BUCKET_MS + LAG_MS)

    reader = SqlRollupReader(store)

    assert reader.latest_sealed_bucket_ms() == run.sealable_max_ms == T
    # Stream C derives windows from this rather than the clock, so the window it
    # builds is bucket-aligned and fully sealed by construction.
    assert reader.read_buckets(_latency_series(experiment_id), T, T + BUCKET_MS)[0].count == 1


def test_sql_reader_matches_the_fake_for_the_same_buckets(
    store: SqlAlchemyStore, experiment_id: int
):
    durations = [800, 1_600, 32_000]
    _seed_traces(store, experiment_id, T, durations, "a")
    _seal(store, through_bucket_ms=T, from_bucket_ms=T)
    window_end = T + BUCKET_MS

    series = _latency_series(experiment_id)
    sql_buckets = SqlRollupReader(store).read_buckets(series, T, window_end)

    fake = FakeRollupReader()
    histogram = hist.empty()
    for value in durations:
        hist.observe(histogram, value, LOG_SKETCH)
    fake.seed(
        series,
        Bucket(
            bucket_start_ms=T,
            count=len(durations),
            sum=float(sum(durations)),
            histogram=hist.to_pairs(histogram),
            boundaries_version=hist.SKETCH_VERSION,
        ),
    )

    assert sql_buckets == fake.read_buckets(series, T, window_end)
    for aggregation in ("COUNT", "SUM", "AVG"):
        assert aggregate_buckets(sql_buckets, aggregation, T, window_end) == aggregate_buckets(
            fake.read_buckets(series, T, window_end), aggregation, T, window_end
        )


###############################################################################
# Tier type parity
###############################################################################


class _DecimalHourRow:
    """One row as the *continuous aggregate* returns it, not as the raw table does.

    The view aggregates with ``sum(count)``, and Postgres types ``sum(bigint)`` as
    ``numeric``, which psycopg hands back as ``decimal.Decimal``. The raw column is
    BIGINT and arrives as ``int``. Nothing in the SQLite-backed tests can see that
    divergence, because SQLite has no continuous aggregate to read.
    """

    def __init__(self, bucket_start_ms: int, count):
        self.bucket_start_ms = bucket_start_ms
        self.count = count
        self.sum = 12.5
        self.histogram = [1, 2]
        self.boundaries_version = 1
        self.is_gap = False


class _StubResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _StubSession:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, *args, **kwargs):
        return _StubResult(self._rows)


def test_an_hourly_bucket_counts_in_int_not_decimal(store: SqlAlchemyStore):
    """A stitched read must not leak ``Decimal`` into ``Bucket.count``.

    ``percentile_count_threshold`` computes ``(1 - p/100) * sample_count``, and
    ``Decimal`` refuses to multiply by ``float``. So an un-coerced hourly count
    raised ``TypeError`` and every PERCENTILE rule whose window spanned a whole
    hour failed to evaluate on Postgres -- silently, since the evaluator logs the
    failure and moves on. ``Bucket.count`` is annotated ``int``; this holds the
    hourly path to it.
    """
    from decimal import Decimal

    from mlflow.genai.alerts.tiers import CoverSpan

    reader = SqlRollupReader(store)
    session = _StubSession([_DecimalHourRow(T, Decimal("4200"))])
    span = CoverSpan(start_ms=T, end_ms=T + 3_600_000, width_ms=3_600_000)

    (bucket,) = reader._read_hours(session, series_id=1, span=span)

    assert isinstance(bucket.count, int)
    assert not isinstance(bucket.count, Decimal)
    assert bucket.count == 4200


def test_a_percentile_threshold_survives_an_hourly_count(store: SqlAlchemyStore):
    """The crash site itself, reached with a count that came off the hourly tier."""
    from decimal import Decimal

    from mlflow.genai.alerts.tiers import CoverSpan

    reader = SqlRollupReader(store)
    session = _StubSession([_DecimalHourRow(T, Decimal("4200"))])
    span = CoverSpan(start_ms=T, end_ms=T + 3_600_000, width_ms=3_600_000)
    (bucket,) = reader._read_hours(session, series_id=1, span=span)

    rule = AlertRule(
        alert_rule_id="p95",
        experiment_id=1,
        name="p95 latency over four hours",
        metric_key="latency",
        dimension_key="TRACES",
        aggregation="PERCENTILE",
        percentile_value=95,
        comparator="GT",
        threshold=45 * 60_000,
        window_seconds=14_400,
        evaluation_interval_seconds=300,
    )
    assert percentile_count_threshold(rule, bucket.count) == pytest.approx(210.0)
