import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from mlflow.genai.alerts import histogram as hist
from mlflow.genai.alerts.aggregator import (
    LAG_MS,
    SOURCE_NAMES,
    AggregationRun,
    RollupAggregator,
    _build_sources,
    build_work_units,
    floor_bucket,
    sealable_max_bucket_ms,
)
from mlflow.genai.alerts.entities import (
    BUCKET_MS,
    MAX_WINDOW_SECONDS,
    METRIC_CATALOGUE,
    AlertRule,
    SeriesKey,
)
from mlflow.genai.alerts.sketch import spec_for
from mlflow.genai.alerts.timescale import setup_timescale
from mlflow.store.tracking.dbmodels.models import (
    SqlAssessments,
    SqlMetricRollup,
    SqlMetricSeries,
    SqlRollupState,
    SqlSpan,
    SqlSpanError,
    SqlSpanMetrics,
    SqlTraceInfo,
    SqlTraceMetrics,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils.time import get_current_time_millis

# A bucket-aligned instant in the past, so `T + offset` lands where the arithmetic
# in the assertions says it does.
T = 1_700_000_040_000
NOW = T + 2 * BUCKET_MS + LAG_MS


@pytest.fixture
def store(tmp_path: Path, db_uri: str) -> SqlAlchemyStore:
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir()
    return SqlAlchemyStore(db_uri, artifact_uri.as_uri())


@pytest.fixture
def experiment_id(store: SqlAlchemyStore) -> int:
    experiment_id = int(store.create_experiment("alerts-aggregator"))
    _subscribe_all(store, experiment_id)
    return experiment_id


def _session(store: SqlAlchemyStore):
    return store.ManagedSessionMaker(read_only=False)


def _subscribe_all(store: SqlAlchemyStore, experiment_id: int):
    """Create one rule per catalogue-expressible family.

    Subscriptions are derived from `alert_rules`, so "subscribe" means "create a
    rule". Most tests here are about the *mechanics* of sealing rather than about
    which families get filtered, so they subscribe broadly and the filtering gets
    its own tests.

    Only catalogue-legal triples can exist as rules, which is now every family the
    aggregator writes: the token and cost components used to have work units but no
    catalogue entry, so they were scanned every minute and discarded.
    """
    for metric_key, spec in METRIC_CATALOGUE.items():
        for dimension_key in sorted(spec.dimension_keys):
            aggregation = "PERCENTILE" if "PERCENTILE" in spec.aggregations else "COUNT"
            if aggregation not in spec.aggregations:
                aggregation = min(spec.aggregations)
            store.create_alert_rule(
                AlertRule(
                    alert_rule_id="",
                    experiment_id=int(experiment_id),
                    name=f"sub-{experiment_id}-{dimension_key}-{metric_key}",
                    metric_key=metric_key,
                    dimension_key=dimension_key,
                    aggregation=aggregation,
                    percentile_value=95.0 if aggregation == "PERCENTILE" else None,
                    comparator="GT",
                    threshold=1.0,
                    window_seconds=600,
                    evaluation_interval_seconds=60,
                )
            )


def _set_watermark(store: SqlAlchemyStore, watermark_ms: int, source: str | None = None) -> None:
    """Seed the watermark for every work unit, or for one source's units."""
    with _session(store) as session:
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


def _watermarks(store: SqlAlchemyStore) -> dict[tuple[str, str, str], int]:
    with store.ManagedSessionMaker() as session:
        return {
            (r.source, r.dimension_key, r.metric_key): r.watermark_ms
            for r in session.query(SqlRollupState).all()
        }


def _seed_trace(
    store: SqlAlchemyStore,
    experiment_id: int,
    trace_id: str,
    *,
    start_ms: int,
    duration_ms: int,
    status: str = "OK",
) -> None:
    with _session(store) as session:
        session.add(
            SqlTraceInfo(
                request_id=trace_id,
                experiment_id=experiment_id,
                timestamp_ms=start_ms,
                execution_time_ms=duration_ms,
                end_time_ms=start_ms + duration_ms,
                status=status,
            )
        )


@dataclass(frozen=True)
class _Row:
    count: int
    sum: float | None
    histogram: list[int] | None
    boundaries_version: int | None
    is_gap: bool


def _rollups(store: SqlAlchemyStore) -> dict[tuple[SeriesKey, int], _Row]:
    """Every rollup row, detached from the session and keyed by (series, bucket)."""
    with store.ManagedSessionMaker() as session:
        rows = (
            session
            .query(SqlMetricSeries, SqlMetricRollup)
            .join(SqlMetricRollup, SqlMetricRollup.series_id == SqlMetricSeries.series_id)
            .all()
        )
        return {
            (
                SeriesKey(
                    dimension_key=series.dimension_key,
                    experiment_id=series.experiment_id,
                    metric_key=series.metric_key,
                    dimension_value=series.dimension_value,
                ),
                rollup.bucket_start_ms,
            ): _Row(
                count=rollup.count,
                sum=rollup.sum,
                histogram=list(rollup.histogram) if rollup.histogram is not None else None,
                boundaries_version=rollup.boundaries_version,
                is_gap=rollup.is_gap,
            )
            for series, rollup in rows
        }


def _expected_histogram(*values: float, metric_key: str = "latency") -> list[int]:
    """The stored form: interleaved (bucket index, count) pairs, index-sorted."""
    spec = spec_for(metric_key)
    h = hist.empty()
    for value in values:
        hist.observe(h, value, spec)
    return hist.to_pairs(h)


@pytest.mark.parametrize(
    ("now_ms", "expected"),
    [
        # Standing inside bucket T+2 at its very start: T+1 is the last one that
        # fully ended, so the extra -BUCKET_MS is what makes this T and not T+1.
        (T + 2 * BUCKET_MS + LAG_MS, T + BUCKET_MS),
        (T + 2 * BUCKET_MS + LAG_MS + BUCKET_MS - 1, T + BUCKET_MS),
        (T + 3 * BUCKET_MS + LAG_MS, T + 2 * BUCKET_MS),
    ],
)
def test_sealable_max_takes_the_last_fully_ended_bucket(now_ms: int, expected: int):
    assert sealable_max_bucket_ms(now_ms) == expected


def test_source_names_match_the_scans(store: SqlAlchemyStore):
    assert tuple(source.name for source in _build_sources(store.db_type)) == SOURCE_NAMES


def test_sealing_a_bucket_produces_exact_rows(store: SqlAlchemyStore, experiment_id: int):
    _seed_trace(store, experiment_id, "t-ok-1", start_ms=T + 1_000, duration_ms=1_500)
    _seed_trace(store, experiment_id, "t-ok-2", start_ms=T + 2_000, duration_ms=40_000)
    _seed_trace(store, experiment_id, "t-err", start_ms=T + 3_000, duration_ms=800, status="ERROR")
    _set_watermark(store, T - BUCKET_MS)

    run = RollupAggregator(store).run_once(now_ms=T + BUCKET_MS + LAG_MS)

    assert run.units["trace_info/TRACES/latency"].sealed_buckets == [T]
    rollups = _rollups(store)
    ok = rollups[(SeriesKey("TRACES", experiment_id, "latency", "OK"), T)]
    assert ok.count == 2
    assert ok.sum == pytest.approx(41_500.0)
    assert ok.histogram == _expected_histogram(1_500, 40_000)
    assert ok.boundaries_version == hist.SKETCH_VERSION
    assert ok.is_gap is False

    err = rollups[(SeriesKey("TRACES", experiment_id, "latency", "ERROR"), T)]
    assert err.count == 1
    assert err.sum == pytest.approx(800.0)

    # The `include_total` series is what makes "number of requests" -- latency+COUNT
    # with no dimension value -- answerable off the same scan.
    total = rollups[(SeriesKey("TRACES", experiment_id, "latency", ""), T)]
    assert total.count == 3
    assert total.sum == pytest.approx(42_300.0)
    assert total.histogram == _expected_histogram(1_500, 40_000, 800)


def test_buckets_by_completion_not_start(store: SqlAlchemyStore, experiment_id: int):
    # Starts in bucket T, finishes in bucket T+1. Start-time bucketing would put it
    # in T and hold that bucket open until the trace finished.
    _seed_trace(store, experiment_id, "t-slow", start_ms=T + 30_000, duration_ms=45_000)
    _set_watermark(store, T - BUCKET_MS)

    RollupAggregator(store).run_once(now_ms=T + 3 * BUCKET_MS + LAG_MS)

    rollups = _rollups(store)
    key = SeriesKey("TRACES", experiment_id, "latency", "OK")
    assert (key, T) not in rollups
    assert rollups[(key, T + BUCKET_MS)].count == 1


def test_half_open_interval_assigns_each_row_to_one_bucket(
    store: SqlAlchemyStore, experiment_id: int
):
    _seed_trace(store, experiment_id, "t-edge-lo", start_ms=T - 10, duration_ms=10)
    _seed_trace(store, experiment_id, "t-edge-hi", start_ms=T + BUCKET_MS - 10, duration_ms=10)
    _set_watermark(store, T - BUCKET_MS)

    RollupAggregator(store).run_once(now_ms=T + 3 * BUCKET_MS + LAG_MS)

    rollups = _rollups(store)
    key = SeriesKey("TRACES", experiment_id, "latency", "OK")
    assert rollups[(key, T)].count == 1
    assert rollups[(key, T + BUCKET_MS)].count == 1


def test_watermark_advances_one_bucket_at_a_time(store: SqlAlchemyStore, experiment_id: int):
    for i in range(3):
        _seed_trace(
            store, experiment_id, f"t-{i}", start_ms=T + i * BUCKET_MS + 100, duration_ms=200
        )
    _set_watermark(store, T - BUCKET_MS)

    seen: list[int] = []
    aggregator = RollupAggregator(store)
    original = aggregator._write_watermark

    def _record(session, unit, watermark_ms):
        if unit.name == "trace_info/TRACES/latency":
            seen.append(watermark_ms)
        return original(session, unit, watermark_ms)

    aggregator._write_watermark = _record
    run = aggregator.run_once(now_ms=T + 3 * BUCKET_MS + LAG_MS)

    assert seen == [T, T + BUCKET_MS, T + 2 * BUCKET_MS]
    assert run.units["trace_info/TRACES/latency"].sealed_buckets == [
        T,
        T + BUCKET_MS,
        T + 2 * BUCKET_MS,
    ]
    assert _watermarks(store)[("trace_info", "TRACES", "latency")] == T + 2 * BUCKET_MS


def test_resealing_a_bucket_is_idempotent(store: SqlAlchemyStore, experiment_id: int):
    _seed_trace(store, experiment_id, "t-1", start_ms=T + 100, duration_ms=250)
    _set_watermark(store, T - BUCKET_MS)
    now = T + BUCKET_MS + LAG_MS

    RollupAggregator(store).run_once(now_ms=now)
    _set_watermark(store, T - BUCKET_MS)
    RollupAggregator(store).run_once(now_ms=now)

    rollups = _rollups(store)
    key = (SeriesKey("TRACES", experiment_id, "latency", "OK"), T)
    assert rollups[key].count == 1
    # Named rather than counted: one OK trace produces its status series, the
    # unscoped latency total, and the unsliced error-rate series. Re-sealing must
    # reuse all three rather than create a second set.
    with store.ManagedSessionMaker() as session:
        assert {
            (s.metric_key, s.dimension_value) for s in session.query(SqlMetricSeries).all()
        } == {("latency", "OK"), ("latency", ""), ("error_rate", "")}


def test_gaps_are_marked_when_the_job_was_down(store: SqlAlchemyStore, experiment_id: int):
    _seed_trace(store, experiment_id, "t-recent", start_ms=T + 100, duration_ms=300)
    now = T + BUCKET_MS + LAG_MS
    aggregator = RollupAggregator(store, max_backfill_buckets=5, max_gap_buckets=10)
    # Run once so the series exists, then rewind the watermark: a marker can only be
    # written against a series some scan has already created.
    _set_watermark(store, T - BUCKET_MS)
    aggregator.run_once(now_ms=now)
    _set_watermark(store, T - 200 * BUCKET_MS)

    run = aggregator.run_once(now_ms=now)

    trace_source = run.units["trace_info/TRACES/latency"]
    resume = T - 4 * BUCKET_MS
    assert trace_source.gap_buckets == [resume - (10 - i) * BUCKET_MS for i in range(10)]
    assert trace_source.sealed_buckets == [resume + i * BUCKET_MS for i in range(5)]
    assert trace_source.watermark_ms == T

    rollups = _rollups(store)
    key = SeriesKey("TRACES", experiment_id, "latency", "OK")
    assert all(rollups[(key, b)].is_gap for b in trace_source.gap_buckets)
    assert all(rollups[(key, b)].count == 0 for b in trace_source.gap_buckets)
    assert rollups[(key, T)].is_gap is False


def test_gap_marking_is_capped(store: SqlAlchemyStore, experiment_id: int):
    _seed_trace(store, experiment_id, "t-recent", start_ms=T + 100, duration_ms=300)
    now = T + BUCKET_MS + LAG_MS
    aggregator = RollupAggregator(store, max_backfill_buckets=2, max_gap_buckets=3)
    _set_watermark(store, T - BUCKET_MS)
    aggregator.run_once(now_ms=now)
    # Days behind: without the cap this would write one row per skipped bucket per
    # series, and none of them could overlap a window anyway.
    _set_watermark(store, T - 5_000 * BUCKET_MS)

    run = aggregator.run_once(now_ms=now)

    assert len(run.units["trace_info/TRACES/latency"].gap_buckets) == 3


def test_a_quiet_minute_writes_no_rows(store: SqlAlchemyStore, experiment_id: int):
    _seed_trace(store, experiment_id, "t-1", start_ms=T + 100, duration_ms=250)
    _set_watermark(store, T - BUCKET_MS)

    RollupAggregator(store).run_once(now_ms=T + 3 * BUCKET_MS + LAG_MS)

    rollups = _rollups(store)
    key = SeriesKey("TRACES", experiment_id, "latency", "OK")
    assert (key, T) in rollups
    assert (key, T + BUCKET_MS) not in rollups


def test_a_fresh_install_does_not_backfill_from_the_epoch(store: SqlAlchemyStore):
    run = RollupAggregator(store).run_once(now_ms=NOW)

    expected = sealable_max_bucket_ms(NOW)
    for unit in build_work_units(store.db_type):
        assert run.units[unit.name].sealed_buckets == [expected]
        assert run.units[unit.name].gap_buckets == []


def test_trace_metrics_scan_groups_by_key(store: SqlAlchemyStore, experiment_id: int):
    _seed_trace(store, experiment_id, "t-1", start_ms=T + 100, duration_ms=200)
    _seed_trace(store, experiment_id, "t-2", start_ms=T + 200, duration_ms=200)
    with _session(store) as session:
        for trace_id, total, inp in (("t-1", 300.0, 100.0), ("t-2", 500.0, 200.0)):
            session.add_all([
                SqlTraceMetrics(
                    request_id=trace_id,
                    key="total_tokens",
                    value=total,
                    experiment_id=experiment_id,
                    timestamp_ms=T + 400,
                ),
                SqlTraceMetrics(
                    request_id=trace_id,
                    key="input_tokens",
                    value=inp,
                    experiment_id=experiment_id,
                    timestamp_ms=T + 400,
                ),
            ])
    _set_watermark(store, T - BUCKET_MS)

    RollupAggregator(store).run_once(now_ms=T + BUCKET_MS + LAG_MS)

    rollups = _rollups(store)
    total = rollups[(SeriesKey("TRACES", experiment_id, "total_tokens", ""), T)]
    assert total.count == 2
    assert total.sum == pytest.approx(800.0)
    # Tokens have a sketch spec, so a percentile over them is answerable.
    assert total.histogram == _expected_histogram(300.0, 500.0, metric_key="total_tokens")
    assert total.boundaries_version == hist.SKETCH_VERSION
    # Interleaved (index, count) pairs, so the counts are the odd positions.
    assert sum(total.histogram[1::2]) == total.count
    # `input_tokens` comes off the same scan, keyed separately. It is its own work
    # unit with its own watermark, so grouping by key is what keeps the two from
    # being summed together.
    inputs = rollups[(SeriesKey("TRACES", experiment_id, "input_tokens", ""), T)]
    assert inputs.count == 2
    assert inputs.sum == pytest.approx(300.0)


def _seed_span(
    store: SqlAlchemyStore,
    experiment_id: int,
    trace_id: str,
    span_id: str,
    *,
    name: str,
    span_type: str,
    end_ms: int,
    duration_ms: int,
    status: str = "OK",
) -> None:
    with _session(store) as session:
        session.add(
            SqlSpan(
                trace_id=trace_id,
                experiment_id=experiment_id,
                span_id=span_id,
                name=name,
                type=span_type,
                status=status,
                start_time_unix_nano=(end_ms - duration_ms) * 1_000_000,
                end_time_unix_nano=end_ms * 1_000_000,
                content="{}",
                dimension_attributes={},
            )
        )


def test_span_name_series_are_written_only_for_tool_spans(
    store: SqlAlchemyStore, experiment_id: int
):
    _seed_trace(store, experiment_id, "t-1", start_ms=T, duration_ms=1_000)
    _seed_span(
        store,
        experiment_id,
        "t-1",
        "s-tool",
        name="search_docs",
        span_type="TOOL",
        end_ms=T + 5_000,
        duration_ms=400,
    )
    _seed_span(
        store,
        experiment_id,
        "t-1",
        "s-llm",
        name="openai.chat",
        span_type="LLM",
        end_ms=T + 6_000,
        duration_ms=900,
    )
    _set_watermark(store, T - BUCKET_MS)

    RollupAggregator(store).run_once(now_ms=T + BUCKET_MS + LAG_MS)

    rollups = _rollups(store)
    assert rollups[(SeriesKey("SPAN_TYPE", experiment_id, "latency", "TOOL"), T)].count == 1
    assert rollups[(SeriesKey("SPAN_TYPE", experiment_id, "latency", "LLM"), T)].count == 1
    tool = rollups[(SeriesKey("SPAN_NAME", experiment_id, "latency", "search_docs"), T)]
    assert tool.count == 1
    assert tool.sum == pytest.approx(400.0)
    assert tool.histogram == _expected_histogram(400)
    # Cardinality control lives in the aggregator, not the schema: without this the
    # SPAN_NAME dimension would grow one member per autologged span name.
    assert (SeriesKey("SPAN_NAME", experiment_id, "latency", "openai.chat"), T) not in rollups


def test_a_source_can_be_sealed_independently_of_the_others(
    store: SqlAlchemyStore, experiment_id: int
):
    """Per-source workers are only safe because the sources are independent.

    Separate watermark rows, disjoint tables, disjoint `series_pairs` -- so sealing
    one must neither advance another's watermark nor write another's series.
    """
    _seed_trace(store, experiment_id, "t-1", start_ms=T, duration_ms=1_000)
    _seed_span(
        store,
        experiment_id,
        "t-1",
        "s-tool",
        name="search_docs",
        span_type="TOOL",
        end_ms=T + 5_000,
        duration_ms=400,
    )
    _set_watermark(store, T - BUCKET_MS)

    run = RollupAggregator(store, only_units=["trace_info/TRACES/latency"]).run_once(
        now_ms=T + BUCKET_MS + LAG_MS
    )

    assert set(run.units) == {"trace_info/TRACES/latency"}
    written = {key.dimension_key for (key, _) in _rollups(store)}
    assert written == {"TRACES"}

    watermarks = _watermarks(store)
    assert watermarks[("trace_info", "TRACES", "latency")] == T
    # Untouched units must not be dragged forward, or their buckets would be
    # skipped without ever being scanned. `spans` shares nothing with `trace_info`,
    # and SPAN_NAME/latency is a *different unit of the same source* -- the case
    # that a per-source watermark could not have represented.
    assert watermarks[("spans", "SPAN_TYPE", "latency")] == T - BUCKET_MS
    assert watermarks[("spans", "SPAN_NAME", "latency")] == T - BUCKET_MS


def test_units_of_one_source_advance_independently(store: SqlAlchemyStore, experiment_id: int):
    """The whole point of keying by metric rather than by source.

    `spans` owns two families; sealing one must leave the other exactly where it was.
    """
    _seed_trace(store, experiment_id, "t-1", start_ms=T, duration_ms=1_000)
    _seed_span(
        store,
        experiment_id,
        "t-1",
        "s-tool",
        name="search_docs",
        span_type="TOOL",
        end_ms=T + 5_000,
        duration_ms=400,
    )
    _set_watermark(store, T - BUCKET_MS)

    RollupAggregator(store, only_units=["spans/SPAN_NAME/latency"]).run_once(
        now_ms=T + BUCKET_MS + LAG_MS
    )

    watermarks = _watermarks(store)
    assert watermarks[("spans", "SPAN_NAME", "latency")] == T
    assert watermarks[("spans", "SPAN_TYPE", "latency")] == T - BUCKET_MS
    assert {key.dimension_key for (key, _) in _rollups(store)} == {"SPAN_NAME"}


def test_an_unknown_source_or_unit_is_rejected_rather_than_silently_doing_nothing(
    store: SqlAlchemyStore,
):
    with pytest.raises(ValueError, match="Unknown rollup source"):
        RollupAggregator(store, only_sources=["not_a_source"])
    with pytest.raises(ValueError, match="Unknown rollup unit"):
        RollupAggregator(store, only_units=["spans/NOPE/latency"])


def test_timing_is_recorded_per_source(store: SqlAlchemyStore, experiment_id: int):
    """Scan and upsert are timed separately because they scale differently.

    The scan grows with traffic and is what sharding addresses; the upsert grows
    with series count and is not helped by sharding at all.
    """
    _seed_trace(store, experiment_id, "t-1", start_ms=T, duration_ms=1_000)
    _set_watermark(store, T - BUCKET_MS)

    run = RollupAggregator(store).run_once(now_ms=T + BUCKET_MS + LAG_MS)

    trace_info = run.units["trace_info/TRACES/latency"]
    assert trace_info.scan_seconds > 0
    assert trace_info.upsert_seconds > 0
    assert run.total_seconds >= trace_info.total_seconds
    assert "trace_info" in run.timing_summary()


def test_dimension_values_match_the_series_the_aggregator_writes(
    store: SqlAlchemyStore, experiment_id: int
):
    """The dropdown and the aggregator must agree on which values exist.

    `_Grouping.only_when` and `_Grouping.only_when_sql` are the same predicate
    written twice -- once in Python for the scan, which fans one result set into
    several groupings, and once in SQL for the editor, which can push it down.
    Nothing but this test stops them drifting apart, and drift here means the
    editor offers a slice that would never produce a series: a rule that looks
    healthy and can never fire.
    """
    # The editor only offers values seen in the last 24h, so this one case has to
    # be seeded near the wall clock rather than at the suite's fixed historical T.
    recent = floor_bucket(get_current_time_millis()) - 2 * BUCKET_MS
    _seed_trace(store, experiment_id, "t-1", start_ms=recent, duration_ms=1_000)
    for span_id, name, span_type in (
        ("s-tool", "search_docs", "TOOL"),
        ("s-tool2", "fetch_page", "TOOL"),
        ("s-llm", "openai.chat", "LLM"),
    ):
        _seed_span(
            store,
            experiment_id,
            "t-1",
            span_id,
            name=name,
            span_type=span_type,
            end_ms=recent + 5_000,
            duration_ms=400,
        )
    _set_watermark(store, recent - BUCKET_MS)
    RollupAggregator(store).run_once(now_ms=recent + BUCKET_MS + LAG_MS)

    written = {
        key.dimension_value
        for (key, _) in _rollups(store)
        if key.dimension_key == "SPAN_NAME" and key.metric_key == "latency"
    }
    offered = set(store.list_alert_dimension_values(experiment_id, "latency", "SPAN_NAME"))

    # The empty-string series is the grouping's `include_total` -- every tool span
    # combined. It is deliberately not in the dropdown: the form expresses it as the
    # "Any tool" choice rather than as a value you could pick by name.
    assert "" in written
    assert offered == written - {""} == {"search_docs", "fetch_page"}
    # The LLM span is the case that distinguishes the two predicates: it produces a
    # SPAN_TYPE series but must not produce a SPAN_NAME one.
    assert "openai.chat" not in offered


def test_span_errors_one_scan_two_groupings(store: SqlAlchemyStore, experiment_id: int):
    _seed_trace(store, experiment_id, "t-1", start_ms=T, duration_ms=1_000)
    rows = [
        ("s-1", "search_docs", "TimeoutError", True),
        ("s-2", "search_docs", "TimeoutError", True),
        ("s-3", "fetch_page", "TimeoutError", True),
        ("s-4", "fetch_page", "RateLimitError", True),
        # A propagating ancestor: counted by neither grouping.
        ("s-5", "agent_run", "TimeoutError", False),
    ]
    with _session(store) as session:
        for span_id, span_name, exception_type, is_origin in rows:
            session.add(
                SqlSpan(
                    trace_id="t-1",
                    experiment_id=experiment_id,
                    span_id=span_id,
                    name=span_name,
                    type="TOOL",
                    status="ERROR",
                    start_time_unix_nano=T * 1_000_000,
                    end_time_unix_nano=(T + 1_000) * 1_000_000,
                    content="{}",
                )
            )
            session.add(
                SqlSpanError(
                    trace_id="t-1",
                    span_id=span_id,
                    exception_type=exception_type,
                    parent_span_id=None,
                    is_origin=is_origin,
                    span_name=span_name,
                    span_type="TOOL",
                    experiment_id=experiment_id,
                    timestamp_ms=T + 1_000,
                )
            )
    _set_watermark(store, T - BUCKET_MS)

    RollupAggregator(store).run_once(now_ms=T + BUCKET_MS + LAG_MS)

    rollups = _rollups(store)
    by_type = rollups[(SeriesKey("ERROR", experiment_id, "error_count", "TimeoutError"), T)]
    assert by_type.count == 3
    assert (
        rollups[(SeriesKey("ERROR", experiment_id, "error_count", "RateLimitError"), T)].count == 1
    )
    by_name = rollups[(SeriesKey("SPAN_NAME", experiment_id, "error_count", "search_docs"), T)]
    assert by_name.count == 2
    assert (
        rollups[(SeriesKey("SPAN_NAME", experiment_id, "error_count", "fetch_page"), T)].count == 2
    )
    assert (SeriesKey("ERROR", experiment_id, "error_count", "TimeoutError"), T) in rollups
    # error_count has no value column, so no sum and no histogram.
    assert by_type.sum is None
    assert by_type.histogram is None


def test_span_metrics_scan_groups_by_model_and_span_type(
    store: SqlAlchemyStore, experiment_id: int
):
    _seed_trace(store, experiment_id, "t-1", start_ms=T, duration_ms=1_000)
    with _session(store) as session:
        for span_id, model, cost in (("s-1", "gpt-5", 0.25), ("s-2", "gpt-5", 0.75)):
            session.add(
                SqlSpan(
                    trace_id="t-1",
                    experiment_id=experiment_id,
                    span_id=span_id,
                    name="chat",
                    type="LLM",
                    status="OK",
                    start_time_unix_nano=T * 1_000_000,
                    end_time_unix_nano=(T + 1_000) * 1_000_000,
                    content="{}",
                    dimension_attributes={"mlflow.llm.model": model},
                )
            )
            session.add(
                SqlSpanMetrics(
                    trace_id="t-1",
                    span_id=span_id,
                    key="total_cost",
                    value=cost,
                    experiment_id=experiment_id,
                    timestamp_ms=T + 1_000,
                )
            )
    _set_watermark(store, T - BUCKET_MS)

    RollupAggregator(store).run_once(now_ms=T + BUCKET_MS + LAG_MS)

    rollups = _rollups(store)
    by_model = rollups[(SeriesKey("SPAN_MODEL", experiment_id, "total_cost", "gpt-5"), T)]
    assert by_model.count == 2
    assert by_model.sum == pytest.approx(1.0)
    by_type = rollups[(SeriesKey("SPAN_TYPE", experiment_id, "total_cost", "LLM"), T)]
    assert by_type.count == 2
    assert by_type.sum == pytest.approx(1.0)


def test_assessments_scan_groups_by_judge_and_skips_invalid(
    store: SqlAlchemyStore, experiment_id: int
):
    _seed_trace(store, experiment_id, "t-1", start_ms=T, duration_ms=1_000)
    with _session(store) as session:
        for i, (value, valid) in enumerate([(True, True), (False, True), (True, False)]):
            session.add(
                SqlAssessments(
                    assessment_id=f"a-{i}",
                    trace_id="t-1",
                    name="safety",
                    assessment_type="feedback",
                    value=json.dumps(value),
                    created_timestamp=T + 1_000,
                    last_updated_timestamp=T + 1_000,
                    source_type="LLM_JUDGE",
                    valid=valid,
                    experiment_id=experiment_id,
                )
            )
    _set_watermark(store, T - BUCKET_MS)

    RollupAggregator(store).run_once(now_ms=T + BUCKET_MS + LAG_MS)

    rollups = _rollups(store)
    safety = rollups[(SeriesKey("ASSESSMENTS", experiment_id, "assessment_value", "safety"), T)]
    assert safety.count == 2
    # AVG = sum/count = 0.5, the pass rate. The overridden verdict is excluded, which
    # is what keeps the number equal to the Overview dashboard's.
    assert safety.sum == pytest.approx(1.0)


def test_each_source_keeps_its_own_watermark(store: SqlAlchemyStore, experiment_id: int):
    _set_watermark(store, T - BUCKET_MS, source="trace_info")
    _set_watermark(store, T - 3 * BUCKET_MS, source="assessments")
    for name in SOURCE_NAMES:
        if name not in ("trace_info", "assessments"):
            _set_watermark(store, T - BUCKET_MS, source=name)

    run = RollupAggregator(store).run_once(now_ms=T + BUCKET_MS + LAG_MS)

    assert run.units["trace_info/TRACES/latency"].sealed_buckets == [T]
    assert run.units["assessments/ASSESSMENTS/assessment_value"].sealed_buckets == [
        T - 2 * BUCKET_MS,
        T - BUCKET_MS,
        T,
    ]


def test_scans_cover_every_experiment_in_one_pass(store: SqlAlchemyStore):
    first = int(store.create_experiment("exp-a"))
    second = int(store.create_experiment("exp-b"))
    _subscribe_all(store, first)
    _subscribe_all(store, second)
    _seed_trace(store, first, "t-a", start_ms=T + 100, duration_ms=200)
    _seed_trace(store, second, "t-b", start_ms=T + 100, duration_ms=200)
    _set_watermark(store, T - BUCKET_MS)

    RollupAggregator(store).run_once(now_ms=T + BUCKET_MS + LAG_MS)

    rollups = _rollups(store)
    assert rollups[(SeriesKey("TRACES", first, "latency", "OK"), T)].count == 1
    assert rollups[(SeriesKey("TRACES", second, "latency", "OK"), T)].count == 1


def test_run_returns_an_aggregation_run(store: SqlAlchemyStore):
    units = build_work_units(store.db_type)
    run = RollupAggregator(store).run_once(now_ms=NOW)

    assert isinstance(run, AggregationRun)
    assert run.sealable_max_ms == sealable_max_bucket_ms(NOW)
    assert set(run.units) == {unit.name for unit in units}
    assert run.sealed_bucket_count == len(units)
    assert set(run.by_source()) == set(SOURCE_NAMES)


def test_every_series_pair_is_owned_by_exactly_one_unit(store: SqlAlchemyStore):
    """Units must partition the series space, not overlap it.

    Two units writing the same series would race on the same rollup row, which is
    what makes them safe to run concurrently in the first place.
    """
    units = build_work_units(store.db_type)
    owners: dict[tuple[str, str], list[str]] = {}
    for unit in units:
        owners.setdefault((unit.grouping.dimension_key, unit.metric_key), []).append(unit.name)

    assert all(len(names) == 1 for names in owners.values()), owners
    expected = {pair for source in _build_sources(store.db_type) for pair in source.series_pairs}
    assert set(owners) == expected


def test_error_rate_stores_attempts_and_failures_not_a_rate(
    store: SqlAlchemyStore, experiment_id: int
):
    """The two halves are stored separately so they can be merged.

    A precomputed per-minute rate could not be summed across buckets, which is what
    every later stage does -- the window fold, the coarse tiers, and the incremental
    cache's subtract all assume additivity.
    """
    _seed_trace(store, experiment_id, "t-1", start_ms=T, duration_ms=1_000)
    for span_id, status in (("s-1", "OK"), ("s-2", "ERROR"), ("s-3", "OK"), ("s-4", "OK")):
        _seed_span(
            store,
            experiment_id,
            "t-1",
            span_id,
            name="search_docs",
            span_type="TOOL",
            end_ms=T + 5_000,
            duration_ms=400,
            status=status,
        )
    _set_watermark(store, T - BUCKET_MS)

    RollupAggregator(store).run_once(now_ms=T + BUCKET_MS + LAG_MS)

    row = _rollups(store)[(SeriesKey("SPAN_NAME", experiment_id, "error_rate", "search_docs"), T)]
    assert row.count == 4  # attempts, including the failure
    assert row.sum == pytest.approx(1.0)  # failures
    # A ratio has nothing to bucket, so no sketch is stored.
    assert row.histogram is None


def test_the_error_rate_is_count_weighted_not_an_average_of_rates(
    store: SqlAlchemyStore, experiment_id: int
):
    """The reason a rate is stored as two numbers rather than one.

    A busy bucket at 1% and a near-idle bucket at 100% must not average to 50%. The
    quiet bucket has one request in it; weighting it equally with a thousand is how a
    rate alert ends up firing every night when traffic drops.
    """
    _seed_trace(store, experiment_id, "t-1", start_ms=T, duration_ms=1_000)
    # Bucket T: 10 calls, 1 failure (10%). Bucket T+1: 1 call, 1 failure (100%).
    for i in range(10):
        _seed_span(
            store,
            experiment_id,
            "t-1",
            f"s-{i}",
            name="search_docs",
            span_type="TOOL",
            end_ms=T + 5_000,
            duration_ms=400,
            status="ERROR" if i == 0 else "OK",
        )
    _seed_span(
        store,
        experiment_id,
        "t-1",
        "s-late",
        name="search_docs",
        span_type="TOOL",
        end_ms=T + BUCKET_MS + 5_000,
        duration_ms=400,
        status="ERROR",
    )
    _set_watermark(store, T - BUCKET_MS)

    RollupAggregator(store).run_once(now_ms=T + 2 * BUCKET_MS + LAG_MS)

    rollups = _rollups(store)
    series = SeriesKey("SPAN_NAME", experiment_id, "error_rate", "search_docs")
    first, second = rollups[(series, T)], rollups[(series, T + BUCKET_MS)]

    assert (first.count, first.sum) == (10, 1.0)
    assert (second.count, second.sum) == (1, 1.0)

    # Folding the window sums both halves, then divides once.
    window_rate = (first.sum + second.sum) / (first.count + second.count)
    assert window_rate == pytest.approx(2 / 11)

    # Averaging the per-bucket rates instead would give 55%, because the bucket
    # holding a single request would count as much as the one holding ten.
    mean_of_rates = ((first.sum / first.count) + (second.sum / second.count)) / 2
    assert mean_of_rates == pytest.approx(0.55)
    assert window_rate < 0.20


def test_the_span_error_rate_divides_tool_errors_by_tool_calls(
    store: SqlAlchemyStore, experiment_id: int
):
    """Numerator and denominator must come from the same population.

    SPAN_NAME series are restricted to TOOL spans, so the rate's denominator has to
    carry the same restriction -- otherwise a tool's failures get divided by every
    span in the trace and every rate reads far too low.
    """
    _seed_trace(store, experiment_id, "t-1", start_ms=T, duration_ms=1_000)
    _seed_span(
        store,
        experiment_id,
        "t-1",
        "s-tool",
        name="search_docs",
        span_type="TOOL",
        end_ms=T + 5_000,
        duration_ms=400,
        status="ERROR",
    )
    for i in range(9):
        _seed_span(
            store,
            experiment_id,
            "t-1",
            f"s-llm-{i}",
            name="openai.chat",
            span_type="LLM",
            end_ms=T + 5_000,
            duration_ms=400,
        )
    _set_watermark(store, T - BUCKET_MS)

    RollupAggregator(store).run_once(now_ms=T + BUCKET_MS + LAG_MS)

    rollups = _rollups(store)
    # One tool call, and it failed: 100%. The nine healthy LLM spans are not tools.
    tool = rollups[(SeriesKey("SPAN_NAME", experiment_id, "error_rate", ""), T)]
    assert (tool.count, tool.sum) == (1, 1.0)
    # By span type the denominator is every span, so the same failure is 1 in 10.
    every_span = rollups[(SeriesKey("SPAN_TYPE", experiment_id, "error_rate", ""), T)]
    assert (every_span.count, every_span.sum) == (10, 1.0)


def test_the_trace_error_rate_is_unsliced(store: SqlAlchemyStore, experiment_id: int):
    """Slicing a rate by status would make the error rate of ERROR traces 100%."""
    _seed_trace(store, experiment_id, "t-ok", start_ms=T, duration_ms=1_000)
    _seed_trace(store, experiment_id, "t-bad", start_ms=T, duration_ms=1_000, status="ERROR")
    _set_watermark(store, T - BUCKET_MS)

    RollupAggregator(store).run_once(now_ms=T + BUCKET_MS + LAG_MS)

    written = {
        key.dimension_value
        for (key, _) in _rollups(store)
        if key.dimension_key == "TRACES" and key.metric_key == "error_rate"
    }
    assert written == {""}
    row = _rollups(store)[(SeriesKey("TRACES", experiment_id, "error_rate", ""), T)]
    assert (row.count, row.sum) == (2, 1.0)


def test_a_bucket_with_no_failures_stores_zero_not_null(store: SqlAlchemyStore, experiment_id: int):
    """`else_=0`, not NULL.

    Both folds coerce a null sum with ``or 0.0``, so a NULL numerator is
    indistinguishable from a genuine zero -- except that it also makes the bucket
    look like it carried no numerator at all.
    """
    _seed_trace(store, experiment_id, "t-1", start_ms=T, duration_ms=1_000)
    _seed_span(
        store,
        experiment_id,
        "t-1",
        "s-1",
        name="search_docs",
        span_type="TOOL",
        end_ms=T + 5_000,
        duration_ms=400,
    )
    _set_watermark(store, T - BUCKET_MS)

    RollupAggregator(store).run_once(now_ms=T + BUCKET_MS + LAG_MS)

    row = _rollups(store)[(SeriesKey("SPAN_NAME", experiment_id, "error_rate", "search_docs"), T)]
    assert row.sum == 0.0
    assert row.sum is not None


def test_error_rate_and_error_count_deliberately_disagree(
    store: SqlAlchemyStore, experiment_id: int
):
    """Different populations, and the difference is by design.

    `error_count` is origin-deduplicated: a failure bubbling through three spans is
    one error. The rate is per-span: all three spans failed, and all three were
    called. Pinning it stops someone later "fixing" one to match the other.
    """
    _seed_trace(store, experiment_id, "t-1", start_ms=T, duration_ms=1_000)
    for span_id, name in (("s-1", "search_docs"), ("s-2", "retrieve"), ("s-3", "agent_run")):
        _seed_span(
            store,
            experiment_id,
            "t-1",
            span_id,
            name=name,
            span_type="TOOL",
            end_ms=T + 5_000,
            duration_ms=400,
            status="ERROR",
        )
    # One origin row, as the dedup would produce for a propagating failure.
    with _session(store) as session:
        session.add(
            SqlSpanError(
                trace_id="t-1",
                span_id="s-1",
                exception_type="TimeoutError",
                is_origin=True,
                span_name="search_docs",
                span_type="TOOL",
                experiment_id=experiment_id,
                timestamp_ms=T + 5_000,
            )
        )
    _set_watermark(store, T - BUCKET_MS)

    RollupAggregator(store).run_once(now_ms=T + BUCKET_MS + LAG_MS)

    rollups = _rollups(store)
    counted = rollups[(SeriesKey("SPAN_NAME", experiment_id, "error_count", ""), T)]
    rated = rollups[(SeriesKey("SPAN_NAME", experiment_id, "error_rate", ""), T)]

    assert counted.count == 1  # the origin only
    assert (rated.count, rated.sum) == (3, 3.0)  # every span that failed, over every call


def test_every_sliced_grouping_also_writes_an_unscoped_total(store: SqlAlchemyStore):
    """ "Any model" has to name a series that exists.

    The form's default for a sliced dimension is "any value", which sends an empty
    `dimension_value`. Only `trace_info` used to emit that series, so an unscoped
    rule on cost, span latency, errors or judge scores read zero rows forever: it
    showed green and could never fire.

    A grouping with no `dim_index` is exempt -- it is unsliced, so its only series
    already *is* the empty-valued one.
    """
    missing = [
        f"{source.name}/{grouping.dimension_key}"
        for source in _build_sources(store.db_type)
        for grouping in source.groupings
        if grouping.dim_index is not None and not grouping.include_total
    ]
    assert missing == []


def test_an_unscoped_series_sums_every_value_of_its_dimension(
    store: SqlAlchemyStore, experiment_id: int
):
    _seed_trace(store, experiment_id, "t-1", start_ms=T, duration_ms=1_000)
    for span_id, name in (("s-1", "search_docs"), ("s-2", "fetch_page")):
        _seed_span(
            store,
            experiment_id,
            "t-1",
            span_id,
            name=name,
            span_type="TOOL",
            end_ms=T + 5_000,
            duration_ms=400,
        )
    _set_watermark(store, T - BUCKET_MS)

    RollupAggregator(store).run_once(now_ms=T + BUCKET_MS + LAG_MS)

    rollups = _rollups(store)
    per_value = [
        row
        for (key, _), row in rollups.items()
        if key.dimension_key == "SPAN_NAME" and key.metric_key == "latency" and key.dimension_value
    ]
    total = rollups[(SeriesKey("SPAN_NAME", experiment_id, "latency", ""), T)]

    assert len(per_value) == 2
    # Not merely present: the total is the sum of the values it stands for, from the
    # same scan rather than a second query.
    assert total.count == sum(r.count for r in per_value)
    assert total.sum == pytest.approx(sum(r.sum for r in per_value))


def test_every_work_unit_can_be_named_by_some_rule(store: SqlAlchemyStore):
    """A unit no rule can reference is work nobody can consume.

    Eight of the seventeen used to be in exactly that state -- the token and cost
    components had work units, sketch grids and a scan, but no `METRIC_CATALOGUE`
    entry, so `validate_metric_triple` rejected every rule naming one. The scan ran
    every minute regardless, because the subscription filter applies to the
    accumulators rather than to the query.
    """
    orphans = [
        unit.name
        for unit in build_work_units(store.db_type)
        if unit.metric_key not in METRIC_CATALOGUE
        or unit.grouping.dimension_key not in METRIC_CATALOGUE[unit.metric_key].dimension_keys
    ]
    assert orphans == []


def test_every_catalogue_pair_has_a_unit_that_writes_it(store: SqlAlchemyStore):
    """The other direction: a rule the form offers must have something feeding it.

    A catalogue entry with no work unit is worse than a missing one -- the rule is
    accepted, the series is never written, and it silently never fires.
    """
    units = {
        (unit.grouping.dimension_key, unit.metric_key) for unit in build_work_units(store.db_type)
    }
    missing = [
        (metric_key, dimension_key)
        for metric_key, spec in METRIC_CATALOGUE.items()
        for dimension_key in sorted(spec.dimension_keys)
        if (dimension_key, metric_key) not in units
    ]
    assert missing == []


def test_the_longest_window_fits_inside_raw_retention():
    """The constraint that bounds ``MAX_WINDOW_SECONDS``, asserted nowhere else.

    The incremental merge path expires buckets by re-reading them at one-minute
    granularity. A window reaching past raw retention therefore reads an empty range,
    subtracts nothing, and silently stops shedding -- the accumulator drifts upward
    until the hourly full recompute snaps it back, which reads as noise rather than
    as a bug.

    The two constants live in different modules with no import between them, so
    nothing but this test stops one moving without the other. It exists because the
    ceiling was briefly set to seven days, four days past retention, and nothing
    failed.
    """
    from mlflow.genai.alerts.timescale import RAW_RETENTION_MS

    assert MAX_WINDOW_SECONDS * 1000 <= RAW_RETENTION_MS


def test_gap_markers_reach_as_far_back_as_the_longest_window():
    """``_gap_range`` keeps only the tail, and that is safe only while these agree.

    Buckets older than the cap are never marked, so they are indistinguishable from
    genuinely quiet minutes. That is correct precisely because no window can reach
    them -- if the cap fell below the longest window, a long rule would silently
    under-count across an outage instead of reporting no-data.
    """
    from mlflow.genai.alerts.aggregator import MAX_GAP_BUCKETS

    assert MAX_GAP_BUCKETS == MAX_WINDOW_SECONDS // (BUCKET_MS // 1000)


def test_setup_timescale_no_longer_builds_a_daily_tier(store: SqlAlchemyStore):
    """The 1d aggregate was built, refreshed and read by nothing, so it was removed.

    Asserted rather than assumed: a continuous aggregate reintroduced without a
    consumer is a background job and an unbounded table earning nothing, and the
    hourly tier's ninety-day retention already covers the longest window.
    """
    import mlflow.genai.alerts.timescale as timescale_module

    source = Path(timescale_module.__file__).read_text(encoding="utf-8")
    # The docstring explains the removal; no statement may create or schedule it.
    assert "CREATE MATERIALIZED VIEW IF NOT EXISTS metric_rollups_1d" not in source
    assert "add_continuous_aggregate_policy('metric_rollups_1d'" not in source


def test_setup_timescale_is_a_no_op_on_sqlite(store: SqlAlchemyStore):
    result = setup_timescale(store.engine)

    assert result.applied is False
    assert "not PostgreSQL" in result.reason
    assert result.statements_run == []
