"""Dedup contract for `span_errors`: exactly one `is_origin` row per
(trace_id, exception_type) chain, whatever order the spans arrive in.
"""

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource as OTelResource
from opentelemetry.sdk.trace import Event as OTelEvent
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

from mlflow.entities.span import Span, create_mlflow_span
from mlflow.genai.alerts.span_errors import extract_span_errors
from mlflow.store.tracking.dbmodels.models import SqlSpanError
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracing.utils import TraceJSONEncoder

pytestmark = pytest.mark.notrackingurimock

TRACE_ID = "tr-dedup"
CHAIN = ["search_docs", "retrieve", "agent_run"]
ARRIVAL_ORDERS = list(itertools.permutations(CHAIN))
_TRACE_NUM = 98765
_MS = 1_000_000


@pytest.fixture
def store(tmp_path: Path, db_uri: str) -> SqlAlchemyStore:
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir(exist_ok=True)
    return SqlAlchemyStore(db_uri, artifact_uri.as_uri())


@pytest.fixture
def experiment_id(store: SqlAlchemyStore) -> str:
    return store.create_experiment("span-error-dedup")


@dataclass(frozen=True)
class Row:
    """A `span_errors` row, detached from the session that read it."""

    span_id: str
    exception_type: str
    parent_span_id: str | None
    is_origin: bool
    span_name: str
    span_type: str | None
    exception_message: str | None
    experiment_id: int
    timestamp_ms: int


def exception_event(
    exception_type: str, message: str = "boom", timestamp_ns: int = 1_500_000_000
) -> OTelEvent:
    """The event shape `SpanEvent.from_exception` produces."""
    return OTelEvent(
        name="exception",
        attributes={
            "exception.message": message,
            "exception.type": exception_type,
            "exception.stacktrace": f"Traceback\n{exception_type}: {message}",
        },
        timestamp=timestamp_ns,
    )


def make_span(
    name: str,
    span_id: int,
    parent_id: int | None = None,
    end_ns: int = 2_000_000_000,
    span_type: str = "TOOL",
    events: list[OTelEvent] | None = None,
    trace_id: str = TRACE_ID,
    start_ns: int = 500_000_000,
) -> Span:
    def context(id_num: int) -> trace_api.SpanContext:
        return trace_api.SpanContext(
            trace_id=_TRACE_NUM,
            span_id=id_num,
            is_remote=False,
            trace_flags=trace_api.TraceFlags(1),
        )

    otel_span = OTelReadableSpan(
        name=name,
        context=context(span_id),
        parent=context(parent_id) if parent_id is not None else None,
        attributes={
            "mlflow.traceRequestId": json.dumps(trace_id),
            "mlflow.spanType": json.dumps(span_type, cls=TraceJSONEncoder),
        },
        events=events or [],
        start_time=start_ns,
        end_time=end_ns,
        status=trace_api.Status(trace_api.StatusCode.UNSET),
        resource=OTelResource.get_empty(),
    )
    return create_mlflow_span(otel_span, trace_id, span_type)


def ingest(store: SqlAlchemyStore, experiment_id: str, spans: list[Span]) -> int:
    """Log a batch of spans, then extract its errors -- what `_log_spans_once` will do."""
    store.log_spans(experiment_id, spans)
    with store.ManagedSessionMaker(read_only=False) as session:
        return extract_span_errors(
            session, spans, {span.trace_id: int(experiment_id) for span in spans}
        )


def read_rows(store: SqlAlchemyStore, trace_id: str = TRACE_ID) -> list[Row]:
    with store.ManagedSessionMaker() as session:
        rows = (
            session
            .query(SqlSpanError)
            .filter(SqlSpanError.trace_id == trace_id)
            .order_by(SqlSpanError.exception_type, SqlSpanError.timestamp_ms)
            .all()
        )
        return [
            Row(
                span_id=row.span_id,
                exception_type=row.exception_type,
                parent_span_id=row.parent_span_id,
                is_origin=row.is_origin,
                span_name=row.span_name,
                span_type=row.span_type,
                exception_message=row.exception_message,
                experiment_id=row.experiment_id,
                timestamp_ms=row.timestamp_ms,
            )
            for row in rows
        ]


def origins(store: SqlAlchemyStore, exception_type: str | None = None) -> set[str]:
    return {
        row.span_name
        for row in read_rows(store)
        if row.is_origin and exception_type in (None, row.exception_type)
    }


def chain_spans() -> dict[str, Span]:
    """search_docs raises, retrieve propagates it, agent_run propagates it further."""
    return {
        "search_docs": make_span(
            "search_docs",
            span_id=3,
            parent_id=2,
            end_ns=1_000 * _MS,
            events=[exception_event("TimeoutError")],
        ),
        "retrieve": make_span(
            "retrieve",
            span_id=2,
            parent_id=1,
            end_ns=2_000 * _MS,
            span_type="RETRIEVER",
            events=[exception_event("TimeoutError")],
        ),
        "agent_run": make_span(
            "agent_run",
            span_id=1,
            end_ns=3_000 * _MS,
            span_type="AGENT",
            events=[exception_event("TimeoutError")],
        ),
    }


def test_in_order_arrival_marks_the_deepest_span_as_origin(store, experiment_id):
    spans = chain_spans()
    for name in CHAIN:
        ingest(store, experiment_id, [spans[name]])

    assert origins(store) == {"search_docs"}
    assert len(read_rows(store)) == 3


def test_reverse_arrival_marks_the_deepest_span_as_origin(store, experiment_id):
    spans = chain_spans()
    for name in ("agent_run", "retrieve", "search_docs"):
        ingest(store, experiment_id, [spans[name]])

    assert origins(store) == {"search_docs"}
    assert len(read_rows(store)) == 3


@pytest.mark.parametrize("arrival", ARRIVAL_ORDERS)
def test_every_arrival_order_converges_to_the_same_state(store, experiment_id, arrival):
    spans = chain_spans()
    for name in arrival:
        ingest(store, experiment_id, [spans[name]])

    assert {(row.span_name, row.is_origin) for row in read_rows(store)} == {
        ("search_docs", True),
        ("retrieve", False),
        ("agent_run", False),
    }


@pytest.mark.parametrize("arrival", ARRIVAL_ORDERS)
def test_one_batch_carrying_the_whole_chain_converges(store, experiment_id, arrival):
    spans = chain_spans()
    assert ingest(store, experiment_id, [spans[name] for name in arrival]) == 3

    assert {(row.span_name, row.is_origin) for row in read_rows(store)} == {
        ("search_docs", True),
        ("retrieve", False),
        ("agent_run", False),
    }


@pytest.mark.parametrize("arrival", ARRIVAL_ORDERS)
def test_identical_end_times_do_not_break_convergence(store, experiment_id, arrival):
    """End-time sorting is an optimization; `_demote_parent` is what makes it correct."""
    spans = {
        "search_docs": make_span(
            "search_docs",
            span_id=3,
            parent_id=2,
            end_ns=_MS,
            events=[exception_event("ValueError")],
        ),
        "retrieve": make_span(
            "retrieve", span_id=2, parent_id=1, end_ns=_MS, events=[exception_event("ValueError")]
        ),
        "agent_run": make_span(
            "agent_run", span_id=1, end_ns=_MS, events=[exception_event("ValueError")]
        ),
    }
    ingest(store, experiment_id, [spans[name] for name in arrival])

    assert origins(store) == {"search_docs"}


def test_two_exception_types_get_one_origin_each(store, experiment_id):
    # search_docs times out; fetch_page hits a rate limit. Both bubble to agent_run,
    # which therefore records two exception events.
    spans = [
        make_span(
            "search_docs",
            span_id=3,
            parent_id=1,
            end_ns=1_000 * _MS,
            events=[exception_event("TimeoutError")],
        ),
        make_span(
            "fetch_page",
            span_id=4,
            parent_id=1,
            end_ns=1_500 * _MS,
            events=[exception_event("RateLimitError")],
        ),
        make_span(
            "agent_run",
            span_id=1,
            end_ns=3_000 * _MS,
            span_type="AGENT",
            events=[exception_event("TimeoutError"), exception_event("RateLimitError")],
        ),
    ]
    assert ingest(store, experiment_id, spans) == 4

    assert origins(store, "TimeoutError") == {"search_docs"}
    assert origins(store, "RateLimitError") == {"fetch_page"}
    assert origins(store) == {"search_docs", "fetch_page"}


def test_sibling_tools_failing_with_the_same_type_are_separate_chains(store, experiment_id):
    """Two independent failures of one type are two origins, not one."""
    spans = [
        make_span(
            "search_docs",
            span_id=3,
            parent_id=1,
            end_ns=1_000 * _MS,
            events=[exception_event("TimeoutError")],
        ),
        make_span(
            "fetch_page",
            span_id=4,
            parent_id=1,
            end_ns=1_500 * _MS,
            events=[exception_event("TimeoutError")],
        ),
        make_span(
            "agent_run",
            span_id=1,
            end_ns=3_000 * _MS,
            span_type="AGENT",
            events=[exception_event("TimeoutError")],
        ),
    ]
    ingest(store, experiment_id, spans)

    rows = read_rows(store)
    assert {(row.span_name, row.is_origin) for row in rows} == {
        ("search_docs", True),
        ("fetch_page", True),
        ("agent_run", False),
    }


def test_span_without_an_exception_writes_nothing(store, experiment_id):
    spans = [
        make_span("agent_run", span_id=1, span_type="AGENT"),
        make_span("search_docs", span_id=3, parent_id=1),
    ]
    assert ingest(store, experiment_id, spans) == 0
    assert read_rows(store) == []


def test_non_exception_events_are_ignored(store, experiment_id):
    span = make_span(
        "agent_run",
        span_id=1,
        events=[OTelEvent(name="cache_miss", attributes={"exception.type": "TimeoutError"})],
    )
    assert ingest(store, experiment_id, [span]) == 0
    assert read_rows(store) == []


def test_caught_exception_writes_only_the_origin(store, experiment_id):
    """The tool failed, the agent handled it, and the trace succeeded."""
    spans = [
        make_span(
            "search_docs",
            span_id=3,
            parent_id=1,
            end_ns=1_000 * _MS,
            events=[exception_event("TimeoutError", "read timed out")],
        ),
        make_span("agent_run", span_id=1, end_ns=3_000 * _MS, span_type="AGENT"),
    ]
    assert ingest(store, experiment_id, spans) == 1

    match read_rows(store):
        case [row]:
            assert row.span_name == "search_docs"
            assert row.is_origin is True
            # The failure never reached the root, so it did not kill the request.
            assert row.parent_span_id is not None
        case rows:
            pytest.fail(f"expected one row, got {rows}")


def test_row_carries_the_fields_the_rollup_reads(store, experiment_id):
    span = make_span(
        "search_docs",
        span_id=3,
        parent_id=1,
        end_ns=4_200 * _MS,
        span_type="TOOL",
        events=[exception_event("TimeoutError", "read timed out after 30s")],
    )
    ingest(store, experiment_id, [span])

    match read_rows(store):
        case [row]:
            assert row.span_id == span.span_id
            assert row.parent_span_id == span.parent_id
            assert row.exception_type == "TimeoutError"
            assert row.exception_message == "read timed out after 30s"
            assert row.span_name == "search_docs"
            assert row.span_type == "TOOL"
            assert row.experiment_id == int(experiment_id)
            # Rollups bucket by completion, so the span's end time is what is stored.
            assert row.timestamp_ms == 4_200
        case rows:
            pytest.fail(f"expected one row, got {rows}")


def test_oversized_values_are_truncated_to_the_column_widths(store, experiment_id):
    span = make_span(
        "n" * 600,
        span_id=3,
        span_type="t" * 80,
        events=[exception_event("E" * 300, "m" * 1500)],
    )
    ingest(store, experiment_id, [span])

    match read_rows(store):
        case [row]:
            assert len(row.exception_type) == 250
            assert len(row.exception_message) == 1000
            assert len(row.span_name) == 500
            assert len(row.span_type) == 50
        case rows:
            pytest.fail(f"expected one row, got {rows}")


def test_repeated_exception_type_on_one_span_writes_one_row(store, experiment_id):
    span = make_span(
        "search_docs",
        span_id=3,
        events=[
            exception_event("TimeoutError", "attempt 1"),
            exception_event("TimeoutError", "attempt 2"),
            exception_event("TimeoutError", "attempt 3"),
        ],
    )
    assert ingest(store, experiment_id, [span]) == 1

    match read_rows(store):
        case [row]:
            assert row.exception_message == "attempt 1"
        case rows:
            pytest.fail(f"expected one row, got {rows}")


def test_exception_event_without_a_type_is_skipped(store, experiment_id):
    span = make_span(
        "search_docs",
        span_id=3,
        events=[
            OTelEvent(name="exception", attributes={"exception.message": "no type recorded"}),
            exception_event("TimeoutError"),
        ],
    )
    assert ingest(store, experiment_id, [span]) == 1
    assert [row.exception_type for row in read_rows(store)] == ["TimeoutError"]


def test_relogging_a_span_does_not_resurrect_a_demoted_origin(store, experiment_id):
    spans = chain_spans()
    ingest(store, experiment_id, [spans["search_docs"], spans["retrieve"]])
    assert origins(store) == {"search_docs"}

    # An export retry re-sends the parent on its own.
    ingest(store, experiment_id, [spans["retrieve"]])

    assert origins(store) == {"search_docs"}
    assert len(read_rows(store)) == 2


def test_spans_whose_trace_is_unknown_are_skipped(store, experiment_id):
    """`_log_spans_once` drops spans whose trace_info could not be created, and
    passes a mapping that omits them.

    Deliberately does not call ``log_spans``: now that the hook is wired into the
    ingest path, logging a span *creates* its trace, so the trace would be known
    and the case under test could not arise. The unknown-trace path is reachable
    only when trace creation itself failed, which is exactly what an omitted
    mapping entry represents.
    """
    span = make_span("search_docs", span_id=3, events=[exception_event("TimeoutError")])

    with store.ManagedSessionMaker(read_only=False) as session:
        assert extract_span_errors(session, [span], {}) == 0

    assert read_rows(store) == []


def test_errors_from_two_traces_are_independent(store, experiment_id):
    other_trace = "tr-other"
    spans = [
        make_span(
            "search_docs",
            span_id=3,
            parent_id=1,
            end_ns=1_000 * _MS,
            events=[exception_event("TimeoutError")],
        ),
        make_span(
            "agent_run",
            span_id=1,
            end_ns=3_000 * _MS,
            events=[exception_event("TimeoutError")],
        ),
        make_span(
            "agent_run",
            span_id=1,
            end_ns=3_000 * _MS,
            trace_id=other_trace,
            events=[exception_event("TimeoutError")],
        ),
    ]
    assert ingest(store, experiment_id, spans) == 3

    assert origins(store) == {"search_docs"}
    # The other trace's root raised on its own; a same-named span elsewhere must
    # not demote it.
    assert [(row.span_name, row.is_origin) for row in read_rows(store, other_trace)] == [
        ("agent_run", True)
    ]
