"""Span error extraction for the ``log_spans`` ingest path.

A tool failure propagates: ``search_docs`` raises ``TimeoutError``, it bubbles
through ``retrieve`` to ``agent_run``, and all three spans record the same
exception event. Counting spans would make the error count depend on how deeply
someone nested their agent, so exactly one span per
``(trace_id, exception_type)`` chain is marked ``is_origin`` -- the deepest one,
which actually raised.

Arrival order cannot be relied on. Children normally export first because they
end first, but the async export queue runs several workers, so a parent can land
ahead of its child and a single ``log_spans`` call can carry both. The algorithm
below therefore converges to the same final state under every arrival order,
using two indexed lookups per erroring span and no tree walk.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass

from sqlalchemy.orm import Session

from mlflow.entities.span import Span
from mlflow.store.tracking.dbmodels.models import SqlSpanError

_logger = logging.getLogger(__name__)

# OTel reserves this event name for exceptions, and `SpanEvent.from_exception`
# emits it along with the two attributes below.
EXCEPTION_EVENT_NAME = "exception"
EXCEPTION_TYPE_ATTRIBUTE = "exception.type"
EXCEPTION_MESSAGE_ATTRIBUTE = "exception.message"

# Column widths from `SqlSpanError`. Values are truncated rather than rejected:
# dropping an error because its message was long would be a worse failure than
# storing a clipped one, and only `exception_type` is ever aggregated on.
_MAX_EXCEPTION_TYPE_LENGTH = 250
_MAX_EXCEPTION_MESSAGE_LENGTH = 1000
_MAX_SPAN_NAME_LENGTH = 500
_MAX_SPAN_TYPE_LENGTH = 50

_NANOSECONDS_PER_MILLISECOND = 1_000_000


@dataclass(frozen=True)
class _SpanError:
    """One prospective ``span_errors`` row, built without touching the database."""

    trace_id: str
    span_id: str
    exception_type: str
    parent_span_id: str | None
    span_name: str
    span_type: str | None
    exception_message: str | None
    experiment_id: int
    timestamp_ms: int


def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[:limit]


def _exception_type(event_attributes: Mapping[str, object]) -> str | None:
    value = event_attributes.get(EXCEPTION_TYPE_ATTRIBUTE)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _span_errors_for_span(span: Span, experiment_id: int) -> list[_SpanError]:
    """Build one row per distinct exception type recorded on ``span``.

    A span may carry several exception events -- a retry loop records one per
    attempt -- but the ``span_errors`` primary key is
    ``(trace_id, span_id, exception_type)``, so repeats of a type collapse into
    the first occurrence.
    """
    by_type: dict[str, _SpanError] = {}
    for event in span.events:
        if event.name != EXCEPTION_EVENT_NAME:
            continue
        attributes = event.attributes or {}
        exception_type = _exception_type(attributes)
        if exception_type is None:
            # An exception event with no `exception.type` cannot be attributed to
            # an error class, and inventing one ("Exception") would silently merge
            # it with genuine `Exception` raises in the ERROR rollup. OTel semantic
            # conventions require the attribute, so this is a malformed event.
            _logger.debug(
                "Skipping exception event without %s on span %s of trace %s",
                EXCEPTION_TYPE_ATTRIBUTE,
                span.span_id,
                span.trace_id,
            )
            continue

        exception_type = _truncate(exception_type, _MAX_EXCEPTION_TYPE_LENGTH)
        if exception_type in by_type:
            continue

        message = attributes.get(EXCEPTION_MESSAGE_ATTRIBUTE)
        # Rollups bucket by completion time. An in-progress span has no end time,
        # so fall back to when the exception was recorded.
        end_time_ns = span.end_time_ns if span.end_time_ns is not None else event.timestamp
        by_type[exception_type] = _SpanError(
            trace_id=span.trace_id,
            span_id=span.span_id,
            exception_type=exception_type,
            parent_span_id=span.parent_id,
            span_name=_truncate(span.name or "", _MAX_SPAN_NAME_LENGTH),
            span_type=_truncate(span.span_type, _MAX_SPAN_TYPE_LENGTH) if span.span_type else None,
            exception_message=(
                None if message is None else _truncate(str(message), _MAX_EXCEPTION_MESSAGE_LENGTH)
            ),
            experiment_id=int(experiment_id),
            timestamp_ms=end_time_ns // _NANOSECONDS_PER_MILLISECOND,
        )
    return list(by_type.values())


def _has_child_carrying_error(session: Session, error: _SpanError) -> bool:
    """Am I a propagation? True when a child of mine already reported this type.

    Served by ``index_span_errors_dedup`` on
    ``(trace_id, exception_type, parent_span_id)``.
    """
    return session.query(
        session
        .query(SqlSpanError)
        .filter(
            SqlSpanError.trace_id == error.trace_id,
            SqlSpanError.exception_type == error.exception_type,
            SqlSpanError.parent_span_id == error.span_id,
        )
        .exists()
    ).scalar()


def _demote_parent(session: Session, error: _SpanError) -> None:
    """My parent, if it reported this error, was propagating mine.

    This exists solely for out-of-order arrival: each span demotes its parent, so
    the origin converges bottom-up no matter what sequence the spans landed in.
    Matched on the primary key ``(trace_id, span_id, exception_type)``.

    Known imprecision: a parent that *independently* raises the same exception
    type is wrongly demoted. Span data cannot distinguish "I propagated X" from
    "I raised my own X", so this is accepted rather than fixed.
    """
    session.query(SqlSpanError).filter(
        SqlSpanError.trace_id == error.trace_id,
        SqlSpanError.exception_type == error.exception_type,
        SqlSpanError.span_id == error.parent_span_id,
    ).update({SqlSpanError.is_origin: False}, synchronize_session="evaluate")


def extract_span_errors(
    session: Session,
    spans: list[Span],
    experiment_ids: Mapping[str, int],
) -> int:
    """Write ``span_errors`` rows for the spans in a batch that recorded exceptions.

    Call this from ``_log_spans_once`` *after* the spans themselves have been
    upserted: ``span_errors`` has a ``(trace_id, span_id)`` foreign key into
    ``spans``, so a row written first would violate it. The write happens in the
    caller's session and is committed with the rest of the batch, so a failed
    ``log_spans`` leaves no error rows behind.

    Args:
        session: The open ingest session. Rows are added to it but not committed.
        spans: The spans being logged. Those without an exception event, and
            those whose trace was dropped (absent from ``experiment_ids``), are
            skipped.
        experiment_ids: Trace ID to experiment ID. ``experiment_id`` is
            denormalized onto ``span_errors`` so the rollup scan needs no join,
            and a trace may already live in an experiment other than the one
            ``log_spans`` was called with, so it is resolved per trace rather
            than taken from the call.

    Returns:
        The number of rows written.
    """
    errors: list[_SpanError] = []
    for span in spans:
        experiment_id = experiment_ids.get(span.trace_id)
        if experiment_id is None:
            continue
        errors.extend(_span_errors_for_span(span, experiment_id))

    if not errors:
        return 0

    # Deepest first: a child always ends before its parent, so end-time order puts
    # every span after the descendants it propagated from. That makes the common
    # case take the `_has_child_carrying_error` fast path. Correctness does not
    # depend on it -- `_demote_parent` repairs any other order.
    errors.sort(key=lambda e: e.timestamp_ms)

    for error in errors:
        is_origin = not _has_child_carrying_error(session, error)
        # `merge` rather than `add`: the same span can be logged more than once
        # (an in-progress span updated later, or an export retry), and a plain
        # insert would fail the primary key and abort the whole ingest. `is_origin`
        # is recomputed on every pass, so a repeat cannot resurrect a demoted span.
        session.merge(
            SqlSpanError(
                trace_id=error.trace_id,
                span_id=error.span_id,
                exception_type=error.exception_type,
                parent_span_id=error.parent_span_id,
                is_origin=is_origin,
                span_name=error.span_name,
                span_type=error.span_type,
                exception_message=error.exception_message,
                experiment_id=error.experiment_id,
                timestamp_ms=error.timestamp_ms,
            )
        )
        if error.parent_span_id is not None:
            _demote_parent(session, error)

    return len(errors)
