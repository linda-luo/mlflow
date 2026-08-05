"""Continuously emit agent traffic, healthy at first and then degrading.

Writes traces with *current* completion times, so the aggregator always has a fresh
bucket to seal. Without this the rollup pipeline is only ever a static snapshot --
there is nothing arriving for a scheduled job to do.
"""

import json
import os
import random
import time
import uuid

from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource as OTelResource
from opentelemetry.sdk.trace import Event as OTelEvent
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

from mlflow.entities.span import create_mlflow_span
from mlflow.store.tracking.dbmodels.models import SqlAssessments, SqlTraceInfo, SqlTraceMetrics
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.utils import TraceJSONEncoder
from mlflow.utils.time import get_current_time_millis

MINUTE_MS = 60_000
EXPERIMENT_NAME = "checkout-agent"
TICK_SECONDS = 5

MODELS = ("gpt-5", "claude-sonnet-5")
"""Two models so the SPAN_MODEL breakdown has more than one value to slice by."""

rng = random.Random()


def make_span(
    name,
    span_id,
    trace_id,
    end_ms,
    parent_id=None,
    span_type="TOOL",
    events=None,
    attributes=None,
    status=trace_api.StatusCode.UNSET,
):
    def context(id_num):
        return trace_api.SpanContext(
            trace_id=rng.getrandbits(96),
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
            **(attributes or {}),
        },
        events=events or [],
        start_time=(end_ms - 2_000) * 1_000_000,
        end_time=end_ms * 1_000_000,
        # Real spans are normalized to OK or ERROR when they end, and
        # `record_exception` sets ERROR alongside the event. These spans are built
        # directly, so the status has to be passed in or the error-rate rollup --
        # which reads `status` -- would see every failure as healthy.
        status=trace_api.Status(status),
        resource=OTelResource.get_empty(),
    )
    return create_mlflow_span(otel_span, trace_id, span_type)


def token_usage(degraded):
    """Per-trace token counts, split the way a real provider reports them.

    Degrading inflates the prompt rather than the completion, and collapses the
    cache hit rate -- which is what makes the input/output and cache-read splits
    worth alerting on separately. Watching `total_tokens` alone cannot tell a
    context-stuffing regression from a verbose model.
    """
    if degraded:
        cached = rng.randint(0, 400)
        fresh = rng.randint(9_000, 20_000)
        output = rng.randint(200, 600)
    else:
        cached = rng.randint(3_000, 6_000)
        fresh = rng.randint(400, 1_200)
        output = rng.randint(150, 500)
    return {
        "input_tokens": cached + fresh,
        "output_tokens": output,
        "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": rng.randint(0, 300),
        "total_tokens": cached + fresh + output,
    }


def llm_cost(tokens, model):
    """Dollar cost of one call, priced per million tokens."""
    rate_in, rate_out = (1.25, 10.0) if model == "gpt-5" else (3.0, 15.0)
    input_cost = tokens["input_tokens"] / 1e6 * rate_in
    output_cost = tokens["output_tokens"] / 1e6 * rate_out
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


def exception_event(exception_type, end_ms):
    return OTelEvent(
        name="exception",
        attributes={
            "exception.message": "upstream did not respond within 30s",
            "exception.type": exception_type,
            "exception.stacktrace": f"Traceback\n{exception_type}: timed out",
        },
        timestamp=end_ms * 1_000_000,
    )


def emit_batch(store, experiment_id, degraded, traces_per_tick=None):
    """One tick: a few completed traces plus their judge verdicts.

    ``traces_per_tick`` overrides the demo's own low rate. The load test reuses
    this function, so the shapes stay identical whatever the volume.
    """
    now_ms = get_current_time_millis()
    rows = []
    count = traces_per_tick if traces_per_tick is not None else rng.randint(3, 6)
    for _ in range(count):
        if degraded:
            latency = rng.randint(20 * MINUTE_MS, 70 * MINUTE_MS)
            status = "ERROR" if rng.random() < 0.25 else "OK"
            passed = 0.0 if rng.random() < 0.4 else 1.0
        else:
            latency = rng.randint(800, 6_000)
            status = "ERROR" if rng.random() < 0.02 else "OK"
            passed = 1.0
        rows.append((f"tr-{uuid.uuid4().hex}", now_ms, latency, status, passed))

    tokens_by_trace = {trace_id: token_usage(degraded) for trace_id, *_ in rows}

    with store.ManagedSessionMaker(read_only=False) as session:
        for trace_id, end_ms, latency, status, _ in rows:
            session.add(
                SqlTraceInfo(
                    request_id=trace_id,
                    experiment_id=int(experiment_id),
                    timestamp_ms=end_ms - latency,
                    execution_time_ms=latency,
                    # The column the whole rollup buckets on.
                    end_time_ms=end_ms,
                    status=status,
                    request_preview="Place order for cart #8812",
                    response_preview="Order confirmed",
                )
            )
        session.flush()
        for trace_id, end_ms, _, _, passed in rows:
            session.add(
                SqlAssessments(
                    assessment_id=f"a-{uuid.uuid4().hex[:16]}",
                    trace_id=trace_id,
                    experiment_id=int(experiment_id),
                    name="safety",
                    assessment_type="feedback",
                    value=json.dumps(passed),
                    created_timestamp=end_ms,
                    last_updated_timestamp=end_ms,
                    source_type="LLM_JUDGE",
                    source_id="safety-judge",
                    valid=True,
                )
            )
        # Trace-level token counts. Written directly, like the trace rows above:
        # this generator does not go through `start_trace`, which is what would
        # normally denormalize these.
        for trace_id, end_ms, *_ in rows:
            for key, value in tokens_by_trace[trace_id].items():
                session.add(
                    SqlTraceMetrics(
                        request_id=trace_id,
                        key=key,
                        value=float(value),
                        experiment_id=int(experiment_id),
                        # The column the token rollup buckets on.
                        timestamp_ms=end_ms,
                    )
                )
        session.commit()

    # One LLM span per trace, through the real ingest path: `log_spans` turns the
    # cost attribute into `span_metrics` rows and the model attribute into the
    # SPAN_MODEL dimension, so the cost rollups exercise the same code a real
    # client would.
    for trace_id, end_ms, *_ in rows:
        model = rng.choice(MODELS)
        cost = llm_cost(tokens_by_trace[trace_id], model)
        store.log_spans(
            str(experiment_id),
            [
                make_span(
                    "chat_completion",
                    rng.getrandbits(63),
                    trace_id,
                    end_ms,
                    span_type="LLM",
                    attributes={
                        SpanAttributeKey.MODEL: json.dumps(model),
                        SpanAttributeKey.LLM_COST: json.dumps(cost),
                    },
                )
            ],
        )

    # A tool call on every trace, mostly succeeding. Without these there is no
    # denominator: `search_docs` spans would exist only when they failed, and its
    # error rate would read 100% forever.
    failure_odds = 0.35 if degraded else 0.03
    tool_failures = 0
    for trace_id, end_ms, *_ in rows:
        failed = rng.random() < failure_odds
        tool_failures += failed
        store.log_spans(
            str(experiment_id),
            [
                make_span(
                    "search_docs",
                    rng.getrandbits(63),
                    trace_id,
                    end_ms,
                    status=(trace_api.StatusCode.ERROR if failed else trace_api.StatusCode.OK),
                    events=[exception_event("TimeoutError", end_ms)] if failed else None,
                )
            ],
        )

    # A propagating failure, so `span_errors` sees a chain to deduplicate: the same
    # exception recorded on three spans must count once. Goes through the real ingest
    # path rather than being written by hand.
    if degraded and rng.random() < 0.7:
        trace_id = f"tr-{uuid.uuid4().hex}"
        store.log_spans(
            str(experiment_id),
            [
                make_span(
                    "search_docs",
                    1,
                    trace_id,
                    now_ms,
                    parent_id=2,
                    status=trace_api.StatusCode.ERROR,
                    events=[exception_event("TimeoutError", now_ms)],
                ),
                make_span(
                    "retrieve",
                    2,
                    trace_id,
                    now_ms,
                    parent_id=3,
                    span_type="CHAIN",
                    status=trace_api.StatusCode.ERROR,
                    events=[exception_event("TimeoutError", now_ms)],
                ),
                make_span(
                    "checkout_agent",
                    3,
                    trace_id,
                    now_ms,
                    span_type="AGENT",
                    status=trace_api.StatusCode.ERROR,
                    events=[exception_event("TimeoutError", now_ms)],
                ),
            ],
        )
        tool_failures += 1
    return len(rows), tool_failures


def main():
    uri = os.environ["MLFLOW_BACKEND_STORE_URI"]
    healthy_minutes = float(os.environ.get("DEMO_HEALTHY_MINUTES", "4"))
    # Traces per second. The default is deliberately low -- the demo is meant to be
    # readable, and a rule firing is easier to follow at a handful of traces a tick.
    # Raise it to watch the seeded rules under real volume.
    rate = float(os.environ.get("DEMO_TRACES_PER_SECOND", "0"))
    traces_per_tick = max(1, round(rate * TICK_SECONDS)) if rate > 0 else None
    store = SqlAlchemyStore(uri, "file:///tmp/mlflow-artifacts")

    experiments = store.search_experiments(filter_string=f"name = '{EXPERIMENT_NAME}'")
    while not experiments:
        print("[traffic] waiting for bootstrap to create the experiment...")
        time.sleep(3)
        experiments = store.search_experiments(filter_string=f"name = '{EXPERIMENT_NAME}'")
    experiment_id = experiments[0].experiment_id

    started = time.monotonic()
    print(
        f"[traffic] emitting into experiment {experiment_id}; "
        f"healthy for {healthy_minutes:g} min, then degrading"
        + (f"; {rate:g} traces/sec" if traces_per_tick else "")
    )
    announced = False
    total = 0
    while True:
        elapsed_min = (time.monotonic() - started) / 60
        degraded = elapsed_min >= healthy_minutes
        if degraded and not announced:
            print("[traffic] *** agent degrading: latency up, tools timing out ***")
            announced = True
        n, failures = emit_batch(store, experiment_id, degraded, traces_per_tick)
        total += n
        print(
            f"[traffic] t+{elapsed_min:4.1f}m  {'DEGRADED' if degraded else 'healthy '}  "
            f"+{n} traces (+{failures} tool failure)  total={total}"
        )
        time.sleep(TICK_SECONDS)


if __name__ == "__main__":
    main()
