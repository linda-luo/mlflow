"""Seed a database with agent traffic and alert rules, so the Alerts page has
something real to show.

Builds one hour of traffic for a checkout agent that degrades partway through:
latency climbs, a tool starts timing out, and a safety judge starts failing. Then
it runs the real aggregator and the real evaluator, so the alerts on the page were
produced by the same code path production would use -- nothing is hand-written
into ``alert_instances``.

    uv run --frozen python dev/alert_demo.py --backend-store-uri sqlite:///demo.db
"""

import argparse
import json
import random
import uuid
from pathlib import Path

from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource as OTelResource
from opentelemetry.sdk.trace import Event as OTelEvent
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

from mlflow.entities.span import create_mlflow_span
from mlflow.genai.alerts.aggregator import RollupAggregator
from mlflow.genai.alerts.entities import AlertRule, derive_evaluation_interval_seconds
from mlflow.genai.alerts.evaluator import AlertEvaluator
from mlflow.genai.alerts.sql_rollup_reader import SqlRollupReader
from mlflow.store.tracking.dbmodels.models import SqlAssessments, SqlTraceInfo
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracing.utils import TraceJSONEncoder
from mlflow.utils.time import get_current_time_millis

BUCKET_MS = 60_000
MINUTE_MS = 60_000
WINDOW_MINUTES = 60
HEALTHY_MINUTES = 40

rng = random.Random(7)


def make_span(name, span_id, trace_id, end_ms, parent_id=None, span_type="TOOL", events=None):
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
        },
        events=events or [],
        start_time=(end_ms - 2_000) * 1_000_000,
        end_time=end_ms * 1_000_000,
        status=trace_api.Status(trace_api.StatusCode.UNSET),
        resource=OTelResource.get_empty(),
    )
    return create_mlflow_span(otel_span, trace_id, span_type)


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


def seed_traffic(store, experiment_id, now_ms):
    """One hour of traffic that degrades in its final third."""
    traces, assessments = [], []
    for minute in range(WINDOW_MINUTES, 0, -1):
        end_base = now_ms - minute * MINUTE_MS
        degraded = minute < (WINDOW_MINUTES - HEALTHY_MINUTES)
        for i in range(5):
            end_ms = end_base + i * 8_000
            if degraded:
                latency = rng.randint(50 * MINUTE_MS, 95 * MINUTE_MS)
                status = "ERROR" if rng.random() < 0.25 else "OK"
            else:
                latency = rng.randint(1_200, 9_000)
                status = "ERROR" if rng.random() < 0.02 else "OK"
            trace_id = f"tr-{uuid.uuid4().hex}"
            traces.append((trace_id, end_ms, latency, status))
            # The safety judge starts failing once the agent degrades.
            passed = 0.0 if (degraded and rng.random() < 0.35) else 1.0
            assessments.append((trace_id, end_ms, "safety", passed))

    with store.ManagedSessionMaker(read_only=False) as session:
        for trace_id, end_ms, latency, status in traces:
            session.add(
                SqlTraceInfo(
                    request_id=trace_id,
                    experiment_id=int(experiment_id),
                    timestamp_ms=end_ms - latency,
                    execution_time_ms=latency,
                    end_time_ms=end_ms,
                    status=status,
                    request_preview="Place order for cart #8812",
                    response_preview="Order confirmed",
                )
            )
        session.flush()
        for trace_id, end_ms, name, value in assessments:
            session.add(
                SqlAssessments(
                    assessment_id=f"a-{uuid.uuid4().hex[:16]}",
                    trace_id=trace_id,
                    experiment_id=int(experiment_id),
                    name=name,
                    assessment_type="feedback",
                    value=json.dumps(value),
                    created_timestamp=end_ms,
                    last_updated_timestamp=end_ms,
                    source_type="LLM_JUDGE",
                    source_id="safety-judge",
                    valid=True,
                )
            )
        session.commit()
    return len(traces)


def seed_tool_failures(store, experiment_id, now_ms):
    """Real spans through the real ingest path, so span_errors is produced by the
    wiring rather than written by hand.
    """
    count = 0
    for minute in range(18, 0, -1):
        end_ms = now_ms - minute * MINUTE_MS
        trace_id = f"tr-{uuid.uuid4().hex}"
        spans = [
            make_span(
                "search_docs", 1, trace_id, end_ms, parent_id=2,
                events=[exception_event("TimeoutError", end_ms)],
            ),
            make_span(
                "retrieve", 2, trace_id, end_ms, parent_id=3, span_type="CHAIN",
                events=[exception_event("TimeoutError", end_ms)],
            ),
            make_span(
                "checkout_agent", 3, trace_id, end_ms, span_type="AGENT",
                events=[exception_event("TimeoutError", end_ms)],
            ),
        ]
        store.log_spans(experiment_id, spans)
        count += 1
    return count


def rule(experiment_id, name, **kwargs):
    window_seconds = kwargs.pop("window_seconds", 3600)
    return AlertRule(
        alert_rule_id=str(uuid.uuid4()),
        experiment_id=int(experiment_id),
        name=name,
        window_seconds=window_seconds,
        evaluation_interval_seconds=derive_evaluation_interval_seconds(window_seconds),
        **kwargs,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-store-uri", required=True)
    parser.add_argument("--artifacts", default="./mlruns")
    args = parser.parse_args()

    Path(args.artifacts).mkdir(parents=True, exist_ok=True)
    store = SqlAlchemyStore(args.backend_store_uri, Path(args.artifacts).absolute().as_uri())

    now_ms = get_current_time_millis()
    experiment_id = store.create_experiment(f"checkout-agent-{uuid.uuid4().hex[:6]}")

    # The default backfill limit is 60 buckets; seeding more than an hour at once
    # would otherwise be recorded as a gap, which correctly reports NO_DATA and
    # would make the demo look broken.
    aggregator = RollupAggregator(store, max_backfill_buckets=WINDOW_MINUTES + 30)
    # A fresh install starts its watermark at "now" and will not scan history, so
    # the watermark has to be established before the traffic is seeded -- exactly
    # as it would be on a server that has been running all along.
    aggregator.run_once(now_ms=now_ms - (WINDOW_MINUTES + 5) * MINUTE_MS)

    n_traces = seed_traffic(store, experiment_id, now_ms)
    n_tool_failures = seed_tool_failures(store, experiment_id, now_ms)

    run = aggregator.run_once(now_ms=now_ms)
    sealed = sum(len(r.sealed_buckets) for r in run.sources.values())

    rules = [
        rule(
            experiment_id, "Checkout p95 latency",
            metric_key="latency", dimension_key="TRACES", aggregation="PERCENTILE",
            percentile_value=95, comparator="GT", threshold=45 * MINUTE_MS,
            severity="HIGH", sustain_seconds=300,
            description="Customers wait too long to get an order confirmation.",
        ),
        rule(
            experiment_id, "Average latency regression",
            metric_key="latency", dimension_key="TRACES", aggregation="AVG",
            comparator="GT", threshold=30 * MINUTE_MS, severity="MEDIUM",
        ),
        rule(
            experiment_id, "search_docs failures",
            metric_key="error_count", dimension_key="SPAN_NAME",
            dimension_value="search_docs", aggregation="COUNT",
            comparator="GTE", threshold=5, severity="HIGH",
        ),
        rule(
            experiment_id, "Safety judge pass rate",
            metric_key="assessment_value", dimension_key="ASSESSMENTS",
            dimension_value="safety", aggregation="AVG",
            comparator="LT", threshold=0.9, severity="HIGH",
        ),
        rule(
            experiment_id, "Traffic dropped",
            metric_key="latency", dimension_key="TRACES", aggregation="COUNT",
            comparator="LT", threshold=10, severity="LOW",
            description="Absence is also a signal: nobody is calling the agent.",
        ),
    ]
    created = [store.create_alert_rule(r) for r in rules]

    evaluator = AlertEvaluator(SqlRollupReader(store), store)
    # Two cycles: the first opens PENDING, the second lets a sustained rule fire.
    evaluator.evaluate_rules(created, now_ms)
    cycle = evaluator.evaluate_rules(created, now_ms + 6 * MINUTE_MS)

    open_instances = store.list_alert_instances(experiment_id)

    print(f"\n  experiment_id : {experiment_id}")
    print(f"  traces        : {n_traces} ({n_tool_failures} failing tool traces)")
    print(f"  buckets sealed: {sealed}")
    print(f"  rules         : {len(created)}")
    print(f"  groups read   : {cycle.groups_read}")
    print(f"\n  open alerts   : {len(open_instances)}")
    by_id = {r.alert_rule_id: r for r in created}
    for instance in open_instances:
        rule_obj = by_id[instance.alert_rule_id]
        value = instance.observed_value
        # Latency is milliseconds; pass rates and counts are not. Rounding a 0.88
        # pass rate to "1" would make a firing rule look like it should not have.
        shown = f"{value / MINUTE_MS:.1f} min" if rule_obj.metric_key == "latency" else f"{value:g}"
        print(f"    [{instance.state:<7}] {rule_obj.name:<28} observed={shown}")
    firing = {i.alert_rule_id for i in open_instances}
    quiet = [r.name for r in created if r.alert_rule_id not in firing]
    for name in quiet:
        print(f"    [ok     ] {name}")
    print()


if __name__ == "__main__":
    main()
