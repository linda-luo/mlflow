"""Load-test harness for the alerting rollup pipeline.

Documented in ``mlflow/genai/alerts/README.md`` alongside the code it tests.

Runs a ladder of traffic bursts (10s, then 30s, then 60s, at rising offered
rates) against the demo stack (see ``compose.yml``), and checks -- against the
real Postgres/Timescale database, not a mock -- that the aggregator kept up:

  1. No loss: every raw row behind a family is reflected in its rollups.
  2. No gaps: the aggregator never fell far enough behind to gap-mark.
  3. Percentile accuracy: the stored sketch's p95 is within 2% of the true one.
  4. Watermark recovery: the aggregator catches back up within a few ticks.
  5. Alerts saw the truth: a fired rule's observed value matches the raw rows.

Escalates only while the previous level passed every assertion, and stops at
the first failure. Never bursts longer than 60 seconds -- that is the ceiling
this script is bound by, not merely a default.

Traffic is written directly to the tracking store, reusing ``traffic.py``'s
trace/span shapes, exactly the way ``dev/alerting/traffic.py`` does it for the
demo. All of that -- and the correctness reads -- run *inside* the ``mlflow``
container, via ``docker compose exec``, so this script needs nothing installed
locally beyond Docker: no local Postgres driver, no local mlflow import.

Usage::

    uv run dev/alerting/load_test/run.py                       # full ladder
    uv run dev/alerting/load_test/run.py --burst 10 --rate 200  # one level only
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from string import Template

COMPOSE_DIR = Path(__file__).resolve().parent.parent
"""Where compose.yml lives -- one level up, since this script moved into its
own folder while the stack stayed put."""
MLFLOW_TRACKING_URI = "http://localhost:5002"
HEALTH_URL = f"{MLFLOW_TRACKING_URI}/health"

BUCKET_MS = 60_000
LAG_MS = 60_000
"""How long the aggregator waits past a bucket's end before sealing it. See
``mlflow.genai.alerts.aggregator.LAG_MS`` -- kept in sync by inspection, not
by import, because this script deliberately runs with nothing local to import."""

PCT_TOLERANCE = 0.02
"""Sketch accuracy: ALPHA in mlflow.genai.alerts.sketch."""

INGEST_TOLERANCE = 0.05
"""How much of what the client sent may fail to reach the database.

Not zero, because a trace flushed at the very end of a burst can land in a bucket
past the checked range. Anything beyond this is real loss -- the SDK's async
export queue is bounded and drops spans ("Queue full, dropping Span") rather than
applying backpressure, which is the right call for an app but means the client's
own count is not evidence that anything arrived."""

DEFAULT_LEVELS = [(10, 50), (30, 150), (60, 300)]
"""(burst_seconds, offered traces/sec), escalating in both duration and rate."""

WATERMARK_POLL_INTERVAL_S = 10
WATERMARK_MAX_TICKS = 14
"""Up to 140s of polling -- comfortably past LAG_MS + one aggregator tick."""

ALERT_POLL_INTERVAL_S = 20
ALERT_MAX_WAIT_S = 480
"""A rule waits a full window (>= 300s) plus jitter plus one evaluator tick
before its first evaluation -- see ``SqlAlchemyStore.create_alert_rule``."""

EXPERIMENT_RULE_NAME = "load-test: AVG latency subscription"


def floor_bucket(ms: int) -> int:
    return (ms // BUCKET_MS) * BUCKET_MS


###############################################################################
# Container plumbing
###############################################################################


def _run_in_container(code: str, *, timeout: float) -> dict:
    """Run ``code`` inside the ``mlflow`` container and parse its last stdout line as JSON.

    Everything data-dependent -- writing traces, reading rollups -- happens in
    here rather than on the host: the container already has mlflow, psycopg2,
    and a working ``MLFLOW_BACKEND_STORE_URI``, so nothing has to be installed
    or reimplemented locally, and the sketch/percentile math stays the product's
    own rather than a second copy of it.
    """
    result = subprocess.run(
        ["docker", "compose", "exec", "-T", "mlflow", "python", "-"],
        cwd=COMPOSE_DIR,
        input=code,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"container call failed (exit {result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"container call produced no output. stderr:\n{result.stderr}")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as e:
        raise RuntimeError(f"could not parse container output as JSON:\n{result.stdout}") from e


def _check_stack_up() -> None:
    try:
        urllib.request.urlopen(HEALTH_URL, timeout=5)
    except (urllib.error.URLError, OSError) as e:
        print(
            f"Could not reach {HEALTH_URL} ({e}).\n"
            "Bring up the demo stack first:\n"
            "    docker compose -f dev/alerting/compose.yml up --build -d",
            file=sys.stderr,
        )
        raise SystemExit(1)


###############################################################################
# Payloads run inside the mlflow container
###############################################################################

_SETUP_CODE = Template(r"""
import json, os, uuid
from mlflow.genai.alerts.entities import AlertRule, derive_evaluation_interval_seconds
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

store = SqlAlchemyStore(os.environ["MLFLOW_BACKEND_STORE_URI"], "file:///tmp/mlflow-artifacts")

experiment_name = "$experiment_name"
existing = store.search_experiments(filter_string=f"name = '{experiment_name}'")
experiment_id = (
    int(existing[0].experiment_id) if existing else int(store.create_experiment(experiment_name))
)

rule_name = "$rule_name"
rule = next((r for r in store.list_alert_rules(experiment_id) if r.name == rule_name), None)
if rule is None:
    # Subscribes this experiment to TRACES/latency, so the aggregator writes rollups
    # for it at all, and always breaches (threshold=-1) so the harness can compare a
    # fired instance's observed_value against the raw rows.
    window_seconds = 300
    rule = store.create_alert_rule(AlertRule(
        alert_rule_id=str(uuid.uuid4()),
        experiment_id=experiment_id,
        name=rule_name,
        metric_key="latency",
        dimension_key="TRACES",
        aggregation="AVG",
        comparator="GT",
        threshold=-1.0,
        window_seconds=window_seconds,
        evaluation_interval_seconds=derive_evaluation_interval_seconds(window_seconds),
        severity="LOW",
    ))

print(json.dumps({
    "experiment_id": experiment_id,
    "rule_id": rule.alert_rule_id,
}))
""")

_BURST_CODE = Template(r"""
import json, os, sys, time
sys.path.insert(0, "dev/alerting")
from traffic import emit_batch
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

store = SqlAlchemyStore(os.environ["MLFLOW_BACKEND_STORE_URI"], "file:///tmp/mlflow-artifacts")

experiment_id = $experiment_id
rate = $rate
duration_s = $duration_s

target_total = max(1, round(rate * duration_s))
written = 0
tool_failures = 0
start = time.monotonic()
deadline = start + duration_s
start_ms = int(time.time() * 1000)

while time.monotonic() < deadline and written < target_total:
    n, f = emit_batch(store, experiment_id, degraded=False)
    written += n
    tool_failures += f
    # Pace to the target rate rather than writing as fast as possible, so the
    # burst is sustained load over duration_s and not an instant dump.
    expected_elapsed = written / rate if rate > 0 else 0
    sleep_for = expected_elapsed - (time.monotonic() - start)
    remaining = deadline - time.monotonic()
    if sleep_for > 0 and remaining > 0:
        time.sleep(min(sleep_for, remaining))

end_ms = int(time.time() * 1000)
elapsed_s = time.monotonic() - start

print(json.dumps({
    "offered": target_total,
    "written": written,
    "elapsed_s": elapsed_s,
    "start_ms": start_ms,
    "end_ms": end_ms,
}))
""")

_WATERMARK_CODE = r"""
import json, os, time
from mlflow.genai.alerts.aggregator import sealable_max_bucket_ms
from mlflow.store.tracking.dbmodels.models import SqlRollupState
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

store = SqlAlchemyStore(os.environ["MLFLOW_BACKEND_STORE_URI"], "file:///tmp/mlflow-artifacts")
now_ms = int(time.time() * 1000)
sealable_max_ms = sealable_max_bucket_ms(now_ms)
with store.ManagedSessionMaker(read_only=True) as session:
    row = session.get(SqlRollupState, ("trace_info", "TRACES"))
    watermark_ms = row.watermark_ms if row is not None else None

print(json.dumps({
    "now_ms": now_ms,
    "sealable_max_ms": sealable_max_ms,
    "watermark_ms": watermark_ms,
}))
"""

_RANGE_CHECK_CODE = Template(r"""
import json, os, statistics
from mlflow.genai.alerts.aggregator import floor_bucket
from mlflow.genai.alerts.entities import BUCKET_MS, SeriesKey
from mlflow.genai.alerts.rollup_reader import aggregate_buckets
from mlflow.genai.alerts.sketch import spec_for
from mlflow.genai.alerts.sql_rollup_reader import SqlRollupReader
from mlflow.store.tracking.dbmodels.models import SqlTraceInfo
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

store = SqlAlchemyStore(os.environ["MLFLOW_BACKEND_STORE_URI"], "file:///tmp/mlflow-artifacts")

experiment_id = $experiment_id
range_start = floor_bucket($start_ms)
range_end = floor_bucket($end_ms - 1) + BUCKET_MS

with store.ManagedSessionMaker(read_only=True) as session:
    values = [
        float(v)
        for (v,) in session.query(SqlTraceInfo.execution_time_ms).filter(
            SqlTraceInfo.experiment_id == experiment_id,
            SqlTraceInfo.execution_time_ms.isnot(None),
            SqlTraceInfo.end_time_ms >= range_start,
            SqlTraceInfo.end_time_ms < range_end,
        ).all()
    ]

raw_count = len(values)
if raw_count >= 2:
    p95_true = statistics.quantiles(sorted(values), n=100, method="inclusive")[94]
elif raw_count == 1:
    p95_true = values[0]
else:
    p95_true = None

series = SeriesKey(
    dimension_key="TRACES", experiment_id=experiment_id, metric_key="latency", dimension_value=""
)
reader = SqlRollupReader(store)
buckets = reader.read_buckets(series, range_start, range_end)
rollup_count = sum(b.count for b in buckets if not b.is_gap)
gap_count = sum(1 for b in buckets if b.is_gap)
obs = aggregate_buckets(
    buckets, "PERCENTILE", range_start, range_end, percentile_value=95, spec=spec_for("latency")
)

print(json.dumps({
    "range_start_ms": range_start,
    "range_end_ms": range_end,
    "raw_count": raw_count,
    "rollup_count": rollup_count,
    "gap_count": gap_count,
    "p95_true": p95_true,
    "p95_sketch": obs.observed_value,
}))
""")

_ALERT_CHECK_CODE = Template(r"""
import json, os
from mlflow.store.tracking.dbmodels.models import SqlTraceInfo
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

store = SqlAlchemyStore(os.environ["MLFLOW_BACKEND_STORE_URI"], "file:///tmp/mlflow-artifacts")

rule_id = "$rule_id"
experiment_id = $experiment_id

instance = store.get_open_alert_instance(rule_id)
result = {"has_instance": instance is not None}
if instance is not None:
    with store.ManagedSessionMaker(read_only=True) as session:
        values = [
            float(v)
            for (v,) in session.query(SqlTraceInfo.execution_time_ms).filter(
                SqlTraceInfo.experiment_id == experiment_id,
                SqlTraceInfo.execution_time_ms.isnot(None),
                SqlTraceInfo.end_time_ms >= instance.window_start_ms,
                SqlTraceInfo.end_time_ms < instance.window_end_ms,
            ).all()
        ]
    true_avg = sum(values) / len(values) if values else None
    result.update({
        "state": instance.state,
        "observed_value": instance.observed_value,
        "sample_count": instance.sample_count,
        "window_start_ms": instance.window_start_ms,
        "window_end_ms": instance.window_end_ms,
        "true_avg": true_avg,
        "true_count": len(values),
    })

print(json.dumps(result))
""")


###############################################################################
# Steps
###############################################################################


def setup(experiment_name: str) -> tuple[int, str]:
    code = _SETUP_CODE.substitute(experiment_name=experiment_name, rule_name=EXPERIMENT_RULE_NAME)
    out = _run_in_container(code, timeout=60)
    return out["experiment_id"], out["rule_id"]


def emit_burst(experiment_id: int, rate: float, duration_s: float) -> dict:
    code = _BURST_CODE.substitute(experiment_id=experiment_id, rate=rate, duration_s=duration_s)
    return _run_in_container(code, timeout=duration_s + 60)


def poll_watermark() -> dict:
    return _run_in_container(_WATERMARK_CODE, timeout=30)


def range_check(experiment_id: int, start_ms: int, end_ms: int) -> dict:
    code = _RANGE_CHECK_CODE.substitute(
        experiment_id=experiment_id, start_ms=start_ms, end_ms=end_ms
    )
    return _run_in_container(code, timeout=60)


def alert_check(rule_id: str, experiment_id: int) -> dict:
    code = _ALERT_CHECK_CODE.substitute(rule_id=rule_id, experiment_id=experiment_id)
    return _run_in_container(code, timeout=30)


def wait_for_seal(burst_end_ms: int) -> list[dict]:
    """Poll the watermark until it catches up to the burst, printing the trend."""
    target_bucket_ms = floor_bucket(burst_end_ms - 1)
    ticks = []
    for i in range(1, WATERMARK_MAX_TICKS + 1):
        time.sleep(WATERMARK_POLL_INTERVAL_S)
        snap = poll_watermark()
        watermark_ms = snap["watermark_ms"]
        sealable_max_ms = snap["sealable_max_ms"]
        lag_buckets = (
            None if watermark_ms is None else (sealable_max_ms - watermark_ms) // BUCKET_MS
        )
        caught_up = watermark_ms is not None and watermark_ms >= target_bucket_ms
        ticks.append({**snap, "lag_buckets": lag_buckets, "caught_up": caught_up})
        elapsed = i * WATERMARK_POLL_INTERVAL_S
        print(
            f"    [watermark] t+{elapsed:3d}s  watermark_ms={watermark_ms}  "
            f"sealable_max_ms={sealable_max_ms}  lag={lag_buckets} bucket(s)"
            f"{'  <- caught up' if caught_up else ''}"
        )
        if caught_up:
            break
    return ticks


###############################################################################
# HTTP transport -- the real ingestion path
###############################################################################

DEFAULT_HTTP_WORKERS = 16
"""Concurrent SDK clients. One synchronous client spends most of its time waiting
on a round trip, so a single thread measures latency, not capacity."""


def emit_burst_http(
    experiment_name: str, rate: float, duration_s: float, workers: int = DEFAULT_HTTP_WORKERS
) -> dict:
    """Drive traces through the server the way an instrumented app does.

    Runs **on the host**, not in the container, so the network hop is real. The
    SDK's exporter batches spans and POSTs them to the OTLP endpoint
    (``/v1/traces``); the server derives the ``trace_info`` row from the batch --
    see ``SqlAlchemyStore._log_spans_once`` -- so this one path populates
    ``trace_info``, ``spans``, ``trace_metrics``, ``span_metrics`` and
    ``span_errors``.

    ``assessments`` are deliberately not driven here. A judge scores traces after
    the fact at its own cadence; emitting one per request at burst rate would be
    modelling something that does not happen. The ``assessment_value`` family is
    therefore uncovered by this transport.
    """
    import json as _json
    import random as _random
    import threading as _threading
    from concurrent.futures import ThreadPoolExecutor

    import mlflow
    from mlflow.tracing.constant import SpanAttributeKey

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    rng = _random.Random(1234)
    counter = {"written": 0, "errors": 0}
    counter_lock = _threading.Lock()
    target_total = max(1, round(rate * duration_s))
    start = time.monotonic()
    deadline = start + duration_s
    start_ms = int(time.time() * 1000)

    def one_trace(i: int) -> None:
        latency_ms = rng.randint(800, 6_000)
        with mlflow.start_span(name="handle_order") as root:
            root.set_attribute("request.index", i)
            with mlflow.start_span(name="gpt-5", span_type="LLM") as llm:
                llm.set_attribute(
                    SpanAttributeKey.CHAT_USAGE,
                    {
                        "input_tokens": rng.randint(400, 1200),
                        "output_tokens": rng.randint(150, 500),
                        "total_tokens": rng.randint(600, 1700),
                    },
                )
                llm.set_attribute(
                    SpanAttributeKey.LLM_COST,
                    _json.dumps({
                        "input_cost": 0.0012,
                        "output_cost": 0.0009,
                        "total_cost": 0.0021,
                    }),
                )
            # A minority of tool calls fail, so `span_errors` is exercised and the
            # error-rate families have both numerator and denominator.
            if rng.random() < 0.05:
                try:
                    with mlflow.start_span(name="search_docs", span_type="TOOL"):
                        raise TimeoutError("search timed out")
                except TimeoutError:
                    pass
            else:
                with mlflow.start_span(name="search_docs", span_type="TOOL"):
                    pass
            time.sleep(latency_ms / 1000 / 200)  # keep spans non-instant, cheaply

    def worker() -> None:
        while True:
            with counter_lock:
                if counter["written"] >= target_total:
                    return
                i = counter["written"]
                counter["written"] += 1
            if time.monotonic() >= deadline:
                with counter_lock:
                    counter["written"] -= 1
                return
            try:
                one_trace(i)
            except Exception:
                with counter_lock:
                    counter["errors"] += 1

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for f in [pool.submit(worker) for _ in range(workers)]:
            f.result()

    emit_elapsed = time.monotonic() - start
    # The exporter is asynchronous: without this the burst "ends" before the
    # server has seen it, and every assertion below would read a partial window.
    # Timed separately -- folding the flush into `elapsed_s` would report a rate
    # of "traces emitted per second spent emitting *and* draining", which
    # understates what the client actually sustained.
    mlflow.flush_trace_async_logging()
    flush_elapsed = time.monotonic() - start - emit_elapsed
    end_ms = int(time.time() * 1000)
    return {
        "offered": target_total,
        "written": counter["written"] - counter["errors"],
        "errors": counter["errors"],
        "start_ms": start_ms,
        "end_ms": end_ms,
        "elapsed_s": emit_elapsed,
        "flush_s": flush_elapsed,
    }


###############################################################################
# Ladder
###############################################################################


@dataclass
class AssertionResult:
    name: str
    passed: bool | None  # None means skipped
    detail: str

    def line(self) -> str:
        status = "SKIP" if self.passed is None else ("PASS" if self.passed else "FAIL")
        return f"    [assert] {self.name:<12} {status:<4} {self.detail}"


def run_level(
    experiment_id: int,
    burst_s: float,
    rate: float,
    *,
    transport: str = "http",
    experiment_name: str = "",
    workers: int = DEFAULT_HTTP_WORKERS,
) -> tuple[bool, list[AssertionResult]]:
    print(f"\n=== Level: {burst_s:g}s burst @ {rate:g} traces/sec offered [{transport}] ===")
    if transport == "http":
        burst = emit_burst_http(experiment_name, rate, burst_s, workers)
    else:
        burst = emit_burst(experiment_id, rate, burst_s)
    achieved_rate = burst["written"] / burst["elapsed_s"] if burst["elapsed_s"] > 0 else 0.0
    errors = burst.get("errors", 0)
    flush_s = burst.get("flush_s")
    print(
        f"    [emit] offered={burst['offered']} written={burst['written']} "
        f"elapsed={burst['elapsed_s']:.2f}s achieved_rate={achieved_rate:.1f}/s"
        + (f" flush={flush_s:.2f}s" if flush_s else "")
        + (f" errors={errors}" if errors else "")
    )

    print(f"    waiting for the aggregator to seal (LAG_MS={LAG_MS}ms + tick)...")
    ticks = wait_for_seal(burst["end_ms"])
    caught_up = bool(ticks) and ticks[-1]["caught_up"]

    check = range_check(experiment_id, burst["start_ms"], burst["end_ms"])
    print(
        f"    [range] [{check['range_start_ms']}, {check['range_end_ms']}) "
        f"raw={check['raw_count']} rollup={check['rollup_count']} gaps={check['gap_count']}"
    )

    results = []

    # Checked FIRST, because it is the assertion that can invalidate the rest:
    # `no_loss` below compares the database against itself and is blind to
    # traffic that never arrived. A run where the exporter dropped 90% of spans
    # would otherwise report a clean sweep.
    written = burst["written"]
    arrived = check["raw_count"]
    if written > 0:
        delivered = arrived / written
        results.append(
            AssertionResult(
                "ingest",
                delivered >= 1 - INGEST_TOLERANCE,
                f"client sent {written}, {arrived} arrived ({delivered * 100:.1f}%)",
            )
        )
    else:
        results.append(AssertionResult("ingest", None, "nothing sent"))

    no_loss = check["raw_count"] == check["rollup_count"]
    results.append(
        AssertionResult(
            "no_loss",
            no_loss,
            f"raw={check['raw_count']} rollup={check['rollup_count']} (aggregator only)",
        )
    )

    no_gaps = check["gap_count"] == 0
    results.append(AssertionResult("no_gaps", no_gaps, f"gap_buckets={check['gap_count']}"))

    p95_true = check["p95_true"]
    p95_sketch = check["p95_sketch"]
    if p95_true is None or p95_sketch is None:
        results.append(
            AssertionResult("percentile", None, "not enough samples in this burst to check")
        )
    else:
        rel_err = abs(p95_sketch - p95_true) / p95_true if p95_true != 0 else abs(p95_sketch)
        results.append(
            AssertionResult(
                "percentile",
                rel_err <= PCT_TOLERANCE + 1e-9,
                f"sketch={p95_sketch:.1f}ms true={p95_true:.1f}ms rel_err={rel_err * 100:.2f}%",
            )
        )

    last_tick = ticks[-1] if ticks else None
    results.append(
        AssertionResult(
            "watermark",
            caught_up,
            (
                f"caught up after {len(ticks)} tick(s) ({len(ticks) * WATERMARK_POLL_INTERVAL_S}s)"
                if caught_up
                else f"still {last_tick['lag_buckets'] if last_tick else '?'} bucket(s) behind "
                f"after {len(ticks)} tick(s)"
            ),
        )
    )

    for r in results:
        print(r.line())

    level_passed = all(r.passed for r in results if r.passed is not None)
    print(f"LEVEL {'PASS' if level_passed else 'FAIL'}: {burst_s:g}s @ {rate:g}/s offered")
    return level_passed, results


def run_alert_verification(rule_id: str, experiment_id: int) -> AssertionResult:
    print(f"\n=== Alerts saw the truth (rule {rule_id}) ===")
    print(
        f"    A newly created rule waits a full window before its first evaluation; "
        f"polling up to {ALERT_MAX_WAIT_S}s..."
    )
    waited = 0
    out = alert_check(rule_id, experiment_id)
    while not out["has_instance"] and waited < ALERT_MAX_WAIT_S:
        time.sleep(ALERT_POLL_INTERVAL_S)
        waited += ALERT_POLL_INTERVAL_S
        out = alert_check(rule_id, experiment_id)
        print(f"    [alert] t+{waited}s  has_instance={out['has_instance']}")

    if not out["has_instance"]:
        result = AssertionResult(
            "alerts_saw_truth", None, f"rule never fired within {ALERT_MAX_WAIT_S}s; not verified"
        )
        print(result.line())
        return result

    observed = out["observed_value"]
    true_avg = out["true_avg"]
    if observed is None or true_avg is None:
        result = AssertionResult(
            "alerts_saw_truth", None, "instance fired but has no observed_value/true_avg to compare"
        )
        print(result.line())
        return result

    rel_err = abs(observed - true_avg) / true_avg if true_avg != 0 else abs(observed)
    passed = rel_err <= 1e-6
    result = AssertionResult(
        "alerts_saw_truth",
        passed,
        f"state={out['state']} observed={observed:.4f}ms true_avg={true_avg:.4f}ms "
        f"(n={out['true_count']}) rel_err={rel_err * 100:.6f}%",
    )
    print(result.line())
    return result


###############################################################################
# Entry point
###############################################################################


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--burst", type=float, default=None, help="run a single level: burst seconds"
    )
    parser.add_argument(
        "--rate", type=float, default=None, help="run a single level: offered traces/sec"
    )
    parser.add_argument(
        "--experiment", default="load-test", help="scratch experiment name (default: load-test)"
    )
    parser.add_argument(
        "--transport",
        choices=("db", "http"),
        default="http",
        help=(
            "how traffic reaches the server. 'http' (default) drives the real ingestion "
            "path -- SDK -> OTLP -> /v1/traces -- and is what a real client does. 'db' "
            "writes rows straight into Postgres, bypassing ingestion entirely; it "
            "isolates the aggregator but cannot tell you anything about the endpoint."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_HTTP_WORKERS,
        help=f"concurrent SDK clients for --transport http (default: {DEFAULT_HTTP_WORKERS})",
    )
    parser.add_argument(
        "--skip-alert-check",
        action="store_true",
        help="skip the final alerts-saw-the-truth check (it can take several minutes)",
    )
    parser.add_argument(
        "--only-alert-check",
        action="store_true",
        help="skip the ladder and just poll the (already-created) rule for a fired instance",
    )
    args = parser.parse_args()

    if (args.burst is None) != (args.rate is None):
        parser.error("--burst and --rate must be given together")
    if args.burst is not None and args.burst > 60:
        parser.error("burst cannot exceed 60 seconds")

    _check_stack_up()

    print(f"Setting up scratch experiment {args.experiment!r} and its subscription rule...")
    experiment_id, rule_id = setup(args.experiment)
    print(f"  experiment_id={experiment_id} rule_id={rule_id}")

    if args.only_alert_check:
        result = run_alert_verification(rule_id, experiment_id)
        raise SystemExit(0 if result.passed is not False else 1)

    levels = [(args.burst, args.rate)] if args.burst is not None else DEFAULT_LEVELS

    broke_at = None
    for burst_s, rate in levels:
        passed, _results = run_level(
            experiment_id,
            burst_s,
            rate,
            transport=args.transport,
            experiment_name=args.experiment,
            workers=args.workers,
        )
        if not passed:
            broke_at = (burst_s, rate)
            break

    if not args.skip_alert_check:
        run_alert_verification(rule_id, experiment_id)

    print("\n=== Summary ===")
    if broke_at is not None:
        print(
            f"Ladder stopped at {broke_at[0]:g}s @ {broke_at[1]:g} traces/sec offered: "
            "assertion failed."
        )
        raise SystemExit(1)
    print(f"All {len(levels)} level(s) passed every assertion.")


if __name__ == "__main__":
    main()
