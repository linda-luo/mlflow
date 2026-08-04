# GenAI Alerting

An alert rule is a saved metric query plus a predicate: _"p95 latency over the last
10 minutes, above 30 minutes, for the `search_docs` tool."_ MLflow already computes
every number a rule needs. This package adds what makes asking continuously
affordable — a rollup layer, a scheduler, and a state machine.

## The problem it solves

Evaluating _"p95 latency over 24 hours"_ every minute by scanning traces means
re-reading ~86,000 rows a minute, forever. Instead, a background aggregator seals
one-minute buckets of counts, sums and histograms, and rules read those. Roughly
60 traces per minute become **10 rollup rows**, and a 24-hour window reads 1,440
small rows rather than 86,000 traces.

## Flow

```
trace_info ─┐
spans ──────┤
assessments ┼─► aggregator ──► metric_rollups ──► metric_rollups_1h
span_errors ┤    (20 tasks,     (1-min buckets)    (continuous aggregate)
metrics ────┘     1/minute)            │
                                       ▼
                                   evaluator ──► state machine ──► alert_instances
                                   (1/minute)                          │
                                                                       ▼
                                                                  notifications
```

## Modules

| File                   | Role                                                                |
| ---------------------- | ------------------------------------------------------------------- |
| `entities.py`          | `AlertRule`, `AlertInstance`, the metric catalogue, window bounds     |
| `aggregator.py`        | Seals raw rows into 1-minute buckets. The 20 periodic work units      |
| `subscriptions.py`     | Which `(experiment, family)` pairs anything actually reads            |
| `rollup_reader.py`     | `Bucket`, and `project_aggregate` — the one projection                |
| `sql_rollup_reader.py` | The Postgres/Timescale reader, incl. tier stitching                   |
| `tiers.py`             | Which tier answers a window; `plan_cover`                             |
| `evaluator.py`         | Claims due rules, folds windows, decides, transitions                 |
| `state_machine.py`     | Pure `(rule, instance, observation) → transition`                     |
| `sketch.py`            | DDSketch spec (α = 2%)                                                |
| `histogram.py`         | Sparse histogram merge/subtract/quantile                              |
| `series.py`            | Rolling-window values for the alert detail chart                      |
| `exemplars.py`         | The traces behind an alert, captured at fire time                     |
| `span_errors.py`       | Error extraction on the `log_spans` ingest path                       |
| `timescale.py`         | Hypertable, `hist_merge`, the hourly tier, retention                  |
| `notifications.py`     | Dispatch, after the instance is committed                             |
| `job.py`               | Wiring for the periodic-task scheduler                                |

## Six things that are easy to get subtly wrong

**Windows derive from the watermark, never from the clock.** The aggregator holds
the newest bucket back by `LAG_MS` (60s) for late arrivals, so the readable edge
trails wall time. Anything that asks about unsealed time gets windows containing
fewer and fewer sealed buckets — and a `COUNT` over one of those is a real,
shrinking number, which draws as a smooth collapse to zero that never happened.

**`project_aggregate` is the only projection.** Two copies of it once disagreed
about the empty window, and the copy the evaluator used was the wrong one: an
absence rule read as insufficient-data at exactly the moment it should fire, so it
could never fire at all. Anything that turns folded aggregates into a number goes
through that function.

**An empty COUNT window is zero; an empty average is not.** That split is what
makes _absence_ expressible ("traffic dropped", "requests < 10"). It is also why
gaps are stored separately from zero counts — `is_gap` means aggregation did not
run, which can neither open nor close an instance.

**Every stored field is invertible.** Counts, sums and sparse histograms all
subtract, which is what lets `WindowAccumulator` maintain a running fold and shed
the expiring edge instead of re-reading the window. It is also why `min`/`max` are
not stored.

**Percentiles merge, they do not average.** The p95 of ten minutes is not the mean
of ten per-minute p95s. Each bucket keeps a DDSketch; merging is "union the
indices, add the counts", exact, and the percentile is read once from the merged
result. Error stays at α whether you merge ten buckets or four thousand.

**The longest window equals raw retention, by construction.**
`MAX_WINDOW_SECONDS * 1000 == RAW_RETENTION_MS`. The incremental path reads the
expiring edge at one-minute granularity, so a window reaching past retention reads
an empty range, subtracts nothing, and silently stops shedding. There is a test
pinning the two constants together.

## Concurrency

One process, one consumer. Cycles are serialized by `huey.lock_task`, and within a
cycle a `ThreadPoolExecutor` fans **read groups** — not rules — across threads, so
two threads never touch the same rule or the same cache entry. Threads are opt-in
via `MLFLOW_ALERT_EVALUATOR_THREADS`, and clamped to 1 on SQLite, where a single
writer means K threads exhaust the busy timeout rather than finishing sooner.

There is deliberately no cross-replica coordination. Alerting previously carried
leases, sticky hash assignment and `SKIP LOCKED` for a deployment shape nothing
else in MLflow supports; that was removed so this package matches every other
periodic task.

## Testing

```bash
uv run pytest tests/genai/alerts/                       # unit + store
uv run --with httpx2 pytest tests/server/test_alert_handlers.py
```

Most of it runs on SQLite against `FakeRollupReader`, with no Timescale and no
scheduler — fast, and enough for the state machine, the projection and the
window arithmetic. What it cannot cover is everything Postgres-only: continuous
aggregates, the hourly tier, the periodic jobs, and the `numeric` vs `bigint`
divergence between tiers that once broke every percentile rule spanning an hour.

## Running it for real — `dev/alerting/`

A Docker stack with real Postgres + TimescaleDB, a real scheduler and real
traffic. It exists because two things cannot be exercised on Windows: the
periodic aggregator/evaluator (MLflow's job backend refuses to start there) and
Timescale itself.

```bash
docker compose -f dev/alerting/compose.yml up --build -d
```

Three services: `timescaledb` (host port 5433), `mlflow` (the tracking server on
host port 5002, with job execution and the alert scheduler on), and `traffic` (a
generator that writes healthy traces into a `checkout-agent` experiment, then
degrades them after a few minutes so a rule crosses green → `PENDING` → `FIRED`
on its own).

**The container serves the API only** — it is a source install with no built
frontend, so `:5002` has no UI. For that, run the dev server and point it at the
container:

```bash
cd mlflow/server/js && MLFLOW_PROXY=http://localhost:5002/ PORT=3001 corepack yarn start
```

Then open <https://localhost:3001/#/experiments/1/alerts> — note **https**, and
it binds IPv6, so `localhost` works where `127.0.0.1` does not.

`bootstrap.py` runs on startup and is idempotent. It creates `checkout-agent`
and five rules, one per rule shape:

| Rule                       | Metric                                 | Demonstrates                    |
| -------------------------- | -------------------------------------- | ------------------------------- |
| Checkout p95 latency       | `latency` p95 (`TRACES`)               | percentile rules, sketch-backed |
| Average latency regression | `latency` avg (`TRACES`)               | plain avg threshold             |
| `search_docs` failures     | `error_count` (`SPAN_NAME`)            | per-tool error counting         |
| Safety judge pass rate     | `assessment_value` avg (`ASSESSMENTS`) | judge / quality signals         |
| Traffic dropped            | `latency` count (`TRACES`) `< 5`       | absence as a signal             |

A freshly created rule reports "No data yet" for one whole window before its
first evaluation — deliberate, since it must not conclude anything from history
it was not recording. Budget for that when demoing, or use the seeded rules.

### Load test — `dev/alerting/load_test/`

Pushes the pipeline past what the demo generator produces and checks correctness
against the database rather than the UI. Uses its own scratch experiment, so the
five rules above stay interpretable.

```bash
uv run dev/alerting/load_test/run.py                       # full ladder
uv run dev/alerting/load_test/run.py --burst 10 --rate 40   # one level, ~3 min
uv run dev/alerting/load_test/run.py --transport db         # bypass ingestion
```

Bursts of 10s, 30s then 60s at rising rates, escalating only while the previous
level passed, never longer than 60s. After each, it waits out the sealing lag
and asserts:

- **ingest** — what the client sent actually arrived. Checked first, because
  every assertion below compares the database against itself and is blind to
  traffic that never showed up.
- **no_loss** — rollup counts equal the raw rows behind them, exactly.
- **no_gaps** — nothing got gap-marked.
- **percentile** — the sketch's p95 is within α of the true p95.
- **watermark** — the aggregator catches back up, with the lag printed per tick
  so the trend is visible rather than just the final number.

`--transport http` (default) drives the real path: the MLflow SDK, OTLP, and
`POST /v1/traces`. `--transport db` writes rows straight into Postgres, which
isolates the aggregator from the endpoint but says nothing about what the server
can accept.

Two things the HTTP transport cannot do, by construction. It **cannot fabricate
latency** — a span's duration is real elapsed time, so latency-*threshold* rules
cannot be exercised through it. And it **does not drive assessments**, since a
judge scores traces on its own cadence rather than once per request, leaving the
`assessment_value` family uncovered.

Reading the output: `achieved_rate` is how fast the client *submitted*; `flush`
is how long the SDK's exporter took to *drain*. They are kept apart on purpose —
a burst can submit in two seconds and take thirteen to deliver, and adding them
together reports a rate neither number means. A `watermark lag` of 1 bucket is
the healthy steady state (`LAG_MS` holds the newest bucket back); what matters
is whether it grows.
