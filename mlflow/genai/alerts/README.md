# GenAI Alerting

> **📐 Design board:**
> https://www.figma.com/board/kINg0ytExVJ8U9f9YomsGV/Alerting---Progress?node-id=65-316&t=0cBcwiF5uAdP2qy0-1
>
> That above link is the **primary source of documentation** for how the system flows and
> why it is built the way it is. Please navigate to the "8/3 - Final" page, which contains the latest architecture diagram and the design decisions.
>Read this README for supported features, build and test instructions.

## Overview

An alert rule is a saved metric query plus a predicate: _"p95 latency over the last
30 minutes, above 3 seconds, for the `search_docs` tool."_ MLflow already computes
every number a rule needs. This package adds what makes asking continuously
affordable — a rollup layer, a scheduler, and a state machine.

Evaluating _"p95 latency over 24 hours"_ every minute by scanning traces means
re-reading ~86,000 rows a minute, forever. Instead, a background aggregator seals
one-minute buckets of counts, sums and histograms, and rules read those. Roughly
60 traces per minute become **10 rollup rows**, and a 24-hour window reads 1,440
small rows rather than 86,000 traces.

See the [design board](https://www.figma.com/board/kINg0ytExVJ8U9f9YomsGV/Alerting---Progress?node-id=65-316&t=0cBcwiF5uAdP2qy0-1)
for the architecture diagram.

The modules worth knowing, in pipeline order:

| File                   | Role                                                                    |
| ---------------------- | ----------------------------------------------------------------------- |
| `entities.py`          | The vocabulary — `AlertRule`, `AlertInstance`, and `METRIC_CATALOGUE`    |
| `span_errors.py`       | Extracts errors on the `log_spans` ingest path                           |
| `aggregator.py`        | Seals raw rows into 1-minute buckets. The 20 periodic work units         |
| `sql_rollup_reader.py` | Reads those buckets back, stitching the hourly tier in where it fits     |
| `evaluator.py`         | Claims due rules, folds each window, decides whether it breached         |
| `state_machine.py`     | Opens, fires, updates and closes alert instances                         |
| `sketch.py`            | The DDSketch grid (α = 2%) that makes percentiles mergeable              |

The rest are supporting: `subscriptions.py`, `rollup_reader.py`, `tiers.py`,
`histogram.py`, `series.py`, `exemplars.py`, `timescale.py`, `notifications.py`
and `job.py`.

## Supported features

A rule is one sentence: **when _[aggregation]_ of _[metric]_ for _[scope]_ over
_[window]_ is _[above/below]_ _[threshold]_, and stays that way for _[duration]_.**

| Slot           | Options                                                                                   |
| -------------- | ----------------------------------------------------------------------------------------- |
| **metric**     | latency · 5 token counts · 3 cost measures · error count · error rate · judge score        |
| **scope**      | whole trace · span type · span name · model · exception type · judge — one value, or all   |
| **aggregation**| `COUNT` · `SUM` · `AVG` · `PERCENTILE` (any percentile)                                    |
| **window**     | 5 minutes to 3 days, rolling                                                               |
| **comparator** | `>` · `>=` · `<` · `<=`, against any threshold                                             |
| **sustain**    | fire on the first breach, or require the condition to hold for N seconds                   |

Not every combination is valid — `METRIC_CATALOGUE` in `entities.py` is the single
source of truth for both the form's dropdowns and the server's validation, so a
rule the evaluator could not answer is rejected at creation rather than silently
never firing.

**Both comparator directions are supported**, so the same grammar expresses
"latency is too high" and "traffic has stopped" — the second being the one rule
whose entire signal is an empty window.

**Percentiles are exact where it counts.** A DDSketch per bucket gives 2% relative
accuracy at every magnitude, but the firing decision restates `p95 > T` as a
counting question answered from exact bucket counts, falling back to raw rows only
when the bounds straddle the threshold.

Evaluation frequency is derived, not configured — `window / 10`, clamped to 1–5
minutes, so consecutive windows overlap and no breach slips between evaluations.

Alerts do not self-resolve. A fired instance becomes `INACTIVE` once the metric
recovers, stays on screen, and is closed only by a person — so a spike that
recovered before anyone looked is still visible afterwards.

In the UI: create, edit, enable/disable and delete rules; view active and
historical instances; dismiss one; and open a detail view with a rolling-window
chart and the trace IDs captured when the alert fired.

**TimescaleDB is an optional cold-cache optimization.** When the evaluator has to
rebuild a whole window from scratch, it reads 1-hour rollups for the whole hours
inside that window, so a 24-hour read no longer has to pull 1,440 single-minute
rows. The system works as-is without it on plain Postgres; hosts can add it for
faster cold starts.

## Build

Everything runs in-process with the tracking server — no separate service. For a
realistic environment (Postgres + TimescaleDB, a real scheduler, real traffic):

```bash
docker compose -f dev/alerting/compose.yml up --build -d
```

It exists because two things cannot be exercised on Windows: the periodic
aggregator/evaluator (MLflow's job backend refuses to start there) and Timescale
itself.

Three services — `timescaledb` (host port 5433), `mlflow` (tracking server on host
port 5002, with job execution and the alert scheduler on), and `traffic` (writes
healthy traces into a `checkout-agent` experiment, then degrades them so a rule
crosses green → `PENDING` → `FIRED` on its own).

**The container serves the API only** — a source install with no built frontend, so
`:5002` has no UI. For that, run the dev server against it:

```bash
cd mlflow/server/js && MLFLOW_PROXY=http://localhost:5002/ PORT=3001 corepack yarn start
```

Then open <https://localhost:3001/#/experiments/1/alerts> — note **https**, and it
binds IPv6, so `localhost` works where `127.0.0.1` does not.

`bootstrap.py` runs on startup, is idempotent, and seeds five rules covering the
distinct rule shapes: percentile, plain average, per-tool error count, judge score,
and absence. A freshly created rule reports "No data yet" for one whole window
before its first evaluation — deliberate, since it must not conclude anything from
history it was not recording. Budget for that when demoing, or use the seeded rules.

## Test

```bash
uv run pytest tests/genai/alerts/                                # unit + store
uv run --with httpx2 pytest tests/server/test_alert_handlers.py  # REST handlers
```

Most of it runs on SQLite against `FakeRollupReader`, with no Timescale and no
scheduler — fast, and enough for the state machine, the projection and the window
arithmetic. What it cannot cover is everything Postgres-only: continuous
aggregates, the hourly tier, the periodic jobs, and the `numeric` vs `bigint`
divergence between tiers that once broke every percentile rule spanning an hour.

For those, the Docker stack above plus the load test:

```bash
uv run dev/alerting/load_test/run.py                        # full ladder
uv run dev/alerting/load_test/run.py --burst 10 --rate 40   # one level, ~3 min
uv run dev/alerting/load_test/run.py --transport db         # bypass ingestion
```

Bursts of 10s, 30s then 60s at rising rates, escalating only while the previous
level passed. After each it waits out the sealing lag and asserts **ingest** (what
was sent arrived — checked first, since every later assertion compares the database
against itself), **no_loss** (rollup counts equal the raw rows exactly), **no_gaps**,
**percentile** (within α of the true p95), and **watermark** (the aggregator catches
back up).

Two things the HTTP transport cannot do: it cannot fabricate latency — a span's
duration is real elapsed time — so latency-threshold rules are not exercised
through it, and it does not drive assessments, leaving the `assessment_value`
family uncovered.
