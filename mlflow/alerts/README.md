# GenAI Alerting

> **📐 Design board:**
> https://www.figma.com/board/kINg0ytExVJ8U9f9YomsGV/Alerting---Progress?node-id=65-316&t=0cBcwiF5uAdP2qy0-1
>
> The above link is the **primary source of documentation** for how the system flows and
> why it is built the way it is. Please navigate to the "8/3 - Final" page, which contains the latest architecture diagram and the design decisions.
>Read this README primarily for build and test instructions.

## Build

Everything runs in-process with the tracking server — no separate service. The
Docker stack below gives a realistic environment (Postgres + TimescaleDB, a real
scheduler, real traffic). It exists because two things cannot be exercised on
Windows: the periodic aggregator/evaluator (MLflow's job backend refuses to start
there) and Timescale itself.

**Run every command from the repository root**, except where a step says otherwise.

**The commands are POSIX shell**, so Git Bash or WSL is the smoothest path on
Windows. PowerShell works too, but it has no `grep` and rejects the
`VAR=value command` prefix; where that matters, the PowerShell form is given
alongside.

### Prerequisites

| Need | Why | Check |
| ---- | --- | ----- |
| Docker Desktop, running | the whole stack | `docker ps` |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | Python entry points | `uv --version` |
| Node ≥ 24.14 with corepack | only for the UI | `node --version` |
| Ports 5002, 5433, 3001 free | server, Postgres, UI | — |

### 1. Start the backend

```bash
docker compose -f dev/alerting/compose.yml up --build -d
```

If you have run this before and want a clean slate, delete the database first —
otherwise the old traces and rules are still there and `bootstrap.py` skips
re-seeding:

```bash
docker compose -f dev/alerting/compose.yml down -v
docker compose -f dev/alerting/compose.yml up --build -d
```

The first build pulls images and installs MLflow from source — allow several
minutes. Wait until both long-running services report `healthy`:

```bash
docker compose -f dev/alerting/compose.yml ps
```

Three services come up in order: `timescaledb` (host port 5433), `mlflow` (the
tracking server on host port 5002, with job execution and the alert scheduler on),
and `traffic`, which waits for the server's healthcheck before starting — so it is
normal for `traffic` to appear a minute after the other two.

SUGGESTED: On startup `bootstrap.py` creates the Timescale objects and a `checkout-agent`
experiment.
Additionally, for demo purposes it will seed five rules covering the distinct rule shapes: percentile, plain average, per-tool error count, judge score, and absence. It is idempotent, so
restarting the stack does not duplicate anything. Confirm it ran:

```bash
docker compose -f dev/alerting/compose.yml logs mlflow | grep bootstrap
# PowerShell: ... logs mlflow | Select-String bootstrap
```

You should see five `[bootstrap] rule '…'` lines.

ALTERNATIVELY: To start with **no demo rules** — for evaluating the rule editor from an empty
page — bring the stack up with:

```bash
DEMO_SEED_RULES=0 docker compose -f dev/alerting/compose.yml up --build -d
# PowerShell: $env:DEMO_SEED_RULES = "0"; docker compose -f dev/alerting/compose.yml up --build -d
```

Migrations, the Timescale tiers and the `checkout-agent` experiment still run;
only the five rules are skipped. Note that aggregation is demand-driven, so with
no rules the aggregator writes no rollups at all — buckets start being sealed for
a family once the first rule reading it exists.

Deleting the rules in the UI is *not* equivalent: bootstrap re-seeds them on the
next restart, because soft-deleted rules do not come back from `list_alert_rules`.

### 2. Open the UI

**The container serves the API only** — it is a source install with no built
frontend, so `:5002` has no UI. Run the dev server against it, in a **second
terminal** (it stays in the foreground), from `mlflow/server/js`:

```bash
cd mlflow/server/js
corepack yarn install    # first time only; several minutes
MLFLOW_PROXY=http://localhost:5002/ PORT=3001 corepack yarn start
```

In PowerShell the last line is instead:

```powershell
$env:MLFLOW_PROXY = "http://localhost:5002/"; $env:PORT = "3001"; corepack yarn start
```

`yarn install` is required — `node_modules` is not checked in, and `yarn start`
fails without it. This `cd` applies to that terminal only; every later step is
again from the repository root, so run them elsewhere or `cd` back — otherwise
`-f dev/alerting/compose.yml` resolves against `mlflow/server/js` and Docker
reports the file missing.

Then open <https://localhost:3001/#/experiments/1/alerts>. Note **https**, and the
dev server binds IPv6, so `localhost` works where `127.0.0.1` does not. The
certificate is self-signed; accept the browser warning.

### 3. Watch an alert fire

If you chose to have the bootstrap seed the 5 rules, you can go view in the UI which should now be online.
The traffic generator is healthy for its first **2 minutes**
(`DEMO_HEALTHY_MINUTES` in `compose.yml`), then degrades. The seeded rules have a 5-minute
window -- reminder that a new rule waits one whole window before its first evaluation is valid. So the page shows **"No data yet" for the first ~5 minutes**, which is
correct rather than broken.

Expect the first `FIRED` alert roughly **6 minutes** after `up` (measured on a
clean bring-up). To watch progress without the UI:

```bash
docker compose -f dev/alerting/compose.yml logs -f traffic
```

which prints one line per tick and announces the switch to `DEGRADED`.

On start, before the alerts have gathered enough data:
![Alerts page shortly after startup: every rule shows "No data yet"](../../../dev/alerting/alerts-no-data.png)

Expected behavior of 5 bootstrap seeded rules:
- Three rules fire on their first breaching evaluation. 
- `Checkout p95 latency` shows "Confirming" (`PENDING`) a couple of minutes longer: it is the only rule with a *sustain*, so the breach must hold for 120s before it fires. It still measures over a 5 minute window, but the condition over that rolling window must be true for 120s. 
- `Traffic dropped`
stays green for as long as traffic keeps flowing — it is the absence rule, and its
signal is an *empty* window. A green `Traffic dropped` means the demo is working,
not that it is stuck.

On finish (note you must refresh to see the updated status):
![Alerts page after the agent degrades: three rules fired, one confirming, the absence rule still green](../../../dev/alerting/alerts-fired.png)

### 4. Look at the tables (optional)

Open a psql shell on the demo's Postgres:

```bash
docker compose -f dev/alerting/compose.yml exec timescaledb psql -U mlflow -d mlflow
```

`\dt` lists the tables, `\x` toggles row-per-line output, `\q` quits.

```sql
SELECT * FROM alert_rules;      -- the five seeded rules
SELECT * FROM alert_instances;  -- firing episodes, one per incident
SELECT * FROM rollup_state;     -- the 12 aggregator work units and their watermarks
SELECT * FROM metric_series;    -- one row per series being aggregated
SELECT * FROM metric_rollups ORDER BY bucket_start_ms DESC LIMIT 10;  -- the sealed buckets
```

That last table is the point of the whole design: a minute of traffic becomes one
row per series, and rules read those instead of the traces.

### 5. Turn up the volume (optional)

The default rate is deliberately low — about 1 trace/sec, so a rule firing is easy
to follow. To drive the seeded rules under real volume, raise it and shorten the
healthy period:

```bash
DEMO_TRACES_PER_SECOND=50 DEMO_HEALTHY_MINUTES=1 \
  docker compose -f dev/alerting/compose.yml up -d traffic --force-recreate
```

```powershell
$env:DEMO_TRACES_PER_SECOND = "50"; $env:DEMO_HEALTHY_MINUTES = "1"
docker compose -f dev/alerting/compose.yml up -d traffic --force-recreate
```

Only the `traffic` container restarts; the server and database keep running.
`logs -f traffic` reports the achieved count per tick.

Rerun the `metric_rollups` query above afterwards: the row count per minute stays
the same, only the `count` and `sum` inside each row grow. That is the compression
the rollup layer exists for.

This writes **straight into Postgres, skipping the server**, so it populates the
demo but proves nothing about what ingestion can take — that is the aggregation
load test's job. It is also the only generator that emits *degraded* traffic,
which is what makes the latency rules breach at all.

## Test

Three things produce traffic or verdicts, and they are not interchangeable:

| | Writes via | Use it for |
| --- | --- | --- |
| **Unit tests** (`pytest`) | in-process SQLite, no Docker | the logic — state machine, projection, window arithmetic |
| **Demo traffic** (step 4 above) | straight into Postgres, **skipping the server** — traces, assessments and token metrics as direct inserts; spans via `log_spans`. Avoids needing real trace logging and a judge | making the seeded rules fire, and the UI look like a live system |
| **Aggregation load test** (`load_test/run.py`) | the real endpoint — SDK → OTLP → `POST /v1/traces` | whether the aggregator keeps up, and seals correctly, under volume |

### Unit tests

```bash
uv sync                                                          # first time only
uv run pytest tests/alerts/                                # unit + store
uv run --with httpx2 pytest tests/server/test_alert_handlers.py  # REST handlers
```

These two commands need no Docker and no database of their own. If an import
fails, install the wider test dependencies —
`uv pip install -r requirements/test-requirements.txt` — though the alerting tests
themselves need only pytest and OpenTelemetry.

Most of it runs on SQLite against `FakeRollupReader`, with no Timescale and no
scheduler — fast, and enough for the state machine, the projection and the window
arithmetic. What it cannot cover is everything Postgres-only: continuous
aggregates, the hourly tier, the periodic jobs, and the `numeric` vs `bigint`
divergence between tiers that once broke every percentile rule spanning an hour.

### Optional: Aggregation load test

**This is a nascent form of testing and would need more time to validate** 
Testing against one burst of 500 traces, not a load
profile. This would be fleshed out with more time.

The thing under load is the **aggregator** — whether it seals every bucket
correctly and keeps its watermark current while traces arrive faster than the demo
produces them. Unlike the demo traffic, this sends through the real endpoint and
then **checks the database against what it sent**, printing PASS/FAIL per check and
exiting non-zero if any fail. It `docker compose exec`s into the running `mlflow`
container, so **the stack must already be up**.

```bash
uv run dev/alerting/load_test/run.py                   # one burst, ~4 min
uv run dev/alerting/load_test/run.py --transport db    # same, but skip ingestion
```

`--transport db` writes rows straight into Postgres instead of going through the
server, so a failure is unambiguously the aggregator's rather than the client's.

**What it creates.** A scratch experiment named `load-test` (`--experiment` to
change it) and one rule inside it, `load-test: AVG latency subscription`. The rule
is not decoration: aggregation is demand-driven, so without it nothing would be
aggregated for the new experiment at all. Its threshold is `-1`, so it always
breaches — that is what lets the harness compare a fired instance's
`observed_value` against the raw rows.

**Where to see it.** In the UI at `/#/experiments/2/alerts` — a separate page from
the demo. In the database, everything is scoped to that experiment:

```sql
SELECT experiment_id, name FROM experiments;                  -- the new 'load-test' row
SELECT * FROM metric_series WHERE experiment_id = 2;          -- its own series
SELECT * FROM alert_rules WHERE experiment_id = 2;            -- the one subscription rule
```

**What the output looks like** — one block per level, then a verdict:

```
=== Level: 10s burst @ 40 traces/sec offered [http] ===
    [emit] offered=400 written=400 elapsed=1.5s achieved_rate=264/s flush=17s
    [watermark] t+ 10s  lag=1 bucket(s)
    [assert] ingest       PASS client sent 400, 400 arrived (100.0%)
    [assert] no_loss      PASS raw=400 rollup=400
    [assert] no_gaps      PASS gap_buckets=0
    [assert] percentile   PASS sketch=110.0ms true=108.0ms rel_err=1.80%
    [assert] watermark    PASS caught up after 6 tick(s)
LEVEL PASS: 10s @ 40/s offered
```

`achieved_rate` is how fast the client *submitted*; `flush` is how long the SDK's
exporter took to *drain*. They are kept apart deliberately — a burst can submit in
two seconds and take seventeen to deliver. A `watermark lag` of 1 bucket is the
healthy steady state; what matters is whether it grows.

Expect latency in the tens of milliseconds, not the 20–70 minutes the demo
fabricates, for the reason below.

A level that fails **ingest** while passing the rest means the client could not
push traces fast enough, not that the pipeline lost them — raise `--workers`.

**The assertions**, after each level: **ingest** (what was sent arrived — checked
first, since every later assertion compares the database against itself),
**no_loss** (rollup counts equal the raw rows exactly), **no_gaps**, **percentile**
(within α of the true p95), and **watermark** (the aggregator catches back up).
Then once at the end, **alerts_saw_truth** — that a fired instance's
`observed_value` matches the raw rows it was computed from.

## Troubleshooting

| Symptom | Cause and fix |
| ------- | ------------- |
| Page says "No data yet" | Expected for the first ~5 minutes. A rule waits one full window before its first evaluation. |
| `traffic` container missing from `ps` | It waits for the server's healthcheck. Give it a minute, then `docker compose -f dev/alerting/compose.yml ps -a`. |
| Browser refuses <https://localhost:3001> | Self-signed certificate — accept the warning. Use `localhost`, not `127.0.0.1`: the dev server binds IPv6. |
| `yarn start` fails immediately | `corepack yarn install` was skipped, or Node is older than 24.14. |
| Port already allocated | Something owns 5002, 5433 or 3001. Stop it, or edit the port mappings in `dev/alerting/compose.yml`. |
| Alerts never fire | Check the traffic generator is actually writing: `docker compose -f dev/alerting/compose.yml logs traffic`. |
| Schema errors after pulling changes | The volume holds the old schema. `docker compose -f dev/alerting/compose.yml down -v` then `up --build -d` — this deletes the demo data, which `bootstrap.py` recreates. |

To stop everything, keeping the data:

```bash
docker compose -f dev/alerting/compose.yml down
```

Add `-v` to delete the database as well.
