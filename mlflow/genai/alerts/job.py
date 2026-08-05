"""Periodic entry points for alerting.

Two jobs run every minute: the aggregator seals the bucket that just closed, and
the evaluator compares due rules against the rollups. They are separate tasks on
purpose -- a slow evaluation must not delay sealing, since a bucket that misses
its window is wrong permanently rather than merely late.
"""

import logging
import threading

from mlflow.environment_variables import (
    MLFLOW_ALERT_EVALUATOR_THREADS,
)
from mlflow.genai.alerts.evaluator import AlertEvaluator, EvaluationCycle
from mlflow.genai.alerts.sql_rollup_reader import SqlRollupReader

_logger = logging.getLogger(__name__)

_evaluator: AlertEvaluator | None = None
_evaluator_lock = threading.Lock()

MAX_EVALUATOR_THREADS = 8
"""Ceiling on ``MLFLOW_ALERT_EVALUATOR_THREADS``.

The SQLAlchemy pool defaults to ``pool_size=5, max_overflow=10`` -- 15 connections
for the whole process, shared with the periodic-task consumer's 5 threads and every
``alert_rollup_*`` task. Past roughly this many evaluator threads the pool starts
handing back timeouts, which surface as TEMPORARILY_UNAVAILABLE and cost the entire
cycle rather than slowing it down.
"""


def _evaluator_threads(db_type: str) -> int:
    """Threads for this process, clamped to what the backend can take.

    SQLite permits one writer, and each rule evaluated is two writes; K threads
    against it exhaust ``busy_timeout`` rather than going faster.
    """
    from mlflow.store.db.db_types import SQLITE

    requested = MLFLOW_ALERT_EVALUATOR_THREADS.get()
    if db_type == SQLITE and requested > 1:
        _logger.warning(
            "MLFLOW_ALERT_EVALUATOR_THREADS=%d ignored: SQLite allows a single writer, "
            "so concurrent evaluation would exhaust the busy timeout rather than "
            "finish sooner. Using 1.",
            requested,
        )
        return 1
    return max(1, min(requested, MAX_EVALUATOR_THREADS))


def _get_evaluator() -> AlertEvaluator:
    """One long-lived evaluator per process.

    The incremental-merge cache is this object's state, so rebuilding it every
    tick would silently discard the optimization it exists to provide -- a 24h
    rule would go back to reading 1,440 buckets a cycle.

    Built under a lock because the unguarded ``if _evaluator is None`` let two
    threads each build one, and the loser's cache -- along with everything it had
    warmed -- was silently discarded.
    """
    global _evaluator
    if _evaluator is not None:
        return _evaluator
    with _evaluator_lock:
        if _evaluator is not None:
            return _evaluator
        # Imported lazily: this module is reachable from the tracking package, and
        # resolving the store at import time would pull the server stack into a
        # client install.
        from mlflow.genai.alerts.notifications import dispatch
        from mlflow.server.handlers import _get_tracking_store

        # The SERVER-side backend store, not the client one. `_tracking_service`'s
        # `_get_store()` honours MLFLOW_TRACKING_URI and hands back a RestStore
        # inside the server process, which has no session maker -- the evaluator
        # then dies with "'RestStore' object has no attribute 'ManagedSessionMaker'".
        # This is the accessor every other periodic task uses.
        store = _get_tracking_store()
        _evaluator = AlertEvaluator(
            SqlRollupReader(store),
            store,
            notifier=dispatch,
            max_workers=_evaluator_threads(store.db_type),
        )
        return _evaluator


def run_alert_evaluation(now_ms: int | None = None) -> EvaluationCycle:
    """Evaluate every rule that is due. Zero-arg, for the scheduler."""
    return _get_evaluator().run_once(now_ms=now_ms)


def reset_evaluator() -> None:
    """Drop the cached evaluator. For tests, and after a config change."""
    global _evaluator
    with _evaluator_lock:
        _evaluator = None
