"""Alerting on agent traces.

An alert rule is a saved metric query plus a predicate. MLflow already computes
every number an alert needs; this package adds the rollup layer that makes long
windows affordable, a scheduler, and a state machine.
"""

from mlflow.genai.alerts.entities import (
    METRIC_CATALOGUE,
    AlertInstance,
    AlertRule,
    Observation,
    SeriesKey,
    Transition,
    derive_evaluation_interval_seconds,
    derive_min_sample_count,
)
from mlflow.genai.alerts.rollup_reader import Bucket, FakeRollupReader, RollupReader

__all__ = [
    "METRIC_CATALOGUE",
    "AlertInstance",
    "AlertRule",
    "Bucket",
    "FakeRollupReader",
    "Observation",
    "RollupReader",
    "SeriesKey",
    "Transition",
    "derive_evaluation_interval_seconds",
    "derive_min_sample_count",
]
