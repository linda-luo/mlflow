"""The traces behind an alert.

Rollups are counts, sums and histograms -- enough to decide whether a rule fires
and nowhere near enough to explain why. The one thing an alert needs that
aggregation deliberately discards is *which requests* were bad, so these ids come
straight from the source table.

They are captured at the moment an instance fires and stored on it, not looked up
when someone opens the alert. Two reasons: the window has usually rolled past by
the time anyone looks, and traces are archived on their own schedule -- the
evidence has to outlive them, which is what ``AlertInstance.exemplar_trace_ids``
is for.

**Best effort, always.** This runs on the fire path, and evidence is worth less
than the alert. Every failure here is swallowed: a rule that fires with no
exemplars is a small loss, one that fails to fire because an exemplar query
errored is the thing alerting exists to prevent.
"""

import logging

from mlflow.genai.alerts.aggregator import build_work_units
from mlflow.genai.alerts.entities import AlertRule

_logger = logging.getLogger(__name__)

MAX_EXEMPLARS = 5
"""Enough to see a pattern, few enough to read. Not a sample -- the *worst* few."""


def _is_worst_first(comparator: str):
    """Order that puts the most damning row first.

    Follows the comparator, exactly as ``state_machine.is_worse`` does for
    ``peak_value``: for a ``LT`` rule on a pass rate, the worst row is the
    *lowest*, and sorting descending would list the best ones.
    """
    return comparator in ("LT", "LTE")


def collect_exemplar_trace_ids(
    store,
    rule: AlertRule,
    window_start_ms: int,
    window_end_ms: int,
    limit: int = MAX_EXEMPLARS,
) -> list[str]:
    """Trace ids for the worst rows in the window that made ``rule`` fire.

    Returns ``[]`` rather than raising, for anything at all -- an unmatched
    source, a value-less metric, a database error. See the module docstring.
    """
    try:
        return _collect(store, rule, window_start_ms, window_end_ms, limit)
    except Exception:
        _logger.debug(
            "Could not collect exemplar traces for rule %s; the alert is unaffected.",
            rule.alert_rule_id,
            exc_info=True,
        )
        return []


def _collect(
    store,
    rule: AlertRule,
    window_start_ms: int,
    window_end_ms: int,
    limit: int,
) -> list[str]:
    unit = next(
        (
            u
            for u in build_work_units(store.db_type)
            if rule.metric_key in u.metric_keys and u.grouping.dimension_key == rule.dimension_key
        ),
        None,
    )
    if unit is None:
        return []

    source = unit.source
    if source.trace_id_column is None:
        return []

    # `time_scale` because spans store nanoseconds while the window is in millis --
    # the same conversion the aggregator's scan does.
    lo = window_start_ms * source.time_scale
    hi = window_end_ms * source.time_scale

    with store.ManagedSessionMaker(read_only=True) as session:
        query = session.query(source.trace_id_column).select_from(source.experiment_column.class_)
        if source.join is not None:
            query = source.join(query)

        filters = [
            source.experiment_column == int(rule.experiment_id),
            source.time_column >= lo,
            source.time_column < hi,
            *source.extra_filters,
        ]
        if source.metric_key_column is not None:
            filters.append(source.metric_key_column == rule.metric_key)
        if unit.grouping.only_when_sql is not None:
            filters.append(unit.grouping.only_when_sql())

        dimension_column = (
            source.dimension_columns[unit.grouping.dim_index]
            if unit.grouping.dim_index is not None
            else None
        )
        if dimension_column is not None and rule.dimension_value:
            filters.append(dimension_column == rule.dimension_value)

        query = query.filter(*filters)

        # A count rule has no value to rank by -- every matching row is equally
        # guilty -- so take the most recent instead of an arbitrary page.
        order_column = (
            source.value_column if source.value_column is not None else source.time_column
        )
        query = query.order_by(
            order_column.asc() if _is_worst_first(rule.comparator) else order_column.desc()
        )

        seen: list[str] = []
        # Distinct in Python rather than SQL: DISTINCT with an ORDER BY on a column
        # that is not selected is invalid on some backends, and one bad trace can
        # own several spans.
        for (trace_id,) in query.limit(limit * 10):
            if trace_id is not None and trace_id not in seen:
                seen.append(trace_id)
                if len(seen) == limit:
                    break
        return seen
