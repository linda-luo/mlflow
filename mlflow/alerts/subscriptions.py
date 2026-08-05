"""Which series are worth aggregating.

Series used to be created by *observation*: whatever an agent emitted became a
series. That is unbounded by construction -- one agent embedding a request id in a
tool name creates a series per request -- and it spends the aggregator's budget on
data nobody reads, which measurement showed to be the cost that actually scales.

A subscription inverts it: the aggregator writes only what some rule reads, so
cardinality is bounded by the number of rules rather than by the number of distinct
strings an agent happens to emit.

**Subscriptions are a query, not a table.** They were briefly materialized into
``rollup_subscriptions``, reconciled on every rule change. That table held nothing a
``SELECT DISTINCT`` over ``alert_rules`` does not already hold: every column was
either copied from the rule, derivable from ``enabled``/``deleted_at_ms``, or
deliberately discarded. Worse, a derived table is the only thing that *can* drift
from the rules, so it created the very failure mode its reconciliation then had to
defend against. Reading the rules directly means there is no hook to miss and
nothing to reconcile.

Bucketing is deliberately not driven from here either. The sketch grid is a constant
per metric, so it cannot move because a rule was created -- which is what stops one
experiment's threshold rebucketing another's data, and one rule's creation
invalidating the history a different rule was reading.
"""

from dataclasses import dataclass

from mlflow.store.tracking.dbmodels.models import SqlAlertRule


@dataclass(frozen=True)
class SeriesFamily:
    """A ``(dimension_key, metric_key)`` pair -- one aggregator work unit's output."""

    dimension_key: str
    metric_key: str


@dataclass(frozen=True)
class Subscription:
    """One experiment's claim on one series family.

    Deliberately not scoped to a rule's ``dimension_value``. A rule targeting a
    single tool still subscribes to the whole dimension: the rule editor has to be
    able to offer the *other* values, and grouped rules -- the named v2, firing
    separately per error type -- need those series to already exist. Holding them
    costs one row per bucket per value.
    """

    experiment_id: int
    dimension_key: str
    metric_key: str

    @property
    def family(self) -> SeriesFamily:
        return SeriesFamily(self.dimension_key, self.metric_key)


def load_active_subscriptions(session) -> list[Subscription]:
    """What every live rule reads, as of now.

    One indexed ``DISTINCT`` over a table holding on the order of a hundred rows,
    called once per aggregation run -- not per unit and not per bucket.

    Scoped per *experiment*, which is what actually bounds the write volume: a
    family is aggregated only for the experiments holding a rule that reads it, so
    an experiment nobody has written a rule for produces no series at all, however
    much traffic it takes.

    This does not narrow the scan, only the write. The units run on a fixed
    schedule so their watermarks stay current, and :func:`is_subscribed` discards
    the unsubscribed series after the scan -- which is why a new rule starts
    producing rows on the next bucket rather than after a backfill.
    """
    rows = (
        session
        .query(
            SqlAlertRule.experiment_id,
            SqlAlertRule.dimension_key,
            SqlAlertRule.metric_key,
        )
        .filter(SqlAlertRule.enabled.is_(True), SqlAlertRule.deleted_at_ms.is_(None))
        .distinct()
        .all()
    )
    return [
        Subscription(experiment_id=int(experiment_id), dimension_key=dim, metric_key=metric)
        for experiment_id, dim, metric in rows
    ]


def is_subscribed(
    subscriptions: list[Subscription],
    family: SeriesFamily,
    experiment_id: int,
    dimension_value: str,
) -> bool:
    """Whether an observed series is worth writing.

    ``dimension_value`` is accepted but unused: a subscription covers every value of
    its dimension. It stays in the signature because grouped rules will need to
    narrow it, and the aggregator's call site should not have to change then.
    """
    return any(s.family == family and s.experiment_id == experiment_id for s in subscriptions)


def subscribed_families(subscriptions: list[Subscription]) -> set[SeriesFamily]:
    return {s.family for s in subscriptions}
