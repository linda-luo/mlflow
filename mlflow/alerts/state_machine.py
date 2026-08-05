"""The alert state machine — a pure function over (rule, open instance, observation).

No database, no clock: ``now_ms`` is passed in, so every transition is
exhaustively unit-testable without fixtures.

::

    OK ─breach─▶ PENDING ─sustained─▶ FIRED ─sustained recovery─▶ INACTIVE
     ▲              │                   │                            │
     │              │                   └────user dismisses──────────┤
     └cleared before┘                                                ▼
      sustaining: nothing was ever notified                      DISMISSED

**Alerts are never erased, and only a person closes one.** An instance that
reaches FIRED and then recovers becomes INACTIVE, not gone: it stays on screen,
keeps its ``peak_value``, and still has to be dismissed by a human. The spike
still happened, and a system that quietly deletes its own history teaches users
to ignore it. What INACTIVE buys is the *second* incident: a rule that breached,
recovered, and breached again used to show one unchanging FIRED row, so the
healthy period and the second episode were both invisible. Only PENDING and
FIRED block a new instance from opening, so once the first episode goes INACTIVE
the second one can open and notify on its own.

**Recovery must be sustained, and only real data counts.** One healthy
evaluation never closes an instance -- a metric oscillating around its threshold
would otherwise open a new alert every interval. Recovery needs at least two
consecutive healthy evaluations *and* ``sustain_seconds`` of wall clock, the
same bar the breach had to clear. Missing data is not healthy: it can neither
close an instance nor break a healthy run, so a broken ingestion pipeline cannot
resolve an incident.

The only silent exit remains a PENDING instance that never sustained: nothing
was notified, so there is nothing to acknowledge.
"""

import uuid
from dataclasses import replace
from typing import Literal

from mlflow.alerts.entities import (
    SYSTEM_DISMISS_NOT_SUSTAINED,
    AlertInstance,
    AlertRule,
    Observation,
    Transition,
)
from mlflow.alerts.histogram import compare

OPEN_STATES = frozenset({"PENDING", "FIRED"})
"""States the partial unique index ``index_alert_instances_open`` covers.

At most one instance per ``alert_rule_id`` may be in one of these, which is what
stops a rule paging twice for the same episode.

Deliberately *not* every undismissed state: INACTIVE is undismissed but not open,
and that is the whole point of it. A recovered-but-unacknowledged episode must not
block the next one from opening, or the second incident would be invisible.
"""

ACTIVE_STATES = frozenset({"PENDING", "FIRED", "INACTIVE"})
"""States that still want a human's attention -- the "Active alerts" view.

``OPEN_STATES`` answers "may another instance open?"; this answers "is this still
on someone's screen?". They were the same set until INACTIVE existed.
"""

ObservationVerdict = Literal["BREACHING", "HEALTHY", "INSUFFICIENT_DATA"]


def classify(
    rule: AlertRule, observation: Observation, breaching: bool | None = None
) -> ObservationVerdict:
    """Three outcomes, not two.

    ``INSUFFICIENT_DATA`` is deliberately distinct from ``HEALTHY``: it is the
    difference between "fine" and "structurally cannot fire", and treating it as
    a non-breach would silently close instances during an outage — precisely when
    data goes missing.

    ``breaching`` lets the evaluator inject a decision it reached from exact
    histogram bounds instead of from the (possibly rounded) observed value.
    """
    if not observation.has_data:
        return "INSUFFICIENT_DATA"
    # ``or 0`` so an unset floor never gates: the field is now the user's, and a
    # rule that simply does not carry one must not be frozen by it.
    if observation.sample_count < (rule.min_sample_count or 0):
        return "INSUFFICIENT_DATA"
    if breaching is None:
        breaching = compare(observation.observed_value, rule.comparator, rule.threshold)
    return "BREACHING" if breaching else "HEALTHY"


def is_worse(candidate: float, incumbent: float | None, comparator: str) -> bool:
    """Direction of "worse" follows the comparator.

    For a ``LT``/``LTE`` rule (pass rate below 0.9) the worst value seen is the
    *minimum*, not the maximum.
    """
    if incumbent is None:
        return True
    if comparator in ("GT", "GTE"):
        return candidate > incumbent
    return candidate < incumbent


def _record(instance: AlertInstance, rule: AlertRule, observation: Observation) -> AlertInstance:
    """Fold a fresh observation into an open instance without changing its state."""
    value = observation.observed_value
    peak = instance.peak_value
    if value is not None and is_worse(value, peak, rule.comparator):
        peak = value
    return replace(
        instance,
        observed_value=value,
        peak_value=peak,
        sample_count=observation.sample_count,
        window_start_ms=observation.window_start_ms,
        window_end_ms=observation.window_end_ms,
    )


def close_instance(instance: AlertInstance, now_ms: int, dismissed_by: str) -> AlertInstance:
    """Close an instance without recovery — user dismissal or a ``system:`` actor.

    System closures reuse DISMISSED with a reserved actor string rather than
    adding a state: the instance really is closed-without-recovery, and the actor
    is enough for the UI to render it differently.
    """
    return replace(
        instance,
        state="DISMISSED",
        dismissed_at_ms=now_ms,
        dismissed_by=dismissed_by,
    )


def evaluate_transition(
    rule: AlertRule,
    open_instance: AlertInstance | None,
    observation: Observation,
    now_ms: int,
    *,
    breaching: bool | None = None,
    new_instance_id: str | None = None,
) -> Transition:
    """Decide what one evaluation of one rule does. Pure — the caller does the I/O.

    ``open_instance`` is the rule's *open* instance, if any -- PENDING or FIRED,
    not merely undismissed. Its presence suppresses re-notification: a later breach
    updates ``observed_value`` and ``peak_value`` on it rather than paging again,
    so a three-hour incident produces one notification and not one per evaluation.
    Anything else passed here (an INACTIVE or DISMISSED instance) is ignored, so a
    breach following a recovery opens a second episode and notifies for it.

    Args:
        rule: the rule being evaluated.
        open_instance: the rule's PENDING/FIRED instance, or None.
        observation: what the evaluator read for this window.
        now_ms: evaluation time, passed in so this function stays pure.
        breaching: overrides the comparison, for callers that decided from exact
            histogram bounds. None means compare ``observed_value`` directly.
        new_instance_id: id for a newly opened instance; generated when omitted.

    Returns:
        A :class:`~mlflow.alerts.entities.Transition`. Its ``instance`` is a
        *new* object — the input is never mutated — and ``should_notify`` is true
        on exactly the transition into FIRED.
    """
    instance = (
        open_instance if open_instance is not None and open_instance.state in OPEN_STATES else None
    )
    verdict = classify(rule, observation, breaching)

    if verdict == "INSUFFICIENT_DATA":
        # Not healthy, so it must not close an open instance; not breaching, so it
        # must not open one. The rule's last_sample_count is what surfaces this.
        #
        # It does not break a healthy run either -- `healthy_since_ms` is carried
        # through untouched. A gap suspends the run rather than resetting it,
        # because resetting would let a flaky ingestion pipeline hold an instance
        # FIRED forever, and no gap can close anything on its own: reaching
        # INACTIVE still requires a *healthy* verdict on this evaluation.
        return Transition(
            event="NONE",
            state=instance.state if instance is not None else None,
            instance=instance,
            should_notify=False,
        )

    if verdict == "HEALTHY":
        if instance is None:
            return Transition(event="NONE", state=None, instance=None, should_notify=False)
        if instance.state == "PENDING":
            closed = close_instance(
                _record(instance, rule, observation), now_ms, SYSTEM_DISMISS_NOT_SUSTAINED
            )
            return Transition(
                event="CLOSED_NOT_SUSTAINED",
                state="DISMISSED",
                instance=closed,
                should_notify=False,
            )
        # FIRED and no longer breaching. observed_value keeps updating either way;
        # what decides the state is how long this has been true.
        updated = _record(instance, rule, observation)
        if instance.healthy_since_ms is None:
            # First healthy evaluation after the breach. Stays FIRED: a single
            # healthy reading is a blip, and a metric sitting on its threshold
            # would otherwise close an instance and open a new one every interval.
            # This floor is what makes "sustained" mean something at the default
            # sustain_seconds of 0.
            return Transition(
                event="UPDATED",
                state="FIRED",
                instance=replace(updated, healthy_since_ms=now_ms),
                should_notify=False,
            )
        if now_ms - instance.healthy_since_ms < rule.effective_sustain_seconds * 1000:
            # Recovery is held to the same bar as the breach: a rule that needed
            # ten minutes of breaching to fire needs ten minutes of health to stop.
            return Transition(event="UPDATED", state="FIRED", instance=updated, should_notify=False)
        # Recovered and stayed recovered. Not closed -- INACTIVE is still
        # unacknowledged, still listed, and still dismissible, and `peak_value`
        # carries how bad it got. `healthy_since_ms` is left at the start of the
        # run, so it reads as "recovered at".
        return Transition(
            event="RECOVERED",
            state="INACTIVE",
            instance=replace(updated, state="INACTIVE"),
            # A de-escalation pages nobody. The next breach opens a *new* instance,
            # and that one notifies normally.
            should_notify=False,
        )

    if instance is None:
        opened = AlertInstance(
            alert_instance_id=new_instance_id or uuid.uuid4().hex,
            alert_rule_id=rule.alert_rule_id,
            experiment_id=rule.experiment_id,
            state="PENDING",
            started_at_ms=now_ms,
            window_start_ms=observation.window_start_ms,
            window_end_ms=observation.window_end_ms,
            observed_value=observation.observed_value,
            peak_value=observation.observed_value,
            threshold=rule.threshold,
            sample_count=observation.sample_count,
        )
        if rule.effective_sustain_seconds == 0:
            return Transition(
                event="FIRED",
                state="FIRED",
                instance=replace(opened, state="FIRED", fired_at_ms=now_ms),
                should_notify=True,
            )
        return Transition(event="OPENED", state="PENDING", instance=opened, should_notify=False)

    # Breaching again, so whatever healthy run was accumulating is over. Without
    # this an instance that alternated healthy/breaching would reach INACTIVE on
    # its second healthy reading no matter how long ago the first one was.
    updated = replace(_record(instance, rule, observation), healthy_since_ms=None)
    sustained = now_ms - instance.started_at_ms >= rule.effective_sustain_seconds * 1000
    if instance.state == "PENDING" and sustained:
        # Snapshot the threshold as of firing so later rule edits cannot rewrite
        # what the notification claimed.
        fired = replace(updated, state="FIRED", fired_at_ms=now_ms, threshold=rule.threshold)
        return Transition(event="FIRED", state="FIRED", instance=fired, should_notify=True)

    return Transition(event="UPDATED", state=instance.state, instance=updated, should_notify=False)
