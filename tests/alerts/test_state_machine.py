import pytest

from mlflow.alerts.entities import (
    SYSTEM_DISMISS_NOT_SUSTAINED,
    SYSTEM_DISMISS_RULE_EDITED,
    AlertInstance,
    AlertRule,
    Observation,
)
from mlflow.alerts.state_machine import (
    ACTIVE_STATES,
    OPEN_STATES,
    classify,
    close_instance,
    evaluate_transition,
    is_worse,
)

MINUTE = 60_000
T0 = 1_700_000_000_000


def make_rule(**overrides) -> AlertRule:
    kwargs = {
        "alert_rule_id": "rule-1",
        "experiment_id": 7,
        "name": "Slow checkout responses",
        "metric_key": "latency",
        "dimension_key": "TRACES",
        "aggregation": "PERCENTILE",
        "comparator": "GT",
        "threshold": 45 * MINUTE,
        "window_seconds": 3600,
        "evaluation_interval_seconds": 300,
        "percentile_value": 95.0,
        "min_sample_count": 200,
    }
    kwargs.update(overrides)
    return AlertRule(**kwargs)


def observation(value: float | None, sample_count: int = 500, at_ms: int = T0) -> Observation:
    return Observation(
        observed_value=value,
        sample_count=sample_count,
        window_start_ms=at_ms - 3600 * 1000,
        window_end_ms=at_ms,
    )


def breach(at_ms: int = T0, value: float = 60 * MINUTE) -> Observation:
    return observation(value, at_ms=at_ms)


def healthy(at_ms: int = T0, value: float = 10 * MINUTE) -> Observation:
    return observation(value, at_ms=at_ms)


class Recorder:
    """A one-rule stand-in for the store, so multi-instance episodes are testable.

    The only thing it has to get right is which instance is handed back: the
    store's ``get_open_alert_instance`` returns PENDING/FIRED only, so an instance
    that has gone INACTIVE stops suppressing the next one. That is the whole
    mechanism by which a second incident becomes visible.
    """

    def __init__(self, rule: AlertRule):
        self.rule = rule
        self.instances: dict[str, AlertInstance] = {}
        self.notifications: list[AlertInstance] = []

    def evaluate(self, obs: Observation, now_ms: int):
        open_instance = next((i for i in self.instances.values() if i.state in OPEN_STATES), None)
        transition = evaluate_transition(
            self.rule,
            open_instance,
            obs,
            now_ms,
            new_instance_id=f"inst-{len(self.instances) + 1}",
        )
        if transition.instance is not None and transition.event != "NONE":
            self.instances[transition.instance.alert_instance_id] = transition.instance
        if transition.should_notify:
            self.notifications.append(transition.instance)
        return transition

    def run(self, breaching_by_minute: list[bool], start_ms: int = T0) -> None:
        """One evaluation per minute, breaching where the list says so."""
        for minute, is_breaching in enumerate(breaching_by_minute):
            now = start_ms + minute * MINUTE
            self.evaluate(breach(at_ms=now) if is_breaching else healthy(at_ms=now), now)

    @property
    def ordered(self) -> list[AlertInstance]:
        return sorted(self.instances.values(), key=lambda i: i.started_at_ms)


def open_pending(rule: AlertRule, started_at_ms: int = T0, **overrides) -> AlertInstance:
    kwargs = {
        "alert_instance_id": "inst-1",
        "alert_rule_id": rule.alert_rule_id,
        "experiment_id": rule.experiment_id,
        "state": "PENDING",
        "started_at_ms": started_at_ms,
        "window_start_ms": started_at_ms - 3600 * 1000,
        "window_end_ms": started_at_ms,
        "observed_value": 60 * MINUTE,
        "peak_value": 60 * MINUTE,
        "threshold": rule.threshold,
        "sample_count": 500,
    }
    kwargs.update(overrides)
    return AlertInstance(**kwargs)


def test_first_breach_opens_pending_without_notifying():
    rule = make_rule(sustain_seconds=600)
    transition = evaluate_transition(rule, None, breach(), T0, new_instance_id="inst-1")

    assert transition.event == "OPENED"
    assert transition.state == "PENDING"
    assert transition.should_notify is False
    assert transition.instance.started_at_ms == T0
    assert transition.instance.fired_at_ms is None
    assert transition.instance.threshold == rule.threshold


def test_pending_does_not_fire_one_evaluation_early():
    rule = make_rule(sustain_seconds=600, evaluation_interval_seconds=300)
    instance = open_pending(rule, started_at_ms=T0)

    early = evaluate_transition(rule, instance, breach(at_ms=T0 + 300_000), T0 + 300_000)
    assert early.event == "UPDATED"
    assert early.state == "PENDING"
    assert early.should_notify is False

    on_time = evaluate_transition(rule, instance, breach(at_ms=T0 + 600_000), T0 + 600_000)
    assert on_time.event == "FIRED"
    assert on_time.state == "FIRED"
    assert on_time.should_notify is True
    assert on_time.instance.fired_at_ms == T0 + 600_000


def test_sustain_is_wall_clock_not_a_count_of_evaluations():
    """Ten evaluations inside the sustain window still do not fire it."""
    rule = make_rule(sustain_seconds=3600, evaluation_interval_seconds=300)
    instance = open_pending(rule, started_at_ms=T0)

    for step in range(1, 11):
        now = T0 + step * 300_000
        transition = evaluate_transition(rule, instance, breach(at_ms=now), now)
        assert transition.event == "UPDATED"
        instance = transition.instance

    now = T0 + 3600 * 1000
    assert evaluate_transition(rule, instance, breach(at_ms=now), now).event == "FIRED"


def test_effective_sustain_rounds_up_to_the_interval():
    rule = make_rule(sustain_seconds=30, evaluation_interval_seconds=300)
    assert rule.effective_sustain_seconds == 300

    instance = open_pending(rule, started_at_ms=T0)
    too_soon = evaluate_transition(rule, instance, breach(at_ms=T0 + 60_000), T0 + 60_000)
    assert too_soon.event == "UPDATED"

    rounded = evaluate_transition(rule, instance, breach(at_ms=T0 + 300_000), T0 + 300_000)
    assert rounded.event == "FIRED"


def test_zero_sustain_fires_immediately_on_first_breach():
    rule = make_rule(sustain_seconds=0)
    transition = evaluate_transition(rule, None, breach(), T0, new_instance_id="inst-1")

    assert transition.event == "FIRED"
    assert transition.state == "FIRED"
    assert transition.should_notify is True
    assert transition.instance.started_at_ms == T0
    assert transition.instance.fired_at_ms == T0


def test_three_hour_incident_notifies_exactly_once():
    rule = make_rule(sustain_seconds=0, evaluation_interval_seconds=60)
    instance = None
    notifications = 0

    for step in range(180):
        now = T0 + step * 60_000
        transition = evaluate_transition(
            rule, instance, breach(at_ms=now), now, new_instance_id="inst-1"
        )
        notifications += int(transition.should_notify)
        instance = transition.instance

    assert notifications == 1
    assert instance.state == "FIRED"
    assert instance.alert_instance_id == "inst-1"


def test_one_healthy_evaluation_does_not_close_a_fired_instance():
    """A single healthy reading is a blip, not a recovery.

    Previously an instance stayed FIRED through *any* amount of recovery; it now
    reaches INACTIVE after a sustained one, so the assertion that matters here is
    that the first healthy evaluation alone is not enough.
    """
    rule = make_rule(sustain_seconds=0)
    fired = evaluate_transition(rule, None, breach(), T0, new_instance_id="inst-1").instance

    recovered = evaluate_transition(rule, fired, healthy(at_ms=T0 + 600_000), T0 + 600_000)

    assert recovered.state == "FIRED"
    assert recovered.event == "UPDATED"
    assert recovered.should_notify is False
    assert recovered.instance.dismissed_at_ms is None
    assert recovered.instance.observed_value == 10 * MINUTE
    assert recovered.instance.peak_value == 60 * MINUTE
    # The clock the second healthy evaluation will be measured against.
    assert recovered.instance.healthy_since_ms == T0 + 600_000


def test_sustained_recovery_moves_a_fired_instance_to_inactive():
    rule = make_rule(sustain_seconds=0, evaluation_interval_seconds=60)
    recorder = Recorder(rule)
    recorder.run([True, False, False])

    (instance,) = recorder.ordered
    assert instance.state == "INACTIVE"
    # Still unacknowledged, and still carrying how bad it got.
    assert instance.dismissed_at_ms is None
    assert instance.dismissed_by is None
    assert instance.peak_value == 60 * MINUTE
    assert instance.fired_at_ms == T0
    assert instance.healthy_since_ms == T0 + MINUTE
    # A de-escalation pages nobody: only the original firing notified.
    assert len(recorder.notifications) == 1


def test_going_inactive_is_a_recovered_transition_that_does_not_notify():
    rule = make_rule(sustain_seconds=0, evaluation_interval_seconds=60)
    recorder = Recorder(rule)
    recorder.evaluate(breach(at_ms=T0), T0)
    recorder.evaluate(healthy(at_ms=T0 + MINUTE), T0 + MINUTE)

    transition = recorder.evaluate(healthy(at_ms=T0 + 2 * MINUTE), T0 + 2 * MINUTE)

    assert transition.event == "RECOVERED"
    assert transition.state == "INACTIVE"
    assert transition.should_notify is False


def test_a_second_incident_opens_a_second_instance_while_the_first_is_inactive():
    """Breaching 0-5, healthy 5-7, breaching 7-10: two episodes, not one row.

    The whole point of INACTIVE. Before it, the single instance stayed FIRED
    throughout, so neither the healthy period nor the second incident appeared
    anywhere.
    """
    rule = make_rule(sustain_seconds=0, evaluation_interval_seconds=60)
    recorder = Recorder(rule)
    recorder.run([True] * 5 + [False] * 2 + [True] * 3)

    first, second = recorder.ordered
    assert first.state == "INACTIVE"
    assert first.started_at_ms == T0
    assert first.healthy_since_ms == T0 + 5 * MINUTE
    assert second.state == "FIRED"
    assert second.started_at_ms == T0 + 7 * MINUTE
    assert second.fired_at_ms == T0 + 7 * MINUTE
    assert second.healthy_since_ms is None
    # The second incident pages, exactly once.
    assert [n.alert_instance_id for n in recorder.notifications] == ["inst-1", "inst-2"]


def test_flapping_produces_one_instance_and_one_notification():
    """A metric oscillating across its threshold must not alert every interval."""
    rule = make_rule(sustain_seconds=0, evaluation_interval_seconds=60)
    recorder = Recorder(rule)
    recorder.run([True, False, True, False, True, False, True, False])

    (instance,) = recorder.ordered
    assert instance.state == "FIRED"
    assert len(recorder.notifications) == 1


def test_a_breach_resets_the_healthy_run():
    """Two healthy evaluations with a breach between them are not a recovery."""
    rule = make_rule(sustain_seconds=0, evaluation_interval_seconds=60)
    recorder = Recorder(rule)
    recorder.run([True, False, True])

    (instance,) = recorder.ordered
    assert instance.state == "FIRED"
    assert instance.healthy_since_ms is None

    # ...and the run has to start over from scratch, so the next single healthy
    # evaluation does not close it either.
    recorder.run([False], start_ms=T0 + 3 * MINUTE)
    assert recorder.ordered[0].state == "FIRED"


def test_recovery_is_held_to_the_same_sustain_as_the_breach():
    rule = make_rule(sustain_seconds=600, evaluation_interval_seconds=300)
    recorder = Recorder(rule)
    # Two breaching evaluations 10 minutes apart to reach FIRED.
    recorder.evaluate(breach(at_ms=T0), T0)
    recorder.evaluate(breach(at_ms=T0 + 600_000), T0 + 600_000)
    assert recorder.ordered[0].state == "FIRED"

    # Healthy, but only for one interval: five minutes is not the ten the rule asks for.
    recorder.evaluate(healthy(at_ms=T0 + 900_000), T0 + 900_000)
    too_soon = recorder.evaluate(healthy(at_ms=T0 + 1_200_000), T0 + 1_200_000)
    assert too_soon.state == "FIRED"
    assert too_soon.event == "UPDATED"

    on_time = recorder.evaluate(healthy(at_ms=T0 + 1_500_000), T0 + 1_500_000)
    assert on_time.state == "INACTIVE"
    assert on_time.instance.healthy_since_ms == T0 + 900_000


@pytest.mark.parametrize(
    ("value", "sample_count"),
    [(10 * MINUTE, 199), (None, 0)],
)
def test_missing_data_never_closes_a_recovering_instance(value, sample_count):
    """Only a genuine healthy reading may resolve an incident.

    A gap suspends the healthy run rather than resetting it -- resetting would let
    a broken ingestion pipeline hold an instance FIRED forever -- but the gap on
    its own can never advance anything, because reaching INACTIVE requires a
    healthy verdict on the evaluation that does it.
    """
    rule = make_rule(min_sample_count=200, sustain_seconds=0, evaluation_interval_seconds=60)
    recorder = Recorder(rule)
    recorder.evaluate(breach(at_ms=T0), T0)
    recorder.evaluate(healthy(at_ms=T0 + MINUTE), T0 + MINUTE)

    gap = recorder.evaluate(
        observation(value, sample_count, at_ms=T0 + 2 * MINUTE), T0 + 2 * MINUTE
    )
    assert gap.event == "NONE"
    assert gap.state == "FIRED"

    resumed = recorder.evaluate(healthy(at_ms=T0 + 3 * MINUTE), T0 + 3 * MINUTE)
    assert resumed.state == "INACTIVE"
    assert resumed.instance.healthy_since_ms == T0 + MINUTE


def test_an_inactive_instance_passed_in_does_not_suppress_a_new_one():
    rule = make_rule(sustain_seconds=0)
    inactive = open_pending(rule, state="INACTIVE", healthy_since_ms=T0 + 60_000)

    transition = evaluate_transition(
        rule, inactive, breach(at_ms=T0 + 120_000), T0 + 120_000, new_instance_id="inst-2"
    )

    assert transition.event == "FIRED"
    assert transition.should_notify is True
    assert transition.instance.alert_instance_id == "inst-2"


def test_an_inactive_instance_is_still_dismissible():
    rule = make_rule()
    inactive = open_pending(rule, state="INACTIVE", healthy_since_ms=T0 + 60_000)

    dismissed = close_instance(inactive, T0 + 300_000, "alice@example.com")

    assert dismissed.state == "DISMISSED"
    assert dismissed.dismissed_by == "alice@example.com"
    # The record of how bad it got survives the whole lifecycle.
    assert dismissed.peak_value == 60 * MINUTE


def test_inactive_is_active_but_not_open():
    assert "INACTIVE" in ACTIVE_STATES
    assert "INACTIVE" not in OPEN_STATES
    assert OPEN_STATES < ACTIVE_STATES


def test_pending_that_clears_before_sustaining_closes_silently():
    rule = make_rule(sustain_seconds=600)
    instance = open_pending(rule, started_at_ms=T0)

    transition = evaluate_transition(rule, instance, healthy(at_ms=T0 + 300_000), T0 + 300_000)

    assert transition.event == "CLOSED_NOT_SUSTAINED"
    assert transition.state == "DISMISSED"
    assert transition.should_notify is False
    assert transition.instance.fired_at_ms is None
    assert transition.instance.dismissed_by == SYSTEM_DISMISS_NOT_SUSTAINED
    assert transition.instance.dismissed_at_ms == T0 + 300_000


def test_healthy_with_no_open_instance_is_a_no_op():
    transition = evaluate_transition(make_rule(), None, healthy(), T0)
    assert transition.event == "NONE"
    assert transition.state is None
    assert transition.instance is None


@pytest.mark.parametrize(
    ("value", "sample_count"),
    [
        (60 * MINUTE, 199),  # breaching, but too few samples to mean anything
        (10 * MINUTE, 199),  # healthy-looking, but equally unmeaningful
        (None, 0),  # a gap: aggregation did not run
        (None, 500),  # data present, no value derivable
    ],
)
def test_insufficient_samples_never_opens_an_instance(value, sample_count):
    rule = make_rule(min_sample_count=200)
    transition = evaluate_transition(rule, None, observation(value, sample_count), T0)

    assert transition.event == "NONE"
    assert transition.instance is None


@pytest.mark.parametrize("state", ["PENDING", "FIRED"])
@pytest.mark.parametrize(
    ("value", "sample_count"),
    [(10 * MINUTE, 199), (None, 0)],
)
def test_insufficient_samples_does_not_close_an_open_instance(state, value, sample_count):
    """Insufficient data is not healthy — closing on it would erase an incident
    precisely when the pipeline feeding it broke.
    """
    rule = make_rule(min_sample_count=200, sustain_seconds=600)
    instance = open_pending(rule, started_at_ms=T0, state=state)

    transition = evaluate_transition(
        rule, instance, observation(value, sample_count, at_ms=T0 + 900_000), T0 + 900_000
    )

    assert transition.event == "NONE"
    assert transition.state == state
    assert transition.instance is instance
    assert transition.instance.dismissed_at_ms is None


def test_insufficient_samples_does_not_advance_a_pending_instance_to_fired():
    rule = make_rule(min_sample_count=200, sustain_seconds=600)
    instance = open_pending(rule, started_at_ms=T0)

    stale = evaluate_transition(
        rule, instance, observation(60 * MINUTE, 3, at_ms=T0 + 600_000), T0 + 600_000
    )

    assert stale.event == "NONE"
    assert stale.state == "PENDING"


@pytest.mark.parametrize(
    ("comparator", "threshold", "values", "expected_peak"),
    [
        ("GT", 45.0, [50.0, 90.0, 60.0], 90.0),
        ("GTE", 45.0, [50.0, 90.0, 60.0], 90.0),
        ("LT", 0.9, [0.8, 0.3, 0.5], 0.3),
        ("LTE", 0.9, [0.8, 0.3, 0.5], 0.3),
    ],
)
def test_peak_value_tracks_the_worst_value_for_the_comparators_direction(
    comparator, threshold, values, expected_peak
):
    rule = make_rule(
        comparator=comparator,
        threshold=threshold,
        aggregation="AVG",
        percentile_value=None,
        min_sample_count=1,
        sustain_seconds=0,
    )
    instance = None
    for step, value in enumerate(values):
        now = T0 + step * 300_000
        instance = evaluate_transition(
            rule,
            instance,
            observation(value, 500, at_ms=now),
            now,
            new_instance_id="inst-1",
        ).instance

    assert instance.peak_value == expected_peak
    assert instance.observed_value == values[-1]


@pytest.mark.parametrize(
    ("comparator", "incumbent", "candidate", "expected"),
    [
        ("GT", 5.0, 6.0, True),
        ("GT", 5.0, 4.0, False),
        ("LT", 5.0, 4.0, True),
        ("LTE", 5.0, 6.0, False),
        ("GT", None, 1.0, True),
        ("LT", None, 1.0, True),
    ],
)
def test_is_worse(comparator, incumbent, candidate, expected):
    assert is_worse(candidate, incumbent, comparator) is expected


def test_dismissal_lets_the_rule_open_a_new_instance():
    rule = make_rule(sustain_seconds=0)
    first = evaluate_transition(rule, None, breach(), T0, new_instance_id="inst-1").instance
    dismissed = close_instance(first, T0 + 600_000, "alice@example.com")

    # The store only ever hands back PENDING/FIRED instances, so a later breach
    # sees None and opens a second episode.
    later = evaluate_transition(
        rule, None, breach(at_ms=T0 + 900_000), T0 + 900_000, new_instance_id="inst-2"
    )

    assert dismissed.state == "DISMISSED"
    assert later.event == "FIRED"
    assert later.should_notify is True
    assert later.instance.alert_instance_id == "inst-2"
    assert later.instance.started_at_ms == T0 + 900_000


def test_a_dismissed_instance_passed_in_does_not_suppress_a_new_one():
    rule = make_rule(sustain_seconds=0)
    dismissed = close_instance(open_pending(rule), T0, SYSTEM_DISMISS_RULE_EDITED)

    transition = evaluate_transition(
        rule, dismissed, breach(at_ms=T0 + 60_000), T0 + 60_000, new_instance_id="inst-2"
    )

    assert transition.instance.alert_instance_id == "inst-2"
    assert transition.event == "FIRED"


def test_an_open_instance_suppresses_a_second_one():
    rule = make_rule(sustain_seconds=0)
    fired = evaluate_transition(rule, None, breach(), T0, new_instance_id="inst-1").instance

    again = evaluate_transition(
        rule, fired, breach(at_ms=T0 + 300_000, value=90 * MINUTE), T0 + 300_000
    )

    assert again.instance.alert_instance_id == "inst-1"
    assert again.should_notify is False
    assert again.instance.peak_value == 90 * MINUTE


def test_threshold_is_snapshotted_at_firing_and_survives_a_later_edit():
    rule = make_rule(sustain_seconds=600)
    opened = evaluate_transition(rule, None, breach(), T0, new_instance_id="inst-1").instance
    fired = evaluate_transition(rule, opened, breach(at_ms=T0 + 600_000), T0 + 600_000).instance

    edited = make_rule(sustain_seconds=600, threshold=2 * 60 * MINUTE)
    after_edit = evaluate_transition(
        edited, fired, breach(at_ms=T0 + 900_000, value=90 * MINUTE), T0 + 900_000
    )

    assert fired.threshold == 45 * MINUTE
    assert after_edit.instance.threshold == 45 * MINUTE


def test_transition_never_mutates_the_instance_it_was_given():
    rule = make_rule(sustain_seconds=0)
    instance = open_pending(rule, state="FIRED", observed_value=60 * MINUTE)

    evaluate_transition(rule, instance, breach(at_ms=T0, value=99 * MINUTE), T0)

    assert instance.observed_value == 60 * MINUTE
    assert instance.peak_value == 60 * MINUTE


def test_breaching_override_wins_over_the_naive_comparison():
    """The evaluator decides percentile breaches from exact histogram bounds, and
    the bucketed observed value can disagree with them.
    """
    rule = make_rule(sustain_seconds=0)

    forced = evaluate_transition(
        rule, None, healthy(), T0, breaching=True, new_instance_id="inst-1"
    )
    suppressed = evaluate_transition(rule, None, breach(), T0, breaching=False)

    assert forced.event == "FIRED"
    assert suppressed.event == "NONE"


@pytest.mark.parametrize(
    ("value", "sample_count", "expected"),
    [
        (60 * MINUTE, 500, "BREACHING"),
        (10 * MINUTE, 500, "HEALTHY"),
        (60 * MINUTE, 12, "INSUFFICIENT_DATA"),
        (None, 500, "INSUFFICIENT_DATA"),
        (None, 0, "INSUFFICIENT_DATA"),
    ],
)
def test_classify(value, sample_count, expected):
    rule = make_rule(min_sample_count=200)
    assert classify(rule, observation(value, sample_count)) == expected


def test_close_instance_records_the_actor():
    rule = make_rule()
    closed = close_instance(open_pending(rule), T0 + 60_000, SYSTEM_DISMISS_RULE_EDITED)

    assert closed.state == "DISMISSED"
    assert closed.dismissed_by == SYSTEM_DISMISS_RULE_EDITED
    assert closed.dismissed_at_ms == T0 + 60_000
