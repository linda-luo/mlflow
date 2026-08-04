import dataclasses
import json
from pathlib import Path
from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.alerts.entities import (
    MAX_WINDOW_SECONDS,
    MIN_WINDOW_SECONDS,
    SYSTEM_DISMISS_RULE_DISABLED,
    SYSTEM_DISMISS_RULE_EDITED,
    AlertInstance,
    AlertRule,
    derive_evaluation_interval_seconds,
    derive_min_sample_count,
)
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.store.tracking.dbmodels.models import (
    SqlAlertInstance,
    SqlAssessments,
    SqlMetricSeries,
    SqlSpan,
    SqlSpanError,
    SqlTraceInfo,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils.time import get_current_time_millis


@pytest.fixture
def store(tmp_path: Path, db_uri: str) -> SqlAlchemyStore:
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir()
    return SqlAlchemyStore(db_uri, artifact_uri.as_uri())


@pytest.fixture
def experiment_id(store: SqlAlchemyStore) -> str:
    return store.create_experiment("alerting")


def _rule(experiment_id: str, **overrides) -> AlertRule:
    kwargs = {
        "alert_rule_id": "",
        "experiment_id": int(experiment_id),
        "name": "Slow checkout responses",
        "metric_key": "latency",
        "dimension_key": "TRACES",
        "aggregation": "PERCENTILE",
        "percentile_value": 95.0,
        "comparator": "GT",
        "threshold": 45 * 60_000,
        "window_seconds": 3600,
        "evaluation_interval_seconds": 300,
    }
    kwargs.update(overrides)
    return AlertRule(**kwargs)


def _open_instance(store: SqlAlchemyStore, rule: AlertRule, state: str = "FIRED") -> str:
    """Seed one instance directly, in whatever state the test needs.

    Instances are opened by the evaluator, not by any of the store's
    alerting methods, so the edit/delete/dismiss behaviors are exercised
    against a hand-seeded row.
    """
    instance_id = f"inst-{state.lower()}-{rule.alert_rule_id[:8]}"
    with store.ManagedSessionMaker(read_only=False) as session:
        session.add(
            SqlAlertInstance(
                alert_instance_id=instance_id,
                alert_rule_id=rule.alert_rule_id,
                experiment_id=rule.experiment_id,
                state=state,
                started_at_ms=get_current_time_millis(),
                fired_at_ms=get_current_time_millis() if state == "FIRED" else None,
                observed_value=99.0,
                peak_value=120.0,
                threshold=rule.threshold,
                sample_count=250,
                window_start_ms=1,
                window_end_ms=2,
            )
        )
    return instance_id


def _error_code(exc: MlflowException) -> str:
    return exc.error_code


def test_create_derives_the_interval_and_jitters_next_evaluation(store, experiment_id):
    before_ms = get_current_time_millis()
    created = store.create_alert_rule(_rule(experiment_id, window_seconds=3600))

    assert created.alert_rule_id
    assert created.evaluation_interval_seconds == derive_evaluation_interval_seconds(3600)
    # The interval is still derived here; the sample floor is not. Suggesting one
    # moved to the REST layer, which is the only place that can tell an omitted
    # field from an explicit zero -- see `tests/server/test_alert_handlers.py`.
    assert created.min_sample_count == 0
    # One window out, then jittered within one interval so rules created together
    # don't stampede the same tick.
    assert (
        before_ms + 3600 * 1000
        <= created.next_evaluation_at_ms
        <= get_current_time_millis() + 3600 * 1000 + created.evaluation_interval_seconds * 1000
    )


@pytest.mark.parametrize("window_seconds", [MIN_WINDOW_SECONDS, 3600, MAX_WINDOW_SECONDS])
def test_the_first_evaluation_waits_out_one_whole_window(store, experiment_id, window_seconds):
    """A new rule may not conclude anything from history that predates it.

    Holds at the ceiling too: a three-day rule waits three days, which is the point
    -- it has no three days of its own history to read.
    """
    before_ms = get_current_time_millis()
    created = store.create_alert_rule(_rule(experiment_id, window_seconds=window_seconds))

    offset_ms = created.next_evaluation_at_ms - before_ms
    assert offset_ms >= window_seconds * 1000
    assert offset_ms <= window_seconds * 1000 + created.evaluation_interval_seconds * 1000 + (
        get_current_time_millis() - before_ms
    )


def test_a_brand_new_rule_is_not_due_on_the_next_tick(store, experiment_id):
    """The regression: an absence rule must not fire off history nobody recorded.

    ``coverage_start_ms`` cannot catch this. It is keyed per family, not per
    ``(experiment, family)``, so on a server that has been aggregating a while it
    reports the window as covered even though no row was ever written for *this*
    experiment's series. An empty COUNT window is a genuine zero, so a
    ``COUNT < 5`` rule fired immediately and reported an outage that never happened.

    Being un-due is what buys the window time to fill.
    """
    created = store.create_alert_rule(
        _rule(
            experiment_id,
            aggregation="COUNT",
            percentile_value=None,
            comparator="LT",
            threshold=5,
            window_seconds=600,
        )
    )

    now_ms = get_current_time_millis()
    assert store.due_alert_rules(now_ms) == []
    # And it is still not due one interval later, which is when it used to run.
    assert store.due_alert_rules(now_ms + created.evaluation_interval_seconds * 1000) == []
    # Due once the window it has to wait out has passed.
    due = store.due_alert_rules(created.next_evaluation_at_ms)
    assert [r.alert_rule_id for r in due] == [created.alert_rule_id]


def test_a_percentile_threshold_is_stored_exactly_as_given(store, experiment_id):
    """The user's number survives verbatim.

    Thresholds used to be snapped onto the nearest histogram boundary so the bounds
    check would collapse -- which worked, but stored and alerted on 45 minutes when
    the user had asked for 44. The sketch's grid is fine enough everywhere that
    nothing needs moving.
    """
    odd = 44 * 60_000

    created = store.create_alert_rule(_rule(experiment_id, threshold=odd))

    assert created.threshold == odd


def test_editing_a_threshold_also_keeps_it_exact(store, experiment_id):
    created = store.create_alert_rule(_rule(experiment_id))
    updated = store.update_alert_rule(created.alert_rule_id, threshold=43 * 60_000)
    assert updated.threshold == 43 * 60_000


def test_min_sample_count_is_stored_as_given(store, experiment_id):
    """The caller's floor wins, including an explicit zero.

    It used to be derived and imposed: a p99 rule on a workload producing 500
    samples a window sat permanently in INSUFFICIENT_DATA, unable to open *or
    close* an instance, with nothing saying which number was responsible.
    """
    created = store.create_alert_rule(_rule(experiment_id, min_sample_count=7))
    assert created.min_sample_count == 7

    unfloored = store.create_alert_rule(_rule(experiment_id, name="no floor", min_sample_count=0))
    assert unfloored.min_sample_count == 0


def test_a_negative_min_sample_count_is_rejected(store, experiment_id):
    with pytest.raises(MlflowException, match="min_sample_count` must not be negative") as exc:
        store.create_alert_rule(_rule(experiment_id, min_sample_count=-1))
    assert _error_code(exc.value) == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_changing_the_percentile_moves_a_floor_the_user_never_set(store, experiment_id):
    created = store.create_alert_rule(_rule(experiment_id, min_sample_count=200))
    updated = store.update_alert_rule(created.alert_rule_id, percentile_value=99.0)
    assert updated.min_sample_count == derive_min_sample_count("PERCENTILE", 99.0)


def test_changing_the_percentile_does_not_overwrite_a_floor_the_user_set(store, experiment_id):
    """The whole point of surfacing the field: an explicit choice survives an edit."""
    created = store.create_alert_rule(_rule(experiment_id, min_sample_count=200))
    updated = store.update_alert_rule(
        created.alert_rule_id, percentile_value=99.0, min_sample_count=25
    )
    assert updated.min_sample_count == 25


def test_the_entity_defaults_to_no_floor(experiment_id):
    """A floor of one makes "traffic dropped to zero" unfireable."""
    assert _rule(experiment_id).min_sample_count == 0


def test_create_leaves_non_percentile_threshold_untouched(store, experiment_id):
    # COUNT reads the `count` column, not the histogram; snapping a request
    # count onto a latency boundary would silently rewrite the user's number.
    created = store.create_alert_rule(
        _rule(
            experiment_id,
            aggregation="COUNT",
            percentile_value=None,
            threshold=7.0,
        )
    )
    assert created.threshold == 7.0
    # Zero, not one: a COUNT rule with a one-sample floor could never fire on
    # "requests dropped to zero", which is exactly when it should.
    assert created.min_sample_count == 0


def test_create_rejects_duplicate_name(store, experiment_id):
    store.create_alert_rule(_rule(experiment_id))
    with pytest.raises(MlflowException, match="already exists") as exc:
        store.create_alert_rule(_rule(experiment_id))
    assert _error_code(exc.value) == ErrorCode.Name(RESOURCE_ALREADY_EXISTS)


def test_a_deleted_rule_frees_its_name(store, experiment_id):
    """Uniqueness is scoped to live rules.

    Deleted rules keep their name so their instance history stays attributable,
    but holding the name reserved forever meant a typo could never be corrected:
    there is no rename that frees it, and recreating under the same name failed.
    """
    first = store.create_alert_rule(_rule(experiment_id))
    store.delete_alert_rule(first.alert_rule_id)

    recreated = store.create_alert_rule(_rule(experiment_id))
    assert recreated.alert_rule_id != first.alert_rule_id
    assert recreated.name == first.name


def test_several_deleted_rules_may_share_one_name(store, experiment_id):
    for _ in range(3):
        created = store.create_alert_rule(_rule(experiment_id))
        store.delete_alert_rule(created.alert_rule_id)

    assert store.create_alert_rule(_rule(experiment_id)).name == "Slow checkout responses"


def test_rename_onto_a_live_name_is_rejected(store, experiment_id):
    store.create_alert_rule(_rule(experiment_id, name="taken"))
    other = store.create_alert_rule(_rule(experiment_id, name="free"))

    with pytest.raises(MlflowException, match="already exists") as exc:
        store.update_alert_rule(other.alert_rule_id, name="taken")
    assert _error_code(exc.value) == ErrorCode.Name(RESOURCE_ALREADY_EXISTS)


def test_rename_onto_a_deleted_name_is_allowed(store, experiment_id):
    gone = store.create_alert_rule(_rule(experiment_id, name="retired"))
    store.delete_alert_rule(gone.alert_rule_id)
    other = store.create_alert_rule(_rule(experiment_id, name="free"))

    assert store.update_alert_rule(other.alert_rule_id, name="retired").name == "retired"


def test_renaming_a_rule_to_its_own_name_is_a_no_op(store, experiment_id):
    created = store.create_alert_rule(_rule(experiment_id, name="stable"))
    assert store.update_alert_rule(created.alert_rule_id, name="stable").name == "stable"


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"metric_key": "nonexistent"}, "Unknown metric"),
        ({"dimension_key": "SPAN_MODEL"}, "cannot be sliced by"),
        ({"metric_key": "error_count", "dimension_key": "ERROR"}, "does not support"),
        ({"comparator": "EQ"}, "Invalid comparator"),
        ({"severity": "CRITICAL"}, "Invalid severity"),
        ({"window_seconds": 60}, "window_seconds` must be between"),
        ({"window_seconds": 300_000}, "window_seconds` must be between"),
        ({"percentile_value": None}, "percentile_value` must be strictly between"),
        ({"sustain_seconds": -1}, "sustain_seconds` must not be negative"),
        ({"name": "  "}, "name` is required"),
    ],
)
def test_create_validates_against_the_metric_catalogue(store, experiment_id, overrides, match):
    with pytest.raises(MlflowException, match=match) as exc:
        store.create_alert_rule(_rule(experiment_id, **overrides))
    assert _error_code(exc.value) == ErrorCode.Name(INVALID_PARAMETER_VALUE)


@pytest.mark.parametrize("window_seconds", [300, 259_200])
def test_the_window_bounds_themselves_are_accepted(store, experiment_id, window_seconds):
    """Both ends, which nothing tested before -- only values well past them."""
    created = store.create_alert_rule(
        _rule(experiment_id, name=f"w{window_seconds}", window_seconds=window_seconds)
    )
    assert created.window_seconds == window_seconds


@pytest.mark.parametrize("window_seconds", [299, 259_201])
def test_one_second_outside_the_window_bounds_is_rejected(store, experiment_id, window_seconds):
    with pytest.raises(MlflowException, match="window_seconds` must be between") as exc:
        store.create_alert_rule(_rule(experiment_id, window_seconds=window_seconds))
    assert _error_code(exc.value) == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_a_value_is_rejected_for_a_dimension_that_is_never_sliced(store, experiment_id):
    """Whole-trace token counts are grouped under TRACES but written with an empty
    `dimension_value`, so a rule naming a status addresses a series that cannot
    exist. It used to be accepted and then silently never fire.
    """
    with pytest.raises(MlflowException, match="aggregated across all of") as exc:
        store.create_alert_rule(
            _rule(
                experiment_id,
                metric_key="total_tokens",
                dimension_key="TRACES",
                dimension_value="OK",
            )
        )
    assert _error_code(exc.value) == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_the_same_dimension_is_sliceable_for_a_metric_that_does_slice_it(store, experiment_id):
    # Latency *is* sliced by trace status, so TRACES is not unsliceable per se --
    # only unsliceable for the metrics whose source has no dimension column.
    created = store.create_alert_rule(
        _rule(experiment_id, metric_key="latency", dimension_key="TRACES", dimension_value="ERROR")
    )
    assert created.dimension_value == "ERROR"


def test_a_dimension_value_wider_than_the_column_is_rejected(store, experiment_id):
    # Observed values are truncated to the column width on the way in, so a longer
    # value could never match the series it names.
    with pytest.raises(MlflowException, match="at most 250 characters") as exc:
        store.create_alert_rule(
            _rule(experiment_id, dimension_key="SPAN_NAME", dimension_value="x" * 251)
        )
    assert _error_code(exc.value) == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_update_validates_the_dimension_value_too(store, experiment_id):
    created = store.create_alert_rule(_rule(experiment_id))
    with pytest.raises(MlflowException, match="aggregated across all of"):
        store.update_alert_rule(
            created.alert_rule_id, metric_key="total_tokens", dimension_value="OK"
        )


@pytest.mark.parametrize(
    ("metric_key", "dimension_key", "match"),
    [
        ("nonexistent", "TRACES", "Unknown metric"),
        ("latency", "ASSESSMENTS", "cannot be sliced by"),
        ("total_tokens", "TRACES", "aggregated across all of"),
    ],
)
def test_dimension_values_rejects_a_pair_that_can_have_none(
    store, experiment_id, metric_key, dimension_key, match
):
    """An empty list must mean "nothing observed yet" and nothing else.

    It previously doubled as the answer for a nonsense request, so a typo in the
    metric name looked identical to an experiment with no traffic.
    """
    with pytest.raises(MlflowException, match=match) as exc:
        store.list_alert_dimension_values(experiment_id, metric_key, dimension_key)
    assert _error_code(exc.value) == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_dimension_values_is_empty_when_nothing_has_been_observed(store, experiment_id):
    assert store.list_alert_dimension_values(experiment_id, "latency", "SPAN_NAME") == []


def test_percentile_on_tokens_is_now_accepted(store, experiment_id):
    """Every metric the catalogue advertises PERCENTILE for now has its own ladder.

    Previously only `latency` did, so three of five metrics could not answer the
    aggregation most people want from them.
    """
    created = store.create_alert_rule(
        _rule(experiment_id, metric_key="total_tokens", dimension_key="TRACES")
    )
    assert created.aggregation == "PERCENTILE"


def test_get_and_list_round_trip(store, experiment_id):
    created = store.create_alert_rule(_rule(experiment_id))
    fetched = store.get_alert_rule(created.alert_rule_id)
    assert fetched == created

    listed = store.list_alert_rules(experiment_id)
    assert [r.alert_rule_id for r in listed] == [created.alert_rule_id]


def test_list_is_scoped_to_the_experiment(store, experiment_id):
    other_experiment = store.create_experiment("other")
    store.create_alert_rule(_rule(experiment_id))
    store.create_alert_rule(_rule(other_experiment, name="Elsewhere"))

    assert [r.name for r in store.list_alert_rules(experiment_id)] == ["Slow checkout responses"]
    assert [r.name for r in store.list_alert_rules(other_experiment)] == ["Elsewhere"]


def test_update_threshold_keeps_an_open_instance_open(store, experiment_id):
    rule = store.create_alert_rule(_rule(experiment_id))
    instance_id = _open_instance(store, rule)

    updated = store.update_alert_rule(rule.alert_rule_id, threshold=30 * 60_000)

    assert updated.threshold == 30 * 60_000
    assert updated.last_updated_timestamp >= rule.last_updated_timestamp
    (still_open,) = store.list_alert_instances(experiment_id)
    assert still_open.alert_instance_id == instance_id
    assert still_open.state == "FIRED"
    # The instance keeps the threshold it fired at, so the history stays truthful.
    assert still_open.threshold == rule.threshold


def test_update_comparator_keeps_an_open_instance_open(store, experiment_id):
    rule = store.create_alert_rule(_rule(experiment_id))
    _open_instance(store, rule)

    store.update_alert_rule(rule.alert_rule_id, comparator="GTE")

    assert [i.state for i in store.list_alert_instances(experiment_id)] == ["FIRED"]


@pytest.mark.parametrize(
    "updates",
    [
        {"window_seconds": 7200},
        {"metric_key": "total_tokens", "aggregation": "SUM", "percentile_value": None},
        {"dimension_key": "SPAN_TYPE"},
        {"dimension_value": "TOOL"},
    ],
)
def test_update_of_what_is_measured_closes_the_open_instance(store, experiment_id, updates):
    rule = store.create_alert_rule(_rule(experiment_id))
    _open_instance(store, rule)

    store.update_alert_rule(rule.alert_rule_id, **updates)

    assert store.list_alert_instances(experiment_id) == []
    (closed,) = store.list_alert_instances(experiment_id, states=["DISMISSED"])
    assert closed.state == "DISMISSED"
    assert closed.dismissed_by == SYSTEM_DISMISS_RULE_EDITED


def test_disabling_a_rule_closes_the_open_instance(store, experiment_id):
    rule = store.create_alert_rule(_rule(experiment_id))
    _open_instance(store, rule, state="PENDING")

    updated = store.update_alert_rule(rule.alert_rule_id, enabled=False)

    assert updated.enabled is False
    (closed,) = store.list_alert_instances(experiment_id, states=["DISMISSED"])
    assert closed.dismissed_by == SYSTEM_DISMISS_RULE_DISABLED


def test_update_window_re_derives_the_interval_and_pulls_the_due_time_in(store, experiment_id):
    rule = store.create_alert_rule(_rule(experiment_id, window_seconds=86_400))
    assert rule.evaluation_interval_seconds == 300

    updated = store.update_alert_rule(rule.alert_rule_id, window_seconds=1200)

    assert updated.evaluation_interval_seconds == 120
    assert updated.next_evaluation_at_ms <= get_current_time_millis() + 120 * 1000


def test_update_validates_the_merged_rule(store, experiment_id):
    rule = store.create_alert_rule(_rule(experiment_id))
    # `error_count` cannot be sliced by TRACES and allows only COUNT, so both
    # patches below are illegal even though each individual field is plausible
    # for some rule -- the merged rule is what gets validated.
    with pytest.raises(MlflowException, match="cannot be sliced by"):
        store.update_alert_rule(rule.alert_rule_id, metric_key="error_count")
    with pytest.raises(MlflowException, match="does not support"):
        store.update_alert_rule(rule.alert_rule_id, metric_key="error_count", dimension_key="ERROR")


def test_update_rejects_unknown_fields(store, experiment_id):
    rule = store.create_alert_rule(_rule(experiment_id))
    with pytest.raises(MlflowException, match="Unsupported alert rule field") as exc:
        # Derived from the window, so it is not the caller's to set.
        store.update_alert_rule(rule.alert_rule_id, evaluation_interval_seconds=5)
    assert _error_code(exc.value) == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_update_missing_rule_raises(store):
    with pytest.raises(MlflowException, match="not found") as exc:
        store.update_alert_rule("does-not-exist", threshold=1.0)
    assert _error_code(exc.value) == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_delete_is_soft_and_keeps_instances_readable(store, experiment_id):
    rule = store.create_alert_rule(_rule(experiment_id))
    instance_id = _open_instance(store, rule)

    store.delete_alert_rule(rule.alert_rule_id)

    assert store.list_alert_rules(experiment_id) == []
    with pytest.raises(MlflowException, match="not found"):
        store.get_alert_rule(rule.alert_rule_id)

    # Open instances are closed: nothing will ever update them again, and an
    # unacknowledged row for a rule that no longer exists is noise the user
    # cannot act on.
    assert store.list_alert_instances(experiment_id) == []

    # But closing is not forgetting. The rule row survives (soft delete) and so
    # does everything it caught, which is the history a postmortem needs.
    (instance,) = store.list_alert_instances(experiment_id, states=["DISMISSED"])
    assert instance.alert_instance_id == instance_id
    assert instance.dismissed_by == "system:rule_deleted"


def test_delete_soft_deletes_rather_than_removing_the_row(store, experiment_id):
    rule = store.create_alert_rule(_rule(experiment_id))
    store.delete_alert_rule(rule.alert_rule_id)

    with store.ManagedSessionMaker() as session:
        from mlflow.store.tracking.dbmodels.models import SqlAlertRule

        row = (
            session
            .query(SqlAlertRule)
            .filter(SqlAlertRule.alert_rule_id == rule.alert_rule_id)
            .one()
        )
        assert row.deleted_at_ms is not None


def test_list_instances_defaults_to_the_undismissed_states(store, experiment_id):
    """INACTIVE is in the default view, not in history.

    It has recovered, but nobody has acknowledged it, so it is still someone's to
    look at. The default used to be the *open* states, which is a narrower notion
    -- open decides whether a second instance may be created, not whether this one
    is still on screen.
    """
    rule = store.create_alert_rule(_rule(experiment_id))
    _open_instance(store, rule, state="FIRED")
    _open_instance(store, rule, state="INACTIVE")
    _open_instance(store, rule, state="DISMISSED")

    active = {i.state for i in store.list_alert_instances(experiment_id)}
    assert active == {"FIRED", "INACTIVE"}
    states = {i.state for i in store.list_alert_instances(experiment_id, states=["DISMISSED"])}
    assert states == {"DISMISSED"}
    assert [i.state for i in store.list_alert_instances(experiment_id, states=["INACTIVE"])] == [
        "INACTIVE"
    ]


def test_an_inactive_instance_does_not_block_a_new_one(store, experiment_id):
    """The invariant the whole feature rests on.

    ``index_alert_instances_open`` and ``_assert_no_open_alert_instance`` both
    scope "open" to PENDING/FIRED, so a recovered instance stays listed while the
    rule opens the next episode. If either counted INACTIVE, the second incident
    would be rejected on Postgres/SQLite and silently absent on MySQL.
    """
    rule = store.create_alert_rule(_rule(experiment_id))
    _open_instance(store, rule, state="INACTIVE")

    second = store.save_alert_instance(
        AlertInstance(
            alert_instance_id="inst-second",
            alert_rule_id=rule.alert_rule_id,
            experiment_id=int(experiment_id),
            state="FIRED",
            started_at_ms=get_current_time_millis(),
            window_start_ms=1,
            window_end_ms=2,
            fired_at_ms=get_current_time_millis(),
        )
    )

    assert second.state == "FIRED"
    assert store.get_open_alert_instance(rule.alert_rule_id).alert_instance_id == "inst-second"
    assert {i.state for i in store.list_alert_instances(experiment_id)} == {"INACTIVE", "FIRED"}


def test_get_open_alert_instance_ignores_inactive(store, experiment_id):
    rule = store.create_alert_rule(_rule(experiment_id))
    _open_instance(store, rule, state="INACTIVE")

    assert store.get_open_alert_instance(rule.alert_rule_id) is None


def test_save_alert_instance_round_trips_the_healthy_run(store, experiment_id):
    rule = store.create_alert_rule(_rule(experiment_id))
    now_ms = get_current_time_millis()
    instance = store.save_alert_instance(
        AlertInstance(
            alert_instance_id="inst-1",
            alert_rule_id=rule.alert_rule_id,
            experiment_id=int(experiment_id),
            state="FIRED",
            started_at_ms=now_ms,
            window_start_ms=1,
            window_end_ms=2,
            fired_at_ms=now_ms,
            healthy_since_ms=now_ms + 60_000,
        )
    )
    assert instance.healthy_since_ms == now_ms + 60_000

    # A fresh breach clears the run, and the null has to survive the round trip --
    # a column that only ever accumulates would make every later blip look like a
    # recovery in progress.
    cleared = store.save_alert_instance(
        dataclasses.replace(instance, healthy_since_ms=None, peak_value=120.0)
    )
    assert cleared.healthy_since_ms is None


def test_dismissing_from_inactive_keeps_the_record(store, experiment_id):
    """INACTIVE stays dismissible: that is what separates "a human reviewed this"
    from "it aged out", and the peak is the part worth keeping either way.
    """
    rule = store.create_alert_rule(_rule(experiment_id))
    instance_id = _open_instance(store, rule, state="INACTIVE")

    dismissed = store.dismiss_alert_instance(instance_id, "alice")

    assert dismissed.state == "DISMISSED"
    assert dismissed.dismissed_by == "alice"
    assert dismissed.dismissed_at_ms is not None
    assert dismissed.peak_value == 120.0
    assert store.list_alert_instances(experiment_id) == []


@pytest.mark.parametrize(
    "action",
    [
        pytest.param(
            lambda store, rule: store.update_alert_rule(rule.alert_rule_id, window_seconds=7200),
            id="edited",
        ),
        pytest.param(
            lambda store, rule: store.update_alert_rule(rule.alert_rule_id, enabled=False),
            id="disabled",
        ),
        pytest.param(lambda store, rule: store.delete_alert_rule(rule.alert_rule_id), id="deleted"),
    ],
)
def test_editing_a_rule_closes_its_inactive_instances_too(store, experiment_id, action):
    """An unacknowledged row for a rule that now measures something else -- or no
    longer exists -- is just as unactionable when it has recovered as when it is
    still firing. They close to DISMISSED with a ``system:`` actor, not to
    INACTIVE, because it was the user's edit that closed them and nothing is known
    about whether the condition cleared.
    """
    rule = store.create_alert_rule(_rule(experiment_id))
    _open_instance(store, rule, state="INACTIVE")

    action(store, rule)

    assert store.list_alert_instances(experiment_id) == []
    (closed,) = store.list_alert_instances(experiment_id, states=["DISMISSED"])
    assert closed.state == "DISMISSED"
    assert closed.dismissed_by.startswith("system:")


def test_list_instances_rejects_an_unknown_state(store, experiment_id):
    with pytest.raises(MlflowException, match="Invalid alert instance state"):
        store.list_alert_instances(experiment_id, states=["OK"])


def test_dismiss_is_idempotent_and_keeps_the_first_actor(store, experiment_id):
    rule = store.create_alert_rule(_rule(experiment_id))
    instance_id = _open_instance(store, rule)

    first = store.dismiss_alert_instance(instance_id, "alice")
    second = store.dismiss_alert_instance(instance_id, "bob")

    assert first.state == "DISMISSED"
    assert first.dismissed_by == "alice"
    assert second.dismissed_by == "alice"
    assert second.dismissed_at_ms == first.dismissed_at_ms


def test_dismiss_requires_an_actor(store, experiment_id):
    rule = store.create_alert_rule(_rule(experiment_id))
    instance_id = _open_instance(store, rule)
    with pytest.raises(MlflowException, match="dismissed_by` is required"):
        store.dismiss_alert_instance(instance_id, "")


def test_dismiss_missing_instance_raises(store):
    with pytest.raises(MlflowException, match="not found") as exc:
        store.dismiss_alert_instance("nope", "alice")
    assert _error_code(exc.value) == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def _seed_dimension_value_sources(store, experiment_id, now_ms):
    """Raw rows only -- deliberately no `metric_series`, which is the point."""
    with store.ManagedSessionMaker(read_only=False) as session:
        # Parents first: assessments and span_errors both carry foreign keys.
        session.add(
            SqlTraceInfo(
                request_id="t-1",
                experiment_id=int(experiment_id),
                timestamp_ms=now_ms - 61_000,
                execution_time_ms=1_000,
                end_time_ms=now_ms - 60_000,
                status="OK",
            )
        )
        session.flush()
        session.add(
            SqlSpan(
                trace_id="t-1",
                experiment_id=int(experiment_id),
                span_id="s-1",
                name="search_docs",
                type="TOOL",
                status="ERROR",
                start_time_unix_nano=(now_ms - 61_000) * 1_000_000,
                end_time_unix_nano=(now_ms - 60_000) * 1_000_000,
                content="{}",
                dimension_attributes={},
            )
        )
        session.flush()
        for i, name in enumerate(("safety", "relevance")):
            session.add(
                SqlAssessments(
                    assessment_id=f"a-{i}",
                    trace_id="t-1",
                    name=name,
                    assessment_type="feedback",
                    value=json.dumps(True),
                    created_timestamp=now_ms - 60_000,
                    last_updated_timestamp=now_ms - 60_000,
                    source_type="LLM_JUDGE",
                    valid=True,
                    experiment_id=int(experiment_id),
                )
            )
        session.add(
            SqlSpanError(
                trace_id="t-1",
                span_id="s-1",
                exception_type="TimeoutError",
                parent_span_id=None,
                is_origin=True,
                span_name="search_docs",
                span_type="TOOL",
                experiment_id=int(experiment_id),
                timestamp_ms=now_ms - 60_000,
            )
        )


def test_dimension_values_come_from_raw_tables_not_from_metric_series(store, experiment_id):
    """The rule editor must work before anything has been aggregated.

    Sourcing it from `metric_series` deadlocks once aggregation is demand-driven:
    no subscription means no series means an empty dropdown means the user cannot
    create the rule that would have subscribed.
    """
    now_ms = get_current_time_millis()
    _seed_dimension_value_sources(store, experiment_id, now_ms)

    with store.ManagedSessionMaker() as session:
        assert session.query(SqlMetricSeries).count() == 0

    assert store.list_alert_dimension_values(experiment_id, "assessment_value", "ASSESSMENTS") == [
        "relevance",
        "safety",
    ]
    assert store.list_alert_dimension_values(experiment_id, "error_count", "ERROR") == [
        "TimeoutError"
    ]
    assert store.list_alert_dimension_values(experiment_id, "error_count", "SPAN_NAME") == [
        "search_docs"
    ]
    # Same tool, reached through a different source: `error_count`/SPAN_NAME comes
    # from `span_errors`, `latency`/SPAN_NAME from the TOOL span itself.
    assert store.list_alert_dimension_values(experiment_id, "latency", "SPAN_NAME") == [
        "search_docs"
    ]
    # A dimension with nothing observed stays empty rather than guessing.
    assert store.list_alert_dimension_values(experiment_id, "total_cost", "SPAN_MODEL") == []


def test_dimension_values_are_scoped_to_the_experiment(store, experiment_id):
    other = store.create_experiment("other-experiment")
    now_ms = get_current_time_millis()
    _seed_dimension_value_sources(store, experiment_id, now_ms)

    assert store.list_alert_dimension_values(other, "assessment_value", "ASSESSMENTS") == []


def test_dimension_values_ignore_data_older_than_the_lookback(store, experiment_id):
    """A judge retired last month should not still be offered as a slice."""
    now_ms = get_current_time_millis()
    _seed_dimension_value_sources(store, experiment_id, now_ms - 8 * 24 * 60 * 60 * 1000)

    assert store.list_alert_dimension_values(experiment_id, "assessment_value", "ASSESSMENTS") == []


# ----------------------------------------------------------------------
# RestStore: the third leg of the store trio. Without it the whole feature
# is silently absent for anyone pointing at a remote tracking server, so
# each method is checked to reach the right verb and path.
# ----------------------------------------------------------------------

_ALERT_STORE_METHODS = [
    "create_alert_rule",
    "get_alert_rule",
    "list_alert_rules",
    "update_alert_rule",
    "delete_alert_rule",
    "list_alert_instances",
    "dismiss_alert_instance",
    "list_alert_dimension_values",
]


@pytest.fixture
def rest_store():
    from mlflow.store.tracking.rest_store import RestStore
    from mlflow.utils.rest_utils import MlflowHostCreds

    return RestStore(lambda: MlflowHostCreds("https://tracking-server"))


@pytest.mark.parametrize("method_name", _ALERT_STORE_METHODS)
def test_rest_store_implements_every_alerting_method(rest_store, method_name):
    from mlflow.store.tracking.abstract_store import AbstractStore

    assert getattr(type(rest_store), method_name) is not getattr(AbstractStore, method_name)


def _rest_response(payload):
    response = mock.MagicMock()
    response.json.return_value = payload
    return response


def test_rest_store_create_posts_the_rule_and_parses_the_response(rest_store):
    rule = _rule("7", alert_rule_id="rule-1")
    with mock.patch(
        "mlflow.store.tracking.rest_store.http_request_safe",
        return_value=_rest_response({"alert_rule": dataclasses.asdict(rule)}),
    ) as mock_request:
        created = rest_store.create_alert_rule(rule)

    mock_request.assert_called_once()
    _, endpoint, method = mock_request.call_args[0]
    assert endpoint == "/ajax-api/3.0/mlflow/alerts/rules"
    assert method == "POST"
    assert mock_request.call_args[1]["json"]["name"] == rule.name
    assert created == rule


@pytest.mark.parametrize(
    ("call", "expected_endpoint", "expected_method", "payload"),
    [
        (
            lambda s: s.get_alert_rule("rule-1"),
            "/ajax-api/3.0/mlflow/alerts/rules/rule-1",
            "GET",
            {"alert_rule": dataclasses.asdict(_rule("7", alert_rule_id="rule-1"))},
        ),
        (
            lambda s: s.list_alert_rules("7"),
            "/ajax-api/3.0/mlflow/alerts/rules",
            "GET",
            {"alert_rules": []},
        ),
        (
            lambda s: s.update_alert_rule("rule-1", threshold=1.0),
            "/ajax-api/3.0/mlflow/alerts/rules/rule-1",
            "PATCH",
            {"alert_rule": dataclasses.asdict(_rule("7", alert_rule_id="rule-1"))},
        ),
        (
            lambda s: s.delete_alert_rule("rule-1"),
            "/ajax-api/3.0/mlflow/alerts/rules/rule-1",
            "DELETE",
            {},
        ),
        (
            lambda s: s.list_alert_instances("7"),
            "/ajax-api/3.0/mlflow/alerts/instances",
            "GET",
            {"alert_instances": []},
        ),
        (
            lambda s: s.dismiss_alert_instance("inst-1", "alice"),
            "/ajax-api/3.0/mlflow/alerts/instances/inst-1/dismiss",
            "POST",
            {
                "alert_instance": {
                    "alert_instance_id": "inst-1",
                    "alert_rule_id": "rule-1",
                    "experiment_id": 7,
                    "state": "DISMISSED",
                    "started_at_ms": 1,
                    "window_start_ms": 1,
                    "window_end_ms": 2,
                }
            },
        ),
        (
            lambda s: s.list_alert_dimension_values("7", "assessment_value", "ASSESSMENTS"),
            "/ajax-api/3.0/mlflow/alerts/dimension-values",
            "GET",
            {"dimension_values": ["safety"]},
        ),
    ],
)
def test_rest_store_targets_the_documented_endpoints(
    rest_store, call, expected_endpoint, expected_method, payload
):
    with mock.patch(
        "mlflow.store.tracking.rest_store.http_request_safe",
        return_value=_rest_response(payload),
    ) as mock_request:
        call(rest_store)

    mock_request.assert_called_once()
    _, endpoint, method = mock_request.call_args[0]
    assert endpoint == expected_endpoint
    assert method == expected_method


def test_rest_store_drops_unknown_response_fields(rest_store):
    payload = dataclasses.asdict(_rule("7", alert_rule_id="rule-1"))
    payload["field_from_a_newer_server"] = 1
    with mock.patch(
        "mlflow.store.tracking.rest_store.http_request_safe",
        return_value=_rest_response({"alert_rule": payload}),
    ) as mock_request:
        rule = rest_store.get_alert_rule("rule-1")

    mock_request.assert_called_once()
    assert rule.alert_rule_id == "rule-1"


# ----------------------------------------------------------------------
# The evaluation side: the AlertEvaluationStore protocol in evaluator.py,
# plus the raw verifier behind bounds-then-verify. Not on rest_store --
# these only ever run in-process on the server.
# ----------------------------------------------------------------------


def test_only_due_enabled_undeleted_rules_come_back(store, experiment_id):
    now_ms = get_current_time_millis()
    due = store.create_alert_rule(_rule(experiment_id, name="due"))
    not_due = store.create_alert_rule(_rule(experiment_id, name="not-due"))
    disabled = store.create_alert_rule(_rule(experiment_id, name="disabled", enabled=False))
    deleted = store.create_alert_rule(_rule(experiment_id, name="deleted"))
    store.delete_alert_rule(deleted.alert_rule_id)

    with store.ManagedSessionMaker(read_only=False) as session:
        from mlflow.store.tracking.dbmodels.models import SqlAlertRule

        for rule_id, next_ms in (
            (due.alert_rule_id, now_ms - 1),
            (disabled.alert_rule_id, now_ms - 1),
            (deleted.alert_rule_id, now_ms - 1),
            (not_due.alert_rule_id, now_ms + 600_000),
        ):
            session.query(SqlAlertRule).filter(SqlAlertRule.alert_rule_id == rule_id).update({
                SqlAlertRule.next_evaluation_at_ms: next_ms
            })

    assert [r.alert_rule_id for r in store.due_alert_rules(now_ms)] == [due.alert_rule_id]

    # Reading is not claiming: asking twice returns the same rule. Cycles are
    # serialized by `alert-evaluator-lock`, not by a lease.
    assert [r.alert_rule_id for r in store.due_alert_rules(now_ms)] == [due.alert_rule_id]


def test_record_evaluated_updates_the_rule(store, experiment_id):
    rule = store.create_alert_rule(_rule(experiment_id))
    now_ms = get_current_time_millis()

    store.record_alert_rule_evaluated(
        alert_rule_id=rule.alert_rule_id,
        last_evaluated_ms=now_ms,
        next_evaluation_at_ms=now_ms + 300_000,
        last_sample_count=412,
    )

    updated = store.get_alert_rule(rule.alert_rule_id)
    assert updated.last_evaluated_ms == now_ms
    assert updated.next_evaluation_at_ms == now_ms + 300_000
    assert updated.last_sample_count == 412


def test_save_alert_instance_inserts_then_updates(store, experiment_id):
    rule = store.create_alert_rule(_rule(experiment_id))
    instance = AlertInstance(
        alert_instance_id="inst-1",
        alert_rule_id=rule.alert_rule_id,
        experiment_id=int(experiment_id),
        state="PENDING",
        started_at_ms=1,
        window_start_ms=1,
        window_end_ms=2,
        observed_value=50.0,
        peak_value=50.0,
        threshold=rule.threshold,
        sample_count=300,
    )
    saved = store.save_alert_instance(instance)
    assert saved.state == "PENDING"
    assert store.get_open_alert_instance(rule.alert_rule_id).alert_instance_id == "inst-1"

    # A later breach updates the open instance rather than opening a second one:
    # the user already knows, and re-notifying on every crossing is what makes an
    # alerting product get muted.
    saved = store.save_alert_instance(
        dataclasses.replace(saved, state="FIRED", fired_at_ms=9, peak_value=120.0)
    )
    assert saved.state == "FIRED"
    assert saved.peak_value == 120.0
    assert len(store.list_alert_instances(experiment_id)) == 1


def test_get_open_alert_instance_ignores_dismissed(store, experiment_id):
    rule = store.create_alert_rule(_rule(experiment_id))
    instance_id = _open_instance(store, rule)
    assert store.get_open_alert_instance(rule.alert_rule_id) is not None

    store.dismiss_alert_instance(instance_id, "alice")

    assert store.get_open_alert_instance(rule.alert_rule_id) is None


def test_raw_verifier_counts_traces_above_the_threshold(store, experiment_id):
    from mlflow.genai.alerts.entities import SeriesKey
    from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyRawValueVerifier

    with store.ManagedSessionMaker(read_only=False) as session:
        from mlflow.store.tracking.dbmodels.models import SqlTraceInfo

        for i, (execution_ms, status) in enumerate([
            (100, "OK"),
            (5_000, "OK"),
            (9_000, "OK"),
            (9_000, "ERROR"),
        ]):
            session.add(
                SqlTraceInfo(
                    request_id=f"tr-{i}",
                    experiment_id=int(experiment_id),
                    timestamp_ms=1_000,
                    execution_time_ms=execution_ms,
                    end_time_ms=1_000 + execution_ms,
                    status=status,
                )
            )
        # Outside the window on the exclusive end: half-open [start, end).
        session.add(
            SqlTraceInfo(
                request_id="tr-late",
                experiment_id=int(experiment_id),
                timestamp_ms=1_000,
                execution_time_ms=9_000,
                end_time_ms=60_000,
                status="OK",
            )
        )

    verifier = SqlAlchemyRawValueVerifier(store)
    series = SeriesKey(
        dimension_key="TRACES", experiment_id=int(experiment_id), metric_key="latency"
    )
    assert verifier.count_above(series, 0, 60_000, 1_000) == 3
    assert verifier.count_above(series, 0, 60_000, 9_000) == 0

    scoped = SeriesKey(
        dimension_key="TRACES",
        experiment_id=int(experiment_id),
        metric_key="latency",
        dimension_value="ERROR",
    )
    assert verifier.count_above(scoped, 0, 60_000, 1_000) == 1


def test_raw_verifier_rejects_metrics_it_cannot_answer_exactly(store):
    from mlflow.genai.alerts.entities import SeriesKey
    from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyRawValueVerifier

    verifier = SqlAlchemyRawValueVerifier(store)
    with pytest.raises(ValueError, match="only implemented for latency"):
        verifier.count_above(
            SeriesKey(dimension_key="SPAN_MODEL", experiment_id=0, metric_key="total_cost"),
            0,
            1,
            1.0,
        )


# ----------------------------------------------------------------------
# The rollup's input columns. These are denormalized onto the ingest
# tables so the 1-minute aggregator scans each source once with no joins;
# unwritten, every scan finds zero rows and every alert stays silent.
# ----------------------------------------------------------------------


def test_start_trace_materializes_completion_time_and_denormalizes_metrics(store, experiment_id):
    import json as json_module

    from mlflow.entities import trace_location
    from mlflow.entities.trace_info import TraceInfo
    from mlflow.entities.trace_state import TraceState
    from mlflow.store.tracking.dbmodels.models import SqlTraceInfo, SqlTraceMetrics
    from mlflow.tracing.constant import TraceMetadataKey

    trace_info = store.start_trace(
        TraceInfo(
            trace_id="tr-rollup-1",
            trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
            request_time=1_700_000_000_000,
            execution_duration=4_500,
            state=TraceState.OK,
            trace_metadata={
                TraceMetadataKey.TOKEN_USAGE: json_module.dumps({
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                })
            },
        )
    )

    with store.ManagedSessionMaker() as session:
        row = (
            session.query(SqlTraceInfo).filter(SqlTraceInfo.request_id == trace_info.trace_id).one()
        )
        # Buckets are keyed on completion, not start: a start-time bucket would
        # stay open until the slowest trace in it finished.
        assert row.end_time_ms == 1_700_000_000_000 + 4_500

        metrics = (
            session
            .query(SqlTraceMetrics)
            .filter(SqlTraceMetrics.request_id == trace_info.trace_id)
            .all()
        )
        assert metrics
        for metric in metrics:
            assert metric.experiment_id == int(experiment_id)
            assert metric.timestamp_ms == row.end_time_ms


def test_create_assessment_denormalizes_the_experiment(store, experiment_id):
    from mlflow.entities import trace_location
    from mlflow.entities.assessment import Feedback
    from mlflow.entities.trace_info import TraceInfo
    from mlflow.entities.trace_state import TraceState
    from mlflow.store.tracking.dbmodels.models import SqlAssessments

    store.start_trace(
        TraceInfo(
            trace_id="tr-assessed",
            trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
            request_time=1_700_000_000_000,
            execution_duration=10,
            state=TraceState.OK,
        )
    )
    created = store.create_assessment(Feedback(trace_id="tr-assessed", name="safety", value=1.0))

    with store.ManagedSessionMaker() as session:
        row = (
            session
            .query(SqlAssessments)
            .filter(SqlAssessments.assessment_id == created.assessment_id)
            .one()
        )
        # The quality rollup already buckets on `created_timestamp`; this was the
        # only column it was missing to avoid joining back to trace_info.
        assert row.experiment_id == int(experiment_id)


def test_log_spans_denormalizes_completion_time_and_span_metrics(store, experiment_id):
    from mlflow.store.tracking.dbmodels.models import SqlSpanMetrics, SqlTraceInfo
    from mlflow.tracing.constant import SpanAttributeKey

    from tests.store.tracking.sqlalchemy_store.conftest import create_test_span

    span = create_test_span(
        "tr-spans-rollup",
        start_ns=1_700_000_000_000_000_000,
        end_ns=1_700_000_004_000_000_000,
        attributes={SpanAttributeKey.LLM_COST: {"total_cost": 0.25}},
    )
    store.log_spans(experiment_id, [span])

    with store.ManagedSessionMaker() as session:
        trace_row = (
            session.query(SqlTraceInfo).filter(SqlTraceInfo.request_id == "tr-spans-rollup").one()
        )
        assert trace_row.end_time_ms == 1_700_000_004_000

        metric_rows = (
            session.query(SqlSpanMetrics).filter(SqlSpanMetrics.trace_id == "tr-spans-rollup").all()
        )
        assert metric_rows
        for row in metric_rows:
            # Resolved per trace rather than from the call's `location`: one
            # log_spans call can touch traces living in other experiments.
            assert row.experiment_id == int(experiment_id)
            assert row.timestamp_ms == 1_700_000_004_000


# ----------------------------------------------------------------------
# Notifications: in-app only. The committed instance row *is* the
# notification, so the channel is a no-op and durability comes from
# ordering (commit, then dispatch) rather than a delivery table.
# ----------------------------------------------------------------------


def test_in_app_channel_is_registered_and_does_nothing():
    from mlflow.genai.alerts.notifications import (
        IN_APP_CHANNEL_TYPE,
        InAppChannel,
        get_notification_channel_registry,
    )

    registry = get_notification_channel_registry()
    assert IN_APP_CHANNEL_TYPE in registry.registered_types()
    assert isinstance(registry.get_channel(IN_APP_CHANNEL_TYPE), InAppChannel)


def test_dispatch_defaults_to_in_app_and_survives_a_failing_channel():
    from mlflow.genai.alerts.entities import AlertInstance
    from mlflow.genai.alerts.notifications import (
        NotificationChannel,
        dispatch,
        get_notification_channel_registry,
    )

    sent = []

    class RecordingChannel(NotificationChannel):
        def send(self, rule, instance):
            sent.append(instance.alert_instance_id)

    class BrokenChannel(NotificationChannel):
        def send(self, rule, instance):
            raise RuntimeError("delivery failed")

    registry = get_notification_channel_registry()
    registry.register("recording", RecordingChannel())
    registry.register("broken", BrokenChannel())

    rule = _rule("1", alert_rule_id="rule-1")
    instance = AlertInstance(
        alert_instance_id="inst-1",
        alert_rule_id="rule-1",
        experiment_id=1,
        state="FIRED",
        started_at_ms=1,
        window_start_ms=1,
        window_end_ms=2,
    )

    # No channels configured: in-app, which cannot fail.
    dispatch(rule, instance)

    # A channel that raises is logged and skipped -- the alert is already
    # recorded, and failing the cycle over an undeliverable message would lose
    # the next cycle's work too.
    rule.channels = [{"type": "broken"}, {"type": "recording"}]
    dispatch(rule, instance)
    assert sent == ["inst-1"]

    rule.channels = [{"type": "not_registered"}]
    dispatch(rule, instance)
    assert sent == ["inst-1"]


def test_the_alerting_schema_declares_no_lease_columns():
    """Alerting used to coordinate across replicas; nothing else in MLflow does.

    Every other periodic task settles for ``huey.lock_task`` and a single consumer,
    which is the deployment shape the product supports. Alerting carried lease
    columns, ``hashtext`` sticky assignment, ``SKIP LOCKED`` and a dialect-split
    query for a shape nothing else offered -- and ``rollup_state``'s lease was never
    read or written at all.

    This pins the removal: reintroducing a lease should be a deliberate decision,
    not something that reappears because a future change assumed it was still there.
    """
    from mlflow.store.tracking.dbmodels.models import SqlAlertInstance, SqlAlertRule, SqlRollupState

    for model in (SqlAlertRule, SqlAlertInstance, SqlRollupState):
        columns = {c.name for c in model.__table__.columns}
        assert not {c for c in columns if "lease" in c}, f"{model.__name__} grew a lease column"
        indexes = {i.name for i in model.__table__.indexes}
        assert "index_rollup_state_claim" not in indexes
