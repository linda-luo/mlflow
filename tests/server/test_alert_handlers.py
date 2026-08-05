import json
from pathlib import Path
from unittest import mock

import pytest
from flask import Flask
from werkzeug.test import Client

from mlflow.genai.alerts.entities import (
    SYSTEM_DISMISS_RULE_DELETED,
    SYSTEM_DISMISS_RULE_EDITED,
)
from mlflow.server import handlers
from mlflow.store.tracking.dbmodels.models import SqlAlertInstance
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils.time import get_current_time_millis

_AJAX = "/ajax-api/3.0/mlflow/alerts"


@pytest.fixture
def store(tmp_path: Path, db_uri: str) -> SqlAlchemyStore:
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir()
    return SqlAlchemyStore(db_uri, artifact_uri.as_uri())


@pytest.fixture
def client(store: SqlAlchemyStore):
    """The real Flask routing table, backed by a real SQLAlchemy store.

    Exercising the handlers through the router (rather than calling them
    directly) is what proves the endpoints are registered and that the
    verb/path pairs resolve -- the failure mode a mocked handler test misses.
    """
    app = Flask(__name__)
    for http_path, handler, methods in handlers.get_endpoints():
        app.add_url_rule(http_path, handler.__name__, handler, methods=methods)
    # The patch is dependency injection, not a call-verification mock: every
    # test below asserts on what the real store actually persisted.
    with mock.patch.object(handlers, "_get_tracking_store", return_value=store):
        yield Client(app)


@pytest.fixture
def experiment_id(store: SqlAlchemyStore) -> str:
    return store.create_experiment("alerting")


def _body(response) -> dict:
    return json.loads(response.data)


def _create_payload(experiment_id: str, **overrides) -> dict:
    payload = {
        "experiment_id": experiment_id,
        "name": "Slow checkout responses",
        "metric_key": "latency",
        "dimension_key": "TRACES",
        "aggregation": "PERCENTILE",
        "percentile_value": 95,
        "comparator": "GT",
        "threshold": 45 * 60_000,
        "window_seconds": 3600,
        "severity": "HIGH",
    }
    payload.update(overrides)
    return payload


def test_rule_round_trips_through_the_api_with_derived_fields(client, experiment_id):
    created = _body(client.post(f"{_AJAX}/rules", json=_create_payload(experiment_id)))[
        "alert_rule"
    ]

    assert created["alert_rule_id"]
    assert created["name"] == "Slow checkout responses"
    assert created["severity"] == "HIGH"
    # Derived server-side; the client never sends these.
    assert created["evaluation_interval_seconds"] == 300
    assert created["min_sample_count"] == 200
    assert created["next_evaluation_at_ms"] is not None
    assert created["threshold"] == 45 * 60_000

    listed = _body(client.get(f"{_AJAX}/rules?experiment_id={experiment_id}"))["alert_rules"]
    assert [r["alert_rule_id"] for r in listed] == [created["alert_rule_id"]]

    fetched = _body(client.get(f"{_AJAX}/rules/{created['alert_rule_id']}"))["alert_rule"]
    assert fetched == created


def test_create_rejects_an_illegal_catalogue_triple(client, experiment_id):
    response = client.post(
        f"{_AJAX}/rules",
        json=_create_payload(experiment_id, metric_key="error_count", dimension_key="ERROR"),
    )
    assert response.status_code == 400
    assert "does not support" in _body(response)["message"]


def test_create_requires_experiment_id(client):
    response = client.post(f"{_AJAX}/rules", json={"name": "x"})
    assert response.status_code == 400


def _rate_payload(experiment_id: str, **overrides) -> dict:
    fields = {
        "metric_key": "error_rate",
        "aggregation": "AVG",
        "percentile_value": None,
        "threshold": 0.05,
    }
    fields.update(overrides)
    return _create_payload(experiment_id, **fields)


@pytest.mark.parametrize("dimension_key", ["TRACES", "SPAN_TYPE", "SPAN_NAME"])
def test_an_error_rate_rule_round_trips(client, experiment_id, dimension_key):
    """The threshold is a fraction on the wire; the form does the percent scaling."""
    payload = _rate_payload(
        experiment_id, name=f"rate {dimension_key}", dimension_key=dimension_key
    )
    response = client.post(f"{_AJAX}/rules", json=payload)

    assert response.status_code == 200
    created = _body(response)["alert_rule"]
    assert created["metric_key"] == "error_rate"
    assert created["threshold"] == 0.05
    # No sketch, so no percentile is derived and no sample floor comes with one.
    assert created["percentile_value"] is None
    assert created["min_sample_count"] == 0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        # A ratio has no distribution to take a percentile of.
        ({"aggregation": "PERCENTILE", "percentile_value": 95}, "does not support"),
        # SUM would read as "failures including propagated", COUNT as invocations --
        # both disagree with metrics that already answer those questions.
        ({"aggregation": "SUM"}, "does not support"),
        ({"aggregation": "COUNT"}, "does not support"),
        # The error rate of ERROR traces is 100% by construction.
        ({"dimension_key": "TRACES", "dimension_value": "ERROR"}, "aggregated across all of"),
        # Sliced by exception type there is no denominator, only share-of-traffic.
        ({"dimension_key": "ERROR"}, "cannot be sliced by"),
    ],
)
def test_an_unanswerable_error_rate_rule_is_rejected(client, experiment_id, overrides, message):
    """Each of these would otherwise be accepted, show as healthy, and never fire.

    That is the same silent failure as a rule naming a series the aggregator never
    writes -- rejecting at creation turns it into a form error instead.
    """
    response = client.post(f"{_AJAX}/rules", json=_rate_payload(experiment_id, **overrides))

    assert response.status_code == 400
    assert message in _body(response)["message"]


@pytest.mark.parametrize("field", ["window_seconds", "sustain_seconds", "threshold"])
def test_a_null_numeric_field_is_a_400_not_a_500(client, experiment_id, field):
    """`JSON.stringify` turns a NaN form field into `null`.

    The type assertions caught only `ValueError`, but `int(None)` and `float(None)`
    raise `TypeError`, and only `AssertionError` is translated upstream -- so a
    user typing letters into the window box got a 500.
    """
    response = client.post(
        f"{_AJAX}/rules", json=_create_payload(experiment_id, **{field: None})
    )
    assert response.status_code == 400


def test_patch_accepts_and_types_the_dimension_value(client, experiment_id):
    """The store lists `dimension_value` as updatable but the patch schema omitted it.

    The handler copies the raw body through, so it reached the store untyped -- a
    client could send an int and provoke a database-level failure.
    """
    rule = _body(
        client.post(
            f"{_AJAX}/rules", json=_create_payload(experiment_id, dimension_value="OK")
        )
    )["alert_rule"]

    patched = _body(
        client.patch(f"{_AJAX}/rules/{rule['alert_rule_id']}", json={"dimension_value": "ERROR"})
    )["alert_rule"]
    assert patched["dimension_value"] == "ERROR"

    assert (
        client.patch(
            f"{_AJAX}/rules/{rule['alert_rule_id']}", json={"dimension_value": 17}
        ).status_code
        == 400
    )


def test_omitting_the_sample_floor_gets_the_statistical_suggestion(client, experiment_id):
    """Only the REST layer can tell "unspecified" from "explicitly zero".

    The store stores the number verbatim, so a client that has never heard of the
    field still gets a sane floor for a percentile.
    """
    response = client.post(f"{_AJAX}/rules", json=_create_payload(experiment_id))
    assert _body(response)["alert_rule"]["min_sample_count"] == 200  # p95


def test_an_explicit_sample_floor_is_honoured_including_zero(client, experiment_id):
    for value in (0, 7, 5000):
        response = client.post(
            f"{_AJAX}/rules",
            json=_create_payload(experiment_id, name=f"floor {value}", min_sample_count=value),
        )
        assert _body(response)["alert_rule"]["min_sample_count"] == value


def test_a_negative_sample_floor_is_rejected(client, experiment_id):
    response = client.post(
        f"{_AJAX}/rules", json=_create_payload(experiment_id, min_sample_count=-1)
    )
    assert response.status_code == 400


def test_patching_the_sample_floor_stops_it_being_re_derived(client, experiment_id):
    """Editing the percentile moves a floor the user never set, but not one they did."""
    rule = _body(client.post(f"{_AJAX}/rules", json=_create_payload(experiment_id)))["alert_rule"]
    assert rule["min_sample_count"] == 200

    moved = _body(
        client.patch(f"{_AJAX}/rules/{rule['alert_rule_id']}", json={"percentile_value": 99})
    )["alert_rule"]
    assert moved["min_sample_count"] == 1000

    kept = _body(
        client.patch(
            f"{_AJAX}/rules/{rule['alert_rule_id']}",
            json={"percentile_value": 50, "min_sample_count": 25},
        )
    )["alert_rule"]
    assert kept["min_sample_count"] == 25


@pytest.mark.parametrize("field", ["percentile_value", "dimension_value"])
def test_create_accepts_an_explicit_null_for_a_nullable_field(client, experiment_id, field):
    """Sending null and omitting the key mean the same thing.

    Rejecting one spelling produced an error naming the wrong field: a create with
    `percentile_value: null` failed on the percentile before any other validation
    ran, so the message pointed nowhere near the real problem.
    """
    overrides = {field: None}
    if field == "percentile_value":
        overrides["aggregation"] = "AVG"
    response = client.post(f"{_AJAX}/rules", json=_create_payload(experiment_id, **overrides))
    assert response.status_code == 200
    assert _body(response)["alert_rule"][field] is None


def test_patch_clearing_a_slice_stores_null_like_create_does(client, experiment_id):
    """"Unscoped" must be one value in the column, not two.

    Create normalizes "" to NULL; a patch used to store the empty string literally,
    so two rules meaning the same thing differed on disk.
    """
    rule = _body(
        client.post(f"{_AJAX}/rules", json=_create_payload(experiment_id, dimension_value="OK"))
    )["alert_rule"]

    for cleared in ("", None):
        patched = _body(
            client.patch(
                f"{_AJAX}/rules/{rule['alert_rule_id']}", json={"dimension_value": cleared}
            )
        )["alert_rule"]
        assert patched["dimension_value"] is None
        client.patch(f"{_AJAX}/rules/{rule['alert_rule_id']}", json={"dimension_value": "OK"})


def test_a_rule_leaving_percentile_can_drop_its_percentile(client, experiment_id):
    rule = _body(client.post(f"{_AJAX}/rules", json=_create_payload(experiment_id)))["alert_rule"]
    assert rule["percentile_value"] == 95

    patched = _body(
        client.patch(
            f"{_AJAX}/rules/{rule['alert_rule_id']}",
            json={"aggregation": "AVG", "percentile_value": None},
        )
    )["alert_rule"]
    assert patched["percentile_value"] is None
    # Derived from the new aggregation: AVG needs no sample floor.
    assert patched["min_sample_count"] == 0


def test_a_percentile_rule_may_not_drop_its_percentile(client, experiment_id):
    rule = _body(client.post(f"{_AJAX}/rules", json=_create_payload(experiment_id)))["alert_rule"]
    response = client.patch(
        f"{_AJAX}/rules/{rule['alert_rule_id']}", json={"percentile_value": None}
    )
    assert response.status_code == 400


def test_a_deleted_rules_name_can_be_reused_through_the_api(client, experiment_id):
    first = _body(client.post(f"{_AJAX}/rules", json=_create_payload(experiment_id)))["alert_rule"]
    client.delete(f"{_AJAX}/rules/{first['alert_rule_id']}")

    response = client.post(f"{_AJAX}/rules", json=_create_payload(experiment_id))
    assert response.status_code == 200
    assert _body(response)["alert_rule"]["alert_rule_id"] != first["alert_rule_id"]


def test_dimension_values_rejects_an_unsliceable_pair(client, experiment_id):
    response = client.get(
        f"{_AJAX}/dimension-values?experiment_id={experiment_id}"
        "&metric=total_tokens&dimension_key=TRACES"
    )
    assert response.status_code == 400
    assert "aggregated across all of" in _body(response)["message"]


def test_patch_threshold_leaves_the_open_instance_open(client, store, experiment_id):
    rule = _body(client.post(f"{_AJAX}/rules", json=_create_payload(experiment_id)))["alert_rule"]
    _seed_open_instance(store, rule)

    patched = _body(
        client.patch(f"{_AJAX}/rules/{rule['alert_rule_id']}", json={"threshold": 30 * 60_000})
    )["alert_rule"]
    assert patched["threshold"] == 30 * 60_000

    instances = _body(client.get(f"{_AJAX}/instances?experiment_id={experiment_id}"))[
        "alert_instances"
    ]
    assert [i["state"] for i in instances] == ["FIRED"]


def test_patch_window_closes_the_open_instance(client, store, experiment_id):
    rule = _body(client.post(f"{_AJAX}/rules", json=_create_payload(experiment_id)))["alert_rule"]
    _seed_open_instance(store, rule)

    client.patch(f"{_AJAX}/rules/{rule['alert_rule_id']}", json={"window_seconds": 7200})

    assert (
        _body(client.get(f"{_AJAX}/instances?experiment_id={experiment_id}"))["alert_instances"]
        == []
    )
    closed = _body(client.get(f"{_AJAX}/instances?experiment_id={experiment_id}&states=DISMISSED"))[
        "alert_instances"
    ]
    assert [i["dismissed_by"] for i in closed] == [SYSTEM_DISMISS_RULE_EDITED]


def test_delete_is_soft_and_the_instance_survives_closed(client, store, experiment_id):
    """Deleting a rule closes its open instances but keeps them readable.

    The user asked to stop hearing about this rule, so nothing should remain in
    the active list -- but the record of what it caught has to survive, since
    that is exactly what a postmortem looks for.
    """
    rule = _body(client.post(f"{_AJAX}/rules", json=_create_payload(experiment_id)))["alert_rule"]
    instance_id = _seed_open_instance(store, rule)

    assert client.delete(f"{_AJAX}/rules/{rule['alert_rule_id']}").status_code == 200
    assert _body(client.get(f"{_AJAX}/rules?experiment_id={experiment_id}"))["alert_rules"] == []
    assert client.get(f"{_AJAX}/rules/{rule['alert_rule_id']}").status_code == 404

    assert (
        _body(client.get(f"{_AJAX}/instances?experiment_id={experiment_id}"))["alert_instances"]
        == []
    )
    closed = _body(client.get(f"{_AJAX}/instances?experiment_id={experiment_id}&states=DISMISSED"))[
        "alert_instances"
    ]
    assert [i["alert_instance_id"] for i in closed] == [instance_id]
    assert closed[0]["dismissed_by"] == SYSTEM_DISMISS_RULE_DELETED


def test_a_recovered_instance_is_still_active_and_still_dismissible(client, store, experiment_id):
    """INACTIVE means "recovered, not acknowledged", so it stays in the active
    list -- and dismissing it is what moves it to history.
    """
    rule = _body(client.post(f"{_AJAX}/rules", json=_create_payload(experiment_id)))["alert_rule"]
    _seed_open_instance(store, rule, state="INACTIVE", instance_id="inst-inactive")
    _seed_open_instance(store, rule, state="FIRED", instance_id="inst-fired")

    active = _body(client.get(f"{_AJAX}/instances?experiment_id={experiment_id}"))[
        "alert_instances"
    ]
    assert {i["state"] for i in active} == {"INACTIVE", "FIRED"}
    assert next(i for i in active if i["state"] == "INACTIVE")["healthy_since_ms"] is not None

    filtered = _body(
        client.get(f"{_AJAX}/instances?experiment_id={experiment_id}&states=INACTIVE")
    )["alert_instances"]
    assert [i["alert_instance_id"] for i in filtered] == ["inst-inactive"]

    dismissed = _body(
        client.post(f"{_AJAX}/instances/inst-inactive/dismiss", json={"dismissed_by": "alice"})
    )["alert_instance"]
    assert dismissed["state"] == "DISMISSED"
    # The record of how bad it got survives the whole lifecycle.
    assert dismissed["peak_value"] == 120.0
    remaining = _body(client.get(f"{_AJAX}/instances?experiment_id={experiment_id}"))[
        "alert_instances"
    ]
    assert [i["alert_instance_id"] for i in remaining] == ["inst-fired"]


def test_dismiss_missing_instance_is_404(client):
    response = client.post(f"{_AJAX}/instances/nope/dismiss", json={"dismissed_by": "alice"})
    assert response.status_code == 404


def test_dimension_values_are_read_from_raw_observations(client, store, experiment_id):
    """Not from `metric_series`: the editor has to work before anything is aggregated."""
    import json

    from mlflow.store.tracking.dbmodels.models import SqlAssessments, SqlTraceInfo
    from mlflow.utils.time import get_current_time_millis

    now_ms = get_current_time_millis()
    with store.ManagedSessionMaker(read_only=False) as session:
        session.add(
            SqlTraceInfo(
                request_id="tr-1",
                experiment_id=int(experiment_id),
                timestamp_ms=now_ms - 61_000,
                execution_time_ms=1_000,
                end_time_ms=now_ms - 60_000,
                status="OK",
            )
        )
        session.flush()
        session.add(
            SqlAssessments(
                assessment_id="asmt-1",
                trace_id="tr-1",
                name="safety",
                assessment_type="feedback",
                value=json.dumps(True),
                created_timestamp=now_ms - 60_000,
                last_updated_timestamp=now_ms - 60_000,
                source_type="LLM_JUDGE",
                valid=True,
                experiment_id=int(experiment_id),
            )
        )

    response = client.get(
        f"{_AJAX}/dimension-values?experiment_id={experiment_id}"
        "&metric=assessment_value&dimension_key=ASSESSMENTS"
    )
    assert _body(response)["dimension_values"] == ["safety"]


def test_endpoints_are_registered_on_both_prefixes(experiment_id):
    paths = {(path, tuple(methods)) for path, _, methods in handlers.get_alert_endpoints()}
    for prefix in ("/api/3.0", "/ajax-api/3.0"):
        assert (f"{prefix}/mlflow/alerts/rules", ("POST",)) in paths
        assert (f"{prefix}/mlflow/alerts/rules", ("GET",)) in paths
        assert (f"{prefix}/mlflow/alerts/rules/<alert_rule_id>", ("PATCH",)) in paths
        assert (f"{prefix}/mlflow/alerts/rules/<alert_rule_id>", ("DELETE",)) in paths
        assert (f"{prefix}/mlflow/alerts/instances", ("GET",)) in paths
        assert (
            f"{prefix}/mlflow/alerts/instances/<alert_instance_id>/dismiss",
            ("POST",),
        ) in paths
        assert (f"{prefix}/mlflow/alerts/dimension-values", ("GET",)) in paths


def _seed_open_instance(
    store: SqlAlchemyStore, rule: dict, state: str = "FIRED", instance_id: str = "inst-1"
) -> str:
    with store.ManagedSessionMaker(read_only=False) as session:
        session.add(
            SqlAlertInstance(
                alert_instance_id=instance_id,
                alert_rule_id=rule["alert_rule_id"],
                experiment_id=rule["experiment_id"],
                state=state,
                started_at_ms=get_current_time_millis(),
                fired_at_ms=get_current_time_millis(),
                healthy_since_ms=get_current_time_millis() if state == "INACTIVE" else None,
                observed_value=99.0,
                peak_value=120.0,
                threshold=rule["threshold"],
                sample_count=250,
                window_start_ms=1,
                window_end_ms=2,
            )
        )
    return instance_id


###############################################################################
# Series
###############################################################################


def test_series_returns_the_threshold_and_the_rule_window(client, experiment_id):
    rule = _body(client.post(f"{_AJAX}/rules", json=_create_payload(experiment_id)))["alert_rule"]

    body = _body(
        client.get(
            f"{_AJAX}/series?alert_rule_id={rule['alert_rule_id']}&start_ms=0&end_ms=600000"
        )
    )

    assert body["threshold"] == rule["threshold"]
    assert body["window_seconds"] == rule["window_seconds"]
    assert body["step_seconds"] > 0


def test_series_points_carry_a_timestamp_and_a_nullable_value(client, experiment_id):
    """A window nobody aggregated is `null`, not `0`.

    The chart has to break the line there; drawing through it invents a slope, and
    for an absence rule a fabricated crash to zero looks exactly like an incident.
    """
    rule = _body(client.post(f"{_AJAX}/rules", json=_create_payload(experiment_id)))["alert_rule"]

    points = _body(
        client.get(
            f"{_AJAX}/series?alert_rule_id={rule['alert_rule_id']}&start_ms=0&end_ms=600000"
        )
    )["points"]

    assert points
    assert all("timestamp_ms" in p and "value" in p for p in points)
    # No rollups exist for this experiment, so a PERCENTILE window has no answer.
    assert all(p["value"] is None for p in points)


def test_series_caps_the_points_it_will_return(client, experiment_id):
    """Three days at the rule's own interval is 864 folds; the step widens instead."""
    from mlflow.genai.alerts.series import MAX_SERIES_POINTS

    rule = _body(client.post(f"{_AJAX}/rules", json=_create_payload(experiment_id)))["alert_rule"]
    three_days_ms = 3 * 24 * 60 * 60 * 1000

    body = _body(
        client.get(
            f"{_AJAX}/series?alert_rule_id={rule['alert_rule_id']}"
            f"&start_ms=0&end_ms={three_days_ms}"
        )
    )

    assert len(body["points"]) <= MAX_SERIES_POINTS


def test_series_rejects_a_backwards_range(client, experiment_id):
    rule = _body(client.post(f"{_AJAX}/rules", json=_create_payload(experiment_id)))["alert_rule"]

    response = client.get(
        f"{_AJAX}/series?alert_rule_id={rule['alert_rule_id']}&start_ms=600000&end_ms=0"
    )

    assert response.status_code == 400
    assert "precedes" in _body(response)["message"]


def test_series_for_a_missing_rule_is_404(client):
    response = client.get(f"{_AJAX}/series?alert_rule_id=nope&start_ms=0&end_ms=600000")
    assert response.status_code == 404
