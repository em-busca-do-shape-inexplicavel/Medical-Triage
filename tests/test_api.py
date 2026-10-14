from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from medical_triage import api
from medical_triage.metrics import REQUEST_COUNT, REQUEST_DURATION

app = api.app


@pytest.fixture
def client(monkeypatch, tmp_path):
    model_path = tmp_path / "model.joblib"
    model_path.touch()

    fake_pipeline = Mock()
    fake_pipeline.predict.return_value = [4]

    artifact = {
        "pipeline": fake_pipeline,
        "label_mapping": {
            4: "cardiovascular diseases",
        },
    }

    load_mock = Mock(return_value=artifact)

    monkeypatch.setattr(api, "MODEL_PATH", model_path)
    monkeypatch.setattr(api.joblib, "load", load_mock)

    with TestClient(app) as test_client:
        load_mock.assert_called_once_with(model_path)
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_category(client):
    response = client.post(
        "/predict",
        json={
            "text": (
                "This study evaluates treatment outcomes "
                "in patients with coronary artery disease."
            )
        },
    )

    assert response.status_code == 200

    body = response.json()
    label_mapping = app.state.label_mapping

    assert set(body) == {"condition_label", "condition_name"}
    assert body["condition_label"] in label_mapping
    assert body["condition_name"] == label_mapping[body["condition_label"]]


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_predict_rejects_blank_text(client, text):
    response = client.post(
        "/predict",
        json={"text": text},
    )

    assert response.status_code == 422


def test_predict_requires_text(client):
    response = client.post(
        "/predict",
        json={},
    )

    assert response.status_code == 422


def test_predict_normalizes_text(client):
    response = client.post(
        "/predict",
        json={"text": "  cardiac   artery\nstudy  "},
    )

    assert response.status_code == 200
    assert response.json() == {
        "condition_label": 4,
        "condition_name": "cardiovascular diseases",
    }

    app.state.pipeline.predict.assert_called_once_with(["cardiac artery study"])


def test_metrics_record_success_and_validation_errors(client):
    from prometheus_client import REGISTRY

    labels = {"method": "POST", "route": "/predict", "status": "200"}
    name = "medical_triage_http_requests_total"
    before = REGISTRY.get_sample_value(name, labels) or 0
    invalid_labels = {**labels, "status": "422"}
    invalid_before = REGISTRY.get_sample_value(name, invalid_labels) or 0
    client.post("/predict", json={"text": "cardiac study"})
    client.post("/predict", json={"text": "   "})
    assert REGISTRY.get_sample_value(name, labels) == before + 1
    assert REGISTRY.get_sample_value(name, invalid_labels) == invalid_before + 1
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "medical_triage_http_request_duration_seconds_bucket" in response.text
    assert "cardiac study" not in response.text


def test_metrics_exclude_scrapes_and_bound_unknown_routes(client):
    from prometheus_client import REGISTRY

    name = "medical_triage_http_requests_total"
    labels = {"method": "GET", "route": "unmatched", "status": "404"}
    before = REGISTRY.get_sample_value(name, labels) or 0
    client.get("/unknown-first")
    client.get("/unknown-second")
    assert REGISTRY.get_sample_value(name, labels) == before + 2

    def samples():
        return [
            sample
            for metric in (REQUEST_COUNT, REQUEST_DURATION)
            for family in metric.collect()
            for sample in family.samples
        ]

    snapshot = samples()
    client.get("/metrics")
    client.get("/metrics")
    assert samples() == snapshot
    assert all(sample.labels.get("route") != "/unknown-first" for sample in snapshot)


def test_metrics_record_internal_errors(client):
    from prometheus_client import REGISTRY

    labels = {"method": "POST", "route": "/predict", "status": "500"}
    name = "medical_triage_http_requests_total"
    before = REGISTRY.get_sample_value(name, labels) or 0
    app.state.pipeline.predict.side_effect = RuntimeError("Inference failed")
    with pytest.raises(RuntimeError, match="Inference failed"):
        client.post("/predict", json={"text": "cardiac study"})
    assert REGISTRY.get_sample_value(name, labels) == before + 1
