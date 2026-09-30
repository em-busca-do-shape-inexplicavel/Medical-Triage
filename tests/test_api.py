from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from medical_triage import api

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
    assert body["condition_name"] == label_mapping[
        body["condition_label"]
    ]


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

    app.state.pipeline.predict.assert_called_once_with(
        ["cardiac artery study"]
    )