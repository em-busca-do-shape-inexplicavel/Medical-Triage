import pytest
from fastapi.testclient import TestClient

from medical_triage.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
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