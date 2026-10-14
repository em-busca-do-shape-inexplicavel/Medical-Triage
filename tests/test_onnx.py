"""Optional conversion tests using synthetic data, never the real artifact."""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("onnxruntime")
pytest.importorskip("skl2onnx")

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

from medical_triage import api
from medical_triage.onnx_inference import OnnxPredictor
from medical_triage.training import train_model


@pytest.fixture
def converted_model(tmp_path):
    data = pd.DataFrame(
        {
            "medical_abstract": [
                "cardiac artery study",
                "cardiac artery research",
                "digestive bowel study",
                "digestive bowel research",
            ],
            "condition_label": [1, 1, 2, 2],
        }
    )
    model = train_model(data)
    converted = convert_sklearn(
        model,
        initial_types=[("text", StringTensorType([None, 1]))],
        options={
            id(model["tfidf"]): {"locale": "C"},
            id(model["classifier"]): {"zipmap": False},
        },
        target_opset=18,
    )
    directory = tmp_path / "models"
    directory.mkdir()
    path = directory / "medical_triage_model.onnx"
    path.write_bytes(converted.SerializeToString())
    path.with_suffix(".onnx.json").write_text(
        json.dumps(
            {
                "label_mapping": {"1": "cardiac", "2": "digestive"},
            }
        )
    )
    return model, path


def test_onnx_predicts_ascii_examples(converted_model):
    model, path = converted_model
    texts = ["cardiac artery", "digestive bowel", "CARDIAC ARTERY", "!!!"]
    assert OnnxPredictor(path).predict(texts).tolist() == model.predict(texts).tolist()


def test_api_onnx_backend(converted_model, monkeypatch):
    _, path = converted_model
    monkeypatch.setenv("MODEL_BACKEND", "onnx")
    monkeypatch.setattr(api, "PROJECT_ROOT", path.parent.parent)
    with TestClient(api.app) as client:
        response = client.post("/predict", json={"text": "  CARDIAC   ARTERY "})
        assert response.status_code == 200
        assert response.json() == {"condition_label": 1, "condition_name": "cardiac"}
        assert client.post("/predict", json={"text": " "}).status_code == 422


def test_unknown_backend_fails_startup(monkeypatch):
    monkeypatch.setenv("MODEL_BACKEND", "invalid")
    with pytest.raises(ValueError, match="Unknown MODEL_BACKEND"):
        with TestClient(api.app):
            pass
