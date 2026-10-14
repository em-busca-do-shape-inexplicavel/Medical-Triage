"""Optional conversion tests using synthetic data, never the real artifact."""

import json
import subprocess
import sys
from pathlib import Path

import joblib
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


def test_export_with_existing_sidecar(converted_model, tmp_path):
    model, _ = converted_model
    source = tmp_path / "source.joblib"
    joblib.dump({"pipeline": model, "label_mapping": {1: "cardiac", 2: "digestive"}}, source)
    output = tmp_path / "export.onnx"
    sidecar = output.with_suffix(".onnx.json")
    sidecar.write_text('{"old": true}')
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts/export_onnx.py"),
        "--input", str(source), "--output", str(output),
    ]
    refused = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert refused.returncode != 0
    assert "Metadata already exists" in refused.stderr
    assert not output.exists()
    assert json.loads(sidecar.read_text()) == {"old": True}

    command.append("--replace-metadata")
    subprocess.run(command, check=True, capture_output=True, timeout=60)
    assert json.loads(sidecar.read_text())["label_mapping"] == {
        "1": "cardiac", "2": "digestive",
    }
    assert OnnxPredictor(output).predict(["cardiac artery"]).tolist() == [1]
    binary_before, metadata_before = output.read_bytes(), sidecar.read_bytes()
    refused = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert refused.returncode != 0
    assert "ONNX output already exists" in refused.stderr
    assert output.read_bytes() == binary_before
    assert sidecar.read_bytes() == metadata_before
