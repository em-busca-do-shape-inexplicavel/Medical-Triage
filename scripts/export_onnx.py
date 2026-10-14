"""Export the fixed trained pipeline; never refit or overwrite the original."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

from medical_triage.onnx_inference import OnnxPredictor


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=root / "models/medical_triage_model.joblib"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "models/medical_triage_model.onnx"
    )
    args = parser.parse_args()
    metadata_path = args.output.with_suffix(".onnx.json")
    if args.output.exists() or metadata_path.exists():
        raise FileExistsError(
            "ONNX output or metadata already exists; choose a new output."
        )
    artifact = joblib.load(args.input)
    model = artifact["pipeline"]
    converted = convert_sklearn(
        model,
        initial_types=[("text", StringTensorType([None, 1]))],
        options={
            id(model["tfidf"]): {"locale": "C"},
            id(model["classifier"]): {"zipmap": False},
        },
        target_opset=18,
    )
    onnx.checker.check_model(converted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(converted, args.output)
    predictor = OnnxPredictor(args.output)
    predictor.predict(["Clinical study of coronary artery disease."])
    metadata = {
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "label_mapping": {int(k): v for k, v in artifact["label_mapping"].items()},
        "versions": {
            p: version(p) for p in ["scikit-learn", "onnx", "onnxruntime", "skl2onnx"]
        },
        "opset": 18,
        "locale": "C",
        "notes": "Full text pipeline. Tokenization/casing may differ for non-English Unicode input; verify corpus parity.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Exported {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
