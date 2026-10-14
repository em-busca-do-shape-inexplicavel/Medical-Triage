"""Compare fixed sklearn and ONNX pipelines on identical inputs and hardware."""

import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import httpx
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from threadpoolctl import threadpool_limits

from medical_triage.onnx_inference import OnnxPredictor


def summary(values):
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def measure(functions, texts, repeats, warmups):
    for fn in functions.values():
        for index in range(warmups):
            fn(texts[index % len(texts)])
    timings = {name: [] for name in functions}
    for repeat in range(repeats):
        current = {name: [] for name in functions}
        for index, text in enumerate(texts):
            order = list(functions)
            if (index + repeat) % 2:
                order.reverse()
            for name in order:
                start = perf_counter()
                functions[name](text)
                current[name].append((perf_counter() - start) * 1000)
        for name in functions:
            timings[name].append(current[name])
    return {
        name: {
            **summary(np.asarray(rounds).ravel()),
            "rounds": [summary(values) for values in rounds],
            "latencies_ms": rounds,
        }
        for name, rounds in timings.items()
    }


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--sklearn-url", default="http://127.0.0.1:8001")
    parser.add_argument("--onnx-url", default="http://127.0.0.1:8002")
    parser.add_argument(
        "--output", type=Path, default=root / "reports/optimization_benchmark.json"
    )
    args = parser.parse_args()
    if min(args.samples, args.repeats, args.warmups) < 1:
        parser.error("samples, repeats and warmups must be positive")
    original = root / "models/medical_triage_model.joblib"
    converted = root / "models/medical_triage_model.onnx"
    metadata = json.loads(converted.with_suffix(".onnx.json").read_text())
    assert (
        hashlib.sha256(original.read_bytes()).hexdigest() == metadata["source_sha256"]
    )
    assert hashlib.sha256(converted.read_bytes()).hexdigest() == metadata["onnx_sha256"]
    sklearn_model = joblib.load(original)["pipeline"]
    onnx_model = OnnxPredictor(converted)
    validation_path = root / "data/processed/validation.csv"
    test_path = root / "data/processed/test.csv"
    validation = pd.read_csv(validation_path)
    test = pd.read_csv(test_path)
    quality = {}
    # No fitting or hyperparameter selection. Re-evaluation checks conversion fidelity.
    with threadpool_limits(limits=1):
        for split, data in {"validation": validation, "test": test}.items():
            texts = data["medical_abstract"].tolist()
            reference = sklearn_model.predict(texts)
            candidate = np.concatenate(
                [
                    onnx_model.predict(texts[i : i + 16])
                    for i in range(0, len(texts), 16)
                ]
            )
            quality[split] = {
                "rows": len(data),
                "agreement": float(np.mean(reference == candidate)),
                "disagreements": int(np.sum(reference != candidate)),
                "sklearn": {
                    "accuracy": accuracy_score(data["condition_label"], reference),
                    "macro_f1": f1_score(
                        data["condition_label"], reference, average="macro"
                    ),
                },
                "onnx": {
                    "accuracy": accuracy_score(data["condition_label"], candidate),
                    "macro_f1": f1_score(
                        data["condition_label"], candidate, average="macro"
                    ),
                },
            }
        texts = (
            validation["medical_abstract"]
            .sample(args.samples, random_state=42)
            .tolist()
        )
        timings = measure(
            {
                "sklearn": lambda text: sklearn_model.predict([text]),
                "onnx": lambda text: onnx_model.predict([text]),
            },
            texts,
            args.repeats,
            args.warmups,
        )
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpus": os.cpu_count(),
        "thread_limit_in_process": 1,
        "versions": {
            p: version(p) for p in ["numpy", "scikit-learn", "onnxruntime", "skl2onnx"]
        },
        "source_sha256": metadata["source_sha256"],
        "onnx_sha256": metadata["onnx_sha256"],
        "data_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [validation_path, test_path]
        },
        "artifact_bytes": {
            "sklearn": original.stat().st_size,
            "onnx": converted.stat().st_size,
        },
        "samples_per_round": args.samples,
        "repeats": args.repeats,
        "warmups_per_backend": args.warmups,
        "sample_seed": 42,
        "batch_size": 1,
        "concurrency": 1,
        "quality": quality,
        "in_process": timings,
        "notes": "Alternating backend order per request/round; model loading excluded. Validation was used in final fitting; test metrics here only check conversion, not model selection. HTTP timings include network/serialization; not comparable to in-process timings. Unicode behavior is not guaranteed equivalent.",
    }
    if args.http:
        with httpx.Client(timeout=30, trust_env=False) as client:

            def http_predict(url, text):
                response = client.post(url + "/predict", json={"text": text})
                response.raise_for_status()
                return response.json()["condition_label"]

            # Confirm the running services match the local artifacts on this sample.
            for text in texts:
                assert http_predict(args.sklearn_url, text) == int(
                    sklearn_model.predict([text])[0]
                )
                assert http_predict(args.onnx_url, text) == int(
                    onnx_model.predict([text])[0]
                )
            report["http"] = measure(
                {
                    "sklearn": lambda text: http_predict(args.sklearn_url, text),
                    "onnx": lambda text: http_predict(args.onnx_url, text),
                },
                texts,
                args.repeats,
                args.warmups,
            )
            report["http_urls"] = {"sklearn": args.sklearn_url, "onnx": args.onnx_url}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(quality, indent=2))
    for scope in ["in_process", "http"]:
        if scope in report:
            for name, result in report[scope].items():
                print(
                    scope, name, {k: result[k] for k in ["mean_ms", "p50_ms", "p95_ms"]}
                )
    print(f"Report saved: {args.output}")


if __name__ == "__main__":
    main()
