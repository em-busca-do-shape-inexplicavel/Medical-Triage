from pathlib import Path
from time import perf_counter
import json

import httpx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "api_latency_baseline.json"

URL = "http://127.0.0.1:8001/predict"
WARMUP_REQUESTS = 10
MEASURED_REQUESTS = 100

validation_df = pd.read_csv(DATA_PATH)

texts = (
    validation_df["medical_abstract"]
    .sample(n=MEASURED_REQUESTS, random_state=42)
    .tolist()
)

durations_ms = []

with httpx.Client(timeout=30.0, trust_env=False) as client:
    for text in texts[:WARMUP_REQUESTS]:
        response = client.post(URL, json={"text": text})
        response.raise_for_status()

    for text in texts:
        start = perf_counter()

        response = client.post(URL, json={"text": text})
        response.raise_for_status()

        durations_ms.append(
            (perf_counter() - start) * 1000
        )

latencies = pd.Series(durations_ms)

results = {
    "url": URL,
    "warmup_requests": WARMUP_REQUESTS,
    "measured_requests": MEASURED_REQUESTS,
    "concurrency": 1,
    "sample_random_state": 42,
    "mean_ms": float(latencies.mean()),
    "p50_ms": float(latencies.quantile(0.50)),
    "p95_ms": float(latencies.quantile(0.95)),
    "max_ms": float(latencies.max()),
    "latencies_ms": durations_ms,
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(
    json.dumps(results, indent=2),
    encoding="utf-8",
)

print(json.dumps(
    {key: value for key, value in results.items() if key != "latencies_ms"},
    indent=2,
))
print(f"Results saved to: {OUTPUT_PATH}")