# Delivery checklist and STAR presentation

This is a preparation guide, not proof of submission. Video publication, remote
CI status, final PR merge and instructor acceptance must be checked by the team.

## Evidence map

| Topic | Repository evidence |
| --- | --- |
| Data understanding and preparation | `notebooks/01*`, `notebooks/02*`, `src/medical_triage/data_preparation.py` |
| Baseline, experiments and final model | `notebooks/03*` through `05*`, `models/model_metadata.json` |
| Reusable training | `scripts/prepare_data.py`, `scripts/train_model.py` |
| Orchestration | `dags/medical_triage_training.py`, `compose.airflow.yaml` |
| Inference service | `src/medical_triage/api.py`, `Dockerfile`, `tests/test_api.py` |
| Continuous integration | `.github/workflows/ci.yml` |
| Monitoring | `monitoring/`, `compose.monitoring.yaml`, `reports/monitoring_validation.md` |
| Optimization | `scripts/export_onnx.py`, `tests/test_onnx.py`, `reports/optimization_benchmark.json` |
| Cloud design, not deployment | [Architecture decision](cloud_architecture.md) |

## Suggested five-minute STAR script

### 0:00–0:40 — Situation

“Medical abstracts contain specialized language and can be organized into
medical categories. We used the Medical Abstracts TC Corpus to build an
educational classification service. Although the repository is named
Medical-Triage, the labels do not represent clinical urgency.”

Show the README and class names. Do not use personal medical information.

### 0:40–1:10 — Task

“Our task was to develop a reproducible workflow from data preparation to a
containerized inference API, including automated training, tests, monitoring,
and a measured inference optimization.”

Show the evidence map and briefly distinguish implemented local components from
the proposed cloud architecture.

### 1:10–3:50 — Action

- Show preparation: normalize whitespace, remove conflicting labels and exact
  duplicates, then create a new stratified split with seed 42.
- Explain model selection using validation results and macro F1. Show TF-IDF
  plus balanced Logistic Regression with `C=10`; show the final held-out result.
- Show a successful Airflow run and both task logs. Explain run isolation and
  that model promotion remains manual.
- Submit one English abstract in API docs, then blank text to demonstrate 422.
  Avoid promising a specific class for an arbitrary example.
- Show the real PR's CI result. Explain that CI's smoke model is synthetic.
- Generate monitoring traffic and show request rate, latency and errors in
  Grafana. Explain that intentional blank requests are client errors.
- Show the paired optimization report: same fixed texts, warmups, alternating
  backend order, five rounds, and no model retraining during conversion.

Prepare the services beforehand; do not spend recording time waiting for builds.
Never show credentials or tokens on screen.

### 3:50–4:40 — Result

“The original final model reached 0.8129 test accuracy and 0.8084 macro F1.
In our paired local benchmark, ONNX reduced mean in-process latency by about
29.5% and mean HTTP latency by about 11.9%. These are sequential local results,
not cloud throughput guarantees.”

Show `reports/optimization_benchmark.json` or the README results table.

### 4:40–5:00 — Limitations and closing

“ONNX changed four of 1,245 test predictions, so sklearn remains the default.
The system is not a clinical tool. Our cloud architecture is a proposal, not a
deployed service; production security, load testing and quality monitoring are
future work.”

## Before submission

- [ ] Confirm the dataset/task adaptation meets the assignment: categories are
  not urgency labels. Resolve any required scope changes with the instructor.
- [ ] Reconcile this evidence map against the original assignment and rubric.
- [ ] Commit and push the final changes; verify remote CI and merge the PR.
- [ ] Check reproduction instructions from a clean checkout/environment with
  separately obtained raw data. Never replace the original model just to test.
- [ ] Keep the historical reports distinct from newly generated measurements.
- [ ] Record the video, verify its duration against the assignment, upload it,
  and test viewer access without the owner's session.
- [ ] Replace the pending video URL in the README with the actual link.
- [ ] Submit the repository and video URLs through the required course channel.

Stop local services when finished using the Compose `down` commands in the
README. Retain named volumes unless their data is deliberately being discarded.
