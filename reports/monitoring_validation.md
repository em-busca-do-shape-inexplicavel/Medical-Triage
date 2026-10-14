# Monitoring validation

Validated locally on 2026-10-14, using the real saved model and
`compose.monitoring.yaml`.

- Ruff passed for `src`, `tests`, and `scripts`.
- 13 tests passed. One existing Starlette/httpx deprecation warning remains.
- Compose configuration and Prometheus `promtool check config` passed.
- API Docker image built successfully and its health check passed.
- Prometheus reported the `medical-triage-api` target as `up`, with no scrape error.
- Grafana provisioned the `medical-triage-api` dashboard with five panels.
- Grafana's datasource health check successfully queried Prometheus.
- All six dashboard query expressions executed successfully against Prometheus.
- 80 demo requests were sent: 72 valid requests and 8 intentional 422 responses.
- The prediction request counter reached 80; p50/p95 queries returned values,
  client errors were visible, server errors were zero, and scrape availability was 1.

Rate panels use sliding windows and may not show exactly the overall 10% client
error fraction, especially near the start or end of traffic. Histogram percentiles
are approximate server-side measurements, not a replacement for a controlled
optimization benchmark. No new model-quality evaluation was performed.

The updated CI smoke test also checks metrics exposure. Remote GitHub Actions
execution for these uncommitted changes has not yet occurred.
