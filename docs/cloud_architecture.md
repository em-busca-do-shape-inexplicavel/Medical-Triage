# Cloud architecture decision

Status: proposed, not deployed. The executable reference is the local Docker
Compose stack. There is no cloud infrastructure provisioning or deployment job
in this repository.

## Decision and rationale

Use synchronous online inference for interactive category routing; use batch
jobs for preparation and retraining. A caller needs a response for one abstract,
so batch-only inference does not meet the interaction pattern. The small CPU
pipeline does not require GPU infrastructure. Keep sklearn as the default until
ONNX's changed predictions have been reviewed against an acceptance policy.

Proposed request and release paths:

```text
Authenticated client -> HTTPS load balancer -> private ECS/Fargate API tasks
                                                   |
                                          per-task metrics -> Prometheus -> Grafana

Versioned data -> isolated training -> quality review -> versioned model
                                                          |
                            approved image in ECR -> controlled service release
```

AWS documents support for Application Load Balancers with ECS services running
on Fargate, including HTTP/HTTPS routing. This makes ALB plus Fargate a compatible
candidate for the containerized API; it is not a deployment validated here.
See [AWS ECS load balancing documentation](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/service-load-balancing.html).

## Proposed responsibilities

| Component | Design responsibility |
| --- | --- |
| ALB and authentication layer | HTTPS entry point, authenticated access, request controls |
| ECS/Fargate | Run immutable API images; one worker per task initially |
| ECR | Store approved images identified by digest |
| Versioned object storage | Preserve dataset snapshots, trusted artifacts, hashes and reports |
| Training orchestrator | Run preparation/training separately from request handling |
| Prometheus/Grafana | Scrape each task privately; aggregate requests, errors and latency |
| Logging and alerting | Track operational failures without recording abstract bodies |

These are design choices, not resources created by this project. The existing
Airflow standalone container must not be exposed as a production orchestrator.

## Release and rollback proposal

1. Freeze dataset versions and the code revision; record dependency versions.
2. Train in an isolated run. Evaluate against predeclared quality criteria;
   do not repeatedly tune on the held-out test set.
3. If converting to ONNX, inspect disagreements, test representative input and
   require acceptance of conversion fidelity and resource use.
4. Package the approved artifact in an immutable image, run smoke checks, and
   require human approval. The current CI synthetic model is not a release model.
5. Deploy gradually, check health/error/latency indicators, and roll back to the
   previous image digest if acceptance criteria fail.

The current CI ends at build/smoke testing. Registry publication, release gates,
deployment automation and rollback automation are future implementation work.

## Security and operating limits

Before any public use, add authentication/authorization, TLS, request-size and
rate limits, least-privilege service identities, secret management, and network
restrictions. Keep metrics and orchestration interfaces private. Do not use real
patient information for the demonstration. The API is not clinically validated.

The local dashboard targets one API process. Multiple replicas require task
discovery and aggregation tests; scraping a load-balanced endpoint alone can
miss individual task counters. Add operational alerts and a separately labeled
evaluation dataset to measure model-quality degradation: latency and availability
cannot establish predictive quality or detect all forms of data drift.

Local sequential timings are not throughput estimates or service-level
objectives. Before sizing, measure concurrent load, cold starts, CPU, memory,
and failure behavior. Capacity, region, availability target, budget and retention
remain undecided; no cloud cost or production SLA is claimed.
