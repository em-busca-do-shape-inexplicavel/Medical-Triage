"""Single-process HTTP metrics; never label metrics with submitted text."""

from time import perf_counter

from prometheus_client import Counter, Histogram
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_COUNT = Counter(
    "medical_triage_http_requests_total",
    "Completed HTTP requests by route and status.",
    ["method", "route", "status"],
)
REQUEST_DURATION = Histogram(
    "medical_triage_http_request_duration_seconds",
    "Server-side request duration, including response body handling.",
    ["method", "route"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)


class MetricsMiddleware:
    """Measure HTTP requests without buffering request or response bodies."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http" or scope["path"] == "/metrics":
            await self.app(scope, receive, send)
            return

        started = perf_counter()
        status = 500

        async def capture_status(message: Message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, capture_status)
        finally:
            route = getattr(scope.get("route"), "path", "unmatched")
            method = scope["method"]
            if method not in {
                "GET",
                "POST",
                "PUT",
                "PATCH",
                "DELETE",
                "HEAD",
                "OPTIONS",
            }:
                method = "OTHER"
            REQUEST_COUNT.labels(method, route, str(status)).inc()
            REQUEST_DURATION.labels(method, route).observe(perf_counter() - started)
