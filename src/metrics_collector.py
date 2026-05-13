"""Metrics collection and aggregation for monitoring dashboard."""

import json
import statistics
from pathlib import Path

from prometheus_client import Counter, Histogram


# Prometheus metrics
REQUEST_COUNT = Counter("rag_requests_total", "Total RAG requests")
REQUEST_ERRORS = Counter("rag_request_errors_total", "Total failed requests")
CITATION_HITS = Counter("rag_citation_hits_total", "Requests with proper citations")
DECLINE_COUNT = Counter("rag_decline_total", "Requests where system declined to answer")

RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_ms",
    "Retrieval latency in ms",
    buckets=[10, 25, 50, 100, 250, 500, 1000, 2500],
)
LLM_LATENCY = Histogram(
    "rag_llm_latency_ms",
    "LLM generation latency in ms",
    buckets=[100, 250, 500, 1000, 2500, 5000, 10000],
)
TOTAL_LATENCY = Histogram(
    "rag_total_latency_ms",
    "Total request latency in ms",
    buckets=[100, 250, 500, 1000, 2500, 5000, 10000, 30000],
)


class MetricsStore:
    """In-memory store for computing percentile metrics."""

    def __init__(self):
        self.latencies: list[float] = []
        self.retrieval_latencies: list[float] = []
        self.llm_latencies: list[float] = []
        self.costs: list[float] = []
        self.citation_count: int = 0
        self.total_count: int = 0
        self.error_count: int = 0

    def record(self, metrics):
        """Record metrics from a single request."""
        self.total_count += 1
        self.latencies.append(metrics.total_latency_ms)
        self.retrieval_latencies.append(metrics.retrieval_latency_ms)
        self.llm_latencies.append(metrics.llm_latency_ms)

        REQUEST_COUNT.inc()
        RETRIEVAL_LATENCY.observe(metrics.retrieval_latency_ms)
        LLM_LATENCY.observe(metrics.llm_latency_ms)
        TOTAL_LATENCY.observe(metrics.total_latency_ms)

        if metrics.has_citations:
            self.citation_count += 1
            CITATION_HITS.inc()
        if metrics.declined_to_answer:
            DECLINE_COUNT.inc()
        if metrics.error:
            self.error_count += 1
            REQUEST_ERRORS.inc()

    def get_summary(self) -> dict:
        """Compute summary statistics."""
        if not self.latencies:
            return {"message": "No data yet"}

        def percentiles(data):
            sorted_data = sorted(data)
            p50 = sorted_data[len(sorted_data) // 2]
            p95_idx = int(len(sorted_data) * 0.95)
            p95 = sorted_data[min(p95_idx, len(sorted_data) - 1)]
            return {
                "p50": round(p50, 2),
                "p95": round(p95, 2),
                "mean": round(statistics.mean(data), 2),
            }

        return {
            "total_requests": self.total_count,
            "error_rate": round(self.error_count / self.total_count, 4)
            if self.total_count
            else 0,
            "citation_coverage": round(self.citation_count / self.total_count, 4)
            if self.total_count
            else 0,
            "total_latency_ms": percentiles(self.latencies),
            "retrieval_latency_ms": percentiles(self.retrieval_latencies),
            "llm_latency_ms": percentiles(self.llm_latencies),
        }

    def save(self, path: str = "reports/metrics_summary.json"):
        """Save current metrics to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.get_summary(), f, indent=2)
