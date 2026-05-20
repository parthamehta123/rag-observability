"""Tracing instrumentation for RAG pipeline using Langfuse SDK v4.

Langfuse v4 uses OpenTelemetry-style context managers:
- start_as_current_observation() creates a span that auto-nests children
- No manual trace_id passing needed — the SDK handles parent-child relationships
"""

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass

from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "http://localhost:3030"),
)


@dataclass
class PipelineMetrics:
    """Collects metrics for a single RAG request."""

    retrieval_latency_ms: float = 0
    rerank_latency_ms: float = 0
    llm_latency_ms: float = 0
    total_latency_ms: float = 0
    input_tokens: int = 0
    output_tokens: int = 0
    chunks_retrieved: int = 0
    chunks_after_rerank: int = 0
    has_citations: bool = False
    declined_to_answer: bool = False
    error: str | None = None


@contextmanager
def traced_span(name: str, metrics: PipelineMetrics, as_type: str = "span", **kwargs):
    """Context manager: creates a Langfuse span, times it, and records latency."""
    start = time.perf_counter()
    try:
        with langfuse.start_as_current_observation(
            name=name, as_type=as_type, **kwargs
        ) as span:
            yield span
    except Exception as e:
        metrics.error = str(e)
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        setattr(metrics, f"{name}_latency_ms", round(elapsed_ms, 2))


@contextmanager
def traced_pipeline(query: str, user_id: str = "anonymous"):
    """Top-level trace context manager for a full RAG pipeline request."""
    with langfuse.start_as_current_observation(
        name="rag-query",
        as_type="chain",
        input={"query": query},
        metadata={"user_id": user_id},
    ) as trace:
        yield trace


def finalize_pipeline(metrics: PipelineMetrics, answer: str):
    """Update the current trace with final output and metrics."""
    metrics.total_latency_ms = round(
        metrics.retrieval_latency_ms
        + metrics.rerank_latency_ms
        + metrics.llm_latency_ms,
        2,
    )

    cost_per_1k_input = 0.005
    cost_per_1k_output = 0.015
    estimated_cost = (metrics.input_tokens / 1000) * cost_per_1k_input + (
        metrics.output_tokens / 1000
    ) * cost_per_1k_output

    langfuse.update_current_span(
        output={"answer": answer[:500]},
        metadata={
            "retrieval_latency_ms": metrics.retrieval_latency_ms,
            "llm_latency_ms": metrics.llm_latency_ms,
            "total_latency_ms": metrics.total_latency_ms,
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "estimated_cost_usd": round(estimated_cost, 6),
            "has_citations": metrics.has_citations,
            "declined_to_answer": metrics.declined_to_answer,
            "chunks_retrieved": metrics.chunks_retrieved,
        },
    )
    langfuse.flush()
