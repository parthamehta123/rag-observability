"""Tracing instrumentation for RAG pipeline using Langfuse."""

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
    host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
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
def trace_step(name: str, metrics: PipelineMetrics, trace=None):
    """Context manager to trace and time a pipeline step."""
    span = trace.span(name=name) if trace else None
    start = time.perf_counter()
    try:
        yield span
    except Exception as e:
        metrics.error = str(e)
        if span:
            span.update(status_message=str(e), level="ERROR")
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        setattr(metrics, f"{name}_latency_ms", round(elapsed_ms, 2))
        if span:
            span.end()


def create_trace(query: str, user_id: str = "anonymous") -> tuple:
    """Create a new Langfuse trace for a RAG request."""
    trace = langfuse.trace(name="rag-query", input={"query": query}, user_id=user_id)
    metrics = PipelineMetrics()
    return trace, metrics


def finalize_trace(trace, metrics: PipelineMetrics, answer: str):
    """Finalize trace with output and metrics."""
    metrics.total_latency_ms = round(
        metrics.retrieval_latency_ms
        + metrics.rerank_latency_ms
        + metrics.llm_latency_ms,
        2,
    )

    cost_per_1k_input = 0.005  # GPT-4o pricing estimate
    cost_per_1k_output = 0.015
    estimated_cost = (metrics.input_tokens / 1000) * cost_per_1k_input + (
        metrics.output_tokens / 1000
    ) * cost_per_1k_output

    trace.update(
        output={"answer": answer},
        metadata={
            "retrieval_latency_ms": metrics.retrieval_latency_ms,
            "rerank_latency_ms": metrics.rerank_latency_ms,
            "llm_latency_ms": metrics.llm_latency_ms,
            "total_latency_ms": metrics.total_latency_ms,
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "estimated_cost_usd": round(estimated_cost, 6),
            "has_citations": metrics.has_citations,
            "declined_to_answer": metrics.declined_to_answer,
        },
    )
    langfuse.flush()
