"""Tests for metrics collection and aggregation."""

import json
import os
import tempfile
from dataclasses import dataclass

from src.metrics_collector import MetricsStore


@dataclass
class FakeMetrics:
    retrieval_latency_ms: float = 100
    llm_latency_ms: float = 500
    total_latency_ms: float = 650
    has_citations: bool = True
    declined_to_answer: bool = False
    error: str | None = None


def test_metrics_store_empty():
    store = MetricsStore()
    summary = store.get_summary()
    assert summary == {"message": "No data yet"}


def test_metrics_store_record_single():
    store = MetricsStore()
    store.record(FakeMetrics())
    summary = store.get_summary()
    assert summary["total_requests"] == 1
    assert summary["citation_coverage"] == 1.0
    assert summary["error_rate"] == 0


def test_metrics_store_record_multiple():
    store = MetricsStore()
    store.record(FakeMetrics(has_citations=True))
    store.record(FakeMetrics(has_citations=False))
    store.record(FakeMetrics(has_citations=True, error="timeout"))
    summary = store.get_summary()
    assert summary["total_requests"] == 3
    assert summary["citation_coverage"] == round(2 / 3, 4)
    assert summary["error_rate"] == round(1 / 3, 4)


def test_metrics_store_percentiles():
    store = MetricsStore()
    for latency in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        store.record(
            FakeMetrics(
                total_latency_ms=latency,
                retrieval_latency_ms=latency * 0.2,
                llm_latency_ms=latency * 0.8,
            )
        )
    summary = store.get_summary()
    assert "total_latency_ms" in summary
    assert "p50" in summary["total_latency_ms"]
    assert "p95" in summary["total_latency_ms"]
    assert summary["total_latency_ms"]["p50"] <= summary["total_latency_ms"]["p95"]


def test_metrics_store_save(tmp_path):
    store = MetricsStore()
    store.record(FakeMetrics())
    path = str(tmp_path / "metrics.json")
    store.save(path)
    with open(path) as f:
        data = json.load(f)
    assert data["total_requests"] == 1
