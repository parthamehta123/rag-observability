"""Streamlit dashboard for RAG observability metrics."""

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="RAG Observability", layout="wide")
st.title("RAG Observability Dashboard")


def load_metrics(path: str = "reports/metrics_summary.json") -> dict:
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


metrics = load_metrics()

if metrics is None:
    st.warning("No metrics data found. Run some queries first.")
    st.stop()

# KPI cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Requests", metrics["total_requests"])
col2.metric("Error Rate", f"{metrics['error_rate'] * 100:.1f}%")
col3.metric("Citation Coverage", f"{metrics['citation_coverage'] * 100:.1f}%")
col4.metric("P95 Latency (ms)", metrics["total_latency_ms"]["p95"])

st.divider()

# Latency breakdown
st.subheader("Latency Breakdown (ms)")
latency_data = {
    "Stage": ["Retrieval", "LLM", "Total"],
    "P50": [
        metrics["retrieval_latency_ms"]["p50"],
        metrics["llm_latency_ms"]["p50"],
        metrics["total_latency_ms"]["p50"],
    ],
    "P95": [
        metrics["retrieval_latency_ms"]["p95"],
        metrics["llm_latency_ms"]["p95"],
        metrics["total_latency_ms"]["p95"],
    ],
}

fig = go.Figure()
fig.add_trace(go.Bar(name="P50", x=latency_data["Stage"], y=latency_data["P50"]))
fig.add_trace(go.Bar(name="P95", x=latency_data["Stage"], y=latency_data["P95"]))
fig.update_layout(barmode="group")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Raw Metrics")
st.json(metrics)
