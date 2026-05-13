# RAG Observability & Monitoring

A comprehensive monitoring and observability layer for RAG applications. Traces every pipeline step, tracks quality metrics over time, and gates deployments on regression tests.

## Features

- **Full Pipeline Tracing**: Track retrieval, reranking, LLM calls, and token usage per request
- **Latency Tracking**: P50/P95 percentiles for each pipeline stage
- **Cost Monitoring**: Per-request cost calculation in dollar terms
- **Citation Coverage**: Percentage of answers grounded in retrieved evidence
- **Quality Dashboard**: Visualize metrics over time
- **Regression Gating**: CI pipeline fails if quality drops below threshold

## Tech Stack

- **Tracing**: Langfuse (open-source, self-hostable)
- **Metrics**: Custom collectors with Prometheus-compatible export
- **Dashboard**: Streamlit
- **CI Integration**: GitHub Actions with eval gating

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Usage

```bash
# Start the traced RAG server
python src/traced_app.py

# View metrics dashboard
streamlit run dashboards/dashboard.py

# Run regression evaluation
python src/regression_eval.py
```

## Metrics Tracked

| Metric | Description |
|--------|-------------|
| `retrieval_latency_ms` | Time to retrieve chunks (P50/P95) |
| `rerank_latency_ms` | Time for cross-encoder reranking |
| `llm_latency_ms` | Time for LLM generation |
| `total_latency_ms` | End-to-end request latency |
| `tokens_used` | Input + output tokens per request |
| `cost_per_request_usd` | Estimated cost per query |
| `citation_coverage` | % of answers with proper citations |
| `failure_rate` | % of requests that error or lack support |
