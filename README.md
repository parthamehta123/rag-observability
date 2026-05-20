# RAG Observability & Monitoring

A comprehensive monitoring and observability layer for a production RAG system processing SEC 10-K financial filings. Traces every pipeline step with Langfuse, tracks quality metrics over time, and gates deployments on regression tests.

## Features

- **Full Pipeline Tracing**: Langfuse spans for retrieval, reranking, and LLM calls with per-span timing
- **Latency Tracking**: P50/P95 percentiles for each pipeline stage (not averages)
- **Cost Monitoring**: Per-request cost estimation in dollar terms
- **Citation Coverage**: Percentage of answers grounded in retrieved evidence
- **Quality Dashboard**: Streamlit dashboard with KPI cards and latency breakdown charts
- **Regression Gating**: CI pipeline fails if quality drops below threshold

## Observability Results (15 traced queries)

| Metric | Value |
|--------|-------|
| Total Requests | 15 |
| Error Rate | 0% |
| Citation Coverage | 86.7% |
| P50 Retrieval Latency | 479ms |
| P95 Retrieval Latency | 831ms |
| P50 LLM Latency | 3,640ms |
| P95 LLM Latency | 6,248ms |
| Regression Gate | PASSED |

Key finding: LLM generation is 88% of total latency. Retrieval is only 12%.

## Tech Stack

- **Tracing**: Langfuse v4 (cloud or self-hosted)
- **Metrics**: Custom collectors with Prometheus-compatible export
- **Dashboard**: Streamlit + Plotly
- **CI Integration**: GitHub Actions with eval gating

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env  # Add Langfuse keys
```

### Langfuse Setup

**Option A: Langfuse Cloud (free tier)**
1. Sign up at https://langfuse.com
2. Create a project, get API keys
3. Add to `.env`:
```
LANGFUSE_PUBLIC_KEY=pk-lf-xxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxx
LANGFUSE_HOST=https://us.cloud.langfuse.com
```

**Option B: Self-host with Docker**
```bash
docker compose -f docker-compose.langfuse.yml up -d
# Open http://localhost:3030, create account, get keys
```

## Usage

Requires `~/production-rag` to be set up with ingested documents first.

```bash
# Run 15 traced queries against production-rag (sends traces to Langfuse)
python src/run_traced_queries.py

# View metrics dashboard
streamlit run dashboards/dashboard.py

# Run regression evaluation (checks quality thresholds)
python src/regression_eval.py

# Run tests
pytest tests/ -v
```

## Metrics Tracked

| Metric | Description |
|--------|-------------|
| `retrieval_latency_ms` | Time to retrieve chunks (P50/P95) |
| `llm_latency_ms` | Time for LLM generation (P50/P95) |
| `total_latency_ms` | End-to-end request latency |
| `input_tokens` / `output_tokens` | Token counts per request |
| `cost_per_request_usd` | Estimated cost per query |
| `citation_coverage` | % of answers with proper citations |
| `error_rate` | % of requests that error |
