"""Run traced queries against the production-rag system and collect observability metrics.

This script bridges rag-observability with production-rag:
1. Loads the production-rag retriever and query pipeline
2. Wraps each step with Langfuse tracing (per-span timing)
3. Collects metrics (latency, citations, costs) per request
4. Saves metrics summary for the Streamlit dashboard
5. Saves eval results for the regression gate

Usage:
    cd ~/rag-observability
    python src/run_traced_queries.py
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from src.metrics_collector import MetricsStore
from src.tracing import PipelineMetrics, finalize_pipeline, traced_pipeline, traced_span

# Import production-rag modules without conflicting with local src namespace
RAG_ROOT = Path(__file__).resolve().parent.parent.parent / "production-rag"

# Load production-rag's .env for API keys, then local .env for Langfuse keys
load_dotenv(RAG_ROOT / ".env")
load_dotenv(override=True)  # local .env overrides for LANGFUSE_* keys

# Check if Langfuse is configured
LANGFUSE_ENABLED = (
    bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
    and os.getenv("LANGFUSE_PUBLIC_KEY") != "your-langfuse-public-key"
)


def _import_from_rag(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(
        module_name, RAG_ROOT / "src" / file_name
    )
    mod = importlib.util.module_from_spec(spec)
    # Register under both the custom name AND src.* so cross-imports work
    sys.modules[module_name] = mod
    base_name = file_name.replace(".py", "")
    sys.modules[f"src.{base_name}"] = mod
    spec.loader.exec_module(mod)
    return mod


# Order matters: retriever first (query.py imports from src.retriever)
rag_ingest = _import_from_rag("rag_ingest", "ingest.py")
rag_retriever = _import_from_rag("rag_retriever", "retriever.py")
rag_query = _import_from_rag("rag_query", "query.py")
generate_answer = rag_query.generate_answer
load_prompts = rag_query.load_prompts
HybridRetriever = rag_retriever.HybridRetriever

QUERIES = [
    "What are Apple's main risk factors?",
    "What are Tesla's main business segments and products?",
    "What is Microsoft's cloud computing strategy and how does Azure fit in?",
    "JPMorgan Chase business segments CCB CIB AWM",
    "What regulatory risks does Goldman Sachs identify in their 10-K?",
    "What factors affect Apple's gross margins?",
    "How does Tesla describe competition in EVs?",
    "What does Microsoft say about AI in their annual report?",
    "What are JPMorgan's total assets and stockholders equity?",
    "How does Goldman Sachs describe its asset and wealth management business?",
    "What does Apple say about tariffs and trade restrictions?",
    "What are the main sources of Apple's revenue?",
    "What cybersecurity threats does Apple describe in their risk factors?",
    "How does Tesla sell its vehicles to customers?",
    "What cybersecurity services does Microsoft offer?",
]


def run_traced_queries():
    """Run all queries with Langfuse tracing and local metrics collection."""
    print("Loading production-rag pipeline...")
    persist_dir = str(RAG_ROOT / "data" / "chroma")
    retriever = HybridRetriever(persist_dir=persist_dir)
    prompts = load_prompts(str(RAG_ROOT / "configs" / "prompts.yaml"))

    store = MetricsStore()
    results = []

    if LANGFUSE_ENABLED:
        print("Langfuse tracing: ENABLED")
    else:
        print("Langfuse tracing: DISABLED (no keys configured)")
        print("  To enable: add LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to .env")

    print(f"\nRunning {len(QUERIES)} traced queries...\n")

    for i, query in enumerate(QUERIES):
        metrics = PipelineMetrics()

        def _run_query():
            """Execute a single traced query."""
            # ── Retrieval span ──
            with traced_span("retrieval", metrics, as_type="retriever"):
                chunks = retriever.retrieve(query)
                metrics.chunks_retrieved = len(chunks)

            # ── LLM generation span ──
            with traced_span("llm", metrics, as_type="generation"):
                result = generate_answer(query, retriever, prompts)

            return result

        # Wrap in pipeline trace if Langfuse enabled, otherwise run directly
        if LANGFUSE_ENABLED:
            with traced_pipeline(query, user_id="benchmark"):
                result = _run_query()
                answer = result["answer"]
                sources = result["sources"]

                metrics.has_citations = "[Source" in answer
                metrics.declined_to_answer = "I cannot answer" in answer

                context_text = " ".join(s["content"] for s in sources)
                metrics.input_tokens = len(context_text + query) // 4
                metrics.output_tokens = len(answer) // 4

                finalize_pipeline(metrics, answer)
        else:
            result = _run_query()
            answer = result["answer"]
            sources = result["sources"]

            metrics.has_citations = "[Source" in answer
            metrics.declined_to_answer = "I cannot answer" in answer

            context_text = " ".join(s["content"] for s in sources)
            metrics.input_tokens = len(context_text + query) // 4
            metrics.output_tokens = len(answer) // 4

        store.record(metrics)

        status = (
            "CITED"
            if metrics.has_citations
            else ("DECLINED" if metrics.declined_to_answer else "NO_CITATION")
        )
        print(
            f"  [{i + 1:2d}/{len(QUERIES)}] {status:10s} "
            f"| retrieval:{metrics.retrieval_latency_ms:5.0f}ms "
            f"| llm:{metrics.llm_latency_ms:5.0f}ms "
            f"| total:{metrics.total_latency_ms:6.0f}ms "
            f"| {query[:50]}"
        )

        results.append(
            {
                "query": query,
                "answer_preview": answer[:200],
                "answer_length": len(answer),
                "has_citations": metrics.has_citations,
                "declined": metrics.declined_to_answer,
                "retrieval_ms": metrics.retrieval_latency_ms,
                "llm_ms": metrics.llm_latency_ms,
                "total_ms": metrics.total_latency_ms,
                "chunks_retrieved": metrics.chunks_retrieved,
                "input_tokens": metrics.input_tokens,
                "output_tokens": metrics.output_tokens,
            }
        )

    # Save metrics summary for dashboard
    store.save("reports/metrics_summary.json")
    print("\nMetrics saved to reports/metrics_summary.json")

    # Save per-query results
    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/query_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Query results saved to reports/query_results.json")

    # Save eval results for regression gate
    summary = store.get_summary()
    eval_results = {
        "faithfulness": summary["citation_coverage"],
        "citation_coverage": summary["citation_coverage"],
    }
    with open("reports/eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    print("Eval results saved to reports/eval_results.json")

    # Print summary
    print(f"\n{'=' * 60}")
    print("OBSERVABILITY SUMMARY")
    print(f"{'=' * 60}")
    print(json.dumps(summary, indent=2))

    if LANGFUSE_ENABLED:
        host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
        print(f"\nView traces in Langfuse: {host}")

    return summary


if __name__ == "__main__":
    run_traced_queries()
