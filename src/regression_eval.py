"""Regression evaluation: run golden dataset and gate on quality thresholds."""

import json
import sys
from pathlib import Path

FAITHFULNESS_THRESHOLD = 0.7
CITATION_COVERAGE_THRESHOLD = 0.8


def run_regression(eval_results_path: str = "reports/eval_results.json"):
    """Check evaluation results against quality thresholds. Exit 1 if below."""
    path = Path(eval_results_path)
    if not path.exists():
        print(f"ERROR: Evaluation results not found at {eval_results_path}")
        print("Run the evaluation pipeline first.")
        sys.exit(1)

    with open(path) as f:
        results = json.load(f)

    print("Regression Gate Results:")
    print(json.dumps(results, indent=2))

    passed = True

    faithfulness = results.get("faithfulness", 0)
    if faithfulness < FAITHFULNESS_THRESHOLD:
        print(f"FAIL: Faithfulness {faithfulness:.4f} < {FAITHFULNESS_THRESHOLD}")
        passed = False
    else:
        print(f"PASS: Faithfulness {faithfulness:.4f} >= {FAITHFULNESS_THRESHOLD}")

    citation_coverage = results.get("citation_coverage", 0)
    if citation_coverage < CITATION_COVERAGE_THRESHOLD:
        print(
            f"FAIL: Citation coverage {citation_coverage:.4f} < {CITATION_COVERAGE_THRESHOLD}"
        )
        passed = False
    else:
        print(
            f"PASS: Citation coverage {citation_coverage:.4f} >= {CITATION_COVERAGE_THRESHOLD}"
        )

    if not passed:
        print("\nRegression gate FAILED. Fix quality issues before merging.")
        sys.exit(1)
    else:
        print("\nRegression gate PASSED.")


if __name__ == "__main__":
    run_regression()
