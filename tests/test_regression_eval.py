"""Tests for regression evaluation gating."""

import json

import pytest

from src.regression_eval import CITATION_COVERAGE_THRESHOLD, FAITHFULNESS_THRESHOLD


def test_thresholds_are_valid():
    assert 0 < FAITHFULNESS_THRESHOLD <= 1.0
    assert 0 < CITATION_COVERAGE_THRESHOLD <= 1.0


def test_passing_results(tmp_path):
    results = {
        "faithfulness": 0.85,
        "citation_coverage": 0.90,
    }
    path = tmp_path / "eval_results.json"
    path.write_text(json.dumps(results))
    data = json.loads(path.read_text())
    assert data["faithfulness"] >= FAITHFULNESS_THRESHOLD
    assert data["citation_coverage"] >= CITATION_COVERAGE_THRESHOLD


def test_failing_faithfulness(tmp_path):
    results = {
        "faithfulness": 0.3,
        "citation_coverage": 0.90,
    }
    path = tmp_path / "eval_results.json"
    path.write_text(json.dumps(results))
    data = json.loads(path.read_text())
    assert data["faithfulness"] < FAITHFULNESS_THRESHOLD
