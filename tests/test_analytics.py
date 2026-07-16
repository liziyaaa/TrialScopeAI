from __future__ import annotations

import pandas as pd

from src.analytics import (
    apply_scenario,
    blocker_counts,
    build_funnel,
    build_markdown_report,
    scenario_comparison,
)
from src.rules import match_dataframe


def test_funnel_is_monotonic(criteria):
    patients = pd.read_csv("data/synthetic_patients.csv")
    funnel = build_funnel(patients, criteria)
    assert funnel.iloc[0]["count"] == 500
    assert funnel["count"].is_monotonic_decreasing


def test_blockers_and_report_are_traceable(criteria):
    patients = pd.read_csv("data/synthetic_patients.csv")
    results = match_dataframe(patients, criteria)
    blockers = blocker_counts(results, criteria)
    assert not blockers.empty
    assert blockers["criterion"].str.len().min() > 0
    report = build_markdown_report("GOLDEN-4", patients, results, criteria)
    assert "合成候选患者：500" in report
    assert "不构成诊断" in report


def test_scenario_comparison_returns_all_statuses(criteria):
    patients = pd.read_csv("data/synthetic_patients.csv")
    scenario = apply_scenario(criteria, {"I01": 50, "I03": 20})
    comparison, baseline, changed = scenario_comparison(patients, criteria, scenario)
    assert len(comparison) == 4
    assert len(baseline) == len(changed) == 500
    assert comparison["change"].sum() == 0
    assert (comparison["change"] != 0).any()
