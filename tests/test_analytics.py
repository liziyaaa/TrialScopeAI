from __future__ import annotations

import pandas as pd

from src.analytics import (
    apply_scenario,
    blocker_counts,
    build_funnel,
    build_markdown_report,
    criterion_marginal_impact,
    scenario_comparison,
    scenario_tradeoff,
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


def test_marginal_impact_is_traceable_and_non_destructive(criteria):
    patients = pd.read_csv("data/synthetic_patients.csv")
    original_values = {item.criterion_id: item.value for item in criteria}
    impact = criterion_marginal_impact(patients, criteria)
    assert not impact.empty
    assert set(impact["criterion_id"]).issubset(original_values)
    assert impact["criterion"].str.len().min() > 0
    assert impact.iloc[0]["eligible_change"] >= 0
    assert {item.criterion_id: item.value for item in criteria} == original_values


def test_scenario_tradeoff_covers_three_decision_dimensions(criteria):
    patients = pd.read_csv("data/synthetic_patients.csv")
    scenario = apply_scenario(criteria, {"I01": 50, "I03": 20})
    _, baseline, changed = scenario_comparison(patients, criteria, scenario)
    tradeoff = scenario_tradeoff(patients, baseline, changed)
    assert {
        "eligible_count",
        "representation_gap",
        "missing_data_count",
    }.issubset(set(tradeoff["metric"]))
    assert (tradeoff["change"] != 0).any()
