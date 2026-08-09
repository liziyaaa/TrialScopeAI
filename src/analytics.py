"""Recruitment funnel, blocker, representation, scenario, and report helpers."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Sequence

import pandas as pd

from .models import Criterion, MatchResult
from .rules import match_dataframe, results_dataframe


STATUS_LABELS = {
    "eligible": "模拟符合",
    "ineligible": "不符合",
    "missing_data": "信息不足",
    "needs_review": "人工复核",
}


def status_counts(results: Sequence[MatchResult]) -> pd.DataFrame:
    counts = Counter(result.overall_status for result in results)
    return pd.DataFrame(
        [
            {"status": status, "label": label, "count": counts.get(status, 0)}
            for status, label in STATUS_LABELS.items()
        ]
    )


def blocker_counts(results: Sequence[MatchResult], criteria: Sequence[Criterion]) -> pd.DataFrame:
    criterion_map = {criterion.criterion_id: criterion for criterion in criteria}
    counts: Counter[str] = Counter()
    for result in results:
        counts.update(item.criterion_id for item in result.evidences if item.status == "fail")
    rows = []
    for criterion_id, count in counts.most_common():
        criterion = criterion_map.get(criterion_id)
        rows.append(
            {
                "criterion_id": criterion_id,
                "count": count,
                "criterion": criterion.source_text if criterion else criterion_id,
            }
        )
    return pd.DataFrame(rows)


def missing_field_counts(results: Sequence[MatchResult]) -> pd.DataFrame:
    counts: Counter[str] = Counter()
    for result in results:
        counts.update(
            item.field or "unsupported_criterion"
            for item in result.evidences
            if item.status == "missing"
        )
    return pd.DataFrame(
        [{"field": field, "count": count} for field, count in counts.most_common()]
    )


FUNNEL_STAGES = [
    ("候选队列", []),
    ("年龄与诊断", ["I01", "I02"]),
    ("吸烟史", ["I03"]),
    ("肺功能", ["I04", "I05", "I06", "I07"]),
    ("近期事件", ["E03", "E05", "E06"]),
    ("其他可执行排除项", ["E02", "E04", "E07", "E08", "E09", "E10", "E11", "E12", "E13", "E15", "E16", "E17"]),
]


def build_funnel(patients: pd.DataFrame, criteria: Sequence[Criterion]) -> pd.DataFrame:
    criterion_map = {criterion.criterion_id: criterion for criterion in criteria}
    active = patients.copy()
    rows = [{"stage": "候选队列", "count": len(active)}]
    accumulated: list[Criterion] = []
    for label, ids in FUNNEL_STAGES[1:]:
        accumulated.extend(criterion_map[item] for item in ids if item in criterion_map)
        matches = match_dataframe(active, accumulated)
        keep_ids = [result.patient_id for result in matches if result.overall_status != "ineligible"]
        active = active[active["patient_id"].astype(str).isin(keep_ids)]
        rows.append({"stage": label, "count": len(active)})
    final_results = match_dataframe(active, criteria)
    potential_ids = [
        result.patient_id
        for result in final_results
        if result.overall_status in {"eligible", "needs_review"}
    ]
    rows.append({"stage": "模拟符合或待复核", "count": len(potential_ids)})
    return pd.DataFrame(rows)


def representation_table(patients: pd.DataFrame, results: Sequence[MatchResult]) -> pd.DataFrame:
    result_frame = results_dataframe(results)
    merged = patients.merge(result_frame[["patient_id", "overall_status"]], on="patient_id")
    selected = merged[merged["overall_status"].isin(["eligible", "needs_review"])]
    rows: list[dict[str, Any]] = []
    for group_name, frame in [("候选队列", merged), ("模拟符合或待复核", selected)]:
        rows.extend(
            [
                {"group": group_name, "metric": "平均年龄", "value": round(frame["age"].mean(), 1) if len(frame) else 0},
                {"group": group_name, "metric": "女性占比", "value": round((frame["sex"] == "Female").mean() * 100, 1) if len(frame) else 0},
                {"group": group_name, "metric": "65岁及以上占比", "value": round((frame["age"] >= 65).mean() * 100, 1) if len(frame) else 0},
                {"group": group_name, "metric": "重度COPD占比", "value": round((frame["disease_severity"] == "severe").mean() * 100, 1) if len(frame) else 0},
            ]
        )
    return pd.DataFrame(rows)


def apply_scenario(criteria: Sequence[Criterion], overrides: dict[str, Any]) -> list[Criterion]:
    cloned = deepcopy(list(criteria))
    for criterion in cloned:
        if criterion.criterion_id in overrides:
            criterion.value = overrides[criterion.criterion_id]
            criterion.note = (criterion.note + " 情景模拟参数已调整。").strip()
    return cloned


def scenario_comparison(
    patients: pd.DataFrame,
    baseline_criteria: Sequence[Criterion],
    scenario_criteria: Sequence[Criterion],
) -> tuple[pd.DataFrame, list[MatchResult], list[MatchResult]]:
    baseline = match_dataframe(patients, baseline_criteria)
    scenario = match_dataframe(patients, scenario_criteria)
    baseline_counts = Counter(item.overall_status for item in baseline)
    scenario_counts = Counter(item.overall_status for item in scenario)
    rows = []
    for status, label in STATUS_LABELS.items():
        before = baseline_counts.get(status, 0)
        after = scenario_counts.get(status, 0)
        rows.append({"status": status, "label": label, "baseline": before, "scenario": after, "change": after - before})
    return pd.DataFrame(rows), baseline, scenario


def _cohort_profile(
    patients: pd.DataFrame,
    results: Sequence[MatchResult],
    *,
    statuses: set[str] | None = None,
) -> dict[str, float]:
    """Return a compact demographic profile for a selected simulated cohort."""

    selected_statuses = statuses or {"eligible"}
    result_frame = results_dataframe(results)
    merged = patients.merge(result_frame[["patient_id", "overall_status"]], on="patient_id")
    selected = merged[merged["overall_status"].isin(selected_statuses)]
    if selected.empty:
        return {
            "mean_age": 0.0,
            "female_pct": 0.0,
            "older_pct": 0.0,
            "severe_pct": 0.0,
        }
    return {
        "mean_age": round(float(selected["age"].mean()), 2),
        "female_pct": round(float((selected["sex"] == "Female").mean() * 100), 2),
        "older_pct": round(float((selected["age"] >= 65).mean() * 100), 2),
        "severe_pct": round(
            float((selected["disease_severity"] == "severe").mean() * 100), 2
        ),
    }


def _representation_gap(patients: pd.DataFrame, results: Sequence[MatchResult]) -> float:
    """Average absolute percentage-point gap from the full synthetic cohort."""

    all_results = [
        MatchResult(patient_id=str(row.patient_id), overall_status="eligible", evidences=[])
        for row in patients.itertuples()
    ]
    population = _cohort_profile(patients, all_results)
    selected = _cohort_profile(patients, results)
    dimensions = ["female_pct", "older_pct", "severe_pct"]
    return round(
        sum(abs(selected[item] - population[item]) for item in dimensions) / len(dimensions),
        2,
    )


def criterion_marginal_impact(
    patients: pd.DataFrame,
    criteria: Sequence[Criterion],
) -> pd.DataFrame:
    """Estimate each executable criterion's marginal effect by omitting it once.

    This is a counterfactual simulation for protocol discussion, not a recommendation
    to remove or relax a clinical criterion.
    """

    baseline = match_dataframe(patients, criteria)
    baseline_counts = Counter(item.overall_status for item in baseline)
    baseline_profile = _cohort_profile(patients, baseline)
    rows: list[dict[str, Any]] = []
    for criterion in criteria:
        if criterion.execution_status != "automated":
            continue
        reduced = [item for item in criteria if item.criterion_id != criterion.criterion_id]
        simulated = match_dataframe(patients, reduced)
        counts = Counter(item.overall_status for item in simulated)
        profile = _cohort_profile(patients, simulated)
        rows.append(
            {
                "criterion_id": criterion.criterion_id,
                "kind": criterion.kind,
                "field": criterion.field or "",
                "criterion": criterion.source_text,
                "eligible_baseline": baseline_counts.get("eligible", 0),
                "eligible_without": counts.get("eligible", 0),
                "eligible_change": counts.get("eligible", 0)
                - baseline_counts.get("eligible", 0),
                "potential_change": (
                    counts.get("eligible", 0) + counts.get("needs_review", 0)
                )
                - (
                    baseline_counts.get("eligible", 0)
                    + baseline_counts.get("needs_review", 0)
                ),
                "missing_change": counts.get("missing_data", 0)
                - baseline_counts.get("missing_data", 0),
                "mean_age_change": round(
                    profile["mean_age"] - baseline_profile["mean_age"], 2
                ),
                "female_pct_change": round(
                    profile["female_pct"] - baseline_profile["female_pct"], 2
                ),
                "older_pct_change": round(
                    profile["older_pct"] - baseline_profile["older_pct"], 2
                ),
                "severe_pct_change": round(
                    profile["severe_pct"] - baseline_profile["severe_pct"], 2
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["eligible_change", "potential_change"], ascending=False
    ).reset_index(drop=True)


def scenario_tradeoff(
    patients: pd.DataFrame,
    baseline_results: Sequence[MatchResult],
    scenario_results: Sequence[MatchResult],
) -> pd.DataFrame:
    """Compare scale, representation and information burden across two scenarios."""

    baseline_counts = Counter(item.overall_status for item in baseline_results)
    scenario_counts = Counter(item.overall_status for item in scenario_results)
    rows = [
        {
            "metric": "eligible_count",
            "baseline": baseline_counts.get("eligible", 0),
            "scenario": scenario_counts.get("eligible", 0),
        },
        {
            "metric": "potential_count",
            "baseline": baseline_counts.get("eligible", 0)
            + baseline_counts.get("needs_review", 0),
            "scenario": scenario_counts.get("eligible", 0)
            + scenario_counts.get("needs_review", 0),
        },
        {
            "metric": "representation_gap",
            "baseline": _representation_gap(patients, baseline_results),
            "scenario": _representation_gap(patients, scenario_results),
        },
        {
            "metric": "missing_data_count",
            "baseline": baseline_counts.get("missing_data", 0),
            "scenario": scenario_counts.get("missing_data", 0),
        },
        {
            "metric": "review_count",
            "baseline": baseline_counts.get("needs_review", 0),
            "scenario": scenario_counts.get("needs_review", 0),
        },
    ]
    frame = pd.DataFrame(rows)
    frame["change"] = frame["scenario"] - frame["baseline"]
    return frame


def build_markdown_report(
    source_title: str,
    patients: pd.DataFrame,
    results: Sequence[MatchResult],
    criteria: Sequence[Criterion],
    *,
    language: str = "zh",
) -> str:
    counts = Counter(result.overall_status for result in results)
    blockers = blocker_counts(results, criteria).head(5)
    blocker_lines = "\n".join(
        f"- {row.criterion_id}: {row['count']} {'records' if language == 'en' else '人'} - {row.criterion}"
        for _, row in blockers.iterrows()
    ) or ("- No dominant failed constraint" if language == "en" else "- 暂无明确排除原因")
    if language == "en":
        return f"""# TrialScopeAI protocol decision brief

## Reference study

{source_title}

## Synthetic cohort baseline

- Synthetic records: {len(patients)}
- Rule-eligible: {counts.get('eligible', 0)}
- Constraint not met: {counts.get('ineligible', 0)}
- Data unresolved: {counts.get('missing_data', 0)}
- Clinical review: {counts.get('needs_review', 0)}

## Leading failed constraints

{blocker_lines}

## Decision boundary

This brief uses public protocol criteria and synthetic records to validate a traceable decision workflow. It does not diagnose, enrol participants or recommend a protocol amendment. Medical, statistical, investigator and ethics review remain required for real decisions.
"""
    return f"""# TrialScopeAI 招募可行性模拟摘要

## 试验

{source_title}

## 队列与结果

- 合成候选患者：{len(patients)} 人
- 模拟符合：{counts.get('eligible', 0)} 人
- 不符合：{counts.get('ineligible', 0)} 人
- 信息不足：{counts.get('missing_data', 0)} 人
- 需要人工复核：{counts.get('needs_review', 0)} 人

## 主要排除原因

{blocker_lines}

## 使用边界

本报告仅基于公开试验标准与合成患者数据，用于方法验证和方案讨论，不构成诊断、入组决定或临床试验方案修改建议。所有真实决策均需由医学、统计、研究者及伦理人员审核。
"""
