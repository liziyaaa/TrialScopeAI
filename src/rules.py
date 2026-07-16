"""Deterministic eligibility rule evaluation and patient evidence generation."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import pandas as pd

from .models import Criterion, CriterionEvidence, MatchResult


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        result = pd.isna(value)
        return bool(result) if not isinstance(result, (list, tuple)) else False
    except (TypeError, ValueError):
        return False


def _condition_matches(patient_value: Any, operator: str, expected: Any) -> bool:
    if operator == "eq":
        return patient_value == expected
    if operator == "neq":
        return patient_value != expected
    if operator == "lt":
        return patient_value < expected
    if operator == "lte":
        return patient_value <= expected
    if operator == "gt":
        return patient_value > expected
    if operator == "gte":
        return patient_value >= expected
    if operator == "between":
        lower, upper = expected
        return lower <= patient_value <= upper
    if operator == "in":
        return patient_value in expected
    if operator == "not_in":
        return patient_value not in expected
    if operator == "is_true":
        return bool(patient_value) is True
    if operator == "is_false":
        return bool(patient_value) is False
    if operator == "within_days":
        return float(patient_value) <= float(expected)
    if operator == "exists":
        return not is_missing(patient_value)
    if operator == "human_review":
        return False
    raise ValueError(f"Unsupported operator: {operator}")


def _format_expected(criterion: Criterion) -> str:
    unit = f" {criterion.unit}" if criterion.unit else ""
    return f"{criterion.operator} {criterion.value}{unit}".strip()


def evaluate_criterion(patient: Mapping[str, Any], criterion: Criterion) -> CriterionEvidence:
    for field, expected in criterion.applicability.items():
        actual = patient.get(field)
        if is_missing(actual):
            return CriterionEvidence(
                criterion_id=criterion.criterion_id,
                criterion_kind=criterion.kind,
                status="missing",
                field=field,
                patient_value=actual,
                expected=expected,
                source_text=criterion.source_text,
                message=f"缺少适用条件字段 {field}，无法判断该标准。",
            )
        if actual != expected:
            return CriterionEvidence(
                criterion_id=criterion.criterion_id,
                criterion_kind=criterion.kind,
                status="not_applicable",
                field=criterion.field,
                patient_value=patient.get(criterion.field) if criterion.field else None,
                expected=criterion.value,
                source_text=criterion.source_text,
                message="该标准不适用于此患者。",
            )

    if not criterion.field or criterion.operator == "human_review":
        return CriterionEvidence(
            criterion_id=criterion.criterion_id,
            criterion_kind=criterion.kind,
            status="review",
            field=criterion.field,
            patient_value=None,
            expected=criterion.value,
            source_text=criterion.source_text,
            message=criterion.note or "该标准需要研究者人工判断。",
        )

    patient_value = patient.get(criterion.field)
    if is_missing(patient_value):
        return CriterionEvidence(
            criterion_id=criterion.criterion_id,
            criterion_kind=criterion.kind,
            status="missing",
            field=criterion.field,
            patient_value=patient_value,
            expected=criterion.value,
            source_text=criterion.source_text,
            message=f"缺少字段 {criterion.field}，无法执行标准。",
        )

    try:
        condition = _condition_matches(patient_value, criterion.operator, criterion.value)
    except (TypeError, ValueError) as exc:
        return CriterionEvidence(
            criterion_id=criterion.criterion_id,
            criterion_kind=criterion.kind,
            status="review",
            field=criterion.field,
            patient_value=patient_value,
            expected=criterion.value,
            source_text=criterion.source_text,
            message=f"字段值无法可靠比较，需要人工复核：{exc}",
        )

    expected_text = _format_expected(criterion)
    if criterion.execution_status == "human_review":
        if criterion.kind == "inclusion":
            status = "pass" if condition else "review"
        else:
            status = "review" if condition else "pass"
        message = (
            "已有合成的人工确认信息，标准通过。"
            if status == "pass"
            else "该患者触及主观标准，需要研究者人工复核。"
        )
    elif criterion.kind == "inclusion":
        status = "pass" if condition else "fail"
        message = (
            f"符合入组标准：患者值 {patient_value}，要求 {expected_text}。"
            if condition
            else f"不符合入组标准：患者值 {patient_value}，要求 {expected_text}。"
        )
    else:
        status = "fail" if condition else "pass"
        message = (
            f"触发排除标准：患者值 {patient_value}，条件 {expected_text}。"
            if condition
            else f"未触发排除标准：患者值 {patient_value}，条件 {expected_text}。"
        )

    return CriterionEvidence(
        criterion_id=criterion.criterion_id,
        criterion_kind=criterion.kind,
        status=status,
        field=criterion.field,
        patient_value=patient_value,
        expected=criterion.value,
        source_text=criterion.source_text,
        message=message,
    )


def match_patient(patient: Mapping[str, Any], criteria: Sequence[Criterion]) -> MatchResult:
    evidences = [evaluate_criterion(patient, criterion) for criterion in criteria]
    statuses = {item.status for item in evidences}
    if "fail" in statuses:
        overall = "ineligible"
    elif "missing" in statuses:
        overall = "missing_data"
    elif "review" in statuses:
        overall = "needs_review"
    else:
        overall = "eligible"
    return MatchResult(
        patient_id=str(patient.get("patient_id", "unknown")),
        overall_status=overall,
        evidences=evidences,
    )


def match_dataframe(patients: pd.DataFrame, criteria: Sequence[Criterion]) -> list[MatchResult]:
    return [match_patient(row.to_dict(), criteria) for _, row in patients.iterrows()]


def results_dataframe(results: Sequence[MatchResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        failures = [item.message for item in result.evidences if item.status == "fail"]
        missing = [item.field for item in result.evidences if item.status == "missing"]
        review = [item.criterion_id for item in result.evidences if item.status == "review"]
        rows.append(
            {
                "patient_id": result.patient_id,
                "overall_status": result.overall_status,
                "failed_criteria": ", ".join(result.failed_criteria),
                "missing_criteria": ", ".join(result.missing_criteria),
                "review_criteria": ", ".join(review),
                "reason_summary": failures[0] if failures else "",
                "missing_fields": ", ".join(sorted({field for field in missing if field})),
            }
        )
    return pd.DataFrame(rows)
