from __future__ import annotations

import pytest

from src.models import Criterion
from src.rules import evaluate_criterion, match_patient


def test_all_50_edge_cases_match_expected_status(criteria, edge_cases):
    assert len(edge_cases) >= 50
    for case in edge_cases:
        actual = match_patient(case["patient"], criteria).overall_status
        assert actual == case["expected_status"], case["case"]


def test_ineligible_takes_priority_over_missing(criteria, edge_cases):
    case = next(item for item in edge_cases if item["case"] == "fail_and_missing")
    result = match_patient(case["patient"], criteria)
    assert result.overall_status == "ineligible"
    assert result.failed_criteria
    assert result.missing_criteria


def test_missing_takes_priority_over_review(criteria, edge_cases):
    case = next(item for item in edge_cases if item["case"] == "review_and_missing")
    result = match_patient(case["patient"], criteria)
    assert result.overall_status == "missing_data"
    assert result.missing_criteria
    assert result.review_criteria


def test_source_traceability_is_complete(criteria):
    assert len(criteria) == 27
    assert all(item.source_text and item.source_reference for item in criteria)


@pytest.mark.parametrize(
    ("operator", "patient_value", "expected", "passes"),
    [
        ("eq", 10, 10, True),
        ("neq", 10, 11, True),
        ("lt", 9, 10, True),
        ("lte", 10, 10, True),
        ("gt", 11, 10, True),
        ("gte", 10, 10, True),
        ("between", 10, [5, 10], True),
        ("in", "a", ["a", "b"], True),
        ("not_in", "c", ["a", "b"], True),
        ("is_true", True, True, True),
        ("is_false", False, False, True),
        ("within_days", 42, 42, True),
        ("exists", 0, None, True),
    ],
)
def test_supported_inclusion_operators(operator, patient_value, expected, passes):
    criterion = Criterion(
        criterion_id="T01",
        kind="inclusion",
        source_text="Test source text",
        source_reference="test",
        field="value",
        operator=operator,
        value=expected,
    )
    evidence = evaluate_criterion({"value": patient_value}, criterion)
    assert (evidence.status == "pass") is passes


def test_applicability_marks_other_sex_not_applicable(criteria, edge_cases):
    patient = edge_cases[0]["patient"]
    female_qtc = next(item for item in criteria if item.criterion_id == "E09")
    evidence = evaluate_criterion(patient, female_qtc)
    assert evidence.status == "not_applicable"
