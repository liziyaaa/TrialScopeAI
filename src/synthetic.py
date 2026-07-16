"""Seeded COPD synthetic cohort and independent edge-case fixtures."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd


SEED = 20260716


def generate_synthetic_patients(count: int = 500, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(64, 10, count).round(), 35, 88).astype(int)
    sex = rng.choice(["Male", "Female"], size=count, p=[0.55, 0.45])
    severity = rng.choice(["mild", "moderate", "severe"], size=count, p=[0.22, 0.53, 0.25])
    severity_base = np.select(
        [severity == "mild", severity == "moderate", severity == "severe"],
        [77, 58, 38],
    )
    fev1_pct = np.clip(severity_base + rng.normal(0, 9, count), 20, 105).round(1)
    predicted_liters = np.where(sex == "Male", 3.1, 2.4) - np.maximum(age - 45, 0) * 0.012
    fev1_liters = np.clip(predicted_liters * fev1_pct / 100, 0.35, 3.8).round(2)
    fev1_fvc = np.clip(0.79 - (100 - fev1_pct) * 0.0037 + rng.normal(0, 0.045, count), 0.38, 0.86).round(3)

    pack_years = np.clip(rng.gamma(shape=3.0, scale=9.5, size=count), 0, 110).round(1)
    smoking_status = np.where(
        pack_years < 5,
        "never",
        rng.choice(["current", "former"], size=count, p=[0.36, 0.64]),
    )
    qtc = np.clip(rng.normal(np.where(sex == "Male", 425, 438), 20, count), 360, 520).round().astype(int)

    patients = pd.DataFrame(
        {
            "patient_id": [f"P{i:04d}" for i in range(1, count + 1)],
            "age": age,
            "sex": sex,
            "disease_severity": severity,
            "copd_diagnosis": rng.random(count) < 0.94,
            "smoking_status": smoking_status,
            "smoking_pack_years": pack_years,
            "post_bd_fev1_pct_predicted": fev1_pct,
            "post_bd_fev1_liters": fev1_liters,
            "post_bd_fev1_fvc": fev1_fvc,
            "spirometry_reproducible": rng.random(count) < 0.94,
            "contraception_confirmed": np.where(sex == "Female", rng.random(count) < 0.94, True),
            "informed_consent_confirmed": rng.random(count) < 0.98,
            "visit_adherence_confirmed": rng.random(count) < 0.95,
            "severe_comorbidity_concern": rng.random(count) < 0.06,
            "other_significant_respiratory_disease": rng.random(count) < 0.08,
            "days_since_copd_exacerbation": rng.integers(5, 500, size=count),
            "oxygen_hours_per_day": np.clip(rng.normal(2.5, 4.5, count), 0, 24).round(1),
            "days_since_respiratory_infection": rng.integers(2, 600, size=count),
            "days_since_systemic_steroids": rng.integers(5, 900, size=count),
            "malignancy_within_5y": rng.random(count) < 0.045,
            "qtc_ms": qtc,
            "bladder_outflow_obstruction_within_6m": rng.random(count) < 0.025,
            "narrow_angle_glaucoma": rng.random(count) < 0.02,
            "aerosol_medication_hypersensitivity": rng.random(count) < 0.018,
            "substance_abuse_within_3m": rng.random(count) < 0.012,
            "psychiatric_completion_concern": rng.random(count) < 0.025,
            "investigational_drug_within_30d": rng.random(count) < 0.035,
            "prior_sun101": rng.random(count) < 0.008,
            "study_drug_class_hypersensitivity": rng.random(count) < 0.018,
        }
    )

    for column in [
        "smoking_pack_years",
        "post_bd_fev1_pct_predicted",
        "post_bd_fev1_liters",
        "post_bd_fev1_fvc",
        "qtc_ms",
    ]:
        missing_rows = rng.choice(count, size=max(1, int(count * 0.02)), replace=False)
        patients.loc[missing_rows, column] = np.nan
    return patients


def _base_edge_patient() -> dict[str, Any]:
    return {
        "patient_id": "EDGE-BASE",
        "age": 60,
        "sex": "Male",
        "disease_severity": "moderate",
        "copd_diagnosis": True,
        "smoking_status": "former",
        "smoking_pack_years": 30.0,
        "post_bd_fev1_pct_predicted": 55.0,
        "post_bd_fev1_liters": 1.4,
        "post_bd_fev1_fvc": 0.58,
        "spirometry_reproducible": True,
        "contraception_confirmed": True,
        "informed_consent_confirmed": True,
        "visit_adherence_confirmed": True,
        "severe_comorbidity_concern": False,
        "other_significant_respiratory_disease": False,
        "days_since_copd_exacerbation": 100,
        "oxygen_hours_per_day": 0.0,
        "days_since_respiratory_infection": 100,
        "days_since_systemic_steroids": 180,
        "malignancy_within_5y": False,
        "qtc_ms": 420,
        "bladder_outflow_obstruction_within_6m": False,
        "narrow_angle_glaucoma": False,
        "aerosol_medication_hypersensitivity": False,
        "substance_abuse_within_3m": False,
        "psychiatric_completion_concern": False,
        "investigational_drug_within_30d": False,
        "prior_sun101": False,
        "study_drug_class_hypersensitivity": False,
    }


def generate_edge_cases() -> list[dict[str, Any]]:
    specs: list[tuple[str, dict[str, Any], str]] = [("baseline", {}, "eligible")]
    specs.extend(
        [
            ("age_below", {"age": 39}, "ineligible"),
            ("no_copd", {"copd_diagnosis": False}, "ineligible"),
            ("pack_years_low", {"smoking_pack_years": 9.9}, "ineligible"),
            ("fev1_pct_boundary_fail", {"post_bd_fev1_pct_predicted": 80.0}, "ineligible"),
            ("fev1_liters_boundary_fail", {"post_bd_fev1_liters": 0.7}, "ineligible"),
            ("ratio_boundary_fail", {"post_bd_fev1_fvc": 0.7}, "ineligible"),
            ("spirometry_not_reproducible", {"spirometry_reproducible": False}, "ineligible"),
            ("other_respiratory_disease", {"other_significant_respiratory_disease": True}, "ineligible"),
            ("recent_exacerbation", {"days_since_copd_exacerbation": 42}, "ineligible"),
            ("oxygen_over_limit", {"oxygen_hours_per_day": 12.1}, "ineligible"),
            ("recent_infection", {"days_since_respiratory_infection": 42}, "ineligible"),
            ("recent_steroids", {"days_since_systemic_steroids": 90}, "ineligible"),
            ("recent_malignancy", {"malignancy_within_5y": True}, "ineligible"),
            ("male_qtc_high", {"qtc_ms": 451}, "ineligible"),
            ("female_qtc_high", {"sex": "Female", "qtc_ms": 471}, "ineligible"),
            ("bladder_obstruction", {"bladder_outflow_obstruction_within_6m": True}, "ineligible"),
            ("glaucoma", {"narrow_angle_glaucoma": True}, "ineligible"),
            ("aerosol_hypersensitivity", {"aerosol_medication_hypersensitivity": True}, "ineligible"),
            ("substance_abuse", {"substance_abuse_within_3m": True}, "ineligible"),
            ("recent_trial", {"investigational_drug_within_30d": True}, "ineligible"),
            ("prior_sun101", {"prior_sun101": True}, "ineligible"),
            ("class_hypersensitivity", {"study_drug_class_hypersensitivity": True}, "ineligible"),
        ]
    )
    specs.extend(
        [
            ("age_boundary_pass", {"age": 40}, "eligible"),
            ("pack_years_boundary_pass", {"smoking_pack_years": 10.0}, "eligible"),
            ("fev1_pct_pass", {"post_bd_fev1_pct_predicted": 79.9}, "eligible"),
            ("fev1_liters_pass", {"post_bd_fev1_liters": 0.701}, "eligible"),
            ("ratio_pass", {"post_bd_fev1_fvc": 0.699}, "eligible"),
            ("oxygen_boundary_pass", {"oxygen_hours_per_day": 12.0}, "eligible"),
            ("exacerbation_outside", {"days_since_copd_exacerbation": 43}, "eligible"),
            ("infection_outside", {"days_since_respiratory_infection": 43}, "eligible"),
            ("steroids_outside", {"days_since_systemic_steroids": 91}, "eligible"),
            ("male_qtc_boundary", {"qtc_ms": 450}, "eligible"),
            ("female_qtc_boundary", {"sex": "Female", "qtc_ms": 470}, "eligible"),
        ]
    )
    for field in [
        "age",
        "smoking_pack_years",
        "post_bd_fev1_pct_predicted",
        "post_bd_fev1_liters",
        "post_bd_fev1_fvc",
        "qtc_ms",
    ]:
        specs.append((f"missing_{field}", {field: None}, "missing_data"))
    specs.extend(
        [
            ("consent_review", {"informed_consent_confirmed": False}, "needs_review"),
            ("adherence_review", {"visit_adherence_confirmed": False}, "needs_review"),
            ("contraception_review", {"sex": "Female", "contraception_confirmed": False}, "needs_review"),
            ("comorbidity_review", {"severe_comorbidity_concern": True}, "needs_review"),
            ("psychiatric_review", {"psychiatric_completion_concern": True}, "needs_review"),
            ("multi_fail_1", {"age": 35, "smoking_pack_years": 2}, "ineligible"),
            ("multi_fail_2", {"post_bd_fev1_pct_predicted": 88, "post_bd_fev1_fvc": 0.74}, "ineligible"),
            ("multi_fail_3", {"oxygen_hours_per_day": 18, "days_since_copd_exacerbation": 10}, "ineligible"),
            ("fail_and_missing", {"age": 35, "qtc_ms": None}, "ineligible"),
            ("review_and_missing", {"informed_consent_confirmed": False, "qtc_ms": None}, "missing_data"),
        ]
    )

    cases: list[dict[str, Any]] = []
    for index, (name, overrides, expected) in enumerate(specs, start=1):
        patient = deepcopy(_base_edge_patient())
        patient.update(overrides)
        patient["patient_id"] = f"EDGE-{index:03d}"
        cases.append({"case": name, "expected_status": expected, "patient": patient})
    return cases
