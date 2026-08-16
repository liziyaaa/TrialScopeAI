"""Generate deterministic, de-identified cohort CSVs for product verification."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "sample_data"
ROW_COUNT = 500


def base_record(index: int, rng: random.Random, prefix: str) -> dict[str, object]:
    age = rng.randint(18, 76)
    return {
        "patient_id": f"{prefix}-{index:04d}",
        "age": age,
        "sex": "female" if index % 2 else "male",
        "copd_diagnosis": rng.random() < 0.42,
        "smoking_pack_years": rng.randint(0, 55),
        "post_bd_fev1_pct_predicted": rng.randint(39, 92),
        "post_bd_fev1_liters": round(rng.uniform(0.95, 3.65), 2),
        "post_bd_fev1_fvc": round(rng.uniform(0.39, 0.85), 3),
        "spirometry_reproducible": rng.random() >= 0.08,
        "contraception_confirmed": rng.random() >= 0.08,
        "informed_consent_confirmed": rng.random() >= 0.04,
        "visit_adherence_confirmed": rng.random() >= 0.12,
        "severe_comorbidity_concern": rng.random() < 0.12,
        "other_significant_respiratory_disease": rng.random() < 0.08,
        "days_since_copd_exacerbation": rng.randint(0, 210),
        "oxygen_hours_per_day": rng.randint(0, 24),
        "days_since_respiratory_infection": rng.randint(0, 120),
        "days_since_systemic_steroids": rng.randint(0, 150),
        "malignancy_within_5y": rng.random() < 0.05,
        "qtc_ms": rng.randint(395, 486),
        "bladder_outflow_obstruction_within_6m": rng.random() < 0.04,
        "narrow_angle_glaucoma": rng.random() < 0.03,
        "aerosol_medication_hypersensitivity": rng.random() < 0.03,
        "substance_abuse_within_3m": rng.random() < 0.04,
        "psychiatric_completion_concern": rng.random() < 0.06,
        "investigational_drug_within_30d": rng.random() < 0.08,
        "prior_sun101": rng.random() < 0.03,
        "study_drug_class_hypersensitivity": rng.random() < 0.03,
        "disease_severity": (
            "mild" if age < 35 else "moderate" if age < 60 else "severe"
        ),
    }


def complete_mixed() -> pd.DataFrame:
    rng = random.Random(2026081601)
    return pd.DataFrame(
        [base_record(index, rng, "MIX") for index in range(1, ROW_COUNT + 1)]
    )


def low_risk() -> pd.DataFrame:
    rng = random.Random(2026081602)
    rows: list[dict[str, object]] = []
    for index in range(1, ROW_COUNT + 1):
        record = base_record(index, rng, "LOW")
        record.update(
            {
                "age": rng.randint(18, 58),
                "copd_diagnosis": False,
                "smoking_pack_years": rng.randint(0, 5),
                "post_bd_fev1_pct_predicted": rng.randint(74, 94),
                "post_bd_fev1_liters": round(rng.uniform(2.5, 3.8), 2),
                "post_bd_fev1_fvc": round(rng.uniform(0.73, 0.88), 3),
                "spirometry_reproducible": True,
                "contraception_confirmed": True,
                "informed_consent_confirmed": True,
                "visit_adherence_confirmed": True,
                "severe_comorbidity_concern": False,
                "other_significant_respiratory_disease": False,
                "days_since_copd_exacerbation": rng.randint(90, 240),
                "oxygen_hours_per_day": 0,
                "days_since_respiratory_infection": rng.randint(60, 150),
                "days_since_systemic_steroids": rng.randint(60, 180),
                "malignancy_within_5y": False,
                "qtc_ms": rng.randint(395, 440),
                "bladder_outflow_obstruction_within_6m": False,
                "narrow_angle_glaucoma": False,
                "aerosol_medication_hypersensitivity": False,
                "substance_abuse_within_3m": False,
                "psychiatric_completion_concern": False,
                "investigational_drug_within_30d": False,
                "prior_sun101": False,
                "study_drug_class_hypersensitivity": False,
                "disease_severity": "mild" if index % 3 else "moderate",
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def missing_fields() -> pd.DataFrame:
    rng = random.Random(2026081603)
    return pd.DataFrame(
        [
            {
                "patient_id": f"MISS-{index:04d}",
                "age": rng.randint(18, 78),
                "sex": "female" if index % 2 else "male",
                "informed_consent_confirmed": index % 17 != 0,
            }
            for index in range(1, ROW_COUNT + 1)
        ]
    )


def boundary_values() -> pd.DataFrame:
    rng = random.Random(2026081604)
    ages = [17, 18, 19, 39, 40, 41, 64, 65, 66, 74, 75, 76]
    qtc_values = [449, 450, 451, 459, 460, 461, 469, 470, 471, 479, 480, 481]
    day_values = [31, 30, 29, 15, 14, 13, 8, 7, 6, 2, 1, 0]
    rows: list[dict[str, object]] = []
    for index in range(1, ROW_COUNT + 1):
        position = (index - 1) % len(ages)
        record = base_record(index, rng, "EDGE")
        record.update(
            {
                "age": ages[position],
                "smoking_pack_years": [9, 10, 11, 19, 20, 21, 29, 30, 31, 39, 40, 41][position],
                "post_bd_fev1_pct_predicted": [81, 80, 79, 61, 60, 59, 51, 50, 49, 41, 40, 39][position],
                "post_bd_fev1_liters": [2.51, 2.5, 2.49, 2.01, 2.0, 1.99, 1.51, 1.5, 1.49, 1.01, 1.0, 0.99][position],
                "post_bd_fev1_fvc": [0.701, 0.7, 0.699, 0.601, 0.6, 0.599, 0.501, 0.5, 0.499, 0.401, 0.4, 0.399][position],
                "days_since_copd_exacerbation": day_values[position],
                "days_since_respiratory_infection": [15, 14, 13, 8, 7, 6, 5, 4, 3, 1, 0, 0][position],
                "days_since_systemic_steroids": day_values[position],
                "qtc_ms": qtc_values[position],
                "spirometry_reproducible": True,
                "contraception_confirmed": True,
                "informed_consent_confirmed": True,
                "visit_adherence_confirmed": True,
                "severe_comorbidity_concern": position == 11,
                "other_significant_respiratory_disease": position == 11,
                "malignancy_within_5y": position == 11,
                "bladder_outflow_obstruction_within_6m": position == 11,
                "narrow_angle_glaucoma": position == 11,
                "aerosol_medication_hypersensitivity": position == 11,
                "substance_abuse_within_3m": position == 11,
                "psychiatric_completion_concern": position == 11,
                "investigational_drug_within_30d": position == 11,
                "prior_sun101": position == 11,
                "study_drug_class_hypersensitivity": position == 11,
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cohorts = {
        "cohort_01_complete_mixed.csv": complete_mixed(),
        "cohort_02_low_risk.csv": low_risk(),
        "cohort_03_missing_fields.csv": missing_fields(),
        "cohort_04_boundary_values.csv": boundary_values(),
    }
    for file_name, cohort in cohorts.items():
        cohort.to_csv(OUTPUT_DIR / file_name, index=False, encoding="utf-8")
        print(f"{file_name}: {len(cohort)} rows")


if __name__ == "__main__":
    main()
