from pathlib import Path

import pandas as pd

from src.llm_parser import ALLOWED_FIELDS, load_cached_demo_criteria
from src.rules import match_dataframe


SAMPLE_DIR = Path("sample_data")


def test_four_sample_cohorts_are_valid_upload_files():
    files = sorted(SAMPLE_DIR.glob("cohort_*.csv"))
    assert len(files) == 4

    for path in files:
        cohort = pd.read_csv(path)
        assert not cohort.empty
        assert "patient_id" in cohort.columns
        assert cohort["patient_id"].astype(str).str.strip().ne("").all()
        assert cohort["patient_id"].is_unique
        assert len(cohort) <= 50_000


def test_complete_samples_cover_all_executable_fields_and_run_rules():
    criteria = load_cached_demo_criteria()
    for file_name in [
        "cohort_01_complete_mixed.csv",
        "cohort_02_low_risk.csv",
        "cohort_04_boundary_values.csv",
    ]:
        cohort = pd.read_csv(SAMPLE_DIR / file_name)
        assert set(ALLOWED_FIELDS).issubset(cohort.columns)
        results = match_dataframe(cohort, criteria)
        assert len(results) == len(cohort)


def test_missing_field_sample_exercises_unresolved_path():
    cohort = pd.read_csv(SAMPLE_DIR / "cohort_03_missing_fields.csv")
    results = match_dataframe(cohort, load_cached_demo_criteria())
    assert any(result.overall_status == "missing_data" for result in results)
