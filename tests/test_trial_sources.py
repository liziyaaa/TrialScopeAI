from __future__ import annotations

import json
from pathlib import Path

import fitz
import httpx
import pytest

from src.config import MAX_PDF_BYTES
from src.trial_sources import (
    PDFLimitError,
    PDFScannedError,
    SourceError,
    extract_searchable_pdf,
    fetch_nct_study,
    source_from_text,
)


ROOT = Path(__file__).resolve().parents[1]


def _pdf_with_text(text: str, pages: int = 1) -> bytes:
    document = fitz.open()
    for _ in range(pages):
        page = document.new_page()
        page.insert_textbox(fitz.Rect(60, 60, 535, 780), text, fontsize=10)
    payload = document.tobytes()
    document.close()
    return payload


def test_demo_pdf_is_searchable_and_section_is_found():
    payload = (ROOT / "output" / "pdf" / "golden4_demo_protocol.pdf").read_bytes()
    result = extract_searchable_pdf(payload, "golden4_demo_protocol.pdf")
    assert result.page_count >= 2
    assert result.section_found is True
    assert "Inclusion Criteria" in result.criteria_text
    assert "Exclusion Criteria" in result.criteria_text


def test_searchable_pdf_without_heading_returns_full_text_warning():
    text = "General study narrative without a recognized heading. " * 20
    result = extract_searchable_pdf(_pdf_with_text(text), "narrative.pdf")
    assert result.section_found is False
    assert result.warning
    assert "General study narrative" in result.criteria_text


def test_blank_pdf_is_reported_as_scanned():
    document = fitz.open()
    document.new_page()
    payload = document.tobytes()
    document.close()
    with pytest.raises(PDFScannedError):
        extract_searchable_pdf(payload, "scan.pdf")


def test_pdf_size_limit_is_checked_before_parsing():
    with pytest.raises(PDFLimitError):
        extract_searchable_pdf(b"%PDF" + b"x" * MAX_PDF_BYTES, "large.pdf")


def test_pdf_page_limit():
    document = fitz.open()
    for _ in range(201):
        document.new_page()
    payload = document.tobytes()
    document.close()
    with pytest.raises(PDFLimitError):
        extract_searchable_pdf(payload, "too-many-pages.pdf")


def test_invalid_pdf_and_short_text_raise_safe_errors():
    with pytest.raises(SourceError):
        extract_searchable_pdf(b"not-a-pdf", "bad.pdf")
    with pytest.raises(SourceError):
        source_from_text("too short")


def test_fetch_nct_validates_id_before_network():
    with pytest.raises(SourceError, match="格式"):
        fetch_nct_study("123")


def test_fetch_nct_maps_official_response(monkeypatch):
    trial = json.loads((ROOT / "data" / "golden4_trial.json").read_text(encoding="utf-8"))
    response_payload = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT02347774", "briefTitle": trial["title"]},
            "eligibilityModule": {
                "eligibilityCriteria": trial["criteria_text"],
                "minimumAge": "40 Years",
                "sex": "ALL",
            },
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Demo Sponsor"}},
            "statusModule": {"overallStatus": "COMPLETED"},
        }
    }

    def fake_get(url, timeout):
        request = httpx.Request("GET", url)
        return httpx.Response(200, request=request, json=response_payload)

    monkeypatch.setattr(httpx, "get", fake_get)
    result = fetch_nct_study("nct02347774")
    assert result.identifier == "NCT02347774"
    assert result.metadata["sponsor"] == "Demo Sponsor"
    assert "Inclusion Criteria" in result.criteria_text
