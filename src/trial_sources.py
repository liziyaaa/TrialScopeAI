"""ClinicalTrials.gov, pasted-text, and searchable-PDF ingestion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import fitz
import httpx

from .config import (
    MAX_PDF_BYTES,
    MAX_PDF_PAGES,
    MIN_EXTRACTED_TEXT_CHARS,
)
from .models import TrialSource


CLINICAL_TRIALS_API = "https://clinicaltrials.gov/api/v2/studies"
NCT_PATTERN = re.compile(r"^NCT\d{8}$", re.IGNORECASE)


class SourceError(ValueError):
    """Base class for safe, user-facing source errors."""


class PDFLimitError(SourceError):
    pass


class PDFScannedError(SourceError):
    pass


@dataclass(frozen=True)
class PDFExtraction:
    filename: str
    page_count: int
    full_text: str
    criteria_text: str
    section_found: bool
    warning: str = ""


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_nct_study(nct_id: str, timeout: float = 20.0) -> TrialSource:
    normalized_id = nct_id.strip().upper()
    if not NCT_PATTERN.fullmatch(normalized_id):
        raise SourceError("NCT 编号格式应为 NCT 加 8 位数字，例如 NCT02347774。")

    try:
        response = httpx.get(f"{CLINICAL_TRIALS_API}/{normalized_id}", timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise SourceError(f"未找到试验 {normalized_id}。") from exc
        raise SourceError("ClinicalTrials.gov 暂时无法返回该试验。") from exc
    except httpx.HTTPError as exc:
        raise SourceError("无法连接 ClinicalTrials.gov，请稍后重试。") from exc

    study: dict[str, Any] = response.json()
    protocol = study.get("protocolSection", {})
    identification = protocol.get("identificationModule", {})
    eligibility = protocol.get("eligibilityModule", {})
    criteria_text = normalize_text(eligibility.get("eligibilityCriteria", ""))
    if not criteria_text:
        raise SourceError("该试验记录未提供可用的入排标准文本。")

    sponsor = (
        protocol.get("sponsorCollaboratorsModule", {})
        .get("leadSponsor", {})
        .get("name", "")
    )
    return TrialSource(
        source_type="clinicaltrials",
        identifier=normalized_id,
        title=identification.get("briefTitle") or normalized_id,
        source_reference=f"https://clinicaltrials.gov/study/{normalized_id}",
        criteria_text=criteria_text,
        metadata={
            "official_title": identification.get("officialTitle", ""),
            "sponsor": sponsor,
            "overall_status": protocol.get("statusModule", {}).get("overallStatus", ""),
            "minimum_age": eligibility.get("minimumAge", ""),
            "maximum_age": eligibility.get("maximumAge", ""),
            "sex": eligibility.get("sex", ""),
        },
    )


def source_from_text(text: str, title: str = "手动输入的入排标准") -> TrialSource:
    normalized = normalize_text(text)
    if len(normalized) < 30:
        raise SourceError("入排标准文本过短，请提供更完整的内容。")
    return TrialSource(
        source_type="text",
        identifier="manual-text",
        title=title,
        source_reference="user-provided-text",
        criteria_text=normalized,
    )


def _find_eligibility_section(full_text: str) -> tuple[str, bool]:
    start_pattern = re.compile(
        r"(?im)^\s*(?:\d+[.)]\s*)?(?:eligibility criteria|key eligibility criteria|"
        r"inclusion criteria|key inclusion criteria)\s*:?[ \t]*$"
    )
    end_pattern = re.compile(
        r"(?im)^\s*(?:\d+[.)]\s*)?(?:outcome measures?|study interventions?|"
        r"study design|statistical (?:analysis|methods?)|contacts? and locations?|"
        r"references|adverse events?|schedule of (?:activities|assessments))\s*:?[ \t]*$"
    )
    start = start_pattern.search(full_text)
    if not start:
        return full_text, False
    end = end_pattern.search(full_text, start.end())
    section = full_text[start.start() : end.start() if end else len(full_text)]
    return section.strip(), True


def extract_searchable_pdf(pdf_bytes: bytes, filename: str = "uploaded.pdf") -> PDFExtraction:
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise PDFLimitError("PDF 超过 20 MB 限制。")
    if not pdf_bytes.startswith(b"%PDF"):
        raise SourceError("文件不是有效的 PDF。")

    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # PyMuPDF exposes several parser exception types.
        raise SourceError("PDF 无法打开或文件已损坏。") from exc

    with document:
        page_count = document.page_count
        if page_count == 0:
            raise SourceError("PDF 没有可读取的页面。")
        if page_count > MAX_PDF_PAGES:
            raise PDFLimitError("PDF 超过 200 页限制。")
        pages = [page.get_text("text", sort=True) for page in document]

    full_text = normalize_text("\n\n".join(pages))
    average_chars = len(full_text) / page_count
    if len(full_text) < MIN_EXTRACTED_TEXT_CHARS or average_chars < 30:
        raise PDFScannedError(
            "该 PDF 很可能是扫描件，首版暂不进行 OCR。请上传可搜索 PDF 或粘贴标准文本。"
        )

    criteria_text, found = _find_eligibility_section(full_text)
    warning = "" if found else "未自动识别到入排标准章节，已展示全文，请手动确认需要解析的部分。"
    return PDFExtraction(
        filename=filename,
        page_count=page_count,
        full_text=full_text,
        criteria_text=criteria_text,
        section_found=found,
        warning=warning,
    )
