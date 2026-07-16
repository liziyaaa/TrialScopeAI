"""Generate deterministic synthetic fixtures and the human-readable demo PDF."""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.synthetic import generate_edge_cases, generate_synthetic_patients  # noqa: E402


DATA_DIR = ROOT / "data"
PDF_DIR = ROOT / "output" / "pdf"


def _draw_header_footer(canvas, document) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#D7E3F0"))
    canvas.line(18 * mm, height - 16 * mm, width - 18 * mm, height - 16 * mm)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.setFillColor(colors.HexColor("#135D75"))
    canvas.drawString(18 * mm, height - 12 * mm, "TrialScopeAI - Searchable Demo Protocol")
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#60758A"))
    canvas.drawRightString(width - 18 * mm, 11 * mm, f"Page {document.page}")
    canvas.line(18 * mm, 15 * mm, width - 18 * mm, 15 * mm)
    canvas.restoreState()


def build_demo_pdf(trial: dict[str, object]) -> Path:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    output = PDF_DIR / "golden4_demo_protocol.pdf"
    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=24 * mm,
        bottomMargin=22 * mm,
        title="GOLDEN-4 Searchable Demo Protocol",
        author="TrialScopeAI",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=25,
        textColor=colors.HexColor("#0B3C5D"),
        alignment=TA_LEFT,
        spaceAfter=10,
    )
    subtitle = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor("#45657B"),
        spaceAfter=12,
    )
    section = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#0B6E75"),
        spaceBefore=12,
        spaceAfter=8,
    )
    body = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#22313F"),
        spaceAfter=6,
    )
    note = ParagraphStyle(
        "Note",
        parent=body,
        backColor=colors.HexColor("#EAF4F6"),
        borderColor=colors.HexColor("#9CC9CF"),
        borderWidth=0.5,
        borderPadding=8,
        textColor=colors.HexColor("#24515A"),
        spaceAfter=14,
    )

    story = [
        Spacer(1, 8 * mm),
        Paragraph("GOLDEN-4 Eligibility Criteria", title),
        Paragraph(html.escape(str(trial["title"])), subtitle),
        Paragraph(
            "This searchable PDF is a competition demonstration asset derived from the public "
            "ClinicalTrials.gov record NCT02347774. It is not an official sponsor protocol and "
            "contains eligibility text only.",
            note,
        ),
        Paragraph("Source and study metadata", section),
        Paragraph(
            f"<b>Registry:</b> ClinicalTrials.gov<br/>"
            f"<b>NCT ID:</b> {html.escape(str(trial['identifier']))}<br/>"
            f"<b>Source URL:</b> {html.escape(str(trial['source_reference']))}<br/>"
            f"<b>Sponsor:</b> {html.escape(str(trial['metadata']['sponsor']))}<br/>"
            f"<b>Phase:</b> {html.escape(str(trial['metadata']['phase']))}<br/>"
            f"<b>Enrollment:</b> {html.escape(str(trial['metadata']['enrollment']))}",
            body,
        ),
        PageBreak(),
    ]

    for line in str(trial["criteria_text"]).splitlines():
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 3 * mm))
        elif stripped.rstrip(":") in {"Inclusion Criteria", "Exclusion Criteria"}:
            story.append(Paragraph(html.escape(stripped), section))
        else:
            story.append(Paragraph(html.escape(stripped), body))

    document.build(story, onFirstPage=_draw_header_footer, onLaterPages=_draw_header_footer)
    return output


def main() -> None:
    trial = json.loads((DATA_DIR / "golden4_trial.json").read_text(encoding="utf-8"))
    patients = generate_synthetic_patients()
    patients.to_csv(DATA_DIR / "synthetic_patients.csv", index=False, encoding="utf-8")

    edge_cases = generate_edge_cases()
    (DATA_DIR / "golden4_edge_cases.json").write_text(
        json.dumps(edge_cases, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "seed": 20260716,
        "synthetic_patient_count": len(patients),
        "edge_case_count": len(edge_cases),
        "source": trial["source_reference"],
        "disclaimer": "All patient records are synthetic and for prototype validation only.",
    }
    (DATA_DIR / "demo_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pdf_path = build_demo_pdf(trial)
    print(f"Generated {len(patients)} patients, {len(edge_cases)} edge cases, and {pdf_path}")


if __name__ == "__main__":
    main()
