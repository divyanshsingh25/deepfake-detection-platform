"""
utils/report_generator.py
=========================
Generates a professional forensic PDF report for each analysis.
Uses ReportLab (free, no API key required).

Report sections:
  1. Header / Branding
  2. File Information
  3. Analysis Result
  4. Confidence & Risk
  5. Grad-CAM heatmap (optional)
  6. Recommendation
  7. Legal / Cybercrime Portal link
  8. Footer
"""

import os
import io
from datetime import datetime
from typing import Optional
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable, PageBreak
)
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF


# ─────────────────────────────────────────────
# Colour Palette
# ─────────────────────────────────────────────
COLOR_DARK_BLUE  = colors.HexColor("#1A237E")
COLOR_ACCENT     = colors.HexColor("#0D47A1")
COLOR_LIGHT_GREY = colors.HexColor("#ECEFF1")
COLOR_RED_HIGH   = colors.HexColor("#B71C1C")
COLOR_ORANGE_MED = colors.HexColor("#E65100")
COLOR_GREEN_LOW  = colors.HexColor("#1B5E20")
COLOR_FAKE       = colors.HexColor("#C62828")
COLOR_REAL       = colors.HexColor("#2E7D32")
COLOR_WHITE      = colors.white
COLOR_BLACK      = colors.black


def _risk_color(risk: str):
    if "High"   in risk: return COLOR_RED_HIGH
    if "Medium" in risk: return COLOR_ORANGE_MED
    return COLOR_GREEN_LOW


def _label_color(label: str):
    return COLOR_FAKE if label.lower() == "fake" else COLOR_REAL


# ─────────────────────────────────────────────
# Style Registry
# ─────────────────────────────────────────────
def _build_styles():
    base = getSampleStyleSheet()

    styles = {
        "Title": ParagraphStyle(
            "Title",
            fontName="Helvetica-Bold",
            fontSize=22,
            textColor=COLOR_WHITE,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle",
            fontName="Helvetica",
            fontSize=11,
            textColor=COLOR_WHITE,
            alignment=TA_CENTER,
            spaceAfter=2,
        ),
        "SectionHeader": ParagraphStyle(
            "SectionHeader",
            fontName="Helvetica-Bold",
            fontSize=13,
            textColor=COLOR_DARK_BLUE,
            spaceBefore=14,
            spaceAfter=6,
            borderPad=4,
        ),
        "Body": ParagraphStyle(
            "Body",
            fontName="Helvetica",
            fontSize=10,
            textColor=COLOR_BLACK,
            leading=16,
            spaceAfter=4,
            alignment=TA_JUSTIFY,
        ),
        "Small": ParagraphStyle(
            "Small",
            fontName="Helvetica",
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER,
        ),
        "Link": ParagraphStyle(
            "Link",
            fontName="Helvetica",
            fontSize=10,
            textColor=COLOR_ACCENT,
            spaceAfter=4,
        ),
    }
    return styles


# ─────────────────────────────────────────────
# Main Report Generator
# ─────────────────────────────────────────────
def generate_report(
    output_path: str,
    file_name: str,
    model_name: str,
    label: str,
    confidence: float,
    risk_level: str,
    recommendation: str,
    frame_count: int = 0,
    real_frames: int = 0,
    fake_frames: int = 0,
    heatmap_image_path: Optional[str] = None,
    face_image_path: Optional[str] = None,
    extra_notes: str = "",
) -> str:
    """
    Generate a forensic PDF report.

    Args:
        output_path  : Where to save the PDF
        file_name    : Original uploaded file name
        model_name   : e.g. 'ResNet50' or 'EfficientNet-B0'
        label        : 'Real' or 'Fake'
        confidence   : 0–100 float
        risk_level   : 'Low' / 'Medium' / 'High'
        recommendation: Text recommendation
        frame_count  : For videos (0 for images)
        real_frames  : Frames classified as real
        fake_frames  : Frames classified as fake
        heatmap_image_path : Path to Grad-CAM image (optional)
        face_image_path    : Path to extracted face image (optional)
        extra_notes  : Additional analyst notes

    Returns:
        output_path (confirmed)
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=1.5*cm, bottomMargin=2*cm,
    )

    styles = _build_styles()
    story  = []
    W = 17 * cm   # usable page width

    # ── Header Banner ─────────────────────────────
    header_table = Table(
        [[
            Paragraph("🛡 DeepShield", styles["Title"]),
            Paragraph("AI-Based Deepfake Detection System<br/>Forensic Analysis Report",
                      styles["Subtitle"]),
        ]],
        colWidths=[6*cm, 11*cm]
    )
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), COLOR_DARK_BLUE),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("ROWPADDING", (0, 0), (-1, -1), 12),
        ("ROUNDEDCORNERS", [6]),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Report Metadata ────────────────────────────
    now = datetime.now()
    meta_data = [
        ["Report Generated", now.strftime("%A, %d %B %Y — %H:%M:%S")],
        ["Report ID",        f"DS-{now.strftime('%Y%m%d%H%M%S')}"],
        ["System Version",   "DeepShield v1.0"],
    ]
    meta_table = Table(meta_data, colWidths=[5*cm, 12*cm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (0, -1), COLOR_LIGHT_GREY),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWPADDING",  (0, 0), (-1, -1), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.3*cm))

    # ── File Information ───────────────────────────
    story.append(Paragraph("1. File Information", styles["SectionHeader"]))
    story.append(HRFlowable(width=W, thickness=1, color=COLOR_ACCENT))
    story.append(Spacer(1, 0.2*cm))

    file_data = [
        ["File Name",   file_name],
        ["Analysis Date", now.strftime("%d/%m/%Y")],
        ["Analysis Time", now.strftime("%H:%M:%S")],
        ["Model Used",  model_name],
        ["Frame Count", str(frame_count) if frame_count > 0 else "N/A (Image)"],
    ]
    file_table = Table(file_data, colWidths=[5*cm, 12*cm])
    file_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (0, -1), COLOR_LIGHT_GREY),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWPADDING",  (0, 0), (-1, -1), 6),
    ]))
    story.append(file_table)

    # ── Analysis Result ────────────────────────────
    story.append(Paragraph("2. Analysis Result", styles["SectionHeader"]))
    story.append(HRFlowable(width=W, thickness=1, color=COLOR_ACCENT))
    story.append(Spacer(1, 0.2*cm))

    verdict_bg = COLOR_FAKE if label.lower() == "fake" else COLOR_REAL
    verdict_table = Table(
        [[Paragraph(f"VERDICT: {label.upper()}", ParagraphStyle(
            "Verdict",
            fontName="Helvetica-Bold",
            fontSize=18,
            textColor=COLOR_WHITE,
            alignment=TA_CENTER,
        ))]],
        colWidths=[W]
    )
    verdict_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), verdict_bg),
        ("ROWPADDING",    (0, 0), (-1, -1), 14),
        ("ROUNDEDCORNERS", [8]),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 0.3*cm))

    # Confidence + Risk side by side
    conf_text  = f"Confidence: {confidence:.1f}%"
    risk_text  = f"Risk Level: {risk_level}"
    rc_table = Table(
        [[
            Paragraph(conf_text, ParagraphStyle("Conf", fontName="Helvetica-Bold",
                       fontSize=13, textColor=COLOR_WHITE, alignment=TA_CENTER)),
            Paragraph(risk_text, ParagraphStyle("Risk", fontName="Helvetica-Bold",
                       fontSize=13, textColor=COLOR_WHITE, alignment=TA_CENTER)),
        ]],
        colWidths=[W/2 - 0.2*cm, W/2 - 0.2*cm],
        spaceBefore=0.2*cm,
    )
    rc_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (0, 0), COLOR_ACCENT),
        ("BACKGROUND",  (1, 0), (1, 0), _risk_color(risk_level)),
        ("ROWPADDING",  (0, 0), (-1, -1), 10),
        ("ROUNDEDCORNERS", [6]),
    ]))
    story.append(rc_table)

    # Frame breakdown (only for videos)
    if frame_count > 0:
        story.append(Spacer(1, 0.3*cm))
        fb_data = [
            ["Total Frames Analysed", str(frame_count)],
            ["Real Frames",           str(real_frames)],
            ["Fake Frames",           str(fake_frames)],
            ["Fake Frame %",          f"{(fake_frames/max(frame_count,1))*100:.1f}%"],
        ]
        fb_table = Table(fb_data, colWidths=[6*cm, 11*cm])
        fb_table.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (0, -1), COLOR_LIGHT_GREY),
            ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 10),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWPADDING",  (0, 0), (-1, -1), 6),
        ]))
        story.append(fb_table)

    # ── Visual Evidence (Face + Heatmap) ──────────
    if face_image_path and os.path.exists(face_image_path):
        story.append(Paragraph("3. Visual Evidence", styles["SectionHeader"]))
        story.append(HRFlowable(width=W, thickness=1, color=COLOR_ACCENT))
        story.append(Spacer(1, 0.2*cm))

        img_row = []
        captions = []

        face_img = RLImage(face_image_path, width=7*cm, height=7*cm)
        img_row.append(face_img)
        captions.append("Extracted Face Region")

        if heatmap_image_path and os.path.exists(heatmap_image_path):
            heat_img = RLImage(heatmap_image_path, width=7*cm, height=7*cm)
            img_row.append(heat_img)
            captions.append("Grad-CAM Attention Map")
        else:
            img_row.append(Paragraph("", styles["Body"]))
            captions.append("")

        img_table = Table([img_row, [
            Paragraph(captions[0], styles["Small"]),
            Paragraph(captions[1], styles["Small"]) if len(captions) > 1 else Paragraph("", styles["Small"]),
        ]], colWidths=[W/2, W/2])
        img_table.setStyle(TableStyle([
            ("ALIGN",    (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",   (0, 0), (-1, -1), "MIDDLE"),
            ("GRID",     (0, 0), (-1, -1), 0.3, colors.lightgrey),
            ("ROWPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(img_table)

    # ── Recommendation ─────────────────────────────
    rec_num = "4" if (face_image_path and os.path.exists(face_image_path or "")) else "3"
    story.append(Paragraph(f"{rec_num}. Recommendation", styles["SectionHeader"]))
    story.append(HRFlowable(width=W, thickness=1, color=COLOR_ACCENT))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(recommendation, styles["Body"]))

    if extra_notes:
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph(f"<b>Analyst Notes:</b> {extra_notes}", styles["Body"]))

    # ── Cybercrime Portal ──────────────────────────
    cyb_num = str(int(rec_num) + 1)
    story.append(Paragraph(f"{cyb_num}. Cybercrime Reporting", styles["SectionHeader"]))
    story.append(HRFlowable(width=W, thickness=1, color=COLOR_ACCENT))
    story.append(Spacer(1, 0.2*cm))

    portal_text = (
        "If this content has been used for harassment, fraud, or any other criminal "
        "activity, you are encouraged to report it to India's National Cyber Crime "
        "Reporting Portal:"
    )
    story.append(Paragraph(portal_text, styles["Body"]))
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph(
        '<link href="https://cybercrime.gov.in" color="blue">'
        '<u>https://cybercrime.gov.in</u></link>',
        styles["Link"]
    ))
    story.append(Paragraph(
        "You can also call the National Cyber Crime Helpline: <b>1930</b>",
        styles["Body"]
    ))

    # ── Ethics & Legal Disclaimer ──────────────────
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width=W, thickness=0.5, color=colors.grey))
    disclaimer = (
        "<b>DISCLAIMER:</b> This report is generated by an AI system and is intended "
        "for informational purposes only. Results should be verified by a qualified "
        "digital forensics expert before any legal proceedings. DeepShield is not "
        "liable for decisions made solely on the basis of this automated analysis. "
        "Misuse of deepfake detection tools for false accusations is a criminal offence."
    )
    story.append(Paragraph(disclaimer, ParagraphStyle(
        "Disclaimer",
        fontName="Helvetica-Oblique",
        fontSize=8,
        textColor=colors.grey,
        leading=12,
        spaceBefore=6,
        alignment=TA_JUSTIFY,
    )))

    # ── Footer ─────────────────────────────────────
    story.append(Spacer(1, 0.3*cm))
    footer_data = [[
        Paragraph("DeepShield v1.0 | AI-Based Deepfake Detection", styles["Small"]),
        Paragraph(f"Generated: {now.strftime('%d/%m/%Y %H:%M')}", styles["Small"]),
    ]]
    footer_table = Table(footer_data, colWidths=[W/2, W/2])
    footer_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), COLOR_LIGHT_GREY),
        ("ROWPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(footer_table)

    # ── Build PDF ──────────────────────────────────
    doc.build(story)
    return output_path


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    path = generate_report(
        output_path="test_report.pdf",
        file_name="suspicious_video.mp4",
        model_name="ResNet50 (Transfer Learning)",
        label="Fake",
        confidence=91.4,
        risk_level="High",
        recommendation="HIGH CONFIDENCE DEEPFAKE DETECTED. Report immediately.",
        frame_count=20,
        real_frames=3,
        fake_frames=17,
    )
    print(f"Report saved: {path}")
