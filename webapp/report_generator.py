from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import hashlib
import datetime
import os

def generate_pdf_report(file_path, result_dict):
    os.makedirs("webapp/reports", exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"AI_Deepfake_Report_{timestamp}.pdf"
    report_path = os.path.join("webapp/reports", report_name)

    c = canvas.Canvas(report_path, pagesize=A4)
    width, height = A4

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "AI-Based Media Authenticity Analysis Report")

    # Metadata
    c.setFont("Helvetica", 11)
    c.drawString(50, height - 90, f"Generated On: {datetime.datetime.now()}")
    c.drawString(50, height - 110, "System: AI-Based Deepfake Detection System")

    # File info
    c.drawString(50, height - 150, f"File Name: {os.path.basename(file_path)}")

    with open(file_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    c.drawString(50, height - 170, "SHA-256 Hash:")
    c.setFont("Helvetica", 9)
    c.drawString(50, height - 185, file_hash)

    # Results
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 220, f"Authenticity Probability: {result_dict['truth']}%")
    c.drawString(50, height - 240, f"Manipulation Probability: {result_dict['fake']}%")

    c.setFont("Helvetica", 10)
    c.drawString(50, height - 270, f"Custom Model Fake Score: {result_dict['custom_fake']}%")

    if result_dict.get("pretrained_fake") is not None:
        c.drawString(
            50, height - 285,
            f"Pretrained Model Fake Score: {result_dict['pretrained_fake']}%"
        )

    # Summary
    c.setFont("Helvetica", 11)
    c.drawString(50, height - 320, "Technical Summary:")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 340, "- Facial artifact and texture inconsistency analysis")
    c.drawString(50, height - 355, "- AI-generated media pattern detection")
    c.drawString(50, height - 370, "- Ensemble deep learning verification applied")

    # Disclaimer
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(
        50, height - 410,
        "Disclaimer: This report provides an AI-based preliminary analysis intended to assist"
    )
    c.drawString(
        50, height - 425,
        "cybercrime investigations. Final verification must be performed by authorized authorities."
    )

    c.save()
    return report_name
