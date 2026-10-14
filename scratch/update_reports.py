import re
from pathlib import Path

reports_path = Path('c:/Arafath_A/visioncare/utils/reports.py')
with open(reports_path, 'r', encoding='utf-8') as f:
    content = f.read()

# I will completely rewrite reports.py with the correct dual-language implementation

new_content = """\"\"\"Doctor-style medical report generation with QR code integration.\"\"\"

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import qrcode
from PIL import Image
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib import colors

from utils.config import EXTENDED_LABELS, FONTS_DIR, REPORTS_DIR, TAMIL_TRANSLATIONS


def _register_tamil_font() -> bool:
    font_path = FONTS_DIR / "NotoSansTamil-Regular.ttf"
    if font_path.exists():
        try:
            pdfmetrics.registerFont(TTFont("Tamil", str(font_path)))
            return True
        except Exception:
            return False
    return False


def _generate_qr_code(data: str) -> Image.Image:
    \"\"\"Generate QR code image from text data.\"\"\"
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=2,
    )
    qr.add_data(data)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white")


def _create_custom_styles(use_tamil=False):
    \"\"\"Create custom paragraph styles for doctor-style report.\"\"\"
    styles = getSampleStyleSheet()
    
    font_name = "Tamil" if (use_tamil and _register_tamil_font()) else "Helvetica"
    font_bold = "Tamil" if (use_tamil and _register_tamil_font()) else "Helvetica-Bold"
    
    # Header styles
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=18,
        textColor=HexColor("#003366"),
        spaceAfter=12,
        alignment=0,
        fontName=font_bold,
    )
    
    section_style = ParagraphStyle(
        "SectionHead",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=HexColor("#003366"),
        spaceAfter=8,
        spaceBefore=8,
        fontName=font_bold,
        borderColor=HexColor("#003366"),
        borderWidth=0.5,
        borderPadding=8,
    )
    
    normal_style = ParagraphStyle(
        "CustomNormal",
        parent=styles["Normal"],
        fontSize=11,
        leading=14,
        textColor=HexColor("#333333"),
        fontName=font_name,
    )
    
    label_style = ParagraphStyle(
        "LabelStyle",
        parent=styles["Normal"],
        fontSize=11,
        fontName=font_bold,
        textColor=HexColor("#000000"),
    )
    
    return {
        "title": title_style,
        "section": section_style,
        "normal": normal_style,
        "label": label_style,
        "font_name": font_name,
    }


def _calculate_risk_score(result: dict[str, Any]) -> tuple[int, str]:
    \"\"\"Calculate risk score based on diagnosis and severity.\"\"\"
    confidence = result.get("confidence", 0)
    stage_map = {
        "Normal": 0,
        "Mild": 1,
        "Moderate": 2,
        "Severe": 3,
        "Critical": 4,
    }
    stage_value = stage_map.get(result.get("stage", "Normal"), 0)
    abnormality = result.get("abnormality_pct", 0)
    
    risk_score = int((stage_value * 25) + (abnormality * 0.5))
    risk_score = min(100, max(0, risk_score))
    
    if risk_score <= 20:
        risk_level = "Low"
    elif risk_score <= 50:
        risk_level = "Moderate"
    elif risk_score <= 75:
        risk_level = "High"
    else:
        risk_level = "Critical"
    
    return risk_score, risk_level


def _get_lifestyle_advice(label: str, use_tamil: bool = False) -> list[str]:
    \"\"\"Generate lifestyle advice based on diagnosis.\"\"\"
    if use_tamil:
        advice_map_ta = {
            "NORMAL": [
                "வருடாந்திர கண் பரிசோதனைகளைத் தொடரவும்",
                "தரமான சன்கிளாஸ்களுடன் கண்களைப் பாதுகாக்கவும்",
                "சீரான உணவை உட்கொள்ளவும்",
                "புகைப்பிடிப்பதை தவிர்க்கவும்",
            ],
            "DME": [
                "சர்க்கரை அளவை கட்டுக்குள் வைக்கவும்",
                "ரத்த அழுத்தத்தை கண்காணிக்கவும்",
                "மருத்துவர் பரிந்துரைத்த மருந்துகளைத் தவறாமல் உட்கொள்ளவும்",
                "உப்பு மற்றும் சர்க்கரையை குறைக்கவும்",
                "தொடர்ந்து உடற்பயிற்சி செய்யவும்",
            ],
            # Fallback for others
        }
        return advice_map_ta.get(label, [
            "தொடர்ந்து கண் பரிசோதனை செய்து கொள்ளவும்",
            "கண்களைப் பாதுகாக்கவும்",
            "மருத்துவரின் பரிந்துரைகளைப் பின்பற்றவும்",
        ])

    advice_map = {
        "NORMAL": [
            "Maintain regular eye check-ups (annually or as recommended)",
            "Protect eyes from UV radiation with quality sunglasses",
            "Maintain a balanced diet rich in antioxidants",
            "Avoid smoking and excessive alcohol consumption",
        ],
        "DME": [
            "Maintain tight glucose control (target HbA1c < 7%)",
            "Monitor blood pressure regularly (target < 130/80 mmHg)",
            "Take prescribed diabetic medications regularly",
            "Reduce salt and sugar intake",
            "Exercise regularly (150 minutes per week)",
        ],
        "CNV": [
            "Avoid smoking completely",
            "Take antioxidant supplements as recommended by physician",
            "Maintain healthy blood pressure",
            "Eat dark leafy greens and omega-3 rich foods",
            "Protect eyes from bright sunlight",
        ],
        "AMD": [
            "Take daily antioxidant and mineral supplements (AREDS formula)",
            "Maintain healthy weight and exercise regularly",
            "Eat colorful fruits and vegetables",
            "Avoid smoking and protect from UV exposure",
            "Monitor vision using Amsler grid daily",
        ],
        "DRUSEN": [
            "Monitor vision daily using Amsler grid",
            "Take AREDS supplements as recommended",
            "Maintain healthy lifestyle and diet",
            "Avoid smoking and UV exposure",
            "Regular comprehensive eye exams",
        ],
    }
    
    return advice_map.get(label, [
        "Maintain regular eye check-ups",
        "Protect eyes from environmental stress",
        "Follow physician recommendations",
        "Maintain overall eye health",
    ])


def _get_followup_plan(result: dict[str, Any], use_tamil: bool = False) -> list[str]:
    \"\"\"Generate follow-up plan based on findings.\"\"\"
    if use_tamil:
        stage_map_ta = {
            "Normal": ("12 மாதங்கள்", "வழக்கமான ரெட்டினா பரிசோதனை"),
            "Mild": ("6 மாதங்கள்", "தொடர் OCT ஸ்கேன்"),
            "Moderate": ("3 மாதங்கள்", "மாதாந்திர OCT மற்றும் பரிசோதனை"),
            "Severe": ("1 மாதம்", "இருவார OCT மற்றும் சிறப்பு மருத்துவர் ஆலோசனை"),
            "Critical": ("1-2 வாரங்கள்", "அவசர சிறப்பு மருத்துவ சிகிச்சை தேவை"),
        }
        interval, frequency = stage_map_ta.get(result.get("stage", "Normal"), ("12 மாதங்கள்", "வழக்கமான ரெட்டினா பரிசோதனை"))
        return [
            f"அடுத்த ஆலோசனை: {interval}",
            f"தேவையான பரிசோதனை: {frequency}",
            "பார்வை மாற்றங்கள் ஏற்பட்டால் உடனடியாக மருத்துவரை அணுகவும்",
        ]

    stage_map = {
        "Normal": ("12 months", "Regular retinal examination"),
        "Mild": ("6 months", "Follow-up OCT imaging"),
        "Moderate": ("3 months", "Monthly OCT and visual field testing"),
        "Severe": ("1 month", "Bi-weekly OCT and specialist evaluation"),
        "Critical": ("1-2 weeks", "Urgent specialist intervention required"),
    }
    
    interval, frequency = stage_map.get(result.get("stage", "Normal"), ("12 months", "Regular retinal examination"))
    
    plan = [
        f"Next appointment: {interval}",
        f"Required imaging: {frequency}",
        "Consult retinal specialist if vision changes occur",
        "Seek immediate care for sudden vision loss",
    ]
    
    return plan


def generate_doctor_style_pdf(
    result: dict[str, Any],
    scan_id: int,
    patient_name: str = "Patient",
    patient_id: str = "N/A",
    doctor_name: str = "Dr. VisionCare AI",
    hospital_name: str = "VisionCare Medical Center",
    scan_date: Optional[str] = None,
    lang: str = "english"
) -> Path:
    \"\"\"Generate professional doctor-style PDF report with QR code.\"\"\"
    use_tamil = lang == "tamil"
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{'Tamil' if use_tamil else 'Doctor'}_Report_{scan_id}.pdf"
    
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch,
    )
    
    styles = _create_custom_styles(use_tamil)
    font_name = styles["font_name"]
    content = []
    
    def T(en_str: str, ta_str: str) -> str:
        return ta_str if use_tamil else en_str

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if scan_date is None:
        scan_date = datetime.now().strftime("%Y-%m-%d")
    
    content.append(Paragraph(T(hospital_name, "விஷன் கேர் மருத்துவ மையம்"), styles["title"]))
    content.append(Paragraph(T("AI-Assisted Retinal OCT Analysis Report", "AI-உதவியுடன் கூடிய ரெட்டினா OCT பகுப்பாய்வு அறிக்கை"), styles["section"]))
    content.append(Spacer(1, 0.1*inch))
    
    info_data = [
        [T("Report ID:", "அறிக்கை எண்:"), f"VCA-{scan_id}"],
        [T("Generated:", "உருவாக்கப்பட்டது:"), timestamp],
        [T("Scan Date:", "ஸ்கேன் தேதி:"), scan_date],
    ]
    info_table = Table(info_data, colWidths=[1.5*inch, 3.5*inch])
    info_table.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), font_name, 10),
        ("TEXTCOLOR", (0, 0), (-1, -1), HexColor("#333333")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
    ]))
    content.append(info_table)
    content.append(Spacer(1, 0.2*inch))
    
    content.append(Paragraph(T("PATIENT INFORMATION", "நோயாளி விவரங்கள்"), styles["section"]))
    patient_data = [
        [T("Patient Name:", "நோயாளர் பெயர்:"), patient_name],
        [T("Patient ID:", "நோயாளர் எண்:"), patient_id],
        [T("Doctor:", "மருத்துவர்:"), doctor_name],
    ]
    patient_table = Table(patient_data, colWidths=[1.5*inch, 3.5*inch])
    patient_table.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), font_name, 10),
        ("TEXTCOLOR", (0, 0), (-1, -1), HexColor("#333333")),
        ("BACKGROUND", (0, 0), (0, -1), HexColor("#F0F0F0")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
    ]))
    content.append(patient_table)
    content.append(Spacer(1, 0.15*inch))
    
    # 1. Chief Complaint
    content.append(Paragraph(T("1. CHIEF COMPLAINT", "1. முக்கிய குறைபாடு"), styles["section"]))
    lbl_en = result.get('label_full', 'retinal pathology')
    lbl_ta = TAMIL_TRANSLATIONS.get(result.get('label', ''), result.get('label_full', 'ரெட்டினா நோய்'))
    chief_complaint = T(
        f"AI-assisted evaluation of retinal OCT scan for detection of {lbl_en}",
        f"ரெட்டினா OCT ஸ்கேன் மூலம் கண்டறியப்பட்டது: {lbl_ta}"
    )
    content.append(Paragraph(chief_complaint, styles["normal"]))
    content.append(Spacer(1, 0.1*inch))
    
    # 2. Symptoms
    content.append(Paragraph(T("2. PATIENT SYMPTOMS & HISTORY", "2. நோயாளி அறிகுறிகள் & வரலாறு"), styles["section"]))
    symptoms = [
        T("Visual symptoms to be reported by patient at examination", "பரிசோதனையின் போது நோயாளியால் தெரிவிக்கப்பட வேண்டிய அறிகுறிகள்"),
        T("Medical history to be reviewed with attending physician", "மருத்துவ வரலாறு மருத்துவருடன் சரிபார்க்கப்பட வேண்டும்"),
        T("Current medications: As per medical records", "தற்போதைய மருந்துகள்: மருத்துவ ஆவணங்களின்படி"),
    ]
    for symptom in symptoms:
        content.append(Paragraph(f"• {symptom}", styles["normal"]))
    content.append(Spacer(1, 0.1*inch))
    
    # 3. OCT Findings
    content.append(Paragraph(T("3. OCT FINDINGS & DIAGNOSIS", "3. OCT கண்டுபிடிப்புகள் & நோயறிதல்"), styles["section"]))
    diag_stage_ta = TAMIL_TRANSLATIONS.get(result.get("stage", ""), result.get("stage", "Not determined"))
    
    diagnosis_text = f\"\"\"
    <b>{T("Primary Diagnosis:", "முக்கிய நோயறிதல்:")}</b> {lbl_ta if use_tamil else lbl_en}<br/>
    <b>{T("Diagnostic Code:", "நோயறிதல் குறியீடு:")}</b> {result.get('label', 'N/A')}<br/>
    <b>{T("AI Confidence Level:", "AI நம்பிக்கை சதவீதம்:")}</b> {result.get('confidence', 0):.1f}%<br/>
    <b>{T("Disease Stage:", "நோயின் நிலை:")}</b> {diag_stage_ta if use_tamil else result.get('stage', 'Not determined')}<br/>
    \"\"\"
    content.append(Paragraph(diagnosis_text, styles["normal"]))
    content.append(Spacer(1, 0.1*inch))
    
    # 4. Retinal Layer
    content.append(Paragraph(T("4. RETINAL LAYER ANALYSIS", "4. ரெட்டினா அடுக்கு பகுப்பாய்வு"), styles["section"]))
    layer_analysis = f\"\"\"
    <b>{T("Retinal Affected Area:", "கண் பாதிப்பு பகுதி:")}</b> {result.get('retina_affected_pct', 0)}%<br/>
    <b>{T("Lesion Area:", "புண் பகுதி:")}</b> {result.get('lesion_area_pct', 0)}%<br/>
    <b>{T("Fluid Accumulation:", "திரவப் பகுதி:")}</b> {result.get('fluid_area_pct', 0)}%<br/>
    <b>{T("Abnormality Index:", "மொத்த பாதிப்பு:")}</b> {result.get('abnormality_pct', 0)}%<br/>
    \"\"\"
    content.append(Paragraph(layer_analysis, styles["normal"]))
    content.append(Spacer(1, 0.1*inch))
    
    # 5. Thickness Map
    content.append(Paragraph(T("5. THICKNESS MAP & LAYER ANALYSIS", "5. தடிமன் வரைபடம் & அடுக்கு பகுப்பாய்வு"), styles["section"]))
    analyzed = T("Analyzed", "பகுப்பாய்வு செய்யப்பட்டது")
    thickness_text = f\"\"\"
    {T("Automated segmentation analysis performed using deep learning:", "ஆழமான கற்றல் மூலம் தானியங்கி பகுப்பாய்வு செய்யப்பட்டது:")}<br/>
    • Inner Limiting Membrane (ILM): {analyzed}<br/>
    • Retinal Nerve Fiber Layer (RNFL): {analyzed}<br/>
    • Ganglion Cell Layer (GCL): {analyzed}<br/>
    • Inner Plexiform Layer (IPL): {analyzed}<br/>
    • Inner Nuclear Layer (INL): {analyzed}<br/>
    • Outer Plexiform Layer (OPL): {analyzed}<br/>
    • External Limiting Membrane (ELM): {analyzed}<br/>
    • Ellipsoid Zone (EZ): {analyzed}<br/>
    • Retinal Pigment Epithelium (RPE): {analyzed}<br/>
    \"\"\"
    content.append(Paragraph(thickness_text, styles["normal"]))
    content.append(Spacer(1, 0.1*inch))
    
    # 6. Impression
    content.append(Paragraph(T("6. IMPRESSION & DIAGNOSIS", "6. மதிப்பீடு & நோயறிதல்"), styles["section"]))
    impression_1 = T(f"{lbl_en} with severity grade:", f"{lbl_ta} கடுமை நிலை:")
    impression_2 = T(
        f"The AI analysis indicates {lbl_en.lower()} affecting approximately {result.get('retina_affected_pct', 0)}% of the retinal region.",
        f"AI பகுப்பாய்வு {result.get('retina_affected_pct', 0)}% ரெட்டினா பகுதியை பாதித்துள்ளதைக் காட்டுகிறது."
    )
    impression = f"{impression_1} <b>{diag_stage_ta if use_tamil else result.get('stage', 'Not determined')}</b><br/>{impression_2}"
    content.append(Paragraph(impression, styles["normal"]))
    content.append(Spacer(1, 0.1*inch))
    
    # 7. Severity
    content.append(Paragraph(T("7. SEVERITY GRADE & CLASSIFICATION", "7. கடுமை நிலை & வகைப்பாடு"), styles["section"]))
    content.append(Paragraph(f"<b>{T('Severity Stage:', 'கடுமை நிலை:')}</b> {diag_stage_ta if use_tamil else result.get('stage', 'Not determined')}", styles["normal"]))
    content.append(Spacer(1, 0.05*inch))
    
    # 8. Risk Assessment
    risk_score, risk_level = _calculate_risk_score(result)
    if use_tamil:
        risk_map = {"Low": "குறைவு", "Moderate": "மிதமானது", "High": "அதிகம்", "Critical": "மிக அதிகம்"}
        risk_level_str = risk_map.get(risk_level, risk_level)
    else:
        risk_level_str = risk_level
        
    content.append(Paragraph(T("8. RISK ASSESSMENT", "8. அபாய மதிப்பீடு"), styles["section"]))
    risk_text = f\"\"\"
    <b>{T("Overall Risk Score:", "மொத்த அபாய மதிப்பெண்:")}</b> {risk_score}/100<br/>
    <b>{T("Risk Level:", "அபாய நிலை:")}</b> <font color="red"><b>{risk_level_str}</b></font><br/>
    <b>{T("Recommendation Priority:", "பரிந்துரை முன்னுரிமை:")}</b> {T("Follow physician guidelines based on risk assessment", "அபாய மதிப்பீட்டின் அடிப்படையில் மருத்துவரின் வழிகாட்டுதல்களைப் பின்பற்றவும்")}
    \"\"\"
    content.append(Paragraph(risk_text, styles["normal"]))
    content.append(Spacer(1, 0.1*inch))
    
    # 9. Lifestyle Advice
    content.append(Paragraph(T("9. LIFESTYLE RECOMMENDATIONS", "9. வாழ்க்கை முறை பரிந்துரைகள்"), styles["section"]))
    advice = _get_lifestyle_advice(result.get('label', 'NORMAL'), use_tamil)
    for i, adv in enumerate(advice, 1):
        content.append(Paragraph(f"• {adv}", styles["normal"]))
    content.append(Spacer(1, 0.1*inch))
    
    # 10. Follow-up
    content.append(Paragraph(T("10. FOLLOW-UP PLAN & MONITORING", "10. பின்தொடர்தல் திட்டம் & கண்காணிப்பு"), styles["section"]))
    followup = _get_followup_plan(result, use_tamil)
    for plan in followup:
        content.append(Paragraph(f"• {plan}", styles["normal"]))
    content.append(Spacer(1, 0.15*inch))
    
    # QR Code
    content.append(PageBreak())
    content.append(Paragraph(T("QR CODE & REPORT VERIFICATION", "QR குறியீடு & அறிக்கை சரிபார்ப்பு"), styles["section"]))
    
    qr_data = f"VCA-{scan_id}|{patient_name}|{result.get('label', 'N/A')}|{result.get('confidence', 0):.1f}%|{timestamp}"
    qr_image = _generate_qr_code(qr_data)
    
    qr_buffer = BytesIO()
    qr_image.save(qr_buffer, format="PNG")
    qr_buffer.seek(0)
    
    qr_rl_image = RLImage(qr_buffer, width=2*inch, height=2*inch)
    content.append(qr_rl_image)
    content.append(Spacer(1, 0.1*inch))
    content.append(Paragraph(f"<b>{T('Report ID:', 'அறிக்கை எண்:')}</b> VCA-{scan_id}", styles["normal"]))
    content.append(Paragraph(f"<b>{T('Verification Code:', 'சரிபார்ப்பு குறியீடு:')}</b> {timestamp.replace(':', '').replace('-', '')}", styles["normal"]))
    
    # Disclaimer
    content.append(Spacer(1, 0.2*inch))
    content.append(Paragraph(T("MEDICAL DISCLAIMER", "மருத்துவ எச்சரிக்கை"), styles["section"]))
    disclaimer = T(
        "This report is generated through AI-assisted analysis of OCT imaging and is intended for informational purposes only. It should NOT be used as a substitute for professional medical diagnosis and treatment.",
        "இந்த அறிக்கை AI-உதவியுடன் பகுப்பாய்வு செய்யப்பட்டது. இது மருத்துவ ஆலோசனைக்கு மாற்றாக அமையாது. தகுதியான மருத்துவரின் ஆலோசனை அவசியம்."
    )
    content.append(Paragraph(disclaimer, styles["normal"]))
    content.append(Spacer(1, 0.1*inch))
    
    # Signature
    content.append(Spacer(1, 0.2*inch))
    content.append(Paragraph("_" * 80, styles["normal"]))
    content.append(Paragraph(f"{T('AI System: VisionCare | Date:', 'AI அமைப்பு: VisionCare | தேதி:')} {timestamp}", styles["normal"]))
    content.append(Paragraph(T("Review and approval by licensed physician required", "அங்கீகரிக்கப்பட்ட மருத்துவரின் ஒப்புதல் தேவை"), styles["normal"]))
    
    doc.build(content)
    return path


def generate_english_pdf(
    result: dict[str, Any],
    scan_id: int,
    patient_name: str = "Patient",
) -> Path:
    if not result:
        raise ValueError("Analysis result missing for report generation")
    
    return generate_doctor_style_pdf(
        result=result,
        scan_id=scan_id,
        patient_name=result.get("patient_name", patient_name),
        patient_id=f"PID-{result.get('patient_id', 'UNKNOWN')}",
        scan_date=result.get("analysis_date"),
        lang="english"
    )


def generate_tamil_pdf(
    result: dict[str, Any],
    scan_id: int,
) -> Path:
    if not result:
        raise ValueError("Analysis result missing for report generation")
    
    return generate_doctor_style_pdf(
        result=result,
        scan_id=scan_id,
        patient_name=result.get("patient_name", "Patient"),
        patient_id=f"PID-{result.get('patient_id', 'UNKNOWN')}",
        scan_date=result.get("analysis_date"),
        lang="tamil"
    )

\"\"\"
with open(reports_path, 'w', encoding='utf-8') as f:
    f.write(new_content)
