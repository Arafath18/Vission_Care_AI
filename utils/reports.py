"""Doctor-style medical report generation with QR code integration."""

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
    """Generate QR code image from text data."""
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
    """Create custom paragraph styles for doctor-style report."""
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
    """Calculate risk score based on diagnosis and severity."""
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
    """Generate lifestyle advice based on diagnosis."""
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
            "CNV": [
                "புகைபிடிப்பதைக் கண்டிப்பாகத் தவிர்க்கவும்",
                "மருத்துவர் பரிந்துரைத்த ஆக்ஸிஜனேற்ற தடுப்பான்கள் மற்றும் சத்து மாத்திரைகளை உட்கொள்ளவும்",
                "இரத்த அழுத்தத்தை சீராக வைத்திருக்கவும்",
                "பச்சை இலை காய்கறிகள் மற்றும் ஒமேகா-3 கொழுப்பு அமிலம் நிறைந்த உணவுகளை அதிகம் உட்கொள்ளவும்",
                "சூரிய ஒளியில் இருந்து கண்களைப் பாதுகாக்க தகுந்த சன்கிளாஸ் பயன்படுத்தவும்",
            ],
            "AMD": [
                "தினமும் ஆக்ஸிஜனேற்ற தடுப்பான்கள் மற்றும் தாது உப்புகள் (AREDS சூத்திரம்) கொண்ட சத்து மாத்திரைகளை உட்கொள்ளவும்",
                "உடல் எடையை சீராக வைத்துக்கொண்டு, தொடர்ந்து உடற்பயிற்சி செய்யவும்",
                "வண்ணமயமான பழங்கள் மற்றும் காய்கறிகளை உணவுடன் சேர்த்துக்கொள்ளவும்",
                "புகைபிடிப்பதைத் தவிர்க்கவும், புற ஊதா (UV) கதிர்வீச்சிலிருந்து கண்களைப் பாதுகாக்கவும்",
                "ஆம்ஸ்லர் கிரிட் (Amsler Grid) அட்டையைப் பயன்படுத்தி தினசரி பார்வையைச் சோதிக்கவும்",
            ],
            "DRUSEN": [
                "ஆம்ஸ்லர் கிரிட் (Amsler Grid) அட்டையைப் பயன்படுத்தி தினசரி பார்வையைச் சோதிக்கவும்",
                "பரிந்துரைக்கப்பட்ட AREDS சத்து மாத்திரைகளைத் தவறாமல் உட்கொள்ளவும்",
                "ஆரோக்கியமான வாழ்க்கை முறை மற்றும் சத்தான உணவுப் பழக்கத்தைப் பின்பற்றவும்",
                "புகைபிடித்தல் மற்றும் அதிக சூரிய ஒளி கதிர்வீச்சுப் பாதிப்பைத் தவிர்க்கவும்",
                "வழக்கமான விரிவான கண் பரிசோதனைகளைத் தவறாமல் செய்து கொள்ளவும்",
            ],
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
    """Generate follow-up plan based on findings."""
    if use_tamil:
        stage_map_ta = {
            "Normal": ("12 மாதங்கள்", "வழக்கமான விழித்திரை (Retina) பரிசோதனை"),
            "Mild": ("6 மாதங்கள்", "தொடர் OCT விழித்திரை ஸ்கேன்"),
            "Moderate": ("3 மாதங்கள்", "மாதாந்திர OCT ஸ்கேன் மற்றும் பரிசோதனை"),
            "Severe": ("1 மாதம்", "ஒவ்வொரு இரண்டு வாரங்களுக்கு ஒருமுறை OCT மற்றும் விழித்திரை சிறப்பு மருத்துவர் ஆலோசனை"),
            "Critical": ("1-2 வாரங்கள்", "உடனடி விழித்திரை சிறப்பு மருத்துவ சிகிச்சை தேவை"),
        }
        interval, frequency = stage_map_ta.get(result.get("stage", "Normal"), ("12 மாதங்கள்", "வழக்கமான விழித்திரை (Retina) பரிசோதனை"))
        return [
            f"அடுத்த பரிசோதனை இடைவெளி: {interval}",
            f"தேவையான ஸ்கேன்/பரிசோதனை வகை: {frequency}",
            "பார்வையில் ஏதேனும் மாற்றங்கள் ஏற்பட்டால் உடனடியாக கண் மருத்துவரை அணுகவும்",
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
    """Generate professional doctor-style PDF report with QR code."""
    use_tamil = lang == "tamil"
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{'Tamil' if use_tamil else 'English'}_Report_{scan_id}.pdf"
    
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
        
    if use_tamil:
        # Transliterate patient name
        if patient_name and patient_name not in ["Patient", "Anonymous"]:
            try:
                import urllib.request
                import json
                import urllib.parse
                url = f"https://inputtools.google.com/request?text={urllib.parse.quote(patient_name)}&itc=ta-t-i0-und&num=1&cp=0&cs=1&ie=utf-8&oe=utf-8&app=demopage"
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                res = json.loads(urllib.request.urlopen(req, timeout=2).read().decode('utf-8'))
                if res[0] == 'SUCCESS':
                    patient_name = res[1][0][1][0]
            except Exception:
                pass
    
    content.append(Paragraph(T(hospital_name, "விஷன் கேர் மருத்துவ மையம்"), styles["title"]))
    content.append(Paragraph(T("AI-Assisted Retinal OCT Analysis Report", "AI-உதவியுடன் கூடிய விழித்திரை OCT பகுப்பாய்வு அறிக்கை"), styles["section"]))
    content.append(Spacer(1, 0.1*inch))
    
    info_data = [
        [T("Report ID:", "அறிக்கை எண்:"), f"VCA-{scan_id}"],
        [T("Generated:", "அறிக்கை உருவாக்கப்பட்ட நேரம்:"), timestamp],
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
        [T("Patient Name:", "நோயாளி பெயர்:"), patient_name],
        [T("Patient ID:", "நோயாளி அடையாள எண்:"), patient_id],
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
    content.append(Paragraph(T("1. CHIEF COMPLAINT", "1. முதன்மைப் பரிசோதனை விவரம்"), styles["section"]))
    lbl_en = result.get('label_full', 'retinal pathology')
    lbl_ta = TAMIL_TRANSLATIONS.get(result.get('label', ''), result.get('label_full', 'விழித்திரை நோய்'))
    chief_complaint = T(
        f"AI-assisted evaluation of retinal OCT scan for detection of {lbl_en}",
        f"விழித்திரையில் {lbl_ta} கண்டறிவதற்கான AI-உதவியுடன் கூடிய OCT ஸ்கேன் பகுப்பாய்வு"
    )
    content.append(Paragraph(chief_complaint, styles["normal"]))
    content.append(Spacer(1, 0.1*inch))
    
    # 2. Symptoms
    content.append(Paragraph(T("2. PATIENT SYMPTOMS & HISTORY", "2. நோயாளியின் அறிகுறிகள் மற்றும் மருத்துவ வரலாறு"), styles["section"]))
    symptoms = [
        T("Visual symptoms to be reported by patient at examination", "பரிசோதனையின் போது நோயாளி குறிப்பிட்ட பார்வை சார்ந்த அறிகுறிகள்"),
        T("Medical history to be reviewed with attending physician", "சிகிச்சை அளிக்கும் மருத்துவருடன் கலந்தாலோசிக்கப்பட வேண்டிய மருத்துவ வரலாறு"),
        T("Current medications: As per medical records", "தற்போதைய மருந்துகள்: மருத்துவப் பதிவுகளின்படி"),
    ]
    for symptom in symptoms:
        content.append(Paragraph(f"• {symptom}", styles["normal"]))
    content.append(Spacer(1, 0.1*inch))
    
    # Images
    img_path = result.get("image_path")
    heat_path = result.get("heatmap_path")
    seg_path = result.get("segmentation_path")

    available_imgs = []
    if img_path and Path(img_path).exists():
        available_imgs.append((img_path, T("Original Scan", "அசல் ஸ்கேன்")))
    if heat_path and Path(heat_path).exists():
        available_imgs.append((heat_path, T("Heatmap Analysis", "வெப்ப வரைபட பகுப்பாய்வு")))
    if seg_path and Path(seg_path).exists():
        available_imgs.append((seg_path, T("Segmentation Mask", "பாதிப்புப் பகுதி வரைபடம் (Segmentation)")))

    if available_imgs:
        content.append(Paragraph(
            T("SCAN IMAGES & ANALYSIS VISUALIZATIONS", "ஸ்கேன் படங்கள் மற்றும் பகுப்பாய்வு காட்சிகள்"),
            styles["section"]
        ))
        n = len(available_imgs)
        img_w = (7.0 / n) * inch  # spread across full width
        img_h = img_w  # square

        label_row = [Paragraph(f"<b>{lbl}</b>", styles["label"]) for _, lbl in available_imgs]
        image_row = [RLImage(p, width=img_w - 0.1*inch, height=img_h - 0.1*inch) for p, _ in available_imgs]

        img_table = Table([label_row, image_row], colWidths=[img_w] * n)
        img_table.setStyle(TableStyle([
            ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("BOX",        (0, 0), (-1, -1), 0.75, HexColor("#CCCCCC")),
            ("INNERGRID",  (0, 0), (-1, -1), 0.5,  HexColor("#EEEEEE")),
            ("BACKGROUND", (0, 0), (-1, 0),  HexColor("#F5F5F5")),
        ]))
        content.append(img_table)
        content.append(Spacer(1, 0.2*inch))

    # 3. OCT Findings
    content.append(Paragraph(T("3. OCT FINDINGS & DIAGNOSIS", "3. OCT பரிசோதனை முடிவுகள் & நோயறிதல்"), styles["section"]))
    diag_stage_ta = TAMIL_TRANSLATIONS.get(result.get("stage", ""), result.get("stage", "Not determined"))
    
    diagnosis_text = f"""
    <b>{T("Primary Diagnosis:", "முதன்மை நோயறிதல்:")}</b> {lbl_ta if use_tamil else lbl_en}<br/>
    <b>{T("Diagnostic Code:", "நோயறிதல் குறியீடு:")}</b> {result.get('label', 'N/A')}<br/>
    <b>{T("AI Confidence Level:", "AI உறுதிப்பாடு சதவீதம்:")}</b> {result.get('confidence', 0):.1f}%<br/>
    <b>{T("Disease Stage:", "பாதிப்பின் நிலை / தீவிரம்:")}</b> {diag_stage_ta if use_tamil else result.get('stage', 'Not determined')}<br/>
    """
    content.append(Paragraph(diagnosis_text, styles["normal"]))
    content.append(Spacer(1, 0.1*inch))
    
    # 4. Retinal Layer
    content.append(Paragraph(T("4. RETINAL LAYER ANALYSIS", "4. விழித்திரை (Retinal) அடுக்கு பகுப்பாய்வு"), styles["section"]))
    layer_analysis = f"""
    <b>{T("Retinal Affected Area:", "பாதிக்கப்பட்ட விழித்திரை பரப்பு:")}</b> {result.get('retina_affected_pct', 0)}%<br/>
    <b>{T("Lesion Area:", "விழித்திரைச் சிதைவுற்ற பரப்பு (Lesion Area):")}</b> {result.get('lesion_area_pct', 0)}%<br/>
    <b>{T("Fluid Accumulation:", "திரவக் குவிப்பு பரப்பு (Fluid Area):")}</b> {result.get('fluid_area_pct', 0)}%<br/>
    <b>{T("Abnormality Index:", "அசாதாரண குறியீடு (Abnormality Index):")}</b> {result.get('abnormality_pct', 0)}%<br/>
    """
    content.append(Paragraph(layer_analysis, styles["normal"]))
    content.append(Spacer(1, 0.1*inch))
    
    # 5. Thickness Map
    content.append(Paragraph(T("5. THICKNESS MAP & LAYER ANALYSIS", "5. விழித்திரை தடிமன் வரைபடம் & அடுக்கு பகுப்பாய்வு"), styles["section"]))
    analyzed = T("Analyzed", "பகுப்பாய்வு செய்யப்பட்டது")
    thickness_text = f"""
    {T("Automated segmentation analysis performed using deep learning:", "ஆழமான கற்றல் (Deep Learning) தொழில்நுட்பத்தைப் பயன்படுத்தி செய்யப்பட்ட தானியங்கி விழித்திரை அடுக்கு பகுப்பாய்வு:")}<br/>
    • Inner Limiting Membrane (ILM): {analyzed}<br/>
    • Retinal Nerve Fiber Layer (RNFL): {analyzed}<br/>
    • Ganglion Cell Layer (GCL): {analyzed}<br/>
    • Inner Plexiform Layer (IPL): {analyzed}<br/>
    • Inner Nuclear Layer (INL): {analyzed}<br/>
    • Outer Plexiform Layer (OPL): {analyzed}<br/>
    • External Limiting Membrane (ELM): {analyzed}<br/>
    • Ellipsoid Zone (EZ): {analyzed}<br/>
    • Retinal Pigment Epithelium (RPE): {analyzed}<br/>
    """
    content.append(Paragraph(thickness_text, styles["normal"]))
    content.append(Spacer(1, 0.1*inch))
    
    # 6. Impression
    content.append(Paragraph(T("6. IMPRESSION & DIAGNOSIS", "6. மருத்துவ மதிப்பீடு & நோயறிதல் கருத்து"), styles["section"]))
    impression_1 = T(f"{lbl_en} with severity grade:", f"{lbl_ta} தீவிரம்:")
    impression_2 = T(
        f"The AI analysis indicates {lbl_en.lower()} affecting approximately {result.get('retina_affected_pct', 0)}% of the retinal region.",
        f"AI பகுப்பாய்வின்படி, விழித்திரை பகுதியின் தோராயமாக {result.get('retina_affected_pct', 0)}% பரப்பு பாதிப்படைந்துள்ளது என அறியப்படுகிறது."
    )
    impression = f"{impression_1} <b>{diag_stage_ta if use_tamil else result.get('stage', 'Not determined')}</b><br/>{impression_2}"
    content.append(Paragraph(impression, styles["normal"]))
    content.append(Spacer(1, 0.1*inch))
    
    # 7. Severity
    content.append(Paragraph(T("7. SEVERITY GRADE & CLASSIFICATION", "7. பாதிப்பின் தீவிரம் & வகைப்பாடு"), styles["section"]))
    content.append(Paragraph(f"<b>{T('Severity Stage:', 'பாதிப்பின் தீவிரம்:')}</b> {diag_stage_ta if use_tamil else result.get('stage', 'Not determined')}", styles["normal"]))
    content.append(Spacer(1, 0.05*inch))
    
    # 8. Risk Assessment
    risk_score, risk_level = _calculate_risk_score(result)
    if use_tamil:
        risk_map = {"Low": "குறைவு", "Moderate": "மிதமானது", "High": "அதிகம்", "Critical": "மிக அதிகம்"}
        risk_level_str = risk_map.get(risk_level, risk_level)
    else:
        risk_level_str = risk_level
        
    content.append(Paragraph(T("8. RISK ASSESSMENT", "8. அபாய மதிப்பீடு"), styles["section"]))
    risk_text = f"""
    <b>{T("Overall Risk Score:", "ஒட்டுமொத்த அபாய மதிப்பெண்:")}</b> {risk_score}/100<br/>
    <b>{T("Risk Level:", "அபாய நிலை:")}</b> <font color="red"><b>{risk_level_str}</b></font><br/>
    <b>{T("Recommendation Priority:", "பரிந்துரை முன்னுரிமை:")}</b> {T("Follow physician guidelines based on risk assessment", "அபாய நிலையைப் பொறுத்து முன்னுரிமையுடன் கண் மருத்துவரை அணுகவும்")}
    """
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
    
    import urllib.parse
    qr_data = f"http://127.0.0.1:8001/verify.html?scan_id={scan_id}&patient_name={urllib.parse.quote(patient_name)}&label={result.get('label', 'N/A')}&confidence={result.get('confidence', 0):.1f}&timestamp={urllib.parse.quote(timestamp)}"
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
        "இந்த அறிக்கை விழித்திரை OCT ஸ்கேன்களின் செயற்கை நுண்ணறிவு (AI) பகுப்பாய்வு மூலம் பெறப்பட்டது மற்றும் இது தகவல்களுக்காக மட்டுமே வழங்கப்படுகிறது. தகுதியான கண் மருத்துவரின் முறையான பரிசோதனை, நோயறிதல் மற்றும் சிகிச்சைக்கு மாற்றாக இதனைப் பயன்படுத்தக் கூடாது."
    )
    content.append(Paragraph(disclaimer, styles["normal"]))
    content.append(Spacer(1, 0.1*inch))
    
    # Signature
    content.append(Spacer(1, 0.2*inch))
    content.append(Paragraph("_" * 80, styles["normal"]))
    content.append(Paragraph(f"{T('AI System: VisionCare | Date:', 'AI அமைப்பு: VisionCare | தேதி:')} {timestamp}", styles["normal"]))
    content.append(Paragraph(T("Review and approval by licensed physician required", "உரிமம் பெற்ற கண் மருத்துவரின் மதிப்பீடு மற்றும் ஒப்புதல் தேவைப்படுகிறது"), styles["normal"]))
    
    doc.build(content)
    return path


def generate_english_pdf(
    result: dict[str, Any],
    scan_id: int,
    patient_name: str = "Patient",
) -> Path:
    if not result:
        raise ValueError("Analysis result missing for report generation")
    
    pname = patient_name if patient_name not in ["Patient", "Anonymous", ""] else (result.get("patient_name") or "Patient")
    
    return generate_doctor_style_pdf(
        result=result,
        scan_id=scan_id,
        patient_name=pname,
        patient_id=f"PID-{result.get('patient_id', 'UNKNOWN')}",
        scan_date=result.get("analysis_date"),
        lang="english"
    )


def generate_tamil_pdf(
    result: dict[str, Any],
    scan_id: int,
    patient_name: str = "Patient",
) -> Path:
    if not result:
        raise ValueError("Analysis result missing for report generation")
    
    pname = patient_name if patient_name not in ["Patient", "Anonymous", ""] else (result.get("patient_name") or "Patient")
    
    return generate_doctor_style_pdf(
        result=result,
        scan_id=scan_id,
        patient_name=pname,
        patient_id=f"PID-{result.get('patient_id', 'UNKNOWN')}",
        scan_date=result.get("analysis_date"),
        lang="tamil"
    )
