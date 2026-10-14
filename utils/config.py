"""Global configuration — CPU-only execution."""

import os
from pathlib import Path

# Force CPU-only
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
SCANS_DIR = OUTPUTS_DIR / "scans"
HEATMAPS_DIR = OUTPUTS_DIR / "heatmaps"
MASKS_DIR = OUTPUTS_DIR / "masks"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FONTS_DIR = ROOT / "fonts"
DATASETS_DIR = ROOT / "datasets"

IMAGE_SIZE = 224
ENSEMBLE_WEIGHTS = (0.7, 0.3)

CLASS_LABELS = [
    "CNV",
    "DME",
    "DRUSEN",
    "NORMAL",
    "AMD",
    "CSCR",
    "ERM",
    "MH",
    "RD",
    "GLAUCOMA",
    "RP",
]

EXTENDED_LABELS = {
    "CNV": "Choroidal Neovascularization",
    "DME": "Diabetic Macular Edema",
    "DRUSEN": "Drusen",
    "NORMAL": "Normal Retina",
    "AMD": "Age-Related Macular Degeneration",
    "CSCR": "Central Serous Chorioretinopathy",
    "ERM": "Epiretinal Membrane",
    "MH": "Macular Hole",
    "RD": "Retinal Detachment",
    "GLAUCOMA": "Glaucoma",
    "RP": "Retinitis Pigmentosa",
    "RETINAL_EDEMA": "Retinal Edema",
    "MACULAR_DEGENERATION": "Macular Degeneration",
    "RETINAL_FLUID": "Retinal Fluid",
    "LESIONS": "Retinal Lesions",
}

STAGES = ["Normal", "Mild", "Moderate", "Severe", "Critical"]

TAMIL_TRANSLATIONS = {
    "CNV": "விழித்திரைக்கீழ் புதிய இரத்தக்குழாய் உருவாக்கம் (CNV)",
    "DME": "நீரிழிவு விழித்திரை மேக்குலா நீர்வீக்கம் (DME)",
    "DRUSEN": "விழித்திரை கொழுப்புப் படிவுகள் (Drusen)",
    "NORMAL": "இயல்பான விழித்திரை",
    "AMD": "வயது முதிர்வு சார்ந்த மேக்குலா சிதைவு (AMD)",
    "CSCR": "மைய விழித்திரை திரவ சேகரிப்பு (CSCR)",
    "ERM": "விழித்திரை மேல் சவ்வு (ERM)",
    "MH": "மேக்குலா துளை (Macular Hole)",
    "RD": "விழித்திரை விலகல் / பிரிதல் (RD)",
    "GLAUCOMA": "கண் அழுத்த நோய் (Glaucoma)",
    "RP": "விழித்திரை நிறமி அழற்சி நோய் (RP)",
    "Normal": "இயல்பானது",
    "Mild": "லேசான பாதிப்பு",
    "Moderate": "மிதமான பாதிப்பு",
    "Severe": "தீவிரமான பாதிப்பு",
    "Critical": "அபாயகரமான பாதிப்பு",
}

INVALID_MSG = "Please upload a valid retinal OCT scan image."
