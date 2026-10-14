import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from utils.reports import generate_tamil_pdf

# Sample analysis result for DME
sample_result_dme = {
    "label": "DME",
    "label_full": "Diabetic Macular Edema",
    "confidence": 87.5,
    "stage": "Moderate",
    "retina_affected_pct": 35.0,
    "lesion_area_pct": 12.0,
    "fluid_area_pct": 23.0,
    "abnormality_pct": 42.0,
    "extended_findings": [
        "Diabetic Macular Edema",
        "Retinal Thickening",
        "Fluid Accumulation",
    ],
    "patient_name": "Anbazhagan",
    "analysis_date": "2026-06-06",
}

# Sample analysis result for CNV
sample_result_cnv = {
    "label": "CNV",
    "label_full": "Choroidal Neovascularization",
    "confidence": 92.4,
    "stage": "Severe",
    "retina_affected_pct": 55.0,
    "lesion_area_pct": 25.0,
    "fluid_area_pct": 30.0,
    "abnormality_pct": 70.0,
    "extended_findings": [
        "Choroidal Neovascularization",
        "Subretinal Fluid",
        "Retinal Lesions",
    ],
    "patient_name": "Raman",
    "analysis_date": "2026-06-06",
}

try:
    print("Generating Tamil DME Report...")
    path_dme = generate_tamil_pdf(sample_result_dme, 999, "Anbazhagan")
    print(f"SUCCESS: Tamil DME report generated at: {path_dme}")
    print(f"File size: {path_dme.stat().st_size / 1024:.2f} KB")

    print("Generating Tamil CNV Report...")
    path_cnv = generate_tamil_pdf(sample_result_cnv, 998, "Raman")
    print(f"SUCCESS: Tamil CNV report generated at: {path_cnv}")
    print(f"File size: {path_cnv.stat().st_size / 1024:.2f} KB")
except Exception as e:
    print(f"ERROR: Failed to generate Tamil report: {e}")
    import traceback
    traceback.print_exc()
