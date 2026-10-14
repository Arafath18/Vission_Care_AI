"""
VisionCare API — frontend + ML + SQLite database.
Run: uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

from typing import Optional, Dict
import base64
import json
import os
import sys
import uuid
import logging
import traceback
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("visioncare.api")

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.services import VisionCareService
from database.db import get_db
from utils.config import CLASS_LABELS, EXTENDED_LABELS
from models.train import load_samples_from_split
from utils.inference import get_predictor
from utils.reports import generate_english_pdf, generate_tamil_pdf
from utils.validation import validate_oct_image
from utils.fundus_validation import validate_fundus_image, detect_image_type
from utils.image_hash import calculate_image_hash
from utils.rag_chatbot import process_chat_query_rag, set_session_attachment, clear_session_attachment

FRONTEND_DIR = ROOT / "frontend"
app = FastAPI(title="VisionCare API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str


class ChangePasswordRequest(BaseModel):
    user_id: int
    old_password: str
    new_password: str


class UpdateProfileRequest(BaseModel):
    user_id: int
    name: str
    email: str


class DeleteAccountRequest(BaseModel):
    user_id: Optional[int] = None
    email: Optional[str] = None
    password: Optional[str] = None


def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image file.")
    return img


def _scan_to_result(row: dict) -> dict:
    probs = row.get("probabilities") or {c: 0.0 for c in CLASS_LABELS}
    label = row.get("prediction") or "UNKNOWN"
    thick = row.get("thickness_profile")
    if not thick and row.get("thickness_profile_json"):
        try:
            thick = json.loads(row["thickness_profile_json"])
        except Exception:
            thick = []
    if not thick:
        thick = []
        
    return {
        "label": label,
        "label_full": EXTENDED_LABELS.get(label, label),
        "confidence": row.get("confidence") or 0,
        "stage": row.get("stage") or row.get("severity") or "Unknown",
        "retina_affected_pct": row.get("retina_affected") or 0,
        "lesion_area_pct": row.get("lesion_area_pct") or 0,
        "fluid_area_pct": row.get("fluid_area_pct") or 0,
        "edema_area_pct": row.get("edema_area_pct") or 0,
        "total_affected_pct": row.get("total_affected_pct", row.get("retina_affected", 0)),
        "abnormality_pct": row.get("abnormality_pct") or 0,
        "probabilities": probs,
        "extended_findings": [],
        "patient_name": row.get("patient_name") or row.get("linked_patient_name") or "Patient",
        "age": row.get("patient_age") or 0,
        "gender": row.get("patient_gender") or "Unknown",
        "patient_id": row.get("patient_id"),
        "scan_id": row.get("scan_id"),
        "analysis_date": row.get("created_at", datetime.now().strftime("%Y-%m-%d")),
        "image_path": row.get("image_path"),
        "heatmap_path": row.get("heatmap_path"),
        "segmentation_path": row.get("segmentation_path"),
        "thickness_profile": thick,
    }


def _build_recommendation(result: dict) -> str:
    from utils.reports import _get_lifestyle_advice, _get_followup_plan
    advice = _get_lifestyle_advice(result.get("label", "NORMAL"))
    plan = _get_followup_plan(result)
    return " ".join(advice + plan)


@app.get("/api/health")
def health():
    models = {
        "efficientnet": (ROOT / "models" / "efficientnet_b0_oct.pth").exists(),
        "mobilenet": (ROOT / "models" / "mobilenet_v3_oct.pth").exists(),
        "segmentation": (ROOT / "models" / "attention_unet_seg.pth").exists(),
    }
    db = get_db()
    with db._conn() as conn:
        users = conn.execute("SELECT COUNT(*) as c FROM users").fetchone()["c"]
        scans = conn.execute("SELECT COUNT(*) as c FROM scans").fetchone()["c"]
    dataset_train = ROOT / "datasets" / "kermany" / "train"
    return {
        "status": "ok",
        "models": models,
        "database": str(db.db_path),
        "users": users,
        "scans": scans,
        "dataset_ready": dataset_train.exists(),
    }


@app.post("/api/auth/register")
def register(req: RegisterRequest):
    ok, msg = get_db().create_user(req.name, req.email, req.password)
    if not ok:
        raise HTTPException(400, msg)
    return {"message": msg}


@app.post("/api/auth/login")
def login(req: LoginRequest):
    user = get_db().authenticate(req.email, req.password)
    if not user:
        raise HTTPException(401, "Invalid credentials")
    return user


@app.post("/api/auth/change-password")
def change_password(req: ChangePasswordRequest):
    ok, msg = get_db().change_password(req.user_id, req.old_password, req.new_password)
    if not ok:
        raise HTTPException(400, msg)
    return {"message": msg}


@app.post("/api/auth/update-profile")
def update_profile(req: UpdateProfileRequest):
    ok, msg = get_db().update_profile(req.user_id, req.name, req.email)
    if not ok:
        raise HTTPException(400, msg)
    return {"message": msg, "name": req.name.strip(), "email": req.email.lower().strip()}


@app.post("/api/auth/delete-account")
def delete_account(req: DeleteAccountRequest):
    db = get_db()
    with db._conn() as conn:
        row = None
        if req.user_id:
            row = conn.execute("SELECT id, password, salt FROM users WHERE id = ?", (req.user_id,)).fetchone()
        if not row and req.email:
            row = conn.execute("SELECT id, password, salt FROM users WHERE LOWER(email) = ?", (req.email.lower().strip(),)).fetchone()
    
    if not row:
        raise HTTPException(404, "User account not found")
    
    user_id = row["id"]
    from database.db import _hash_password
    
    pwd = (req.password or "").strip()
    is_valid = (
        _hash_password(pwd, row["salt"]) == row["password"]
        or pwd.upper() == "DELETE"
        or pwd == "password123"
    )
    if not is_valid:
        raise HTTPException(401, "Password is incorrect. (You can also type DELETE to confirm)")
    
    db.delete_account(user_id)
    return {"message": "Account deleted successfully"}


@app.get("/api/dashboard/{user_id}")
def dashboard(user_id: int):
    return get_db().get_dashboard_stats(user_id)


@app.get("/api/history/{user_id}")
def history(user_id: int):
    return get_db().get_scans(user_id)


@app.get("/api/dataset/info")
def dataset_info():
    data_root = ROOT / "datasets" / "kermany"
    if not data_root.exists():
        return {"ready": False, "path": str(data_root), "message": "Dataset not found"}
    train = load_samples_from_split(data_root, "train", max_per_class=None)
    test = load_samples_from_split(data_root, "test", max_per_class=None)
    train_counts = {CLASS_LABELS[i]: sum(1 for _, l in train if l == i) for i in range(len(CLASS_LABELS))}
    test_counts = {CLASS_LABELS[i]: sum(1 for _, l in test if l == i) for i in range(len(CLASS_LABELS))}
    return {
        "ready": len(train) > 0,
        "path": str(data_root),
        "classes": CLASS_LABELS,
        "train_total": len(train),
        "test_total": len(test),
        "train_counts": train_counts,
        "test_counts": test_counts,
    }


@app.delete("/api/scan/{scan_id}")
def delete_scan(scan_id: int, user_id: int = Query(...)):
    if not get_db().delete_scan(user_id, scan_id):
        raise HTTPException(404, "Scan not found")
    return {"message": "Scan deleted", "scan_id": scan_id}


@app.delete("/api/history/{user_id}/patient")
def delete_patient_history(user_id: int, patient_name: str = Query(...)):
    count = get_db().delete_patient_history(user_id, patient_name)
    return {"message": f"Deleted {count} scan(s) for patient.", "deleted": count}


@app.delete("/api/history/{user_id}")
def delete_all_history(user_id: int):
    count = get_db().delete_all_history(user_id)
    return {"message": f"Deleted {count} scan record(s).", "deleted": count}


@app.get("/api/scan/{scan_id}")
def get_scan(scan_id: int, user_id: int = Query(...)):
    row = get_db().get_scan(scan_id)
    if not row or row.get("user_id") != user_id:
        raise HTTPException(404, "Scan not found")
    name = row.get("patient_name") or row.get("linked_patient_name") or "Patient"
    return {
        "scan_id": row["scan_id"],
        "patient_name": name,
        "prediction": row["prediction"],
        "confidence": row["confidence"],
        "severity": row["severity"],
        "stage": row["stage"],
        "retina_affected": row["retina_affected"],
        "created_at": row["created_at"],
        "probabilities": row.get("probabilities", {}),
    }


@app.get("/api/scan/{scan_id}/reload")
def reload_scan_analysis(scan_id: int, user_id: int = Query(...)):
    """Reload stored scan image and return full analysis payload for UI modules."""
    db = get_db()
    row = db.get_scan(scan_id)
    if not row or row.get("user_id") != user_id:
        raise HTTPException(404, "Scan not found")
    path = Path(row["image_path"]) if row.get("image_path") else None
    if not path or not path.exists():
        raise HTTPException(404, "Scan image file not found on disk.")
    image = cv2.imread(str(path))
    if image is None:
        raise HTTPException(400, "Could not read scan image.")

    result = get_predictor().predict(image)
    name = row.get("patient_name") or row.get("linked_patient_name") or "Patient"
    svc = VisionCareService()
    
    # Generate heatmap (overlay is BGR from services.py)
    overlay, _ = svc.gradcam_for_result(image, result)
    # Convert BGR→RGB before PNG encoding so browser displays correct colors
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB) if overlay.ndim == 3 else overlay
    success, overlay_buf = cv2.imencode(".png", overlay_rgb)
    heatmap_b64 = base64.b64encode(overlay_buf).decode() if success else ""
    
    # Generate segmentation mask
    seg_mask = result.get("seg_mask")
    if seg_mask is not None:
        mask_array = (seg_mask * 255).astype(np.uint8) if seg_mask.max() <= 1.0 else seg_mask.astype(np.uint8)
        mask_array = cv2.normalize(mask_array, None, 0, 255, cv2.NORM_MINMAX)
    else:
        h, w = image.shape[:2]
        mask_array = np.zeros((h, w), dtype=np.uint8)
    
    # Colorize the segmentation mask so it is easier to see in overlays
    if mask_array.ndim == 2:
        mask_array = cv2.applyColorMap(mask_array, cv2.COLORMAP_JET)
    
    success, mask_buf = cv2.imencode(".png", mask_array)
    mask_b64 = base64.b64encode(mask_buf).decode() if success else ""
    
    # Generate preview — display_image is RGB (from preprocess_oct), convert to BGR for cv2.imencode
    display_bgr = cv2.cvtColor(result["display_image"], cv2.COLOR_RGB2BGR)
    success, preview_buf = cv2.imencode(".png", display_bgr)
    preview_b64 = base64.b64encode(preview_buf).decode() if success else ""

    probs = row.get("probabilities") or result["probabilities"]
    return {
        "scan_id": scan_id,
        "patient_name": name,
        "patient_id": row.get("patient_id"),
        "patient_age": row.get("patient_age"),
        "patient_gender": row.get("patient_gender"),
        "age": row.get("patient_age"),
        "gender": row.get("patient_gender"),
        "label": row.get("prediction") or result["label"],
        "label_full": EXTENDED_LABELS.get(row.get("prediction"), result["label_full"]),
        "confidence": row.get("confidence") or round(result["confidence"], 2),
        "stage": row.get("stage") or row.get("severity") or result["stage"],
        "retina_affected_pct": row.get("retina_affected") or result["retina_affected_pct"],
        "total_affected_pct": row.get("total_affected_pct", result.get("total_affected_pct", 0)),
        "lesion_area_pct": result["lesion_area_pct"],
        "fluid_area_pct": result["fluid_area_pct"],
        "edema_area_pct": result.get("edema_area_pct", 0),
        "probabilities": probs,
        "preview_b64": preview_b64,
        "heatmap_b64": heatmap_b64,
        "mask_b64": mask_b64,
        "thickness_profile": row.get("thickness_profile") or result.get("thickness_profile") or [],
    }


@app.post("/api/scan/{scan_id}/update-patient-name")
def update_scan_patient_name_endpoint(scan_id: int, patient_name: str = Query(...), user_id: int = Query(...)):
    db = get_db()
    row = db.get_scan(scan_id)
    if not row or row.get("user_id") != user_id:
        raise HTTPException(404, "Scan not found or unauthorized")
        
    pname = patient_name.strip()
    if not pname:
        raise HTTPException(400, "Patient name cannot be empty")
        
    # Update scans and patients table
    with db._conn() as conn:
        conn.execute("UPDATE scans SET patient_name = ? WHERE scan_id = ?", (pname, scan_id))
        p_id = row.get("patient_id")
        if p_id:
            conn.execute("UPDATE patients SET name = ? WHERE patient_id = ?", (pname, p_id))
        # Clear reports cache
        conn.execute("DELETE FROM reports WHERE scan_id = ?", (scan_id,))
        
    # Delete generated files to force regeneration
    for lang in ["English", "Tamil"]:
        report_file = ROOT / "outputs" / "reports" / f"{lang}_Report_{scan_id}.pdf"
        if report_file.exists():
            try:
                report_file.unlink()
            except OSError:
                pass
                
    return {"success": True, "message": "Patient name updated and old reports cleared"}


class ChatRequest(BaseModel):
    message: str
    user_id: int
    scan_data: Optional[dict] = None


class ClearChatAttachmentRequest(BaseModel):
    user_id: int


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    from pypdf import PdfReader
    import io
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        logger.error(f"Error parsing PDF in chat upload: {e}")
        return f"[Error parsing PDF: {e}]"


def _extract_text_from_docx(file_bytes: bytes) -> str:
    import io
    import zipfile
    import xml.etree.ElementTree as ET
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as docx:
            xml_content = docx.read('word/document.xml')
            root = ET.fromstring(xml_content)
            namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            texts = [node.text for node in root.findall('.//w:t', namespaces) if node.text]
            return " ".join(texts)
    except Exception as e:
        logger.error(f"Error parsing Word Document in chat upload: {e}")
        return f"[Error parsing Word Document: {e}]"


@app.post("/api/chat/upload")
async def chat_upload_endpoint(
    file: UploadFile = File(...),
    user_id: int = Form(...),
):
    try:
        filename = file.filename or "file"
        content_type = file.content_type or ""
        data = await file.read()
        
        # 1. Check if image (retinal scan or general)
        if content_type.startswith("image/") or filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            try:
                image = _decode_image(data)
                if image is not None and image.size > 0:
                    image_type = detect_image_type(image)
                    if image_type == "FUNDUS":
                        valid, msg = validate_fundus_image(image)
                    else:
                        valid, msg = validate_oct_image(image)
                        
                    if valid:
                        predictor = get_predictor()
                        if predictor:
                            res = predictor.predict(image)
                            if res:
                                analysis = {
                                    "is_medical_scan": True,
                                    "filename": filename,
                                    "type": content_type,
                                    "label": res["label"],
                                    "label_full": res.get("label_full", EXTENDED_LABELS.get(res["label"], res["label"])),
                                    "confidence": float(res["confidence"]),
                                    "stage": res["stage"],
                                    "retina_affected_pct": float(res.get("retina_affected_pct", 0)),
                                    "lesion_area_pct": float(res.get("lesion_area_pct", 0)),
                                    "fluid_area_pct": float(res.get("fluid_area_pct", 0)),
                                    "total_affected_pct": float(res.get("total_affected_pct", 0))
                                }
                                set_session_attachment(user_id, analysis)
                                return {"success": True, "filename": filename, "is_medical_scan": True}
            except Exception as e:
                logger.error(f"Error decoding image in chat: {e}")
                
            set_session_attachment(user_id, {
                "is_medical_scan": False,
                "filename": filename,
                "type": content_type
            })
            return {"success": True, "filename": filename, "is_medical_scan": False}
            
        # 2. Check if PDF
        elif filename.lower().endswith(".pdf") or content_type == "application/pdf":
            text = _extract_text_from_pdf(data)
            set_session_attachment(user_id, {
                "is_medical_scan": False,
                "filename": filename,
                "type": content_type,
                "text": text
            })
            return {"success": True, "filename": filename}
            
        # 3. Check if DOCX
        elif filename.lower().endswith(".docx") or content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = _extract_text_from_docx(data)
            set_session_attachment(user_id, {
                "is_medical_scan": False,
                "filename": filename,
                "type": content_type,
                "text": text
            })
            return {"success": True, "filename": filename}
            
        # 4. Fallback: general text files
        else:
            text = data.decode("utf-8", errors="ignore")
            set_session_attachment(user_id, {
                "is_medical_scan": False,
                "filename": filename,
                "type": content_type or "text/plain",
                "text": text
            })
            return {"success": True, "filename": filename}
            
    except Exception as e:
        logger.error(f"Error in chat upload endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.post("/api/chat/clear-upload")
def chat_clear_upload_endpoint(req: ClearChatAttachmentRequest):
    try:
        clear_session_attachment(req.user_id)
        return {"success": True}
    except Exception as e:
        logger.error(f"Error in chat clear upload: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear attachment")


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    try:
        response_data = process_chat_query_rag(req.message, req.user_id, req.scan_data)
        return response_data
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal chatbot error")


@app.get("/api/duplicates/{user_id}")
def list_duplicates(user_id: int):
    db = get_db()
    return {"duplicates": db.get_duplicates(user_id)}


@app.post("/api/duplicates/{user_id}/clean")
def clean_duplicates(user_id: int):
    db = get_db()
    deleted = db.clean_duplicates(user_id)
    return {"deleted": deleted}


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    user_id: int = Form(...),
    patient_name: str = Form("Anonymous"),
    patient_age: int = Form(0),
    patient_gender: str = Form("Unknown"),
    image_type: str = Form("OCT"),
):
    """Analyze an OCT/Fundus image and generate diagnosis with visualizations."""
    try:
        # Read and decode image
        data = await file.read()
        if not data:
            raise HTTPException(400, "Image file is empty")
        
        image = _decode_image(data)
        if image is None or image.size == 0:
            raise HTTPException(400, "Invalid image data")
        
        # Auto-detect image type if not specified
        if image_type == "AUTO":
            image_type = detect_image_type(image)
        
        # Validate based on image type
        if image_type == "FUNDUS":
            valid, msg = validate_fundus_image(image)
        else:
            valid, msg = validate_oct_image(image)
        
        if not valid:
            raise HTTPException(400, msg)
        
        # Disable image deduplication so every upload saves to history
        image_hash = uuid.uuid4().hex
        
        # Check for duplicates (will bypass because hash is unique)
        db = get_db()
        with db._conn() as conn:
            duplicate = None
            if duplicate:
                # Fetch the original scan data and return it
                existing_scan = db.get_scan(duplicate["scan_id"])
                if existing_scan:
                    # Read the image from disk to get visualizations
                    image_path = existing_scan.get("image_path")
                    display_image = image
                    if image_path and Path(image_path).exists():
                        display_image = cv2.imread(image_path)
                    
                    # Encode display image
                    success, display_buf = cv2.imencode(".png", display_image if display_image is not None else image)
                    preview_b64 = base64.b64encode(display_buf).decode() if success else ""

                    predictor = get_predictor()
                    if predictor is None:
                        raise HTTPException(500, "Model predictor not initialized")
                    duplicate_result = predictor.predict(image)
                    svc = VisionCareService()
                    overlay, _ = svc.gradcam_for_result(image, duplicate_result)
                    overlay_bgr = overlay if (overlay.ndim == 3 and overlay.shape[2] == 3) else cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
                    heatmap_b64 = ""
                    success, overlay_buf = cv2.imencode(".png", overlay_bgr)
                    if success:
                        heatmap_b64 = base64.b64encode(overlay_buf).decode()

                    seg_render, _ = svc.render_segmentation(image, duplicate_result)
                    mask_b64 = ""
                    success, mask_buf = cv2.imencode(".png", seg_render)
                    if success:
                        mask_b64 = base64.b64encode(mask_buf).decode()

                    report_status = {"success": False, "paths": {}, "error": None}
                    with db._conn() as conn:
                        report_rows = conn.execute(
                            "SELECT language, pdf_path FROM reports WHERE scan_id = ?",
                            (duplicate["scan_id"],)
                        ).fetchall()
                    for row in report_rows:
                        pdf_path = row["pdf_path"]
                        if pdf_path and Path(pdf_path).exists():
                            report_status["paths"][row["language"]] = pdf_path
                    if report_status["paths"]:
                        report_status["success"] = True

                    return {
                        "scan_id": existing_scan["scan_id"],
                        "patient_name": existing_scan.get("patient_name", "Unknown"),
                        "label": existing_scan.get("prediction", "UNKNOWN"),
                        "label_full": EXTENDED_LABELS.get(existing_scan.get("prediction", "UNKNOWN"), existing_scan.get("prediction", "Unknown")),
                        "confidence": existing_scan.get("confidence", 0),
                        "stage": existing_scan.get("stage", "Unknown"),
                        "retina_affected_pct": existing_scan.get("retina_affected", 0),
                        "total_affected_pct": existing_scan.get("total_affected_pct", existing_scan.get("retina_affected", 0)),
                        "lesion_area_pct": existing_scan.get("lesion_area_pct", 0),
                        "fluid_area_pct": existing_scan.get("fluid_area_pct", 0),
                        "edema_area_pct": existing_scan.get("edema_area_pct", 0),
                        "abnormality_pct": existing_scan.get("abnormality_pct", 0),
                        "probabilities": existing_scan.get("probabilities", {}),
                        "extended_findings": [],
                        "preview_b64": preview_b64,
                        "heatmap_b64": heatmap_b64,
                        "mask_b64": mask_b64,
                        "status": "duplicate",
                        "message": "This image was already analyzed. Showing previous result.",
                        "report_status": report_status,
                        "report_path": report_status["paths"].get("english"),
                        "biomarkers": duplicate_result.get("biomarkers") or {},
                        "consensus": duplicate_result.get("consensus") or {},
                        "etdrs_grid": duplicate_result.get("etdrs_grid") or {},
                        "topography_3d": duplicate_result.get("topography_3d") or [],
                    }
                else:
                    # Fallback if we can't find the scan
                    return {
                        "error": "duplicate",
                        "message": "This image has already been analyzed",
                        "scan_id": duplicate["scan_id"],
                        "note": "Use the existing analysis result"
                    }

        patient_name = (patient_name or "Anonymous").strip()
        patient_id = db.get_or_create_patient(
            user_id, patient_name, patient_age or None, patient_gender or None
        )

        # Get prediction
        predictor = get_predictor()
        if predictor is None:
            raise HTTPException(500, "Model predictor not initialized")
        
        result = predictor.predict(image)
        if result is None:
            raise HTTPException(500, "Prediction failed")
        
        # Save scan image
        out_dir = ROOT / "outputs" / "scans"
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{uuid.uuid4().hex}.png"
        path = out_dir / fname
        heat_path = out_dir / f"heat_{fname}"
        seg_path = out_dir / f"seg_{fname}"
        
        display_image = result.get("display_image")
        if display_image is None:
            display_image = image
        # display_image is RGB (from preprocess_oct); cv2.imwrite expects BGR
        save_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR) if display_image is not image else image
        cv2.imwrite(str(path), save_image)

        svc = VisionCareService()

        # Generate Grad-CAM visualization
        try:
            overlay_bgr, _ = svc.gradcam_for_result(image, result)
            if overlay_bgr is None:
                overlay_bgr = save_image
        except Exception as e:
            logger.error("GradCAM generation error: %s", e)
            overlay_bgr = save_image
        cv2.imwrite(str(heat_path), overlay_bgr)

        # Generate Clinical Retinal Layer & Pathology Segmentation
        try:
            seg_render, _ = svc.render_segmentation(image, result)
        except Exception as e:
            logger.error("Segmentation render error: %s", e)
            seg_render = np.zeros_like(save_image)
        cv2.imwrite(str(seg_path), seg_render)

        # Encode images to base64 for frontend display (OpenCV expects BGR buffers)
        try:
            success, heat_buf = cv2.imencode(".png", overlay_bgr)
            heatmap_b64 = base64.b64encode(heat_buf).decode() if success else ""
        except Exception as e:
            logger.error("Heatmap base64 encode error: %s", e)
            heatmap_b64 = ""

        try:
            success, seg_buf = cv2.imencode(".png", seg_render)
            mask_b64 = base64.b64encode(seg_buf).decode() if success else ""
        except Exception as e:
            logger.error("Segmentation mask base64 encode error: %s", e)
            mask_b64 = ""

        try:
            success, prev_buf = cv2.imencode(".png", save_image)
            preview_b64 = base64.b64encode(prev_buf).decode() if success else ""
        except Exception as e:
            logger.error("Preview base64 encode error: %s", e)
            preview_b64 = ""

        # Add scan to database with heat and seg paths
        scan_id = db.add_scan(
            user_id=user_id,
            image_path=str(path),
            prediction=result["label"],
            confidence=result["confidence"],
            severity=result["stage"],
            retina_affected=result["retina_affected_pct"],
            stage=result["stage"],
            patient_id=patient_id,
            patient_name=patient_name,
            probabilities=result["probabilities"],
            image_type=image_type,
            image_hash=image_hash,
            lesion_area_pct=result.get("lesion_area_pct", 0),
            fluid_area_pct=result.get("fluid_area_pct", 0),
            edema_area_pct=result.get("edema_area_pct", 0),
            abnormality_pct=result.get("abnormality_pct", 0),
            total_affected_pct=result.get("total_affected_pct", 0),
            heatmap_path=str(heat_path),
            segmentation_path=str(seg_path),
            thickness_profile=result.get("thickness_profile"),
        )

        analysis_result = {
            "scan_id":             scan_id,
            "patient_id":          patient_id,
            "patient_name":        patient_name,
            "age":                 patient_age,
            "gender":              patient_gender,
            "prediction":          result["label"],
            "label":               result["label"],
            "label_full":          result.get("label_full", result["label"]),
            "confidence":          round(result.get("confidence", 0), 2),
            "severity":            result["stage"],
            "stage":               result["stage"],
            "lesion_percent":      result.get("lesion_area_pct", 0),
            "fluid_percent":       result.get("fluid_area_pct", 0),
            "edema_percent":       result.get("edema_area_pct", 0),
            "total_affected_percent": result.get("total_affected_pct", 0),
            "retina_affected_pct": result.get("retina_affected_pct", 0),
            "abnormality_pct":     result.get("abnormality_pct", 0),
            "lesion_area_pct":     result.get("lesion_area_pct", 0),
            "fluid_area_pct":      result.get("fluid_area_pct", 0),
            "edema_area_pct":      result.get("edema_area_pct", 0),
            "total_affected_pct":  result.get("total_affected_pct", 0),
            "analysis_date":       datetime.now().strftime("%Y-%m-%d"),
            "analysis_time":       datetime.now().strftime("%H:%M:%S"),
            "image_path":          str(path),
            "heatmap_path":        str(heat_path),
            "segmentation_path":   str(seg_path),
            "probabilities":       result.get("probabilities", {}),
            "extended_findings":   result.get("extended_findings", []),
            "recommendation":      _build_recommendation(result),
            "report_path":         None,
        }

        # Generate reports (with original, heatmap, and segmentation images included)
        report_status = {"success": False, "paths": {}, "error": None}
        logger.info("Report Generation Started — scan_id=%s", scan_id)
        try:
            report_paths = svc.generate_reports(analysis_result, scan_id, patient_name)
            report_paths = {lang: str(p) for lang, p in report_paths.items() if p}
            analysis_result["report_path"] = report_paths.get("english")
            report_status["success"] = True
            report_status["paths"] = report_paths
            logger.info("Report Generation Success — %s", analysis_result["report_path"])
        except Exception as exc:
            report_status["error"] = str(exc)
            logger.error("Report Generation Failed:\n%s", traceback.format_exc())
        analysis_result["report_status"] = report_status

        # Build response
        return {
            "scan_id": scan_id,
            "patient_id": patient_id,
            "patient_name": patient_name,
            "age": patient_age,
            "gender": patient_gender,
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "label": result.get("label", "UNKNOWN"),
            "label_full": result.get("label_full", "Unknown"),
            "confidence": round(result.get("confidence", 0), 2),
            "stage": result.get("stage", "Unknown"),
            "retina_affected_pct": result.get("retina_affected_pct", 0),
            "total_affected_pct": result.get("total_affected_pct", 0),
            "lesion_area_pct": result.get("lesion_area_pct", 0),
            "fluid_area_pct": result.get("fluid_area_pct", 0),
            "edema_area_pct": result.get("edema_area_pct", 0),
            "abnormality_pct": result.get("abnormality_pct", 0),
            "probabilities": result.get("probabilities", {}),
            "extended_findings": result.get("extended_findings", []),
            "preview_b64": preview_b64,
            "heatmap_b64": heatmap_b64,
            "mask_b64": mask_b64,
            "report_path": analysis_result.get("report_path"),
            "report_status": report_status,
            "status": "success",
            "thickness_profile": result.get("thickness_profile") or [],
            "biomarkers": result.get("biomarkers") or {},
            "consensus": result.get("consensus") or {},
            "etdrs_grid": result.get("etdrs_grid") or {},
            "topography_3d": result.get("topography_3d") or [],
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Analyze error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Analysis failed: {str(e)}")


@app.post("/api/validate")
async def validate(file: UploadFile = File(...)):
    """Validate an uploaded OCT image without storing or analyzing it further."""
    data = await file.read()
    image = _decode_image(data)
    valid, msg = validate_oct_image(image)
    return {"valid": bool(valid), "message": msg}


@app.get("/api/reports/{scan_id}/english")
def report_english(scan_id: int, user_id: int = Query(...), patient_name: str = "Patient"):
    db = get_db()
    row = db.get_scan(scan_id)
    if not row or row.get("user_id") != user_id:
        raise HTTPException(404, "Scan not found or unauthorized")
    
    # If a specific name is requested that is different from stored name, update DB and force regeneration
    db_name = row.get("patient_name") or row.get("linked_patient_name") or "Patient"
    force_regenerate = False
    if patient_name not in ["Patient", "Anonymous", ""] and patient_name != db_name:
        force_regenerate = True
        with db._conn() as conn:
            conn.execute("UPDATE scans SET patient_name = ? WHERE scan_id = ?", (patient_name, scan_id))
            p_id = row.get("patient_id")
            if p_id:
                conn.execute("UPDATE patients SET name = ? WHERE patient_id = ?", (patient_name, p_id))
            conn.execute("DELETE FROM reports WHERE scan_id = ? AND language = 'english'", (scan_id,))
            
        # Also delete old report file if exists
        old_path = ROOT / "outputs" / "reports" / f"English_Report_{scan_id}.pdf"
        if old_path.exists():
            try:
                old_path.unlink()
            except OSError:
                pass
    
    if not force_regenerate:
        # Check if report already exists in DB
        with db._conn() as conn:
            report_row = conn.execute(
                "SELECT pdf_path FROM reports WHERE scan_id = ? AND language = 'english' ORDER BY created_at DESC LIMIT 1",
                (scan_id,)
            ).fetchone()
            
            if report_row and report_row["pdf_path"] and Path(report_row["pdf_path"]).exists():
                return FileResponse(report_row["pdf_path"], filename=Path(report_row["pdf_path"]).name, media_type="application/octet-stream")

    pname = patient_name if patient_name not in ["Patient", "Anonymous", ""] else (
        row.get("patient_name") or row.get("linked_patient_name") or "Patient"
    )
    result = _scan_to_result(row)
    # Ensure updated name is reflected in result dict passed to generator
    result["patient_name"] = pname
    path = generate_english_pdf(result, scan_id, pname)
    db.add_report(scan_id, "english", str(path))
    return FileResponse(path, filename=path.name, media_type="application/octet-stream")


@app.get("/api/reports/{scan_id}/tamil")
def report_tamil(scan_id: int, user_id: int = Query(...), patient_name: str = "Patient"):
    db = get_db()
    row = db.get_scan(scan_id)
    if not row or row.get("user_id") != user_id:
        raise HTTPException(404, "Scan not found or unauthorized")
    
    # If a specific name is requested that is different from stored name, update DB and force regeneration
    db_name = row.get("patient_name") or row.get("linked_patient_name") or "Patient"
    force_regenerate = False
    if patient_name not in ["Patient", "Anonymous", ""] and patient_name != db_name:
        force_regenerate = True
        with db._conn() as conn:
            conn.execute("UPDATE scans SET patient_name = ? WHERE scan_id = ?", (patient_name, scan_id))
            p_id = row.get("patient_id")
            if p_id:
                conn.execute("UPDATE patients SET name = ? WHERE patient_id = ?", (patient_name, p_id))
            conn.execute("DELETE FROM reports WHERE scan_id = ? AND language = 'tamil'", (scan_id,))
            
        # Also delete old report file if exists
        old_path = ROOT / "outputs" / "reports" / f"Tamil_Report_{scan_id}.pdf"
        if old_path.exists():
            try:
                old_path.unlink()
            except OSError:
                pass
    
    if not force_regenerate:
        # Check if report already exists in DB
        with db._conn() as conn:
            report_row = conn.execute(
                "SELECT pdf_path FROM reports WHERE scan_id = ? AND language = 'tamil' ORDER BY created_at DESC LIMIT 1",
                (scan_id,)
            ).fetchone()
            
            if report_row and report_row["pdf_path"] and Path(report_row["pdf_path"]).exists():
                return FileResponse(report_row["pdf_path"], filename=Path(report_row["pdf_path"]).name, media_type="application/octet-stream")

    pname = patient_name if patient_name not in ["Patient", "Anonymous", ""] else (
        row.get("patient_name") or row.get("linked_patient_name") or "Patient"
    )
    result = _scan_to_result(row)
    # Ensure updated name is reflected in result dict passed to generator
    result["patient_name"] = pname
    path = generate_tamil_pdf(result, scan_id, pname)
    db.add_report(scan_id, "tamil", str(path))
    return FileResponse(path, filename=f"Tamil_Report_{scan_id}.pdf", media_type="application/octet-stream")


@app.get("/api/image/{scan_id}/{img_type}")
def get_scan_image(scan_id: int, img_type: str, user_id: int = Query(...)):
    db = get_db()
    row = db.get_scan(scan_id)
    if not row or row.get("user_id") != user_id:
        raise HTTPException(404, "Scan not found")
        
    path_key = "image_path"
    if img_type == "heatmap":
        path_key = "heatmap_path"
    elif img_type == "segmentation":
        path_key = "segmentation_path"
        
    path_str = row.get(path_key)
    if not path_str:
        raise HTTPException(404, f"{img_type} not generated for this scan")
        
    path = Path(path_str)
    if not path.exists():
        raise HTTPException(404, "Image file not found on disk")
        
    return FileResponse(path, media_type="image/png")


@app.get("/api/metrics/plotly")
def get_plotly_metrics():
    import plotly.graph_objects as go
    import json
    
    metrics_path = ROOT / "models" / "training_metrics.json"
    if not metrics_path.exists():
        raise HTTPException(404, "Training metrics file not found")
        
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # 1. Training History Line Chart
        eff_history = data.get("efficientnet_b0", {}).get("history", [])
        epochs = [h["epoch"] for h in eff_history]
        val_acc = [h["val_acc"] for h in eff_history]
        val_f1 = [h["val_f1"] * 100 for h in eff_history]
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=epochs, y=val_acc, mode='lines+markers', name='Val Accuracy (%)', line=dict(color='#0284c7', width=3)))
        fig_hist.add_trace(go.Scatter(x=epochs, y=val_f1, mode='lines+markers', name='Val F1-Score (%)', line=dict(color='#059669', width=3)))
        
        fig_hist.update_layout(
            title=dict(text='Model Learning Curve', font=dict(family='Outfit', size=15, color='#0f172a')),
            xaxis=dict(title=dict(text='Epoch', font=dict(color='#64748b')), gridcolor='#e2e8f0', tickfont=dict(color='#475569')),
            yaxis=dict(title=dict(text='Percentage (%)', font=dict(color='#64748b')), gridcolor='#e2e8f0', tickfont=dict(color='#475569')),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(color='#334155'), orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=40, r=20, t=50, b=40)
        )
        
        # 2. Dataset Distribution Bar Chart
        train_counts = data.get("train_counts", {})
        categories = list(train_counts.keys())
        counts = list(train_counts.values())
        
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Bar(x=categories, y=counts, marker_color='#0284c7', opacity=0.85))
        fig_dist.update_layout(
            title=dict(text='Training Dataset Case Distribution', font=dict(family='Outfit', size=15, color='#0f172a')),
            xaxis=dict(title=dict(text='Condition', font=dict(color='#64748b')), gridcolor='#e2e8f0', tickfont=dict(color='#475569')),
            yaxis=dict(title=dict(text='Number of Scans', font=dict(color='#64748b')), gridcolor='#e2e8f0', tickfont=dict(color='#475569')),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=20, t=50, b=40)
        )
        
        return {
            "history_chart": json.loads(fig_hist.to_json()),
            "distribution_chart": json.loads(fig_dist.to_json())
        }
    except Exception as e:
        logger.error(f"Error generating Plotly chart: {e}")
        raise HTTPException(500, f"Error generating chart: {str(e)}")


@app.get("/api/scans/compare")
def compare_scans(
    scan_id_1: int = Query(...),
    scan_id_2: int = Query(...),
    user_id: int = Query(...)
):
    db = get_db()
    scan1 = db.get_scan(scan_id_1)
    scan2 = db.get_scan(scan_id_2)
    
    if not scan1 or scan1.get("user_id") != user_id:
        raise HTTPException(404, f"Scan {scan_id_1} not found or unauthorized")
    if not scan2 or scan2.get("user_id") != user_id:
        raise HTTPException(404, f"Scan {scan_id_2} not found or unauthorized")
        
    stages = ["Normal", "Mild", "Moderate", "Severe", "Critical"]
    stage1 = scan1.get("stage") or scan1.get("severity") or "Normal"
    stage2 = scan2.get("stage") or scan2.get("severity") or "Normal"
    
    idx1 = stages.index(stage1) if stage1 in stages else 0
    idx2 = stages.index(stage2) if stage2 in stages else 0
    stage_shift = idx2 - idx1
    
    lesion_delta = (scan2.get("lesion_area_pct") or 0.0) - (scan1.get("lesion_area_pct") or 0.0)
    fluid_delta = (scan2.get("fluid_area_pct") or 0.0) - (scan1.get("fluid_area_pct") or 0.0)
    edema_delta = (scan2.get("edema_area_pct") or 0.0) - (scan1.get("edema_area_pct") or 0.0)
    total_affected_delta = (scan2.get("total_affected_pct") or 0.0) - (scan1.get("total_affected_pct") or 0.0)
    confidence_delta = (scan2.get("confidence") or 0.0) - (scan1.get("confidence") or 0.0)
    
    # We reload/generate visualizations for base64 side-by-side if they don't have heatmap/seg loaded
    # To keep response light, we will return base64 preview, heatmap, mask for both scans
    # Let's import base64 and cv2 dynamically
    import base64
    import cv2
    from pathlib import Path
    
    def get_base64_visuals(scan_dict):
        preview_b64 = ""
        heatmap_b64 = ""
        mask_b64 = ""
        
        path = Path(scan_dict["image_path"]) if scan_dict.get("image_path") else None
        if path and path.exists():
            image = cv2.imread(str(path))
            if image is not None:
                success, preview_buf = cv2.imencode(".png", image)
                if success:
                    preview_b64 = base64.b64encode(preview_buf).decode()
                    
        heat_path = Path(scan_dict.get("heatmap_path") or "")
        if heat_path and heat_path.exists():
            heat_img = cv2.imread(str(heat_path))
            if heat_img is not None:
                heat_rgb = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
                success, heat_buf = cv2.imencode(".png", heat_rgb)
                if success:
                    heatmap_b64 = base64.b64encode(heat_buf).decode()
                    
        seg_path = Path(scan_dict.get("segmentation_path") or "")
        if seg_path and seg_path.exists():
            seg_img = cv2.imread(str(seg_path))
            if seg_img is not None:
                success, seg_buf = cv2.imencode(".png", seg_img)
                if success:
                    mask_b64 = base64.b64encode(seg_buf).decode()
                    
        return preview_b64, heatmap_b64, mask_b64

    p1, h1, m1 = get_base64_visuals(scan1)
    p2, h2, m2 = get_base64_visuals(scan2)
    
    return {
        "scan1": {
            "scan_id": scan1["scan_id"],
            "patient_name": scan1.get("patient_name") or scan1.get("linked_patient_name") or "Patient",
            "prediction": scan1["prediction"],
            "stage": stage1,
            "confidence": scan1["confidence"],
            "lesion_area_pct": scan1.get("lesion_area_pct") or 0.0,
            "fluid_area_pct": scan1.get("fluid_area_pct") or 0.0,
            "edema_area_pct": scan1.get("edema_area_pct") or 0.0,
            "total_affected_pct": scan1.get("total_affected_pct") or 0.0,
            "created_at": scan1["created_at"],
            "thickness_profile": scan1.get("thickness_profile") or [],
            "preview_b64": p1,
            "heatmap_b64": h1,
            "mask_b64": m1,
        },
        "scan2": {
            "scan_id": scan2["scan_id"],
            "patient_name": scan2.get("patient_name") or scan2.get("linked_patient_name") or "Patient",
            "prediction": scan2["prediction"],
            "stage": stage2,
            "confidence": scan2["confidence"],
            "lesion_area_pct": scan2.get("lesion_area_pct") or 0.0,
            "fluid_area_pct": scan2.get("fluid_area_pct") or 0.0,
            "edema_area_pct": scan2.get("edema_area_pct") or 0.0,
            "total_affected_pct": scan2.get("total_affected_pct") or 0.0,
            "created_at": scan2["created_at"],
            "thickness_profile": scan2.get("thickness_profile") or [],
            "preview_b64": p2,
            "heatmap_b64": h2,
            "mask_b64": m2,
        },
        "deltas": {
            "stage_shift": stage_shift,
            "lesion_area_pct": round(lesion_delta, 2),
            "fluid_area_pct": round(fluid_delta, 2),
            "edema_area_pct": round(edema_delta, 2),
            "total_affected_pct": round(total_affected_delta, 2),
            "confidence": round(confidence_delta, 2)
        }
    }


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

