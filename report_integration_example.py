#!/usr/bin/env python3
"""
Integration example — Doctor-style PDF report generation with VisionCare API.

This module demonstrates how to integrate the doctor-style report generator
with the VisionCare FastAPI backend.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from utils.reports import generate_doctor_style_pdf


# ============================================================================
# Pydantic Models
# ============================================================================

class AnalysisResult(BaseModel):
    """Analysis result from the prediction model."""
    label: str
    label_full: str
    confidence: float
    stage: str
    retina_affected_pct: int
    lesion_area_pct: int
    fluid_area_pct: int
    abnormality_pct: int
    extended_findings: list[str] = []


class PatientInfo(BaseModel):
    """Patient demographic information."""
    name: str
    patient_id: str
    age: Optional[int] = None
    gender: Optional[str] = None
    email: Optional[str] = None


class ReportGenerationRequest(BaseModel):
    """Request to generate a doctor-style report."""
    scan_id: int
    patient_info: PatientInfo
    analysis_result: AnalysisResult
    doctor_name: str = "Dr. VisionCare AI"
    hospital_name: str = "VisionCare Medical Center"
    scan_date: Optional[str] = None


class ReportGenerationResponse(BaseModel):
    """Response from report generation."""
    status: str
    report_id: str
    report_path: str
    patient_name: str
    diagnosis: str
    confidence: float
    message: Optional[str] = None


# ============================================================================
# Report Generation Functions
# ============================================================================

def generate_report_from_request(request: ReportGenerationRequest) -> ReportGenerationResponse:
    """
    Generate a doctor-style report from a request object.
    
    Args:
        request: ReportGenerationRequest with patient and analysis data
        
    Returns:
        ReportGenerationResponse with report details
    """
    try:
        # Prepare scan date
        scan_date = request.scan_date or datetime.now().strftime("%Y-%m-%d")
        
        # Convert analysis result to dictionary
        result_dict = request.analysis_result.model_dump()
        
        # Generate PDF report
        report_path = generate_doctor_style_pdf(
            result=result_dict,
            scan_id=request.scan_id,
            patient_name=request.patient_info.name,
            patient_id=request.patient_info.patient_id,
            doctor_name=request.doctor_name,
            hospital_name=request.hospital_name,
            scan_date=scan_date,
        )
        
        # Return success response
        return ReportGenerationResponse(
            status="success",
            report_id=f"VCA-{request.scan_id}",
            report_path=str(report_path),
            patient_name=request.patient_info.name,
            diagnosis=request.analysis_result.label_full,
            confidence=request.analysis_result.confidence,
            message=f"Report generated successfully for {request.patient_info.name}",
        )
        
    except Exception as e:
        return ReportGenerationResponse(
            status="error",
            report_id=f"VCA-{request.scan_id}",
            report_path="",
            patient_name=request.patient_info.name,
            diagnosis="",
            confidence=0.0,
            message=f"Error generating report: {str(e)}",
        )


# ============================================================================
# FastAPI Integration Example
# ============================================================================

"""
To use with FastAPI, add the following to your main.py:

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

app = FastAPI()

@app.post("/api/v1/generate-report", response_model=ReportGenerationResponse)
async def generate_report_endpoint(request: ReportGenerationRequest):
    '''
    Generate a professional doctor-style PDF report.
    
    POST /api/v1/generate-report
    
    Request body:
    {
        "scan_id": 12345,
        "patient_info": {
            "name": "John Doe",
            "patient_id": "PID-2024-001",
            "age": 55,
            "gender": "M"
        },
        "analysis_result": {
            "label": "DME",
            "label_full": "Diabetic Macular Edema",
            "confidence": 87.5,
            "stage": "Moderate",
            "retina_affected_pct": 35,
            "lesion_area_pct": 12,
            "fluid_area_pct": 23,
            "abnormality_pct": 42,
            "extended_findings": ["Retinal Thickening", "Fluid Accumulation"]
        },
        "doctor_name": "Dr. Sarah Johnson, MD",
        "hospital_name": "Vision Care Medical Center"
    }
    
    Response:
    {
        "status": "success",
        "report_id": "VCA-12345",
        "report_path": "/path/to/report.pdf",
        "patient_name": "John Doe",
        "diagnosis": "Diabetic Macular Edema",
        "confidence": 87.5
    }
    '''
    response = generate_report_from_request(request)
    
    if response.status != "success":
        raise HTTPException(status_code=500, detail=response.message)
    
    return response


@app.get("/api/v1/download-report/{report_id}")
async def download_report(report_id: str):
    '''
    Download a generated report PDF.
    
    GET /api/v1/download-report/VCA-12345
    '''
    from utils.config import REPORTS_DIR
    
    # Extract scan_id from report_id (VCA-12345 -> 12345)
    try:
        scan_id = report_id.split("-")[-1]
        report_path = REPORTS_DIR / f"Doctor_Report_{scan_id}.pdf"
        
        if not report_path.exists():
            raise HTTPException(status_code=404, detail="Report not found")
        
        return FileResponse(
            path=report_path,
            filename=f"Doctor_Report_{scan_id}.pdf",
            media_type="application/pdf"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
"""


# ============================================================================
# CLI Usage Example
# ============================================================================

def example_cli_usage():
    """Example of using the report generator from CLI."""
    
    # Create sample request
    request = ReportGenerationRequest(
        scan_id=12345,
        patient_info=PatientInfo(
            name="John Doe",
            patient_id="PID-2024-001",
            age=55,
            gender="M",
        ),
        analysis_result=AnalysisResult(
            label="DME",
            label_full="Diabetic Macular Edema",
            confidence=87.5,
            stage="Moderate",
            retina_affected_pct=35,
            lesion_area_pct=12,
            fluid_area_pct=23,
            abnormality_pct=42,
            extended_findings=[
                "Diabetic Macular Edema",
                "Retinal Thickening",
                "Fluid Accumulation",
            ],
        ),
        doctor_name="Dr. Sarah Johnson, MD",
        hospital_name="Vision Care Medical Center",
    )
    
    # Generate report
    response = generate_report_from_request(request)
    
    # Print results
    print(f"Report Generation Status: {response.status}")
    print(f"Report ID: {response.report_id}")
    print(f"Report Path: {response.report_path}")
    print(f"Patient: {response.patient_name}")
    print(f"Diagnosis: {response.diagnosis} ({response.confidence:.1f}% confidence)")
    
    return response


if __name__ == "__main__":
    print("=" * 70)
    print("VisionCare — Report Generation Integration Example")
    print("=" * 70)
    print()
    
    response = example_cli_usage()
    
    print()
    print("=" * 70)
    print("✓ Integration example completed successfully!")
    print("=" * 70)
