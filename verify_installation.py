#!/usr/bin/env python3
"""
Verification Script — Doctor-Style Report Generator Implementation

This script verifies that all components are properly installed and functional.
"""

import sys
from pathlib import Path

# Add visioncare to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

print("=" * 70)
print("VisionCare Doctor-Style Report Generator — Implementation Verification")
print("=" * 70)
print()

# Check 1: Dependencies
print("✓ Checking dependencies...")
dependencies_ok = True

try:
    import reportlab
    print("  ✓ reportlab installed")
except ImportError:
    print("  ✗ reportlab NOT installed")
    dependencies_ok = False

try:
    import fpdf
    print("  ✓ fpdf2 installed")
except ImportError:
    print("  ✗ fpdf2 NOT installed")
    dependencies_ok = False

try:
    import qrcode
    print("  ✓ qrcode installed")
except ImportError:
    print("  ✗ qrcode NOT installed")
    dependencies_ok = False

try:
    from PIL import Image
    print("  ✓ Pillow installed")
except ImportError:
    print("  ✗ Pillow NOT installed")
    dependencies_ok = False

print()

# Check 2: Files
print("✓ Checking files...")
files_ok = True

required_files = [
    "utils/reports.py",
    "utils/config.py",
    "utils/inference.py",
    "outputs/reports",
    "requirements.txt",
]

for file_path in required_files:
    full_path = ROOT / file_path
    if full_path.exists():
        print(f"  ✓ {file_path}")
    else:
        print(f"  ✗ {file_path} NOT FOUND")
        files_ok = False

print()

# Check 3: Documentation
print("✓ Checking documentation...")
docs_ok = True

doc_files = [
    "QUICK_REFERENCE.md",
    "REPORT_GENERATION_GUIDE.md",
    "API_INTEGRATION_GUIDE.md",
    "DOCTOR_REPORT_README.md",
    "IMPLEMENTATION_SUMMARY.md",
    "README_NAVIGATION.md",
]

for doc in doc_files:
    doc_path = ROOT / doc
    if doc_path.exists():
        try:
            lines = doc_path.read_text(encoding='utf-8').split('\n')
        except UnicodeDecodeError:
            lines = doc_path.read_text(encoding='latin-1').split('\n')
        print(f"  ✓ {doc} ({len(lines)} lines)")
    else:
        print(f"  ✗ {doc} NOT FOUND")
        docs_ok = False

print()

# Check 4: Test Files
print("✓ Checking test files...")
test_ok = True

test_files = [
    "test_doctor_report.py",
    "report_integration_example.py",
]

for test in test_files:
    test_path = ROOT / test
    if test_path.exists():
        lines = test_path.read_text().split('\n')
        print(f"  ✓ {test} ({len(lines)} lines)")
    else:
        print(f"  ✗ {test} NOT FOUND")
        test_ok = False

print()

# Check 5: Generated Reports
print("✓ Checking generated reports...")
reports_dir = ROOT / "outputs" / "reports"
if reports_dir.exists():
    reports = list(reports_dir.glob("*.pdf"))
    print(f"  ✓ {len(reports)} PDF reports found")
    for report in sorted(reports)[:5]:
        size_kb = report.stat().st_size / 1024
        print(f"    - {report.name} ({size_kb:.2f} KB)")
    if len(reports) > 5:
        print(f"    ... and {len(reports) - 5} more")
else:
    print("  ℹ No reports generated yet (this is OK)")

print()

# Check 6: Import Test
print("✓ Testing imports...")
import_ok = True

try:
    from utils.reports import generate_doctor_style_pdf
    print("  ✓ generate_doctor_style_pdf imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import generate_doctor_style_pdf: {e}")
    import_ok = False

try:
    from utils.reports import generate_english_pdf
    print("  ✓ generate_english_pdf imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import generate_english_pdf: {e}")
    import_ok = False

try:
    from utils.reports import generate_tamil_pdf
    print("  ✓ generate_tamil_pdf imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import generate_tamil_pdf: {e}")
    import_ok = False

print()

# Summary
print("=" * 70)
print("Verification Summary")
print("=" * 70)

checks = {
    "Dependencies": dependencies_ok,
    "Files": files_ok,
    "Documentation": docs_ok,
    "Test Files": test_ok,
    "Imports": import_ok,
}

all_ok = all(checks.values())

for check, status in checks.items():
    symbol = "✓" if status else "✗"
    print(f"{symbol} {check}: {'OK' if status else 'ISSUES FOUND'}")

print()

if all_ok:
    print("✅ ALL CHECKS PASSED - System is ready to use!")
    print()
    print("Next steps:")
    print("  1. Read QUICK_REFERENCE.md for quick start")
    print("  2. Run: python test_doctor_report.py")
    print("  3. Check outputs/reports/ for generated PDFs")
    sys.exit(0)
else:
    print("⚠️  SOME CHECKS FAILED - Please review issues above")
    print()
    print("Common fixes:")
    print("  - Install dependencies: pip install -r requirements.txt")
    print("  - Ensure you're in the visioncare directory")
    print("  - Check file paths in error messages")
    sys.exit(1)
