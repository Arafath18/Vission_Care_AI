import os
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import unittest
from visioncare.utils.rag_chatbot import process_chat_query_rag, set_session_attachment, clear_session_attachment
from visioncare.api.main import _extract_text_from_pdf, _extract_text_from_docx

class TestFileQAAndGeneralQA(unittest.TestCase):
    def test_pdf_extraction(self):
        # Check if test.pdf or any pdf exists
        test_pdf = ROOT / "visioncare" / "test.pdf"
        if test_pdf.exists():
            with open(test_pdf, "rb") as f:
                data = f.read()
            text = _extract_text_from_pdf(data)
            print(f"\nPDF Extraction Test: Extracted {len(text)} chars from {test_pdf.name}")
            self.assertTrue(len(text) > 0)
        else:
            print("\nPDF Extraction Test: No test.pdf found, skipping file read check.")

    def test_docx_extraction(self):
        import io
        import zipfile
        
        # Build a mock docx in memory
        docx_buffer = io.BytesIO()
        with zipfile.ZipFile(docx_buffer, "w") as docx:
            xml_data = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
            <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
                <w:body>
                    <w:p>
                        <w:r>
                            <w:t>This is a mock Word Document text content for chatbot testing.</w:t>
                        </w:r>
                    </w:p>
                </w:body>
            </w:document>"""
            docx.writestr("word/document.xml", xml_data)
            
        data = docx_buffer.getvalue()
        text = _extract_text_from_docx(data)
        print(f"\nDOCX Extraction Test: Extracted text: '{text}'")
        self.assertIn("mock Word Document", text)

    def test_general_qa_greetings(self):
        res = process_chat_query_rag("Hello assistant", user_id=999)
        print(f"\nGeneral QA Greeting (English): {res['response']}")
        self.assertIn("AI clinical assistant", res["response"])
        self.assertEqual(res["language"], "english")
        
        res_tamil = process_chat_query_rag("வணக்கம்", user_id=999)
        print(f"\nGeneral QA Greeting (Tamil): {res_tamil['response']}")
        self.assertIn("வணக்கம்", res_tamil["response"])
        self.assertEqual(res_tamil["language"], "tamil")

    def test_general_qa_science(self):
        res = process_chat_query_rag("What is photosynthesis?", user_id=999)
        print(f"\nGeneral QA Science: {res['response']}")
        self.assertIn("process", res["response"])
        self.assertIn("sun", res["response"])

    def test_general_qa_code(self):
        res = process_chat_query_rag("binary search in python", user_id=999)
        print(f"\nGeneral QA Code: {res['response']}")
        self.assertIn("def binary_search", res["response"])
        self.assertIn("Complexity", res["response"])

    def test_attached_file_qa(self):
        # Set file attachment in session
        set_session_attachment(user_id=999, attachment_data={
            "filename": "patient_report.txt",
            "type": "text/plain",
            "text": "Patient Name: Rajesh Kumar. Blood Pressure: 130/85 mmHg. HbA1c: 6.8%. Severe macular edema detected in left eye."
        })
        
        # Query about name
        res_name = process_chat_query_rag("What is the patient name?", user_id=999)
        print(f"\nAttached File QA (Name): {res_name['response']}")
        self.assertIn("Rajesh Kumar", res_name["response"])
        
        # Query about BP
        res_bp = process_chat_query_rag("What is the blood pressure?", user_id=999)
        print(f"\nAttached File QA (BP): {res_bp['response']}")
        self.assertIn("130/85", res_bp["response"])
        
        # Clear attachment
        clear_session_attachment(user_id=999)
        
        # Query again, should not see name
        res_after = process_chat_query_rag("What is the patient name?", user_id=999)
        print(f"\nQA After Clearing Attachment: {res_after['response']}")
        self.assertNotIn("Rajesh Kumar", res_after["response"])

if __name__ == "__main__":
    unittest.main()
