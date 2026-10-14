"""Verification script for Retinal Care RAG Chatbot Assistant."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.rag_chatbot import process_chat_query_rag, get_faiss_index, is_medical_query

def run_tests():
    print("=== Chatbot Verification Started ===")
    
    # 1. Test Medical Query Classification
    print("\n1. Testing Medical Query Classification:")
    queries = [
        ("What is Glaucoma?", True),
        ("How to treat DME?", True),
        ("கண் அழுத்தம் சிகிச்சை என்ன?", True),
        ("france capital", False),
        ("who is the president of USA?", False)
    ]
    for q, expected in queries:
        res = is_medical_query(q)
        print(f"  Query: '{q}' -> Is Medical: {res} (Expected: {expected})")
        assert res == expected, f"Classification failed for: {q}"
        
    # 2. Test RAG Vector Search & FAISS Indexing
    print("\n2. Initializing FAISS & SentenceTransformers...")
    index = get_faiss_index()
    print(f"  FAISS Index size: {index.index.ntotal} chunks")
    
    # 3. Test Chatbot Endpoint logic
    print("\n3. Testing Chatbot responses in multiple languages:")
    
    # English
    print("\n[English Query]: 'Explain Glaucoma treatment options'")
    res = process_chat_query_rag("Explain Glaucoma treatment options", user_id=123)
    print("Response:")
    print(res["response"])
    print(f"Detected Language: {res['language']}")
    assert res["language"] == "english", "Should detect English"
    
    # Tamil
    print("\n[Tamil Query]: 'விழித்திரை நரம்பு அடைப்பு என்றால் என்ன?'")
    res = process_chat_query_rag("விழித்திரை நரம்பு அடைப்பு என்றால் என்ன?", user_id=123)
    print("Response:")
    print(res["response"])
    print(f"Detected Language: {res['language']}")
    assert res["language"] == "tamil", "Should detect Tamil"
    
    # Tanglish
    print("\n[Tanglish Query]: 'AMD prevention pathi sollunga'")
    res = process_chat_query_rag("AMD prevention pathi sollunga", user_id=123)
    print("Response:")
    print(res["response"])
    print(f"Detected Language: {res['language']}")
    assert res["language"] == "tanglish", "Should detect Tanglish"
    
    # Non-medical query decline
    print("\n[Non-Medical Query]: 'Tell me a joke'")
    res = process_chat_query_rag("Tell me a joke", user_id=123)
    print("Response:")
    print(res["response"])
    assert "Medical AI Assistant" in res["response"] or "medical-related questions" in res["response"], "Should decline non-medical topic"

    # Active Scan Explanation
    print("\n[Active Scan Query]: 'Explain my prediction result'")
    scan_data = {
        "label": "DME",
        "confidence": 0.942,
        "stage": "Severe",
        "total_affected_pct": 38.5,
        "fluid_area_pct": 12.4,
        "lesion_area_pct": 8.2,
        "patient_name": "Ramesh Kumar"
    }
    res = process_chat_query_rag("Explain my prediction result at beginner level", user_id=123, scan_data=scan_data)
    print("Response:")
    print(res["response"])

    print("\n=== Chatbot Verification Succeeded! ===")

if __name__ == "__main__":
    run_tests()
