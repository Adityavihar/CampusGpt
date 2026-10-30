import sys
import os
sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pymupdf
from services.document_parser import extract_text_from_pdf

def verify_pdf():
    pdf_path = "data/documents/VITAP_Fee_Structure_and_Scholarships.pdf"
    print("=" * 70)
    print("DIAGNOSTIC 1: PDF CONTENT & TEXT EXTRACTION VERIFICATION")
    print("=" * 70)
    print(f"Checking file: {pdf_path}")
    print(f"File exists: {os.path.exists(pdf_path)}")
    
    # Raw PyMuPDF inspection
    doc = pymupdf.open(pdf_path)
    print(f"Total pages in PDF: {len(doc)}")
    raw_text = doc[0].get_text()
    doc.close()
    
    print("\n--- RAW PDF TEXT (PAGE 1) ---")
    print(raw_text.strip())
    
    # Document parser extraction
    print("\n--- DOCUMENT PARSER EXTRACTION TEST ---")
    pages = extract_text_from_pdf(pdf_path)
    print(f"Pages returned by parser: {len(pages)}")
    parsed_text = pages[0]["text"]
    print(f"Parsed character count: {len(parsed_text)}")
    print(f"OCR triggered: {pages[0].get('ocr_used')}")
    
    # Keyword checks
    keywords = ["GV School Development Programme", "GVSDP", "STARS", "Merit Awards", "100%", "75%"]
    print("\n--- KEYWORD PRESENCE IN EXTRACTED TEXT ---")
    for kw in keywords:
        found = kw.lower() in parsed_text.lower()
        print(f"'{kw}': {'FOUND' if found else 'MISSING'}")

if __name__ == "__main__":
    verify_pdf()
