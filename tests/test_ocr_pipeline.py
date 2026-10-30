import os
import pytest
from PIL import Image, ImageDraw
import pymupdf
from services.document_parser import extract_text_from_pdf
from services.ocr_service import ocr_pdf_page, extract_text_with_ocr


@pytest.fixture
def scanned_and_mixed_pdfs(tmp_path):
    # 1. Scanned PDF (single page image)
    img = Image.new("RGB", (600, 200), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((30, 40), "VIT-AP SCANNED DOCUMENT CONFIRMATION", fill=(0, 0, 0))
    d.text((30, 80), "Attendance threshold is 75 percent minimum.", fill=(0, 0, 0))
    img_path = tmp_path / "page_image.png"
    img.save(str(img_path))

    scanned_pdf_path = tmp_path / "scanned_doc.pdf"
    doc_scanned = pymupdf.open()
    p = doc_scanned.new_page(width=600, height=300)
    p.insert_image(pymupdf.Rect(0, 0, 600, 200), filename=str(img_path))
    doc_scanned.save(str(scanned_pdf_path))
    doc_scanned.close()

    # 2. Mixed PDF (page 1 text, page 2 scanned image)
    mixed_pdf_path = tmp_path / "mixed_doc.pdf"
    doc_mixed = pymupdf.open()
    p1 = doc_mixed.new_page(width=595, height=842)
    p1.insert_text((50, 80), "Page 1 Digital Text for VIT-AP University")
    p2 = doc_mixed.new_page(width=600, height=300)
    p2.insert_image(pymupdf.Rect(0, 0, 600, 200), filename=str(img_path))
    doc_mixed.save(str(mixed_pdf_path))
    doc_mixed.close()

    return str(scanned_pdf_path), str(mixed_pdf_path)


def test_scanned_pdf_ocr(scanned_and_mixed_pdfs):
    scanned_pdf, _ = scanned_and_mixed_pdfs
    pages = extract_text_from_pdf(scanned_pdf)
    assert len(pages) == 1
    assert pages[0]["page"] == 1
    assert pages[0]["ocr_used"] is True
    assert "attendance" in pages[0]["text"].lower() or "vit-ap" in pages[0]["text"].lower()


def test_mixed_pdf_ocr(scanned_and_mixed_pdfs):
    _, mixed_pdf = scanned_and_mixed_pdfs
    pages = extract_text_from_pdf(mixed_pdf)
    assert len(pages) == 2
    # Page 1 was digital text
    assert pages[0]["page"] == 1
    assert pages[0]["ocr_used"] is False
    assert "Digital Text" in pages[0]["text"]
    # Page 2 was scanned image
    assert pages[1]["page"] == 2
    assert pages[1]["ocr_used"] is True
