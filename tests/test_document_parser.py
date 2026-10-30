import pytest
from pathlib import Path
from services.document_parser import clean_text, extract_text_from_pdf


def test_clean_text():
    raw = "Hello\x00World\r\n\r\n\n\nTest   Multiple   Spaces"
    cleaned = clean_text(raw)
    assert "\x00" not in cleaned
    assert "\r" not in cleaned
    assert "Hello World" in cleaned


def test_extract_text_missing_file():
    with pytest.raises(FileNotFoundError):
        extract_text_from_pdf("data/documents/non_existent_file.pdf")


def test_extract_text_existing_pdf():
    pdf_path = "data/documents/VITAP_Academic_Regulations_and_FFCS.pdf"
    pages = extract_text_from_pdf(pdf_path)
    assert len(pages) > 0
    assert pages[0]["page"] == 1
    assert "FFCS" in pages[0]["text"] or "Regulations" in pages[0]["text"]
