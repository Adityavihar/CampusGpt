import os
import re
import logging
from pathlib import Path
import pymupdf

logger = logging.getLogger(__name__)

# Minimum characters on a page to consider it as containing extractable digital text
MIN_TEXT_CHARS = 30


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing null bytes, normalizing spaces,
    and trimming extraneous blank lines.
    """
    if not text:
        return ""
    # Remove null bytes
    text = text.replace("\x00", " ")
    # Replace carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading and trailing whitespace
    return text.strip()


def extract_text_from_pdf(pdf_path: str, min_chars_for_ocr: int = MIN_TEXT_CHARS, force_ocr: bool = False):
    """
    Extract text from a PDF document, preserving page numbers.
    Uses PyMuPDF for native digital text.
    If a page has insufficient text (scanned or image-based), falls back
    to OCR strictly for that page, preserving mixed and scanned document integrity.

    Args:
        pdf_path: Path to the PDF file.
        min_chars_for_ocr: Character threshold below which page is treated as scanned.
        force_ocr: If True, forces OCR on every page.

    Returns:
        list[dict]: List of {"page": page_number, "text": cleaned_text, "source": filename, "ocr_used": bool}
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        document = pymupdf.open(str(pdf_file))
    except Exception as e:
        logger.error(f"Failed to open PDF {pdf_path}: {e}")
        raise ValueError(f"Could not open PDF file {pdf_file.name}: {e}") from e

    pages = []
    pages_needing_ocr = []

    try:
        for page_number, page in enumerate(document, start=1):
            raw_text = "" if force_ocr else page.get_text().strip()
            cleaned = clean_text(raw_text)

            pages.append({
                "page": page_number,
                "text": cleaned,
                "source": pdf_file.name,
                "ocr_used": False
            })

            # Check if page lacks sufficient text
            if len(cleaned) < min_chars_for_ocr or force_ocr:
                pages_needing_ocr.append(page_number)
    finally:
        document.close()

    # Per-page OCR fallback strictly for pages requiring it
    if pages_needing_ocr:
        logger.info(
            f"Invoking targeted OCR for {len(pages_needing_ocr)} pages in {pdf_file.name}: {pages_needing_ocr}"
        )
        from services.ocr_service import ocr_pdf_page

        for page_num in pages_needing_ocr:
            try:
                ocr_text = ocr_pdf_page(str(pdf_file), page_num)
                cleaned_ocr = clean_text(ocr_text)
                if cleaned_ocr:
                    # Update the specific page record
                    for p in pages:
                        if p["page"] == page_num:
                            p["text"] = cleaned_ocr
                            p["ocr_used"] = True
                            break
            except Exception as ocr_err:
                logger.error(f"OCR failed for page {page_num} of {pdf_file.name}: {ocr_err}")

    # Retain pages that have extracted content
    valid_pages = [page for page in pages if page["text"].strip()]
    return valid_pages
