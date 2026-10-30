import os
import io
import logging
from pathlib import Path
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)

# Configure Tesseract command
DEFAULT_TESSERACT_WIN = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CMD = os.getenv("TESSERACT_CMD")

if not TESSERACT_CMD:
    if os.path.exists(DEFAULT_TESSERACT_WIN):
        TESSERACT_CMD = DEFAULT_TESSERACT_WIN
    else:
        TESSERACT_CMD = "tesseract"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Configure Poppler bin path
DEFAULT_POPPLER_WIN = r"C:\poppler\poppler-26.09.0\Library\bin"
POPPLER_PATH = os.getenv("POPPLER_PATH")
if not POPPLER_PATH and os.path.exists(DEFAULT_POPPLER_WIN):
    POPPLER_PATH = DEFAULT_POPPLER_WIN

# Default OCR language
OCR_LANG = os.getenv("OCR_LANG", "eng")


def ocr_image(image: Image.Image, lang: str = None) -> str:
    """
    Extract text from a PIL Image using Tesseract OCR.
    """
    selected_lang = lang or OCR_LANG
    try:
        text = pytesseract.image_to_string(image, lang=selected_lang)
        return text.strip()
    except Exception as e:
        logger.error(f"Tesseract OCR failed on image: {e}")
        raise RuntimeError(f"OCR processing failed: {e}") from e


def ocr_pdf_page_with_pymupdf(pdf_path: str, page_number: int, dpi: int = 200, lang: str = None) -> str:
    """
    Extract text from a specific PDF page using PyMuPDF pixmap rendering and Tesseract OCR.
    Page numbers are 1-indexed.
    """
    import pymupdf

    doc = pymupdf.open(pdf_path)
    try:
        if page_number < 1 or page_number > len(doc):
            raise IndexError(f"Page number {page_number} out of range (1..{len(doc)})")
        
        page = doc[page_number - 1]
        pix = page.get_pixmap(dpi=dpi)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        return ocr_image(image, lang=lang)
    finally:
        doc.close()


def ocr_pdf_page_with_pdf2image(pdf_path: str, page_number: int, dpi: int = 200, lang: str = None) -> str:
    """
    Extract text from a specific PDF page using pdf2image and Poppler.
    Page numbers are 1-indexed.
    """
    from pdf2image import convert_from_path

    images = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        first_page=page_number,
        last_page=page_number,
        poppler_path=POPPLER_PATH if POPPLER_PATH else None
    )
    if not images:
        return ""
    return ocr_image(images[0], lang=lang)


def ocr_pdf_page(pdf_path: str, page_number: int, dpi: int = 200, lang: str = None) -> str:
    """
    Extract text from a single PDF page with OCR.
    Tries PyMuPDF pixmap first (fast, in-memory, self-contained),
    falling back to pdf2image if needed.
    """
    try:
        return ocr_pdf_page_with_pymupdf(pdf_path, page_number, dpi=dpi, lang=lang)
    except Exception as pymupdf_err:
        logger.warning(f"PyMuPDF pixmap OCR failed for page {page_number}, falling back to pdf2image: {pymupdf_err}")
        return ocr_pdf_page_with_pdf2image(pdf_path, page_number, dpi=dpi, lang=lang)


def extract_text_with_ocr(pdf_path, page_numbers=None, dpi=200, lang=None):
    """
    Extract text from PDF pages using Tesseract OCR.
    
    Args:
        pdf_path: Path to PDF file.
        page_numbers: Optional list of 1-indexed page numbers to OCR.
                      If None, all pages in the PDF are OCR-processed.
        dpi: Resolution for rendering.
        lang: Tesseract language code.

    Returns:
        list[dict]: List of {"page": page_number, "text": extracted_text}
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    import pymupdf
    doc = pymupdf.open(str(pdf_path))
    total_pages = len(doc)
    doc.close()

    target_pages = page_numbers if page_numbers is not None else list(range(1, total_pages + 1))
    extracted_pages = []

    for page_num in target_pages:
        if page_num < 1 or page_num > total_pages:
            logger.warning(f"Skipping invalid page number: {page_num}")
            continue
        try:
            text = ocr_pdf_page(str(pdf_path), page_num, dpi=dpi, lang=lang)
            extracted_pages.append({
                "page": page_num,
                "text": text
            })
            print(f"OCR processed page {page_num}: {len(text)} characters")
        except Exception as e:
            logger.error(f"Failed to OCR page {page_num} of {pdf_path}: {e}")
            extracted_pages.append({
                "page": page_num,
                "text": ""
            })

    return extracted_pages
