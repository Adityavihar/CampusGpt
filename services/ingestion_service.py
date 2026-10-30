import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional

from services.document_parser import extract_text_from_pdf
from services.chunking import chunk_text
from services.embeddings import get_embeddings
from services.vector_store import add_documents, delete_document

logger = logging.getLogger(__name__)

DOCUMENTS_DIR = Path("data/documents")
CATALOG_PATH = Path("data/catalog.json")


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file for change detection."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_catalog() -> dict:
    """Load or initialize document catalog JSON."""
    if CATALOG_PATH.exists():
        try:
            with open(CATALOG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load catalog.json: {e}")
    return {"documents": {}, "last_updated": None}


def save_catalog(catalog: dict):
    """Save document catalog JSON."""
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)


def get_document_category(filename: str, catalog: dict) -> str:
    """Get category for a document filename from catalog or infer by convention."""
    if filename in catalog.get("documents", {}):
        return catalog["documents"][filename].get("category", "General University Information")
    
    lower = filename.lower()
    if "program" in lower or "course" in lower:
        return "Academic Programs"
    elif "admission" in lower or "eligibility" in lower:
        return "Admissions"
    elif "fee" in lower or "scholarship" in lower:
        return "Fees & Scholarships"
    elif "calendar" in lower or "schedule" in lower:
        return "Academic Calendar"
    elif "regulation" in lower or "policy" in lower or "act" in lower:
        return "Academic Regulations"
    elif "hostel" in lower or "transport" in lower or "facility" in lower:
        return "Campus Facilities"
    elif "attendance" in lower:
        return "Attendance Regulations"
    elif "disclosure" in lower or "overview" in lower:
        return "University Overview"
    return "Institutional Information"


def index_single_pdf(
    pdf_path: Path,
    category: Optional[str] = None,
    force: bool = False
) -> dict:
    """
    Ingest, parse, chunk, embed, and index a single PDF document.
    Idempotent: Replaces previous chunks for the document.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    filename = pdf_path.name
    catalog = load_catalog()
    doc_meta = catalog.get("documents", {}).get(filename, {})
    current_hash = compute_file_hash(pdf_path)

    # Check if unchanged and force is False
    if not force and doc_meta.get("sha256") == current_hash and doc_meta.get("status") == "indexed":
        logger.info(f"Skipping {filename} — already indexed and unchanged.")
        return {
            "source": filename,
            "status": "skipped",
            "reason": "unchanged",
            "chunks_stored": doc_meta.get("chunks_count", 0),
            "pages_processed": doc_meta.get("pages_count", 0)
        }

    doc_category = category or doc_meta.get("category") or get_document_category(filename, catalog)

    # Step 1: Parse PDF pages
    pages = extract_text_from_pdf(str(pdf_path))
    if not pages:
        return {
            "source": filename,
            "status": "failed",
            "reason": "No extractable text or content in PDF",
            "chunks_stored": 0,
            "pages_processed": 0
        }

    ocr_pages_count = sum(1 for p in pages if p.get("ocr_used", False))

    # Step 2: Split text into chunks
    all_chunks = []
    for page in pages:
        page_chunks = chunk_text(page["text"])
        for idx, chunk in enumerate(page_chunks):
            all_chunks.append({
                "source": filename,
                "page": page["page"],
                "text": chunk,
                "category": doc_category,
                "chunk_index": idx
            })

    if not all_chunks:
        return {
            "source": filename,
            "status": "failed",
            "reason": "Text extracted but no chunks created",
            "chunks_stored": 0,
            "pages_processed": len(pages)
        }

    # Step 3: Embed chunks in batches
    chunk_texts = [c["text"] for c in all_chunks]
    embeddings = get_embeddings(chunk_texts)

    # Step 4: Add to ChromaDB with replace_existing=True
    add_documents(all_chunks, embeddings, replace_existing=True)

    # Update catalog
    import datetime
    if "documents" not in catalog:
        catalog["documents"] = {}

    catalog["documents"][filename] = {
        "title": doc_meta.get("title", filename.replace("_", " ").replace(".pdf", "")),
        "category": doc_category,
        "source_url": doc_meta.get("source_url", "https://vitap.ac.in"),
        "sha256": current_hash,
        "pages_count": len(pages),
        "ocr_pages_count": ocr_pages_count,
        "chunks_count": len(all_chunks),
        "status": "indexed",
        "last_indexed": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    catalog["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save_catalog(catalog)

    return {
        "source": filename,
        "status": "indexed",
        "pages_processed": len(pages),
        "pages_ocr_processed": ocr_pages_count,
        "chunks_stored": len(all_chunks),
        "category": doc_category
    }


def index_all_documents(force: bool = False) -> dict:
    """
    Scan DOCUMENTS_DIR and index all discovered PDF files.
    """
    if not DOCUMENTS_DIR.exists():
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        return {
            "total_documents": 0,
            "documents_indexed": 0,
            "skipped": 0,
            "errors": [],
            "details": []
        }

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))
    total_docs = len(pdf_files)
    indexed_count = 0
    skipped_count = 0
    errors = []
    details = []

    for pdf_path in sorted(pdf_files):
        try:
            result = index_single_pdf(pdf_path, force=force)
            details.append(result)
            if result.get("status") == "indexed":
                indexed_count += 1
            elif result.get("status") == "skipped":
                skipped_count += 1
            else:
                errors.append(f"{pdf_path.name}: {result.get('reason')}")
        except Exception as e:
            logger.exception(f"Error indexing {pdf_path.name}")
            errors.append(f"{pdf_path.name}: {str(e)}")
            details.append({
                "source": pdf_path.name,
                "status": "error",
                "error": str(e)
            })

    return {
        "total_documents": total_docs,
        "documents_indexed": indexed_count,
        "skipped": skipped_count,
        "errors": errors,
        "details": details
    }
