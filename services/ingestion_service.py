import os
import json
import hashlib
import logging
import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from services.document_parser import extract_text_from_pdf
from services.content_extractor import (
    extract_text_from_txt,
    extract_text_from_docx,
    fetch_and_extract_webpage,
    is_safe_url
)
from services.chunking import chunk_text
from services.embeddings import get_embeddings
from services.vector_store import add_documents, delete_document

logger = logging.getLogger(__name__)

DOCUMENTS_DIR = Path("data/documents")
UPLOADS_DIR = Path("data/uploads")
CATALOG_PATH = Path("data/catalog.json")

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# 15 Standard University Knowledge Base Categories
STANDARD_CATEGORIES = [
    "Academics and Courses",
    "Course Registration and FFCS",
    "Examinations",
    "Academic Calendar",
    "Admissions",
    "Fees and Scholarships",
    "Hostel",
    "Transport",
    "Library",
    "Attendance",
    "Departments and Contacts",
    "Placements",
    "Campus Facilities",
    "University Policies",
    "Other"
]

# Map legacy categories to standard names
CATEGORY_NORMALIZATION_MAP = {
    "academic regulations": "University Policies",
    "academic programs": "Academics and Courses",
    "fees & scholarships": "Fees and Scholarships",
    "fees and scholarships": "Fees and Scholarships",
    "campus facilities": "Campus Facilities",
    "university overview & governance": "University Policies",
    "general university information": "Other",
    "institutional information": "Other",
    "attendance regulations": "Attendance"
}


def normalize_category(cat: Optional[str]) -> str:
    """Normalize user or inferred category to one of the 15 standard categories."""
    if not cat:
        return "Other"
    cat_clean = cat.strip()
    if cat_clean in STANDARD_CATEGORIES:
        return cat_clean
    lower = cat_clean.lower()
    if lower in CATEGORY_NORMALIZATION_MAP:
        return CATEGORY_NORMALIZATION_MAP[lower]
    # Substring matches
    for std in STANDARD_CATEGORIES:
        if std.lower() in lower or lower in std.lower():
            return std
    return "Other"


def compute_bytes_hash(data: bytes) -> str:
    """Compute SHA-256 hash of raw bytes."""
    return hashlib.sha256(data).hexdigest()


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
        return normalize_category(catalog["documents"][filename].get("category"))
    
    lower = filename.lower()
    if "program" in lower or "course" in lower:
        return "Academics and Courses"
    elif "ffcs" in lower or "registration" in lower:
        return "Course Registration and FFCS"
    elif "exam" in lower or "cat" in lower or "fat" in lower:
        return "Examinations"
    elif "admission" in lower or "eligibility" in lower:
        return "Admissions"
    elif "fee" in lower or "scholarship" in lower:
        return "Fees and Scholarships"
    elif "calendar" in lower or "schedule" in lower:
        return "Academic Calendar"
    elif "regulation" in lower or "policy" in lower or "act" in lower:
        return "University Policies"
    elif "hostel" in lower:
        return "Hostel"
    elif "transport" in lower or "bus" in lower:
        return "Transport"
    elif "library" in lower:
        return "Library"
    elif "attendance" in lower:
        return "Attendance"
    elif "placement" in lower:
        return "Placements"
    elif "facility" in lower:
        return "Campus Facilities"
    elif "department" in lower or "contact" in lower:
        return "Departments and Contacts"
    return "Other"


def find_duplicate_by_hash(content_hash: str) -> Optional[dict]:
    """Check if an active document with the same SHA-256 hash already exists."""
    catalog = load_catalog()
    for doc_id, meta in catalog.get("documents", {}).items():
        if meta.get("sha256") == content_hash and meta.get("status") == "indexed":
            return {"doc_id": doc_id, **meta}
    return None


# =========================================================
# CORE EXTRACTION HELPER
# =========================================================

def extract_pages_and_text(file_path: Path) -> Tuple[List[Dict[str, Any]], int]:
    """
    Extract pages/text from supported file formats.
    Returns: (pages_list, ocr_pages_count)
    """
    ext = file_path.suffix.lower()
    pages = []
    ocr_count = 0

    if ext == ".pdf":
        pages = extract_text_from_pdf(str(file_path))
        ocr_count = sum(1 for p in pages if p.get("ocr_used", False))
    elif ext in (".txt", ".md"):
        text = extract_text_from_txt(file_path)
        if text.strip():
            pages = [{"page": 1, "text": text, "source": file_path.name, "ocr_used": False}]
    elif ext == ".docx":
        text = extract_text_from_docx(file_path)
        if text.strip():
            pages = [{"page": 1, "text": text, "source": file_path.name, "ocr_used": False}]
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Supported: .pdf, .txt, .md, .docx")

    return pages, ocr_count


# =========================================================
# INCREMENTAL INGESTION: UPLOADED FILE
# =========================================================

def index_uploaded_file(
    file_bytes: bytes,
    original_filename: str,
    title: Optional[str] = None,
    category: Optional[str] = None,
    publisher: Optional[str] = None,
    publication_date: Optional[str] = None,
    effective_date: Optional[str] = None,
    source_url: Optional[str] = None,
    doc_id_to_replace: Optional[str] = None,
    force: bool = False
) -> dict:
    """
    Incrementally ingest an uploaded file (PDF, TXT, DOCX) into ChromaDB and update catalog.
    Idempotent and atomic: processes new chunks first before updating catalog.
    """
    content_hash = compute_bytes_hash(file_bytes)
    ext = Path(original_filename).suffix.lower()

    if ext not in (".pdf", ".txt", ".md", ".docx"):
        return {
            "success": False,
            "error": f"Unsupported file type '{ext}'. Allowed: PDF, TXT, MD, DOCX."
        }

    # Duplicate check if not replacing or forcing
    if not force and not doc_id_to_replace:
        dup = find_duplicate_by_hash(content_hash)
        if dup:
            return {
                "success": False,
                "is_duplicate": True,
                "existing_doc_id": dup["doc_id"],
                "existing_title": dup.get("title"),
                "message": f"This exact document has already been indexed as '{dup.get('title')}' (ID: {dup['doc_id']})."
            }

    # Determine unique document ID
    catalog = load_catalog()
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    if doc_id_to_replace:
        doc_id = doc_id_to_replace
        old_meta = catalog.get("documents", {}).get(doc_id, {})
        old_version = old_meta.get("version", "1.0")
        try:
            ver_num = float(old_version.replace("v", ""))
            version = f"{ver_num + 1.0:.1f}"
        except Exception:
            version = "2.0"
    else:
        # Generate safe document ID
        base_id = Path(original_filename).stem.strip().replace(" ", "_")
        clean_base = "".join(c for c in base_id if c.isalnum() or c in ("_", "-"))
        doc_id = f"{clean_base}_{content_hash[:8]}"
        version = "1.0"

    doc_title = title.strip() if title and title.strip() else Path(original_filename).stem.replace("_", " ")
    doc_category = normalize_category(category)

    # Save physical copy in data/documents
    target_path = DOCUMENTS_DIR / f"{doc_id}{ext}"
    try:
        with open(target_path, "wb") as f:
            f.write(file_bytes)
    except Exception as e:
        logger.error(f"Failed to save document to {target_path}: {e}")
        return {"success": False, "error": f"Failed to save file on server: {e}"}

    # Step 1: Content Extraction
    try:
        pages, ocr_count = extract_pages_and_text(target_path)
    except Exception as extract_err:
        logger.exception("Text extraction failed")
        if target_path.exists():
            target_path.unlink()
        return {"success": False, "error": f"Extraction failed: {extract_err}"}

    if not pages:
        if target_path.exists():
            target_path.unlink()
        return {"success": False, "error": "No readable text content could be extracted from document."}

    # Step 2: Section-Aware Chunking
    all_chunks = []
    for page in pages:
        page_chunks = chunk_text(page["text"])
        for idx, chunk in enumerate(page_chunks):
            all_chunks.append({
                "source": doc_id,
                "doc_id": doc_id,
                "title": doc_title,
                "page": page["page"],
                "text": chunk,
                "category": doc_category,
                "chunk_index": idx
            })

    if not all_chunks:
        return {"success": False, "error": "Document extracted but zero chunks were created."}

    # Step 3: Embeddings Generation
    try:
        chunk_texts = [c["text"] for c in all_chunks]
        embeddings = get_embeddings(chunk_texts)
    except Exception as emb_err:
        logger.exception("Jina Embeddings API failed")
        return {"success": False, "error": f"Embedding generation failed: {emb_err}"}

    # Step 4: Atomic ChromaDB Storage
    try:
        add_documents(all_chunks, embeddings, replace_existing=True)
    except Exception as db_err:
        logger.exception("ChromaDB storage failed")
        return {"success": False, "error": f"Vector database storage failed: {db_err}"}

    # Step 5: Update Master Catalog
    if "documents" not in catalog:
        catalog["documents"] = {}

    catalog["documents"][doc_id] = {
        "doc_id": doc_id,
        "title": doc_title,
        "category": doc_category,
        "source_type": "file",
        "source_url": source_url or "https://vitap.ac.in",
        "original_filename": original_filename,
        "publisher": publisher or "VIT-AP University",
        "publication_date": publication_date or None,
        "effective_date": effective_date or None,
        "version": version,
        "sha256": content_hash,
        "pages_count": len(pages),
        "ocr_pages_count": ocr_count,
        "chunks_count": len(all_chunks),
        "status": "indexed",
        "error_message": None,
        "created_at": catalog["documents"].get(doc_id, {}).get("created_at", now_iso),
        "last_indexed": now_iso
    }
    catalog["last_updated"] = now_iso
    save_catalog(catalog)

    return {
        "success": True,
        "doc_id": doc_id,
        "title": doc_title,
        "category": doc_category,
        "version": version,
        "pages_processed": len(pages),
        "ocr_pages_count": ocr_count,
        "chunks_stored": len(all_chunks),
        "status": "indexed"
    }


# =========================================================
# INCREMENTAL INGESTION: WEBPAGE URL
# =========================================================

def index_webpage(
    url: str,
    title: Optional[str] = None,
    category: Optional[str] = None,
    publisher: Optional[str] = None,
    publication_date: Optional[str] = None,
    effective_date: Optional[str] = None,
    doc_id_to_replace: Optional[str] = None,
    force: bool = False
) -> dict:
    """
    Safely fetch, validate, chunk, embed, and index a webpage URL.
    Enforces SSRF protection, duplicate detection, and catalog tracking.
    """
    safe, msg = is_safe_url(url)
    if not safe:
        return {"success": False, "error": f"URL rejected by SSRF protection: {msg}"}

    try:
        extracted = fetch_and_extract_webpage(url)
    except Exception as e:
        logger.exception(f"Webpage fetch failed for {url}")
        return {"success": False, "error": f"Failed to fetch webpage: {e}"}

    content_text = extracted["text"]
    content_hash = compute_bytes_hash(content_text.encode("utf-8"))

    # Duplicate check
    if not force and not doc_id_to_replace:
        dup = find_duplicate_by_hash(content_hash)
        if dup:
            return {
                "success": False,
                "is_duplicate": True,
                "existing_doc_id": dup["doc_id"],
                "existing_title": dup.get("title"),
                "message": f"This webpage content has already been indexed as '{dup.get('title')}' (ID: {dup['doc_id']})."
            }

    catalog = load_catalog()
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    if doc_id_to_replace:
        doc_id = doc_id_to_replace
        old_meta = catalog.get("documents", {}).get(doc_id, {})
        old_version = old_meta.get("version", "1.0")
        try:
            ver_num = float(old_version.replace("v", ""))
            version = f"{ver_num + 1.0:.1f}"
        except Exception:
            version = "2.0"
    else:
        url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]
        doc_id = f"web_{url_hash}_{content_hash[:6]}"
        version = "1.0"

    doc_title = title.strip() if title and title.strip() else extracted.get("title", "Webpage Information")
    doc_category = normalize_category(category)

    # Chunking
    chunks_text = chunk_text(content_text)
    if not chunks_text:
        return {"success": False, "error": "Extracted webpage text resulted in zero chunks."}

    all_chunks = []
    for idx, c_text in enumerate(chunks_text):
        all_chunks.append({
            "source": doc_id,
            "doc_id": doc_id,
            "title": doc_title,
            "page": 1,
            "text": c_text,
            "category": doc_category,
            "chunk_index": idx
        })

    # Embeddings
    try:
        chunk_texts = [c["text"] for c in all_chunks]
        embeddings = get_embeddings(chunk_texts)
    except Exception as emb_err:
        logger.exception("Jina Embeddings API failed")
        return {"success": False, "error": f"Embedding generation failed: {emb_err}"}

    # ChromaDB
    try:
        add_documents(all_chunks, embeddings, replace_existing=True)
    except Exception as db_err:
        logger.exception("ChromaDB storage failed")
        return {"success": False, "error": f"Vector database storage failed: {db_err}"}

    if "documents" not in catalog:
        catalog["documents"] = {}

    catalog["documents"][doc_id] = {
        "doc_id": doc_id,
        "title": doc_title,
        "category": doc_category,
        "source_type": "webpage",
        "source_url": url,
        "original_filename": None,
        "publisher": publisher or "Official Website",
        "publication_date": publication_date or None,
        "effective_date": effective_date or None,
        "version": version,
        "sha256": content_hash,
        "pages_count": 1,
        "ocr_pages_count": 0,
        "chunks_count": len(all_chunks),
        "status": "indexed",
        "error_message": None,
        "created_at": catalog["documents"].get(doc_id, {}).get("created_at", now_iso),
        "last_indexed": now_iso
    }
    catalog["last_updated"] = now_iso
    save_catalog(catalog)

    return {
        "success": True,
        "doc_id": doc_id,
        "title": doc_title,
        "category": doc_category,
        "version": version,
        "pages_processed": 1,
        "chunks_stored": len(all_chunks),
        "status": "indexed"
    }


# =========================================================
# DOCUMENT MANAGEMENT: REINDEX & DELETE
# =========================================================

def reindex_document(doc_id: str) -> dict:
    """
    Re-extract, re-chunk, and re-embed an individual document by its doc_id.
    """
    catalog = load_catalog()
    doc_meta = catalog.get("documents", {}).get(doc_id)
    if not doc_meta:
        return {"success": False, "error": f"Document ID '{doc_id}' not found in catalog."}

    source_type = doc_meta.get("source_type", "file")
    if source_type == "webpage":
        url = doc_meta.get("source_url")
        if not url:
            return {"success": False, "error": "Webpage document is missing source_url."}
        return index_webpage(
            url=url,
            title=doc_meta.get("title"),
            category=doc_meta.get("category"),
            publisher=doc_meta.get("publisher"),
            publication_date=doc_meta.get("publication_date"),
            effective_date=doc_meta.get("effective_date"),
            doc_id_to_replace=doc_id,
            force=True
        )
    else:
        # Find file on disk
        candidates = list(DOCUMENTS_DIR.glob(f"{doc_id}.*"))
        if not candidates:
            # Check original filename
            orig = doc_meta.get("original_filename")
            if orig and (DOCUMENTS_DIR / orig).exists():
                candidates = [DOCUMENTS_DIR / orig]
            elif (DOCUMENTS_DIR / doc_id).exists():
                candidates = [DOCUMENTS_DIR / doc_id]

        if not candidates:
            return {"success": False, "error": f"Underlying document file for '{doc_id}' not found on disk."}

        target_file = candidates[0]
        with open(target_file, "rb") as f:
            file_bytes = f.read()

        return index_uploaded_file(
            file_bytes=file_bytes,
            original_filename=doc_meta.get("original_filename") or target_file.name,
            title=doc_meta.get("title"),
            category=doc_meta.get("category"),
            publisher=doc_meta.get("publisher"),
            publication_date=doc_meta.get("publication_date"),
            effective_date=doc_meta.get("effective_date"),
            source_url=doc_meta.get("source_url"),
            doc_id_to_replace=doc_id,
            force=True
        )


def delete_document_from_kb(doc_id: str) -> dict:
    """
    Safely delete a document's chunks from ChromaDB and remove from catalog.
    Does NOT touch chunks belonging to other documents.
    """
    catalog = load_catalog()
    doc_meta = catalog.get("documents", {}).get(doc_id)
    if not doc_meta:
        # Also check if doc_id was stored as filename key
        found_key = None
        for k in catalog.get("documents", {}):
            if k == doc_id or catalog["documents"][k].get("doc_id") == doc_id:
                found_key = k
                doc_meta = catalog["documents"][k]
                break
        if not found_key:
            return {"success": False, "error": f"Document '{doc_id}' not found in catalog."}
        doc_id = found_key

    # Delete chunks in ChromaDB
    try:
        deleted_count = delete_document(doc_id)
    except Exception as e:
        logger.error(f"Failed to delete ChromaDB chunks for {doc_id}: {e}")
        return {"success": False, "error": f"Vector database deletion failed: {e}"}

    # Remove from catalog
    deleted_title = doc_meta.get("title", doc_id)
    del catalog["documents"][doc_id]
    catalog["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save_catalog(catalog)

    # Optional: remove file from disk if uploaded
    for cand in list(DOCUMENTS_DIR.glob(f"{doc_id}.*")):
        try:
            cand.unlink()
        except Exception as file_err:
            logger.warning(f"Note: Could not delete disk file {cand}: {file_err}")

    return {
        "success": True,
        "doc_id": doc_id,
        "title": deleted_title,
        "chunks_removed": deleted_count,
        "message": f"Document '{deleted_title}' and all its {deleted_count} chunks have been removed from the knowledge base."
    }


def list_catalog_documents(
    category_filter: Optional[str] = None,
    status_filter: Optional[str] = None,
    search_query: Optional[str] = None
) -> List[dict]:
    """
    List and filter all indexed documents in catalog.
    """
    catalog = load_catalog()
    results = []

    for doc_key, meta in catalog.get("documents", {}).items():
        doc_info = {
            "doc_id": meta.get("doc_id", doc_key),
            "title": meta.get("title", doc_key),
            "category": meta.get("category", "Other"),
            "source_type": meta.get("source_type", "file"),
            "source_url": meta.get("source_url", "https://vitap.ac.in"),
            "original_filename": meta.get("original_filename", doc_key),
            "publisher": meta.get("publisher", "VIT-AP University"),
            "publication_date": meta.get("publication_date"),
            "effective_date": meta.get("effective_date"),
            "version": meta.get("version", "1.0"),
            "pages_count": meta.get("pages_count", 1),
            "chunks_count": meta.get("chunks_count", 0),
            "status": meta.get("status", "indexed"),
            "last_indexed": meta.get("last_indexed"),
            "sha256": meta.get("sha256")
        }

        # Filtering
        if category_filter and category_filter != "all" and doc_info["category"] != category_filter:
            continue
        if status_filter and status_filter != "all" and doc_info["status"] != status_filter:
            continue
        if search_query:
            q = search_query.lower()
            if q not in doc_info["title"].lower() and q not in doc_info["category"].lower() and q not in doc_info["doc_id"].lower():
                continue

        results.append(doc_info)

    # Sort newest first
    results.sort(key=lambda d: d.get("last_indexed") or "", reverse=True)
    return results


# =========================================================
# BACKWARD-COMPATIBLE PDF SCANNING WRAPPERS
# =========================================================

def index_single_pdf(
    pdf_path: Path,
    category: Optional[str] = None,
    force: bool = False
) -> dict:
    """Backward compatible wrapper for legacy route."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    res = index_uploaded_file(
        file_bytes=file_bytes,
        original_filename=pdf_path.name,
        category=category,
        force=force
    )
    if res.get("success"):
        return {
            "source": pdf_path.name,
            "status": "indexed",
            "pages_processed": res.get("pages_processed", 0),
            "pages_ocr_processed": res.get("ocr_pages_count", 0),
            "chunks_stored": res.get("chunks_stored", 0),
            "category": res.get("category")
        }
    elif res.get("is_duplicate"):
        return {
            "source": pdf_path.name,
            "status": "skipped",
            "reason": "unchanged"
        }
    else:
        return {
            "source": pdf_path.name,
            "status": "failed",
            "reason": res.get("error", "Unknown error")
        }


def index_all_documents(force: bool = False) -> dict:
    """Backward compatible wrapper scanning data/documents/."""
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
            details.append({"source": pdf_path.name, "status": "error", "error": str(e)})

    return {
        "total_documents": total_docs,
        "documents_indexed": indexed_count,
        "skipped": skipped_count,
        "errors": errors,
        "details": details
    }
