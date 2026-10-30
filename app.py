import os
import logging
from pathlib import Path
from flask import Flask, jsonify, request, render_template, session, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("campusgpt")

from services.auth import (
    is_admin_authenticated,
    admin_required,
    admin_api_required,
    verify_admin_password,
    login_admin_session,
    logout_admin_session
)
from services.embeddings import get_embeddings
from services.document_parser import extract_text_from_pdf
from services.chunking import chunk_text
from services.vector_store import add_documents, get_collection_stats, delete_document
from services.retriever import retrieve_documents, retrieve_formatted_chunks
from services.llm import generate_answer
from services.ingestion_service import (
    STANDARD_CATEGORIES,
    index_single_pdf,
    index_all_documents,
    index_uploaded_file,
    index_webpage,
    reindex_document,
    delete_document_from_kb,
    list_catalog_documents,
    load_catalog
)
from services.content_extractor import fetch_and_extract_webpage, is_safe_url
from services.attendance_service import process_attendance_file, parse_attendance_text

app = Flask(__name__)

# Security: Read session secret key from environment
app.secret_key = os.getenv("SECRET_KEY", "campusgpt-secure-session-key-vitap")

# Upload configurations
DOCUMENTS_DIR = Path("data/documents")
UPLOAD_FOLDER = Path("data/uploads")
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload limit
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".txt", ".md", ".docx"}


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def is_safe_document_path(path_str: str) -> bool:
    """Ensure path stays strictly within data/documents to prevent directory traversal."""
    try:
        resolved = Path(path_str).resolve()
        base_resolved = DOCUMENTS_DIR.resolve()
        return base_resolved in resolved.parents or resolved == base_resolved
    except Exception:
        return False


# =========================================================
# STUDENT HOME & CHAT PAGE
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# ADMIN AUTHENTICATION ROUTES
# =========================================================

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login_page():
    if request.method == "POST":
        data = request.get_json(silent=True) or request.form
        password = data.get("password", "")
        if verify_admin_password(password):
            login_admin_session()
            if request.is_json or request.headers.get("Accept") == "application/json":
                return jsonify({"success": True, "message": "Authenticated successfully."})
            return redirect(url_for("admin_knowledge_base_page"))
        if request.is_json or request.headers.get("Accept") == "application/json":
            return jsonify({"success": False, "message": "Invalid administrator password."}), 401
        return render_template("admin_login.html", error="Invalid administrator password."), 401

    if is_admin_authenticated():
        return redirect(url_for("admin_knowledge_base_page"))
    return render_template("admin_login.html")


@app.route("/admin/logout", methods=["POST", "GET"])
def admin_logout():
    logout_admin_session()
    if request.is_json or request.headers.get("Accept") == "application/json":
        return jsonify({"success": True, "message": "Logged out successfully."})
    return redirect(url_for("admin_login_page"))


# =========================================================
# ADMIN KNOWLEDGE MANAGER PAGE & ENDPOINTS
# =========================================================

@app.route("/admin/knowledge-base")
@admin_required
def admin_knowledge_base_page():
    return render_template("admin_knowledge_base.html", categories=STANDARD_CATEGORIES)


@app.route("/api/admin/stats")
@admin_api_required
def admin_stats():
    stats = get_collection_stats()
    catalog = load_catalog()
    docs = catalog.get("documents", {})
    categories = {d.get("category") for d in docs.values() if d.get("category")}
    return jsonify({
        "success": True,
        "stats": {
            "total_documents": len(docs),
            "total_chunks": stats.get("total_chunks", 0),
            "total_categories": len(categories),
            "collection_status": "Healthy (HNSW Cosine)"
        }
    })


@app.route("/api/admin/documents")
@admin_api_required
def admin_list_documents():
    category = request.args.get("category")
    status = request.args.get("status")
    search_q = request.args.get("search")

    docs = list_catalog_documents(
        category_filter=category,
        status_filter=status,
        search_query=search_q
    )
    return jsonify({
        "success": True,
        "count": len(docs),
        "documents": docs
    })


@app.route("/api/admin/upload", methods=["POST"])
@admin_api_required
def admin_upload_document():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part in request."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    orig_name = secure_filename(file.filename)
    if not allowed_file(orig_name):
        return jsonify({
            "success": False,
            "error": "Invalid file type. Allowed: .pdf, .txt, .md, .docx"
        }), 400

    title = request.form.get("title")
    category = request.form.get("category", "").strip()
    if not category:
        return jsonify({"success": False, "error": "Category parameter is required."}), 400

    publisher = request.form.get("publisher")
    pub_date = request.form.get("publication_date")
    eff_date = request.form.get("effective_date")
    source_url = request.form.get("source_url")
    doc_id_to_replace = request.form.get("doc_id_to_replace")
    force = request.form.get("force", "false").lower() in ("true", "1")

    file_bytes = file.read()
    if not file_bytes:
        return jsonify({"success": False, "error": "Uploaded file is empty."}), 400

    result = index_uploaded_file(
        file_bytes=file_bytes,
        original_filename=orig_name,
        title=title,
        category=category,
        publisher=publisher,
        publication_date=pub_date,
        effective_date=eff_date,
        source_url=source_url,
        doc_id_to_replace=doc_id_to_replace,
        force=force
    )

    status_code = 200 if result.get("success") or result.get("is_duplicate") else 500
    return jsonify(result), status_code


@app.route("/api/admin/webpage/preview", methods=["POST"])
@admin_api_required
def admin_preview_webpage():
    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"success": False, "error": "Missing 'url' parameter."}), 400

    safe, msg = is_safe_url(url)
    if not safe:
        return jsonify({"success": False, "error": f"SSRF Protection: {msg}"}), 400

    try:
        extracted = fetch_and_extract_webpage(url)
        # Show first 800 characters as preview snippet
        snippet = extracted["text"][:800] + ("..." if len(extracted["text"]) > 800 else "")
        return jsonify({
            "success": True,
            "preview": {
                "title": extracted["title"],
                "description": extracted.get("description", ""),
                "text": snippet,
                "total_chars": len(extracted["text"])
            }
        })
    except Exception as e:
        logger.exception("Webpage preview error")
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/admin/webpage/index", methods=["POST"])
@admin_api_required
def admin_index_webpage():
    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()
    category = data.get("category", "").strip()
    title = data.get("title")
    publisher = data.get("publisher")
    pub_date = data.get("publication_date")
    eff_date = data.get("effective_date")
    doc_id_to_replace = data.get("doc_id_to_replace")
    force = bool(data.get("force", False))

    if not url:
        return jsonify({"success": False, "error": "URL parameter is required."}), 400
    if not category:
        return jsonify({"success": False, "error": "Category parameter is required."}), 400

    result = index_webpage(
        url=url,
        title=title,
        category=category,
        publisher=publisher,
        publication_date=pub_date,
        effective_date=eff_date,
        doc_id_to_replace=doc_id_to_replace,
        force=force
    )

    status_code = 200 if result.get("success") or result.get("is_duplicate") else 500
    return jsonify(result), status_code


@app.route("/api/admin/document/reindex", methods=["POST"])
@admin_api_required
def admin_reindex():
    data = request.get_json(silent=True) or {}
    doc_id = data.get("doc_id")
    if not doc_id:
        return jsonify({"success": False, "error": "Missing 'doc_id' parameter."}), 400

    result = reindex_document(doc_id)
    status_code = 200 if result.get("success") else 500
    return jsonify(result), status_code


@app.route("/api/admin/document/<path:doc_id>", methods=["DELETE"])
@admin_api_required
def admin_delete_document(doc_id):
    if not doc_id:
        return jsonify({"success": False, "error": "Invalid document ID."}), 400

    result = delete_document_from_kb(doc_id)
    if not result.get("success"):
        if "not found in catalog" in result.get("error", "").lower():
            return jsonify(result), 404
        return jsonify(result), 500
    return jsonify(result), 200


@app.route("/api/admin/test-retrieval", methods=["POST"])
@admin_api_required
def admin_test_retrieval():
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    category = data.get("category")
    top_k = min(int(data.get("top_k", 5)), 20)

    if not query:
        return jsonify({"success": False, "error": "Query cannot be empty."}), 400

    try:
        results = retrieve_formatted_chunks(
            query=query,
            n_results=top_k,
            category=category if category and category != "all" else None
        )
        return jsonify({
            "success": True,
            "query": query,
            "count": len(results),
            "results": results
        })
    except Exception as e:
        logger.exception("Admin test retrieval error")
        return jsonify({"success": False, "error": str(e)}), 500


# =========================================================
# PUBLIC SYSTEM HEALTH & CATALOG
# =========================================================

@app.route("/api/health")
def health():
    stats = get_collection_stats()
    catalog = load_catalog()
    return jsonify({
        "status": "healthy",
        "service": "CampusGPT API",
        "version": "2.1.0",
        "vector_store": stats,
        "catalog_documents_count": len(catalog.get("documents", {}))
    })


@app.route("/api/documents")
def list_documents():
    catalog = load_catalog()
    stats = get_collection_stats()
    return jsonify({
        "catalog": catalog.get("documents", {}),
        "indexed_stats": stats,
        "last_updated": catalog.get("last_updated")
    })


# =========================================================
# DIAGNOSTIC ENDPOINTS (PROTECTED & PATH-VALIDATED)
# =========================================================

@app.route("/api/test-pdf")
@admin_api_required
def test_pdf():
    pdf_param = request.args.get("file", "VITAP_Public_Self_Disclosure.pdf")
    pdf_path = (DOCUMENTS_DIR / Path(pdf_param).name).resolve()

    if not pdf_path.exists():
        return jsonify({"error": f"File not found: {pdf_path.name}"}), 404

    try:
        pages = extract_text_from_pdf(str(pdf_path))
        return jsonify({
            "file": pdf_path.name,
            "total_pages": len(pages),
            "first_page": pages[0] if pages else None
        })
    except Exception as e:
        logger.exception("test-pdf failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/test-chunks")
@admin_api_required
def test_chunks():
    pdf_param = request.args.get("file", "VITAP_Public_Self_Disclosure.pdf")
    pdf_path = (DOCUMENTS_DIR / Path(pdf_param).name).resolve()

    if not pdf_path.exists():
        return jsonify({"error": f"File not found: {pdf_path.name}"}), 404

    try:
        pages = extract_text_from_pdf(str(pdf_path))
        all_chunks = []
        for page in pages:
            chunks = chunk_text(page["text"])
            for chunk in chunks:
                all_chunks.append({
                    "source": pdf_path.name,
                    "page": page["page"],
                    "text": chunk
                })

        return jsonify({
            "total_chunks": len(all_chunks),
            "chunks": all_chunks[:10]
        })
    except Exception as e:
        logger.exception("test-chunks failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/test-embedding")
@admin_api_required
def test_embedding():
    pdf_path = DOCUMENTS_DIR / "VITAP_Public_Self_Disclosure.pdf"
    if not pdf_path.exists():
        return jsonify({"error": "Default PDF not found"}), 404

    try:
        pages = extract_text_from_pdf(str(pdf_path))
        if not pages:
            return jsonify({"error": "No pages found in PDF"}), 500

        chunks = chunk_text(pages[0]["text"])[:2]
        if not chunks:
            return jsonify({"error": "No chunks generated"}), 500

        embeddings = get_embeddings(chunks)
        return jsonify({
            "chunks_sent": len(chunks),
            "embedding_count": len(embeddings),
            "embedding_dimension": len(embeddings[0]) if embeddings else 0,
            "first_embedding": embeddings[0] if embeddings else None
        })
    except Exception as e:
        logger.exception("test-embedding failed")
        return jsonify({"error": str(e)}), 500


# =========================================================
# INDEX DOCUMENT (PROTECTED)
# =========================================================

@app.route("/api/index-document", methods=["POST"])
@admin_api_required
def index_document():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename")
    force = bool(data.get("force", False))
    index_all = bool(data.get("all", False))

    try:
        if index_all or not filename:
            result = index_all_documents(force=force)
            return jsonify({
                "message": "Full document collection ingestion complete",
                "result": result
            }), 200
        else:
            safe_name = secure_filename(filename)
            pdf_path = DOCUMENTS_DIR / safe_name
            if not pdf_path.exists():
                return jsonify({"error": f"Document not found: {safe_name}"}), 404

            result = index_single_pdf(pdf_path, force=force)
            return jsonify({
                "message": f"Document {safe_name} indexed successfully",
                "result": result
            }), 200

    except Exception as error:
        logger.exception("Document indexing failed")
        return jsonify({
            "error": "Document indexing failed",
            "details": str(error)
        }), 500


# =========================================================
# PUBLIC SEARCH
# =========================================================

@app.route("/api/search")
def search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Please provide a query parameter (?q=...)"}), 400

    try:
        category = request.args.get("category")
        top_k = min(int(request.args.get("limit", 5)), 20)

        matches = retrieve_formatted_chunks(
            query=query,
            n_results=top_k,
            category=category
        )

        return jsonify({
            "query": query,
            "count": len(matches),
            "results": matches
        })

    except Exception as e:
        logger.exception("Search error")
        return jsonify({"error": str(e)}), 500


# =========================================================
# STUDENT CHAT (ENRICHED WITH GENUINE SOURCE METADATA)
# =========================================================

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not data or "question" not in data:
        return jsonify({"error": "Please provide a JSON body with a 'question' field."}), 400

    question = str(data["question"]).strip()
    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    try:
        logger.info(f"Incoming chat question: {question}")

        # 1. Conversation History from Flask Session
        history = session.get("chat_history", [])

        # 2. Build retrieval query: Use question directly for standalone queries;
        # attach the immediate prior question only if the student asks an anaphoric follow-up.
        retrieval_query = question
        if history:
            q_lower = question.lower().strip()
            followup_starters = (
                "what about", "how about", "and for", "which one", "tell me more",
                "what are they", "how much", "where can i", "why is that", "what else"
            )
            pronouns = ("it", "its", "they", "them", "these", "those", "this", "that")
            
            is_followup = q_lower.startswith(followup_starters) or any(
                f" {p} " in f" {q_lower} " for p in pronouns
            )

            if is_followup:
                for m in reversed(history):
                    if m.get("role") == "user" and m.get("content"):
                        last_q = m["content"].strip()
                        retrieval_query = f"{last_q} - {question}"
                        break

        # 3. Retrieve relevant chunks
        results = retrieve_documents(retrieval_query, n_results=10)
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        # Load catalog to enrich sources
        catalog = load_catalog().get("documents", {})

        # 4. Format Document Context
        context_parts = []
        for i, doc in enumerate(documents):
            meta = metadatas[i] if i < len(metadatas) else {}
            src = meta.get("source", "Unknown Document")
            page = meta.get("page", 1)
            doc_title = meta.get("title") or catalog.get(src, {}).get("title", src)
            context_parts.append(f"Source: {doc_title} (File: {src})\nPage: {page}\nContent:\n{doc}\n")

        docs_context = "\n".join(context_parts)

        # 5. Format Conversation Context
        conversation_context = ""
        if history:
            conversation_lines = []
            for msg in history:
                role = "Student" if msg.get("role") == "user" else "CampusGPT"
                conversation_lines.append(f"{role}: {msg.get('content', '')}")
            conversation_context = (
                "\n================ PREVIOUS CONVERSATION ================\n"
                + "\n".join(conversation_lines)
                + "\n============== END PREVIOUS CONVERSATION ==============\n"
            )

        # 6. Compose Final Context for LLM
        final_context = (
            conversation_context
            + "\n================ UNIVERSITY DOCUMENTS ================\n"
            + docs_context
            + "\n============== END UNIVERSITY DOCUMENTS ==============\n"
        )

        # 7. Generate Answer
        answer = generate_answer(question, final_context)

        # 8. Update Session History (keep last 10 messages)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        session["chat_history"] = history[-10:]
        session.modified = True

        # 9. Format Deduplicated Genuine Sources for UI
        sources = []
        seen_sources = set()
        for meta in metadatas:
            if not meta:
                continue
            src = meta.get("source", "Unknown source")
            page = meta.get("page", 1)
            key = (src, page)
            if key not in seen_sources:
                seen_sources.add(key)
                cat_entry = catalog.get(src) or catalog.get(meta.get("doc_id")) or {}
                sources.append({
                    "source": src,
                    "page": page,
                    "title": meta.get("title") or cat_entry.get("title") or src.replace("_", " ").replace(".pdf", ""),
                    "source_url": cat_entry.get("source_url") or "https://vitap.ac.in",
                    "category": meta.get("category") or cat_entry.get("category", "General"),
                    "publication_date": cat_entry.get("publication_date"),
                    "effective_date": cat_entry.get("effective_date")
                })

        return jsonify({
            "question": question,
            "answer": answer,
            "sources": sources
        })

    except Exception as e:
        logger.exception("Chat endpoint failed")
        return jsonify({
            "error": "Failed to process chat request",
            "details": str(e)
        }), 500


# =========================================================
# CLEAR CHAT
# =========================================================

@app.route("/api/clear-chat", methods=["POST"])
def clear_chat():
    session.pop("chat_history", None)
    return jsonify({"message": "Chat history cleared successfully."})


# =========================================================
# ATTENDANCE EXTRACTION (SUPPORTING FEATURE)
# =========================================================

@app.route("/api/attendance/extract", methods=["POST"])
def extract_attendance():
    """
    Dedicated endpoint for attendance document or text extraction.
    Supports file uploads (images/PDFs) or JSON with raw text.
    Privacy: Uploaded files are processed in-memory / temporary files and deleted.
    """
    temp_file_path = None
    try:
        # Option A: Uploaded file
        if "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "No file selected."}), 400

            if not allowed_file(file.filename):
                return jsonify({
                    "error": "Invalid file format. Allowed: PDF, PNG, JPG, JPEG, WEBP, TXT"
                }), 400

            safe_name = secure_filename(file.filename)
            temp_file_path = UPLOAD_FOLDER / f"temp_{os.getpid()}_{safe_name}"
            file.save(str(temp_file_path))

            result = process_attendance_file(temp_file_path)
            return jsonify(result)

        # Option B: JSON payload with raw text
        data = request.get_json(silent=True)
        if data and "text" in data:
            raw_text = str(data["text"]).strip()
            if not raw_text:
                return jsonify({"error": "Text payload is empty."}), 400
            result = parse_attendance_text(raw_text)
            return jsonify(result)

        return jsonify({
            "error": "Provide either a 'file' multipart upload or a JSON body with 'text'."
        }), 400

    except Exception as e:
        logger.exception("Attendance extraction failed")
        return jsonify({"error": f"Attendance extraction failed: {str(e)}"}), 500

    finally:
        # Cleanup temporary uploaded file to ensure student privacy
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception as clean_err:
                logger.warning(f"Failed to delete temporary file {temp_file_path}: {clean_err}")


# =========================================================
# APPLICATION ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() in ["1", "true"]
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
