import os
import logging
from pathlib import Path
from flask import Flask, jsonify, request, render_template, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("campusgpt")

from services.embeddings import get_embeddings
from services.document_parser import extract_text_from_pdf
from services.chunking import chunk_text
from services.vector_store import add_documents, get_collection_stats
from services.retriever import retrieve_documents, retrieve_formatted_chunks
from services.llm import generate_answer
from services.ingestion_service import index_single_pdf, index_all_documents, load_catalog
from services.attendance_service import process_attendance_file, parse_attendance_text

app = Flask(__name__)

# Security: Read session secret key from environment
app.secret_key = os.getenv("SECRET_KEY", "campusgpt-secure-session-key-vitap")

# Upload configurations
UPLOAD_FOLDER = Path("data/uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload limit
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".txt"}


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# HEALTH & STATUS
# =========================================================

@app.route("/api/health")
def health():
    stats = get_collection_stats()
    catalog = load_catalog()
    return jsonify({
        "status": "healthy",
        "service": "CampusGPT API",
        "version": "2.0.0",
        "vector_store": stats,
        "catalog_documents_count": len(catalog.get("documents", {}))
    })


# =========================================================
# DOCUMENT CATALOG & LISTING
# =========================================================

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
# TEST PDF (BACKWARD COMPATIBLE)
# =========================================================

@app.route("/api/test-pdf")
def test_pdf():
    pdf_path = request.args.get("file", "data/documents/VITAP_Public_Self_Disclosure.pdf")
    if not os.path.exists(pdf_path):
        return jsonify({"error": f"File not found: {pdf_path}"}), 404

    try:
        pages = extract_text_from_pdf(pdf_path)
        return jsonify({
            "file": Path(pdf_path).name,
            "total_pages": len(pages),
            "first_page": pages[0] if pages else None
        })
    except Exception as e:
        logger.exception("test-pdf failed")
        return jsonify({"error": str(e)}), 500


# =========================================================
# TEST CHUNKS (BACKWARD COMPATIBLE)
# =========================================================

@app.route("/api/test-chunks")
def test_chunks():
    pdf_path = request.args.get("file", "data/documents/VITAP_Public_Self_Disclosure.pdf")
    if not os.path.exists(pdf_path):
        return jsonify({"error": f"File not found: {pdf_path}"}), 404

    try:
        pages = extract_text_from_pdf(pdf_path)
        all_chunks = []
        for page in pages:
            chunks = chunk_text(page["text"])
            for chunk in chunks:
                all_chunks.append({
                    "source": Path(pdf_path).name,
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


# =========================================================
# TEST EMBEDDING (BACKWARD COMPATIBLE)
# =========================================================

@app.route("/api/test-embedding")
def test_embedding():
    pdf_path = "data/documents/VITAP_Public_Self_Disclosure.pdf"
    if not os.path.exists(pdf_path):
        return jsonify({"error": "Default PDF not found"}), 404

    try:
        pages = extract_text_from_pdf(pdf_path)
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
# INDEX DOCUMENT / INGESTION
# =========================================================

@app.route("/api/index-document", methods=["POST"])
def index_document():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename")
    force = bool(data.get("force", False))
    index_all = bool(data.get("all", False))

    try:
        if index_all or not filename:
            # If all=True or no single file requested, index the full documents directory
            result = index_all_documents(force=force)
            return jsonify({
                "message": "Full document collection ingestion complete",
                "result": result
            }), 200
        else:
            safe_name = secure_filename(filename)
            pdf_path = Path("data/documents") / safe_name
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
# SEARCH
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
# CHAT
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

        # 4. Format Document Context
        context_parts = []
        for i, doc in enumerate(documents):
            meta = metadatas[i] if i < len(metadatas) else {}
            source = meta.get("source", "Unknown Document")
            page = meta.get("page", "Unknown")
            context_parts.append(f"Source: {source}\nPage: {page}\nContent:\n{doc}\n")

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

        # 9. Format Deduplicated Sources for UI
        sources = []
        seen_sources = set()
        for meta in metadatas:
            if not meta:
                continue
            src = meta.get("source", "Unknown source")
            page = meta.get("page", "Unknown")
            key = (src, page)
            if key not in seen_sources:
                seen_sources.add(key)
                sources.append({"source": src, "page": page})

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
