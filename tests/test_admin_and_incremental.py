import pytest
import io
import json
from unittest.mock import patch, MagicMock
from app import app
from services.content_extractor import is_safe_url, extract_text_from_txt, CleanHTMLParser
from services.auth import verify_admin_password, get_admin_password


@pytest.fixture(autouse=True)
def preserve_catalog():
    import json
    from pathlib import Path
    from services.ingestion_service import CATALOG_PATH, load_catalog, save_catalog, DOCUMENTS_DIR
    original_catalog = load_catalog()
    existing_files = set(DOCUMENTS_DIR.glob("*"))
    yield
    # Restore original catalog
    save_catalog(original_catalog)
    # Remove any test files created during test runs
    for f in DOCUMENTS_DIR.glob("*"):
        if f not in existing_files:
            try:
                f.unlink()
            except Exception:
                pass


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test-secret-key"
    with app.test_client() as client:
        yield client


@pytest.fixture
def auth_client():
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test-secret-key"
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["is_admin"] = True
        yield client


# =========================================================
# 1. ADMIN AUTHENTICATION & ACCESS CONTROL
# =========================================================

def test_admin_password_verification():
    admin_pw = get_admin_password()
    assert verify_admin_password(admin_pw) is True
    assert verify_admin_password("wrong-password-1234") is False
    assert verify_admin_password("") is False
    assert verify_admin_password(None) is False


def test_admin_pages_require_auth(client):
    # Unauthenticated should redirect to login
    response = client.get("/admin/knowledge-base")
    assert response.status_code == 302
    assert "/admin/login" in response.headers["Location"]


def test_admin_api_requires_auth(client):
    # Unauthenticated should return 401
    endpoints = [
        ("GET", "/api/admin/stats"),
        ("GET", "/api/admin/documents"),
        ("POST", "/api/admin/upload"),
        ("POST", "/api/admin/webpage/preview"),
        ("POST", "/api/admin/webpage/index"),
        ("POST", "/api/admin/document/reindex"),
        ("DELETE", "/api/admin/document/test-doc"),
        ("POST", "/api/admin/test-retrieval"),
        ("GET", "/api/test-pdf"),
        ("GET", "/api/test-chunks"),
        ("GET", "/api/test-embedding"),
        ("POST", "/api/index-document"),
    ]
    for method, path in endpoints:
        if method == "GET":
            res = client.get(path)
        elif method == "POST":
            res = client.post(path, json={})
        elif method == "DELETE":
            res = client.delete(path)
        assert res.status_code == 401, f"Expected 401 for {method} {path}, got {res.status_code}"


def test_admin_login_flow(client):
    # Bad password
    res = client.post("/admin/login", data={"password": "incorrect-password"})
    assert res.status_code == 401
    assert b"Invalid administrator password" in res.data

    # Good password
    res = client.post("/admin/login", data={"password": get_admin_password()}, follow_redirects=False)
    assert res.status_code == 302
    assert "/admin/knowledge-base" in res.headers["Location"]


def test_admin_header_api_key_auth(client):
    # Authenticated via X-Admin-Key header
    res = client.get("/api/admin/stats", headers={"X-Admin-Key": get_admin_password()})
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True


def test_admin_logout(auth_client):
    res = auth_client.get("/admin/logout", follow_redirects=False)
    assert res.status_code == 302
    assert "/admin/login" in res.headers["Location"]

    # Now verify session is cleared
    res2 = auth_client.get("/api/admin/stats")
    assert res2.status_code == 401


# =========================================================
# 2. SSRF PROTECTION & URL VALIDATION
# =========================================================

def test_ssrf_prohibited_destinations():
    bad_urls = [
        "http://localhost:5000/api",
        "http://127.0.0.1:8000/secret",
        "http://0.0.0.0:80",
        "http://169.254.169.254/latest/meta-data",
        "http://192.168.1.1/admin",
        "http://10.0.0.5/api",
        "http://172.16.0.10/status",
        "ftp://vitap.ac.in/resource.pdf",
        "file:///etc/passwd",
        "file://C:/Windows/win.ini",
        "javascript:alert(1)",
        "",
        "not_a_valid_url"
    ]
    for url in bad_urls:
        safe, err = is_safe_url(url)
        assert not safe, f"Expected {url} to be blocked by SSRF protection, but was allowed"
        assert err is not None


def test_clean_html_parser():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head><title>VIT-AP Exam Guidelines</title><script>alert('bad');</script></head>
    <body>
        <nav><a href="/">Home</a><a href="/login">Login</a></nav>
        <header>Header content</header>
        <main>
            <h1>End Term Examination Rules</h1>
            <p>Students must carry their physical ID cards to the exam hall.</p>
            <p>No electronic devices or smartwatches are permitted.</p>
        </main>
        <footer>Copyright 2026 VIT-AP</footer>
    </body>
    </html>
    """
    parser = CleanHTMLParser()
    parser.feed(html_content)
    text = parser.get_text()
    title = parser.get_title()

    assert title == "VIT-AP Exam Guidelines"
    assert "End Term Examination Rules" in text
    assert "Students must carry their physical ID cards" in text
    assert "alert('bad')" not in text
    assert "Header content" not in text
    assert "Copyright 2026" not in text


# =========================================================
# 3. CONTENT EXTRACTOR HELPERS
# =========================================================

def test_extract_text_from_txt_encodings():
    utf8_data = "VIT-AP Library is open from 8:00 AM to 11:00 PM on weekdays.".encode("utf-8")
    assert "Library is open" in extract_text_from_txt(utf8_data)

    latin1_data = "Fee structure per semester is \xa31000 or \x80500.".encode("latin-1")
    extracted = extract_text_from_txt(latin1_data)
    assert "Fee structure" in extracted


# =========================================================
# 4. ADMIN INGESTION ENDPOINTS & VALIDATION
# =========================================================

def test_admin_upload_rejects_unsupported_extensions(auth_client):
    data = {
        "file": (io.BytesIO(b"malicious script"), "hack.exe"),
        "category": "Academics and Courses"
    }
    res = auth_client.post("/api/admin/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    json_data = res.get_json()
    assert "Invalid file type" in json_data["error"]


def test_admin_upload_requires_category(auth_client):
    data = {
        "file": (io.BytesIO(b"Valid test text content"), "notice.txt"),
        "category": ""
    }
    res = auth_client.post("/api/admin/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    json_data = res.get_json()
    assert "Category parameter is required" in json_data["error"]


@patch("services.ingestion_service.get_embeddings")
@patch("services.ingestion_service.add_documents")
def test_admin_upload_valid_txt(mock_add_docs, mock_embed, auth_client):
    import uuid
    mock_embed.return_value = [[0.01] * 1024]
    mock_add_docs.return_value = None

    unique_content = f"VIT-AP Placement Cell contact: placements@vitap.ac.in. Code: {uuid.uuid4().hex}".encode("utf-8")
    data = {
        "file": (io.BytesIO(unique_content), f"placements_{uuid.uuid4().hex[:6]}.txt"),
        "category": "Placements",
        "title": "VIT-AP Placement Highlights 2026",
        "publisher": "Career Development Centre"
    }
    res = auth_client.post("/api/admin/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    json_data = res.get_json()
    assert json_data["success"] is True
    assert json_data["chunks_stored"] >= 1
    assert "doc_id" in json_data


@patch("services.ingestion_service.get_embeddings")
@patch("services.ingestion_service.add_documents")
def test_admin_duplicate_detection(mock_add_docs, mock_embed, auth_client):
    import uuid
    mock_embed.return_value = [[0.01] * 1024]
    mock_add_docs.return_value = None

    nonce = uuid.uuid4().hex
    content = f"Unique notice text for duplicate test {nonce}: All hostelers must report by 9:00 PM.".encode("utf-8")
    data1 = {
        "file": (io.BytesIO(content), f"hostel_rules_{nonce[:4]}.txt"),
        "category": "Hostel",
        "title": "Hostel Rules"
    }
    res1 = auth_client.post("/api/admin/upload", data=data1, content_type="multipart/form-data")
    assert res1.status_code == 200

    # Upload identical content
    data2 = {
        "file": (io.BytesIO(content), f"hostel_rules_{nonce[:4]}_copy.txt"),
        "category": "Hostel",
        "title": "Hostel Rules Copy"
    }
    res2 = auth_client.post("/api/admin/upload", data=data2, content_type="multipart/form-data")
    assert res2.status_code == 200
    json_data2 = res2.get_json()
    assert json_data2.get("is_duplicate") is True
    assert "already been indexed" in json_data2.get("message", "").lower()


def test_admin_webpage_preview_ssrf_blocked(auth_client):
    res = auth_client.post("/api/admin/webpage/preview", json={"url": "http://127.0.0.1:8080/secret"})
    assert res.status_code == 400
    assert "SSRF Protection" in res.get_json()["error"]


# =========================================================
# 5. RETRIEVAL & DELETION SAFETY
# =========================================================

@patch("services.retriever.retrieve_documents")
def test_admin_test_retrieval(mock_retrieve, auth_client):
    mock_retrieve.return_value = {
        "documents": [["VIT-AP central library has 50,000+ volumes."]],
        "metadatas": [[{
            "doc_id": "test_lib_1",
            "source": "library.pdf",
            "page": 3,
            "category": "Library",
            "title": "Library Guide"
        }]],
        "distances": [[0.08]]
    }

    res = auth_client.post("/api/admin/test-retrieval", json={
        "query": "Where is the library located?",
        "category": "Library"
    })
    assert res.status_code == 200
    json_data = res.get_json()
    assert json_data["success"] is True
    assert json_data["count"] == 1
    chunk = json_data["results"][0]
    assert "Very High" in chunk["distance_meaning"]
    assert chunk["page"] == 3


@patch("services.ingestion_service.delete_document")
def test_admin_delete_document_safety(mock_delete, auth_client):
    mock_delete.return_value = True

    # Attempt delete of non-existent
    res = auth_client.delete("/api/admin/document/non_existent_doc_id_9999")
    assert res.status_code == 404
    assert res.get_json()["success"] is False


def test_admin_upload_rejects_oversized_file(auth_client):
    # Simulate file exceeding 25MB MAX_CONTENT_LENGTH
    oversized = io.BytesIO(b"x" * (26 * 1024 * 1024))
    res = auth_client.post("/api/admin/upload", data={
        "file": (oversized, "huge_file.pdf"),
        "category": "Academics and Courses"
    }, content_type="multipart/form-data")
    assert res.status_code == 413 or res.status_code == 400


@patch("services.ingestion_service.get_embeddings")
def test_embedding_failure_handled_gracefully(mock_embed, auth_client):
    mock_embed.side_effect = RuntimeError("Jina API rate limit exceeded")
    
    data = {
        "file": (io.BytesIO(b"Some text for rate limit test"), "test_fail.txt"),
        "category": "Academics and Courses"
    }
    res = auth_client.post("/api/admin/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 500
    json_data = res.get_json()
    assert json_data["success"] is False
    assert "Jina API rate limit" in json_data["error"]


@patch("services.ingestion_service.get_embeddings")
@patch("services.ingestion_service.add_documents")
@patch("services.ingestion_service.delete_document")
def test_document_replacement_workflow(mock_del, mock_add, mock_embed, auth_client):
    import uuid
    mock_embed.return_value = [[0.01] * 1024]
    mock_add.return_value = None
    mock_del.return_value = 1

    doc_id = f"test_replace_{uuid.uuid4().hex[:6]}"
    
    # Ingest v1
    res1 = auth_client.post("/api/admin/upload", data={
        "file": (io.BytesIO(b"Version 1 syllabus content: Modules 1 to 3."), "syllabus.txt"),
        "category": "Academics and Courses",
        "title": "Data Structures Syllabus"
    }, content_type="multipart/form-data")
    assert res1.status_code == 200
    orig_id = res1.get_json()["doc_id"]

    # Ingest v2 replacing v1
    res2 = auth_client.post("/api/admin/upload", data={
        "file": (io.BytesIO(b"Version 2 updated syllabus: Modules 1 to 5."), "syllabus_v2.txt"),
        "category": "Academics and Courses",
        "title": "Data Structures Syllabus Revised",
        "doc_id_to_replace": orig_id
    }, content_type="multipart/form-data")
    assert res2.status_code == 200
    json_v2 = res2.get_json()
    assert json_v2["success"] is True
    assert json_v2["version"] in (2, "2.0")
    assert mock_add.called
    # Verify replacement was requested to overwrite existing chunks
    _, kwargs = mock_add.call_args
    assert kwargs.get("replace_existing") is True


@patch("app.retrieve_documents")
@patch("app.generate_answer")
def test_student_chatbot_with_new_knowledge_and_followup(mock_gen, mock_ret, client):
    # Turn 1: Initial Question
    mock_ret.return_value = {
        "documents": [["VIT-AP central library operates from 8:00 AM to 11:00 PM."]],
        "metadatas": [[{
            "doc_id": "lib_timings",
            "source": "Library_Timings_2026.pdf",
            "page": 1,
            "title": "Central Library Timings 2026",
            "category": "Library"
        }]],
        "distances": [[0.05]]
    }
    mock_gen.return_value = "The central library is open from 8:00 AM to 11:00 PM on weekdays."

    res1 = client.post("/api/chat", json={"question": "What are the library timings?"})
    assert res1.status_code == 200
    data1 = res1.get_json()
    assert "8:00 AM" in data1["answer"]
    assert len(data1["sources"]) == 1
    assert data1["sources"][0]["title"] == "Central Library Timings 2026"
    assert data1["sources"][0]["page"] == 1

    # Turn 2: Conversational Follow-up (anaphoric)
    mock_ret.return_value = {
        "documents": [["On Saturdays and Sundays, the library closes at 6:00 PM."]],
        "metadatas": [[{
            "doc_id": "lib_timings",
            "source": "Library_Timings_2026.pdf",
            "page": 1,
            "title": "Central Library Timings 2026",
            "category": "Library"
        }]],
        "distances": [[0.07]]
    }
    mock_gen.return_value = "On weekends, it is open until 6:00 PM."

    res2 = client.post("/api/chat", json={"question": "What about on weekends?"})
    assert res2.status_code == 200
    data2 = res2.get_json()
    assert "weekends" in data2["answer"].lower()
    # Verify retrieval query enriched with prior question
    assert mock_ret.call_count == 2
    last_query_called = mock_ret.call_args[0][0]
    assert "What are the library timings?" in last_query_called

