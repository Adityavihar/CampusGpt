import pytest
import json
from unittest.mock import patch
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"CampusGPT" in response.data


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "vector_store" in data


def test_documents_endpoint(client):
    response = client.get("/api/documents")
    assert response.status_code == 200
    data = response.get_json()
    assert "catalog" in data


def test_search_missing_query(client):
    response = client.get("/api/search")
    assert response.status_code == 400


def test_chat_empty_body(client):
    response = client.post("/api/chat", json={})
    assert response.status_code == 400


@patch("app.retrieve_documents")
@patch("app.generate_answer")
def test_chat_successful_turn(mock_generate, mock_retrieve, client):
    mock_retrieve.return_value = {
        "documents": [["VIT-AP requires minimum 75% attendance."]],
        "metadatas": [[{"source": "VITAP_Academic_Regulations_and_FFCS.pdf", "page": 2}]],
        "distances": [[0.1]]
    }
    mock_generate.return_value = "Students must maintain 75% attendance [VITAP_Academic_Regulations_and_FFCS.pdf, Page 2]."

    payload = {"question": "What is the attendance policy?"}
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert "attendance" in data["answer"].lower()
    assert len(data["sources"]) == 1
    assert data["sources"][0]["page"] == 2


def test_clear_chat(client):
    response = client.post("/api/clear-chat")
    assert response.status_code == 200
    assert response.get_json()["message"] == "Chat history cleared successfully."


def test_attendance_endpoint_with_text(client):
    payload = {
        "text": "Data Structures: 28 / 35\nOperating Systems: 22 / 32"
    }
    response = client.post("/api/attendance/extract", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["records_count"] == 2
    assert "overall" in data
