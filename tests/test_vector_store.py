import pytest
from services.vector_store import (
    add_documents,
    delete_document,
    search_documents,
    get_collection_stats,
    sanitize_id_part
)


def test_sanitize_id_part():
    assert sanitize_id_part("Test File (1).pdf") == "Test_File__1_.pdf"
    assert sanitize_id_part("SimpleName") == "SimpleName"


def test_add_and_search_documents():
    chunks = [
        {"source": "test_doc.pdf", "page": 1, "text": "VIT-AP Engineering admissions overview.", "category": "Admissions"},
        {"source": "test_doc.pdf", "page": 2, "text": "Tuition fees for Category 1 are Rs 198000.", "category": "Fees"}
    ]
    # Simulated 1024-dimensional embeddings
    embeddings = [[0.01] * 1024, [0.05] * 1024]

    # Add documents (idempotent)
    add_documents(chunks, embeddings, replace_existing=True)

    stats = get_collection_stats()
    assert stats["total_chunks"] >= 2
    assert "test_doc.pdf" in stats["documents"]

    # Search with dummy query vector
    query_vector = [0.01] * 1024
    results = search_documents(query_vector, n_results=2)
    assert len(results["documents"][0]) > 0

    # Clean up test doc
    delete_document("test_doc.pdf")
    post_stats = get_collection_stats()
    assert "test_doc.pdf" not in post_stats["documents"]
