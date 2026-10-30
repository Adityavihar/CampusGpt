import pytest
from unittest.mock import patch
from services.retriever import retrieve_documents, retrieve_formatted_chunks


def test_empty_query():
    res = retrieve_documents("")
    assert res["documents"] == [[]]


@patch("services.retriever.get_query_embedding")
@patch("services.retriever.search_documents")
def test_retrieve_formatted_chunks(mock_search, mock_embed):
    mock_embed.return_value = [0.1] * 1024
    mock_search.return_value = {
        "documents": [["VIT-AP B.Tech programs in CSE", "Duplicate B.Tech programs in CSE"]],
        "metadatas": [[{"source": "programs.pdf", "page": 1, "category": "Programs"}, {"source": "programs.pdf", "page": 1, "category": "Programs"}]],
        "distances": [[0.15, 0.15]]
    }

    chunks = retrieve_formatted_chunks("What B.Tech programs are offered?", n_results=5)
    assert len(chunks) == 2
    assert chunks[0]["source"] == "programs.pdf"
    assert chunks[0]["page"] == 1
