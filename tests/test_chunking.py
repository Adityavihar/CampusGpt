import pytest
from services.chunking import chunk_text


def test_chunk_empty_text():
    assert chunk_text("") == []
    assert chunk_text("   ") == []
    assert chunk_text(None) == []


def test_chunk_short_text():
    text = "Short text under 500 characters."
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_long_text_overlap():
    text = "A" * 1000
    chunks = chunk_text(text, chunk_size=400, overlap=100)
    assert len(chunks) >= 3
    for chunk in chunks:
        assert len(chunk) <= 400


def test_chunk_overlap_safety():
    # If overlap >= chunk_size, function should safely handle and not loop infinitely
    text = "Testing boundary conditions for overlap" * 10
    chunks = chunk_text(text, chunk_size=50, overlap=60)
    assert len(chunks) > 0
