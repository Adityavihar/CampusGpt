import logging
from typing import Optional
from services.embeddings import get_query_embedding
from services.vector_store import search_documents

logger = logging.getLogger(__name__)


def retrieve_documents(query: str, n_results: int = 5, category: Optional[str] = None) -> dict:
    """
    Retrieve the most relevant document chunks from ChromaDB for a query.

    Args:
        query: User question or search phrase.
        n_results: Number of chunks to retrieve (default: 5).
        category: Optional category filter (e.g. 'Programs', 'Admissions').

    Returns:
        dict: Standard ChromaDB query result dictionary containing
              'documents', 'metadatas', 'distances', and 'ids'.
    """
    if not query or not query.strip():
        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "ids": [[]]
        }

    # Generate query embedding
    query_embedding = get_query_embedding(query.strip())

    # Query persistent vector store
    results = search_documents(
        query_embedding=query_embedding,
        n_results=n_results,
        category=category
    )

    if not isinstance(results, dict):
        raise TypeError(f"ChromaDB returned unexpected type: {type(results)}")

    for key in ["documents", "metadatas", "distances"]:
        if key not in results:
            results[key] = [[]]

    return results


def retrieve_formatted_chunks(
    query: str,
    n_results: int = 5,
    category: Optional[str] = None,
    max_distance: Optional[float] = None
) -> list[dict]:
    """
    Retrieve documents and return a clean, deduplicated list of chunk dicts
    with source attribution and relevance distance.
    """
    raw_results = retrieve_documents(query=query, n_results=n_results, category=category)
    
    docs = raw_results.get("documents", [[]])[0]
    metas = raw_results.get("metadatas", [[]])[0]
    dists = raw_results.get("distances", [[]])[0]

    items = []
    seen_texts = set()

    for i in range(len(docs)):
        text = docs[i].strip() if docs[i] else ""
        if not text:
            continue

        # Prevent exact duplicate chunks in context
        first_60 = text[:60]
        if first_60 in seen_texts:
            continue
        seen_texts.add(first_60)

        dist = dists[i] if i < len(dists) else None
        if max_distance is not None and dist is not None and dist > max_distance:
            continue

        meta = metas[i] if i < len(metas) else {}
        items.append({
            "text": text,
            "source": meta.get("source", "Unknown Document"),
            "page": meta.get("page", "Unknown"),
            "category": meta.get("category", "General"),
            "distance": dist
        })

    return items
