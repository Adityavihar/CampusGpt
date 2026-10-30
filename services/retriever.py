import logging
from typing import Optional
from services.embeddings import get_query_embedding
from services.vector_store import search_documents

logger = logging.getLogger(__name__)


def interpret_distance(distance: Optional[float]) -> str:
    """Provide human-friendly interpretation of ChromaDB cosine distance."""
    if distance is None:
        return "Not available"
    if distance < 0.5:
        return "Very High Relevance (Strong semantic match)"
    elif distance < 0.85:
        return "High Relevance (Direct topic match)"
    elif distance < 1.2:
        return "Moderate Relevance (Related contextual match)"
    else:
        return "Low / Broad Relevance (General context)"


def retrieve_documents(query: str, n_results: int = 5, category: Optional[str] = None) -> dict:
    """
    Retrieve the most relevant document chunks from ChromaDB for a query.

    Args:
        query: User question or search phrase.
        n_results: Number of chunks to retrieve (default: 5).
        category: Optional category filter (e.g. 'Admissions', 'Hostel').

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
    with rich metadata, source attribution, and distance interpretation.
    """
    raw_results = retrieve_documents(query=query, n_results=n_results, category=category)
    
    docs = raw_results.get("documents", [[]])[0]
    metas = raw_results.get("metadatas", [[]])[0]
    dists = raw_results.get("distances", [[]])[0]

    # Load catalog for rich metadata enrichment
    catalog_docs = {}
    try:
        from services.ingestion_service import load_catalog
        catalog_docs = load_catalog().get("documents", {})
    except Exception:
        pass

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
        src = meta.get("source", "Unknown Document")
        doc_id = meta.get("doc_id", src)
        
        # Cross-reference catalog metadata
        cat_meta = catalog_docs.get(doc_id) or catalog_docs.get(src) or {}

        doc_title = meta.get("title") or cat_meta.get("title") or src.replace("_", " ").replace(".pdf", "")
        doc_category = meta.get("category") or cat_meta.get("category", "Other")
        source_url = cat_meta.get("source_url") or "https://vitap.ac.in"
        effective_date = cat_meta.get("effective_date")
        publication_date = cat_meta.get("publication_date")

        items.append({
            "text": text,
            "source": src,
            "doc_id": doc_id,
            "title": doc_title,
            "page": meta.get("page", 1),
            "category": doc_category,
            "source_url": source_url,
            "publication_date": publication_date,
            "effective_date": effective_date,
            "distance": round(dist, 4) if dist is not None else None,
            "distance_meaning": interpret_distance(dist)
        })

    return items
