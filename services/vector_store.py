import re
import logging
from typing import Optional
import chromadb

logger = logging.getLogger(__name__)

# =========================================================
# CHROMA CLIENT & COLLECTION
# =========================================================

CHROMA_PATH = "vectorstore/chroma"

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name="campus_documents",
    metadata={"hnsw:space": "cosine"}
)


def sanitize_id_part(value: str) -> str:
    """
    Sanitize strings for ChromaDB ID generation.
    """
    clean = re.sub(r"[^a-zA-Z0-9_\-.]", "_", str(value))
    return clean.strip("_")


# =========================================================
# ADD DOCUMENTS (IDEMPOTENT & STRUCTURED)
# =========================================================

def add_documents(chunks: list[dict], embeddings: list[list[float]], replace_existing: bool = True):
    """
    Store document chunks and embeddings in ChromaDB idempotently.

    Each chunk dict must have:
      - 'source': str (e.g. 'VITAP_Public_Self_Disclosure.pdf')
      - 'page': int
      - 'text': str
      - optional 'category': str
      - optional 'chunk_index': int (index within page or document)

    Args:
        chunks: List of chunk dictionaries.
        embeddings: List of embedding vectors matching chunks.
        replace_existing: If True, removes previous records for the same source before upserting.
    """
    if not chunks:
        return

    if len(chunks) != len(embeddings):
        raise ValueError(f"Mismatch between chunks count ({len(chunks)}) and embeddings count ({len(embeddings)})")

    # If replace_existing is requested, delete old chunks for each unique source in this batch
    if replace_existing:
        unique_sources = {chunk["source"] for chunk in chunks if "source" in chunk}
        for src in unique_sources:
            try:
                collection.delete(where={"source": src})
                logger.info(f"Purged previous chunks for source: {src}")
            except Exception as e:
                logger.warning(f"Note on clearing source {src}: {e}")

    ids = []
    documents = []
    metadatas = []

    # Track chunk index per page for stable deterministic IDs
    page_chunk_counters: dict[tuple[str, int], int] = {}

    for i, chunk in enumerate(chunks):
        source = chunk.get("source", "unknown_source")
        page = int(chunk.get("page", 1))
        category = chunk.get("category", "General")
        
        counter_key = (source, page)
        chunk_idx = chunk.get("chunk_index")
        if chunk_idx is None:
            chunk_idx = page_chunk_counters.get(counter_key, 0)
            page_chunk_counters[counter_key] = chunk_idx + 1

        clean_source = sanitize_id_part(source)
        chunk_id = f"{clean_source}_p{page}_c{chunk_idx}"

        ids.append(chunk_id)
        documents.append(chunk["text"])
        metadatas.append({
            "source": source,
            "page": page,
            "category": category,
            "chunk_index": int(chunk_idx)
        })

    logger.info(f"Upserting {len(documents)} chunks to ChromaDB collection 'campus_documents'")

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    logger.info(f"Upsert complete. Total collection count: {collection.count()}")


# =========================================================
# DELETE DOCUMENT
# =========================================================

def delete_document(source: str) -> int:
    """
    Remove all chunks corresponding to a specific document source.
    """
    try:
        existing = collection.get(where={"source": source})
        count = len(existing.get("ids", []))
        if count > 0:
            collection.delete(where={"source": source})
            logger.info(f"Deleted {count} chunks for source: {source}")
        return count
    except Exception as e:
        logger.error(f"Error deleting document {source}: {e}")
        raise


# =========================================================
# SEARCH DOCUMENTS
# =========================================================

def search_documents(query_embedding: list[float], n_results: int = 5, category: Optional[str] = None) -> dict:
    """
    Search ChromaDB using a query embedding with optional category filter.
    """
    total = collection.count()
    if total == 0:
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

    actual_n = min(n_results, total)
    where_filter = {"category": category} if category else None

    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": actual_n,
        "include": ["documents", "metadatas", "distances"]
    }
    if where_filter:
        kwargs["where"] = where_filter

    results = collection.query(**kwargs)
    return results


# =========================================================
# STATS & CATALOG
# =========================================================

def get_collection_stats() -> dict:
    """
    Return statistics about the ChromaDB index.
    """
    total = collection.count()
    sources_summary: dict[str, int] = {}
    
    if total > 0:
        all_meta = collection.get(include=["metadatas"])
        for meta in all_meta.get("metadatas", []):
            if meta:
                src = meta.get("source", "Unknown")
                sources_summary[src] = sources_summary.get(src, 0) + 1

    return {
        "total_chunks": total,
        "indexed_documents_count": len(sources_summary),
        "documents": sources_summary
    }
