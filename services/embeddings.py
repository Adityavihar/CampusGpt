import os
import time
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_URL = "https://api.jina.ai/v1/embeddings"
MODEL_NAME = "jina-embeddings-v3"
EMBEDDING_DIMENSIONS = 1024
DEFAULT_BATCH_SIZE = 32


def _get_api_key():
    key = os.getenv("JINA_API_KEY")
    if not key:
        raise ValueError("JINA_API_KEY is not set in the environment or .env file")
    return key


def get_embeddings(texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE) -> list[list[float]]:
    """
    Generate passage embeddings for a list of texts in batches.
    Maintains exact 1:1 order alignment with input texts.
    """
    if not texts:
        return []

    api_key = _get_api_key()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        payload = {
            "model": MODEL_NAME,
            "task": "retrieval.passage",
            "input": batch,
            "embedding_type": "float",
            "dimensions": EMBEDDING_DIMENSIONS
        }

        # Retry logic for network timeouts or transient API errors
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                response = requests.post(JINA_URL, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                batch_embeddings = [item["embedding"] for item in result["data"]]
                all_embeddings.extend(batch_embeddings)
                last_error = None
                break
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Embedding batch {i // batch_size + 1} attempt {attempt + 1} failed: {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        if last_error:
            raise RuntimeError(
                f"Failed to generate embeddings for batch starting at index {i}: {last_error}"
            ) from last_error

    return all_embeddings


def get_query_embedding(query: str) -> list[float]:
    """
    Generate a query embedding for a search or retrieval question.
    """
    if not query or not query.strip():
        raise ValueError("Query string cannot be empty")

    api_key = _get_api_key()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": MODEL_NAME,
        "task": "retrieval.query",
        "input": [query.strip()],
        "embedding_type": "float",
        "dimensions": EMBEDDING_DIMENSIONS
    }

    response = requests.post(JINA_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    result = response.json()
    return result["data"][0]["embedding"]
