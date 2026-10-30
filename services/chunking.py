import re

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 50) -> list[str]:
    """
    Split text into clean, semantically coherent chunks.
    Respects natural boundaries (numbered sections '1. ', '2. ', '3. ',
    paragraph breaks '\n\n', or logical line breaks) so that complete
    topics (e.g. scholarship schemes) remain unbroken.
    Falls back safely to sliding window for unstructured text blocks.

    Args:
        text: Input string.
        chunk_size: Target maximum character length per chunk (default: 800).
        overlap: Character overlap between consecutive chunks if sliding window is needed (default: 50).

    Returns:
        list[str]: Cleaned, non-empty text chunks.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    lines = text.split("\n")
    chunks = []
    curr = []
    curr_len = 0
    min_split_len = max(200, chunk_size // 3)

    for line in lines:
        line_clean = line.strip()
        is_new_section = bool(re.match(r"^\d+\.\s+[A-Z]", line_clean))

        if is_new_section and curr_len >= min_split_len:
            chunk_str = "\n".join(curr).strip()
            if chunk_str:
                chunks.append(chunk_str)
            curr = [line]
            curr_len = len(line)
        elif curr_len + len(line) > chunk_size and curr_len >= min_split_len:
            chunk_str = "\n".join(curr).strip()
            if chunk_str:
                chunks.append(chunk_str)
            curr = [line]
            curr_len = len(line)
        else:
            curr.append(line)
            curr_len += len(line) + 1

    if curr:
        remainder = "\n".join(curr).strip()
        # If remainder is tiny (< 100 chars) and previous chunk exists, merge to avoid dangling fragments
        if remainder:
            if len(remainder) < 100 and chunks:
                chunks[-1] = chunks[-1] + "\n" + remainder
            else:
                chunks.append(remainder)

    # Fallback to character sliding window if section/line chunking produced only 1 oversized chunk
    if len(chunks) == 1 and len(chunks[0]) > chunk_size * 1.5:
        return _character_sliding_chunk(text, chunk_size=chunk_size, overlap=overlap)

    return chunks


def _character_sliding_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Fallback character-based sliding window."""
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 10)

    step = max(1, chunk_size - overlap)
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start += step

    return chunks
