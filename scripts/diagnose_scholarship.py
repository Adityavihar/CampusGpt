import sys
import os
sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import chromadb
from services.document_parser import extract_text_from_pdf
from services.retriever import retrieve_documents
from services.llm import generate_answer

def run_diagnostics():
    query = "What scholarships are available for top-ranking students?"
    print("=" * 60)
    print("DIAGNOSTIC QUERY:", query)
    print("=" * 60)

    # 1. Document inspection
    pdf_path = "data/documents/VITAP_Fee_Structure_and_Scholarships.pdf"
    pages = extract_text_from_pdf(pdf_path)
    full_text = pages[0]["text"]
    print("\n--- 1. PDF TEXT EXTRACTION ---")
    print(f"File: {pdf_path}")
    print(f"Pages: {len(pages)}")
    print(f"Length: {len(full_text)} characters")
    print("Contains 'GVSDP':", "GVSDP" in full_text)
    print("Contains 'STARS':", "STARS" in full_text)
    print("Contains 'Merit Awards':", "Merit Awards" in full_text)

    # 2. ChromaDB inspection
    client = chromadb.PersistentClient(path="vectorstore/chroma")
    col = client.get_or_create_collection("campus_documents")
    all_chunks = col.get(where={"source": "VITAP_Fee_Structure_and_Scholarships.pdf"}, include=["documents", "metadatas"])
    print("\n--- 2. CHROMADB CHUNKS IN STORE ---")
    print(f"Total chunks stored for {pdf_path}: {len(all_chunks['ids'])}")
    for i, (cid, doc) in enumerate(zip(all_chunks["ids"], all_chunks["documents"])):
        print(f"\nChunk {i} ID: {cid}")
        print(f"Snippet ({len(doc)} chars): {repr(doc[:120])}")
        if "GVSDP" in doc or "STARS" in doc or "Merit" in doc or "scholarship" in doc.lower():
            print(">>> Matches scholarship content! Full chunk text:")
            print(doc)

    # 3. Retrieval test
    print("\n--- 3. RETRIEVAL FOR QUERY ---")
    res = retrieve_documents(query, n_results=10)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    print(f"Retrieved {len(docs)} chunks:")
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
        has_kw = any(k in doc for k in ["GVSDP", "STARS", "Merit", "Scholarship", "scholarship"])
        print(f"Rank {i+1}: {meta.get('source')} (p.{meta.get('page')}) - Dist: {dist:.4f} - HasScholarshipKW: {has_kw}")

    # 4. Context formation & LLM call
    print("\n--- 4. LLM GENERATION ---")
    ctx_parts = []
    for i in range(len(docs)):
        m = metas[i] if i < len(metas) else {}
        ctx_parts.append(f"Source: {m.get('source')}\nPage: {m.get('page')}\nContent:\n{docs[i]}\n")
    context = "UNIVERSITY DOCUMENTS:\n" + "\n".join(ctx_parts)

    print("Context length passed to LLM:", len(context))
    ans = generate_answer(query, context)
    print("\n=== FINAL LLM ANSWER ===")
    print(ans)

if __name__ == "__main__":
    run_diagnostics()
