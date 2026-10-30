import sys
import os
sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import chromadb
from services.retriever import retrieve_documents

def verify_chroma_and_retrieval():
    print("=" * 70)
    print("DIAGNOSTIC 2: CHROMADB CHUNKS & RETRIEVAL COMPARISON")
    print("=" * 70)
    
    # 1. Inspect existing chunks in ChromaDB
    client = chromadb.PersistentClient(path="vectorstore/chroma")
    col = client.get_or_create_collection("campus_documents")
    doc_res = col.get(where={"source": "VITAP_Fee_Structure_and_Scholarships.pdf"}, include=["documents", "metadatas"])
    
    print(f"\n--- 1. STORED CHUNKS FOR 'VITAP_Fee_Structure_and_Scholarships.pdf' ---")
    print(f"Total chunks stored: {len(doc_res['ids'])}")
    for i, (cid, doc) in enumerate(zip(doc_res["ids"], doc_res["documents"])):
        print(f"\n[Chunk {i} | ID: {cid} | Length: {len(doc)} chars]")
        print("Text preview:")
        print(doc.strip())
        print("-" * 50)

    # 2. Retrieval WITHOUT conversation history (Standalone query)
    query = "What scholarships are available for top-ranking students?"
    print("\n--- 2. RETRIEVAL WITHOUT CONVERSATION HISTORY ---")
    print(f"Query: {query}")
    
    res_clean = retrieve_documents(query, n_results=10)
    docs_clean = res_clean["documents"][0]
    metas_clean = res_clean["metadatas"][0]
    dists_clean = res_clean["distances"][0]
    
    print(f"Retrieved {len(docs_clean)} chunks:")
    for i in range(len(docs_clean)):
        src = metas_clean[i].get("source")
        page = metas_clean[i].get("page")
        dist = dists_clean[i]
        has_sch = any(k in docs_clean[i].lower() for k in ["scholarship", "gvsdp", "stars", "merit"])
        print(f"  Rank {i+1}: {src} (p.{page}) | Dist: {dist:.4f} | HasScholarshipInfo: {has_sch}")
        print(f"          Snippet: {repr(docs_clean[i][:90])}")

    # 3. Retrieval WITH conversation history prepended (as app.py currently does)
    prev_questions = [
        "What is the minimum attendance requirement and the debarment policy at VIT-AP?"
    ]
    polluted_query = (
        "Previous student questions:\n"
        + "\n".join(prev_questions)
        + "\n\nCurrent student question:\n"
        + query
    )
    print("\n--- 3. RETRIEVAL WITH CONVERSATION HISTORY PREPENDED (app.py behavior) ---")
    print(f"Query sent to Jina Embeddings API:\n{repr(polluted_query)}")
    
    res_hist = retrieve_documents(polluted_query, n_results=10)
    docs_hist = res_hist["documents"][0]
    metas_hist = res_hist["metadatas"][0]
    dists_hist = res_hist["distances"][0]
    
    print(f"\nRetrieved {len(docs_hist)} chunks:")
    found_any_scholarship = False
    for i in range(len(docs_hist)):
        src = metas_hist[i].get("source")
        page = metas_hist[i].get("page")
        dist = dists_hist[i]
        has_sch = any(k in docs_hist[i].lower() for k in ["scholarship", "gvsdp", "stars", "merit"])
        if has_sch:
            found_any_scholarship = True
        print(f"  Rank {i+1}: {src} (p.{page}) | Dist: {dist:.4f} | HasScholarshipInfo: {has_sch}")
        print(f"          Snippet: {repr(docs_hist[i][:90])}")
        
    print(f"\nDid retrieval with history include scholarship chunks?: {found_any_scholarship}")

if __name__ == "__main__":
    verify_chroma_and_retrieval()
