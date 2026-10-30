import sys
import os
sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from services.retriever import retrieve_documents
from services.llm import generate_answer

def test_session_history_effect():
    # In the screenshot provided by user, the previous question in chat was:
    # "What is the minimum attendance requirement and the debarment policy at VIT-AP?"
    prev_q = "What is the minimum attendance requirement and the debarment policy at VIT-AP?"
    curr_q = "What scholarships are available for top-ranking students?"

    # Case 1: With session history prepended to retrieval query (as app.py currently does)
    polluted_query = f"Previous student questions:\n{prev_q}\n\nCurrent student question:\n{curr_q}"
    print("=" * 60)
    print("CASE 1: WITH SESSION HISTORY PREPENDED TO RETRIEVAL QUERY")
    print("=" * 60)
    print("Query sent to embedder:\n", repr(polluted_query))
    
    res1 = retrieve_documents(polluted_query, n_results=10)
    docs1 = res1["documents"][0]
    metas1 = res1["metadatas"][0]
    dists1 = res1["distances"][0]

    print(f"\nRetrieved {len(docs1)} documents:")
    has_scholarship_chunk = False
    for i, (doc, meta, dist) in enumerate(zip(docs1, metas1, dists1)):
        src = meta.get("source")
        has_sch = "scholarship" in doc.lower() or "gvsdp" in doc.lower() or "stars" in doc.lower()
        if has_sch:
            has_scholarship_chunk = True
        print(f"Rank {i+1}: {src} (p.{meta.get('page')}) - Dist: {dist:.4f} - HasScholarshipText: {has_sch}")
        print(f"   Snippet: {repr(doc[:80])}")

    print("\nDid Case 1 retrieve scholarship text?", has_scholarship_chunk)

    ctx1 = "\n".join([f"Source: {metas1[i].get('source')}\nPage: {metas1[i].get('page')}\n{docs1[i]}\n" for i in range(len(docs1))])
    full_ctx1 = f"PREVIOUS CONVERSATION:\nStudent: {prev_q}\nCampusGPT: Minimum 75% attendance...\n\nUNIVERSITY DOCUMENTS:\n{ctx1}"
    ans1 = generate_answer(curr_q, full_ctx1)
    print("\n--- LLM ANSWER FOR CASE 1 ---")
    print(ans1)

if __name__ == "__main__":
    test_session_history_effect()
