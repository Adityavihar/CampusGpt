import sys
import os
sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from services.retriever import retrieve_documents
from services.llm import generate_answer

def test_multi_history():
    recent_history = [
        "What B.Tech programs and specializations are offered by VIT-AP?",
        "What is the minimum attendance requirement and the debarment policy at VIT-AP?",
        "Explain the Fully Flexible Credit System (FFCS) at VIT-AP."
    ]
    curr_q = "What scholarships are available for top-ranking students?"

    # app.py logic:
    retrieval_query = (
        "Previous student questions:\n"
        + "\n".join(recent_history)
        + "\n\nCurrent student question:\n"
        + curr_q
    )

    print("Retrieval query sent:\n", repr(retrieval_query))
    res = retrieve_documents(retrieval_query, n_results=10)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
        has_sch = "scholarship" in doc.lower() or "gvsdp" in doc.lower() or "stars" in doc.lower()
        print(f"Rank {i+1}: {meta.get('source')} (p.{meta.get('page')}) - Dist: {dist:.4f} - HasSch: {has_sch}")

    ctx = "UNIVERSITY DOCUMENTS:\n" + "\n".join([f"Source: {metas[i].get('source')}\nPage: {metas[i].get('page')}\n{docs[i]}\n" for i in range(len(docs))])
    ans = generate_answer(curr_q, ctx)
    print("\n--- LLM ANSWER ---")
    print(ans)

if __name__ == "__main__":
    test_multi_history()
