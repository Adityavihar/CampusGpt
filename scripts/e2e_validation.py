import sys
import os
sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
from services.retriever import retrieve_documents
from services.llm import generate_answer

def test_query(question: str, history_context: str = ""):
    print(f"\n==========================================")
    print(f"QUERY: {question}")
    print(f"==========================================")
    
    # Retrieve
    res = retrieve_documents(question, n_results=5)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    ctx_parts = []
    for i in range(len(docs)):
        m = metas[i] if i < len(metas) else {}
        ctx_parts.append(
            f"Source: {m.get('source')}\nPage: {m.get('page')}\nContent:\n{docs[i]}\n"
        )
    
    full_context = ""
    if history_context:
        full_context += f"PREVIOUS CONVERSATION:\n{history_context}\n"
    full_context += "UNIVERSITY DOCUMENTS:\n" + "\n".join(ctx_parts)

    answer = generate_answer(question, full_context)
    print("ANSWER:")
    print(answer)
    print("\nRETRIEVED SOURCES:")
    for m in metas:
        print(f"- {m.get('source')} (Page {m.get('page')})")
    return answer

if __name__ == "__main__":
    # Test A: B.Tech Programs
    ans_a = test_query("What B.Tech programs are offered by VIT-AP University?")

    # Test B: Follow-up question
    history = f"Student: What B.Tech programs are offered by VIT-AP University?\nCampusGPT: {ans_a}\n"
    test_query("What about M.Tech programs?", history_context=history)

    # Test C: Academic Regulations & Attendance
    test_query("What is the minimum attendance requirement and the debarment policy at VIT-AP?")

    # Test D: Unsupported question
    test_query("What is the undergraduate tuition fee at Stanford University?")
