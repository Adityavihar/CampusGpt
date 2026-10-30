import sys
import os
sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from services.retriever import retrieve_documents
from services.llm import generate_answer

def verify_llm_generation():
    print("=" * 70)
    print("DIAGNOSTIC 3: CONTEXT PASSED TO GROQ LLM & ANSWER GENERATION")
    print("=" * 70)
    
    query = "What scholarships are available for top-ranking students?"
    prev_q = "What is the minimum attendance requirement and the debarment policy at VIT-AP?"

    # =========================================================================
    # SCENARIO A: Context retrieved WITH history pollution (Exact issue user reported)
    # =========================================================================
    print("\n" + "=" * 50)
    print("SCENARIO A: LLM RUN WITH HISTORY POLLUTED CONTEXT")
    print("=" * 50)
    
    polluted_query = f"Previous student questions:\n{prev_q}\n\nCurrent student question:\n{query}"
    res_a = retrieve_documents(polluted_query, n_results=10)
    docs_a = res_a["documents"][0]
    metas_a = res_a["metadatas"][0]
    
    ctx_parts_a = []
    for i in range(len(docs_a)):
        m = metas_a[i] if i < len(metas_a) else {}
        ctx_parts_a.append(f"Source: {m.get('source')}\nPage: {m.get('page')}\nContent:\n{docs_a[i]}\n")
        
    context_a = "UNIVERSITY DOCUMENTS:\n" + "\n".join(ctx_parts_a)
    print(f"Total context length passed to LLM: {len(context_a)} characters")
    print("Contains 'scholarship' in context:", "scholarship" in context_a.lower())
    print("Contains 'GVSDP' in context:", "gvsdp" in context_a.lower())
    print("Contains 'STARS' in context:", "stars" in context_a.lower())
    
    print("\nCalling Groq LLM for Scenario A...")
    answer_a = generate_answer(query, context_a)
    print("\n--- GROQ LLM ANSWER (SCENARIO A) ---")
    print(answer_a)

    # =========================================================================
    # SCENARIO B: Context retrieved WITHOUT history pollution (Clean query)
    # =========================================================================
    print("\n" + "=" * 50)
    print("SCENARIO B: LLM RUN WITH CLEAN QUERY RETRIEVAL")
    print("=" * 50)
    
    res_b = retrieve_documents(query, n_results=10)
    docs_b = res_b["documents"][0]
    metas_b = res_b["metadatas"][0]
    
    ctx_parts_b = []
    for i in range(len(docs_b)):
        m = metas_b[i] if i < len(metas_b) else {}
        ctx_parts_b.append(f"Source: {m.get('source')}\nPage: {m.get('page')}\nContent:\n{docs_b[i]}\n")
        
    context_b = "UNIVERSITY DOCUMENTS:\n" + "\n".join(ctx_parts_b)
    print(f"Total context length passed to LLM: {len(context_b)} characters")
    print("Contains 'scholarship' in context:", "scholarship" in context_b.lower())
    print("Contains 'GVSDP' in context:", "gvsdp" in context_b.lower())
    print("Contains 'STARS' in context:", "stars" in context_b.lower())
    
    print("\nCalling Groq LLM for Scenario B...")
    answer_b = generate_answer(query, context_b)
    print("\n--- GROQ LLM ANSWER (SCENARIO B) ---")
    print(answer_b)

if __name__ == "__main__":
    verify_llm_generation()
