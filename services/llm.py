import os
import logging
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

api_key = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


def _get_client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY is not set in the environment or .env file")
    return Groq(api_key=key)


SYSTEM_PROMPT = """You are CampusGPT, the official-information academic assistant for VIT-AP University.

Your job is to provide accurate, helpful, and concise answers to student questions based STRICTLY on the provided university document excerpts.

CRITICAL RULES:
1. GROUNDING: The UNIVERSITY DOCUMENTS provided in the prompt are your SOLE source of factual information. Never fabricate policies, fees, programs, cutoffs, or deadlines.
2. CLEAN DIRECT ANSWERS: Deliver clean, direct, and well-structured answers using bullet points, clear headings, or tables. Do not include citation tags, footnotes, or source document references in the answer.
3. FOLLOW-UP UNDERSTANDING: You may reference the PREVIOUS CONVERSATION solely to understand ambiguous pronouns or follow-up references (e.g., "What about M.Tech?", "Which one offers AI?", "How much does it cost?"). However, every fact in your answer must be substantiated by the university documents.
4. HONESTY ON MISSING INFO: If the provided documents do not contain sufficient evidence to answer the student's question, return EXACTLY:
"I couldn't find this information in the available VIT-AP university documents. Please refer to the official university portal at https://vitap.ac.in for the most current information."
5. FORMATTING: Use structured markdown (bullet points, clear headings, bold text, or tables where appropriate) for readability.
6. NO SYSTEM LEAKS: Do not mention internal concepts such as "vector database", "ChromaDB", "embeddings", "top-k", "RAG", or "chunks".
"""


def generate_answer(question: str, context: str) -> str:
    """
    Generate an answer using university documents and Groq LLM.
    Handles conversation history and enforces grounding with citations.
    """
    if not question or not question.strip():
        return "Please ask a question about VIT-AP University."

    # If context is empty or contains no university documents
    if not context or "UNIVERSITY DOCUMENTS" not in context or not context.strip():
        return (
            "I couldn't find this information in the available VIT-AP university documents. "
            "Please refer to the official university portal at https://vitap.ac.in for the most current information."
        )

    user_prompt = f"""CONTEXT INFORMATION:
{context}

STUDENT QUESTION:
{question}

Answer the student's question clearly and directly based strictly on the university documents above. Do not include citation tags, footnotes, or source filenames in the text. If the information is missing from the documents, respond with the exact unavailable notice."""

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            reasoning_effort="low",
            include_reasoning=False,
            temperature=0.1,
            max_tokens=1500
        )

        message = response.choices[0].message
        answer = message.content

        if not answer or not answer.strip():
            logger.warning("Empty response received from Groq LLM")
            return (
                "I couldn't find this information in the available VIT-AP university documents. "
                "Please refer to the official university portal at https://vitap.ac.in."
            )

        return answer.strip()

    except Exception as e:
        logger.error(f"Groq LLM generation error: {e}")
        return (
            "I couldn't generate an answer at this time due to an upstream service error. "
            "Please verify your connection and try again."
        )
