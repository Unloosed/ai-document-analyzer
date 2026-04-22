import logging
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from config import CHAT_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from store import search_docs

logger = logging.getLogger(__name__)

# Dedicated client for chat
chat_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    default_headers={
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "ai-document-analyzer",
    },
)

def answer_question(
    query: str, strict_mode: bool = False, min_confidence: float = 0.0
) -> Tuple[str, List[Tuple[str, Dict[str, Any], float]]]:
    """Answers a question using retrieved context.

    Args:
        query (str): The user's question.
        strict_mode (bool): If True, strictly follows the context.
        min_confidence (float): Minimum similarity score (distance threshold).

    Returns:
        Tuple[str, List]: The generated answer and the retrieved chunks.
    """
    retrieved = search_docs(query, k=5)

    # Filter by confidence
    # Since we use cosine distance, lower is better (0.0 is exact match, 1.0/2.0 is opposite)
    # min_confidence in UI is 0.0 to 1.0 (similarity).
    # We map it: threshold = 1.0 - min_confidence
    if min_confidence > 0:
        threshold = 1.0 - min_confidence
        retrieved = [r for r in retrieved if r[2] <= threshold]

    if not retrieved and strict_mode:
        return (
            "I couldn't find enough relevant information to answer your question.",
            [],
        )

    context_parts = []
    for i, (doc, meta, _) in enumerate(retrieved, start=1):
        context_parts.append(
            f"--- Chunk [{i}] (Source: {meta['source']}) ---\n{doc}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = "You are a careful document assistant."
    if strict_mode:
        system_prompt += (
            " Answer the user's question using ONLY the context provided. "
            "If the answer is not in the context, say you don't know. "
            "Every statement you make must be supported by the context."
        )
    else:
        system_prompt += (
            " Use the provided context to answer the question. "
            "If the answer isn't fully covered, use your knowledge but prioritize context. "
            "Be transparent about what comes from the context and what is general knowledge."
        )

    system_prompt += (
        "\n\nYou MUST cite your sources using the chunk numbers in brackets, "
        "like [1] or [1, 3], when referring to information from the context."
    )

    user_prompt = f"""Context:
{context}

Question:
{query}

Please provide a detailed answer with citations.
"""

    try:
        response = chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content, retrieved
    except Exception as e:
        logger.error("Chat completion failed: %s", e)
        return f"Error generating answer: {str(e)}", retrieved
