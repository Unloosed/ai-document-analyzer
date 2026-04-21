import csv
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import chromadb
from chromadb.api.types import EmbeddingFunction, Embeddings
from openai import OpenAI

from utils.file_ops import get_env_variable

logger = logging.getLogger(__name__)

try:
    OPENROUTER_API_KEY = get_env_variable("OPENROUTER_API_KEY", required=True)
    OPENROUTER_BASE_URL = get_env_variable(
        "OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1"
    )
    EMBEDDING_MODEL = get_env_variable(
        "EMBEDDING_MODEL", default="nomic-embed-text-v1.5"
    )
    CHAT_MODEL = get_env_variable("CHAT_MODEL", default="openrouter/free")
except Exception as e:
    logger.warning("Environment variables not fully set: %s", e)
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "missing")
    OPENROUTER_BASE_URL = os.getenv(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )
    EMBEDDING_MODEL = get_env_variable(
        "EMBEDDING_MODEL", default="nomic-embed-text-v1.5"
    )
    CHAT_MODEL = get_env_variable("CHAT_MODEL", default="openrouter/free")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    default_headers={
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "ai-document-analyzer",
    },
)


class OpenAIEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input=None, texts=None) -> Embeddings:
        if input is None and texts is not None:
            input = texts
        elif input is None:
            raise ValueError("No input provided to embedding function")

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=input,
        )
        return [item.embedding for item in response.data]


# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./data")
collection = chroma_client.get_or_create_collection(
    name="docs", embedding_function=OpenAIEmbeddingFunction()
)


def add_document(filename: str, file_hash: str, chunks: List[str]) -> bool:
    """Adds a document to the vector store if its hash doesn't already exist.

    Args:
        filename (str): The name of the file.
        file_hash (str): The SHA-256 hash of the file content.
        chunks (List[str]): The text chunks to index.

    Returns:
        bool: True if the document was added, False if it was a duplicate.
    """
    # Check for duplicate by hash in metadata
    existing = collection.get(where={"file_hash": file_hash})
    if existing and existing["ids"]:
        logger.info("Document %s already exists (duplicate hash).", filename)
        return False

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {"source": filename, "chunk_index": i, "file_hash": file_hash}
        for i in range(len(chunks))
    ]

    collection.add(ids=ids, documents=chunks, metadatas=metadatas)
    return True


def search_docs(query: str, k: int = 4) -> List[Tuple[str, Dict[str, Any], float]]:
    """Searches the vector store for relevant document chunks.

    Args:
        query (str): The search query.
        k (int): The number of results to return.

    Returns:
        List[Tuple[str, Dict[str, Any], float]]: A list of (document, metadata, distance).
    """
    results = collection.query(query_texts=[query], n_results=k)

    if not results["documents"]:
        return []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0] if results["distances"] else [0.0] * len(docs)

    return list(zip(docs, metas, distances))


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

    # Filter by confidence if needed (Note: Chroma distances are typically squared L2)
    # Lower distance means higher similarity.
    if min_confidence > 0:
        # This is a naive thresholding
        retrieved = [r for r in retrieved if r[2] <= (1.0 - min_confidence)]

    if not retrieved and strict_mode:
        return (
            "I couldn't find enough relevant information to answer your question.",
            [],
        )

    context_parts = []
    for doc, meta, _ in retrieved:
        context_parts.append(
            f"[Source: {meta['source']} | Chunk: {meta['chunk_index']}]\n{doc}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = "You are a careful document assistant."
    if strict_mode:
        system_prompt += (
            " Answer the user's question using ONLY the context below. "
            "If the answer is not in the context, say you don't know."
        )
    else:
        system_prompt += (
            " Use the provided context to answer the question. "
            "If the answer isn't fully covered, use your knowledge but prioritize context."
        )

    prompt = f"""Include brief citations like (source: filename, chunk N).

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content, retrieved


def log_feedback(
    question: str, answer: str, feedback: str, path: str = "feedback.csv"
) -> None:
    """Logs user feedback to a CSV file.

    Args:
        question (str): The original question.
        answer (str): The generated answer.
        feedback (str): 'thumbs_up' or 'thumbs_down'.
        path (str): Path to the feedback CSV file.
    """
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "question", "answer", "feedback"])
        writer.writerow(
            [datetime.now(timezone.utc).isoformat(), question, answer, feedback]
        )


def get_stats() -> Dict[str, Any]:
    """Returns basic statistics about the vector store.

    Returns:
        Dict[str, Any]: Statistics including doc count and chunk count.
    """
    count = collection.count()
    metas = collection.get(include=["metadatas"])["metadatas"]
    unique_docs = len(set(m["file_hash"] for m in metas)) if metas else 0

    return {
        "total_chunks": count,
        "unique_documents": unique_docs,
    }
