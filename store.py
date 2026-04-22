import json
import logging
import os
import uuid
from typing import Any, Dict, List, Tuple

import chromadb

from embeddings import OpenAIEmbeddingFunction

logger = logging.getLogger(__name__)

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./data")
collection = chroma_client.get_or_create_collection(
    name="docs",
    embedding_function=OpenAIEmbeddingFunction(),
    metadata={"hnsw:space": "cosine"}
)

MANIFEST_PATH = "./data/manifest.json"

def _load_manifest() -> Dict[str, Any]:
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load manifest: %s", e)
    return {"hashes": {}, "total_chunks": 0}

def _save_manifest(manifest: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    try:
        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2)
    except Exception as e:
        logger.error("Failed to save manifest: %s", e)

def is_document_indexed(file_hash: str) -> bool:
    manifest = _load_manifest()
    return file_hash in manifest["hashes"]

def add_document(filename: str, file_hash: str, chunks: List[str]) -> bool:
    """Adds a document to the vector store if its hash doesn't already exist.

    Args:
        filename (str): The name of the file.
        file_hash (str): The SHA-256 hash of the file content.
        chunks (List[str]): The text chunks to index.

    Returns:
        bool: True if the document was added, False if it was a duplicate.
    """
    manifest = _load_manifest()
    if file_hash in manifest["hashes"]:
        logger.info("Document %s already exists (duplicate hash).", filename)
        return False

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {"source": filename, "chunk_index": i, "file_hash": file_hash}
        for i in range(len(chunks))
    ]

    collection.add(ids=ids, documents=chunks, metadatas=metadatas)

    # Update manifest
    manifest["hashes"][file_hash] = filename
    manifest["total_chunks"] += len(chunks)
    _save_manifest(manifest)

    return True

def search_docs(query: str, k: int = 5) -> List[Tuple[str, Dict[str, Any], float]]:
    """Searches the vector store for relevant document chunks.

    Args:
        query (str): The search query.
        k (int): The number of results to return.

    Returns:
        List[Tuple[str, Dict[str, Any], float]]: A list of (document, metadata, distance).
    """
    results = collection.query(query_texts=[query], n_results=k)

    if not results["documents"] or not results["documents"][0]:
        return []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0] if results["distances"] else [0.0] * len(docs)

    return list(zip(docs, metas, distances, strict=False))

def get_stats() -> Dict[str, Any]:
    """Returns basic statistics about the vector store.

    Returns:
        Dict[str, Any]: Statistics including doc count and chunk count.
    """
    manifest = _load_manifest()
    return {
        "total_chunks": manifest["total_chunks"],
        "unique_documents": len(manifest["hashes"]),
    }
