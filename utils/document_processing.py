"""
document_processing.py

Utilities for reading and processing different document formats,
including PDF, TXT, and MD, as well as text chunking and file hashing.
"""

import hashlib
import logging
from typing import Any, List

from pypdf import PdfReader

logger = logging.getLogger(__name__)


def read_pdf(file: Any) -> str:
    """Reads a PDF file and returns its text content.

    Args:
        file: A file-like object or path to a PDF file.

    Returns:
        str: The extracted text from the PDF.
    """
    reader = PdfReader(file)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def read_txt(file: Any) -> str:
    """Reads a text file and returns its content.

    Args:
        file: A file-like object or path to a text file.

    Returns:
        str: The content of the text file.
    """
    if hasattr(file, "read"):
        content = file.read()
        if isinstance(content, bytes):
            return content.decode("utf-8")
        return str(content)

    with open(file, "r", encoding="utf-8") as f:
        return f.read()


def read_md(file: Any) -> str:
    """Reads a markdown file and returns its content.

    Args:
        file: A file-like object or path to a markdown file.

    Returns:
        str: The content of the markdown file.
    """
    return read_txt(file)


def get_file_hash(file: Any) -> str:
    """Generates a SHA-256 hash for the file content.

    Args:
        file: A file-like object or path to a file.

    Returns:
        str: The SHA-256 hash of the file content.
    """
    sha256_hash = hashlib.sha256()
    if hasattr(file, "read"):
        # Store current position if possible
        pos = 0
        if hasattr(file, "tell"):
            pos = file.tell()
        if hasattr(file, "seek"):
            file.seek(0)

        for byte_block in iter(lambda: file.read(4096), b""):
            if isinstance(byte_block, str):
                byte_block = byte_block.encode("utf-8")
            sha256_hash.update(byte_block)

        # Restore position
        if hasattr(file, "seek"):
            file.seek(pos)
    else:
        with open(file, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """Chunks text into smaller pieces with a specified overlap.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum size of each chunk.
        overlap (int): The number of characters to overlap between chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    # Normalize whitespace
    text = " ".join(text.split())
    chunks = []
    start = 0

    if not text:
        return []

    if chunk_size <= overlap:
        logger.warning("chunk_size must be greater than overlap. Using overlap = 0.")
        overlap = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)

        if end >= len(text):
            break

        start += chunk_size - overlap

    return chunks
