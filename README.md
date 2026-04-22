# AI Document Analyzer

| | |
| --- | --- |
| **Testing** | [![CI Status](https://github.com/Unloosed/ai-document-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/Unloosed/ai-document-analyzer/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/Unloosed/ai-document-analyzer/graph/badge.svg?token=VXecHu2XBp)](https://codecov.io/gh/Unloosed/ai-document-analyzer) |
| **Code Quality** | [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![Security: Bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit) |
| **Meta** | [![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) |
| **Docs** | [![Documentation Status](https://github.com/Unloosed/ai-document-analyzer/actions/workflows/deploy-docs.yml/badge.svg)](https://Unloosed.github.io/ai-document-analyzer/) |

---

A Streamlit-based RAG (Retrieval-Augmented Generation) assistant that answers questions over uploaded PDF, TXT, and MD documents using embedding search and grounded generation.

## Features

- **Multi-format Support**: Upload and index PDFs, Text files, and Markdown files.
- **RAG Architecture**: Uses OpenAI embeddings and ChromaDB for semantic retrieval.
- **Duplicate Detection**: Content-based hashing to prevent redundant document indexing.
- **Grounded Answers**: Provides citations (source and chunk) to reduce hallucinations.
- **Configurable Search**: Adjust chunk sizes, overlap, and similarity confidence via the UI.
- **Feedback Loop**: Log user feedback (thumbs up/down) for answer quality tracking.
- **Admin Dashboard**: Real-time statistics on indexed documents and chunks.

## Software Architecture

The project follows a modular design to separate concerns:

- **Frontend (`app.py`)**: Streamlit-based user interface for document upload, indexing, and querying.
- **RAG Logic (`rag.py`)**: Orchestrates the retrieval-augmented generation process, including prompt construction and LLM interaction.
- **Vector Store (`store.py`)**: Manages document indexing and retrieval using ChromaDB. It includes a manifest system for deduplication.
- **Embeddings (`embeddings.py`)**: Custom wrapper for generating text embeddings via OpenRouter/OpenAI.
- **Feedback (`feedback.py`)**: Simple utility to log user feedback on generated answers.
- **Configuration (`config.py`)**: Centralized environment variable management.
- **Utilities (`utils/`)**: Low-level document processing (PDF/TXT/MD reading, chunking, hashing) and logging setup.

## Deep Dive: RAG Implementation

### What is RAG?
Retrieval-Augmented Generation (RAG) is a technique that enhances Large Language Model (LLM) responses by retrieving relevant information from a specific dataset before generating an answer. This "grounds" the model in facts and reduces hallucinations.

### How it works in this project

1.  **Ingestion & Processing**:
    - Files are uploaded via Streamlit.
    - `utils.document_processing` extracts text based on file type.
    - Content is hashed (`SHA-256`) to prevent re-indexing the same file.
2.  **Chunking**:
    - Text is split into smaller, overlapping segments (chunks). Overlap ensures context isn't lost at the boundaries of chunks.
3.  **Embedding & Storage**:
    - Each chunk is converted into a numerical vector (embedding) using `openai/text-embedding-3-small`.
    - Vectors and metadata (source name, chunk index) are stored in **ChromaDB**.
4.  **Retrieval**:
    - When a user asks a question, the query is also embedded.
    - ChromaDB performs a semantic search to find the top $k$ chunks most similar to the query vector (using cosine distance).
5.  **Generation**:
    - The retrieved chunks are formatted into a prompt as "Context".
    - The LLM (e.g., `openrouter/free`) is instructed to answer the question using only the provided context.
    - The system enforces bracketed citations (e.g., [1]) to link statements back to specific chunks.

### Design Choice Justifications

- **ChromaDB**: Chosen for being lightweight, open-source, and capable of running locally without complex infrastructure (PersistentClient). It supports metadata filtering and efficient vector search.
- **OpenRouter**: Used as a gateway to access various LLMs (like OpenAI, Anthropic, or free models) through a unified API, providing flexibility and cost management.
- **Manual Implementation (Vanilla Python)**: Instead of using high-level frameworks like LangChain or LlamaIndex, this project implements the RAG pipeline manually. This provides:
    - **Transparency**: Clear visibility into how prompts are built and how retrieval works.
    - **Minimal Dependencies**: Reduced "dependency bloat" and faster startup times.
    - **Control**: Easier customization of the manifest-based deduplication and citation logic.

### Alternatives Considered

- **FAISS**: Extremely fast for vector search but lacks the built-in metadata management and persistence features that ChromaDB provides out-of-the-box.
- **Pinecone**: A great managed vector database, but it requires an external service and API keys, which conflicts with the goal of a self-contained local-first analyzer.
- **LangChain**: While powerful, LangChain's abstractions can often make debugging and fine-grained control more difficult. For a focused project like this, the direct implementation is more maintainable.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Unloosed/ai-document-analyzer.git
   cd ai-document-analyzer
   ```

2. **Set up a virtual environment**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -e .
   ```

## Configuration

Create a `.env` file in the root directory with your credentials:

```text
OPENROUTER_API_KEY=your_openrouter_api_key_here
EMBEDDING_MODEL=openai/text-embedding-3-small
CHAT_MODEL=openrouter/free
```

## Running the App

Start the Streamlit application:

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

## Development

### Project Structure

- `utils/`: Core utility modules for file operations, document processing, and logging.
- `tests/`: Test suite using `pytest`.
- `source/`: `Sphinx` documentation source.
- `.github/workflows/`: CI/CD pipeline definitions for GitHub Actions.
- `pyproject.toml`: Project metadata, dependency management, and tool configurations.
- `Makefile`: Common development tasks.
- `.pre-commit-config.yaml`: Pre-commit hook configuration.
- `.editorconfig`: Editor formatting configuration.
- `.devcontainer/`: VS Code development container setup.

### Running Tests

We use `pytest` for testing. To run all tests:

```bash
PYTHONPATH=. pytest
```

### Linting and Formatting

We use `ruff` for linting and formatting.

```bash
# Check for linting issues
ruff check .

# Apply automatic formatting
ruff format .
```

### Security Audit

To run security checks with `bandit` and `pip-audit`:

```bash
# Static analysis
bandit -c pyproject.toml -r .

# Dependency audit
pip-audit
```

### Static Type Checking

We use `mypy` for static type checking.

```bash
mypy .
```

### Pre-commit Hooks

We use `pre-commit` to ensure code quality before every commit.

```bash
# Install the hooks
pip install pre-commit
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

### Makefile

A `Makefile` is provided to simplify common development tasks.

```bash
make help        # Show all available commands
make dev-install # Install all dev dependencies
make lint        # Run ruff check
make format      # Run ruff format
make type-check  # Run mypy
make test        # Run pytest
make docs        # Build documentation
make clean       # Remove build and cache artifacts
```
