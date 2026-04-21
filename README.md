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
OPENAI_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
```

## Running the App

Start the Streamlit application:

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

## Development

### Running Tests

We use `pytest` for testing. To run all tests with coverage:

```bash
pytest
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
# or
make pre-commit
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

## Project Structure

- `utils/`: Core utility modules for file operations and logging.
- `tests/`: Test suite using `pytest`.
- `source/`: `Sphinx` documentation source.
- `.github/workflows/`: CI/CD pipeline definitions for GitHub Actions.
- `pyproject.toml`: Project metadata, dependency management, and tool configurations.
- `Makefile`: Common development tasks.
- `.pre-commit-config.yaml`: Pre-commit hook configuration.
- `.editorconfig`: Editor formatting configuration.
- `.devcontainer/`: VS Code development container setup.
