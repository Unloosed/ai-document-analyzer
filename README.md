# General Python Programming

| | |
| --- | --- |
| **Testing** | [![CI Status](https://github.com/Unloosed/ai-document-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/Unloosed/ai-document-analyzer/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/Unloosed/ai-document-analyzer/graph/badge.svg?token=VXecHu2XBp)](https://codecov.io/gh/Unloosed/ai-document-analyzer) |
| **Code Quality** | [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![Security: Bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit) |
| **Meta** | [![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) |
| **Docs** | [![Documentation Status](https://github.com/Unloosed/ai-document-analyzer/actions/workflows/deploy-docs.yml/badge.svg)](https://Unloosed.github.io/ai-document-analyzer/) |

---

A modern Python project template with utility functions for file operations, logging, and environment management.

## Features

- **Logging Setup**: Standardized logging with console colors and file rotation.
- **File Operations**: Utilities for directory management, environment variable loading, and Git integration.
- **Modern Plumbing**: Uses `pyproject.toml` for dependency management and project configuration.
- **CI/CD Ready**: Includes configurations for GitHub Actions (linting, testing, type-checking, and security checks).
- **Static Analysis**: Configured with `ruff`, `mypy`, and `bandit` for high code quality and security.
- **Pre-commit Hooks**: Automatic quality checks before every commit.
- **Standardized Dev Environment**: `Makefile` for common tasks and VS Code Dev Container for an instant setup.
- **Automated Documentation**: Utilizes `Sphinx` and GitHub Actions for generation and publication of documentation.

## Template Setup

To use this project as a template, please follow these steps:

1. **Update `pyproject.toml`**:
    - Modify the `[project]` section, including `name`, `version`, `description`, and `dependencies`.
    - Update the `[project.urls]` section with your repository URL.
2. **Configure CI/CD**:
    - **GitHub Actions**: Ensure the `.github/workflows/ci.yml` file is configured correctly for your branch names and environments.
    - **Codecov**:
        - Create a [Codecov](https://codecov.io/) account and link it to your repository.
        - Retrieve your `CODECOV_TOKEN` from the Codecov repository settings.
        - Add the token as a repository secret in GitHub: `Settings > Secrets and variables > Actions > New repository secret`.
        - Modify the badge in `README.md` by getting a token from Codecov's `Config > Badges & Graphs`
3. **Update `Sphinx` documentation**:
    - Update the `copyright`, `author`, and `release` variables in `source/conf.py`.
    - Update the project name at the top of `source/index.rst`
4. **Update `README.md`**:
    - Replace the badges at the top with your own (testing, code quality, etc.).
    - Adjust the project description and usage examples to match your specific application.
5. **Manage Dependencies**:
    - Add project-specific dependencies to the `dependencies` list in `pyproject.toml`.
    - If additional dev tools are needed, add them to `[project.optional-dependencies]`.
6. **Add credentials**:
    - Create a `.env` file in the project root and store credentials in there if not using key vaults like Azure
7. **Enable GitHub Pages**:
    - Navigate to *Settings > Pages* and change the `Source` to `GitHub Actions`

## Installation

To install the project in editable mode with development dependencies:

```bash
# Using pip
pip install -e ".[dev,sphinx]"

# Or using the tasks.py invocation
python -m invoke --list
```

## Usage

### Logging

```python
import logging
from utils.logger_setup import setup_universal_logging

setup_universal_logging(default_level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Universal logging is ready!")
```

### Environment Variables

```python
from utils.file_ops import get_env_variable

# Loads from .env if present
db_url = get_env_variable("DATABASE_URL", required=True)
```

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
