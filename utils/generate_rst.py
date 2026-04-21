"""
generate_rst.py

A utility module for generating reStructuredText (.rst) files for Python modules
to be used with Sphinx documentation.

This script scans a given directory for Python modules (excluding specified directories)
and automatically creates corresponding .rst files using the `.. automodule::` directive.
These files are intended to be included in a Sphinx documentation build.

Usage:
    Run this script as a standalone module to generate .rst files:

        $ export PYTHONPATH=$PYTHONPATH:.
        $ python utils/generate_rst.py

    This will:
        - Search for all `.py` files (excluding `__init__.py`) in the specified source directory.
        - Skip excluded directories like `tests`, `build`, and `__pycache__`.
        - Output `.rst` files to the appropriate subdirectories in `source/`.

Functions:
    - generate_module_rst(module_path, output_dir):
        Generates a .rst file for a single module using the Sphinx automodule directive.

    - find_modules(root_dir, current_dir, exclude_dirs=None):
        Recursively finds Python modules in a directory, excluding specified directories.

Note:
    The generated .rst files are intended to be included in the Sphinx build.
"""

import os
from pathlib import Path

from utils.file_ops import get_git_repo_name, get_git_root

PROJECT_NAME = get_git_repo_name()

# List of subdirectory names to exclude from Sphinx documentation and sys.path
EXCLUDED_DIRS = [
    ".git",
    ".github",
    ".idea",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".venv",
    ".virtual_documents",
    ".vscode",
    f"{PROJECT_NAME}.egg-info",
    "__pycache__",
    "build",
    "docs",
    "environments",
    "examples",
    "input",
    "img",
    "js",
    "output",
    "tests",
    "third",
    "source/_static",
    "source/_templates",
    "source/reports",
]


def generate_module_rst(module_path: str, output_dir: str) -> None:
    """Generates a reStructuredText file for a Python module.

    Args:
        module_path (str): The relative path to the Python module (e.g., 'utils/file_ops').
        output_dir (str): The directory where the .rst file will be created.
    """
    os.makedirs(output_dir, exist_ok=True)
    module_name = os.path.basename(module_path)
    # The automodule directive needs the importable name.
    # If it's in a subdirectory like utils/, it should be 'utils.file_ops'
    import_path = module_path.replace(os.sep, ".")

    filename = os.path.join(output_dir, f"{module_name}.rst")
    escape_character_module_name = module_name.replace("_", "\\_")
    with open(filename, "w") as f:
        f.write(f"{escape_character_module_name} Module\n")
        f.write("=" * (len(escape_character_module_name) + 7) + "\n\n")
        f.write(f".. automodule:: {import_path}\n")
        f.write("   :members:\n")
        f.write("   :undoc-members:\n")
        f.write("   :show-inheritance:\n")
    print(f"Generated: {filename}")


def find_modules(
    root_dir: str, current_dir: str, exclude_dirs: list[str] | None = None
) -> list[str]:
    """Recursively finds Python modules (.py files) within a given root directory.

    Args:
        root_dir (str): The root directory of the project.
        current_dir (str): The current directory to search.
        exclude_dirs (list, optional): A list of directory names to exclude.
            Defaults to None.

    Returns:
        list: A list of module relative paths (e.g., ['utils/file_ops']).
    """
    modules = []
    if exclude_dirs is None:
        exclude_dirs = []

    for item in os.listdir(current_dir):
        item_path = os.path.join(current_dir, item)
        rel_item_path = os.path.relpath(item_path, root_dir)

        if os.path.isdir(item_path):
            if item in exclude_dirs or rel_item_path in exclude_dirs:
                continue
            modules.extend(find_modules(root_dir, item_path, exclude_dirs))
        elif item.endswith(".py") and item != "__init__.py":
            relative_path = os.path.relpath(item_path, root_dir)
            modules.append(relative_path[:-3])  # Remove .py
    return modules


def run_generator(git_root: str | None, excluded_directories: list[str]) -> None:
    """Recursively finds Python modules and generates corresponding .rst files.

    Args:
        git_root (str): The root directory of the project.
        excluded_directories (list): A list of directory names to exclude.
    """
    if git_root is None:
        print("No git root found, cannot generate documentation.")
        return

    # Find all modules and packages
    modules_to_document = find_modules(git_root, git_root, excluded_directories)

    if not modules_to_document:
        print(f"No Python modules or packages found in: {git_root}")
    else:
        for module_rel_path in sorted(modules_to_document):
            # Determine output directory based on module location
            parts = Path(module_rel_path).parts
            if module_rel_path in [
                os.path.join("source", "conf"),
            ]:
                output_subdir = os.path.join(git_root, "source", "reference")
                # Adjust module path for these
                module_rel_path = os.path.basename(module_rel_path)
            elif parts[0] == "utils":
                output_subdir = os.path.join(git_root, "source", "utils")
            else:
                # Default to api/ for others
                output_subdir = os.path.join(git_root, "source", "api")

            generate_module_rst(module_rel_path, output_subdir)

        print("\nDocumentation generation complete.")


if __name__ == "__main__":
    run_generator(get_git_root(), EXCLUDED_DIRS)
