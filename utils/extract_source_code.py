"""
extract_source_code.py

This module provides utilities for cleaning up project directories by removing
unwanted files and directories, and clearing Jupyter Notebook outputs.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

# The specific file extensions to keep (lowercase for comparison)
ALLOWED_EXTENSIONS = {
    ".ipynb",
    ".py",
    ".sh",
    ".html",
    ".css",
    ".js",
    ".json",
    ".svg",
    ".png",
}


def clear_jupyter_outputs(file_path: str) -> None:
    """Parses a Jupyter Notebook as JSON and clears cell outputs/execution counts.

    Args:
        file_path (str): The path to the Jupyter Notebook file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            notebook = json.load(f)

        changed = False
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                if cell.get("outputs"):
                    cell["outputs"] = []
                    changed = True
                if cell.get("execution_count") is not None:
                    cell["execution_count"] = None
                    changed = True

        if changed:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(notebook, f, indent=1)
            print(f"Cleared outputs: {file_path}")

    except Exception as e:
        print(f"Error processing notebook {file_path}: {e}")


def contains_template_html(dir_path: str) -> bool:
    """Checks if a directory recursively contains any '*template*.html' file.

    Args:
        dir_path (str): The path to the directory to check.

    Returns:
        bool: True if a matching template file is found, False otherwise.
    """
    # Convert string path to a Path object if it isn't one already
    path = Path(dir_path)

    # rglob yields a generator of Path objects matching the pattern
    # We use .name.lower() to ensure the check is case-insensitive
    for file in path.rglob("*"):
        if (
            file.is_file()
            and "template" in file.name.lower()
            and file.suffix == ".html"
        ):
            return True

    return False


def should_delete_dir(name: str, path: str) -> bool:
    """Encapsulates the logic for directory deletion criteria.

    Args:
        name (str): The name of the directory.
        path (str): The full path to the directory.

    Returns:
        bool: True if the directory should be deleted, False otherwise.
    """
    name_lower = name.lower()

    if name_lower == "output":
        return True

    if name_lower == "input":
        # Delete if it DOES NOT contain the template
        return not contains_template_html(path)

    return False


def clean_directory(target_dir: str) -> None:
    """Walks the directory to clean files, remove folders, and clear notebooks.

    Args:
        target_dir (str): The root directory to start cleaning from.
    """
    for root, dirs, files in os.walk(target_dir, topdown=True):
        # 1. Handle Directories
        # We iterate over a copy of 'dirs' so we can safely modify the original
        for d in list(dirs):
            dir_path = os.path.join(root, d)

            if should_delete_dir(d, dir_path):
                print(f"Deleting folder: {dir_path}")
                shutil.rmtree(dir_path, ignore_errors=True)
                dirs.remove(d)  # Prevents os.walk from entering this deleted dir

        # 2. Handle Files
        for f in files:
            file_path = os.path.join(root, f)
            _, ext = os.path.splitext(f)
            ext_lower = ext.lower()

            # Logic for allowed extensions
            if ext_lower not in ALLOWED_EXTENSIONS:
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Could not delete {file_path}: {e}")
                continue  # Move to the next file

            # Logic for notebooks
            if ext_lower == ".ipynb":
                clear_jupyter_outputs(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean directories, prune unneeded files, and clear Jupyter outputs."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Target directory to clean (defaults to current directory)",
    )

    args = parser.parse_args()

    target_directory = os.path.abspath(args.path)
    print(f"Starting cleanup in: {target_directory}\n{'-' * 40}")

    # SECURITY WARNING: Add a small confirmation prompt to prevent accidental data loss
    confirm = input(
        "WARNING: This script will permanently delete files and directories. Continue? (y/n): "
    )
    if confirm.lower() == "y":
        clean_directory(target_directory)
        print(f"{'-' * 40}\nCleanup complete.")
    else:
        print("Aborted.")
