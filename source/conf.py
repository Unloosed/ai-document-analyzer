# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import logging
import os
import shutil
import sys

from utils.file_ops import get_git_repo_name, get_git_root
from utils.generate_rst import EXCLUDED_DIRS, run_generator

_PROJECT_ROOT = get_git_root()
if _PROJECT_ROOT is None:
    _PROJECT_ROOT = os.getcwd()

PROJECT_ROOT: str = _PROJECT_ROOT
PROJECT_NAME: str = get_git_repo_name()
DESTINATION: str = os.path.join(PROJECT_ROOT, "source/reports")
sys.path.insert(0, PROJECT_ROOT)
print(f"Root directory: {PROJECT_ROOT}")


def add_all_subdirectories_to_path(
    root_dir: str | None = PROJECT_ROOT, excluded_dirs: list[str] = EXCLUDED_DIRS
) -> None:
    """Recursively adds all subdirectories within the root directory to sys.path.

    This is so that the .rst files and Sphinx can see modules in other directories.

    Args:
        root_dir (str): The root directory to start searching from. Defaults to
            PROJECT_ROOT.
        excluded_dirs (list, optional): A list of directory names to exclude.
            Defaults to EXCLUDED_DIRS.
    """
    if root_dir is None:
        return

    excluded_dirs_set = set(excluded_dirs)

    for root, dirs, _files in os.walk(root_dir):
        # Modify dirs in-place to prevent os.walk from descending into excluded directories
        dirs[:] = [
            d
            for d in dirs
            if d not in excluded_dirs_set
            and os.path.relpath(os.path.join(root, d), root_dir)
            not in excluded_dirs_set
        ]
        sys.path.append(root)


INCLUDED_NOTEBOOKS: list[
    str
] = [  # Notebooks to include in the source/reports/ directory
    # "example_notebook.ipynb",
]

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def clear_destination() -> None:
    """Removes old copies from the destination folder.

    Raises:
        Exception: If the destination folder cannot be cleared.
    """
    try:
        shutil.rmtree(DESTINATION, ignore_errors=True)
        os.makedirs(DESTINATION, exist_ok=True)
        logging.info(f"Cleared and recreated directory: {DESTINATION}")
    except Exception as e:
        logging.error(f"Failed to clear destination folder: {e}")
        raise


def is_included_notebook(filename: str) -> bool:
    """Checks if a file is a specified Jupyter Notebook.

    Args:
        filename (str): The name of the file to check.

    Returns:
        bool: True if the file is in INCLUDED_NOTEBOOKS, False otherwise.
    """
    return filename in INCLUDED_NOTEBOOKS


def should_skip_directory(directory: str) -> bool:
    """Checks if a directory should be skipped based on EXCLUDED_DIRS.

    Args:
        directory (str): The directory name to check.

    Returns:
        bool: True if the directory should be skipped, False otherwise.
    """
    return any(excluded in directory for excluded in EXCLUDED_DIRS)


def copy_notebooks() -> None:
    """Walks through project directories and copies specified .ipynb files.

    Raises:
        Exception: If a file copy operation fails.
    """
    try:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            # Convert root to absolute path for reliable comparison
            abs_root = os.path.abspath(root)

            # Skip processing files within the destination directory
            if abs_root.startswith(os.path.abspath(DESTINATION)):
                continue

            # Exclude directories that should be skipped
            dirs[:] = [
                d
                for d in dirs
                if not should_skip_directory(d)
                and os.path.abspath(os.path.join(root, d))
                != os.path.abspath(DESTINATION)
            ]

            for file in files:
                if is_included_notebook(file):
                    # Get source path and relative path
                    src = os.path.join(root, file)
                    relative_path = os.path.relpath(root, PROJECT_ROOT)

                    # Create destination directory in source/
                    dest_dir = os.path.join(DESTINATION, relative_path)
                    os.makedirs(dest_dir, exist_ok=True)
                    shutil.copy(src, dest_dir)  # Copy the file
                    logging.info(f"Copied {src} to {dest_dir}")
                else:
                    src = os.path.join(root, file)
                    # logging.info(f"Skipped {src} (not in the included list)")
    except Exception as e:
        logging.error(f"Failed during notebook copy: {e}")
        raise


def import_notebooks() -> None:
    """Executes the clear and copy process for Jupyter Notebooks to be included.

    Clears the destination directory and then copies the specified notebooks.
    """
    clear_destination()
    copy_notebooks()
    logging.info("All notebooks successfully copied.")


def main() -> None:
    """Sets up working directories and imports notebooks.

    This is the main entry point for the custom configuration logic in conf.py.
    """
    add_all_subdirectories_to_path()
    run_generator(PROJECT_ROOT, EXCLUDED_DIRS)
    import_notebooks()


# if __name__ == "__main__":
# Unfortunately the guard does means it won't run using the make file (terminal `make html`)
main()


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = PROJECT_NAME
copyright = "2026, Michael Evans"
author = "Michael Evans"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",  # To support NumPy and Google style docstrings, REMOVE if using reST format
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx_copybutton",  # Adds a copy button for code blocks
    "myst_parser",  # Allows Sphinx to read markdown files (README.md)
    # 'myst_nb', # Use Myst for notebooks (doesn't seem to work properly, use pandoc instead)
    "nbsphinx",  # Allows Sphinx to document Jupyter Notebook files
    "pydata_sphinx_theme",  # Pydata theme (used by pandas and XlsxWriter)
    # 'sphinx_rtd_theme', # Readthedocs theme
]


nbsphinx_execute = (
    "never"  # 'never' if you don't want to execute notebooks during build ('auto')
)


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "jinja2": ("https://jinja.palletsprojects.com/en/stable/", None),
    # 'plotly': ('https://plotly.com/python/all-about-plotly/', None), # No .inv found
    # 'looker_sdk': ('https://looker-open-source.github.io/looker-sdk-py/', None), # No .inv found
    # Add other libraries as needed
}

templates_path = ["_templates"]
exclude_patterns: list[str] = []

todo_include_todos = True

autodoc_mock_imports = ["invoke"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
# html_theme = 'sphinx_rtd_theme' # Readthedocs theme
html_static_path = ["_static"]
for path in html_static_path:
    if not os.path.exists(path):
        os.makedirs(path)
