"""
Unit tests for the environment management and file operation utilities.
"""

import logging
import os
from pathlib import Path

import pytest

from utils.file_ops import (
    create_directories,
    get_env_variable,
    get_git_repo_name,
    get_git_root,
    zip_files_with_password,
)
from utils.logger_setup import setup_universal_logging


@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    """Sets up logging for the test session."""
    setup_universal_logging(default_level=logging.DEBUG)


def test_create_directories():
    """Tests the creation of default directories."""
    create_directories()
    assert os.path.isdir("input")
    assert os.path.isdir("output")


def test_get_env_variable_existing():
    """Tests retrieval of an existing environment variable."""
    # Set a temporary environment variable for testing
    os.environ["TEST_CREDENTIAL"] = "doisa782ubfdsa0q"
    val = get_env_variable("TEST_CREDENTIAL")
    assert val == "doisa782ubfdsa0q"


def test_get_env_variable_existing_unredacted():
    """Tests retrieval of an existing environment variable without redaction."""
    # Set a temporary environment variable for testing
    os.environ["TEST_CREDENTIAL"] = "doisa782ubfdsa0q"
    val = get_env_variable("TEST_CREDENTIAL", redact=False)
    assert val == "doisa782ubfdsa0q"


def test_get_env_variable_missing_required():
    """Tests that a missing required environment variable raises an exception."""
    from utils.file_ops import MissingEnvironmentVariable

    with pytest.raises(MissingEnvironmentVariable):
        get_env_variable("NON_EXISTENT_VAR", required=True)


def test_get_env_variable_missing_not_required():
    """Tests retrieval of a missing non-required environment variable with a default value."""
    val = get_env_variable("NON_EXISTENT_VAR", required=False, default="default_val")
    assert val == "default_val"


def test_get_git_root():
    """Tests that the correct Git root directory is returned."""
    # Current directory is in a Git repository
    git_root = get_git_root()
    assert git_root is not None
    assert os.path.isdir(git_root)
    assert os.path.exists(os.path.join(git_root, ".git"))

    # Test with a non-existent path
    non_existent_path = "/path/to/non/existent/directory"
    git_root = get_git_root(non_existent_path)
    assert git_root is None


def test_get_git_repo_name():
    """Tests that the correct Git repository name is returned."""
    # Current directory is in a Git repository
    git_repo_name = get_git_repo_name()
    assert git_repo_name is not None

    # Test with a non-existent path
    non_existent_path = "/path/to/non/existent/directory"
    git_repo_name = get_git_repo_name(non_existent_path)
    assert git_repo_name is None


def test_zip_files_with_password():
    """Tests the creation of a password-protected ZIP archive."""
    # Get the directory where THIS test file lives
    current_dir = Path(__file__).parent

    # Anchor your paths!
    # This looks for README.md one level up from the test folder
    readme_path = current_dir.parent / "README.md"

    # Define your output relative to the test folder
    output_dir = current_dir / "test_output"
    zip_filename = "test_zip.zip"
    password = "test_password"

    # Convert Path objects to strings for your function
    files = [str(readme_path), "/path/to/non/existent/directory"]

    zip_files_with_password(files, zip_filename, password, str(output_dir))

    # Assertion using the anchored path
    expected_zip = output_dir / zip_filename
    assert expected_zip.exists()
