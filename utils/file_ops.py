"""
file_ops.py

This module provides utility functions for handling file operations commonly
needed in data processing workflows, including compression, encryption,
directory management, file movement, extraction, and data import.

It supports operations such as:
- Creating password-protected ZIP archives.
- Reading from .env file.
- Copying, moving, and renaming files.
- Creating directories safely.
- Unzipping CSV files from archives.
- Reading Excel files into pandas DataFrames.
- Retrieving and sorting files by name or creation date.
"""

import logging
import os
from pathlib import Path
from typing import List

import pyzipper
from dotenv import load_dotenv
from git import InvalidGitRepositoryError, NoSuchPathError, Repo

logger = logging.getLogger(__name__)

_ENV_LOADED = False


def create_directories(directories: List[str] | None = None) -> None:
    """Creates multiple directories if they don't exist.

    Creates input and output directories by default.

    Args:
        directories (list, optional): A list of directory paths to be created.
            Defaults to ["input", "output"].
    """
    if directories is None:
        directories = ["input", "output"]

    for directory_str in directories:
        directory_path = Path(directory_str)
        directory_path.mkdir(parents=True, exist_ok=True)


def get_git_root(path: str | None = None) -> str | None:
    """Returns the root directory of the Git repository containing the path.

    Args:
        path (str, optional): The path to start searching from. Defaults to the current
            working directory.

    Returns:
        str: The root directory of the Git repository, or None if not in a repository.
    """
    if path is None:
        path = os.getcwd()
    logger.debug("Getting git root for %s", path)
    try:
        # search_parent_directories finds the .git folder by walking up the tree
        git_repo = Repo(path, search_parent_directories=True)
        git_root: str = git_repo.git.rev_parse("--show-toplevel")
        return git_root
    except (InvalidGitRepositoryError, NoSuchPathError):
        # These are the expected "failures" if the path isn't in a repo
        return None


def get_git_repo_name(path: str | None = None) -> str:
    """Returns the name of the current Git repository.

    Returns:
        str: The name of the Git repository, or None if not in a repository.
    """
    git_root = get_git_root(path)
    if git_root is None:
        return None
    return os.path.basename(git_root)


class MissingEnvironmentVariable(RuntimeError):
    """Exception raised when a required environment variable is missing."""

    pass


def load_env_once() -> None:
    """Loads .env file from the project root if it exists.

    Does NOT override existing environment variables. Safe to call multiple times.
    """
    global _ENV_LOADED

    if _ENV_LOADED:
        return

    git_root = get_git_root()
    if git_root is None:
        logger.warning("Could not find git root. Skipping .env load.")
        _ENV_LOADED = True
        return

    env_path = Path(git_root).resolve() / ".env"

    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
        logger.info("Loaded .env from %s", env_path)
    else:
        logger.info("No .env found at %s (this is normal in CI)", env_path)

    _ENV_LOADED = True


def get_env_variable(
    name: str,
    *,
    required: bool = True,
    default: str | None = None,
    redact: bool = True,
) -> str:
    """Fetches an environment variable with strong diagnostics.

    Args:
        name (str): The name of the environment variable.
        required (bool, optional): If True, raises an exception if the variable
            is missing. Defaults to True.
        default (any, optional): The value to return if the variable is missing
            and not required. Defaults to None.
        redact (bool, optional): If True, hides the value in logs. Defaults to True.

    Returns:
        str: The value of the environment variable.

    Raises:
        MissingEnvironmentVariable: If the variable is required and missing.
    """

    load_env_once()

    value = os.environ.get(name)

    if value is None:
        if required:
            logger.critical(
                "Missing required environment variable: %s\n"
                "This usually means:\n"
                "  - Key Vault variable group not mapped with env:\n"
                "  - Secret name mismatch\n"
                "  - Local .env not loaded\n",
                name,
            )
            raise MissingEnvironmentVariable(name)
        return default if default is not None else ""

    if not redact:
        logger.debug("Loaded env var %s = %s", name, value)
    else:
        logger.debug("Loaded env var %s = [REDACTED]", name)

    return value


def zip_files_with_password(
    files: List[str], zip_filename: str, password: str, output_dir: str = "."
) -> None:
    """
    Creates a password-protected ZIP archive containing the specified files.

    Args:
        files (List[str]): A list of file paths to be included in the ZIP archive.
        zip_filename (str): The name of the resulting ZIP file.
        password (str): The password to protect the ZIP archive.
        output_dir (str, optional): The directory where the ZIP file will be saved.
            Defaults to the current directory (".").
    """
    # 1. Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. Combine the directory and filename
    full_zip_path = os.path.join(output_dir, zip_filename)

    secret_password = password.encode("utf-8")

    with pyzipper.AESZipFile(
        full_zip_path,
        "w",
        compression=pyzipper.ZIP_DEFLATED,
        encryption=pyzipper.WZ_AES,
    ) as zf:
        zf.setpassword(secret_password)
        for file in files:
            if os.path.exists(file):
                # Using os.path.basename so the ZIP doesn't contain
                # the full local path of the source files
                zf.write(file, arcname=os.path.basename(file))
            else:
                logger.warning("File not found: %s", file)
