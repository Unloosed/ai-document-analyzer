"""
This module provides standardized logging configuration for the project.
Includes console output with colors and a rotating file log.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler

from colorlog import ColoredFormatter


def setup_universal_logging(
    default_level: int = logging.INFO, log_file: str = "app.log"
) -> None:
    """Sets up a global logging configuration for the entire project.

    Includes colorized console output and plain text file output.

    Args:
        default_level (int, optional): The logging level for the root logger.
            Defaults to logging.INFO.
        log_file (str, optional): The name of the log file to be created.
            Defaults to "app.log".
    """
    # 1. Create the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(default_level)

    # Clean up any existing handlers (prevents double-logging if called twice)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 2. Define Formats
    # Console format: Level [Module Name] Message
    console_format = (
        "%(log_color)s%(levelname)-8s %(cyan)s[%(name)s]%(reset)s %(blue)s%(message)s"
    )
    console_formatter = ColoredFormatter(
        console_format,
        log_colors={
            "DEBUG": "white",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )

    # File format: Standard ISO timestamp and detailed info
    file_format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    file_formatter = logging.Formatter(file_format)

    # 3. Create Handlers
    # Console (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    # File (Rotating to save space)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=2
    )
    file_handler.setFormatter(file_formatter)

    # 4. Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.debug("Universal logging initialized.")
