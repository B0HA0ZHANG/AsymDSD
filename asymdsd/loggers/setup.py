import logging
import sys
from typing import TextIO

DEFAULT_LOGGER_NAME = "asymdsd"


def setup_logger(
    name: str = DEFAULT_LOGGER_NAME,
    level: int | str | None = logging.INFO,
    info_output: str | TextIO | None = sys.stdout,
    warn_output: str | TextIO | None = sys.stderr,
) -> None:
    if level is None:
        level = logging.INFO

    logging.captureWarnings(True)
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level)

    def create_handler(output, default_stream):
        if isinstance(output, str):  # File path
            return logging.FileHandler(output)
        elif output is None:
            return logging.StreamHandler(default_stream)
        else:  # Assume a TextIO stream
            return logging.StreamHandler(output)

    info_handler = create_handler(info_output, sys.stdout)
    info_handler.setLevel(logging.INFO)
    # info_handler.addFilter(lambda record: record.levelno == logging.INFO)
    logger.addHandler(info_handler)

    # WARNING and above handler
    warn_handler = create_handler(warn_output, sys.stderr)
    warn_handler.setLevel(logging.WARNING)
    logger.addHandler(warn_handler)


def get_default_logger() -> logging.Logger:
    return logging.getLogger(DEFAULT_LOGGER_NAME)
