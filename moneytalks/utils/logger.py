"""Unified logging configuration using loguru."""

import sys

from loguru import logger

# Remove default handler
logger.remove()

# Add console handler with formatting
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>",
    level="INFO",
)


def get_logger(name: str = "moneytalks"):
    """Get a logger instance with the given name."""
    return logger.bind(name=name)
