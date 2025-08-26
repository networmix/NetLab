"""Logging configuration for NetLab.

This module provides a thin wrapper around Python's ``logging`` to keep
NetLab's logging consistent across subpackages.
"""

from __future__ import annotations

import logging
import os
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger.

    Args:
        name: Logger name, typically ``__name__``.

    Returns:
        A logger instance bound to ``name``.
    """
    return logging.getLogger(name)


def set_global_log_level(level: int | str) -> None:
    """Set the global logging level for NetLab-related loggers.

    Args:
        level: Logging level (e.g., ``logging.DEBUG`` or ``"INFO"``).
    """
    resolved: int
    if isinstance(level, int):
        resolved = level
    else:
        # Map common names to numeric levels; default to INFO on unknown
        try:
            resolved = int(getattr(logging, str(level).upper()))
        except Exception:
            resolved = logging.INFO

    # Configure root logger
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )

    # Set level for NetLab packages explicitly
    for pkg in ("netlab", "metrics"):
        logging.getLogger(pkg).setLevel(resolved)


def configure_from_env(
    var_name: str = "NETLAB_LOG_LEVEL", default: int | str = logging.INFO
) -> None:
    """Configure logging using an environment variable.

    If ``var_name`` is present in the environment, its value is used as the
    desired log level; otherwise ``default`` is applied.

    Args:
        var_name: Environment variable name to read (default: ``NETLAB_LOG_LEVEL``).
        default: Level to use when the variable is unset or invalid.
    """
    value = os.environ.get(var_name)
    if value is None or value.strip() == "":
        set_global_log_level(default)
        return
    set_global_log_level(value.strip())
