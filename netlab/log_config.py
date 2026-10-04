"""Configure shared logging for NetLab packages."""

from __future__ import annotations

import logging
import os
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a logger for ``name``, usually the caller's ``__name__``."""
    return logging.getLogger(name)


def set_global_log_level(level: int | str) -> None:
    """Set NetLab logger levels, preserving existing application handlers."""
    resolved: int
    if isinstance(level, int):
        resolved = level
    else:
        try:
            resolved = int(getattr(logging, str(level).upper()))
        except (AttributeError, ValueError, TypeError):
            resolved = logging.INFO

    logging.basicConfig(
        level=resolved,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    logging.getLogger("netlab").setLevel(resolved)


def configure_from_env(
    var_name: str = "NETLAB_LOG_LEVEL", default: int | str = logging.INFO
) -> None:
    """Set the level from ``var_name``; use ``default`` if unset, INFO if invalid."""
    value = os.environ.get(var_name)
    if value is None or value.strip() == "":
        set_global_log_level(default)
        return
    set_global_log_level(value.strip())
