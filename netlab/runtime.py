"""Runtime helpers for locating external executables."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def resolve_executable(
    name: str,
    *,
    explicit: str | None = None,
    env_var: str | None = None,
) -> str | None:
    """Resolve an executable from explicit path, env, PATH, or current venv."""
    if explicit:
        return explicit

    if env_var:
        env_value = os.environ.get(env_var, "").strip()
        if env_value:
            return env_value

    path_value = shutil.which(name)
    if path_value is not None:
        return path_value

    venv_bin = Path(sys.executable).parent / name
    if venv_bin.exists():
        return str(venv_bin)

    return None


def require_executable(
    name: str,
    *,
    explicit: str | None = None,
    env_var: str | None = None,
    display_name: str | None = None,
) -> str:
    """Resolve an executable or raise a clear RuntimeError."""
    resolved = resolve_executable(name, explicit=explicit, env_var=env_var)
    if resolved is not None:
        return resolved

    label = display_name or name
    env_hint = f" or ${env_var}" if env_var else ""
    raise RuntimeError(
        f"{label} executable not found. Set it explicitly{env_hint}, "
        f"put '{name}' on PATH, or install it in the current Python environment."
    )


def resolve_invoke(
    name: str,
    *,
    explicit: str | None = None,
    env_var: str | None = None,
    python_module: str | None = None,
) -> list[str] | None:
    """Resolve a command invocation, optionally falling back to ``sys.executable -m``."""
    resolved = resolve_executable(name, explicit=explicit, env_var=env_var)
    if resolved is not None:
        return [resolved]

    if python_module and sys.executable:
        return [sys.executable, "-m", python_module]

    return None
