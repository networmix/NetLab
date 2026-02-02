"""netlab package: CLI and utilities for building and running NetGraph experiments."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("netlab")
except PackageNotFoundError:
    __version__ = "0.0.0.dev"  # fallback for editable installs before first pip install

# Expose submodules declared in __all__
from . import (
    cli,  # noqa: F401
    comparison,  # noqa: F401
    experiment,  # noqa: F401
    metrics_failure,  # noqa: F401
    scenario,  # noqa: F401
    visualize,  # noqa: F401
)

__all__ = [
    "__version__",
    "cli",
    "comparison",
    "experiment",
    "metrics_failure",
    "scenario",
    "visualize",
]
