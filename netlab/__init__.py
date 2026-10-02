"""NetLab: metrics and experiment tooling for NetGraph scenarios."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("netlab")
except PackageNotFoundError:
    __version__ = "0.0.0.dev"  # Package metadata is unavailable before installation.

__all__ = ["__version__"]
