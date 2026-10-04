"""Atomic artifact publication and content-based provenance."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from importlib.metadata import distribution
from importlib.util import find_spec
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterator


@contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    """Publish a same-filesystem temporary file only after a successful write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as f:
        temporary = Path(f.name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_text_atomic(path: Path, content: str) -> None:
    with atomic_path(path) as temporary:
        temporary.write_text(content, encoding="utf-8")


def write_json_atomic(path: Path, data: Any) -> None:
    write_text_atomic(path, json.dumps(data, indent=2) + "\n")


def write_csv_atomic(path: Path, data: Any) -> None:
    import pandas as pd

    table = data if isinstance(data, (pd.DataFrame, pd.Series)) else pd.DataFrame(data)
    with atomic_path(path) as temporary:
        table.to_csv(temporary)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def package_versions() -> dict[str, dict[str, Any]]:
    """Identify installed revisions, including current editable package contents."""
    packages = {}
    for name in ("netlab", "topogen", "ngraph", "netgraph-core"):
        package = distribution(name)
        record: dict[str, Any] = {"version": package.version}
        direct_url = package.read_text("direct_url.json")
        if direct_url:
            record["source"] = json.loads(direct_url)
            if record["source"].get("dir_info", {}).get("editable"):
                spec = find_spec(name.replace("-", "_"))
                if spec is None or not spec.submodule_search_locations:
                    raise ValueError(f"Cannot fingerprint editable package {name}")
                files = {}
                for location in spec.submodule_search_locations:
                    root = Path(location)
                    for path in sorted(root.rglob("*")):
                        if path.is_file() and path.suffix in {".py", ".so", ".pyd"}:
                            files[str(path.relative_to(root))] = sha256_file(path)
                record["content_sha256"] = fingerprint(files)
        packages[name] = record
    return packages


def ensure_run_identity(path: Path, inputs: Any) -> None:
    """Bind a resumable output directory to exactly one set of inputs."""
    key = fingerprint(inputs)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous["key"] != key:
            raise ValueError(
                f"Research inputs or dependencies changed: {path}. "
                "Use a new output directory for this experiment."
            )
    else:
        write_json_atomic(path, {"key": key, "inputs": inputs})


def cache_path(artifact: Path) -> Path:
    return artifact.with_suffix(artifact.suffix + ".cache.json")


def cache_matches(artifact: Path, key: str) -> bool:
    try:
        manifest = json.loads(cache_path(artifact).read_text(encoding="utf-8"))
        return manifest["key"] == key and manifest["sha256"] == sha256_file(artifact)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def record_cache(artifact: Path, key: str) -> None:
    write_json_atomic(
        cache_path(artifact), {"key": key, "sha256": sha256_file(artifact)}
    )
