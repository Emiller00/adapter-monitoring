from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_json(path: str | Path, value: Any) -> Path:
    output = Path(path)
    _atomic_text(output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    return output


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    output = Path(path)
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    _atomic_text(output, text)
    return output


def write_text(path: str | Path, text: str) -> Path:
    output = Path(path)
    _atomic_text(output, text)
    return output


def portable_path(path: str | Path, root: str | Path) -> str:
    target, base = Path(path).resolve(), Path(root).resolve()
    try:
        return str(target.relative_to(base))
    except ValueError:
        return str(target)


def artifact_record(path: str | Path, root: str | Path) -> dict[str, Any]:
    target = Path(path)
    return {
        "path": portable_path(target, root),
        "sha256": sha256_file(target),
        "bytes": target.stat().st_size,
    }
