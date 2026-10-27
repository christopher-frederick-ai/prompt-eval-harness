"""Append-only JSONL storage. Append-only + keyed records is what makes runs resumable."""

from __future__ import annotations

import json
import threading
from pathlib import Path

_lock = threading.Lock()


def append_jsonl(path: Path, record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False)
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def latest_by_key(path: Path) -> dict[str, dict]:
    """Last record wins, so a retry after an error supersedes the failed attempt."""
    return {r["key"]: r for r in read_jsonl(path)}
