"""Per-companion JSONL transcript of every LLM commentary call.

Disabled by default. The Qt entry point calls :func:`enable` with
``<companion_dir>/prompts.jsonl`` when ``--debug`` is set; until then
:func:`log_call` is a no-op so production runs pay nothing.

Each line is a single JSON object: timestamp, kind (``event`` /
``idle`` / ``reply``), provider, model, system prompt, user prompt,
and the model's raw response (``null`` when the call failed).
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_lock = threading.Lock()
_path: Path | None = None


def enable(path: Path) -> None:
    """Start appending every commentary call to ``path``.

    Creates parent directories if needed. Subsequent calls overwrite
    the target — only the most recently set path receives writes.
    """
    global _path
    path.parent.mkdir(parents=True, exist_ok=True)
    _path = path


def log_call(
    *,
    kind: str,
    system: str,
    user: str,
    response: str | None,
    provider: str,
    model: str,
) -> None:
    """Append one record to the active transcript. No-op when disabled."""
    if _path is None:
        return
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "kind": kind,
        "provider": provider,
        "model": model,
        "system": system,
        "user": user,
        "response": response,
    }
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _lock, _path.open("a", encoding="utf-8") as f:
        f.write(line)
