"""JSONL session line parser."""

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MAX_SUMMARY_LENGTH = 2000

# Tool-result text gets pre-truncated to this before the broader
# ``MAX_SUMMARY_LENGTH`` cap kicks in — a Read of a 5 KB file would
# otherwise drown out everything else in the rolling context window.
_TOOL_RESULT_PREVIEW_MAX = 300

# Event types to skip (noise)
SKIP_TYPES = {
    "progress",
    "system",
    "file-history-snapshot",
    "update",
    "last-prompt",
    "queue-operation",
    "attachment",
}


def _basename(path: str) -> str:
    """Last path component, for compact tool labels."""
    if not path:
        return ""
    return path.rsplit("/", 1)[-1] or path


def _format_tool_use(block: dict[str, object]) -> str:
    """Render a ``tool_use`` block as a short ``[Tool arg]`` label.

    The companion uses these to narrate what the assistant *did*
    between text turns ("read view.py", "ran make checkall"). Keep
    each label compact — one per tool call ends up in the rolling
    context window so verbose labels add up fast.
    """
    name = str(block.get("name", "")) or "tool"
    raw_input = block.get("input", {})
    inp: dict[str, object] = raw_input if isinstance(raw_input, dict) else {}

    if name == "Read":
        path = _basename(str(inp.get("file_path", "")))
        return f"[Read {path}]" if path else "[Read]"
    if name == "Edit":
        path = _basename(str(inp.get("file_path", "")))
        return f"[Edit {path}]" if path else "[Edit]"
    if name == "Write":
        path = _basename(str(inp.get("file_path", "")))
        return f"[Write {path}]" if path else "[Write]"
    if name == "Bash":
        cmd = str(inp.get("command", "")).strip().split("\n", 1)[0]
        if len(cmd) > 80:
            cmd = cmd[:79] + "…"
        return f"[Bash: {cmd}]" if cmd else "[Bash]"
    if name == "Grep":
        pat = str(inp.get("pattern", ""))
        path = _basename(str(inp.get("path", "")))
        if pat and path:
            return f"[Grep {pat!r} in {path}]"
        if pat:
            return f"[Grep {pat!r}]"
        return "[Grep]"
    if name == "Glob":
        pat = str(inp.get("pattern", ""))
        return f"[Glob {pat}]" if pat else "[Glob]"
    if name == "WebFetch":
        url = str(inp.get("url", ""))
        return f"[WebFetch {url}]" if url else "[WebFetch]"
    if name == "WebSearch":
        q = str(inp.get("query", ""))
        return f"[WebSearch {q!r}]" if q else "[WebSearch]"
    if name == "Agent":
        desc = str(inp.get("description", "")).strip()
        return f"[Agent: {desc}]" if desc else "[Agent]"
    if name == "TodoWrite":
        todos = inp.get("todos", [])
        n = len(todos) if isinstance(todos, list) else 0
        return f"[TodoWrite {n} items]"
    return f"[{name}]"


def _extract_tool_result_text(block: dict[str, object]) -> str:
    """Pull a short text preview out of a single ``tool_result`` block."""
    body = block.get("content", "")
    text = ""
    if isinstance(body, str):
        text = body
    elif isinstance(body, list):
        chunks: list[str] = []
        for sub in body:
            if not isinstance(sub, dict):
                continue
            if sub.get("type") == "text":
                chunks.append(str(sub.get("text", "")))
        text = " ".join(chunks)
    text = text.strip().replace("\n", " ")
    if len(text) > _TOOL_RESULT_PREVIEW_MAX:
        text = text[: _TOOL_RESULT_PREVIEW_MAX - 1] + "…"
    return text


@dataclass
class SessionEvent:
    """A parsed session event relevant for commentary."""

    event_type: str
    role: str
    summary: str
    timestamp: str
    is_tool_result: bool = False
    """``True`` for ``role=user`` events whose content is purely SDK
    tool-result feedback (no human text). Lets the coalescer
    distinguish a real human message (which closes a turn) from a
    tool-result coming back from the assistant's prior tool_use."""
    touched_paths: list[str] = field(default_factory=list)
    """File paths the assistant edited / wrote / created in this turn,
    extracted from ``Edit`` / ``Write`` / ``MultiEdit`` / ``NotebookEdit``
    tool_use blocks. Absolute paths as the SDK reports them. Used by
    :mod:`commentary.changes` to attach diffs to the user prompt;
    coalesced (de-duped, order-preserving) when the turn coalescer
    merges multi-block turns."""


def _extract_user_text(content: str | list[dict[str, object]]) -> str:
    """Extract text + tool-result previews from a user message.

    A user message can be:
    * Plain text (the developer typed something) → return as-is.
    * A list of ``tool_result`` blocks (the SDK feeding tool output back
      to the assistant) → render each as ``[result: <preview>]``.
    * Mixed (rare) → text first, then result previews.
    """
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type", "")
        if block_type == "text":
            text = str(block.get("text", "")).strip()
            if text:
                parts.append(text)
        elif block_type == "tool_result":
            preview = _extract_tool_result_text(block)
            if block.get("is_error"):
                parts.append(f"[result error: {preview}]" if preview else "[result error]")
            else:
                parts.append(f"[result: {preview}]" if preview else "[result]")

    return " ".join(parts)


_FILE_EDITING_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})


def _extract_touched_paths(content: str | list[dict[str, object]]) -> list[str]:
    """Pull file paths from edit/write tool_use blocks, in arrival order.

    Reads/Greps/Globs intentionally do *not* count as touched — the
    diff layer only wants paths the assistant actually changed.
    De-duped while preserving first-seen order so a multi-block turn
    that edits the same file twice surfaces it once.
    """
    if not isinstance(content, list):
        return []
    seen: set[str] = set()
    paths: list[str] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        if str(block.get("name", "")) not in _FILE_EDITING_TOOLS:
            continue
        raw_input = block.get("input", {})
        if not isinstance(raw_input, dict):
            continue
        path = str(raw_input.get("file_path", "")).strip()
        if not path or path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def _extract_assistant_text(content: str | list[dict[str, object]]) -> str:
    """Extract text + tool-call labels from an assistant message.

    Tool-use blocks become compact ``[Read foo.py]`` / ``[Bash: ls]``
    tags inline with any text the assistant produced, so the companion
    sees the *work* the assistant did instead of just its prose.
    """
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type", "")
        if block_type == "text":
            text = str(block.get("text", "")).strip()
            if text:
                parts.append(text)
        elif block_type == "tool_use":
            parts.append(_format_tool_use(block))

    return " ".join(parts)


def _truncate(text: str, max_len: int = MAX_SUMMARY_LENGTH) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def parse_jsonl_line(line: str) -> SessionEvent | None:
    """Parse a single JSONL line into a SessionEvent.

    Surfaces user text + tool-result previews and assistant text + tool
    activity tags. Skips bookkeeping lines listed in ``SKIP_TYPES``.
    """
    try:
        data = json.loads(line.strip())
    except (json.JSONDecodeError, ValueError):
        logger.debug("Failed to parse JSONL line")
        return None

    if not isinstance(data, dict):
        return None

    event_type = data.get("type", "")
    if event_type in SKIP_TYPES:
        return None

    message = data.get("message", {})
    if not isinstance(message, dict):
        return None

    role = message.get("role", "unknown")
    content = message.get("content", "")

    is_tool_result = False
    touched_paths: list[str] = []
    if role == "user":
        summary = _truncate(_extract_user_text(content))
        # If every block in the user content is a tool_result, this is
        # SDK tool-output feedback rather than a fresh human turn.
        if isinstance(content, list) and content:
            is_tool_result = all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
    elif role == "assistant":
        summary = _truncate(_extract_assistant_text(content))
        touched_paths = _extract_touched_paths(content)
    else:
        return None

    if not summary:
        return None

    timestamp = data.get("timestamp", "")

    return SessionEvent(
        event_type=event_type,
        role=role,
        summary=summary,
        timestamp=timestamp,
        is_tool_result=is_tool_result,
        touched_paths=touched_paths,
    )
