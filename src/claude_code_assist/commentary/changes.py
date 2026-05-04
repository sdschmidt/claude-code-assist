"""Render git-diff and surrounding-code context for a focal event.

Layered above ``monitor.parser`` (which extracts touched file paths
from ``tool_use`` blocks) and below ``commentary.prompts`` (which only
splices the rendered string into the user prompt). Keeping git
mechanics in their own module means the prompt builder stays pure
text substitution and the backend stays a tick-loop scheduler.

Responsibilities:

* Filter touched paths against a denylist (secrets, lockfiles,
  binaries) before any disk read.
* Run ``git diff HEAD -- <path>`` per file and parse the unified
  diff into hunks so truncation never cuts mid-hunk.
* Apply per-file and global hunk-line budgets, always keeping the
  first hunk in full.
* Fall back to a synthetic full-file diff for new / untracked files
  so the model sees the actual content.
* Render one small surrounding-code window for the largest hunk in
  the largest diff so commentary can anchor on real line numbers.

Public surface is intentionally one function: :func:`build_change_context_block`.
All paths return ``""`` on any failure (no git, not a repo, file
deleted, etc.) so commentary generation degrades gracefully.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Denylist — applied before any disk read or git invocation.
# ---------------------------------------------------------------------------


_DENY_NAME_GLOBS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.env",
    "*.lock",
    "*.min.*",
    "*.map",
    "*.pem",
    "*.key",
    "id_rsa*",
    "*.p12",
    "*credentials*",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.webp",
    "*.ico",
    "*.svg",
    "*.bin",
    "*.so",
    "*.dylib",
    "*.dll",
    "*.exe",
    "*.zip",
    "*.tar",
    "*.tar.gz",
    "*.tgz",
    "*.gz",
    "*.pdf",
    "*.mp3",
    "*.mp4",
    "*.mov",
    "*.pyc",
    "*.class",
    "*.o",
)

_DENY_PATH_SEGMENTS: tuple[str, ...] = (
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".tox",
    "target",
    ".next",
    ".gradle",
    ".idea",
    ".vscode",
)

_MAX_FILE_BYTES = 1_000_000
"""Skip files larger than ~1 MB on disk before even running git diff."""


def _is_denied(path: Path, cwd: Path) -> bool:
    """True if the path should never be sent to the LLM.

    Checks both filename globs (secrets, binaries, lockfiles) and any
    parent-segment denylist (``.git``, ``node_modules``, …). Files
    above ``_MAX_FILE_BYTES`` on disk are also denied — large
    minified files / generated artifacts dominate the prompt budget
    without adding signal.
    """
    name = path.name
    if any(fnmatch(name, glob) for glob in _DENY_NAME_GLOBS):
        return True
    try:
        rel_parts = path.resolve().relative_to(cwd.resolve()).parts
    except ValueError:
        # Path escapes cwd — treat as denied; we never diff outside the project.
        return True
    if any(seg in _DENY_PATH_SEGMENTS for seg in rel_parts[:-1]):
        return True
    if path.exists() and path.is_file():
        try:
            if path.stat().st_size > _MAX_FILE_BYTES:
                return True
        except OSError:
            return True
    return False


def _resolve(raw: str, cwd: Path) -> Path | None:
    """Resolve ``raw`` (absolute or cwd-relative) to a Path inside cwd.

    Returns ``None`` when the path can't be normalised under ``cwd``
    (escape attempt, empty string, etc.) so the caller can skip it.
    """
    if not raw:
        return None
    try:
        p = Path(raw)
        p = (cwd / p).resolve() if not p.is_absolute() else p.resolve()
    except (OSError, ValueError):
        return None
    try:
        p.relative_to(cwd.resolve())
    except ValueError:
        return None
    return p


def _rel_to_cwd(path: Path, cwd: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(cwd.resolve()))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Git plumbing
# ---------------------------------------------------------------------------


def _run_git(args: list[str], cwd: Path, timeout: float = 3.0) -> str | None:
    """Run ``git <args>`` in ``cwd`` and return stdout, or None on failure.

    A nonzero exit with empty stdout returns ``None`` so callers can
    detect "no diff" cleanly. ``--no-index`` exits nonzero by design
    when files differ, so we keep stdout in that case.
    """
    try:
        result = subprocess.run(  # noqa: S603 — cmd is fixed args, cwd is project root
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0 and not result.stdout:
        return None
    return result.stdout


def _file_diff(path: Path, cwd: Path) -> str | None:
    """Best-effort unified diff for ``path`` against HEAD.

    Falls back to a synthetic full-add diff (``--no-index /dev/null``)
    for untracked files so the model still sees newly-added content.
    Returns ``None`` when neither produces output.
    """
    rel = _rel_to_cwd(path, cwd)
    if rel is None:
        return None
    out = _run_git(["diff", "--no-color", "-U3", "HEAD", "--", rel], cwd)
    if out and out.strip():
        return out
    # Untracked / no-HEAD-yet fallback. Treats the whole file as added.
    out = _run_git(["diff", "--no-color", "-U3", "--no-index", "/dev/null", rel], cwd)
    if out and out.strip():
        return out
    return None


# ---------------------------------------------------------------------------
# Hunk parsing + truncation
# ---------------------------------------------------------------------------


_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


@dataclass
class _Hunk:
    """One ``@@``-delimited hunk inside a unified diff."""

    header: str
    body: list[str] = field(default_factory=list)
    new_start: int = 1
    new_count: int = 1

    def line_count(self) -> int:
        """Header line + body lines — the unit of the truncation budget."""
        return 1 + len(self.body)


def _parse_hunks(diff_text: str) -> tuple[list[str], list[_Hunk]]:
    """Split a unified diff into ``(preamble, hunks)``.

    Preamble is whatever comes before the first ``@@`` (``diff --git``,
    ``index``, ``---``, ``+++``). It's returned separately so callers
    can drop it — the model already knows which file is being shown
    from the ``<diff path="...">`` wrapper.
    """
    preamble: list[str] = []
    hunks: list[_Hunk] = []
    current: _Hunk | None = None
    for line in diff_text.splitlines():
        if line.startswith("@@"):
            if current is not None:
                hunks.append(current)
            m = _HUNK_HEADER_RE.match(line)
            new_start = int(m.group(1)) if m else 1
            new_count = int(m.group(2)) if m and m.group(2) else 1
            current = _Hunk(header=line, new_start=new_start, new_count=new_count)
        elif current is None:
            preamble.append(line)
        else:
            current.body.append(line)
    if current is not None:
        hunks.append(current)
    return preamble, hunks


def _truncate_to_budget(hunks: list[_Hunk], budget: int) -> tuple[list[_Hunk], int]:
    """Keep whole hunks until ``budget`` is spent.

    The first hunk is always kept in full even if it overruns the
    budget — the first change is usually the most informative and
    cutting it mid-hunk would leave the model with half a story.
    Returns the kept hunks and the total line count consumed.
    """
    if not hunks:
        return [], 0
    kept: list[_Hunk] = [hunks[0]]
    used = hunks[0].line_count()
    for h in hunks[1:]:
        size = h.line_count()
        if used + size > budget:
            break
        kept.append(h)
        used += size
    return kept, used


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_diff_block(rel: str, kept: list[_Hunk], total: int) -> str:
    """Render kept hunks back into the unified-diff body, with an omit marker."""
    lines: list[str] = []
    for h in kept:
        lines.append(h.header)
        lines.extend(h.body)
    omitted = total - len(kept)
    if omitted > 0:
        suffix = "" if omitted == 1 else "s"
        lines.append(f"(diff truncated, {omitted} more hunk{suffix} omitted)")
    body = "\n".join(lines)
    return f'<diff path="{rel}">\n{body}\n</diff>'


def _render_surrounding_window(
    path: Path,
    hunk: _Hunk,
    *,
    window: int,
    cwd: Path,
) -> str:
    """Line-numbered ±window view around ``hunk`` in the working-tree file.

    Used once per event for the single largest hunk — gives the model a
    stable anchor to cite (``api.py:47``) without having to resolve
    diff line numbers itself.
    """
    if not path.exists() or not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    file_lines = text.splitlines()
    if not file_lines:
        return ""
    start = max(1, hunk.new_start - window)
    end = min(len(file_lines), hunk.new_start + max(hunk.new_count, 1) + window - 1)
    if start > end:
        return ""
    rel = _rel_to_cwd(path, cwd) or path.name
    width = len(str(end))
    body = "\n".join(f"{str(i).rjust(width)}  {file_lines[i - 1]}" for i in range(start, end + 1))
    return f"SURROUNDING CODE ({rel}:{start}-{end}):\n{body}"


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def build_change_context_block(
    paths: list[str] | None,
    cwd: Path | None,
    *,
    per_file_diff_lines: int = 200,
    total_diff_lines: int = 400,
    surrounding_window_lines: int = 10,
) -> str:
    """Render the change-context block for the focal event's user prompt.

    Returns either the empty string (no usable diff) or a self-contained
    block ending with ``\\n\\n`` so the surrounding template can drop the
    placeholder in directly without worrying about blank lines.

    Args:
        paths: Absolute or cwd-relative file paths the focal turn
            touched, in arrival order. Duplicates and denied paths are
            filtered.
        cwd: Project working directory (the same directory the user
            launched ``companion`` from). When ``None`` — text-watcher
            mode, no project — the function returns ``""`` immediately.
        per_file_diff_lines: Hard cap on diff lines per file. The
            first hunk is always kept whole even if it overruns.
        total_diff_lines: Soft cap on total diff lines across all
            files in this turn. Each file gets ``total // n`` as a fair
            share, capped by ``per_file_diff_lines``.
        surrounding_window_lines: ± lines around the largest single
            hunk to render as the anchor window. Set to 0 to skip.
    """
    if cwd is None or not paths:
        return ""

    seen: set[str] = set()
    resolved: list[Path] = []
    for raw in paths:
        p = _resolve(raw, cwd)
        if p is None or _is_denied(p, cwd):
            continue
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(p)
    if not resolved:
        return ""

    per_file_share = max(40, total_diff_lines // len(resolved))
    blocks: list[str] = []
    largest: tuple[Path, _Hunk] | None = None
    largest_size = 0
    remaining = total_diff_lines

    for p in resolved:
        rel = _rel_to_cwd(p, cwd) or p.name
        if remaining <= 0:
            blocks.append(f'<diff path="{rel}">\n(omitted — turn diff budget exhausted)\n</diff>')
            continue
        diff_text = _file_diff(p, cwd)
        if diff_text is None:
            continue
        _, hunks = _parse_hunks(diff_text)
        if not hunks:
            continue
        budget = min(per_file_share, per_file_diff_lines, remaining)
        kept, used = _truncate_to_budget(hunks, budget)
        if not kept:
            continue
        blocks.append(_render_diff_block(rel, kept, total=len(hunks)))
        remaining -= used
        for h in kept:
            size = h.line_count()
            if size > largest_size:
                largest_size = size
                largest = (p, h)

    if not blocks:
        return ""

    body = "\n".join(blocks)
    out = f"FILES TOUCHED THIS TURN — diff against HEAD:\n{body}\n\n"

    if largest is not None and surrounding_window_lines > 0:
        path, hunk = largest
        win = _render_surrounding_window(path, hunk, window=surrounding_window_lines, cwd=cwd)
        if win:
            out += f"{win}\n\n"

    return out
