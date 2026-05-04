"""Session file watcher using watchdog."""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from watchdog.events import FileCreatedEvent, FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from claude_code_assist.monitor.parser import MAX_SUMMARY_LENGTH, SessionEvent, parse_jsonl_line

if TYPE_CHECKING:
    from queue import Queue

    from watchdog.observers.api import BaseObserver

logger = logging.getLogger(__name__)

# Coalescer debounce — how long the assistant has to be silent before
# we flush a buffered turn. 1.5 s is comfortably longer than a typical
# pause between text and the next tool_use, but short enough that
# commentary still feels prompt at end-of-turn.
_COALESCE_DEBOUNCE_SECONDS = 1.5


class _TurnCoalescer:
    """Buffers per-block ``SessionEvent``s into one event per turn.

    Why: Claude Code writes one JSONL line per content block, so a
    single assistant turn ("explain → read → edit → run tests → wrap")
    arrives as 5+ tiny fragments. Reacting to each fragment makes the
    companion narrate every pause; coalescing lets it react to the
    whole turn.

    Flush triggers:
    * A real user message (``role=user`` with non-tool-result content)
      arrives — it's emitted *immediately* (no buffering, so commentary
      on direct address stays snappy) and any prior buffered turn is
      flushed first.
    * The assistant has been silent for ``_COALESCE_DEBOUNCE_SECONDS``
      since the last buffered block.
    """

    def __init__(
        self, output_queue: Queue[SessionEvent], *, debounce_seconds: float = _COALESCE_DEBOUNCE_SECONDS
    ) -> None:
        self._queue = output_queue
        self._debounce = debounce_seconds
        self._lock = threading.Lock()
        self._buffer: list[SessionEvent] = []
        self._timer: threading.Timer | None = None

    def feed(self, event: SessionEvent) -> None:
        """Process one parsed event."""
        with self._lock:
            if event.role == "user" and not event.is_tool_result:
                # Fresh human input — close any open turn first, then
                # emit this event without buffering so the React loop
                # sees it on the next tick.
                self._flush_locked()
                self._queue.put(event)
                return
            self._buffer.append(event)
            self._reset_timer_locked()

    def flush_now(self) -> None:
        """Forced flush — used on shutdown so an in-progress turn isn't lost."""
        with self._lock:
            self._flush_locked()

    def _reset_timer_locked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self._debounce, self._flush_due_to_debounce)
        self._timer.daemon = True
        self._timer.start()

    def _flush_due_to_debounce(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if not self._buffer:
            return
        coalesced = _coalesce(self._buffer)
        self._buffer = []
        if coalesced is not None:
            self._queue.put(coalesced)


def _coalesce(events: list[SessionEvent]) -> SessionEvent | None:
    """Combine same-turn events into one focal SessionEvent.

    Tool_result events get inline ``[result: …]`` framing already
    (via the parser); we just stitch the summaries together in arrival
    order so the LLM sees the turn as one continuous narrative.
    """
    if not events:
        return None
    has_assistant = any(ev.role == "assistant" for ev in events)
    role = "assistant" if has_assistant else events[0].role
    parts: list[str] = []
    for ev in events:
        text = ev.summary.strip()
        if text:
            parts.append(text)
    summary = " ".join(parts)
    if len(summary) > MAX_SUMMARY_LENGTH:
        summary = summary[: MAX_SUMMARY_LENGTH - 1] + "…"
    seen: set[str] = set()
    touched_paths: list[str] = []
    for ev in events:
        for path in ev.touched_paths:
            if path and path not in seen:
                seen.add(path)
                touched_paths.append(path)
    return SessionEvent(
        event_type=events[0].event_type,
        role=role,
        summary=summary,
        timestamp=events[-1].timestamp,
        is_tool_result=False,
        touched_paths=touched_paths,
    )


def encode_project_path(project_path: str) -> str:
    """Encode a project path the way Claude Code stores it.

    Args:
        project_path: Absolute project directory path.

    Returns:
        Encoded path with ``/``, ``_``, and ``.`` replaced by ``-``.
    """
    return re.sub(r"[/_.]", "-", project_path)


def find_newest_session(session_dir: Path) -> Path | None:
    """Find the most recently modified JSONL session file.

    Args:
        session_dir: Directory to search for .jsonl files.

    Returns:
        Path to newest file, or None if no .jsonl files exist.
    """
    jsonl_files = [f for f in session_dir.iterdir() if f.is_file() and f.suffix == ".jsonl"]
    if not jsonl_files:
        return None
    return max(jsonl_files, key=lambda f: f.stat().st_mtime)


class SessionWatcher:
    """Watches Claude Code session files for new events."""

    def __init__(self, session_dir: Path, event_queue: Queue[SessionEvent]) -> None:
        self._session_dir = session_dir
        self._event_queue = event_queue
        self._file_positions: dict[Path, int] = {}
        self._observer: BaseObserver | None = None
        self._coalescer = _TurnCoalescer(event_queue)

    def _process_file(self, path: Path) -> None:
        """Read new lines from a session file and parse them."""
        if not path.exists() or path.suffix != ".jsonl":
            return

        last_pos = self._file_positions.get(path, 0)
        try:
            with path.open("r", encoding="utf-8") as f:
                f.seek(last_pos)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    event = parse_jsonl_line(line)
                    if event is not None:
                        self._coalescer.feed(event)
                self._file_positions[path] = f.tell()
        except Exception:
            logger.exception("Error reading session file %s", path)

    def process_file(self, path: Path) -> None:
        """Read new lines from a session file and parse them.

        Public interface used by the file-system event handler.

        Args:
            path: Path to the JSONL session file to process.
        """
        self._process_file(path)

    def start(self) -> None:
        """Start watching the session directory.

        If the session directory doesn't exist yet (e.g. Claude Code hasn't
        started for this project), polls for up to 60 seconds waiting for it
        to appear, then starts the watchdog observer.
        """
        if not self._session_dir.exists():
            self._poll_for_session_dir()

        if not self._session_dir.exists():
            logger.warning("Session directory not found after polling: %s", self._session_dir)
            return

        newest = find_newest_session(self._session_dir)
        if newest:
            self._file_positions[newest] = newest.stat().st_size
            logger.info("Watching session file: %s", newest)

        handler = _SessionFileHandler(self)
        self._observer = Observer()
        self._observer.schedule(handler, str(self._session_dir), recursive=False)
        self._observer.daemon = True
        self._observer.start()
        logger.info("Started watching %s", self._session_dir)

    def _poll_for_session_dir(self, timeout: float = 60.0, interval: float = 2.0) -> None:
        """Poll until the session directory appears or timeout elapses.

        Args:
            timeout: Maximum seconds to wait.
            interval: Seconds between checks.
        """
        import time

        logger.info("Session directory not found, polling for up to %.0fs: %s", timeout, self._session_dir)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._session_dir.exists():
                logger.info("Session directory appeared: %s", self._session_dir)
                return
            time.sleep(interval)
        logger.warning("Session directory still not found after %.0fs: %s", timeout, self._session_dir)

    def stop(self) -> None:
        """Stop watching."""
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
            logger.info("Stopped session watcher")
        # Flush any partial turn we were buffering — better to surface
        # it late than to drop it.
        self._coalescer.flush_now()


class _SessionFileHandler(FileSystemEventHandler):
    """Handles file system events for session files."""

    def __init__(self, watcher: SessionWatcher) -> None:
        self._watcher = watcher

    def on_created(self, event: FileCreatedEvent) -> None:  # type: ignore[override]
        if event.is_directory:
            return
        path = Path(str(event.src_path))
        if path.suffix == ".jsonl":
            logger.info("New session file detected: %s", path)
            self._watcher.process_file(path)

    def on_modified(self, event: FileModifiedEvent) -> None:  # type: ignore[override]
        if event.is_directory:
            return
        path = Path(str(event.src_path))
        if path.suffix == ".jsonl":
            self._watcher.process_file(path)
