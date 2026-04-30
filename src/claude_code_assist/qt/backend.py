"""Session watcher + commentary backend, polled from the Qt app loop.

Replaces the Swift ``CommentBus`` Unix-socket protocol. Reuses the
existing ``tpet.monitor`` watchers and ``tpet.commentary`` LLM-call
plumbing — the same pieces ``tpet run`` uses — and exposes a single
``poll()`` method the Qt app calls each animation tick.

Threading:
* Watchdog observers run in their own threads inside the chosen
  watcher; they push ``SessionEvent`` instances onto a queue.
* LLM calls go through the shared single-worker ``ThreadPoolExecutor``
  in ``tpet.commentary.generator`` (one event loop per worker thread).
* ``poll()`` does only non-blocking work on the Qt main thread:
  drain at most one event, check if either pending future is done,
  maybe submit new work.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import TYPE_CHECKING

from claude_code_assist.commentary.generator import (
    submit_comment,
    submit_idle_chatter,
    submit_memory_update,
    submit_reply,
)
from claude_code_assist.monitor.text_watcher import TextFileWatcher
from claude_code_assist.monitor.watcher import SessionWatcher, encode_project_path

if TYPE_CHECKING:
    from concurrent.futures import Future
    from pathlib import Path

    from claude_code_assist.config import CompanionConfig
    from claude_code_assist.models.companion import CompanionProfile
    from claude_code_assist.monitor.parser import SessionEvent

logger = logging.getLogger(__name__)

_COMMENT_HISTORY_MAX = 20  # match tpet.app.COMMENT_HISTORY_MAX
_RECENT_EVENTS_MAX = 5
"""Rolling window of session turns the companion sees as context.

The most recent event is always the focal one passed to ``submit_*``;
the prior 4 are surfaced as conversation history so the LLM can react
to patterns (e.g. "they've been stuck on this for several turns")
rather than only to the single line that just arrived.
"""


def _is_direct_address(text: str, companion: CompanionProfile) -> bool:
    """True when ``text`` mentions the companion's name or role as a whole word."""
    import re  # noqa: PLC0415

    targets: list[str] = [companion.name]
    if companion.role is not None:
        targets.append(companion.role.value)
    for target in targets:
        if not target:
            continue
        if re.search(rf"\b{re.escape(target)}\b", text, re.IGNORECASE):
            return True
    return False


@dataclass
class BackendUpdate:
    """Summary of what changed during one ``poll()``.

    Both fields can be set together (e.g. an event arrived this tick AND
    a previous future just resolved); the app handles each independently.
    """

    new_comment: str | None = None
    """Fresh comment string from the LLM, ready to display. ``None`` if
    no comment came back this tick."""

    had_event: bool = False
    """``True`` when a session event was just drained from the queue.
    Triggers the REACTING animation regardless of whether a comment is
    in flight."""


class CommentaryBackend:
    """Owns the watcher, the per-session counters, and the in-flight futures."""

    def __init__(
        self,
        *,
        config: CompanionConfig,
        companion: CompanionProfile,
        project_path: str,
        watch_dir: Path | None = None,
        follow_file: Path | None = None,
    ) -> None:
        self._config = config
        self._companion = companion
        self._event_queue: Queue[SessionEvent] = Queue()

        self._watcher: SessionWatcher | TextFileWatcher
        if follow_file is not None:
            logger.info("Following text file: %s", follow_file)
            self._watcher = TextFileWatcher(file_path=follow_file, event_queue=self._event_queue)
        else:
            from pathlib import Path

            session_dir = watch_dir or (Path.home() / ".claude" / "projects" / encode_project_path(project_path))
            logger.info("Watching Claude session dir: %s", session_dir)
            self._watcher = SessionWatcher(session_dir=session_dir, event_queue=self._event_queue)

        self._pending_comment: Future[str | None] | None = None
        self._pending_idle: Future[str | None] | None = None
        self._pending_memory: Future[str | None] | None = None
        self._recent_events: list[SessionEvent] = []
        self._comment_count = 0
        # Use ``-inf`` so the first event is never blocked by the cooldown.
        self._last_comment_time = float("-inf")
        self._last_idle_time = time.monotonic()
        # When ``False`` the autonomous "idle chatter" generator is
        # paused — only real session events drive comments. Toggled
        # by the Qt app based on the controller's sleep state.
        self._idle_chatter_enabled: bool = True

    def set_idle_chatter_enabled(self, enabled: bool) -> None:
        self._idle_chatter_enabled = enabled

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._watcher.start()
        logger.info("CommentaryBackend started (%s)", type(self._watcher).__name__)

    def stop(self) -> None:
        self._watcher.stop()
        # Tear down the shared LLM executor so the interpreter doesn't
        # block on a pending Claude Agent SDK call during shutdown.
        from claude_code_assist.commentary.generator import shutdown_executor  # noqa: PLC0415

        shutdown_executor()
        logger.info("CommentaryBackend stopped (%d comments)", self._comment_count)

    @property
    def comment_count(self) -> int:
        return self._comment_count

    # ------------------------------------------------------------------
    # Manual trigger
    # ------------------------------------------------------------------

    def request_comment_now(self) -> bool:
        """Submit a fresh comment immediately, bypassing the cooldown.

        Used by the tray's "React now" menu item. Picks the best source:

        * If a session event is queued, drain and use it (matches the
          regular tick flow but jumps the queue).
        * Else if we've seen any user events this session, comment on
          the most recent one.
        * Else fall back to idle chatter so the companion still says something.

        No-op when a comment future is already in flight or the session
        budget is exhausted; returns ``False`` in that case so the caller
        can show feedback if desired.
        """
        if not self._within_budget():
            logger.info("React-now ignored: comment budget exhausted")
            return False
        if self._pending_comment is not None:
            logger.info("React-now ignored: a comment is already in flight")
            return False

        # Try to drain a queued event first so this acts like an immediate
        # version of the normal flow.
        try:
            event: SessionEvent | None = self._event_queue.get_nowait()
        except Empty:
            event = None

        if event is not None:
            self._record_event(event)
            self._pending_comment = submit_comment(
                self._companion,
                event,
                config=self._config,
                max_length=self._config.max_comment_length,
                recent_events=list(self._recent_events[:-1]),
            )
            return True

        if self._recent_events:
            focal = self._recent_events[-1]
            self._pending_comment = submit_comment(
                self._companion,
                focal,
                config=self._config,
                max_length=self._config.max_comment_length,
                recent_events=list(self._recent_events[:-1]),
            )
            return True

        # No session context yet — use the same idle slot as background chatter
        # so the next ``poll()`` picks it up via the normal harvest path.
        if self._pending_idle is None:
            self._pending_idle = submit_idle_chatter(
                self._companion,
                config=self._config,
                max_length=self._config.max_idle_length,
            )
            self._last_idle_time = time.monotonic()
            return True

        return False

    # ------------------------------------------------------------------
    # Per-tick poll
    # ------------------------------------------------------------------

    def poll(self) -> BackendUpdate:
        """Drain at most one event, harvest pending futures, submit new work."""
        update = BackendUpdate()
        now = time.monotonic()

        # 1. Harvest a completed comment future.
        if self._pending_comment is not None and self._pending_comment.done():
            try:
                comment = self._pending_comment.result()
            except RuntimeError:
                logger.exception("Comment future raised")
                comment = None
            self._pending_comment = None
            # Apply the cooldown whenever an LLM call completes — even
            # when the model chose ``<skip>`` (returns None) — so a
            # silent decision doesn't trigger an instant retry on the
            # next event.
            self._last_comment_time = now
            if comment:
                self._record_comment(comment)
                update.new_comment = comment

        # 2. Harvest a completed idle-chatter future.
        if self._pending_idle is not None and self._pending_idle.done():
            try:
                idle_text = self._pending_idle.result()
            except RuntimeError:
                logger.exception("Idle chatter future raised")
                idle_text = None
            self._pending_idle = None
            self._last_idle_time = now
            # Prefer a real comment over idle chatter if both came back this tick.
            if idle_text and update.new_comment is None:
                self._record_comment(idle_text)
                update.new_comment = idle_text

        # 2b. Harvest a completed memory-update future. Mutation lives
        # on the main thread — the executor only computed the new
        # string. Failed updates leave the existing memory untouched.
        if self._pending_memory is not None and self._pending_memory.done():
            try:
                new_memory = self._pending_memory.result()
            except RuntimeError:
                logger.exception("Memory update future raised")
                new_memory = None
            self._pending_memory = None
            if new_memory:
                self._companion.session_memory = new_memory

        # 3. Drain one queued session event so a burst doesn't lock up the tick.
        try:
            event: SessionEvent | None = self._event_queue.get_nowait()
        except Empty:
            event = None

        if event is not None:
            update.had_event = True
            self._record_event(event)
            prior_events = list(self._recent_events[:-1])

            # Direct address: the developer mentioned the companion by
            # name or by role in their message. Skip the cooldown and
            # dispatch a *reply* instead of a third-person comment.
            direct_address = (
                event.role == "user"
                and self._pending_comment is None
                and self._within_budget()
                and _is_direct_address(event.summary, self._companion)
            )
            if direct_address:
                self._pending_comment = submit_reply(
                    self._companion,
                    event.summary,
                    config=self._config,
                    max_length=self._config.max_comment_length,
                    recent_events=prior_events,
                )
            elif (
                self._pending_comment is None
                and now - self._last_comment_time >= self._config.comment_interval_seconds
                and self._within_budget()
            ):
                self._pending_comment = submit_comment(
                    self._companion,
                    event,
                    config=self._config,
                    max_length=self._config.max_comment_length,
                    recent_events=prior_events,
                )

        # 4. Idle chatter when the session has gone quiet.
        if (
            self._idle_chatter_enabled
            and self._pending_idle is None
            and now - self._last_idle_time >= self._config.idle_chatter_interval_seconds
            and self._within_budget()
        ):
            self._pending_idle = submit_idle_chatter(
                self._companion,
                config=self._config,
                max_length=self._config.max_idle_length,
            )
            self._last_idle_time = now  # don't re-submit while in flight

        return update

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _record_event(self, event: SessionEvent) -> None:
        """Append ``event`` to the rolling session-context window.

        When the window overflows, the oldest event is about to be
        forgotten — fold it into ``session_memory`` first via a
        background LLM call so the companion keeps long-horizon
        context. Skips the update if a memory call is already in
        flight; the next overflow will catch up.
        """
        self._recent_events.append(event)
        if len(self._recent_events) <= _RECENT_EVENTS_MAX:
            return
        rotated = self._recent_events[0]
        self._recent_events = self._recent_events[-_RECENT_EVENTS_MAX:]
        if self._pending_memory is None and rotated.summary:
            observation = f"[{rotated.role}] {rotated.summary}"
            self._pending_memory = submit_memory_update(
                self._companion,
                observation,
                self._config,
            )

    def _record_comment(self, comment: str) -> None:
        self._companion.last_comment = comment
        self._companion.comment_history.append(comment)
        if len(self._companion.comment_history) > _COMMENT_HISTORY_MAX:
            self._companion.comment_history = self._companion.comment_history[-_COMMENT_HISTORY_MAX:]
        self._comment_count += 1

        # Bump the rolling counter that gates the next player-driven
        # level-up. ``record_comment`` no longer auto-levels — the
        # player consumes the eligibility at startup (see
        # ``cli/_levelup_flow.py``).
        from claude_code_assist.profile.leveling import record_comment

        record_comment(self._companion)

    def _within_budget(self) -> bool:
        return self._config.max_comments_per_session == 0 or self._comment_count < self._config.max_comments_per_session
