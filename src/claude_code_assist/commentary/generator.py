"""Commentary generation routed through the configured LLM provider."""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from claude_code_assist.commentary import transcript
from claude_code_assist.commentary.changes import build_change_context_block
from claude_code_assist.commentary.prompts import (
    MEMORY_MAX_LENGTH,
    RECENT_COMMENTS_MAX,
    build_event_prompt,
    build_idle_prompt,
    build_memory_update_system_prompt,
    build_memory_update_user_prompt,
    build_reply_prompt,
    build_system_prompt,
)

if TYPE_CHECKING:
    from pathlib import Path

    from claude_code_assist.config import CompanionConfig
    from claude_code_assist.models.companion import CompanionProfile
    from claude_code_assist.monitor.parser import SessionEvent

logger = logging.getLogger(__name__)


@dataclass
class SessionUsage:
    """Cumulative token usage and cost for a tpet session."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_cost_usd: float = 0.0
    api_calls: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed (input + output)."""
        return self.input_tokens + self.output_tokens


# Module-level usage tracker — accumulates across all commentary calls.
_session_usage = SessionUsage()
_usage_lock = threading.Lock()


def get_session_usage() -> SessionUsage:
    """Return the current cumulative session usage (thread-safe snapshot).

    Returns:
        Copy of the session usage dataclass.
    """
    with _usage_lock:
        return SessionUsage(
            input_tokens=_session_usage.input_tokens,
            output_tokens=_session_usage.output_tokens,
            total_cost_usd=_session_usage.total_cost_usd,
            api_calls=_session_usage.api_calls,
        )


def reset_session_usage() -> None:
    """Reset session usage counters to zero."""
    with _usage_lock:
        _session_usage.input_tokens = 0
        _session_usage.output_tokens = 0
        _session_usage.total_cost_usd = 0.0
        _session_usage.api_calls = 0


def _record_usage(result_msg: ResultMessage) -> None:
    """Accumulate usage from a ResultMessage into the session tracker.

    Args:
        result_msg: The ResultMessage from an Agent SDK query.
    """
    with _usage_lock:
        _session_usage.api_calls += 1
        if result_msg.total_cost_usd is not None:
            _session_usage.total_cost_usd += result_msg.total_cost_usd
        if result_msg.usage:
            _session_usage.input_tokens += result_msg.usage.get("input_tokens", 0)
            _session_usage.output_tokens += result_msg.usage.get("output_tokens", 0)


# Shared thread-pool executor for non-blocking LLM calls (single worker to
# prevent concurrent Claude Agent SDK event-loop conflicts).
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cca-llm")


def shutdown_executor() -> None:
    """Best-effort shutdown — cancel queued tasks, don't wait on running ones.

    Called from the Qt ``aboutToQuit`` hook. ``cancel_futures=True``
    drops anything still in the queue; ``wait=False`` lets the
    in-flight Claude Agent SDK call carry on but lets the rest of the
    interpreter exit. The hard ``os._exit`` in the Qt entry point
    finishes the kill.
    """
    _executor.shutdown(wait=False, cancel_futures=True)


# Common preamble patterns models produce despite instructions not to
_PREAMBLE_RE = re.compile(
    r"^(?:"
    r"(?:terminal\s+companion|companion)\s+says?:\s*"  # "Terminal companion says:"
    r"|[*]{2,}|[_`\"]+\s*"  # leading bold (**) or formatting, but NOT single * (italics)
    r"|(?:as\s+\w+|speaking\s+as\s+\w+|here'?s?\s+\w+)[,:]\s*"  # "As Knurling:"
    r"|(?:the\s+)?(?:companion|creature|axolotl|cat|dog|dragon)\s+(?:says|replies|responds|thinks|mutters|whispers|quips):\s*"
    r")",
    re.IGNORECASE,
)


def _clean_comment(text: str, max_length: int) -> str | None:
    """Enforce single-line and length limit on model output.

    Strips preamble patterns, collapses to first line, trims quotes.
    Returns ``None`` when the model emitted ``<skip>`` (the "stay
    silent" escape \u2014 see ``build_system_prompt`` SILENCE clause) or
    when the output is empty after cleaning.
    """
    # Collapse to first line only (no multi-paragraph responses)
    first_line = text.strip().split("\n")[0].strip()
    # Skip token: model chose silence. Match permissively \u2014 some models
    # wrap the token in quotes / backticks despite the rule.
    stripped = first_line.strip().strip("`'\"").lower()
    if stripped == "<skip>" or stripped == "skip":
        return None
    # Strip preamble patterns (e.g. "Terminal companion says: ")
    first_line = _PREAMBLE_RE.sub("", first_line).strip()
    # Strip surrounding quotes if present
    if len(first_line) >= 2 and first_line[0] == '"' and first_line[-1] == '"':
        first_line = first_line[1:-1]
    # Strip leading ** (bold preamble) but preserve single * (markdown italics)
    if first_line.startswith("**"):
        first_line = first_line.lstrip("*").rstrip()
    if not first_line:
        return None
    # Enforce hard length limit
    if len(first_line) > max_length:
        first_line = first_line[: max_length - 1] + "\u2026"
    return first_line


async def _generate_text_claude(
    system_prompt: str,
    user_prompt: str,
    model: str,
    setting_sources: list[str] | None,
) -> str | None:
    """Run a single-turn Agent SDK query and return the result text.

    The generator returned by ``query()`` spawns a Claude subprocess
    and yields stream messages. We bail out as soon as we see a
    ``ResultMessage`` — without explicit ``aclose()`` the half-iterated
    generator gets cleaned up at loop teardown, racing with the SDK's
    own subprocess-reading task and producing
    ``RuntimeError: aclose(): asynchronous generator is already running``.
    """
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt,
        allowed_tools=[],
        max_turns=1,
        permission_mode="dontAsk",
        setting_sources=setting_sources,
        plugins=[],
    )

    gen = query(prompt=user_prompt, options=options)
    try:
        async for message in gen:
            if isinstance(message, ResultMessage):
                _record_usage(message)
                return message.result.strip() if message.result else None
        return None
    finally:
        # ``aclose`` swallows ``StopAsyncIteration`` / ``GeneratorExit``;
        # we just need to make sure it runs while the loop is alive.
        await gen.aclose()


def _generate_text_gemini(system_prompt: str, user_prompt: str, model: str, api_key: str) -> str | None:
    """Generate text using Google Gemini via the genai SDK.

    All errors are caught and logged — returns None so the caller can
    silently skip the failed generation.

    Args:
        system_prompt: System prompt for the model.
        user_prompt: User prompt for the model.
        model: Gemini model name (e.g. "gemini-2.5-flash").
        api_key: Google API key for authentication.

    Returns:
        Result text, or None if generation failed.
    """
    from google import genai
    from google.genai import types as gx

    try:
        client = genai.Client(api_key=api_key)
        config = gx.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=256,
        )
        response = client.models.generate_content(
            model=model,
            contents=[user_prompt],
            config=config,
        )
        text = response.text.strip() if response.text else None
        if text:
            with _usage_lock:
                _session_usage.api_calls += 1
                usage_meta = getattr(response, "usage_metadata", None)
                if usage_meta:
                    _session_usage.input_tokens += getattr(usage_meta, "prompt_token_count", 0) or 0
                    _session_usage.output_tokens += getattr(usage_meta, "candidates_token_count", 0) or 0
            return text
        return None
    except Exception:  # noqa: BLE001
        logger.warning("Gemini call failed (model=%s)", model, exc_info=True)
        return None


def _call_llm(system_prompt: str, user_prompt: str, config: CompanionConfig, *, kind: str) -> str | None:
    """Route text generation to the configured provider.

    Args:
        system_prompt: System prompt for the model.
        user_prompt: User prompt for the model.
        config: Application configuration (determines provider and model).
        kind: Tag recorded in the prompt transcript (``event`` / ``idle`` / ``reply``).

    Returns:
        Result text, or None if generation failed.
    """
    from claude_code_assist.config import LLMProvider
    from claude_code_assist.llm_client import generate_text_openai_compat
    from claude_code_assist.qt.settings import SettingsStore, settings_to_sdk_arg

    resolved = config.resolved_provider

    response: str | None
    if resolved.uses_agent_sdk:
        setting_sources = settings_to_sdk_arg(SettingsStore(config.config_dir).load().claude_setting_sources)
        response = asyncio.run(_generate_text_claude(system_prompt, user_prompt, resolved.model, setting_sources))
    elif resolved.is_openai_compat:
        response = generate_text_openai_compat(system_prompt, user_prompt, resolved)
    elif resolved.provider == LLMProvider.GEMINI:
        api_key = resolved.api_key
        if not api_key:
            logger.warning("Gemini API key not set (expected env var: %s)", resolved.api_key_env)
            response = None
        else:
            response = _generate_text_gemini(system_prompt, user_prompt, resolved.model, api_key)
    else:
        logger.warning("Unsupported commentary provider: %s", resolved.provider)
        response = None

    transcript.log_call(
        kind=kind,
        system=system_prompt,
        user=user_prompt,
        response=response,
        provider=resolved.provider.value,
        model=resolved.model,
    )
    return response


def _recent_comments_for(companion: CompanionProfile) -> list[str]:
    """Last few comments the companion has produced, oldest first.

    Surfaced in every user prompt so the LLM can avoid repeating
    itself across consecutive calls.
    """
    return list(companion.comment_history[-RECENT_COMMENTS_MAX:])


def _run_generate_comment(
    companion: CompanionProfile,
    event: SessionEvent,
    config: CompanionConfig,
    max_length: int,
    recent_events: list[SessionEvent],
    cwd: Path | None,
) -> str | None:
    """Blocking worker — runs in a background thread.

    The change-context build (``git diff`` + file reads) is blocking
    I/O and so lives here, never on the Qt main thread.
    """
    try:
        change_context = build_change_context_block(event.touched_paths, cwd)
        raw = _call_llm(
            system_prompt=build_system_prompt(companion, max_comment_length=max_length),
            user_prompt=build_event_prompt(
                event,
                companion=companion,
                recent_events=recent_events,
                recent_comments=_recent_comments_for(companion),
                change_context=change_context,
            ),
            config=config,
            kind="event",
        )
        return _clean_comment(raw, max_length) if raw else None
    except (RuntimeError, asyncio.CancelledError, OSError):
        logger.exception("Failed to generate comment")
        return None


def _run_generate_idle_chatter(companion: CompanionProfile, config: CompanionConfig, max_length: int) -> str | None:
    """Blocking worker — runs in a background thread."""
    try:
        raw = _call_llm(
            system_prompt=build_system_prompt(companion, max_comment_length=max_length),
            user_prompt=build_idle_prompt(
                companion=companion,
                recent_comments=_recent_comments_for(companion),
                max_length=max_length,
            ),
            config=config,
            kind="idle",
        )
        return _clean_comment(raw, max_length) if raw else None
    except (RuntimeError, asyncio.CancelledError, OSError):
        logger.exception("Failed to generate idle chatter")
        return None


def generate_comment(
    companion: CompanionProfile,
    event: SessionEvent,
    config: CompanionConfig,
    max_length: int = 150,
    recent_events: list[SessionEvent] | None = None,
    cwd: Path | None = None,
) -> str | None:
    """Generate an in-character comment for a session event (blocking).

    Prefer ``submit_comment`` for use inside the display loop.

    Args:
        companion: The companion profile to generate a comment for.
        event: The session event to comment on.
        config: Application configuration.
        max_length: Maximum comment character length.
        recent_events: Prior session turns (oldest first) to surface as
            context. Used in place of the older single
            ``last_user_event`` shortcut.
        cwd: Project working directory used to render the diff /
            surrounding-code blocks. ``None`` skips the change context
            (text-watcher mode, or no project).

    Returns:
        Generated comment string, or None if generation failed.
    """
    return _run_generate_comment(companion, event, config, max_length, list(recent_events or []), cwd)


def generate_idle_chatter(companion: CompanionProfile, config: CompanionConfig, max_length: int = 100) -> str | None:
    """Generate idle chatter when no session is active (blocking).

    Prefer ``submit_idle_chatter`` for use inside the display loop.

    Args:
        companion: The companion profile to generate idle chatter for.
        config: Application configuration.
        max_length: Maximum idle chatter character length.

    Returns:
        Generated idle chatter string, or None if generation failed.
    """
    return _run_generate_idle_chatter(companion, config, max_length)


def submit_comment(
    companion: CompanionProfile,
    event: SessionEvent,
    config: CompanionConfig,
    max_length: int = 150,
    recent_events: list[SessionEvent] | None = None,
    cwd: Path | None = None,
) -> Future[str | None]:
    """Submit a comment-generation task to the background executor.

    Returns immediately with a ``Future``.  The caller should check
    ``future.done()`` each tick and retrieve the result with
    ``future.result()`` when ready.

    Args:
        companion: The companion profile.
        event: The session event to comment on.
        config: Application configuration.
        max_length: Maximum comment character length.
        recent_events: Prior session turns (oldest first) to thread
            into the user prompt as context.
        cwd: Project working directory used by the diff layer. ``None``
            skips the change context block.

    Returns:
        Future resolving to the comment string, or None.
    """
    return _executor.submit(
        _run_generate_comment,
        companion,
        event,
        config,
        max_length,
        list(recent_events or []),
        cwd,
    )


def _run_generate_reply(
    companion: CompanionProfile,
    message: str,
    config: CompanionConfig,
    max_length: int,
    recent_events: list[SessionEvent],
    cwd: Path | None,
) -> str | None:
    """Generate an in-character reply to a direct address.

    Pulls the most recent assistant turn's touched-paths so the reply
    sees the same diff context an event-driven comment would. The
    developer asking "what do you think?" usually means the latest
    change.
    """
    focal_paths: list[str] = []
    for ev in reversed(recent_events):
        if ev.touched_paths:
            focal_paths = ev.touched_paths
            break
    try:
        change_context = build_change_context_block(focal_paths, cwd)
        raw = _call_llm(
            system_prompt=build_system_prompt(companion, max_comment_length=max_length),
            user_prompt=build_reply_prompt(
                companion,
                message,
                recent_events=recent_events,
                recent_comments=_recent_comments_for(companion),
                change_context=change_context,
                max_length=max_length,
            ),
            config=config,
            kind="reply",
        )
        return _clean_comment(raw, max_length) if raw else None
    except (RuntimeError, asyncio.CancelledError, OSError):
        logger.exception("Failed to generate reply")
        return None


def submit_reply(
    companion: CompanionProfile,
    message: str,
    config: CompanionConfig,
    max_length: int = 200,
    recent_events: list[SessionEvent] | None = None,
    cwd: Path | None = None,
) -> Future[str | None]:
    """Submit a direct-reply task to the background executor."""
    return _executor.submit(
        _run_generate_reply,
        companion,
        message,
        config,
        max_length,
        list(recent_events or []),
        cwd,
    )


def submit_idle_chatter(
    companion: CompanionProfile,
    config: CompanionConfig,
    max_length: int = 100,
) -> Future[str | None]:
    """Submit an idle-chatter task to the background executor.

    Returns immediately with a ``Future``.

    Args:
        companion: The companion profile.
        config: Application configuration.
        max_length: Maximum idle chatter character length.

    Returns:
        Future resolving to the idle chatter string, or None.
    """
    return _executor.submit(_run_generate_idle_chatter, companion, config, max_length)


def _run_update_memory(
    companion: CompanionProfile,
    observation: str,
    config: CompanionConfig,
) -> str | None:
    """Blocking memory update — runs in a background thread.

    Returns the new memory string, or None on failure (caller keeps
    the existing memory untouched).
    """
    try:
        raw = _call_llm(
            system_prompt=build_memory_update_system_prompt(companion),
            user_prompt=build_memory_update_user_prompt(companion, companion.session_memory, observation),
            config=config,
            kind="memory",
        )
    except (RuntimeError, asyncio.CancelledError, OSError):
        logger.exception("Failed to update session memory")
        return None
    if not raw:
        return None
    text = raw.strip()
    if len(text) > MEMORY_MAX_LENGTH:
        text = text[: MEMORY_MAX_LENGTH - 1] + "…"
    return text


def submit_memory_update(
    companion: CompanionProfile,
    observation: str,
    config: CompanionConfig,
) -> Future[str | None]:
    """Submit a memory-update task to the background executor.

    The caller is responsible for assigning the result back onto
    ``companion.session_memory`` when the future resolves — this
    keeps the executor stateless and keeps mutation on the main thread.
    """
    return _executor.submit(_run_update_memory, companion, observation, config)
