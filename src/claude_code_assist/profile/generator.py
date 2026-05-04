"""Companion profile generation using configurable LLM providers.

Public surface:

* :func:`generate_companion` — build a fresh :class:`CompanionProfile`
  (sync wrapper with retries around the async LLM call).
* :func:`ensure_locomotion_descriptors` — backfill the four locomotion
  fields on a legacy profile that doesn't have them yet.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING

from claude_code_assist.config import LLMProvider, ResolvedProviderConfig
from claude_code_assist.models.companion import CompanionProfile
from claude_code_assist.models.rarity import Rarity
from claude_code_assist.models.stats import DEFAULT_STAT_NAMES, STAT_DEFINITIONS, shape_stats

if TYPE_CHECKING:
    from claude_code_assist.config import CompanionConfig

logger = logging.getLogger(__name__)

_LOCOMOTION_INSTRUCTIONS = (
    "LOCOMOTION FIELDS (used to drive the desktop companion's walk/fall/landing animation): "
    "describe how THIS specific creature moves. Be concrete and species-appropriate — "
    "a 4-legged animal trots, a snake slithers, a centipede ripples its many legs, "
    "a bird flaps and glides, a balloon drifts. Keep each description to 1-3 sentences "
    "and do NOT include cell/frame numbering — the consumer wraps these into a sprite-sheet prompt.\n"
    "- body_plan: anatomy in one sentence (limb count, wings, tail, body shape, etc.).\n"
    "- walk_description: a two-stride walk cycle describing stride A and stride B for THIS creature's "
    "locomotion (4-leg diagonal trot, 6-leg tripod gait, slither S-curve A vs S-curve B, hover bob, etc.). "
    "Always FACING RIGHT. Do not flip or mirror between strides.\n"
    "- fall_description: how it behaves mid-air. If it has wings or is naturally floaty (cloud, balloon, "
    "feather, jellyfish), it GLIDES or DRIFTS calmly. Otherwise it PLUMMETS with limbs splayed and a "
    "startled expression.\n"
    "- landing_description: how it hits the ground. Describe only one frame, the moment of impact."
    "Soft for gliders/floaters (touches down with wings folded / drifts to rest, neutral expression). "
    "Hard for everything else, choosing the reaction that matches the body: splatted into a puddle for squishy/gooey"
    " shattered shards for brittle/crystalline, dented + sparks for metallic/mechanical, "
    "dazed with X-eyes and stars otherwise.\n"
)

_STAT_LINES = "\n".join(f"- {name}: {desc}" for name, desc in STAT_DEFINITIONS.items())

_SYSTEM_PROMPT = (
    "You are a creative desktop-companion designer. Generate a unique, charming creature "
    "that would live on a developer's desktop. The creature should have a distinct personality "
    "that relates to software development.\n\n"
    "STATS — five integer values 0-100 reflecting the creature's personality. The post-"
    "processor will identify the *highest* stat as the companion's signature peak and "
    "the *lowest* as the signature dump stat (the rest are mid-range). Pick relative "
    "ordering that matches the persona — the exact numbers will be re-rolled within "
    "rarity-appropriate ranges, so what matters is which stat is highest and which is "
    "lowest:\n"
    f"{_STAT_LINES}\n\n" + _LOCOMOTION_INSTRUCTIONS + "\n"
    "IMPORTANT: Output ONLY a single JSON object with no markdown fencing, no explanation, "
    "and no extra text. The JSON must have these exact keys:\n"
    '- "name": string (creative creature name; never any of these reserved CLI words: '
    '"new", "art", "roster", "archive", "help" — case-insensitive)\n'
    '- "creature_type": string (species like axolotl, phoenix, goblin)\n'
    '- "personality": string (2-3 sentence personality summary)\n'
    '- "backstory": string (3-5 sentence origin story)\n'
    '- "accent_color": string (Rich color name like cyan, red, bright_magenta)\n'
    '- "stats": object with all five keys (DEBUGGING, PATIENCE, CHAOS, WISDOM, SNARK) '
    "mapped to integer values 0-100\n"
    '- "body_plan": string (anatomy in one sentence)\n'
    '- "walk_description": string (two-stride walk cycle for the macos-desktop sprite)\n'
    '- "fall_description": string (mid-air behavior — glide vs plummet)\n'
    '- "landing_description": string (touchdown reaction — soft vs hard impact)\n'
)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
_MAX_RETRIES = 3
_RESERVED_COMPANION_NAMES = frozenset({"new", "art", "roster", "archive", "help", "default"})


class LLMConfigError(RuntimeError):
    """LLM call failed for a reason that retrying won't fix.

    Wraps auth, missing-CLI, model-not-found, and unreachable-endpoint
    errors with provider/model context plus a one-line hint so the user
    knows where to look. ``_run_with_retries`` bails out immediately
    when this is raised — retrying a misconfigured provider just adds
    noise.
    """


def _provider_context(resolved: ResolvedProviderConfig) -> str:
    """One-line ``provider=… model=… [base_url=…]`` summary for error messages."""
    parts = [f"provider={resolved.provider.value}", f"model={resolved.model}"]
    if resolved.base_url:
        parts.append(f"base_url={resolved.base_url}")
    return ", ".join(parts)


def _extract_json_from_result(result_msg: object, context: str) -> dict[str, object]:
    """Extract and parse JSON from an Agent SDK ResultMessage."""
    structured_output = getattr(result_msg, "structured_output", None)
    if structured_output is not None:
        return structured_output if isinstance(structured_output, dict) else {}

    result_text = getattr(result_msg, "result", None)
    if result_text:
        cleaned = _extract_json(result_text)
        try:
            data: dict[str, object] = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse %s JSON. Raw result:\n%s", context, result_text)
            raise RuntimeError(f"Agent SDK returned invalid JSON for {context}: {exc}") from exc
        else:
            return data

    raise RuntimeError(f"Agent SDK returned empty result for {context}")


def _parse_json_response(text: str | None, context: str) -> dict[str, object]:
    """Extract and parse JSON from a raw text response."""
    if not text:
        raise RuntimeError(f"LLM returned empty result for {context}")

    cleaned = _extract_json(text)
    try:
        data: dict[str, object] = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse %s JSON. Raw result:\n%s", context, text)
        raise RuntimeError(f"LLM returned invalid JSON for {context}: {exc}") from exc
    else:
        return data


async def _generate_text_openai_compat(
    system_prompt: str,
    user_prompt: str,
    resolved: ResolvedProviderConfig,
    max_tokens: int = 4096,
) -> str:
    """Generate text using an OpenAI-compatible API, raising on failure."""
    from openai import APIConnectionError, AuthenticationError, NotFoundError, OpenAI

    ctx = _provider_context(resolved)
    if resolved.api_key_env and not resolved.api_key:
        raise LLMConfigError(
            f"Missing API key for {resolved.provider.value} — "
            f"set the {resolved.api_key_env} environment variable. ({ctx})"
        )

    try:
        client = OpenAI(base_url=resolved.base_url, api_key=resolved.api_key or "unused")
        response = client.chat.completions.create(
            model=resolved.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            timeout=120,
        )
    except AuthenticationError as exc:
        raise LLMConfigError(
            f"Authentication failed for {resolved.provider.value}. "
            f"Check the {resolved.api_key_env or 'API key'}. ({ctx}): {exc}"
        ) from exc
    except NotFoundError as exc:
        raise LLMConfigError(
            f"Model not found on {resolved.provider.value}: {resolved.model!r}. "
            f"Pick a different model in `companion settings`. ({ctx}): {exc}"
        ) from exc
    except APIConnectionError as exc:
        hint = "Is the Ollama server running?" if resolved.provider.value == "ollama" else "Check network / base_url."
        raise LLMConfigError(f"Could not reach {resolved.provider.value}. {hint} ({ctx}): {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"OpenAI-compatible API call failed ({ctx}): {exc}") from exc

    choice = response.choices[0] if response.choices else None
    if choice and choice.message and choice.message.content:
        return choice.message.content.strip()
    raise RuntimeError(f"OpenAI-compatible API returned no content ({ctx})")


async def _generate_text_gemini(
    system_prompt: str,
    user_prompt: str,
    resolved: ResolvedProviderConfig,
) -> str:
    """Generate text using Google Gemini API, raising on failure."""
    from google import genai
    from google.genai import types as gx

    ctx = _provider_context(resolved)
    if not resolved.api_key:
        raise LLMConfigError(
            f"Missing API key for gemini — set the {resolved.api_key_env} environment variable. ({ctx})"
        )

    try:
        client = genai.Client(api_key=resolved.api_key)
        config = gx.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="text/plain",
        )
        response = client.models.generate_content(
            model=resolved.model,
            contents=[user_prompt],
            config=config,
        )
    except Exception as exc:
        msg = str(exc).lower()
        if "api key" in msg or "unauthorized" in msg or "permission" in msg:
            raise LLMConfigError(f"Gemini auth failed — check {resolved.api_key_env}. ({ctx}): {exc}") from exc
        if "not found" in msg or "invalid model" in msg:
            raise LLMConfigError(
                f"Gemini model not found: {resolved.model!r}. "
                f"Pick a different model in `companion settings`. ({ctx}): {exc}"
            ) from exc
        raise RuntimeError(f"Gemini API call failed ({ctx}): {exc}") from exc

    text = response.text if hasattr(response, "text") else None
    if text:
        return text.strip()
    raise RuntimeError(f"Gemini API returned no text content ({ctx})")


async def _call_profile_llm(
    system_prompt: str,
    user_prompt: str,
    config: CompanionConfig,
    context: str = "LLM call",
) -> dict[str, object]:
    """Route profile LLM call to the configured provider and parse JSON response."""
    resolved = config.resolved_provider
    ctx = _provider_context(resolved)

    if resolved.uses_agent_sdk:
        from claude_agent_sdk import (
            ClaudeAgentOptions,
            CLINotFoundError,
            ProcessError,
            ResultMessage,
            query,
        )

        from claude_code_assist.qt.settings import SettingsStore, settings_to_sdk_arg

        setting_sources = settings_to_sdk_arg(SettingsStore(config.config_dir).load().claude_setting_sources)
        options = ClaudeAgentOptions(
            model=resolved.model,
            system_prompt=system_prompt,
            allowed_tools=[],
            max_turns=1,
            permission_mode="dontAsk",
            setting_sources=setting_sources,
            plugins=[],
        )
        result_msg: object | None = None
        try:
            async for message in query(prompt=user_prompt, options=options):
                if isinstance(message, ResultMessage):
                    result_msg = message
        except CLINotFoundError as exc:
            raise LLMConfigError(
                "Claude Code CLI not found. Install it with `npm install -g @anthropic-ai/claude-code` "
                f"or switch provider in `companion settings`. ({ctx}): {exc}"
            ) from exc
        except ProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stderr_hint = f"\n  stderr: {stderr}" if stderr and stderr != "Check stderr output for details" else ""
            raise LLMConfigError(
                "Claude Agent SDK subprocess failed. This usually means the `claude` CLI is "
                "missing, not authenticated (`claude login`), or the model is unavailable. "
                f"Try `companion settings` to switch provider. ({ctx}, exit_code={exc.exit_code})"
                f"{stderr_hint}"
            ) from exc

        if result_msg is None:
            raise RuntimeError(f"Agent SDK did not return a result for {context} ({ctx})")
        is_error = getattr(result_msg, "is_error", False)
        if is_error:
            errors_val = getattr(result_msg, "errors", None) or []
            errors = ", ".join(str(e) for e in errors_val) if errors_val else "unknown error"
            raise RuntimeError(f"Agent SDK returned an error ({ctx}): {errors}")

        return _extract_json_from_result(result_msg, context)

    if resolved.is_openai_compat:
        text = await _generate_text_openai_compat(system_prompt, user_prompt, resolved)
        return _parse_json_response(text, context)

    if resolved.provider == LLMProvider.GEMINI:
        text = await _generate_text_gemini(system_prompt, user_prompt, resolved)
        return _parse_json_response(text, context)

    raise RuntimeError(f"Unsupported LLM provider for profile generation: {resolved.provider}")


def _run_with_retries(fn: object, *args: object, context: str = "Operation") -> object:
    """Run an async function via asyncio.run with retry logic.

    ``LLMConfigError`` is re-raised immediately — retrying a misconfigured
    provider (missing API key, wrong model, dead endpoint) just produces
    the same failure three times.
    """
    last_error: Exception | None = None
    attempt = 0
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return asyncio.run(fn(*args))  # type: ignore[operator]
        except LLMConfigError:
            raise
        except Exception as exc:
            last_error = exc
            logger.warning("%s attempt %d/%d failed: %s", context, attempt, _MAX_RETRIES, exc)
            msg = str(exc).lower()
            if any(s in msg for s in ("model", "invalid", "unauthorized", "permission", "not found")):
                break
    raise RuntimeError(f"{context} failed after {attempt} attempts: {last_error}") from last_error


def _extract_json(text: str) -> str:
    """Extract JSON from text that may be wrapped in markdown code fences."""
    match = _JSON_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()
    return text.strip()


async def _generate_companion_async(
    config: CompanionConfig,
    rarity: Rarity,
    criteria: str | None = None,
) -> tuple[CompanionProfile, str]:
    """Generate a companion profile using the configured LLM provider."""
    if rarity == Rarity.COMMON:
        complexity = "simple"
    elif rarity == Rarity.LEGENDARY:
        complexity = "impressive"
    else:
        complexity = "interesting"
    rarity_hint = f"This is a {rarity.value} creature ({rarity.stars}). Make it appropriately {complexity}."

    import random

    seed = config.seed
    rng = random.Random(seed)

    adjectives = [
        "whimsical",
        "mischievous",
        "stoic",
        "energetic",
        "lazy",
        "curious",
        "grumpy",
        "cheerful",
        "sassy",
        "shy",
        "brave",
        "clumsy",
        "wise",
        "chaotic",
        "gentle",
        "fierce",
        "dreamy",
        "sneaky",
        "noble",
        "quirky",
    ]
    species_hints = [
        "elemental",
        "sprite",
        "familiar",
        "spirit",
        "golem",
        "slime",
        "dragon",
        "cat",
        "bird",
        "bug",
        "robot",
        "ghost",
        "plant",
        "fish",
        "demon",
        "angel",
        "goblin",
        "fox",
        "blob",
        "crystal",
    ]
    themes = [
        "code",
        "bugs",
        "compilers",
        "networks",
        "databases",
        "cloud",
        "pixels",
        "shaders",
        "threads",
        "memory",
        "stacks",
        "loops",
        "regex",
        "syntax",
        "debugging",
        "recursion",
        "linting",
        "types",
        "containers",
        "terminals",
        "git",
        "merge conflicts",
        "deploys",
    ]

    adj = rng.choice(adjectives)
    species = rng.choice(species_hints)
    theme = rng.choice(themes)
    prompt_parts = [
        f"Generate a desktop companion creature. {rarity_hint}",
        f"Inspiration for this companion: a {adj} {species} related to {theme}. "
        f"Use this as a starting point but feel free to deviate creatively.",
    ]
    if criteria:
        prompt_parts.append(f"Additional creation criteria from the user:\n{criteria}")
    user_prompt = "\n\n".join(prompt_parts)

    data = await _call_profile_llm(_SYSTEM_PROMPT, user_prompt, config, context="companion generation")

    name = str(data.get("name", "")).strip()
    if name.lower() in _RESERVED_COMPANION_NAMES:
        raise RuntimeError(
            f"LLM returned reserved name {name!r}; reserved tokens cannot be used as "
            "companion names because they would shadow CLI subcommands."
        )

    raw_stats = data.get("stats")
    hints: dict[str, int] | None = None
    if isinstance(raw_stats, dict) and raw_stats:
        hints = {str(k): int(v) for k, v in raw_stats.items() if isinstance(v, int | float)}
    stats = shape_stats(rarity, hints, names=list(DEFAULT_STAT_NAMES))

    companion = CompanionProfile(
        name=name,
        creature_type=str(data["creature_type"]),
        rarity=rarity,
        shiny=rng.random() < 0.01,
        personality=str(data["personality"]),
        backstory=str(data["backstory"]),
        stats=stats,
        accent_color=str(data.get("accent_color", "cyan")),
        body_plan=str(data.get("body_plan", "")).strip(),
        walk_description=str(data.get("walk_description", "")).strip(),
        fall_description=str(data.get("fall_description", "")).strip(),
        landing_description=str(data.get("landing_description", "")).strip(),
    )
    full_prompt = f"=== SYSTEM PROMPT ===\n{_SYSTEM_PROMPT}\n\n=== USER PROMPT ===\n{user_prompt}\n"
    return companion, full_prompt


def generate_companion(
    config: CompanionConfig,
    rarity: Rarity,
    criteria: str | None = None,
) -> tuple[CompanionProfile, str]:
    """Generate a new companion profile using the configured LLM provider.

    Returns ``(companion, full_prompt_text)`` so callers that opt into
    creation-prompt logging can persist the exact prompt the LLM saw.
    Retries up to 3 times on failure (e.g. invalid JSON from the model).
    """
    result = _run_with_retries(_generate_companion_async, config, rarity, criteria, context="Companion generation")
    return result  # type: ignore[return-value]


_LOCOMOTION_BACKFILL_SYSTEM_PROMPT = (
    "You are a creature designer filling in missing animation fields for an existing companion. "
    "You will be given the companion's name, creature type, personality, and backstory. "
    "Produce four short descriptors used by an image-generation pipeline.\n\n" + _LOCOMOTION_INSTRUCTIONS + "\n"
    "IMPORTANT: Output ONLY a single JSON object with no markdown fencing, no explanation, "
    "and no extra text. The JSON must have these exact keys:\n"
    '- "body_plan": string\n'
    '- "walk_description": string\n'
    '- "fall_description": string\n'
    '- "landing_description": string\n'
)


async def _backfill_locomotion_async(config: CompanionConfig, companion: CompanionProfile) -> dict[str, str]:
    """Ask the configured profile LLM for the four locomotion descriptors."""
    user_prompt = (
        f"Fill in the locomotion fields for this existing companion:\n"
        f"- Name: {companion.name}\n"
        f"- Creature type: {companion.creature_type}\n"
        f"- Personality: {companion.personality}\n"
        f"- Backstory: {companion.backstory}\n"
        f"- Rarity: {companion.rarity.value} ({companion.rarity.stars})\n"
    )
    data = await _call_profile_llm(
        _LOCOMOTION_BACKFILL_SYSTEM_PROMPT, user_prompt, config, context="locomotion backfill"
    )
    return {
        "body_plan": str(data.get("body_plan", "")).strip(),
        "walk_description": str(data.get("walk_description", "")).strip(),
        "fall_description": str(data.get("fall_description", "")).strip(),
        "landing_description": str(data.get("landing_description", "")).strip(),
    }


def ensure_locomotion_descriptors(
    config: CompanionConfig, companion: CompanionProfile
) -> tuple[CompanionProfile, bool]:
    """Return ``(companion, updated)`` with locomotion fields filled in."""
    if all(
        [companion.body_plan, companion.walk_description, companion.fall_description, companion.landing_description]
    ):
        return companion, False

    descriptors = _run_with_retries(_backfill_locomotion_async, config, companion, context="Locomotion backfill")
    if not isinstance(descriptors, dict):
        return companion, False
    return companion.model_copy(update=descriptors), True
