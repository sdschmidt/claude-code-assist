"""Premade art-set discovery, LLM matching, and copy-into-slot.

The bundled ``assets/placeholder_frames/<slug>/`` directories each hold
a complete 10-frame sprite set, an ``icon.png`` and a ``descriptor.txt``
of the form ``<creature_type>; anatomy: <body_plan>``. ``companion art
→ premade`` first tries a deterministic creature-type shortcut from
those descriptors; if it's ambiguous, it asks the configured profile
LLM which slug best matches the active companion (shape > limb count >
material > color), then copies that slug's files into the companion's
art dir.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import TYPE_CHECKING

from claude_code_assist.art.meta import ArtMeta, write_meta

if TYPE_CHECKING:
    from pathlib import Path

    from claude_code_assist.config import CompanionConfig
    from claude_code_assist.models.companion import CompanionProfile

logger = logging.getLogger(__name__)

PREMADE_PACKAGE = "claude_code_assist.assets.placeholder_frames"
FRAME_COUNT = 10
ICON_FILENAME = "icon.png"
DESCRIPTOR_FILENAME = "descriptor.txt"


@dataclass(frozen=True)
class PremadeOption:
    """One bundled art set the matcher can pick from."""

    slug: str
    description: str  # full ``<creature_type>; anatomy: ...`` line for the LLM
    creature_type: str  # leading segment — used by the heuristic shortcut
    has_icon: bool


def _split_descriptor(descriptor: str, slug: str) -> tuple[str, str]:
    """Return ``(creature_type, full_descriptor)`` from a descriptor.txt body.

    Format is ``<creature_type>; anatomy: <body_plan>``. The prefix before
    the first ``;`` is the creature_type. If the file is malformed (no ``;``)
    the whole string is treated as the creature_type.
    """
    text = descriptor.strip()
    if not text:
        fallback = slug.replace("_", " ")
        return fallback, fallback
    head, _, _ = text.partition(";")
    return head.strip() or slug.replace("_", " "), text


@lru_cache(maxsize=1)
def list_premade_options() -> tuple[PremadeOption, ...]:
    """Return all bundled premade sets that ship 10 frames + descriptor.txt.

    Cached because the bundled directory is read-only at runtime — calling
    this repeatedly during a single CLI invocation should be free.
    """
    package = resources.files(PREMADE_PACKAGE)
    options: list[PremadeOption] = []
    for entry in sorted(package.iterdir(), key=lambda p: p.name.lower()):
        if not entry.is_dir() or entry.name.startswith("__"):
            continue
        frames_present = all(entry.joinpath(f"frame_{i}.png").is_file() for i in range(FRAME_COUNT))
        descriptor_path = entry.joinpath(DESCRIPTOR_FILENAME)
        if not (frames_present and descriptor_path.is_file()):
            logger.debug(
                "Skipping premade %s (incomplete: frames=%s, descriptor=%s)",
                entry.name,
                frames_present,
                descriptor_path.is_file(),
            )
            continue
        try:
            descriptor_text = descriptor_path.read_text(encoding="utf-8")
        except OSError:
            logger.warning("Skipping premade %s: descriptor.txt unreadable", entry.name, exc_info=True)
            continue
        creature_type, descriptor = _split_descriptor(descriptor_text, entry.name)
        options.append(
            PremadeOption(
                slug=entry.name,
                description=descriptor,
                creature_type=creature_type,
                has_icon=entry.joinpath(ICON_FILENAME).is_file(),
            )
        )
    return tuple(options)


def copy_premade(option: PremadeOption, art_dir: Path) -> list[Path]:
    """Copy a premade set's frames + icon + meta into ``art_dir``."""
    art_dir.mkdir(parents=True, exist_ok=True)
    package = resources.files(PREMADE_PACKAGE)
    source = package.joinpath(option.slug)

    out: list[Path] = []
    for i in range(FRAME_COUNT):
        name = f"frame_{i}.png"
        dest = art_dir / name
        dest.write_bytes(source.joinpath(name).read_bytes())
        out.append(dest)

    if option.has_icon:
        (art_dir / ICON_FILENAME).write_bytes(source.joinpath(ICON_FILENAME).read_bytes())

    write_meta(art_dir, ArtMeta(model=f"premade:{option.slug}", prompt=option.description))
    return out


_MATCH_SYSTEM_PROMPT = (
    "You match a desktop-companion profile to the closest bundled pixel-art set. "
    "You will receive (1) a companion profile and (2) a list of premade options "
    "with descriptions.\n\n"
    "RULES — apply in this strict priority order:\n"
    "1. OVERALL SHAPE / SILHOUETTE is most important. Pick the option whose body "
    "outline best matches the profile's creature_type and body_plan. A green dragon "
    "must pick a dragon-shaped option over a frog-shaped option, even when the frog "
    "is also green.\n"
    "2. LIMB COUNT and LOCOMOTION (bipedal vs quadrupedal vs slither vs winged "
    "vs hover) come second.\n"
    "3. MATERIAL / TEXTURE (skeleton, slime, crystal, plush, metal) comes third.\n"
    "4. COLOR is a tiebreaker only — never a primary signal.\n\n"
    "Output ONLY a JSON object with these exact keys:\n"
    '- "slug": one of the listed option slugs, exactly as written\n'
    '- "reason": one short sentence on why this slug wins under the rules above\n'
    "No markdown, no commentary, no extra keys."
)


def _build_match_user_prompt(companion: CompanionProfile, options: tuple[PremadeOption, ...]) -> str:
    profile_block = (
        "Companion profile:\n"
        f"- name: {companion.name}\n"
        f"- creature_type: {companion.creature_type}\n"
        f"- body_plan: {companion.body_plan or '(empty)'}\n"
        f"- personality: {companion.personality}\n"
    )
    options_block = "Premade options:\n" + "\n".join(f'- slug="{opt.slug}": {opt.description}' for opt in options)
    return f"{profile_block}\n{options_block}\n"


_TOKEN_SPLIT_RE = re.compile(r"[\s_\-]+")


def _tokenize(text: str) -> set[str]:
    """Split into lowercase word tokens (≥3 chars) — for word-level matching."""
    return {t for t in _TOKEN_SPLIT_RE.split(text.lower()) if len(t) >= 3}


def _heuristic_match(companion: CompanionProfile, options: tuple[PremadeOption, ...]) -> PremadeOption | None:
    """Return an option when ``companion.creature_type`` uniquely identifies one slug.

    Word-level matching (not substring) so ``imp`` matches ``black_imp`` but
    not ``important``. Returns ``None`` when zero or multiple options match —
    those need the LLM to disambiguate.
    """
    needle_tokens = _tokenize(companion.creature_type or "")
    if not needle_tokens:
        return None

    matches: list[PremadeOption] = []
    for opt in options:
        haystack_tokens = _tokenize(opt.slug) | _tokenize(opt.creature_type)
        if needle_tokens <= haystack_tokens:
            matches.append(opt)
    if len(matches) == 1:
        return matches[0]
    return None


def match_premade(
    companion: CompanionProfile,
    options: tuple[PremadeOption, ...] | list[PremadeOption],
    config: CompanionConfig,
) -> tuple[PremadeOption, str]:
    """Pick the closest premade slug. Returns ``(option, reason)``.

    Fast path: a deterministic creature-type shortcut (no LLM call).
    Slow path: a single LLM call (no retries — matching is best-effort and
    falls back to the first option on failure).
    """
    if not options:
        raise RuntimeError("No premade options available")
    options_tuple: tuple[PremadeOption, ...] = tuple(options)

    shortcut = _heuristic_match(companion, options_tuple)
    if shortcut is not None:
        logger.debug("Premade heuristic shortcut: %s -> %s", companion.creature_type, shortcut.slug)
        return shortcut, f"creature_type '{companion.creature_type}' uniquely matches {shortcut.slug}"

    # LLM path. Imported lazily to avoid pulling asyncio at art-pkg import time.
    from claude_code_assist.profile.generator import _call_profile_llm  # noqa: PLC2701

    user_prompt = _build_match_user_prompt(companion, options_tuple)
    try:
        data = asyncio.run(_call_profile_llm(_MATCH_SYSTEM_PROMPT, user_prompt, config, context="premade match"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Premade LLM match failed: %s", exc)
        return options_tuple[0], f"(LLM match failed: {exc}; using {options_tuple[0].slug})"

    if not isinstance(data, dict):
        return options_tuple[0], f"(unexpected matcher response; using {options_tuple[0].slug})"

    slug = str(data.get("slug", "")).strip()
    reason = str(data.get("reason", "")).strip()

    by_slug = {opt.slug: opt for opt in options_tuple}
    if slug in by_slug:
        return by_slug[slug], reason
    logger.warning("Matcher returned unknown slug %r; falling back to %r", slug, options_tuple[0].slug)
    return options_tuple[0], f"(matcher returned unknown slug {slug!r}; using {options_tuple[0].slug})"
