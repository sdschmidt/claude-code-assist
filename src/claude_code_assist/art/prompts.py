"""Sprite-sheet prompt builder + locomotion override surface.

The prompt asks Gemini to render a 2x5 portrait grid (10 cells) on a
flat magenta background. The four locomotion descriptors come from the
companion's profile by default; ``LocomotionOverrides`` lets the
``companion art → adapt`` flow override them per-generation without
mutating the saved profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claude_code_assist.models.companion import CompanionProfile

CHROMA_KEY = (255, 0, 255)
CHROMA_KEY_HEX = "#FF00FF"


@dataclass
class LocomotionOverrides:
    """Optional per-generation overrides for the four locomotion fields."""

    body_plan: str | None = None
    walk_description: str | None = None
    fall_description: str | None = None
    landing_description: str | None = None


def _resolve(value: str | None, fallback: str) -> str:
    return (value or "").strip() or fallback


def _layout_instructions(companion: CompanionProfile, overrides: LocomotionOverrides | None) -> str:
    """Render the row-by-row 10-cell layout instructions."""
    body_plan = _resolve(
        overrides.body_plan if overrides else None,
        companion.body_plan or f"a {companion.creature_type}",
    )
    walk = _resolve(
        overrides.walk_description if overrides else None,
        companion.walk_description
        or "Two-stride walk cycle facing right; pose A and pose B differ in stride only, no horizontal mirroring.",
    )
    fall = _resolve(
        overrides.fall_description if overrides else None,
        companion.fall_description or "Airborne with limbs splayed and a startled expression — no ground under it.",
    )
    landing = _resolve(
        overrides.landing_description if overrides else None,
        companion.landing_description
        or "Dazed on the ground: eyes as two small X shapes, 2-3 small yellow stars around its head, mouth slightly open.",
    )

    return (
        "CRITICAL FORMATTING RULES (read these first):\n"
        "1. NO GRID LINES, NO BORDERS, NO PANEL OUTLINES, NO FRAME SEPARATORS, NO DIVIDERS. "
        "The 10 cells must be visually contiguous — only the magenta background and the "
        "characters appear. Do NOT draw any line, stroke, or border between cells.\n"
        "2. NO LABELS, NO TEXT, NO ROW/COLUMN NUMBERS, NO CAPTIONS anywhere on the image.\n"
        "3. The character MUST appear at the SAME SIZE in every cell — same body proportions, "
        "same scale, same number of pixels tall and wide. Do NOT zoom in for some frames or "
        "out for others. The bounding box of the character should be roughly identical across "
        "all 10 cells.\n"
        "4. Center the character horizontally within its cell, with consistent vertical placement "
        "(idle/walk: feet near the cell bottom; fall: mid-cell; landing: floor of the cell).\n\n"
        "Layout: 2 columns x 5 rows grid (10 panels). The overall image must be portrait "
        "orientation (taller than wide, approximately 2:5 aspect ratio). "
        f"The character has this anatomy: {body_plan}. Use this body plan consistently across "
        "every cell.\n"
        "Each panel shows a different animation frame:\n"
        "- Row 1, Left: Idle pose (relaxed, eyes fully open, on the ground)\n"
        "- Row 1, Right: Idle shift (same pose with tiny body/head tilt, eyes fully open)\n"
        "- Row 2, Left: BLINK — pixel-for-pixel IDENTICAL to Row 1 Left with ONE change: the eyes "
        "are closed (same closed-eye style as the sleeping frame). Nothing else changes.\n"
        "- Row 2, Right: BLINK — pixel-for-pixel IDENTICAL to Row 1 Right with ONE change: the "
        "eyes are closed. Nothing else changes.\n"
        "- Row 3, Left: Excited / surprised — eyes wide open, mouth open in a happy or "
        "startled expression. Same body and colors as idle.\n"
        "- Row 3, Right: Sleeping — eyes closed, tiny zzz bubbles above the head.\n"
        f"- Row 4, Left: Walk stride A. {walk} Eyes open, FACING TO THE RIGHT.\n"
        "- Row 4, Right: Walk stride B — FACING TO THE RIGHT, the SAME direction as Row 4 Left. "
        "Do NOT flip or mirror the character horizontally. This is the mid-stride pose.\n"
        f"- Row 5, Left: Falling. {fall} Keep the same character, same size, same style, same colors.\n"
        f"- Row 5, Right: Landing. {landing} Keep the same character, same colors, same style.\n"
        "\n"
        "Style: Clean pixel art, cute and charming. The creature must look identical across all "
        "frames (same proportions, colors, style). Only the expression and pose should differ.\n"
        "The creature MUST have a thick black outline around the entire body and major features. "
        "NO shadows, NO drop shadows, NO ambient occlusion. Flat solid colors only.\n\n"
        "IMPORTANT:\n"
        "NO LABELS\n"
        "Don't label the generated image or the individual frames\n"
        "Don't separate the individual frames at all -- no outline, no grid"
    )


def _default_subject(companion: CompanionProfile) -> str:
    return (
        f"Create a pixel art sprite sheet of a {companion.creature_type} named {companion.name}. "
        f"Personality: {companion.personality}. "
        f"Rarity: {companion.rarity.value}."
    )


def build_sprite_prompt(
    companion: CompanionProfile,
    overrides: LocomotionOverrides | None = None,
) -> str:
    """Return the full prompt for a 2x5 magenta-bg sprite sheet."""
    subject = _default_subject(companion)
    layout = _layout_instructions(companion, overrides)
    return (
        f"{subject}\n\n"
        f"{layout}\n\n"
        "Clean edges, no anti-aliasing into background.\n\n"
        f"CRITICAL: The background MUST be solid pure magenta {CHROMA_KEY_HEX} RGB(255,0,255) "
        "with absolutely no gradients, no shadows, no shading, no variations. "
        "A single flat magenta color across the entire background. "
        "Isolated subject, centered composition within each cell. "
        "This is essential for automated background removal processing."
    )
