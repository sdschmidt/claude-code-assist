"""Companion profile data model."""

from datetime import UTC, date, datetime

from pydantic import BaseModel, Field

from claude_code_assist.models.rarity import Rarity
from claude_code_assist.models.role import Role


class CompanionProfile(BaseModel):
    """Complete companion profile with personality, stats, and art."""

    name: str = Field(description="Generated creature name.")
    creature_type: str = Field(description="Species or creature type (e.g. axolotl, phoenix).")
    rarity: Rarity = Field(description="Rarity tier: Common, Uncommon, Rare, Epic, or Legendary.")
    personality: str = Field(description="2-3 sentence personality summary used in commentary prompts.")
    backstory: str = Field(description="3-5 sentence origin story.")
    stats: dict[str, int] = Field(
        description="Stat name to integer value mapping (e.g. HUMOR: 72). Values are clamped to the rarity range."
    )
    accent_color: str = Field(description="Rich color name used for the art panel border (e.g. 'bright_cyan').")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="UTC timestamp of companion creation."
    )
    last_activated_at: datetime | None = Field(
        default=None,
        description=(
            "UTC timestamp of the most recent activation (``set_active_slot``). "
            "``None`` for legacy profiles created before this field existed; the "
            "roster picker falls back to ``created_at`` for sorting in that case."
        ),
    )
    last_comment: str | None = Field(default=None, description="Most recent comment text, persisted across sessions.")
    comment_history: list[str] = Field(
        default_factory=list, description="Rolling history of recent comments (max 20 entries)."
    )
    level: int = Field(
        default=1,
        ge=1,
        description="Current level. Starts at 1; +1 per 100 comments and +1 per new calendar day.",
    )
    comment_counter: int = Field(
        default=0,
        ge=0,
        description="Comments produced since last level-up (resets on level-up; excess carries over).",
    )
    last_seen_date: date | None = Field(
        default=None,
        description=(
            "Calendar date of the most recent app start that triggered the daily level check. "
            "``None`` until first launch — the first daily check seeds it without leveling up."
        ),
    )
    role: Role | None = Field(
        default=None,
        description=(
            "Optional commentary specialisation (Archmage, Scholar, Thief, "
            "Sentinel, Paladin, Bard). Picked after creature generation; "
            "appended to the system prompt at commentary time. ``None`` for "
            "legacy profiles created before roles existed."
        ),
    )
    body_plan: str = Field(
        default="",
        description=(
            "Anatomical description of the creature for image generation (limbs, wings, body shape, "
            "locomotion mode). Generated lazily by the LLM if missing. Empty for legacy profiles."
        ),
    )
    walk_description: str = Field(
        default="",
        description=(
            "Two-stride walk-cycle description used in the macos-desktop sprite prompt. "
            "Should reflect the creature's locomotion (4-leg trot, 6-leg scuttle, slither, hover, etc.)."
        ),
    )
    fall_description: str = Field(
        default="",
        description=(
            "Description of how the creature falls (glides if winged, drifts if floaty, plummets otherwise) "
            "used in the macos-desktop sprite prompt."
        ),
    )
    landing_description: str = Field(
        default="",
        description=(
            "Description of how the creature lands (soft-touch for gliders/floaters, "
            "hard impact reaction otherwise — splat, shatter, dent, dazed) used in the macos-desktop sprite prompt."
        ),
    )
