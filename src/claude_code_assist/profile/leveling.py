"""Player-driven leveling.

The companion no longer levels up automatically. Instead the
**player** initiates each level-up at startup (or via the
``companion levelup`` debug command), picks one stat to boost by 1,
and the rarity is recomputed from the new stat block. The counters
(``comment_counter``, ``last_seen_date``) gate eligibility and reset
on each successful level-up.

Eligibility = either condition is met:

* ``last_seen_date`` is set AND differs from today (a new calendar day)
* ``comment_counter`` has reached :data:`COMMENT_LEVEL_THRESHOLD`

A freshly created companion has ``last_seen_date=None``; the first
launch after creation seeds it via :func:`seed_last_seen_date` without
leveling up — otherwise the player would be prompted immediately after
``companion new``.

Both counters are reset on a successful level-up.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from claude_code_assist.models.rarity import Rarity, compute_rarity_from_stats

if TYPE_CHECKING:
    from claude_code_assist.models.companion import CompanionProfile

COMMENT_LEVEL_THRESHOLD = 100
MAX_STAT_VALUE = 100

XP_BAR_CELLS = 14
_BAR_FILLED = "█"
_BAR_EMPTY = "░"


def format_xp_bar_segments(
    counter: int,
    *,
    cells: int = XP_BAR_CELLS,
    threshold: int = COMMENT_LEVEL_THRESHOLD,
) -> tuple[str, str]:
    """Return ``(filled, empty)`` strings for a unicode XP progress bar.

    ``counter`` is clamped to ``[0, threshold]`` so excess comments past
    a missed level-up still render as a full bar without overflowing.
    Filled and empty segments are returned separately so the caller can
    color them independently (e.g. rarity-colored fill, dim empty).
    """
    if threshold <= 0 or cells <= 0:
        return "", _BAR_EMPTY * max(0, cells)
    progress = max(0, min(counter, threshold))
    filled = max(0, min(cells, int((progress / threshold) * cells)))
    return _BAR_FILLED * filled, _BAR_EMPTY * (cells - filled)


def is_eligible_for_levelup(companion: CompanionProfile, today: date | None = None) -> bool:
    """Return ``True`` when the player has earned at least one level-up.

    A ``last_seen_date`` of ``None`` (freshly created companion) is
    *not* eligible — the first launch seeds the field via
    :func:`seed_last_seen_date` instead.
    """
    today = today or date.today()
    new_day = companion.last_seen_date is not None and companion.last_seen_date != today
    enough_comments = companion.comment_counter >= COMMENT_LEVEL_THRESHOLD
    return new_day or enough_comments


def seed_last_seen_date(companion: CompanionProfile, today: date | None = None) -> bool:
    """Mark a freshly created companion as seen today without leveling up.

    Returns ``True`` if the profile was mutated (caller should save).
    """
    if companion.last_seen_date is not None:
        return False
    companion.last_seen_date = today or date.today()
    return True


def eligibility_reason(companion: CompanionProfile, today: date | None = None) -> str:
    """Human-readable string describing *why* a level-up is on the table."""
    today = today or date.today()
    reasons: list[str] = []
    if companion.last_seen_date is not None and companion.last_seen_date != today:
        reasons.append("new day")
    if companion.comment_counter >= COMMENT_LEVEL_THRESHOLD:
        reasons.append(f"{companion.comment_counter} comments since last level")
    return " · ".join(reasons) if reasons else "not eligible"


def record_comment(companion: CompanionProfile) -> None:
    """Increment the rolling comment counter. No automatic level-up."""
    companion.comment_counter += 1


def apply_player_levelup(
    companion: CompanionProfile,
    stat_name: str,
    today: date | None = None,
) -> tuple[Rarity, Rarity]:
    """Boost ``stat_name`` by 1, increment level, recompute rarity, reset counters.

    Returns ``(old_rarity, new_rarity)`` so the caller can decide
    whether to surface a "rarity changed" message. The stat is
    capped at :data:`MAX_STAT_VALUE`.
    """
    today = today or date.today()
    if stat_name not in companion.stats:
        raise KeyError(f"{stat_name!r} is not a stat on this companion")

    companion.stats[stat_name] = min(MAX_STAT_VALUE, companion.stats[stat_name] + 1)
    companion.level += 1

    old_rarity = companion.rarity
    new_rarity = compute_rarity_from_stats(companion.stats)
    companion.rarity = new_rarity

    # Counters reset: today's date is now "seen"; comment counter
    # back to zero (excess carries forward — keep extras over the
    # threshold so the player isn't punished for bursty sessions).
    companion.last_seen_date = today
    companion.comment_counter = max(0, companion.comment_counter - COMMENT_LEVEL_THRESHOLD)

    return old_rarity, new_rarity
