"""Shared interactive level-up flow.

Used by:
* ``qt/app.py`` at startup, gated by :func:`is_eligible_for_levelup`
* ``cli/levelup.py`` (the ``companion levelup`` debug command), which
  bypasses the eligibility check (``force=True``)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import questionary
from rich.console import Console

from claude_code_assist.profile.leveling import (
    apply_player_levelup,
    eligibility_reason,
    is_eligible_for_levelup,
)

if TYPE_CHECKING:
    from claude_code_assist.models.companion import CompanionProfile

console = Console()


def run_levelup_interactive(companion: CompanionProfile, *, force: bool = False) -> bool:
    """Prompt the player to level up and pick a stat.

    Returns ``True`` when the profile was mutated (level + stat +
    counters), so the caller knows to save. ``False`` if the player
    declined, cancelled, or there's nothing to do.
    """
    if not force and not is_eligible_for_levelup(companion):
        return False

    color = companion.rarity.color
    reason = eligibility_reason(companion) if not force else "forced"
    console.print(
        f"[bold {color}]{companion.name}[/bold {color}]"
        f"  [dim]· Lv. {companion.level} → Lv. {companion.level + 1}"
        f"  ·  {reason}[/dim]"
    )

    # No "Level up your companion?" prompt: reaching this function
    # already implies the user picked "levelup" from the menu (or ran
    # the ``companion levelup`` debug command), so a second confirm
    # is just friction. Skip straight to the stat picker.

    stat_choices = [
        questionary.Choice(
            title=f"{name}: {value}/100",
            value=name,
        )
        for name, value in companion.stats.items()
    ]
    try:
        chosen_stat = questionary.select(
            "Boost which stat by 1?",
            choices=stat_choices,
        ).ask()
    except (KeyboardInterrupt, EOFError):
        return False
    if chosen_stat is None:
        return False

    old_rarity, new_rarity = apply_player_levelup(companion, chosen_stat)

    console.print(
        f"[green]+1 {chosen_stat}[/green] "
        f"[dim](now {companion.stats[chosen_stat]}/100)[/dim]   "
        f"[green]Lv. {companion.level}[/green]"
    )
    if new_rarity != old_rarity:
        old_color = old_rarity.color
        new_color = new_rarity.color
        console.print(
            f"[{old_color}]{old_rarity.value}[/{old_color}] → "
            f"[bold {new_color}]{new_rarity.value}[/bold {new_color}]"
            "  [dim](rarity recomputed from new stats)[/dim]"
        )
    return True
