"""Domain models — companion profile, rarity, stats."""

from claude_code_assist.models.companion import CompanionProfile
from claude_code_assist.models.rarity import (
    DEFAULT_RARITY_WEIGHTS,
    Rarity,
    compute_rarity_from_stats,
    pick_rarity,
    rarity_for_dump_stat,
    rarity_for_peak_stat,
)
from claude_code_assist.models.role import ROLE_CATALOG, Role, RoleDef, picker_label, picker_label_styled
from claude_code_assist.models.stats import (
    DEFAULT_STAT_NAMES,
    STAT_DEFINITIONS,
    StatConfig,
    generate_stats,
    shape_stats,
)

__all__ = [
    "DEFAULT_RARITY_WEIGHTS",
    "DEFAULT_STAT_NAMES",
    "ROLE_CATALOG",
    "STAT_DEFINITIONS",
    "CompanionProfile",
    "Rarity",
    "Role",
    "RoleDef",
    "StatConfig",
    "compute_rarity_from_stats",
    "generate_stats",
    "picker_label",
    "picker_label_styled",
    "pick_rarity",
    "rarity_for_dump_stat",
    "rarity_for_peak_stat",
    "shape_stats",
]
