"""Companion roles — commentary specialisations.

A companion's role narrows what it pays attention to during the
session. The flavour names (``Thief``, ``Druid``, ``Archmage``…) live
in the UI — picker, tray, profile — but the prompt fragment fed to
the model is plain functional language. Voice comes from the
companion's personality and stats; the role only decides *what* to
look at.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Role(StrEnum):
    ARCHMAGE = "Archmage"
    SCHOLAR = "Scholar"
    THIEF = "Thief"
    SENTINEL = "Sentinel"
    BERSERKER = "Berserker"
    PALADIN = "Paladin"
    BARD = "Bard"
    LOREKEEPER = "Lorekeeper"
    DRUID = "Druid"
    SAGE = "Sage"


@dataclass(frozen=True)
class RoleDef:
    role: Role
    domain: str
    description: str
    prompt: str
    color: str  # ``#rrggbb`` — kept muted so rarity colors stay dominant


ROLE_CATALOG: dict[Role, RoleDef] = {
    Role.ARCHMAGE: RoleDef(
        role=Role.ARCHMAGE,
        domain="Architecture",
        description="weaver of grand systems, sees the shape of the design",
        prompt=(
            "Your concern is architecture: layering, coupling, abstraction "
            "boundaries, the shape of the system as a whole. Watch for "
            "leaky abstractions, circular dependencies, a function or "
            "class taking on too many responsibilities, hidden coupling "
            "between modules, missing seams between layers, business "
            "logic ending up in transport / IO code."
        ),
        color="#7388b3",
    ),
    Role.SCHOLAR: RoleDef(
        role=Role.SCHOLAR,
        domain="Style & idiom",
        description="pedant of style, has read a thousand codebases",
        prompt=(
            "Your concern is style and idiom at the textual level. Watch "
            "for inconsistent naming, mixed paradigms within one file, "
            "dead code, magic numbers, copy-paste duplication, stale "
            "comments contradicting the code, unidiomatic constructs. "
            "Stay at the surface — don't escalate into architecture."
        ),
        color="#b39673",
    ),
    Role.THIEF: RoleDef(
        role=Role.THIEF,
        domain="Debugging",
        description="bug-hunter in the shadows, finds hidden flaws",
        prompt=(
            "Your concern is bugs and error-hunting. Watch for off-by-one "
            "in loops or ranges, unchecked None / null / empty, races on "
            "shared state, swallowed exceptions, wrong loop bounds, "
            "implicit type coercion, time-of-check vs time-of-use gaps, "
            "missed branches, silent failures. Name the leak and where "
            "it leaks."
        ),
        color="#736b85",
    ),
    Role.SENTINEL: RoleDef(
        role=Role.SENTINEL,
        domain="Security",
        description="vigilant watcher, knows the threats",
        prompt=(
            "Your concern is security. Watch for untrusted input flowing "
            "into queries / shell / eval / HTML, secrets in code or logs, "
            "missing or broken access checks, dangerous defaults, missing "
            "rate limits, weak crypto, trust placed in client-supplied "
            "data."
        ),
        color="#6b8a96",
    ),
    Role.BERSERKER: RoleDef(
        role=Role.BERSERKER,
        domain="Performance",
        description="obsessed with speed, hates wasted cycles",
        prompt=(
            "Your concern is performance. Watch for O(n²) hidden in "
            "nested loops, repeated work that could be cached or hoisted, "
            "allocations inside hot loops, blocking I/O on a hot path, "
            "eager queries inside a tight scope, string concatenation in "
            "a loop, regex compiled per call."
        ),
        color="#b3736b",
    ),
    Role.PALADIN: RoleDef(
        role=Role.PALADIN,
        domain="Testing",
        description="purifier, demands proof through trials",
        prompt=(
            "Your concern is testing. Watch for new code without tests, "
            "untested error / edge branches, mocks that hide real "
            "behaviour, missing negative tests, asserts that don't "
            "actually assert, contract changes with no test update, flaky "
            "time-based tests."
        ),
        color="#b3a673",
    ),
    Role.BARD: RoleDef(
        role=Role.BARD,
        domain="Creativity",
        description="lateral thinker, sparks new approaches",
        prompt=(
            "Your concern is creative alternatives. Suggest simpler "
            "reframings of the current approach, alternative data shapes "
            "(table → tree, list → set), a library function that already "
            "does this, an entirely different angle, or a constraint "
            "worth questioning. Spark ideas; don't critique."
        ),
        color="#b37388",
    ),
    Role.LOREKEEPER: RoleDef(
        role=Role.LOREKEEPER,
        domain="Documentation",
        description="records the saga, watches for lost knowledge",
        prompt=(
            "Your concern is documentation and recorded knowledge. Watch "
            "for public APIs without docstrings, stale comments "
            "contradicting the code, opaque names that need a one-line "
            "rationale, non-obvious decisions with no recorded why, "
            "TODOs without context."
        ),
        color="#8a9670",
    ),
    Role.DRUID: RoleDef(
        role=Role.DRUID,
        domain="Refactoring",
        description="tends the codebase like a forest — prunes, regrows",
        prompt=(
            "Your concern is refactoring. Watch for dead code, "
            "near-duplicates that want unification, overgrown functions "
            "ready to split, patterns repeated three or more times, "
            "deeply nested branches that flatten with early return, "
            "parameters threaded through many layers that want a struct."
        ),
        color="#6ba65a",
    ),
    Role.SAGE: RoleDef(
        role=Role.SAGE,
        domain="Teaching",
        description="wise elder, gentle explainer",
        prompt=(
            "Your concern is teaching. When you see something instructive, "
            "surface the underlying concept, the implicit tradeoff, the "
            "subtle reason something works, or a useful name for a pattern "
            "the developer is reinventing. Favour curiosity over critique."
        ),
        color="#8a73b3",
    ),
}


def picker_label(definition: RoleDef) -> str:
    """Plain-text label — kept for callers that don't want the styled form."""
    return f"{definition.role.value} - {definition.description} ({definition.domain})"


def picker_label_styled(definition: RoleDef) -> list[tuple[str, str]]:
    """Styled label for ``questionary.select`` — role name colored.

    Returns the prompt-toolkit ``[(style, text), …]`` form, with the
    role name in the role's hex color and the rest dimmed so the eye
    snaps to the role first.
    """
    return [
        (f"fg:{definition.color} bold", definition.role.value),
        ("fg:ansibrightblack", f" - {definition.description} ({definition.domain})"),
    ]
