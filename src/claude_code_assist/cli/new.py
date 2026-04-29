"""``companion new`` — interactive companion generation.

Modes:

* ``--mode quiz`` (default) — Claude asks 3-5 short questions; answers
  feed into the generation prompt as user criteria.
* ``--mode free`` — single free-form description box.
* ``--mode random`` — no user input; pure random generation.

Each successful generation shows the companion's bio / backstory / stats /
rarity in the terminal. The user can ``proceed`` (saves to
``profile.json``) or ``re-roll`` (returns to the mode picker).

When ``profile.json`` already exists the user is prompted to confirm,
and on confirmation the previous profile is moved to
``<config>/archive/<timestamp>_<name>/profile.json`` so it can be
restored later via ``companion roster``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from claude_code_assist.cli._picker import PICKER_STYLE, bind_shortcuts, menu_title
from claude_code_assist.config import CompanionConfig, load_config
from claude_code_assist.models.rarity import pick_rarity
from claude_code_assist.models.role import ROLE_CATALOG, Role, picker_label_styled
from claude_code_assist.profile.generator import _call_profile_llm, generate_companion
from claude_code_assist.profile.storage import (
    PROFILE_FILENAME,
    allocate_companion_slot,
    get_active_companion_dir,
    load_profile,
    migrate_legacy_layout,
    save_profile,
    set_active_slot,
)

if TYPE_CHECKING:
    from claude_code_assist.models.companion import CompanionProfile

logger = logging.getLogger(__name__)
console = Console()

_QUIZ_FALLBACK_QUESTIONS = [
    "What kind of creature feels right? (any species, real or imaginary)",
    "What's its personality like in one phrase?",
    "What does it do when nothing's happening?",
    "What's it bad at?",
    "Anything else you'd like to add? (optional)",
]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="companion new",
        description="Generate a new companion for the desktop companion.",
    )
    parser.add_argument(
        "--mode",
        choices=("free", "quiz", "random"),
        default=None,
        help=("How to gather creation criteria. If omitted, you'll be asked interactively (arrow-key menu)."),
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Config directory override. Defaults to $XDG_CONFIG_HOME/claude-companion.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the overwrite confirmation when a companion already exists.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for the inspiration tags inside the generation prompt.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args(argv)


def run(argv: list[str]) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    config = _load_or_default_config(args.config_dir)
    if args.seed is not None:
        config = config.model_copy(update={"seed": args.seed})

    # Migrate any legacy layout into the current roster/<slot>/ shape
    # before we read or write anything.
    migrate_legacy_layout(config.config_dir)

    # Surface the active companion (if any) so the user knows what
    # they're potentially leaving behind. With the roster layout the
    # *previous* companion is never deleted — it just stops being
    # active when this run completes.
    active_dir = get_active_companion_dir(config.config_dir)
    if active_dir is not None and not args.yes:
        existing = load_profile(active_dir / PROFILE_FILENAME)
        existing_name = existing.name if existing is not None else active_dir.name
        prompt = (
            f"A companion named '{existing_name}' is currently active. "
            "Generate a new one and switch to it? "
            "(the previous companion stays in your roster, just not active)"
        )
        if _pick_new_or_quit(prompt) != "new":
            console.print("[yellow]Aborted — keeping the current companion.[/yellow]")
            return 0

    from claude_code_assist.qt.settings import SettingsStore  # noqa: PLC0415

    settings = SettingsStore(config.config_dir).load()

    mode = args.mode or _pick_mode_interactive()
    if mode is None or mode == "quit":
        return 0  # Ctrl-C, Esc, or explicit quit
    while True:
        criteria = _collect_criteria(mode, config)
        rarity = pick_rarity(config.rarity_weights)
        with console.status(f"[cyan]Generating a {rarity.value.lower()} companion…[/cyan]", spinner="dots"):
            try:
                companion, creation_prompt = generate_companion(config, rarity=rarity, criteria=criteria)
            except RuntimeError as exc:
                console.print(f"[red]Generation failed:[/red] {exc}")
                if not _ask_confirm("Try again?", default=True):
                    return 1
                continue

        _show_companion(companion)
        choice = _pick_proceed_action()
        if choice == "quit":
            console.print("[yellow]Aborted — nothing saved.[/yellow]")
            return 0
        # Treat ``None`` (Ctrl-C / Esc) as proceed: by the time the user
        # has seen a generated companion they probably don't want to lose
        # the roll just because they hit the wrong key. Explicit "quit"
        # is still the way to discard.
        if choice in ("proceed", None):
            # Ask for the role *after* the user has decided to keep the
            # companion — no point asking on rolls that get discarded.
            role = _pick_role_interactive()
            if role is not None:
                companion.role = role
            slot_dir = allocate_companion_slot(config.config_dir, companion.name)
            slot_dir.mkdir(parents=True, exist_ok=True)
            profile_path = slot_dir / PROFILE_FILENAME
            save_profile(companion, profile_path)
            set_active_slot(config.config_dir, slot_dir.name)
            if settings.creation_prompt_log:
                (slot_dir / "creation_prompt.txt").write_text(creation_prompt, encoding="utf-8")
            if choice is None:
                console.print(
                    f"[green]Saved {companion.name} to {profile_path}.[/green] "
                    "[dim](Ctrl-C treated as proceed — companion preserved.)[/dim]"
                )
            else:
                console.print(f"[green]Saved {companion.name} to {profile_path}.[/green]")
            console.print(
                "Run [bold]companion art[/bold] to generate sprite frames, then "
                "[bold]companion[/bold] to start the companion."
            )
            return 0
        # Reroll — let the user pick a different mode (or keep the current one).
        next_mode = _pick_mode_interactive(default=mode)
        if next_mode is None or next_mode == "quit":
            console.print("[yellow]Aborted — nothing saved.[/yellow]")
            return 0
        mode = next_mode


# ---------------------------------------------------------------------------
# Mode picker (arrow-key menu)
# ---------------------------------------------------------------------------


# (value, shortcut, label, description).
_MODE_CHOICES: tuple[tuple[str, str, str, str], ...] = (
    ("free", "p", "prompt", "describe your companion freely"),
    ("quiz", "z", "quiz", "answer a few short questions"),
    ("random", "r", "random", "pure roll, no input"),
    ("quit", "q", "quit", "leave without generating"),
)


_ACTION_CHOICES: tuple[tuple[str, str, str, str], ...] = (
    ("proceed", "p", "proceed", "save this companion"),
    ("reroll", "r", "reroll", "try a different generation"),
    ("quit", "q", "quit", "discard and exit"),
)


def _ask_confirm(message: str, *, default: bool = False) -> bool:
    """Yes/no prompt rendered with questionary. ``False`` on cancel."""
    try:
        answer = questionary.confirm(message, default=default).ask()
    except (KeyboardInterrupt, EOFError):
        return False
    return bool(answer) if answer is not None else False


_NEW_OR_QUIT_CHOICES: tuple[tuple[str, str, str, str], ...] = (
    ("new", "n", "new", "generate a new companion"),
    ("quit", "q", "quit", ""),
)


def _pick_new_or_quit(message: str) -> str | None:
    """Two-row select used in place of a yes/no overwrite confirmation."""
    choices = [
        questionary.Choice(title=menu_title(label, description, shortcut=key), value=value)
        for value, key, label, description in _NEW_OR_QUIT_CHOICES
    ]
    shortcut_map = {key: value for value, key, _, _ in _NEW_OR_QUIT_CHOICES}
    try:
        question = questionary.select(message, choices=choices, style=PICKER_STYLE)
        bind_shortcuts(question, shortcut_map)
        return question.ask()
    except (KeyboardInterrupt, EOFError):
        return None


def _pick_role_interactive() -> Role | None:
    """Arrow-key picker for the companion's role. ``None`` if user cancels."""
    choices = [questionary.Choice(title=picker_label_styled(defn), value=role) for role, defn in ROLE_CATALOG.items()]
    try:
        return questionary.select(
            "Choose a role for this companion:",
            choices=choices,
        ).ask()
    except (KeyboardInterrupt, EOFError):
        return None


def _pick_proceed_action() -> str | None:
    """Arrow-key picker for proceed / reroll / quit. ``None`` on cancel."""
    labels = [
        questionary.Choice(title=menu_title(label, description, shortcut=key), value=value)
        for value, key, label, description in _ACTION_CHOICES
    ]
    shortcut_map = {key: value for value, key, _, _ in _ACTION_CHOICES}
    try:
        question = questionary.select(
            "What now?",
            choices=labels,
            default=labels[0],
            style=PICKER_STYLE,
        )
        bind_shortcuts(question, shortcut_map)
        choice = question.ask()
    except (KeyboardInterrupt, EOFError):
        return None
    return None if choice is None else str(choice)


def _pick_mode_interactive(default: str | None = None) -> str | None:
    """Show an arrow-key menu of generation modes.

    The cursor opens on the first option unless ``default`` names a
    specific mode (used by the reroll path so the user's previous
    choice stays preselected). Arrow keys move the highlight; Enter
    confirms; Ctrl-C / Esc cancels and returns ``None``.
    """
    labels = [
        questionary.Choice(title=menu_title(label, description, shortcut=key), value=value)
        for value, key, label, description in _MODE_CHOICES
    ]
    shortcut_map = {key: value for value, key, _, _ in _MODE_CHOICES}
    select_kwargs: dict[str, object] = {"style": PICKER_STYLE}
    if default is not None:
        select_kwargs["default"] = next((c for c in labels if c.value == default), labels[0])
    try:
        question = questionary.select(
            "How do you want to design your companion?",
            choices=labels,
            **select_kwargs,  # type: ignore[arg-type]
        )
        bind_shortcuts(question, shortcut_map)
        choice = question.ask()
    except (KeyboardInterrupt, EOFError):
        return None
    if choice is None:
        return None
    return str(choice)


# ---------------------------------------------------------------------------
# Mode collectors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _QuizQuestion:
    text: str
    suggestions: list[str] = field(default_factory=list)


_OTHER_LABEL = "✎  Other (type your own answer)"
_SKIP_LABEL = "↷  Skip this question"


def _collect_criteria(mode: str, config: CompanionConfig) -> str | None:
    if mode == "random":
        return None
    if mode == "free":
        text = Prompt.ask("Describe your companion (or leave blank for random)").strip()
        return text or None
    # quiz mode (default) — the LLM call to design the questions can take
    # a few seconds, so wrap it in a spinner; otherwise the prompt looks
    # like it just hung after the menu picker.
    with console.status("[cyan]Asking Claude to draft the quiz…[/cyan]", spinner="dots"):
        questions = _generate_quiz_questions(config)

    answers: list[str] = []
    for i, question in enumerate(questions, start=1):
        answer = _ask_quiz_question(i, question)
        if answer is _CANCELLED:
            return None  # Ctrl-C abandons the whole quiz
        if answer:
            answers.append(f"{question.text} → {answer}")
    if not answers:
        return None
    return "User answered the following design questions:\n" + "\n".join(answers)


# Sentinel so the quiz can distinguish "user pressed Esc/Ctrl-C" from "user
# chose to skip this single question". Returning ``None`` means "skip";
# returning this sentinel means "abort the whole quiz".
_CANCELLED: object = object()


def _ask_quiz_question(index: int, question: _QuizQuestion) -> str | object | None:
    """Render one quiz question.

    Suggestions are shown as a single-choice select with two extra
    rows: ``Other`` (opens a free-text prompt) and ``Skip`` (drops the
    question). Without suggestions we go straight to free text.

    Returns the answer string, ``None`` for skip, or :data:`_CANCELLED`
    when the user pressed Ctrl-C / Esc.
    """
    prompt = f"Q{index}. {question.text}"

    if not question.suggestions:
        try:
            answer = questionary.text(prompt, instruction="(blank to skip)").ask()
        except (KeyboardInterrupt, EOFError):
            return _CANCELLED
        if answer is None:
            return _CANCELLED
        return answer.strip() or None

    choices: list[questionary.Choice] = [questionary.Choice(title=s, value=s) for s in question.suggestions]
    choices.append(questionary.Choice(title=_OTHER_LABEL, value=_OTHER_LABEL))
    choices.append(questionary.Choice(title=_SKIP_LABEL, value=_SKIP_LABEL))
    try:
        choice = questionary.select(prompt, choices=choices).ask()
    except (KeyboardInterrupt, EOFError):
        return _CANCELLED
    if choice is None:
        return _CANCELLED
    if choice == _SKIP_LABEL:
        return None
    if choice == _OTHER_LABEL:
        try:
            text = questionary.text("Your answer:", instruction="(blank to skip)").ask()
        except (KeyboardInterrupt, EOFError):
            return _CANCELLED
        if text is None:
            return _CANCELLED
        return text.strip() or None
    return str(choice)


def _generate_quiz_questions(config: CompanionConfig) -> list[_QuizQuestion]:
    """Ask the configured profile LLM for 3-5 questions + per-question samples.

    Falls back to a hardcoded list (no suggestions) if the LLM call
    fails or returns something unparseable — we don't want a flaky
    network connection to block ``companion new``.
    """
    system = (
        "You are helping a developer design a desktop companion. Generate 3-5 short, "
        "open-ended questions that would yield enough flavor for a creative companion "
        "designer to invent a memorable creature. Avoid yes/no questions.\n\n"
        "For each question also provide 3-4 short sample answers a developer might give "
        "(2-6 words each, no full sentences, no leading dashes). The user will be able to "
        "pick one of the samples or type their own.\n\n"
        "Output ONLY a JSON array of objects with keys:\n"
        '  - "question": string\n'
        '  - "suggestions": array of 3-4 short strings\n'
    )
    user = "Generate the quiz now."
    try:
        data = asyncio.run(_call_profile_llm(system, user, config, context="quiz questions"))
        items: list[object] = []
        if isinstance(data, list):
            items = list(data)
        elif isinstance(data, dict):
            for value in data.values():
                if isinstance(value, list):
                    items = list(value)
                    break
        questions: list[_QuizQuestion] = []
        for raw in items:
            if isinstance(raw, dict):
                q_text = str(raw.get("question", "")).strip()
                raw_suggestions = raw.get("suggestions", [])
                suggestions: list[str] = []
                if isinstance(raw_suggestions, list):
                    suggestions = [str(s).strip() for s in raw_suggestions if str(s).strip()]
                if q_text:
                    questions.append(_QuizQuestion(text=q_text, suggestions=suggestions[:6]))
            elif isinstance(raw, str) and raw.strip():
                questions.append(_QuizQuestion(text=raw.strip(), suggestions=[]))
        if 1 <= len(questions) <= 8:
            return questions
    except Exception as exc:  # noqa: BLE001 — fall back is intentional
        logger.warning("Quiz question generation failed, using fallback: %s", exc)
    return [_QuizQuestion(text=q) for q in _QUIZ_FALLBACK_QUESTIONS]


# ---------------------------------------------------------------------------
# Companion preview
# ---------------------------------------------------------------------------


def _safe_rich_color(name: str | None, fallback: str = "cyan") -> str:
    """Return ``name`` if Rich can parse it, else ``fallback``.

    LLMs occasionally invent plausible-sounding color names (``"storm_blue"``)
    that crash Rich's style parser. Fall back so a creative-but-invalid
    accent doesn't take down the preview panel.
    """
    if not name:
        return fallback
    from rich.color import Color, ColorParseError

    try:
        Color.parse(name)
    except ColorParseError:
        return fallback
    return name


def _show_companion(companion: CompanionProfile) -> None:
    # Frame, name, stars, and stat bars all use the rarity color so the
    # tier reads at a glance. ``accent_color`` from the LLM is ignored
    # for this view — rarity wins.
    color = _safe_rich_color(companion.rarity.color)

    header = Text()
    header.append(companion.name, style=f"bold {color}")
    header.append(f"  {companion.rarity.stars}", style=color)
    header.append(f"  · {companion.rarity.value.title()}", style=color)
    header.append(f"  · {companion.creature_type}", style="dim")

    body = Text()
    body.append("Personality\n", style="bold")
    body.append(companion.personality.strip() + "\n\n")
    body.append("Backstory\n", style="bold")
    body.append(companion.backstory.strip() + "\n")

    console.print(Panel(body, title=header, border_style=color, padding=(1, 2)))

    if companion.stats:
        table = Table(show_header=True, header_style="bold", title="Stats", expand=False)
        table.add_column("Stat")
        table.add_column("Value", justify="right")
        table.add_column("Bar")
        for name, value in companion.stats.items():
            clamped = max(0, min(100, int(value)))
            filled = clamped // 10
            bar = Text("█" * filled, style=color) + Text("░" * (10 - filled), style="dim")
            table.add_row(name, str(clamped), bar)
        console.print(table)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_or_default_config(config_dir: Path | None) -> CompanionConfig:
    """Load ``config.json`` if present, otherwise return defaults bound to ``config_dir``."""
    if config_dir is None:
        from claude_code_assist.paths import default_config_dir  # noqa: PLC0415

        config_dir = default_config_dir()

    # Run the legacy migration first so a config.yaml or companion_settings.json
    # left over from earlier launches gets folded into config.json.
    migrate_legacy_layout(config_dir)

    cfg_path = config_dir / "config.json"
    if cfg_path.is_file():
        return load_config(cfg_path).model_copy(update={"config_dir": config_dir})
    return CompanionConfig(config_dir=config_dir)
