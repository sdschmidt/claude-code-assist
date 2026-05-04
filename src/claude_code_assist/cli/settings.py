"""``companion settings`` — edit companion configuration from the CLI.

Two persistence layers live in the same ``config.json`` file:

- The tray-toggle block (``settings`` key) is owned by
  :class:`SettingsStore` in ``qt/settings.py``. Gravity, walking, scale,
  and the prompt-log toggles live here.
- The model-field block (everything else, including ``provider_config``)
  is owned by :class:`CompanionConfig`. The LLM provider/model picker
  writes here via :func:`save_config`, which preserves the tray block
  on round-trip.

The scale picker snaps to the same 80 %–200 % / 20 % grid the tray
slider uses (see ``qt/tray.py::_SCALE_PCT_*``); keeping the two ranges
in sync means a value chosen here looks identical when the slider
re-reads it.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import questionary
from rich.console import Console

from claude_code_assist.cli._picker import PICKER_STYLE, bind_shortcuts, menu_title
from claude_code_assist.config import (
    PROVIDER_MODEL_DEFAULTS,
    CompanionConfig,
    LLMProvider,
    ProviderConfig,
    default_model_for,
    load_config,
    save_config,
)
from claude_code_assist.qt.settings import CompanionSettings, SettingsStore

logger = logging.getLogger(__name__)
console = Console()

_SCALE_PCT_MIN = 80
_SCALE_PCT_MAX = 200
_SCALE_PCT_STEP = 20

_CUSTOM_MODEL_SENTINEL = "__custom__"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="companion settings",
        description="Edit companion settings (LLM provider, gravity, walking, scale).",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Config directory override. Defaults to $XDG_CONFIG_HOME/claude-companion.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args(argv)


def _resolve_config_dir(override: Path | None) -> Path:
    if override is not None:
        return override
    from claude_code_assist.paths import default_config_dir  # noqa: PLC0415

    return default_config_dir()


def _format_bool(value: bool) -> str:
    return "on" if value else "off"


def _scale_choices(current: float) -> list[questionary.Choice]:
    current_pct = int(round(current * 100))
    out: list[questionary.Choice] = []
    for pct in range(_SCALE_PCT_MIN, _SCALE_PCT_MAX + 1, _SCALE_PCT_STEP):
        marker = "  ← current" if pct == current_pct else ""
        out.append(questionary.Choice(title=f"{pct}%{marker}", value=pct))
    return out


def _toggle_gravity(settings: CompanionSettings) -> bool:
    settings.gravity_enabled = not settings.gravity_enabled
    return True


def _toggle_walking(settings: CompanionSettings) -> bool:
    settings.walking_enabled = not settings.walking_enabled
    return True


def _edit_scale(settings: CompanionSettings) -> bool:
    try:
        chosen = questionary.select(
            "Companion scale:",
            choices=_scale_choices(settings.companion_scale),
        ).ask()
    except (KeyboardInterrupt, EOFError):
        return False
    if chosen is None:
        return False
    new_scale = float(chosen) / 100.0
    if abs(new_scale - settings.companion_scale) < 1e-6:
        return False
    settings.companion_scale = new_scale
    return True


def _edit_provider(config: CompanionConfig) -> bool:
    """Pick a provider. Resets model to the new provider's default."""
    current = config.provider_config.provider
    choices = [
        questionary.Choice(
            title=f"{p.value}{'  ← current' if p == current else ''}",
            value=p.value,
        )
        for p in LLMProvider
    ]
    try:
        chosen = questionary.select("LLM provider:", choices=choices, style=PICKER_STYLE).ask()
    except (KeyboardInterrupt, EOFError):
        return False
    if chosen is None:
        return False
    new_provider = LLMProvider(chosen)
    if new_provider == current:
        return False
    # Reset model to empty so resolve() picks the new provider's default.
    # The user can then customise via the model picker.
    config.provider_config = ProviderConfig(provider=new_provider, model="")
    return True


def _edit_model(config: CompanionConfig) -> bool:
    """Pick a model from the provider's curated list, or enter a custom string."""
    provider = config.provider_config.provider
    current = config.provider_config.model or default_model_for(provider)
    curated = PROVIDER_MODEL_DEFAULTS[provider]

    choices: list[questionary.Choice] = [
        questionary.Choice(
            title=f"{m}{'  ← current' if m == current else ''}",
            value=m,
        )
        for m in curated
    ]
    if current not in curated:
        choices.append(questionary.Choice(title=f"{current}  ← current (custom)", value=current))
    choices.append(questionary.Choice(title="Custom…", value=_CUSTOM_MODEL_SENTINEL))

    try:
        chosen = questionary.select(f"Model for {provider.value}:", choices=choices, style=PICKER_STYLE).ask()
    except (KeyboardInterrupt, EOFError):
        return False
    if chosen is None:
        return False

    if chosen == _CUSTOM_MODEL_SENTINEL:
        try:
            entered = questionary.text(
                "Model name:",
                default=current,
            ).ask()
        except (KeyboardInterrupt, EOFError):
            return False
        if entered is None:
            return False
        chosen = entered.strip()
        if not chosen:
            return False

    if chosen == config.provider_config.model:
        return False
    config.provider_config = ProviderConfig(provider=provider, model=chosen)
    return True


# (value, shortcut, label).
_SETTINGS_ROWS: tuple[tuple[str, str, str], ...] = (
    ("provider", "p", "provider"),
    ("model", "m", "model"),
    ("gravity", "g", "gravity"),
    ("walking", "w", "walking"),
    ("scale", "s", "scale"),
    ("commentary_prompt_log", "c", "commentary prompt log"),
    ("art_prompt_log", "a", "art prompt log"),
    ("creation_prompt_log", "r", "creation prompt log"),
)


def _row_description(value: str, settings: CompanionSettings, config: CompanionConfig) -> str:
    if value == "scale":
        return f"{int(round(settings.companion_scale * 100))}%"
    if value == "gravity":
        return _format_bool(settings.gravity_enabled)
    if value == "walking":
        return _format_bool(settings.walking_enabled)
    if value == "commentary_prompt_log":
        return _format_bool(settings.commentary_prompt_log)
    if value == "art_prompt_log":
        return _format_bool(settings.art_prompt_log)
    if value == "creation_prompt_log":
        return _format_bool(settings.creation_prompt_log)
    if value == "provider":
        return config.provider_config.provider.value
    if value == "model":
        return config.provider_config.model or default_model_for(config.provider_config.provider)
    return ""


def _menu_choices(
    settings: CompanionSettings, config: CompanionConfig
) -> tuple[list[questionary.Choice], dict[str, str]]:
    choices: list[questionary.Choice] = [
        questionary.Choice(
            title=menu_title(label, _row_description(value, settings, config), shortcut=key),
            value=value,
        )
        for value, key, label in _SETTINGS_ROWS
    ]
    choices.append(questionary.Choice(title=menu_title("quit", "", shortcut="q"), value="quit"))
    shortcut_map = {key: value for value, key, _ in _SETTINGS_ROWS}
    shortcut_map["q"] = "quit"
    return choices, shortcut_map


def run(argv: list[str]) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    config_dir = _resolve_config_dir(args.config_dir)
    config_path = config_dir / "config.json"

    store = SettingsStore(config_dir)
    settings = store.load()
    config = load_config(config_path)
    # Pin config_dir so save_config writes back to the right place even
    # when the loaded file didn't carry one.
    config.config_dir = config_dir

    console.print("[bold]Companion settings[/bold]")

    while True:
        choices, shortcut_map = _menu_choices(settings, config)
        try:
            question = questionary.select(
                "Pick a setting to change:",
                choices=choices,
                style=PICKER_STYLE,
            )
            bind_shortcuts(question, shortcut_map)
            choice = question.ask()
        except (KeyboardInterrupt, EOFError):
            choice = None

        if choice is None or choice == "quit":
            store.save(settings)
            save_config(config, config_path)
            return 0

        settings_changed = False
        config_changed = False
        if choice == "gravity":
            settings_changed = _toggle_gravity(settings)
        elif choice == "walking":
            settings_changed = _toggle_walking(settings)
        elif choice == "scale":
            settings_changed = _edit_scale(settings)
        elif choice == "commentary_prompt_log":
            settings.commentary_prompt_log = not settings.commentary_prompt_log
            settings_changed = True
        elif choice == "art_prompt_log":
            settings.art_prompt_log = not settings.art_prompt_log
            settings_changed = True
        elif choice == "creation_prompt_log":
            settings.creation_prompt_log = not settings.creation_prompt_log
            settings_changed = True
        elif choice == "provider":
            config_changed = _edit_provider(config)
        elif choice == "model":
            config_changed = _edit_model(config)

        if settings_changed:
            store.save(settings)
        if config_changed:
            save_config(config, config_path)
