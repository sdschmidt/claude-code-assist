"""Shared questionary picker styling and helpers.

Importing this module also installs a one-time monkey-patch that hides
questionary's hardcoded ``- `` prefix on disabled rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import questionary
from questionary.prompts.common import InquirerControl

if TYPE_CHECKING:
    from prompt_toolkit.key_binding import KeyBindings

# ``disabled`` rows recede in faint grey + italic; ``shortcut`` colours the
# inline shortcut letter orange; ``description`` fades the hyphen tagline.
PICKER_STYLE = questionary.Style(
    [
        ("disabled", "fg:#5a5a5a italic"),
        ("shortcut", "fg:#ff8c00 bold"),
        ("description", "fg:#888"),
    ]
)


def menu_title(
    label: str,
    description: str,
    *,
    shortcut: str | None = None,
    label_style: str = "",
) -> list[tuple[str, str]]:
    """Build a ``Choice`` title with an inline-highlighted shortcut letter.

    ``shortcut`` is matched case-insensitively against ``label``; the
    first matching character is rendered with ``class:shortcut``. The
    optional ``description`` follows after `` - `` in ``class:description``.
    """
    parts: list[tuple[str, str]] = []
    idx = label.lower().find(shortcut.lower()) if shortcut else -1
    if idx < 0:
        parts.append((label_style, label))
    else:
        if idx > 0:
            parts.append((label_style, label[:idx]))
        parts.append(("class:shortcut", label[idx]))
        if idx + 1 < len(label):
            parts.append((label_style, label[idx + 1 :]))
    if description:
        parts.append(("class:description", f"  -  {description}"))
    return parts


def bind_shortcuts(question: questionary.Question, shortcut_map: dict[str, Any]) -> None:
    """Eager-bind single-key shortcuts on a built questionary ``Question``.

    Pressing a registered key submits the picker immediately with the
    mapped value — no Enter required. Bindings are added to the
    Application's existing ``key_bindings`` so questionary's arrow-key
    navigation still works alongside.
    """
    bindings = cast("KeyBindings", question.application.key_bindings)

    def _make_handler(value: Any):
        def _handler(event):  # type: ignore[no-untyped-def]
            event.app.exit(result=value)

        return _handler

    for letter, value in shortcut_map.items():
        bindings.add(letter, eager=True)(_make_handler(value))


def _patch_questionary_disabled_prefix() -> None:
    """Strip questionary's hardcoded ``- `` prefix from disabled rows."""
    if getattr(InquirerControl._get_choice_tokens, "__ccassist_patched__", False):
        return
    original = InquirerControl._get_choice_tokens

    def patched(self):  # type: ignore[no-untyped-def]
        tokens = original(self)
        out = []
        for style, text, *rest in tokens:
            if style == "class:disabled" and text.startswith("- "):
                text = text[2:]
            out.append((style, text, *rest))
        return out

    patched.__ccassist_patched__ = True  # type: ignore[attr-defined]
    InquirerControl._get_choice_tokens = patched  # type: ignore[method-assign]


_patch_questionary_disabled_prefix()
