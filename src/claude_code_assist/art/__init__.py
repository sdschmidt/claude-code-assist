"""Sprite-art pipeline — Gemini generation, chroma-key, frame splitting.

Top-level imports are lazy via ``__getattr__`` so installing the project
without optional image deps (``pillow``, ``numpy``, ``google-genai``)
still lets the rest of the package load — only ``companion art`` then
errors out with a friendly hint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claude_code_assist.art.generator import generate_frames, split_and_clean
    from claude_code_assist.art.premade import (
        PremadeOption,
        copy_premade,
        list_premade_options,
        match_premade,
    )

__all__ = [
    "PremadeOption",
    "copy_premade",
    "generate_frames",
    "list_premade_options",
    "match_premade",
    "split_and_clean",
]


def __getattr__(name: str) -> Any:
    if name == "generate_frames":
        from claude_code_assist.art.generator import generate_frames

        return generate_frames
    if name == "split_and_clean":
        from claude_code_assist.art.generator import split_and_clean

        return split_and_clean
    if name in {"PremadeOption", "copy_premade", "list_premade_options", "match_premade"}:
        from claude_code_assist.art import premade

        return getattr(premade, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
