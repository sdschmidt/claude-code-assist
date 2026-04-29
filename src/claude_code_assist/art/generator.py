"""End-to-end art pipeline: prompt → Gemini → split → chroma-key → save."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from claude_code_assist.art.client import GeminiImageClient
from claude_code_assist.art.meta import ArtMeta, write_meta
from claude_code_assist.art.process import (
    CHROMA_BG,
    detect_2x5_cells,
    paint_over_grid_lines,
    remove_chroma_key,
    split_sprite_sheet_2x5,
)
from claude_code_assist.art.prompts import build_sprite_prompt

if TYPE_CHECKING:
    from claude_code_assist.art.prompts import LocomotionOverrides
    from claude_code_assist.models.companion import CompanionProfile

logger = logging.getLogger(__name__)

SPRITE_FILENAME = "sprite.png"
FRAME_COUNT = 10


def generate_frames(
    companion: CompanionProfile,
    art_dir: Path,
    *,
    overrides: LocomotionOverrides | None = None,
    api_key: str | None = None,
    write_prompt_log: bool = False,
) -> list[Path]:
    """Call Gemini, save the raw sheet, then split + clean. Returns frame paths.

    When ``write_prompt_log`` is true, the full sprite prompt is also
    saved alongside the frames as ``<art_dir>/prompt.txt`` — gated on
    the ``art_prompt_log`` setting so a normal install doesn't litter
    the art folder with a redundant copy of what's already in
    ``meta.json``.
    """
    art_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_sprite_prompt(companion, overrides)
    sprite_path = art_dir / SPRITE_FILENAME

    client = GeminiImageClient(api_key=api_key)
    client.generate_sprite(prompt, sprite_path, aspect_ratio="9:16", resolution="2k")

    sprite = Image.open(sprite_path)
    frame_paths = split_and_clean(
        sprite,
        art_dir,
        remove_grid=True,
        smart_split=True,
        contiguous_chroma=False,
        feather_px=2,
    )

    write_meta(
        art_dir,
        ArtMeta(model=client.model, prompt=prompt),
    )
    if write_prompt_log:
        (art_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    return frame_paths


def split_and_clean(
    sprite: Image.Image,
    art_dir: Path,
    *,
    remove_grid: bool,
    smart_split: bool,
    contiguous_chroma: bool,
    feather_px: int = 0,
) -> list[Path]:
    """Split a sprite sheet into 10 cleaned ``frame_{N}.png`` files.

    When ``smart_split=True`` the splitter detects cell boundaries by
    locating magenta corridors between cells and falls back to the
    fixed 2×5 fractional split if detection fails. ``remove_grid``
    paints over any dark dividers Gemini drew with chroma so both the
    detector and chroma-key removal see a continuous magenta
    background.
    """
    art_dir.mkdir(parents=True, exist_ok=True)

    sheet = sprite
    if remove_grid:
        sheet = paint_over_grid_lines(sprite, bg_color=CHROMA_BG)

    frames: list[Image.Image] = []
    if smart_split:
        cells = detect_2x5_cells(sheet.convert("RGB"))
        if cells is not None and len(cells) == FRAME_COUNT:
            logger.debug("Smart-split: detected %d cells", len(cells))
            sheet_rgba = sheet.convert("RGBA")
            for left, top, right, bottom in cells:
                frames.append(sheet_rgba.crop((left, top, right, bottom)))
        else:
            logger.debug("Smart-split detection failed; falling back to fixed split")

    if not frames:
        frames = split_sprite_sheet_2x5(sheet, inset_px=0)

    if len(frames) != FRAME_COUNT:
        raise RuntimeError(f"Expected {FRAME_COUNT} frames after split, got {len(frames)}")

    out_paths: list[Path] = []
    for i, frame in enumerate(frames):
        cleaned = remove_chroma_key(frame, contiguous=contiguous_chroma, feather_px=feather_px)
        path = art_dir / f"frame_{i}.png"
        cleaned.save(path, format="PNG")
        out_paths.append(path)

    return out_paths
