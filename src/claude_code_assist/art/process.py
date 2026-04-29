"""Image processing — chroma-key removal, grid-line painting, 2x5 splitting."""

from __future__ import annotations

import logging
from collections import deque

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

CHROMA_BG = (255, 0, 255)

# Fixed-fraction layout: 2 cols x 5 rows.
_FRAME_POSITIONS_2x5 = [
    (0.0, 0.0, 0.5, 0.2),  # 0 idle A
    (0.5, 0.0, 1.0, 0.2),  # 1 idle B
    (0.0, 0.2, 0.5, 0.4),  # 2 blink A
    (0.5, 0.2, 1.0, 0.4),  # 3 blink B
    (0.0, 0.4, 0.5, 0.6),  # 4 excited
    (0.5, 0.4, 1.0, 0.6),  # 5 sleep
    (0.0, 0.6, 0.5, 0.8),  # 6 walk A
    (0.5, 0.6, 1.0, 0.8),  # 7 walk B
    (0.0, 0.8, 0.5, 1.0),  # 8 fall
    (0.5, 0.8, 1.0, 1.0),  # 9 stunned
]


# ---------------------------------------------------------------------------
# Background detection + chroma-key removal
# ---------------------------------------------------------------------------


def _detect_bg_color(
    image: Image.Image,
    target_color: tuple[int, int, int] | None = None,
    target_radius: int = 120,
) -> tuple[int, int, int]:
    """Detect bg color by sampling the image's 2px border."""
    from collections import Counter  # noqa: PLC0415

    img = image.convert("RGB")
    arr = np.array(img)

    border_pixels = np.concatenate(
        [
            arr[:2, :, :].reshape(-1, 3),
            arr[-2:, :, :].reshape(-1, 3),
            arr[:, :2, :].reshape(-1, 3),
            arr[:, -2:, :].reshape(-1, 3),
        ]
    )

    if target_color is not None:
        target = np.array(target_color, dtype=np.float64)
        dists = np.sqrt(np.sum((border_pixels.astype(np.float64) - target) ** 2, axis=1))
        close_mask = dists <= target_radius
        if close_mask.sum() >= max(10, int(border_pixels.shape[0] * 0.01)):
            avg = border_pixels[close_mask].astype(np.float64).mean(axis=0)
            return (int(avg[0]), int(avg[1]), int(avg[2]))

    quantized = (border_pixels.astype(np.int32) // 16 * 16).astype(np.uint8)
    tuples = [tuple(p) for p in quantized]
    dominant = Counter(tuples).most_common(1)[0][0]
    mask = np.all(quantized == np.array(dominant), axis=1)
    bg_pixels = border_pixels[mask].astype(np.float64)
    avg = bg_pixels.mean(axis=0)
    return (int(avg[0]), int(avg[1]), int(avg[2]))


def remove_chroma_key(
    image: Image.Image,
    *,
    bg_color: tuple[int, int, int] | None = None,
    tolerance: int = 80,
    target_color: tuple[int, int, int] | None = CHROMA_BG,
    contiguous: bool = True,
    feather_px: int = 0,
) -> Image.Image:
    """Remove the magenta background.

    With ``contiguous=True`` (default) only edge-connected background
    pixels are removed (preserves interior chroma). With
    ``contiguous=False`` every pixel within tolerance is cleared,
    including pockets enclosed by the character outline.

    ``feather_px`` shrinks the opaque region inward by that many pixels
    with a linear alpha ramp at the new edge. Useful for killing the
    last sliver of pink halo that survives the binary tolerance check
    — e.g. anti-aliased outline pixels whose color is character +
    chroma blend but falls outside ``tolerance``. ``0`` disables the
    feather entirely.
    """
    detected_bg = bg_color or _detect_bg_color(image, target_color=target_color)

    img = image.convert("RGBA")
    arr = np.array(img)
    h, w = arr.shape[:2]

    bg = np.array(detected_bg, dtype=np.float64)
    dist = np.sqrt(np.sum((arr[:, :, :3].astype(np.float64) - bg) ** 2, axis=2))
    is_bg_local = dist <= tolerance

    if contiguous:
        background = _flood_fill_from_edges(is_bg_local)
    else:
        background = is_bg_local.copy()

    # Anti-alias edge erosion: pixels neighboring transparent + within
    # 1.5x tolerance get cleared too. This kills the pink halo Gemini
    # leaves around outlines.
    alpha = np.ones((h, w), dtype=np.float64)
    alpha[background] = 0.0
    for _ in range(2):
        padded = np.pad(alpha, 1, constant_values=1.0)
        has_transparent_neighbor = (
            (padded[:-2, 1:-1] == 0) | (padded[2:, 1:-1] == 0) | (padded[1:-1, :-2] == 0) | (padded[1:-1, 2:] == 0)
        )
        edge_mask = has_transparent_neighbor & (~background) & (dist <= tolerance * 1.5)
        alpha[edge_mask] = 0.0
        background = background | edge_mask

    result = arr.copy()
    result[background, 3] = 0

    if feather_px > 0:
        feather = _feather_alpha(background, feather_px)
        result[:, :, 3] = np.clip(result[:, :, 3].astype(np.float64) * feather, 0, 255).astype(np.uint8)

    return Image.fromarray(result, mode="RGBA")


def _feather_alpha(background: np.ndarray, radius: int) -> np.ndarray:
    """Return float [0,1] alpha that ramps from 0 at ``background`` to 1 ``radius`` pixels inward.

    Iteratively shrinks the opaque region using a 4-neighborhood
    min-filter with a fixed step of ``1 / (radius + 1)``. After
    ``radius`` iterations the boundary of opaque pixels has values
    ``1/(radius+1), 2/(radius+1), …, radius/(radius+1)`` while pixels
    deeper inside stay at ``1.0``.
    """
    alpha = np.where(background, 0.0, 1.0)
    step = 1.0 / (radius + 1)
    for _ in range(radius):
        padded = np.pad(alpha, 1, constant_values=0.0)
        neighbor_min = np.minimum(
            np.minimum(padded[:-2, 1:-1], padded[2:, 1:-1]),
            np.minimum(padded[1:-1, :-2], padded[1:-1, 2:]),
        )
        alpha = np.minimum(alpha, neighbor_min + step)
    return alpha


def _flood_fill_from_edges(is_bg_local: np.ndarray) -> np.ndarray:
    h, w = is_bg_local.shape
    visited = np.zeros((h, w), dtype=bool)
    background = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for y in range(h):
        for x in (0, w - 1):
            if is_bg_local[y, x] and not visited[y, x]:
                visited[y, x] = True
                background[y, x] = True
                queue.append((y, x))
    for x in range(w):
        for y in (0, h - 1):
            if is_bg_local[y, x] and not visited[y, x]:
                visited[y, x] = True
                background[y, x] = True
                queue.append((y, x))

    while queue:
        cy, cx = queue.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and is_bg_local[ny, nx]:
                visited[ny, nx] = True
                background[ny, nx] = True
                queue.append((ny, nx))

    return background


# ---------------------------------------------------------------------------
# Grid-line removal
# ---------------------------------------------------------------------------


def paint_over_grid_lines(
    image: Image.Image,
    *,
    bg_color: tuple[int, int, int] = CHROMA_BG,
    line_search_thickness: int = 8,
    darkness_threshold: int = 80,
) -> Image.Image:
    """Detect and paint over horizontal/vertical grid bands with ``bg_color``.

    Looks for dark horizontal/vertical strips at the expected 2x5 grid
    positions and fills them with the chroma color so the splitter sees
    a continuous magenta canvas instead of dark dividers.
    """
    img = image.convert("RGB")
    arr = np.array(img).astype(np.int16)
    h, w = arr.shape[:2]

    luma = arr.mean(axis=2)

    # Vertical line at x=0.5 ± line_search_thickness
    v_center = w // 2
    v_strip_lo = max(0, v_center - line_search_thickness)
    v_strip_hi = min(w, v_center + line_search_thickness)

    # Horizontal lines at y = h/5, 2h/5, 3h/5, 4h/5
    h_centers = [int(h * (i + 1) / 5) for i in range(4)]

    bg_arr = np.array(bg_color, dtype=np.uint8)
    out = np.array(img)

    # Vertical
    col_luma = luma[:, v_strip_lo:v_strip_hi].mean(axis=0)
    if col_luma.size and col_luma.min() < darkness_threshold:
        offset = int(col_luma.argmin())
        x_dark = v_strip_lo + offset
        # Expand outward while pixels stay dark
        x0 = x_dark
        while x0 > 0 and luma[:, x0 - 1].mean() < darkness_threshold:
            x0 -= 1
        x1 = x_dark
        while x1 < w - 1 and luma[:, x1 + 1].mean() < darkness_threshold:
            x1 += 1
        out[:, x0 : x1 + 1] = bg_arr

    # Horizontal
    for hc in h_centers:
        lo = max(0, hc - line_search_thickness)
        hi = min(h, hc + line_search_thickness)
        row_luma = luma[lo:hi, :].mean(axis=1)
        if row_luma.size and row_luma.min() < darkness_threshold:
            offset = int(row_luma.argmin())
            y_dark = lo + offset
            y0 = y_dark
            while y0 > 0 and luma[y0 - 1, :].mean() < darkness_threshold:
                y0 -= 1
            y1 = y_dark
            while y1 < h - 1 and luma[y1 + 1, :].mean() < darkness_threshold:
                y1 += 1
            out[y0 : y1 + 1, :] = bg_arr

    return Image.fromarray(out, mode="RGB")


# ---------------------------------------------------------------------------
# Cell detection + splitting
# ---------------------------------------------------------------------------


def detect_2x5_cells(
    image: Image.Image,
    *,
    chroma_color: tuple[int, int, int] = CHROMA_BG,
    chroma_tolerance: int = 40,
    row_corridor_threshold: float = 0.985,
) -> list[tuple[int, int, int, int]] | None:
    """Find the 10 cell rectangles by locating horizontal magenta corridors.

    Assumes any grid lines have already been painted over with the
    chroma color (see ``paint_over_grid_lines``); the splitter looks for
    bands of near-pure ``chroma_color`` between rows, not dark lines.

    Algorithm:

    1. Build a per-pixel mask of pixels within ``chroma_tolerance`` of
       ``chroma_color``.
    2. **Horizontal corridors** — contiguous rows whose chroma fraction
       is at least ``row_corridor_threshold``. Stored as
       ``(ymin, ymax)``; these are the gaps between frame rows (plus
       any top/bottom borders).
    3. **Frame rows** — the strips of non-corridor pixels between
       consecutive corridors. Expect exactly 5 for a 2×5 sheet.
    4. **Column split** — fixed at the geometric midline. For an even
       width both cells are exactly ``w // 2`` pixels wide; for an odd
       width the single middle pixel is dropped (left ends at
       ``w // 2``, right starts at ``(w + 1) // 2``) so the cells stay
       equal-sized. Per-row corridor search would let mildly
       misaligned cells drift, while the artist always lays out
       exactly two equal columns.

    Returns ``None`` if 5 frame rows can't be identified.
    """
    img = image.convert("RGB")
    arr = np.array(img).astype(np.int16)
    h, w = arr.shape[:2]

    chroma = np.array(chroma_color, dtype=np.int16)
    is_chroma = np.abs(arr - chroma).max(axis=2) <= chroma_tolerance

    is_corridor_row = is_chroma.mean(axis=1) >= row_corridor_threshold
    corridors = _find_runs(is_corridor_row)
    if not corridors:
        logger.debug("No horizontal magenta corridors detected")
        return None

    frame_rows: list[tuple[int, int]] = []
    prev_end = 0
    for cstart, cend in corridors:
        if cstart > prev_end:
            frame_rows.append((prev_end, cstart))
        prev_end = cend
    if prev_end < h:
        frame_rows.append((prev_end, h))

    min_row_height = max(4, h // 50)
    frame_rows = [r for r in frame_rows if r[1] - r[0] >= min_row_height]

    if len(frame_rows) != 5:
        logger.debug(
            "Detected %d frame rows from %d corridors; expected 5",
            len(frame_rows),
            len(corridors),
        )
        return None

    left_end = w // 2
    right_start = (w + 1) // 2
    cells: list[tuple[int, int, int, int]] = []
    for ystart, yend in frame_rows:
        cells.append((0, ystart, left_end, yend))
        cells.append((right_start, ystart, w, yend))
    return cells


def _find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return ``[(start, end_exclusive), …]`` of contiguous True runs in a 1D bool mask."""
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, v in enumerate(mask):
        if v:
            if not in_run:
                in_run = True
                start = i
        elif in_run:
            in_run = False
            runs.append((start, i))
    if in_run:
        runs.append((start, len(mask)))
    return runs


def split_sprite_sheet_2x5(image: Image.Image, *, inset_px: int = 0) -> list[Image.Image]:
    """Split using fixed 2×5 fractional positions."""
    img = image.convert("RGBA")
    w, h = img.size
    frames: list[Image.Image] = []
    for left_f, top_f, right_f, bottom_f in _FRAME_POSITIONS_2x5:
        left = int(left_f * w) + inset_px
        top = int(top_f * h) + inset_px
        right = int(right_f * w) - inset_px
        bottom = int(bottom_f * h) - inset_px
        frames.append(img.crop((left, top, right, bottom)))
    return frames
