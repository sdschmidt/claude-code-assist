"""macOS window-surface enumeration for companion perching.

Returns a list of :class:`Surface` describing on-screen app windows
the companion can land / walk on top of. Excludes a caller-supplied
set of CGWindow IDs (the companion's own sprite + bubble windows).

Returns ``[]`` on non-macOS platforms or if pyobjc's Quartz binding
isn't installed — the rest of the app falls back to the screen floor.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

from PySide6.QtCore import QRect

logger = logging.getLogger(__name__)

# Minimum perchable surface size — matches Swift PetController.
_MIN_PERCH_W = 80
_MIN_PERCH_H = 20


@dataclass(frozen=True)
class Surface:
    """A window the companion can perch on. ``id`` is the CGWindowID."""

    id: int
    rect: QRect


def current_surfaces(*, excluding: frozenset[int] = frozenset()) -> list[Surface]:
    """Return on-screen perchable windows in z-order (frontmost first).

    Filters out: our own windows (``excluding``), windows on non-app
    layers (Dock, menu bar, wallpaper), undersized windows, fully
    transparent windows.
    """
    if sys.platform != "darwin":
        return []
    try:
        # pyobjc's Quartz stubs don't declare these symbols at the
        # type-stub level, but they exist at runtime. Access them via
        # the module so pyright doesn't trip on each name.
        import Quartz  # type: ignore[import-not-found]
    except ImportError:
        return []

    options = (
        Quartz.kCGWindowListOptionOnScreenOnly  # pyright: ignore[reportAttributeAccessIssue]
        | Quartz.kCGWindowListExcludeDesktopElements  # pyright: ignore[reportAttributeAccessIssue]
    )
    try:
        infos = (
            Quartz.CGWindowListCopyWindowInfo(  # pyright: ignore[reportAttributeAccessIssue]
                options,
                Quartz.kCGNullWindowID,  # pyright: ignore[reportAttributeAccessIssue]
            )
            or []
        )
    except Exception:  # noqa: BLE001
        logger.debug("CGWindowListCopyWindowInfo failed", exc_info=True)
        return []

    surfaces: list[Surface] = []
    for info in infos:
        wid = int(info.get("kCGWindowNumber", 0))
        if wid == 0 or wid in excluding:
            continue
        # Layer 0 is the normal app layer; everything else (Dock, menu
        # bar, wallpaper, screensaver shields) we skip.
        if int(info.get("kCGWindowLayer", -1)) != 0:
            continue
        bounds = info.get("kCGWindowBounds")
        if bounds is None:
            continue
        try:
            x = int(bounds["X"])
            y = int(bounds["Y"])
            w = int(bounds["Width"])
            h = int(bounds["Height"])
        except (KeyError, TypeError):
            continue
        if w < _MIN_PERCH_W or h < _MIN_PERCH_H:
            continue
        if float(info.get("kCGWindowAlpha", 1.0)) <= 0.0:
            continue
        surfaces.append(Surface(id=wid, rect=QRect(x, y, w, h)))
    return surfaces


def get_window_number(widget) -> int:  # type: ignore[no-untyped-def]
    """Return the CGWindowID of a Qt widget's NSWindow, or ``0`` on failure.

    Used to keep the companion from perching on its own sprite or
    speech bubble. The NSWindow must already exist (call ``.show()``
    before this).
    """
    if sys.platform != "darwin":
        return 0
    try:
        from AppKit import NSApp  # type: ignore[import-not-found]
    except ImportError:
        return 0

    app = NSApp()
    if app is None:
        return 0
    try:
        ns_view_id = int(widget.winId())
    except Exception:  # noqa: BLE001
        return 0

    for win in app.windows():
        try:
            content_view = win.contentView()
            if content_view is None:
                continue
            if int(content_view.__c_void_p__().value) != ns_view_id:
                continue
            return int(win.windowNumber())
        except Exception:  # noqa: BLE001
            continue
    return 0
