"""Frameless transparent window that hosts the companion sprite."""

from __future__ import annotations

import time
from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent, QPixmap
from PySide6.QtWidgets import QLabel, QWidget

from claude_code_assist.qt.shiny import apply_shiny
from claude_code_assist.qt.sprites import (
    SPRITE_CANVAS,
    SPRITE_RENDER_SCALE,
    scale_frame,
)

MouseHook = Callable[[int, int], None]
DoubleClickHook = Callable[[], None]


class CompanionWindow(QWidget):
    """A frameless top-level window that paints the active sprite frame.

    The window is sized ``(SPRITE_CANVAS × scale × aspect, SPRITE_CANVAS × scale)``.
    The sprite itself is rendered ``SPRITE_RENDER_SCALE × bigger`` than
    the window — overflow is clipped by the hosting QLabel so the
    character fills the canvas without resizing the canvas.
    """

    on_mouse_press: MouseHook | None
    on_mouse_move: MouseHook | None
    on_mouse_release: MouseHook | None
    on_mouse_double_click: DoubleClickHook | None

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self._label = QLabel(self)
        self._label.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # Bottom-aligned: the character sits at the bottom of the
        # canvas regardless of how much SPRITE_RENDER_SCALE overflows
        # the canvas vertically.
        self._label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom)
        self._label.setScaledContents(False)

        self._scale: float = 1.0
        self._aspect: float = 1.0
        self._sprite_height: int = SPRITE_CANVAS
        self._sprite_width: int = SPRITE_CANVAS
        self._current_pixmap: QPixmap | None = None
        self._mirrored: bool = False
        self._dragging_button = False
        self._shiny: bool = False
        self._shiny_phase_start: float = time.monotonic()

        self.on_mouse_press = None
        self.on_mouse_move = None
        self.on_mouse_release = None
        self.on_mouse_double_click = None

        self._apply_dimensions()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_aspect(self, aspect: float) -> None:
        self._aspect = max(0.05, aspect)
        self._apply_dimensions()
        self._refresh_pixmap()

    def set_scale(self, scale: float) -> None:
        self._scale = max(0.1, scale)
        self._apply_dimensions()
        self._refresh_pixmap()

    def set_frame(self, pixmap: QPixmap, *, mirrored: bool = False) -> None:
        self._current_pixmap = pixmap
        self._mirrored = mirrored
        self._refresh_pixmap()

    def enable_shiny(self) -> None:
        self._shiny = True
        self._shiny_phase_start = time.monotonic()
        self._refresh_pixmap()

    def set_position(self, x: int, y: int) -> None:
        self.move(int(x), int(y))

    def sprite_width(self) -> int:
        return self._sprite_width

    def sprite_height(self) -> int:
        return self._sprite_height

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_dimensions(self) -> None:
        self._sprite_height = max(1, int(round(SPRITE_CANVAS * self._scale)))
        self._sprite_width = max(1, int(round(self._sprite_height * self._aspect)))
        # ``setFixedSize`` (rather than ``resize``) — Qt's frameless
        # hint doesn't strip the macOS resize style mask, so without
        # this the user can grab the invisible top corners and resize
        # the sprite window.
        self.setFixedSize(self._sprite_width, self._sprite_height)
        self._label.setGeometry(0, 0, self._sprite_width, self._sprite_height)

    def _refresh_pixmap(self) -> None:
        if self._current_pixmap is None or self._current_pixmap.isNull():
            self._label.clear()
            return
        # Render the sprite ``SPRITE_RENDER_SCALE``× the canvas size.
        # The QLabel clips overflow, so the canvas (window) stays the
        # same logical size while the character visually fills (and
        # extends past) it.
        render_w = max(1, int(round(self._sprite_width * SPRITE_RENDER_SCALE)))
        render_h = max(1, int(round(self._sprite_height * SPRITE_RENDER_SCALE)))
        scaled = scale_frame(self._current_pixmap, render_w, render_h, mirrored=self._mirrored)
        if self._shiny:
            scaled = apply_shiny(scaled, time.monotonic() - self._shiny_phase_start)
        self._label.setPixmap(scaled)

    # ------------------------------------------------------------------
    # Mouse → callbacks
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging_button = True
            if self.on_mouse_press is not None:
                pos = event.globalPosition().toPoint()
                self.on_mouse_press(pos.x(), pos.y())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._dragging_button and self.on_mouse_move is not None:
            pos = event.globalPosition().toPoint()
            self.on_mouse_move(pos.x(), pos.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging_button = False
            if self.on_mouse_release is not None:
                pos = event.globalPosition().toPoint()
                self.on_mouse_release(pos.x(), pos.y())
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self.on_mouse_double_click is not None:
            self.on_mouse_double_click()
        super().mouseDoubleClickEvent(event)
