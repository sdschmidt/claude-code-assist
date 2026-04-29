"""Translucent speech bubble that floats next to the companion."""

from __future__ import annotations

from PySide6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QRect,
    Qt,
    QTimer,
)
from PySide6.QtGui import QColor, QFont, QFontDatabase, QFontMetrics, QPainter, QPainterPath
from PySide6.QtWidgets import QGraphicsOpacityEffect, QLabel, QWidget

_FADE_MS = 180
_AUTO_HIDE_S = 10.0
_PADDING = 10
_RADIUS = 10
_TAIL = 8
_GAP = 6
_MAX_WIDTH = 240
_BASE_FONT_PT = 11


class SpeechBubble(QWidget):
    """A frameless translucent bubble with fade in/out and auto-hide."""

    def __init__(self) -> None:
        super().__init__()
        # See ``view.py`` — Tool windows hide when the (accessory) app
        # isn't active. Use a plain frameless top-level window instead.
        # ``WA_ShowWithoutActivating`` keeps the bubble from stealing
        # focus when it pops up automatically; we deliberately do NOT
        # set ``WindowDoesNotAcceptFocus`` so the user can click the
        # bubble to focus it and copy selected text with ⌘C / Ctrl+C.
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self._text: str = ""
        self._scale: float = 1.0
        # Use the platform's default fixed-width font (Menlo on macOS,
        # DejaVu Sans Mono on Linux, Consolas on Windows). Falling back
        # to ``StyleHint.Monospace`` keeps things sane if the system
        # font lookup fails.
        self._font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        self._font.setStyleHint(QFont.StyleHint.Monospace)
        self._font.setPointSize(_BASE_FONT_PT)
        # ``True`` when the bubble is positioned to the *left* of the
        # companion (right-edge fallback). Drives which side the tail
        # is rendered on so it always points at the sprite.
        self._tail_on_right: bool = False

        # Child label renders the text so it can be selected with the
        # mouse. The parent widget still paints the rounded background
        # + tail behind it.
        self._label = QLabel(self)
        self._label.setFont(self._font)
        self._label.setWordWrap(True)
        self._label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard,
        )
        self._label.setStyleSheet("background: transparent; color: rgba(255, 255, 255, 220);")
        self._label.setCursor(Qt.CursorShape.IBeamCursor)

        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity)

        self._fade = QPropertyAnimation(self._opacity, b"opacity", self)
        self._fade.setDuration(_FADE_MS)
        self._fade.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._start_fade_out)

        self.resize(1, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_scale(self, scale: float) -> None:
        """Scale the bubble's font + max width to track the companion size."""
        self._scale = max(0.1, scale)
        self._font.setPointSize(max(1, int(round(_BASE_FONT_PT * self._scale))))
        self._label.setFont(self._font)
        if self._text:
            self._size_to_text()
            self.update()

    def show_comment(self, text: str, duration_s: float = _AUTO_HIDE_S) -> None:
        """Show ``text`` in the bubble; fade out after ``duration_s``."""
        self._text = text
        self._size_to_text()
        self.update()
        if not self.isVisible():
            self.show()
        self._fade.stop()
        self._fade.setStartValue(self._opacity.opacity())
        self._fade.setEndValue(1.0)
        self._fade.start()
        self._hide_timer.start(int(duration_s * 1000))

    def reposition(self, companion_rect: QRect, screen_rect: QRect) -> None:
        """Anchor next to the companion; flip side + tail when out of room."""
        if not self._text:
            return
        # Default: bubble on the right of the companion, tail points left.
        x = companion_rect.right() + _GAP
        y = companion_rect.top() - self.height() // 2
        tail_on_right = False
        # Flip to the left if we'd run off the right edge of the screen.
        if x + self.width() > screen_rect.right():
            x = companion_rect.left() - _GAP - self.width()
            tail_on_right = True
        x = max(screen_rect.left(), min(x, screen_rect.right() - self.width()))
        y = max(screen_rect.top(), min(y, screen_rect.bottom() - self.height()))
        if tail_on_right != self._tail_on_right:
            self._tail_on_right = tail_on_right
            self._position_label()
            self.update()
        self.move(x, y)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _start_fade_out(self) -> None:
        self._fade.stop()
        self._fade.setStartValue(self._opacity.opacity())
        self._fade.setEndValue(0.0)
        self._fade.finished.connect(self._after_fade_out)
        self._fade.start()

    def _after_fade_out(self) -> None:
        try:
            self._fade.finished.disconnect(self._after_fade_out)
        except (TypeError, RuntimeError):
            pass
        if self._opacity.opacity() <= 0.01:
            self.hide()

    def _size_to_text(self) -> None:
        metrics = QFontMetrics(self._font)
        max_width = max(60, int(round(_MAX_WIDTH * self._scale)))
        rect = metrics.boundingRect(
            QRect(0, 0, max_width, 10_000),
            Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
            self._text,
        )
        w = rect.width() + _PADDING * 2 + _TAIL
        h = rect.height() + _PADDING * 2
        self.resize(max(48, w), max(28, h))
        self._label.setText(self._text)
        self._position_label()

    def _body_rect(self) -> QRect:
        """Bubble body minus the side reserved for the tail."""
        if self._tail_on_right:
            return self.rect().adjusted(0, 0, -_TAIL, 0)
        return self.rect().adjusted(_TAIL, 0, 0, 0)

    def _position_label(self) -> None:
        text_rect = self._body_rect().adjusted(_PADDING, _PADDING, -_PADDING, -_PADDING)
        self._label.setGeometry(text_rect)

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt API)
        if not self._text:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Reserve ``_TAIL`` px on whichever side the tail lives. The
        # bubble body + tail together always occupy the full widget
        # rect, so swapping sides only changes which margin is used.
        body_rect = self._body_rect()

        path = QPainterPath()
        path.addRoundedRect(body_rect, _RADIUS, _RADIUS)
        tail_y = body_rect.height() // 2
        if self._tail_on_right:
            # Tail points right toward the companion (companion is to
            # the right of the bubble).
            path.moveTo(body_rect.right(), tail_y - 5)
            path.lineTo(body_rect.right() + _TAIL, tail_y)
            path.lineTo(body_rect.right(), tail_y + 5)
        else:
            path.moveTo(body_rect.left(), tail_y - 5)
            path.lineTo(body_rect.left() - _TAIL, tail_y)
            path.lineTo(body_rect.left(), tail_y + 5)
        path.closeSubpath()

        painter.fillPath(path, QColor(20, 20, 22, 230))
