"""Shiny variant rendering — sweep a holographic band across the sprite.

The shimmer stays inside the sprite's own alpha mask: the companion
window is fixed-size and clips at its edges, so anything painted past
the silhouette would just disappear at the window boundary.

Driven by a single ``phase_seconds`` float so callers can advance time
however they like (the view uses ``time.monotonic()`` minus a
per-window start time).

The pulsing gold halo is currently disabled (commented out below).
Flip ``_HALO_ENABLED`` to ``True`` to bring it back.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPixmap

_SHIMMER_SWEEP_S = 2.5
_SHIMMER_PAUSE_S = 3.0
_PULSE_PERIOD_S = 12.0
_HALO_ENABLED = True


def apply_shiny(pixmap: QPixmap, phase_seconds: float) -> QPixmap:
    """Return a same-size pixmap with the shimmer composited on top."""
    if pixmap.isNull():
        return pixmap

    out = QPixmap(pixmap.size())
    out.setDevicePixelRatio(pixmap.devicePixelRatio())
    out.fill(Qt.GlobalColor.transparent)

    painter = QPainter(out)
    painter.drawPixmap(0, 0, pixmap)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)

    if _HALO_ENABLED:
        pulse = (math.sin(phase_seconds * 2 * math.pi / _PULSE_PERIOD_S) + 1.0) * 0.5
        halo_alpha = int(round(75 * pulse))
        painter.drawPixmap(0, 0, _silhouette(pixmap, QColor(255, 220, 130, halo_alpha)))

    painter.drawPixmap(0, 0, _shimmer(pixmap, phase_seconds))
    painter.end()
    return out


def _silhouette(source: QPixmap, color: QColor) -> QPixmap:
    """Same-size pixmap filled with ``color`` then alpha-clipped to ``source``."""
    layer = QPixmap(source.size())
    layer.setDevicePixelRatio(source.devicePixelRatio())
    layer.fill(Qt.GlobalColor.transparent)
    p = QPainter(layer)
    p.fillRect(layer.rect(), color)
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
    p.drawPixmap(0, 0, source)
    p.end()
    return layer


def _shimmer(source: QPixmap, phase_seconds: float) -> QPixmap:
    """Diagonal rainbow band at ``phase`` position, masked by ``source`` alpha.

    Cycle is sweep-then-pause: the band sweeps once in ``_SHIMMER_SWEEP_S``,
    then nothing is drawn for ``_SHIMMER_PAUSE_S`` before the next sweep.
    """
    layer = QPixmap(source.size())
    layer.setDevicePixelRatio(source.devicePixelRatio())
    layer.fill(Qt.GlobalColor.transparent)

    cycle_pos = phase_seconds % (_SHIMMER_SWEEP_S + _SHIMMER_PAUSE_S)
    if cycle_pos >= _SHIMMER_SWEEP_S:
        return layer

    p = QPainter(layer)
    diag = float(source.width() + source.height())
    t = cycle_pos / _SHIMMER_SWEEP_S
    band_pos = -diag * 0.5 + diag * 1.5 * t
    band_width = diag * 0.55

    grad = QLinearGradient(
        QPointF(band_pos, band_pos),
        QPointF(band_pos + band_width, band_pos + band_width),
    )
    grad.setColorAt(0.00, QColor(255, 90, 200, 0))
    grad.setColorAt(0.20, QColor(255, 90, 200, 90))
    grad.setColorAt(0.40, QColor(120, 200, 255, 120))
    grad.setColorAt(0.55, QColor(180, 255, 140, 130))
    grad.setColorAt(0.75, QColor(255, 220, 100, 100))
    grad.setColorAt(1.00, QColor(255, 220, 100, 0))
    p.fillRect(layer.rect(), grad)

    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
    p.drawPixmap(0, 0, source)
    p.end()
    return layer
