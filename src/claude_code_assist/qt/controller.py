"""Companion controller — 30 Hz state machine for position + animation.

State machine: IDLE → WALKING / SLEEPING / DRAGGING / FALLING / LANDED / REACTING.

Frame indices match the 2x5 sprite grid order::

    0 idle_a   1 idle_b
    2 blink_a  3 blink_b
    4 excited  5 sleep
    6 walk_a   7 walk_b
    8 fall     9 stunned
"""

from __future__ import annotations

import random
from enum import Enum, auto

from PySide6.QtCore import QRect

# Frame indices
F_IDLE_A = 0
F_IDLE_B = 1
F_BLINK_A = 2
F_BLINK_B = 3
F_EXCITED = 4
F_SLEEP = 5
F_WALK_A = 6
F_WALK_B = 7
F_FALL = 8
F_STUNNED = 9

# Physics / behavior
_TICK_HZ = 30
_WALK_SPEED = 1.6
_GRAVITY = 1.6
_STUN_VELOCITY = 36.0
_STUN_FRAMES = int(_TICK_HZ * 1.0)
_AWAKE_FRAMES = int(_TICK_HZ * 60.0)  # 1 min of inactivity before sleep
_REACT_FRAMES = int(_TICK_HZ * 2.0)  # ~2 s of "excited" pose
_BLINK_INTERVAL_MIN = int(_TICK_HZ * 1.5)
_BLINK_INTERVAL_MAX = int(_TICK_HZ * 4.0)
_BLINK_DURATION = int(_TICK_HZ * 0.18)
_IDLE_TO_WALK_MIN = int(_TICK_HZ * 1.0)
_IDLE_TO_WALK_MAX = int(_TICK_HZ * 3.0)
_WALK_DURATION_MIN = int(_TICK_HZ * 3.0)
_WALK_DURATION_MAX = int(_TICK_HZ * 8.0)
_WALK_FRAME_RATE = int(_TICK_HZ / 6)  # toggle walk_a/walk_b ~6 Hz


class _State(Enum):
    IDLE = auto()
    WALKING = auto()
    SLEEPING = auto()
    DRAGGING = auto()
    FALLING = auto()
    LANDED = auto()
    REACTING = auto()


class CompanionController:
    """Owns position + state for the on-screen companion."""

    def __init__(self, *, screen_rect: QRect, sprite_width: int, sprite_height: int) -> None:
        self._screen = screen_rect
        self.sprite_width = max(1, sprite_width)
        self.sprite_height = max(1, sprite_height)

        self.gravity_enabled: bool = True
        self.walking_enabled: bool = True

        # Top-left position; bottom-center spawn near screen bottom-center.
        self._x: float = (screen_rect.left() + screen_rect.right()) / 2 - self.sprite_width / 2
        self._y: float = screen_rect.bottom() - self.sprite_height
        self._vy: float = 0.0
        self._mirrored: bool = False

        self._state: _State = _State.IDLE
        self._state_frames: int = 0
        self._next_state_at: int = random.randint(_IDLE_TO_WALK_MIN, _IDLE_TO_WALK_MAX)
        self._awake_frames: int = _AWAKE_FRAMES
        # Blink timer is tracked separately from ``_state_frames`` —
        # they used to share, which meant every blink reset the walk
        # timer and the companion never started walking.
        self._frames_since_blink: int = 0
        self._next_blink_at: int = random.randint(_BLINK_INTERVAL_MIN, _BLINK_INTERVAL_MAX)
        self._blink_remaining: int = 0
        self._walk_dir: int = 1
        self._walk_anim_counter: int = 0
        # Cursor-to-window-top-left offset, captured at ``begin_drag``
        # and held for the rest of the drag so the sprite doesn't jump
        # under the cursor.
        self._drag_offset_x: float = 0.0
        self._drag_offset_y: float = 0.0

    # ------------------------------------------------------------------
    # Public read-state
    # ------------------------------------------------------------------

    def position(self) -> tuple[int, int]:
        return int(round(self._x)), int(round(self._y))

    def mirrored(self) -> bool:
        return self._mirrored

    @property
    def state_name(self) -> str:
        """Current state name (e.g. ``"IDLE"``, ``"WALKING"``)."""
        return self._state.name

    # ------------------------------------------------------------------
    # External signals
    # ------------------------------------------------------------------

    def set_sprite_dimensions(self, width: int, height: int) -> None:
        """Pin the sprite's bottom-center across rescales."""
        old_w = self.sprite_width
        old_h = self.sprite_height
        center_x = self._x + old_w / 2
        bottom_y = self._y + old_h
        self.sprite_width = max(1, width)
        self.sprite_height = max(1, height)
        self._x = center_x - self.sprite_width / 2
        self._y = bottom_y - self.sprite_height

    def react(self) -> None:
        """Transition to ``REACTING`` for ~2s, stopping any in-flight motion.

        Falls / landings / drags own their own animations and ignore
        the reaction. Walking / idle / sleeping all yield to it: the
        sprite stops, shows the excited frame, then drops back to
        IDLE (which can immediately transition to WALKING again).
        """
        self._awake_frames = _AWAKE_FRAMES
        if self._state in (_State.DRAGGING, _State.FALLING, _State.LANDED, _State.REACTING):
            return
        self._state = _State.REACTING
        self._state_frames = 0

    def begin_drag(self, x: int, y: int) -> None:
        self._state = _State.DRAGGING
        self._state_frames = 0
        # Lock in where the user grabbed the sprite — the cursor stays
        # at the same relative point on the window for the whole drag.
        self._drag_offset_x = x - self._x
        self._drag_offset_y = y - self._y
        self._vy = 0.0
        self._awake_frames = _AWAKE_FRAMES

    def update_drag(self, x: int, y: int) -> None:
        if self._state != _State.DRAGGING:
            return
        self._x = x - self._drag_offset_x
        self._y = y - self._drag_offset_y

    def end_drag(self) -> None:
        if self._state != _State.DRAGGING:
            return
        # Drop with zero momentum — no upward jump from drag velocity.
        self._vy = 0.0
        if self.gravity_enabled:
            self._state = _State.FALLING
            self._state_frames = 0
        else:
            self._state = _State.IDLE
            self._state_frames = 0

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self, screen_rect: QRect) -> int:
        self._screen = screen_rect
        self._state_frames += 1

        if self._state != _State.SLEEPING:
            self._awake_frames = max(0, self._awake_frames - 1)

        # Toggling gravity on while the sprite is parked mid-air (e.g. the
        # user disabled gravity, dragged it up, dropped it, then re-enabled
        # gravity) should drop it. DRAGGING / FALLING / LANDED already own
        # their own physics, so leave them alone.
        if (
            self.gravity_enabled
            and self._state in (_State.IDLE, _State.WALKING, _State.REACTING, _State.SLEEPING)
            and self._y < self._screen.bottom() - self.sprite_height
        ):
            self._state = _State.FALLING
            self._state_frames = 0
            self._vy = 0.0

        if self._state == _State.IDLE:
            return self._tick_idle()
        if self._state == _State.WALKING:
            return self._tick_walking()
        if self._state == _State.SLEEPING:
            return F_SLEEP
        if self._state == _State.DRAGGING:
            return F_EXCITED
        if self._state == _State.FALLING:
            return self._tick_falling()
        if self._state == _State.LANDED:
            return self._tick_landed()
        if self._state == _State.REACTING:
            return self._tick_reacting()
        return F_IDLE_A

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _tick_idle(self) -> int:
        # Sleep when awake window expires
        if self._awake_frames <= 0:
            self._state = _State.SLEEPING
            self._state_frames = 0
            return F_SLEEP

        # Blink
        self._frames_since_blink += 1
        if self._blink_remaining > 0:
            self._blink_remaining -= 1
            return F_BLINK_A if (self._blink_remaining // 4) % 2 == 0 else F_BLINK_B
        if self._frames_since_blink >= self._next_blink_at:
            self._blink_remaining = _BLINK_DURATION
            self._frames_since_blink = 0
            self._next_blink_at = random.randint(_BLINK_INTERVAL_MIN, _BLINK_INTERVAL_MAX)
            return F_BLINK_A

        # Maybe transition to walking
        if self.walking_enabled and self._state_frames >= self._next_state_at:
            self._state = _State.WALKING
            self._state_frames = 0
            self._walk_dir = random.choice([-1, 1])
            self._mirrored = self._walk_dir < 0
            self._walk_anim_counter = 0
            self._next_state_at = random.randint(_WALK_DURATION_MIN, _WALK_DURATION_MAX)
            return F_WALK_A

        # Soft idle bobble
        return F_IDLE_A if (self._state_frames // 15) % 2 == 0 else F_IDLE_B

    def _tick_walking(self) -> int:
        self._x += _WALK_SPEED * self._walk_dir
        left_bound = self._screen.left()
        right_bound = self._screen.right() - self.sprite_width
        if self._x <= left_bound:
            self._x = left_bound
            self._walk_dir = 1
            self._mirrored = False
        elif self._x >= right_bound:
            self._x = right_bound
            self._walk_dir = -1
            self._mirrored = True

        if self._state_frames >= self._next_state_at:
            self._state = _State.IDLE
            self._state_frames = 0
            self._next_state_at = random.randint(_IDLE_TO_WALK_MIN, _IDLE_TO_WALK_MAX)
            return F_IDLE_A

        self._walk_anim_counter += 1
        return F_WALK_A if (self._walk_anim_counter // _WALK_FRAME_RATE) % 2 == 0 else F_WALK_B

    def _tick_falling(self) -> int:
        self._vy += _GRAVITY
        self._y += self._vy
        floor = self._screen.bottom() - self.sprite_height
        if self._y >= floor:
            self._y = floor
            stunned = abs(self._vy) >= _STUN_VELOCITY
            self._vy = 0.0
            self._state_frames = 0
            if stunned:
                self._state = _State.LANDED
                return F_STUNNED
            # Soft landing — there's no dedicated frame between F_FALL
            # and idle, so skip LANDED entirely instead of holding the
            # airborne pose on the ground for ~250 ms.
            self._state = _State.IDLE
            return F_IDLE_A
        return F_FALL

    def _tick_landed(self) -> int:
        if self._state_frames >= _STUN_FRAMES:
            self._state = _State.IDLE
            self._state_frames = 0
            return F_IDLE_A
        return F_STUNNED

    def _tick_reacting(self) -> int:
        if self._state_frames >= _REACT_FRAMES:
            self._state = _State.IDLE
            self._state_frames = 0
            return F_IDLE_A
        return F_EXCITED
