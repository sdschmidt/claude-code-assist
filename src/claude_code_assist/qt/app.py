"""Entry point for the Qt desktop companion.

Run with::

    python -m claude_code_assist.qt --companion-name <name>

Wires together a 30 Hz timer, the ``CompanionController`` state machine, the
``CompanionWindow``, the speech ``SpeechBubble``, the in-process commentary
backend, and the full tray menu (header, info rows, sprite preview,
stats bars, bio/backstory, gravity/walk toggles, Open Config Folder,
Quit). Window perching lands in M5.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Tick rate constant — matches Swift ``CompanionController`` (30 Hz).
_TICK_HZ = 30
_TICK_INTERVAL_MS = round(1000 / _TICK_HZ)


def _install_stdin_quit(app) -> None:  # type: ignore[no-untyped-def]
    """Wire ``q`` (in the launching TTY) to ``app.quit()``.

    Switches stdin to cbreak so single keystrokes arrive without echo;
    a ``QSocketNotifier`` polls the fd from inside Qt's event loop. The
    original termios state is restored on app exit.
    """
    if not sys.stdin.isatty():
        return
    try:
        import termios  # noqa: PLC0415
        import tty  # noqa: PLC0415

        from PySide6.QtCore import QSocketNotifier  # noqa: PLC0415
    except ImportError:
        return

    fd = sys.stdin.fileno()
    try:
        original = termios.tcgetattr(fd)
    except termios.error:
        return
    try:
        tty.setcbreak(fd)
    except termios.error:
        return

    def _restore() -> None:
        import contextlib  # noqa: PLC0415

        with contextlib.suppress(termios.error):
            termios.tcsetattr(fd, termios.TCSADRAIN, original)

    notifier = QSocketNotifier(fd, QSocketNotifier.Type.Read, app)

    def _on_ready(_socket: int) -> None:
        try:
            ch = sys.stdin.read(1)
        except (OSError, ValueError):
            return
        if ch and ch.lower() == "q":
            app.quit()

    notifier.activated.connect(_on_ready)
    app.aboutToQuit.connect(_restore)


def _build_status_panel(companion, comment: str, state: str, *, console_width: int):  # type: ignore[no-untyped-def]
    """Build the rounded status box rendered via ``rich.live``.

    Title border: ``Name ★… · Rarity · Lv. N · creature_type · Role``,
    using the rarity color for the name/stars/rarity-name and the role
    catalog's color for the role suffix (matching ``companion new`` /
    ``companion roster``). Body: the most recent commentary. Subtitle
    (bottom border): ``XP <bar>`` left-anchored next to the controller
    state — both packed into a single Rich ``Text`` because Panel
    only supports one subtitle slot. Padding is computed against
    ``console_width`` so the state ends up flush against the right
    corner regardless of terminal width.
    """
    from rich.panel import Panel  # noqa: PLC0415
    from rich.text import Text  # noqa: PLC0415

    from claude_code_assist.models.role import ROLE_CATALOG  # noqa: PLC0415
    from claude_code_assist.profile.leveling import format_xp_bar_segments  # noqa: PLC0415

    color = companion.rarity.color

    title = Text()
    title.append(companion.name, style=f"bold {color}")
    title.append(f"  {companion.rarity.stars}", style=color)
    title.append(f"  ·  {companion.rarity.value.title()}", style=color)
    title.append(f"  ·  Lv. {companion.level}", style="dim")
    title.append(f"  ·  {companion.creature_type}", style="dim")
    if companion.role is not None:
        defn = ROLE_CATALOG.get(companion.role)
        role_color = defn.color if defn else "#888"
        title.append("  ·  ", style="dim")
        title.append(companion.role.value, style=role_color)

    body = Text(comment) if comment else Text("(no commentary yet)", style="dim italic")

    xp_filled, xp_empty = format_xp_bar_segments(companion.comment_counter)
    xp_segment = Text()
    xp_segment.append(" XP ", style="dim")
    xp_segment.append(xp_filled, style=color)
    xp_segment.append(xp_empty, style="dim")

    state_segment = Text(f" {state} ", style="bold dim")

    # Rich's Panel reserves the two corner glyphs and a single space on
    # each side of the subtitle, so the usable inner width is
    # ``console_width - 4``. Anything left over after the two segments
    # becomes a run of ``─`` characters (one per cell) in the gap, which
    # matches the look of the existing border.
    used = xp_segment.cell_len + state_segment.cell_len
    fill = max(2, console_width - 4 - used)
    subtitle = xp_segment + Text("─" * fill, style="dim") + state_segment

    return Panel(
        body,
        title=title,
        subtitle=subtitle,
        subtitle_align="left",
        border_style=color,
        padding=(1, 2),
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI args. New flags get added here as milestones land."""
    import os

    from claude_code_assist.paths import default_config_dir  # noqa: PLC0415

    cfg = default_config_dir()
    parser = argparse.ArgumentParser(prog="companion", description=__doc__)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=cfg,
        help="config directory. Defaults to $XDG_CONFIG_HOME/claude-companion.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(os.getcwd()),
        help="Project directory whose Claude Code sessions to watch. Defaults to cwd.",
    )
    parser.add_argument(
        "--watch-dir",
        type=Path,
        default=None,
        help="Override Claude session dir. Defaults to ~/.claude/projects/<encoded-project>.",
    )
    parser.add_argument(
        "--follow",
        type=Path,
        default=None,
        help="Follow this plain text file instead of Claude sessions.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging to stderr.")
    return parser.parse_args(argv)


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Return ``(art_dir, profile_path)``, falling back to config-dir defaults.

    Defaults follow the new layout: ``<config>/companion/profile.json`` and
    ``<config>/companion/art/`` (see ``profile.storage`` for the full tree).
    """
    from claude_code_assist.profile.storage import companion_art_dir, get_profile_path

    art_dir = companion_art_dir(args.config_dir)
    profile_path = get_profile_path(args.config_dir)
    return art_dir, profile_path


def _resolve_session_label(
    args: argparse.Namespace,
    encode_project_path,  # type: ignore[no-untyped-def]
    find_newest_session,  # type: ignore[no-untyped-def]
) -> str:
    """Best-effort 'where am I watching?' string for the tray info row.

    Prefers the explicit ``--follow`` file, then ``--watch-dir`` plus the
    most recent ``.jsonl`` session inside it, then the auto-derived
    Claude Code session dir. Returns ``"—"`` if nothing is set up yet.
    """
    if args.follow is not None:
        return str(args.follow)
    session_dir = args.watch_dir or (Path.home() / ".claude" / "projects" / encode_project_path(str(args.project)))
    if session_dir.is_dir():
        newest = find_newest_session(session_dir)
        if newest is not None:
            return f"{newest.name}"
    return str(session_dir)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    # Default level is WARNING so a normal launch is silent (the user
    # said the INFO chatter should be gated behind ``--debug``). With
    # the flag we drop to DEBUG so both DEBUG and INFO records show, and
    # we mirror the stream to ``<config>/debug.log`` for after-the-fact
    # inspection.
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.debug:
        try:
            args.config_dir.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(args.config_dir / "debug.log"))
        except OSError:
            sys.stderr.write(f"Could not open debug.log under {args.config_dir}\n")
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )

    # One-shot migration of legacy <config>/profile.yaml / <config>/art /
    # <config>/pet/ / config.yaml / *_settings.json into the current
    # <config>/companion/ + config.json layout. Idempotent.
    from claude_code_assist.profile.storage import migrate_legacy_layout

    migrate_legacy_layout(args.config_dir)

    # Per-companion JSONL transcript of every LLM call (system +
    # user prompt + raw response). Enabled when either ``--debug`` is
    # passed (the historical knob) or the ``commentary_prompt_log``
    # setting is on (so users can opt in without restarting under
    # debug); production runs without either flip stay silent.
    from claude_code_assist.qt.settings import SettingsStore  # noqa: PLC0415

    transcript_enabled = args.debug or SettingsStore(args.config_dir).load().commentary_prompt_log
    if transcript_enabled:
        from claude_code_assist.commentary import transcript  # noqa: PLC0415
        from claude_code_assist.profile.storage import get_profile_path  # noqa: PLC0415

        transcript.enable(get_profile_path(args.config_dir).parent / "prompts.jsonl")

    # Lazy imports so ``import claude_code_assist.qt`` works even when PySide6 is not
    # installed (e.g. someone using only the terminal renderers).
    try:
        from PySide6.QtCore import QRect, QTimer
        from PySide6.QtGui import QCursor
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        sys.stderr.write(f"PySide6 is required. Install with:\n    uv sync\n(import error: {exc})\n")
        return 1

    from claude_code_assist.config import load_config
    from claude_code_assist.monitor.watcher import encode_project_path, find_newest_session
    from claude_code_assist.profile.storage import load_profile, save_profile
    from claude_code_assist.qt.backend import CommentaryBackend
    from claude_code_assist.qt.bubble import SpeechBubble
    from claude_code_assist.qt.controller import CompanionController
    from claude_code_assist.qt.macos_polish import promote_window_level, set_accessory_activation_policy
    from claude_code_assist.qt.settings import SettingsStore
    from claude_code_assist.qt.sprites import SPRITE_CANVAS, Frame, load_frames
    from claude_code_assist.qt.tray import install_tray
    from claude_code_assist.qt.view import CompanionWindow

    art_dir, profile_path = _resolve_paths(args)

    # Load the companion config (provider + cooldowns + budgets). Falling back
    # to defaults is fine — the LLM provider stack will surface any missing
    # API keys when commentary submission tries to run.
    config_path = args.config_dir / "config.json"
    if config_path.is_file():
        config = load_config(config_path).model_copy(update={"config_dir": args.config_dir})
    else:
        from claude_code_assist.config import CompanionConfig

        config = CompanionConfig(config_dir=args.config_dir)

    companion = load_profile(profile_path)
    if companion is None:
        sys.stderr.write(f"No companion profile found at {profile_path}. Run `companion new` to create one.\n")
        return 1

    # Refuse to start without a complete sprite set — running with empty
    # art produces an invisible companion (the user reported this).
    # ``companion art`` is the canonical fix.
    missing_frames = [i for i in range(10) if not (art_dir / f"frame_{i}.png").is_file()]
    if missing_frames:
        sys.stderr.write(
            f"Companion art is missing in {art_dir} "
            f"(missing {len(missing_frames)}/10 frames). "
            f"Run `companion art` to generate or prefill the sprite set.\n"
        )
        return 1

    # Seed ``last_seen_date`` on a freshly created companion so the
    # eligibility check has a baseline to compare against. Level-ups
    # themselves are no longer prompted at startup — the tray surfaces
    # them via a yellow "Level up available!" entry + halo on the icon
    # and routes the player through ``LevelUpDialog`` when they choose.
    # ``companion levelup`` (questionary) is still available as a
    # debug-only force path.
    from claude_code_assist.profile.leveling import seed_last_seen_date  # noqa: PLC0415

    if seed_last_seen_date(companion):
        save_profile(companion, profile_path)

    # QApplication must exist before any QPixmap is constructed.
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    # macOS only: hide the Dock icon and stop the app from grabbing the
    # menu bar. No-op on Linux / when pyobjc isn't installed.
    set_accessory_activation_policy()

    # Use the screen the cursor is on so multi-monitor setups land sensibly.
    screen = app.screenAt(QCursor.pos()) or app.primaryScreen()
    if screen is None:
        sys.stderr.write("No screen detected.\n")
        return 1
    screen_rect = screen.availableGeometry()

    # High-DPI rendering: give the sprite loader the screen's device pixel
    # ratio so it can produce pixmaps with full Retina resolution. Without
    # this the companion looks soft on a 2× display.
    dpr = screen.devicePixelRatio()
    logger.debug("Using device pixel ratio %s (screen %s)", dpr, screen.name())

    try:
        frames, sprite_aspect = load_frames(art_dir, device_pixel_ratio=dpr)
    except FileNotFoundError as exc:
        sys.stderr.write(f"Error loading sprites: {exc}\n")
        return 1

    # Backend: session watcher + LLM call pipeline. In-process replacement
    # for the Swift CommentBus IPC. Started below after Qt is set up.
    backend = CommentaryBackend(
        config=config,
        companion=companion,
        project_path=str(args.project),
        watch_dir=args.watch_dir,
        follow_file=args.follow,
    )

    # Load persisted toggle state before constructing the controller so
    # the first physics tick respects the user's previous choice.
    settings_store = SettingsStore(args.config_dir)
    companion_settings = settings_store.load()

    initial_h = max(1, int(round(SPRITE_CANVAS * companion_settings.companion_scale)))
    initial_w = max(1, int(round(initial_h * sprite_aspect)))
    controller = CompanionController(
        screen_rect=screen_rect,
        sprite_width=initial_w,
        sprite_height=initial_h,
    )
    controller.gravity_enabled = companion_settings.gravity_enabled
    controller.walking_enabled = companion_settings.walking_enabled
    window = CompanionWindow()
    window.set_aspect(sprite_aspect)
    window.set_scale(companion_settings.companion_scale)
    window.set_frame(frames[Frame.IDLE_A])
    px, py = controller.position()
    window.set_position(px, py)
    window.show()
    # Must run *after* show() so the underlying NSWindow exists.
    promote_window_level(window)
    # Re-promote whenever the app's activation state changes — Qt
    # occasionally re-applies window properties on focus events and we
    # need to overwrite those changes immediately.
    app.applicationStateChanged.connect(lambda _state: promote_window_level(window))

    # Speech bubble — frameless translucent companion window. Force-create
    # the underlying NSWindow with a no-op show/hide cycle so we can
    # promote its level once; future show_comment() calls reuse the same
    # NSWindow and inherit the promoted state.
    bubble = SpeechBubble()
    bubble.set_scale(companion_settings.companion_scale)
    bubble.show()
    promote_window_level(bubble)
    bubble.hide()
    app.applicationStateChanged.connect(lambda _state: promote_window_level(bubble))

    # Tray keeps a strong ref via ``parent=app``; we keep one too so the icon
    # doesn't disappear on garbage-collection.
    def _react_now() -> None:
        # Ask the backend for a fresh comment; if it can't (budget /
        # in-flight) show a one-line bubble so the click feels responsive.
        if not backend.request_comment_now():
            bubble.show_comment("…thinking already, hang on.", duration_s=2.0)

    def _set_gravity(enabled: bool) -> None:
        controller.gravity_enabled = enabled

    def _set_walking(enabled: bool) -> None:
        controller.walking_enabled = enabled

    def _set_scale(scale: float) -> None:
        sprite_h = max(1, int(round(SPRITE_CANVAS * scale)))
        sprite_w = max(1, int(round(sprite_h * sprite_aspect)))
        controller.set_sprite_dimensions(sprite_w, sprite_h)
        window.set_scale(scale)
        bubble.set_scale(scale)

    def _on_levelup_requested() -> None:
        from claude_code_assist.qt.levelup_dialog import LevelUpDialog  # noqa: PLC0415

        dialog = LevelUpDialog(companion)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        save_profile(companion, profile_path)
        tray.refresh_levelup()  # type: ignore[attr-defined]

        # Speech bubble: same news the dialog showed, surfaced next to
        # the sprite so it lingers after the modal closes.
        if dialog.chosen_stat is not None and dialog.new_stat_value is not None and dialog.new_level is not None:
            parts = [f"+1 {dialog.chosen_stat} (now {dialog.new_stat_value}/100) — Lv. {dialog.new_level}"]
            if (
                dialog.new_rarity is not None
                and dialog.old_rarity is not None
                and dialog.new_rarity != dialog.old_rarity
            ):
                parts.append(
                    f"{dialog.old_rarity.value.title()} → {dialog.new_rarity.value.title()} {dialog.new_rarity.stars}"
                )
            controller.react()
            bubble.show_comment(" · ".join(parts), duration_s=8.0)

    session_label = _resolve_session_label(args, encode_project_path, find_newest_session)
    tray = install_tray(  # noqa: F841 — strong ref for lifetime
        app,
        frames[Frame.IDLE_A],
        companion=companion,
        config_dir=args.config_dir,
        art_dir=art_dir,
        settings=companion_settings,
        settings_store=settings_store,
        session_label=session_label,
        cwd_label=str(args.project),
        cwd_path=args.project,
        on_quit=app.quit,
        on_react_now=_react_now,
        on_gravity_toggled=_set_gravity,
        on_walking_toggled=_set_walking,
        on_scale_changed=_set_scale,
        on_levelup_requested=_on_levelup_requested,
    )

    # Mouse → controller. The controller owns position; the view just relays
    # global coordinates and reads them back next tick via ``set_position``.
    window.on_mouse_press = controller.begin_drag
    window.on_mouse_move = controller.update_drag
    window.on_mouse_release = lambda _x, _y: controller.end_drag()

    # Double-click cycles through four behaviors:
    #   0 → start a react (fresh comment, same path as the tray button)
    #   1 → read the last message back
    #   2 → "…thinking please hang on"
    #   3 → read the last message back
    # Phase wraps to 0 after 3.
    double_click_phase = 0

    def _on_double_click() -> None:
        nonlocal double_click_phase
        phase = double_click_phase
        double_click_phase = (phase + 1) % 4

        if phase == 0:
            controller.react()
            _react_now()
            return

        if phase == 2:
            controller.react()
            bubble.show_comment("...", duration_s=2.5)
            return

        # phases 1 and 3 — replay the last comment if we have one.
        if companion.last_comment:
            controller.react()
            bubble.show_comment(companion.last_comment)
        else:
            # No history yet: treat like a regular react so the user
            # still gets feedback on the click.
            controller.react()
            _react_now()

    window.on_mouse_double_click = _on_double_click

    # Persistent terminal status panel. Re-renders only when the
    # commentary or controller state changes so the redraw cost stays
    # tiny even though we're called at 30 Hz.
    from rich.console import Console  # noqa: PLC0415
    from rich.live import Live  # noqa: PLC0415

    panel_state = {
        "comment": companion.last_comment or "",
        "state": controller.state_name,
        "xp": companion.comment_counter,
    }
    panel_console = Console()
    live = Live(
        _build_status_panel(
            companion,
            panel_state["comment"],
            panel_state["state"],
            console_width=panel_console.width,
        ),
        console=panel_console,
        auto_refresh=False,
        transient=False,
    )
    live.start(refresh=True)

    def on_tick() -> None:
        # Re-query each tick so the companion follows window moves between screens.
        active = app.screenAt(QCursor.pos()) or app.primaryScreen()
        rect = active.availableGeometry() if active is not None else screen_rect

        # Pause autonomous idle chatter while sleeping — only real
        # session events should rouse the companion.
        backend.set_idle_chatter_enabled(controller.state_name != "SLEEPING")

        # Drain a session event + harvest pending LLM futures.
        update = backend.poll()
        if update.had_event:
            controller.react()
        if update.new_comment:
            bubble.show_comment(update.new_comment)
            controller.react()

        idx = controller.tick(rect)
        window.set_frame(frames[idx], mirrored=controller.mirrored())
        x, y = controller.position()
        window.set_position(x, y)
        bubble.reposition(QRect(x, y, controller.sprite_width, controller.sprite_height), rect)

        # Repaint the status panel only on commentary / state / XP transitions.
        needs_redraw = False
        if update.new_comment and update.new_comment != panel_state["comment"]:
            panel_state["comment"] = update.new_comment
            needs_redraw = True
        current_state = controller.state_name
        if current_state != panel_state["state"]:
            panel_state["state"] = current_state
            needs_redraw = True
        if companion.comment_counter != panel_state["xp"]:
            panel_state["xp"] = companion.comment_counter
            needs_redraw = True
        if needs_redraw:
            live.update(
                _build_status_panel(
                    companion,
                    panel_state["comment"],
                    panel_state["state"],
                    console_width=panel_console.width,
                ),
                refresh=True,
            )

    timer = QTimer()
    timer.setInterval(_TICK_INTERVAL_MS)
    timer.timeout.connect(on_tick)
    timer.start()

    # Make Ctrl-C from the terminal kill the app cleanly. Without this Qt's
    # event loop swallows SIGINT and the only way out is the tray menu.
    # First ``SIGINT`` triggers a graceful Qt quit; a second one
    # hard-exits in case the in-flight Claude Agent SDK call wedges
    # interpreter shutdown.
    sigint_count = {"n": 0}

    def _on_sigint(*_args: object) -> None:
        sigint_count["n"] += 1
        if sigint_count["n"] == 1:
            app.quit()
        else:
            import os as _os  # noqa: PLC0415

            _os._exit(130)

    signal.signal(signal.SIGINT, _on_sigint)
    # A no-op timer keeps Python's signal handlers running while Qt is in
    # native code (Qt blocks Python signal delivery during ``exec()``).
    sigint_pump = QTimer()
    sigint_pump.start(100)
    sigint_pump.timeout.connect(lambda: None)

    # Press ``q`` in the launching terminal to quit. Only enabled if stdin
    # is a TTY (so piping or daemonising won't trip on the cbreak switch).
    # ``cbreak`` disables ICANON + ECHO so single keys arrive immediately
    # and the user doesn't see typed characters echoed onto the Live panel.
    _install_stdin_quit(app)

    # Persist the rolling comment_history when the app quits so the next
    # launch carries on with the same context the LLM saw, and stop the
    # Live status panel so the terminal isn't left mid-render.
    app.aboutToQuit.connect(
        lambda: (backend.stop(), save_profile(companion, profile_path), live.stop()),
    )

    backend.start()
    rc = app.exec()
    # Hard-exit instead of letting the interpreter unwind: the
    # commentary executor's worker thread can be mid-Claude-Agent-SDK
    # async-generator call, which makes ``ThreadPoolExecutor.shutdown``
    # block at interpreter shutdown for minutes.
    import os as _os  # noqa: PLC0415

    _os._exit(rc)


if __name__ == "__main__":
    raise SystemExit(main())
