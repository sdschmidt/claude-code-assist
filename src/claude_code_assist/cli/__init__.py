"""``companion`` CLI dispatcher.

The top-level ``companion`` script peeks at ``sys.argv[1]``:

* known subcommand (``start`` / ``new`` / ``art`` / ``roster`` /
  ``settings`` / ``levelup``) — dispatch to the matching module;
* no args — open the interactive menu (``cli/menu.py``);
* anything else (e.g. a bare flag like ``--debug``) — fall through to
  the Qt companion entry point with the original argv, preserving the
  legacy "just run the companion" path.

We don't use a real subparser here on purpose — the companion entry
point has its own argparse and adopting subparsers globally would force
every flag to be redeclared at the top level.
"""

from __future__ import annotations

import sys

_SUBCOMMANDS = ("start", "new", "art", "roster", "settings", "levelup")


def _print_top_level_help() -> None:
    print(
        "companion — Claude Code desktop companion\n"
        "\n"
        "Usage:\n"
        "  companion              Open the interactive menu.\n"
        "  companion start        Run the companion (exits to the shell).\n"
        "  companion new          Generate a new companion (archives the existing one).\n"
        "  companion art          Generate sprite art for the current companion.\n"
        "  companion roster       List + switch between archived companions.\n"
        "  companion settings     Edit gravity / walking / scale.\n"
        "  companion levelup      Force a level-up + stat boost (debug; skips eligibility).\n"
        "\n"
        "Run 'companion <subcommand> --help' for subcommand-specific options.\n"
    )


def main(argv: list[str] | None = None) -> int:
    raw = sys.argv[1:] if argv is None else argv

    if raw and raw[0] in _SUBCOMMANDS:
        cmd, sub_argv = raw[0], raw[1:]
        if cmd == "start":
            from claude_code_assist.qt.app import main as run_companion

            return run_companion(sub_argv)
        if cmd == "new":
            from claude_code_assist.cli.new import run as run_new

            return run_new(sub_argv)
        if cmd == "art":
            from claude_code_assist.cli.art import run as run_art

            return run_art(sub_argv)
        if cmd == "roster":
            from claude_code_assist.cli.roster import run as run_roster

            return run_roster(sub_argv)
        if cmd == "settings":
            from claude_code_assist.cli.settings import run as run_settings

            return run_settings(sub_argv)
        if cmd == "levelup":
            from claude_code_assist.cli.levelup import run as run_levelup

            return run_levelup(sub_argv)

    if raw and raw[0] in ("help", "--commands"):
        _print_top_level_help()
        return 0

    if not raw:
        from claude_code_assist.cli.menu import run as run_menu

        return run_menu([])

    from claude_code_assist.qt.app import main as run_companion

    return run_companion(raw)


if __name__ == "__main__":
    raise SystemExit(main())
