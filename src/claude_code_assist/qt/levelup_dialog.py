"""Modal level-up dialog launched from the tray.

Two-page flow: pick a stat, then a results page that announces the new
level and (when it changed) the rarity transition. The dialog mutates
the companion in place on Confirm — the caller is responsible for
persisting the profile and surfacing the same news in the speech
bubble after the dialog closes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from claude_code_assist.profile.leveling import apply_player_levelup, eligibility_reason

if TYPE_CHECKING:
    from claude_code_assist.models.companion import CompanionProfile
    from claude_code_assist.models.rarity import Rarity


class LevelUpDialog(QDialog):
    """Pick a stat to boost; show results (with rarity change) on confirm.

    After ``exec()`` returns ``QDialog.Accepted`` the caller can read
    :attr:`chosen_stat`, :attr:`old_rarity`, :attr:`new_rarity`, and
    :attr:`new_stat_value` to drive the post-dialog announcement.
    """

    def __init__(self, companion: CompanionProfile, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._companion = companion
        self.chosen_stat: str | None = None
        self.old_rarity: Rarity | None = None
        self.new_rarity: Rarity | None = None
        self.new_stat_value: int | None = None
        self.new_level: int | None = None

        self.setWindowTitle(f"Level up — {companion.name}")
        self.setModal(True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(10)

        self._stack = QStackedWidget(self)
        outer.addWidget(self._stack)

        self._stack.addWidget(self._build_pick_page())
        self._stack.addWidget(self._build_result_page())
        self._stack.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Page 0 — stat picker
    # ------------------------------------------------------------------

    def _build_pick_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        color = self._companion.rarity.color
        header = QLabel()
        header.setTextFormat(Qt.TextFormat.RichText)
        header.setText(
            f'<span style="color:{color}; font-weight:700; font-size:14px;">'
            f"{_html_escape(self._companion.name)}"
            f'</span> <span style="color:#888;">'
            f"· Lv. {self._companion.level} → "
            f'</span><span style="color:#1eff00; font-weight:700;">'
            f"Lv. {self._companion.level + 1}</span>"
        )
        layout.addWidget(header)

        reason_label = QLabel()
        reason_label.setTextFormat(Qt.TextFormat.RichText)
        reason_label.setText(
            f'<span style="color:#aaa; font-size:12px;">{_html_escape(eligibility_reason(self._companion))}</span>'
        )
        layout.addWidget(reason_label)

        prompt = QLabel("Boost which stat by 1?")
        prompt.setStyleSheet("font-weight:600; color:#cccccc; margin-top:6px;")
        layout.addWidget(prompt)

        # One push button per stat — clicking the button is itself the
        # commit action, so there's no separate Confirm/Cancel row at
        # the bottom. Esc / window close (which trigger ``reject()``)
        # remain available as the implicit "back out" path.
        for name, value in self._companion.stats.items():
            button = QPushButton(f"{name}  ·  {value}/100")
            if value >= 100:
                button.setEnabled(False)
                button.setText(f"{name}  ·  100/100  (maxed)")
            button.clicked.connect(lambda _checked=False, stat=name: self._apply_stat(stat))
            layout.addWidget(button)

        return page

    def _apply_stat(self, stat_name: str) -> None:
        old, new = apply_player_levelup(self._companion, stat_name)
        self.chosen_stat = stat_name
        self.old_rarity = old
        self.new_rarity = new
        self.new_stat_value = self._companion.stats[stat_name]
        self.new_level = self._companion.level
        self._populate_result_page()
        self._stack.setCurrentIndex(1)

    # ------------------------------------------------------------------
    # Page 1 — results
    # ------------------------------------------------------------------

    def _build_result_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._result_header = QLabel()
        self._result_header.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self._result_header)

        self._result_body = QLabel()
        self._result_body.setTextFormat(Qt.TextFormat.RichText)
        self._result_body.setWordWrap(True)
        layout.addWidget(self._result_body)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok, parent=page)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
        return page

    def _populate_result_page(self) -> None:
        assert self.chosen_stat is not None
        assert self.old_rarity is not None
        assert self.new_rarity is not None
        assert self.new_stat_value is not None
        assert self.new_level is not None

        new_color = self.new_rarity.color
        self._result_header.setText(
            f'<span style="color:{new_color}; font-weight:700; font-size:14px;">'
            f"Leveled up!"
            f'</span> <span style="color:#888;">· Lv. {self.new_level}</span>'
        )

        lines = [
            f'<span style="color:#1eff00;">+1 {_html_escape(self.chosen_stat)}</span>'
            f' <span style="color:#888;">(now {self.new_stat_value}/100)</span>'
        ]
        if self.new_rarity != self.old_rarity:
            old_color = self.old_rarity.color
            lines.append(
                f'<br><span style="color:{old_color};">{self.old_rarity.value.title()}</span>'
                f' <span style="color:#888;">→</span> '
                f'<span style="color:{new_color}; font-weight:700;">'
                f"{self.new_rarity.value.title()} {self.new_rarity.stars}</span>"
                f'<br><span style="color:#888; font-size:11px;">rarity recomputed from new stats</span>'
            )
        self._result_body.setText("".join(lines))


def _html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
