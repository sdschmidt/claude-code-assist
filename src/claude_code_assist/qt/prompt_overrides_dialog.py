"""Modal dialog for editing per-companion prompt overrides.

Six sections (commentary system / event user / idle user / reply user /
memory system / memory user). Each section has a per-kind enabled
checkbox, a multi-line template editor, a *Reset to default* button
that pastes the default template, and a placeholder hint.

Save commits to ``companion.prompt_overrides`` and persists via the
caller-supplied save callback. Cancel discards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from claude_code_assist.commentary.prompts import (
    PLACEHOLDERS_BY_KIND,
    export_default_template,
)
from claude_code_assist.models.companion import PromptOverride

if TYPE_CHECKING:
    from collections.abc import Callable

    from claude_code_assist.models.companion import CompanionProfile

# Order + display names for the six sections. Tuple-of-tuples so the
# dialog renders deterministically regardless of dict iteration order.
_SECTIONS: tuple[tuple[str, str, str], ...] = (
    ("system", "Commentary system prompt", "Persona, role, stats, memory, rules. Shared by event/idle/reply."),
    ("event_user", "Event user prompt", "Sent when a session turn fires (assistant or developer)."),
    ("idle_user", "Idle user prompt", "Sent when the session has been quiet long enough."),
    ("reply_user", "Reply user prompt", "Sent when the developer addresses the companion by name."),
    ("memory_system", "Memory system prompt", "Summariser system prompt — third person, not in-character."),
    ("memory_user", "Memory user prompt", "Folds a rotated-out event into the running session memory."),
)


def _monospace_font() -> QFont:
    """Best-effort cross-platform monospace; falls back to system default."""
    f = QFont()
    f.setStyleHint(QFont.StyleHint.Monospace)
    f.setFamilies(["Menlo", "Monaco", "Consolas", "DejaVu Sans Mono", "monospace"])
    f.setPointSize(11)
    return f


def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


class _PromptSection(QWidget):
    """One kind's editor: header row + textarea + placeholder hint."""

    def __init__(self, kind: str, label: str, subtitle: str, override: PromptOverride) -> None:
        super().__init__()
        self.kind = kind

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(6)

        # Header row: title + enabled checkbox + reset button.
        header = QHBoxLayout()
        header.setSpacing(8)
        title = QLabel(f"<b>{label}</b>")
        header.addWidget(title)
        header.addStretch()
        self.enabled_checkbox = QCheckBox("enabled")
        self.enabled_checkbox.setChecked(override.enabled)
        self.enabled_checkbox.setToolTip(
            "When off the default template is used regardless of what's in the editor below."
        )
        header.addWidget(self.enabled_checkbox)
        reset_btn = QPushButton("Reset to default")
        reset_btn.setToolTip("Replace the editor contents with the default template (placeholders preserved).")
        reset_btn.clicked.connect(self._on_reset)
        header.addWidget(reset_btn)
        layout.addLayout(header)

        # Subtitle: one-line description of when this prompt fires.
        sub = QLabel(f"<span style='color:#888;font-size:11px;'>{subtitle}</span>")
        layout.addWidget(sub)

        # Editor: monospace, scrollable, decent default height.
        self.text_edit = QPlainTextEdit()
        self.text_edit.setFont(_monospace_font())
        self.text_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.text_edit.setMinimumHeight(180)
        # Pre-fill with the user's saved template, falling back to the
        # default so a user opening the dialog for the first time sees a
        # known-good starting point instead of a blank box.
        initial = override.template or export_default_template(kind)
        self.text_edit.setPlainText(initial)
        layout.addWidget(self.text_edit)

        # Placeholder hint — shows only the placeholders this kind
        # actually exposes so users don't get confused trying to use
        # ``{{focal_block}}`` in the idle prompt.
        placeholders = PLACEHOLDERS_BY_KIND.get(kind, ())
        hint_text = "  ".join(f"{{{{{p}}}}}" for p in placeholders)
        hint = QLabel(f"<span style='color:#666;font-size:11px;'>Placeholders: {hint_text}</span>")
        hint.setWordWrap(True)
        hint.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(hint)

    def _on_reset(self) -> None:
        self.text_edit.setPlainText(export_default_template(self.kind))

    def collect(self) -> PromptOverride:
        """Snapshot the editor state into a ``PromptOverride``."""
        return PromptOverride(
            enabled=self.enabled_checkbox.isChecked(),
            template=self.text_edit.toPlainText(),
        )


class PromptOverridesDialog(QDialog):
    """Editor for ``companion.prompt_overrides``.

    Caller is responsible for persisting the change after ``exec()``
    returns ``Accepted`` — typically by calling ``save_profile``.
    """

    def __init__(
        self,
        companion: CompanionProfile,
        on_save: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._companion = companion
        self._on_save = on_save
        self.setWindowTitle(f"Prompts — {companion.name}")
        self.setModal(True)
        self.resize(760, 820)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        intro = QLabel(
            f"<span style='color:#888;font-size:11px;'>"
            f"Customise the prompts {companion.name} uses. Toggle a section off to fall back "
            f"to the default. Use <code>{{{{placeholder}}}}</code> tokens — unknown ones are left "
            f"literal."
            f"</span>"
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        # Scroll area wraps the stack of sections — six textareas don't
        # fit on a small screen.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll, stretch=1)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        self._sections: dict[str, _PromptSection] = {}
        for i, (kind, label, subtitle) in enumerate(_SECTIONS):
            override = getattr(companion.prompt_overrides, kind)
            section = _PromptSection(kind, label, subtitle, override)
            self._sections[kind] = section
            body_layout.addWidget(section)
            if i < len(_SECTIONS) - 1:
                body_layout.addWidget(_hline())

        body_layout.addStretch(1)
        scroll.setWidget(body)

        # Standard button box — Save commits to companion + fires the
        # on_save callback (which writes profile.json). Cancel just
        # closes; nothing has been mutated until accept().
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def accept(self) -> None:
        # Snapshot every section back onto the companion profile.
        for kind, section in self._sections.items():
            setattr(self._companion.prompt_overrides, kind, section.collect())
        if self._on_save is not None:
            self._on_save()
        super().accept()
