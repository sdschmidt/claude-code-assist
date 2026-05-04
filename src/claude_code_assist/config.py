"""Configuration system for the desktop companion."""

from __future__ import annotations

import json
import logging
import os
import time
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from claude_code_assist.models.rarity import DEFAULT_RARITY_WEIGHTS, Rarity
from claude_code_assist.models.stats import StatConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM Provider system
# ---------------------------------------------------------------------------


class LLMProvider(StrEnum):
    """Supported LLM providers."""

    CLAUDE = "claude"
    OLLAMA = "ollama"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    GEMINI = "gemini"


# Claude model aliases → full Agent SDK model IDs
_CLAUDE_MODEL_ALIASES: dict[str, str] = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6-20250514",
    "opus": "claude-opus-4-6-20250514",
}

# Curated default model lists per provider — surfaced in the CLI picker.
# First entry is treated as the provider's default if the user hasn't picked one.
PROVIDER_MODEL_DEFAULTS: dict[LLMProvider, tuple[str, ...]] = {
    LLMProvider.CLAUDE: ("haiku", "sonnet", "opus"),
    LLMProvider.OPENAI: ("gpt-5", "gpt-4o", "gpt-4o-mini"),
    LLMProvider.OPENROUTER: (
        "anthropic/claude-haiku-4-5",
        "anthropic/claude-sonnet-4-6",
        "openai/gpt-4o",
    ),
    LLMProvider.GEMINI: ("gemini-2.5-flash", "gemini-2.5-pro"),
    LLMProvider.OLLAMA: ("llama3.2", "qwen2.5", "mistral"),
}

# Per-provider connection defaults (base_url + which env var holds the API key).
_PROVIDER_CONNECTION_DEFAULTS: dict[LLMProvider, dict[str, str]] = {
    LLMProvider.CLAUDE: {"base_url": "", "api_key_env": ""},
    LLMProvider.OLLAMA: {"base_url": "http://localhost:11434/v1", "api_key_env": ""},
    LLMProvider.OPENAI: {"base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY"},
    LLMProvider.OPENROUTER: {"base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY"},
    LLMProvider.GEMINI: {"base_url": "", "api_key_env": "GEMINI_API_KEY"},
}


def default_model_for(provider: LLMProvider) -> str:
    """First curated model for a provider — used when the user hasn't picked one."""
    return PROVIDER_MODEL_DEFAULTS[provider][0]


def resolve_api_key(provider: LLMProvider, api_key_env: str) -> str | None:
    """Resolve the API key for a provider.

    Args:
        provider: The LLM provider.
        api_key_env: Environment variable name containing the key.

    Returns:
        API key string, or None if not applicable (e.g. Claude Agent SDK).
    """
    if provider == LLMProvider.CLAUDE:
        return None  # Agent SDK handles auth internally
    if provider == LLMProvider.OLLAMA:
        return "ollama"  # Ollama accepts any non-empty string
    if api_key_env:
        return os.environ.get(api_key_env)
    return None


class ProviderConfig(BaseModel):
    """User-facing LLM provider selection.

    Stores only the provider + model the user picked. Connection
    details (base_url, api_key_env) are derived per-provider on
    resolve, so switching providers doesn't require re-typing them.
    """

    provider: LLMProvider = LLMProvider.CLAUDE
    model: str = ""

    def resolve(self) -> ResolvedProviderConfig:
        """Fill in defaults: provider's first curated model if model is empty,
        plus connection details (base_url, api_key_env)."""
        connection = _PROVIDER_CONNECTION_DEFAULTS[self.provider]
        resolved_model = self.model or default_model_for(self.provider)

        # Resolve Claude model aliases (haiku/sonnet/opus → full IDs)
        if self.provider == LLMProvider.CLAUDE:
            resolved_model = _CLAUDE_MODEL_ALIASES.get(resolved_model.lower().strip(), resolved_model)

        return ResolvedProviderConfig(
            provider=self.provider,
            model=resolved_model,
            base_url=connection["base_url"],
            api_key_env=connection["api_key_env"],
        )


class ResolvedProviderConfig(BaseModel):
    """Fully-resolved provider config with all defaults filled in."""

    provider: LLMProvider
    model: str
    base_url: str
    api_key_env: str

    @property
    def api_key(self) -> str | None:
        """Resolve and return the API key for this provider."""
        return resolve_api_key(self.provider, self.api_key_env)

    @property
    def is_openai_compat(self) -> bool:
        """Whether this provider uses the OpenAI-compatible client."""
        return self.provider in (LLMProvider.OLLAMA, LLMProvider.OPENAI, LLMProvider.OPENROUTER)

    @property
    def uses_agent_sdk(self) -> bool:
        """Whether this provider uses the Claude Agent SDK."""
        return self.provider == LLMProvider.CLAUDE


# ---------------------------------------------------------------------------
# Main config
# ---------------------------------------------------------------------------


def _default_config_dir() -> Path:
    from claude_code_assist.paths import default_config_dir  # noqa: PLC0415

    return default_config_dir()


class CompanionConfig(BaseModel):
    """Application configuration."""

    # Paths
    config_dir: Path = Field(default_factory=_default_config_dir)
    project_dir: Path | None = Field(
        default=None,
        description="Project-local config directory when using a project-specific companion. Set by CLI resolution.",
    )

    # Stats
    stat_config: StatConfig = Field(default_factory=StatConfig)
    rarity_weights: dict[Rarity, float] = Field(default_factory=lambda: dict(DEFAULT_RARITY_WEIGHTS))

    # Generation
    seed: int = Field(
        default_factory=lambda: int(time.time()),
        description="Random seed for companion generation. Defaults to current timestamp.",
    )

    # Timing
    comment_interval_seconds: float = 30.0
    idle_chatter_interval_seconds: float = 300.0
    max_comments_per_session: int = 0

    # Commentary
    max_comment_length: int = 300
    max_idle_length: int = 150

    # Animation
    idle_duration_seconds: float = 3.0
    reaction_duration_seconds: float = 0.5
    sleep_duration_seconds: float = 60.0
    sleep_threshold_seconds: int = 120

    # Logging
    log_level: str = "WARNING"
    log_file: str = "debug.log"

    # --- Single global LLM provider (shared by commentary + profile) ---
    provider_config: ProviderConfig = Field(
        default_factory=ProviderConfig,
        description="LLM provider for commentary and profile generation. Image art uses Gemini directly.",
    )

    # Art
    art_size: int = Field(default=120, description="Target pixel height for art sprites. Must be a multiple of 6.")
    art_dir_path: str = Field(
        default="art", description="Subdirectory name under config_dir where art frame files are stored."
    )
    art_prompt: str = Field(
        default="",
        description=(
            "Custom prompt override for image generation. "
            "Frame layout instructions are always appended. "
            "Empty string uses auto-generated prompt."
        ),
    )

    # --- Resolved provider convenience property ---

    @property
    def resolved_provider(self) -> ResolvedProviderConfig:
        """Resolved LLM provider config used by commentary and profile generation."""
        return self.provider_config.resolve()

    @field_validator("log_file")
    @classmethod
    def _validate_log_file(cls, v: str) -> str:
        """Ensure log_file is a bare filename with no directory components."""
        p = Path(v)
        if p.name != v or v.startswith(".") or "/" in v or "\\" in v:
            raise ValueError(f"log_file must be a bare filename (no path separators or leading dots), got: {v!r}")
        return v

    @field_validator("art_dir_path")
    @classmethod
    def _validate_art_dir_path(cls, v: str) -> str:
        """Ensure art_dir_path is a single directory name with no traversal."""
        p = Path(v)
        if p.name != v or v.startswith(".") or "/" in v or "\\" in v or ".." in v.split("/"):
            raise ValueError(
                f"art_dir_path must be a single directory name (no path separators or leading dots), got: {v!r}"
            )
        return v

    @property
    def companion_data_dir(self) -> Path:
        """Base directory for companion data (art, logs). Uses project dir when active."""
        return self.project_dir if self.project_dir else self.config_dir

    @property
    def art_dir(self) -> Path:
        """Path to the art directory, scoped to project if active."""
        return self.companion_data_dir / self.art_dir_path

    @property
    def profile_path(self) -> Path:
        """Path to the global companion profile."""
        return self.config_dir / "companion" / "profile.json"

    @property
    def log_file_path(self) -> Path:
        """Full path to the log file, scoped to project if active."""
        return self.companion_data_dir / self.log_file

    @property
    def config_file_path(self) -> Path:
        """Full path to the config file."""
        return self.config_dir / "config.json"


# Fields the runtime sets that we don't want in the persisted file.
_EXCLUDE_FROM_FILE = {"config_dir"}

# Legacy keys silently dropped on load — replaced by `provider_config`.
_LEGACY_DROPPED_KEYS = {
    "profile_provider_config",
    "commentary_provider_config",
    "image_art_provider_config",
    "art_mode",
    "ascii_art_frames",
    "art_max_width_pct",
    "halfblock_size",
    "chroma_tolerance",
    "bubble_placement",
}


def save_config(config: CompanionConfig, path: Path) -> None:
    """Save configuration to ``config.json``, preserving unknown keys.

    Drops the legacy ``settings`` sub-object on write — that data now
    lives in ``settings.json`` (owned by :class:`SettingsStore`) and is
    migrated out the first time :meth:`SettingsStore.load` runs against
    an old config dir.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.is_file():
        try:
            text = path.read_text(encoding="utf-8")
            if text.strip():
                existing = json.loads(text)
            if not isinstance(existing, dict):
                existing = {}
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read existing %s; rewriting from scratch", path, exc_info=True)
            existing = {}

    # Drop legacy keys (including the old settings block) so they don't
    # keep round-tripping through the file.
    for key in _LEGACY_DROPPED_KEYS:
        existing.pop(key, None)
    existing.pop("settings", None)

    data = config.model_dump(mode="json", exclude=_EXCLUDE_FROM_FILE)
    if "rarity_weights" in data:
        data["rarity_weights"] = {str(k): v for k, v in data["rarity_weights"].items()}

    merged = {**existing, **data}
    path.write_text(json.dumps(merged, indent=2), encoding="utf-8")


def load_config(path: Path) -> CompanionConfig:
    """Load configuration from ``config.json``, falling back to defaults.

    The legacy ``settings`` sub-object and other dropped keys are
    silently ignored — they're owned elsewhere or no longer used.
    """
    if not path.exists():
        logger.debug("Config file not found at %s, using defaults", path)
        return CompanionConfig()

    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text) if text.strip() else {}
        if not isinstance(data, dict):
            logger.warning("Top-level config in %s is not an object; using defaults", path)
            return CompanionConfig()
        data = {k: v for k, v in data.items() if k != "settings" and k not in _LEGACY_DROPPED_KEYS}
        if "rarity_weights" in data:
            data["rarity_weights"] = {Rarity(k): v for k, v in data["rarity_weights"].items()}
        return CompanionConfig(**data)
    except json.JSONDecodeError:
        logger.exception("Malformed JSON in config %s, using defaults", path)
        return CompanionConfig()
    except ValidationError:
        logger.exception("Config validation failed for %s, using defaults", path)
        return CompanionConfig()
    except OSError:
        logger.exception("Could not read config file %s, using defaults", path)
        return CompanionConfig()
