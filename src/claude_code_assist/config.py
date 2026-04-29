"""Configuration system for the desktop companion."""

from __future__ import annotations

import json
import logging
import os
import time
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from claude_code_assist.models.rarity import DEFAULT_RARITY_WEIGHTS, Rarity
from claude_code_assist.models.stats import StatConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM Provider system
# ---------------------------------------------------------------------------

# Pipeline type determines which model default to use (text vs image)
PipelineType = Literal["text", "image"]


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

# Provider defaults keyed by (provider, pipeline_type)
_PROVIDER_DEFAULTS: dict[tuple[LLMProvider, PipelineType], dict[str, str]] = {
    # Claude — text only (Agent SDK handles auth)
    (LLMProvider.CLAUDE, "text"): {
        "model": "claude-haiku-4-5",
        "base_url": "",
        "api_key_env": "",
    },
    (LLMProvider.CLAUDE, "image"): {
        "model": "claude-haiku-4-5",
        "base_url": "",
        "api_key_env": "",
    },
    # Ollama — local, no real API key needed
    (LLMProvider.OLLAMA, "text"): {
        "model": "llama3.2",
        "base_url": "http://localhost:11434/v1",
        "api_key_env": "",
    },
    (LLMProvider.OLLAMA, "image"): {
        "model": "llama3.2",
        "base_url": "http://localhost:11434/v1",
        "api_key_env": "",
    },
    # OpenAI
    (LLMProvider.OPENAI, "text"): {
        "model": "gpt-4o",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    (LLMProvider.OPENAI, "image"): {
        "model": "gpt-image-1.5",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    # OpenRouter
    (LLMProvider.OPENROUTER, "text"): {
        "model": "anthropic/claude-haiku-4-5",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    (LLMProvider.OPENROUTER, "image"): {
        "model": "openai/dall-e-3",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    # Gemini
    (LLMProvider.GEMINI, "text"): {
        "model": "gemini-2.5-flash",
        "base_url": "",
        "api_key_env": "GEMINI_API_KEY",
    },
    (LLMProvider.GEMINI, "image"): {
        "model": "gemini-3.1-flash-image-preview",
        "base_url": "",
        "api_key_env": "GEMINI_API_KEY",
    },
}


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


class PipelineProviderConfig(BaseModel):
    """Provider configuration for a single LLM pipeline."""

    provider: LLMProvider = LLMProvider.CLAUDE
    model: str = ""
    base_url: str = ""
    api_key_env: str = ""

    def resolve(self, pipeline: PipelineType = "text") -> ResolvedProviderConfig:
        """Fill in provider-specific defaults for any empty fields.

        Args:
            pipeline: Whether this is a "text" or "image" pipeline (affects default model).

        Returns:
            Fully resolved config with all fields populated.
        """
        defaults = _PROVIDER_DEFAULTS.get((self.provider, pipeline), {})
        resolved_model = self.model or defaults.get("model", "")

        # Resolve Claude model aliases
        if self.provider == LLMProvider.CLAUDE:
            resolved_model = _CLAUDE_MODEL_ALIASES.get(resolved_model.lower().strip(), resolved_model)

        return ResolvedProviderConfig(
            provider=self.provider,
            model=resolved_model,
            base_url=self.base_url or defaults.get("base_url", ""),
            api_key_env=self.api_key_env or defaults.get("api_key_env", ""),
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
# Display enums
# ---------------------------------------------------------------------------


class ArtMode(StrEnum):
    """Supported display modes for companion art."""

    ASCII = "ascii"
    SIXEL_ART = "sixel-art"
    MACOS_DESKTOP = "macos-desktop"


class BubblePlacement(StrEnum):
    """Position of the speech bubble relative to the companion art."""

    TOP = "top"
    RIGHT = "right"
    BOTTOM = "bottom"


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
    ascii_art_frames: int = 6
    idle_duration_seconds: float = 3.0
    reaction_duration_seconds: float = 0.5
    sleep_duration_seconds: float = 60.0
    sleep_threshold_seconds: int = 120

    # Logging
    log_level: str = "WARNING"
    log_file: str = "debug.log"

    # --- Per-pipeline LLM provider configs ---
    profile_provider_config: PipelineProviderConfig = Field(
        default_factory=PipelineProviderConfig,
        description="Provider for companion profile generation and ASCII art regeneration.",
    )
    commentary_provider_config: PipelineProviderConfig = Field(
        default_factory=PipelineProviderConfig,
        description="Provider for commentary and idle chatter generation.",
    )
    image_art_provider_config: PipelineProviderConfig = Field(
        default_factory=lambda: PipelineProviderConfig(provider=LLMProvider.OPENAI),
        description="Provider for sixel-art image generation.",
    )

    # Art mode
    art_mode: ArtMode = Field(default=ArtMode.ASCII, description="Display mode: ascii or sixel-art.")
    art_max_width_pct: int = Field(
        default=40, description="Percentage of terminal width allocated to the art panel (1-100)."
    )
    art_size: int = Field(default=120, description="Target pixel height for art sprites. Must be a multiple of 6.")
    halfblock_size: int = Field(
        default=48,
        description=(
            "Target pixel height for sixel-art half-block rendering. Must be even; each terminal row covers 2 pixels."
        ),
    )
    chroma_tolerance: int = Field(
        default=30,
        description=(
            "Tolerance for chroma key background removal during art processing (0-255). "
            "Higher values remove more of the background color."
        ),
    )
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

    # Display
    bubble_placement: BubblePlacement = Field(
        default=BubblePlacement.BOTTOM,
        description="Speech bubble position relative to companion art: top, right, or bottom.",
    )

    # --- Resolved provider convenience properties ---

    @property
    def resolved_profile_provider(self) -> ResolvedProviderConfig:
        """Resolved provider config for profile/art regeneration pipeline."""
        return self.profile_provider_config.resolve("text")

    @property
    def resolved_commentary_provider(self) -> ResolvedProviderConfig:
        """Resolved provider config for commentary pipeline."""
        return self.commentary_provider_config.resolve("text")

    @property
    def resolved_image_art_provider(self) -> ResolvedProviderConfig:
        """Resolved provider config for image art pipeline."""
        return self.image_art_provider_config.resolve("image")

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


def save_config(config: CompanionConfig, path: Path) -> None:
    """Save configuration to ``config.json``, preserving unknown keys.

    The tray-toggle settings (``settings`` sub-object) are owned by
    :class:`SettingsStore` in ``qt/settings.py``, which mutates the same
    file. To avoid clobbering them when ``companion new`` writes the
    rest of the config, we read the existing file first, layer the
    model fields on top, and write the merged result.
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

    data = config.model_dump(mode="json", exclude=_EXCLUDE_FROM_FILE)
    if "rarity_weights" in data:
        data["rarity_weights"] = {str(k): v for k, v in data["rarity_weights"].items()}

    merged = {**existing, **data}
    # The tray-managed ``settings`` block is preserved unless ``data`` set
    # it explicitly (which the model never does — it's owned elsewhere).
    if "settings" in existing and "settings" not in data:
        merged["settings"] = existing["settings"]

    path.write_text(json.dumps(merged, indent=2), encoding="utf-8")


def load_config(path: Path) -> CompanionConfig:
    """Load configuration from ``config.json``, falling back to defaults.

    Unknown keys (notably the tray-managed ``settings`` block) are
    ignored by the model — they're preserved on save by ``save_config``.
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
        # Drop the tray-settings block and convert rarity keys.
        data = {k: v for k, v in data.items() if k != "settings"}
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
