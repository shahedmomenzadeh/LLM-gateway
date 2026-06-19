"""
Configuration schema and loader for the LLM Gateway.

Reads ./config.yaml (path can be overridden by the GATEWAY_CONFIG env var or
the --config CLI flag). Validates the structure with Pydantic and exposes a
typed Config object to the rest of the gateway.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

class GatewayConfig(BaseModel):
    rate_limit_rpm: int = 0
    key_cooldown_seconds: int = 20
    timeout: float = 60.0
    waterfall_max_cycles: int = 10
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    host: str = "0.0.0.0"
    port: int = 8000

    @field_validator("rate_limit_rpm")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("rate_limit_rpm must be >= 0")
        return v

    @field_validator("waterfall_max_cycles")
    @classmethod
    def _at_least_one(cls, v: int) -> int:
        if v < 1:
            raise ValueError("waterfall_max_cycles must be >= 1")
        return v


class ProviderConfig(BaseModel):
    base_url: str
    timeout: float = 60.0
    max_retries: int = 1
    api_keys: List[str]
    extra_headers: Dict[str, str] = Field(default_factory=dict)

    @field_validator("api_keys")
    @classmethod
    def _non_empty_keys(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("at least one api_key is required")
        for k in v:
            if not isinstance(k, str) or not k.strip():
                raise ValueError("api_keys must be non-empty strings")
        return v

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")


class WaterfallStep(BaseModel):
    provider: str
    models: List[str]

    @field_validator("models")
    @classmethod
    def _non_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("waterfall step must have at least one model")
        return v


class Config(BaseModel):
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    providers: Dict[str, ProviderConfig]
    waterfall: List[WaterfallStep]

    @model_validator(mode="after")
    def _check_waterfall_providers_exist(self) -> "Config":
        for step in self.waterfall:
            if step.provider not in self.providers:
                raise ValueError(
                    f"waterfall step references unknown provider '{step.provider}'"
                )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG_PATH = Path("config.yaml")


def resolve_config_path(explicit: Optional[str] = None) -> Path:
    """Resolve which config file to load.

    Priority: explicit arg > $GATEWAY_CONFIG > ./config.yaml
    """
    if explicit:
        return Path(explicit)
    env = os.environ.get("GATEWAY_CONFIG")
    if env:
        return Path(env)
    return DEFAULT_CONFIG_PATH


def load_config(path: Optional[Path | str] = None) -> Config:
    """Load and validate the config file. Raises on missing file or bad schema."""
    p = Path(path) if path else resolve_config_path()
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}
    return Config.model_validate(raw)


def load_config_or_none(path: Optional[Path | str] = None) -> Optional[Config]:
    """Like load_config but returns None on error (used during hot-reload)."""
    try:
        return load_config(path)
    except Exception as e:  # noqa: BLE001
        # Log to stderr; the caller (watcher) decides what to do
        import sys
        print(f"[gateway.config] failed to reload config: {e}", file=sys.stderr)
        return None
