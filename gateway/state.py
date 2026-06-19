"""
Runtime state for the LLM Gateway.

Tracks per-key cooldowns, per-key/model/provider error counts, and global
request counters. All mutations go through an asyncio.Lock so the state is
safe under concurrent requests.

The state is intentionally kept separate from the Config object so that
hot-reloads of the config don't wipe out runtime cooldowns (cooldowns are
keyed by `(provider, key_string)` and persist across reloads when the same
key is still present in the new config).
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import Config, ProviderConfig


@dataclass
class KeyState:
    """Mutable per-key runtime state."""

    provider: str
    key: str
    key_index: int  # position in the provider's api_keys list at load time
    cooldown_until: float = 0.0  # epoch seconds; 0 = not cooling down
    cooldown_set_by: Optional[str] = None  # request_id that set the cooldown
    consecutive_errors: int = 0
    total_requests: int = 0
    total_errors: int = 0


@dataclass
class ModelState:
    """Mutable per-(provider, model) runtime state."""

    provider: str
    model: str
    total_requests: int = 0
    total_errors: int = 0


@dataclass
class ProviderState:
    """Mutable per-provider runtime state."""

    name: str
    total_requests: int = 0
    total_errors: int = 0
    keys: List[KeyState] = field(default_factory=list)
    models: Dict[str, ModelState] = field(default_factory=dict)


@dataclass
class GatewayStats:
    total_requests: int = 0
    total_errors: int = 0
    total_fallbacks: int = 0  # number of times we advanced to a different key/model/provider
    total_cycles: int = 0
    started_at: float = field(default_factory=time.time)
    last_request_at: float = 0.0


class State:
    """Thread-async-safe runtime state container."""

    def __init__(self, config: Config) -> None:
        self._lock = asyncio.Lock()
        self._config: Config = config
        self._providers: Dict[str, ProviderState] = self._build_provider_state(config)
        self._stats = GatewayStats()
        # Track last-used position so /status can report "where we are"
        self._last_step_index: int = 0
        self._last_model_index: int = 0
        self._last_key_index: int = 0

    # ── construction / reload ────────────────────────────────────────────

    @staticmethod
    def _build_provider_state(
        config: Config, previous: Optional[Dict[str, ProviderState]] = None
    ) -> Dict[str, ProviderState]:
        """Build provider state from config. Preserves cooldowns from `previous`
        when a (provider, key) pair is unchanged."""
        out: Dict[str, ProviderState] = {}
        for name, pcfg in config.providers.items():
            prev_p = previous.get(name) if previous else None
            ps = ProviderState(name=name)
            for idx, key in enumerate(pcfg.api_keys):
                # Try to find a matching key in the previous state to preserve cooldown
                prev_k: Optional[KeyState] = None
                if prev_p:
                    for pk in prev_p.keys:
                        if pk.key == key:
                            prev_k = pk
                            break
                ks = KeyState(
                    provider=name,
                    key=key,
                    key_index=idx,
                    cooldown_until=prev_k.cooldown_until if prev_k else 0.0,
                    consecutive_errors=prev_k.consecutive_errors if prev_k else 0,
                    total_requests=prev_k.total_requests if prev_k else 0,
                    total_errors=prev_k.total_errors if prev_k else 0,
                )
                ps.keys.append(ks)
            for step in config.waterfall:
                if step.provider == name:
                    for model in step.models:
                        prev_m = prev_p.models.get(model) if prev_p else None
                        ps.models[model] = ModelState(
                            provider=name,
                            model=model,
                            total_requests=prev_m.total_requests if prev_m else 0,
                            total_errors=prev_m.total_errors if prev_m else 0,
                        )
            out[name] = ps
        return out

    def reload(self, new_config: Config) -> None:
        """Atomically swap the config and rebuild provider state, preserving
        cooldowns for unchanged (provider, key) pairs."""
        # NOTE: must be called under self._lock
        new_providers = self._build_provider_state(new_config, self._providers)
        self._config = new_config
        self._providers = new_providers

    # ── read accessors ───────────────────────────────────────────────────

    @property
    def config(self) -> Config:
        return self._config

    @property
    def stats(self) -> GatewayStats:
        return self._stats

    def get_provider(self, name: str) -> Optional[ProviderState]:
        return self._providers.get(name)

    def get_provider_config(self, name: str) -> Optional[ProviderConfig]:
        return self._config.providers.get(name)

    # ── key cooldown logic ───────────────────────────────────────────────

    async def is_key_available(self, provider: str, key_index: int) -> bool:
        async with self._lock:
            ps = self._providers.get(provider)
            if not ps or key_index >= len(ps.keys):
                return False
            ks = ps.keys[key_index]
            return ks.cooldown_until <= time.time()

    async def is_key_available_for_request(
        self, provider: str, key_index: int, request_id: str
    ) -> bool:
        """Return True if the key can be tried by THIS request.

        A key is unavailable for this request if:
          - it is in cooldown AND
          - the cooldown was set by a DIFFERENT request
        Cooldowns set by THIS request are ignored (so the same request can try
        the same key against multiple models within a single cycle, per the
        waterfall spec).
        """
        async with self._lock:
            ps = self._providers.get(provider)
            if not ps or key_index >= len(ps.keys):
                return False
            ks = ps.keys[key_index]
            if ks.cooldown_until <= time.time():
                return True
            # In cooldown — but only block if set by a different request
            return ks.cooldown_set_by == request_id

    async def mark_key_error(
        self,
        provider: str,
        key_index: int,
        cooldown_base: int,
        request_id: str,
    ) -> float:
        """Mark a key as having errored. Returns the new cooldown_until timestamp."""
        async with self._lock:
            ps = self._providers.get(provider)
            if not ps or key_index >= len(ps.keys):
                return 0.0
            ks = ps.keys[key_index]
            ks.consecutive_errors += 1
            ks.total_errors += 1
            # Exponential backoff: base * 2^(errors-1), capped at base*60
            backoff = min(cooldown_base * (2 ** (ks.consecutive_errors - 1)), cooldown_base * 60)
            ks.cooldown_until = time.time() + backoff
            ks.cooldown_set_by = request_id
            return ks.cooldown_until

    async def mark_key_success(self, provider: str, key_index: int) -> None:
        async with self._lock:
            ps = self._providers.get(provider)
            if not ps or key_index >= len(ps.keys):
                return
            ks = ps.keys[key_index]
            ks.consecutive_errors = 0
            ks.cooldown_until = 0.0
            ks.total_requests += 1

    async def mark_model_attempt(self, provider: str, model: str, error: bool) -> None:
        async with self._lock:
            ps = self._providers.get(provider)
            if not ps:
                return
            ms = ps.models.get(model)
            if not ms:
                ms = ModelState(provider=provider, model=model)
                ps.models[model] = ms
            ms.total_requests += 1
            if error:
                ms.total_errors += 1
            ps.total_requests += 1
            if error:
                ps.total_errors += 1

    async def reset_all_cooldowns(self, request_id: Optional[str] = None) -> None:
        """Reset cooldowns at the start of a new cycle.

        If `request_id` is provided, only resets cooldowns that were set by
        THAT request — cooldowns set by other concurrent requests are left
        alone, so we don't disrupt them.
        If `request_id` is None, resets ALL cooldowns (used at startup / reload).
        """
        async with self._lock:
            for ps in self._providers.values():
                for ks in ps.keys:
                    if request_id is None or ks.cooldown_set_by == request_id:
                        ks.cooldown_until = 0.0
                        ks.cooldown_set_by = None

    async def record_request_start(self) -> None:
        async with self._lock:
            self._stats.total_requests += 1
            self._stats.last_request_at = time.time()

    async def record_request_error(self) -> None:
        async with self._lock:
            self._stats.total_errors += 1

    async def record_fallback(self) -> None:
        async with self._lock:
            self._stats.total_fallbacks += 1

    async def record_cycle(self) -> None:
        async with self._lock:
            self._stats.total_cycles += 1

    async def update_position(
        self, step_index: int, model_index: int, key_index: int
    ) -> None:
        async with self._lock:
            self._last_step_index = step_index
            self._last_model_index = model_index
            self._last_key_index = key_index

    # ── snapshot for /status ─────────────────────────────────────────────

    async def snapshot(self) -> dict:
        async with self._lock:
            now = time.time()
            providers_snapshot = []
            for name, ps in self._providers.items():
                pcfg = self._config.providers.get(name)
                keys_snap = []
                for ks in ps.keys:
                    cooldown_remaining = max(0.0, ks.cooldown_until - now)
                    keys_snap.append(
                        {
                            "key_index": ks.key_index,
                            "key_preview": _mask_key(ks.key),
                            "available": ks.cooldown_until <= now,
                            "cooldown_remaining_seconds": round(cooldown_remaining, 1),
                            "consecutive_errors": ks.consecutive_errors,
                            "total_requests": ks.total_requests,
                            "total_errors": ks.total_errors,
                        }
                    )
                models_snap = []
                for mname, ms in ps.models.items():
                    models_snap.append(
                        {
                            "model": mname,
                            "total_requests": ms.total_requests,
                            "total_errors": ms.total_errors,
                        }
                    )
                providers_snapshot.append(
                    {
                        "name": name,
                        "base_url": pcfg.base_url if pcfg else None,
                        "timeout": pcfg.timeout if pcfg else None,
                        "total_requests": ps.total_requests,
                        "total_errors": ps.total_errors,
                        "keys": keys_snap,
                        "models": models_snap,
                    }
                )
            waterfall_snap = [
                {"provider": s.provider, "models": s.models}
                for s in self._config.waterfall
            ]
            return {
                "uptime_seconds": round(now - self._stats.started_at, 1),
                "totals": {
                    "total_requests": self._stats.total_requests,
                    "total_errors": self._stats.total_errors,
                    "total_fallbacks": self._stats.total_fallbacks,
                    "total_cycles": self._stats.total_cycles,
                    "last_request_at": self._stats.last_request_at,
                },
                "waterfall": {
                    "max_cycles": self._config.gateway.waterfall_max_cycles,
                    "last_position": {
                        "step_index": self._last_step_index,
                        "model_index": self._last_model_index,
                        "key_index": self._last_key_index,
                    },
                    "steps": waterfall_snap,
                },
                "providers": providers_snapshot,
            }


def _mask_key(key: str) -> str:
    """Mask a key for display: show first 4 and last 4 chars."""
    if len(key) <= 10:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"
