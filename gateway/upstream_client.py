"""
Singleton accessor for the UpstreamClient.

The UpstreamClient wraps a shared httpx.AsyncClient that should live for the
entire gateway process. This module exposes get_upstream_client() which
returns that singleton, lazily initializing it on first call.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from .client import UpstreamClient

_client: Optional[UpstreamClient] = None
_init_lock = asyncio.Lock()


async def init_upstream_client() -> UpstreamClient:
    """Initialize the global UpstreamClient. Call once at startup."""
    global _client
    if _client is None:
        async with _init_lock:
            if _client is None:
                _client = UpstreamClient()
    return _client


def get_upstream_client() -> UpstreamClient:
    """Return the singleton UpstreamClient. Must have been initialized first."""
    global _client
    if _client is None:
        # Lazy init (sync) — used in routes after startup has initialized it.
        # If called before init_upstream_client, we create one synchronously.
        _client = UpstreamClient()  # type: ignore[assignment]
    return _client


async def close_upstream_client() -> None:
    """Close the singleton client at shutdown."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
