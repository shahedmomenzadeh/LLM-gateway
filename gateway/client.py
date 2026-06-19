"""
Async upstream HTTP client.

Wraps httpx.AsyncClient. Handles:
  - Non-streaming requests: returns (status, headers, body_bytes)
  - Streaming requests: returns an async iterator of raw SSE bytes

On any HTTP 4xx/5xx response, raises UpstreamError so the waterfall can catch
it and move to the next key/model/provider. Network errors raise UpstreamError
too (with a distinct category field for logging).

The client intentionally does NOT parse the OpenAI response — it just forwards
bytes. This keeps the proxy transparent and lets us support any
OpenAI-compatible upstream without translation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Optional, Tuple

import httpx

from .config import ProviderConfig
from .logging_setup import get_logger

log = get_logger("gateway.client")


# ─────────────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────────────

class UpstreamError(Exception):
    """Raised when an upstream call fails (HTTP 4xx/5xx or network error)."""

    def __init__(
        self,
        message: str,
        *,
        category: str,  # 'http' | 'network' | 'timeout' | 'stream'
        status_code: Optional[int] = None,
        upstream_body: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.status_code = status_code
        self.upstream_body = upstream_body


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NonStreamResult:
    status: int
    headers: Dict[str, str]
    body: bytes
    latency_ms: float


@dataclass
class StreamResult:
    """A successfully opened SSE stream. Caller iterates `chunks` to get raw bytes."""

    status: int
    headers: Dict[str, str]
    chunks: AsyncIterator[bytes]
    latency_ms: float
    # We hold a reference to the response so it doesn't get GC'd mid-stream
    _response: httpx.Response

    async def aclose(self) -> None:
        try:
            await self._response.aclose()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Client
# ─────────────────────────────────────────────────────────────────────────────

class UpstreamClient:
    """Async upstream client. One shared httpx.AsyncClient per gateway process."""

    def __init__(self) -> None:
        # Limits tuned for many concurrent small requests
        limits = httpx.Limits(max_connections=200, max_keepalive_connections=50)
        self._http = httpx.AsyncClient(
            limits=limits,
            follow_redirects=False,
            # Per-request timeout is set explicitly in each call
            timeout=httpx.Timeout(60.0),
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    # ── internal helpers ─────────────────────────────────────────────────

    def _build_headers(
        self, provider: ProviderConfig, api_key: str, content_type: str
    ) -> Dict[str, str]:
        h: Dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": content_type,
            "Accept": content_type if content_type == "application/json" else "text/event-stream",
        }
        # Merge provider-specific extra headers (e.g. custom auth)
        for k, v in provider.extra_headers.items():
            h[k] = v
        return h

    def _build_url(self, provider: ProviderConfig, path: str) -> str:
        return f"{provider.base_url}{path}"

    # ── non-streaming ────────────────────────────────────────────────────

    async def post_non_stream(
        self,
        provider: ProviderConfig,
        path: str,
        json_body: dict,
        api_key: str,
    ) -> NonStreamResult:
        url = self._build_url(provider, path)
        headers = self._build_headers(provider, api_key, "application/json")
        t0 = time.perf_counter()
        try:
            resp = await self._http.post(
                url,
                json=json_body,
                headers=headers,
                timeout=provider.timeout,
            )
        except httpx.TimeoutException as e:
            raise UpstreamError(f"timeout: {e}", category="timeout") from e
        except httpx.HTTPError as e:
            raise UpstreamError(f"network error: {e}", category="network") from e
        latency_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code >= 400:
            body_text = ""
            try:
                body_text = resp.text
            except Exception:
                pass
            raise UpstreamError(
                f"upstream returned {resp.status_code}",
                category="http",
                status_code=resp.status_code,
                upstream_body=body_text[:2000],
            )

        return NonStreamResult(
            status=resp.status_code,
            headers={k: v for k, v in resp.headers.items()},
            body=resp.content,
            latency_ms=latency_ms,
        )

    # ── streaming ────────────────────────────────────────────────────────

    async def post_stream(
        self,
        provider: ProviderConfig,
        path: str,
        json_body: dict,
        api_key: str,
    ) -> StreamResult:
        """Open an SSE stream. Raises UpstreamError if the upstream returns a
        non-2xx status OR if the connection itself fails.

        Once a StreamResult is returned, the stream is considered "opened
        successfully" — mid-stream errors will surface as exceptions during
        iteration and are NOT converted to UpstreamError (the client has
        already begun receiving data)."""
        url = self._build_url(provider, path)
        headers = self._build_headers(provider, api_key, "application/json")
        # We need to read the response headers to check the status before
        # streaming the body, so we use stream() context.
        t0 = time.perf_counter()
        try:
            # Note: stream() returns an async context manager. We enter it,
            # send the request, check the status, and then hand off the
            # stream to the caller. The caller is responsible for closing.
            req = self._http.build_request(
                "POST", url, json=json_body, headers=headers
            )
            resp = await self._http.send(req, stream=True)
        except httpx.TimeoutException as e:
            raise UpstreamError(f"timeout opening stream: {e}", category="timeout") from e
        except httpx.HTTPError as e:
            raise UpstreamError(f"network error opening stream: {e}", category="network") from e
        latency_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code >= 400:
            # Read the error body for logging, then close
            try:
                err_body = (await resp.aread()).decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            await resp.aclose()
            raise UpstreamError(
                f"upstream returned {resp.status_code} (stream)",
                category="http",
                status_code=resp.status_code,
                upstream_body=err_body[:2000],
            )

        async def _aiter() -> AsyncIterator[bytes]:
            try:
                async for chunk in resp.aiter_raw():
                    if chunk:
                        yield chunk
            finally:
                try:
                    await resp.aclose()
                except Exception:
                    pass

        return StreamResult(
            status=resp.status_code,
            headers={k: v for k, v in resp.headers.items()},
            chunks=_aiter(),
            latency_ms=latency_ms,
            _response=resp,
        )
