"""
Structured logging for the LLM Gateway.

Two sinks:
  1. Console (stdout) — human-readable, INFO level by default.
  2. JSONL file (logs/requests.jsonl) — one record per upstream attempt with
     provider/model/key index/latency/status. Useful for debugging fallbacks.

The JSONL writer is async-safe (single asyncio.Lock around appends).
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

LOG_DIR = Path("logs")
REQUEST_LOG_FILE = LOG_DIR / "requests.jsonl"


class _StdoutFormatter(logging.Formatter):
    """Compact, color-free formatter suitable for systemd / Docker / dev shell."""

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
        level = record.levelname[0]  # D / I / W / E / C
        msg = record.getMessage()
        extras = ""
        if record.__dict__.get("extra_fields"):
            extras = " " + " ".join(
                f"{k}={v}" for k, v in record.__dict__["extra_fields"].items()
            )
        return f"{ts} {level} [{record.name}] {msg}{extras}"


class _ExtraAdapter(logging.LoggerAdapter):
    """Allow passing extra fields via logger.info(msg, extra={...})."""

    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        extra = kwargs.pop("extra", {}) or {}
        # merge with adapter's own extra
        merged = {**(self.extra or {}), **extra}
        # Promote to a known key so the formatter can read it
        if merged:
            kwargs["extra"] = {**merged, "extra_fields": merged}
        else:
            kwargs["extra"] = {"extra_fields": {}}
        return msg, kwargs


def setup_logging(level: str = "INFO") -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)
    # Remove any pre-existing handlers (uvicorn adds its own; we keep them)
    # but ensure our stdout handler is present once.
    seen = False
    for h in root.handlers:
        if getattr(h, "_gateway_handler", False):
            seen = True
            break
    if not seen:
        h = logging.StreamHandler(sys.stdout)
        h._gateway_handler = True  # type: ignore[attr-defined]
        h.setFormatter(_StdoutFormatter())
        root.addHandler(h)


def get_logger(name: str) -> logging.LoggerAdapter:
    return _ExtraAdapter(logging.getLogger(name), {})


# ─────────────────────────────────────────────────────────────────────────────
# JSONL request log (async-safe)
# ─────────────────────────────────────────────────────────────────────────────

import asyncio

_log_lock: Optional[asyncio.Lock] = None


def _get_log_lock() -> asyncio.Lock:
    global _log_lock
    if _log_lock is None:
        _log_lock = asyncio.Lock()
    return _log_lock


async def log_request_attempt(record: dict[str, Any]) -> None:
    """Append a single attempt record to logs/requests.jsonl.

    Expected keys: ts, request_id, provider, model, key_index, status,
    latency_ms, error, stream, cycle, step_index.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str, ensure_ascii=False) + "\n"
    lock = _get_log_lock()
    async with lock:
        # Append mode — open/write/close per record is fine at this scale.
        with REQUEST_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line)
