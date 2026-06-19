"""
Admin endpoints: /health, /status, /admin/reload.

  GET  /health          — liveness probe (always 200 if the process is up)
  GET  /status          — full runtime state snapshot (key cooldowns, counters, etc.)
  POST /admin/reload    — manually trigger a config reload
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .config import load_config_or_none
from .logging_setup import get_logger

log = get_logger("gateway.admin")
router = APIRouter()


@router.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(content={"status": "ok"})


@router.get("/status")
async def status(request: Request) -> JSONResponse:
    state = request.app.state.gateway_state
    snap = await state.snapshot()
    return JSONResponse(content=snap)


@router.post("/admin/reload")
async def reload_config(request: Request) -> JSONResponse:
    state = request.app.state.gateway_state
    config_path = request.app.state.config_path
    new_cfg = load_config_or_none(config_path)
    if new_cfg is None:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": "config reload failed (see server logs)"},
        )
    state.reload(new_cfg)
    log.info(
        "manual config reload succeeded",
        extra={"providers": list(new_cfg.providers.keys())},
    )
    return JSONResponse(
        content={
            "ok": True,
            "providers": list(new_cfg.providers.keys()),
            "waterfall_steps": len(new_cfg.waterfall),
        }
    )
