"""
LLM Gateway — FastAPI entry point.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000
or
    python main.py
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gateway.admin import router as admin_router
from gateway.config import load_config, resolve_config_path
from gateway.logging_setup import get_logger, setup_logging
from gateway.routes import router as api_router
from gateway.state import State
from gateway.upstream_client import close_upstream_client, init_upstream_client
from gateway.watcher import ConfigWatcher

log = get_logger("gateway.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup + shutdown lifecycle."""
    config_path: Path = app.state.config_path
    setup_logging("INFO")
    log.info(f"LLM Gateway starting — config={config_path.resolve()}")

    try:
        config = load_config(config_path)
    except Exception as e:
        log.error(f"failed to load config: {e}")
        sys.exit(1)

    state = State(config)
    app.state.gateway_state = state
    log.info(
        "config loaded",
        extra={
            "providers": list(config.providers.keys()),
            "waterfall_steps": len(config.waterfall),
            "max_cycles": config.gateway.waterfall_max_cycles,
        },
    )

    # Init the upstream HTTP client
    await init_upstream_client()

    # Start the config file watcher (hot reload)
    loop = asyncio.get_running_loop()
    watcher = ConfigWatcher(config_path, state, loop)
    watcher.start()
    app.state.config_watcher = watcher

    log.info(
        f"LLM Gateway ready — listening on {config.gateway.host}:{config.gateway.port}"
    )
    yield

    # Shutdown
    log.info("LLM Gateway shutting down")
    try:
        watcher.stop()
    except Exception:
        pass
    await close_upstream_client()


def create_app(config_path: Path) -> FastAPI:
    app = FastAPI(
        title="LLM Gateway",
        version="1.0.0",
        description="OpenAI-compatible proxy with multi-provider waterfall fallback.",
        lifespan=lifespan,
    )
    app.state.config_path = config_path

    # CORS — allow any client (coding agents run locally)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)
    app.include_router(admin_router)

    @app.get("/")
    async def root() -> dict:
        return {
            "name": "LLM Gateway",
            "version": "1.0.0",
            "endpoints": [
                "POST /v1/chat/completions",
                "POST /v1/completions",
                "POST /v1/embeddings",
                "GET  /v1/models",
                "POST /v1/messages",
                "GET  /health",
                "GET  /status",
                "POST /admin/reload",
            ],
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Gateway proxy server")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: ./config.yaml or $GATEWAY_CONFIG)",
    )
    parser.add_argument("--host", type=str, default=None, help="Override host")
    parser.add_argument("--port", type=int, default=None, help="Override port")
    parser.add_argument("--log-level", type=str, default="info")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)

    # Pre-load to get host/port for uvicorn (will be re-loaded inside lifespan)
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"[gateway.main] failed to load config: {e}", file=sys.stderr)
        sys.exit(1)

    host = args.host or config.gateway.host
    port = args.port or config.gateway.port

    app = create_app(config_path)

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=args.log_level,
        # Pass SSL config if provided
        ssl_certfile=config.gateway.ssl_certfile or None,
        ssl_keyfile=config.gateway.ssl_keyfile or None,
    )


if __name__ == "__main__":
    main()
