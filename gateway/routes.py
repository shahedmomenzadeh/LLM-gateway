"""
FastAPI routes for the LLM Gateway.

Exposes OpenAI-compatible endpoints:
  POST /v1/chat/completions   — chat (streaming + non-streaming)
  POST /v1/completions        — legacy text completions
  POST /v1/embeddings         — embeddings (best-effort; only some providers support)
  GET  /v1/models             — list models from the waterfall
  POST /v1/messages           — Anthropic-style (translated to OpenAI internally)

The user-sent `model` field is IGNORED — the waterfall picks the model.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .anthropic import (
    anthropic_request_to_openai,
    openai_response_to_anthropic,
    translate_stream_openai_to_anthropic,
)
from .client import NonStreamResult, StreamResult, UpstreamError
from .logging_setup import get_logger
from .state import State
from .upstream_client import get_upstream_client
from .waterfall import (
    WaterfallExhaustedError,
    execute_non_stream,
    execute_stream,
)

log = get_logger("gateway.routes")
router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_state(request: Request) -> State:
    return request.app.state.gateway_state


def _is_stream(body: Dict[str, Any]) -> bool:
    return bool(body.get("stream", False))


def _select_content_type(headers: Dict[str, str], is_stream: bool) -> str:
    if is_stream:
        return "text/event-stream"
    return "application/json"


def _filter_response_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Pass through content-type and a few safe headers; drop hop-by-hop ones."""
    allowed = {"content-type", "x-request-id", "openai-organization", "openai-processing-ms"}
    out: Dict[str, str] = {}
    for k, v in headers.items():
        if k.lower() in allowed:
            out[k] = v
    # Always force the correct content-type for our response
    return out


async def _read_body(request: Request) -> Dict[str, Any]:
    try:
        raw = await request.body()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to read body: {e}")
    if not raw:
        raise HTTPException(status_code=400, detail="empty request body")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# /v1/chat/completions
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    state = _get_state(request)
    body = await _read_body(request)
    if "messages" not in body:
        raise HTTPException(status_code=400, detail="'messages' field is required")

    stream = _is_stream(body)
    client = get_upstream_client()

    try:
        if stream:
            result = await execute_stream(
                client=client, state=state, path="/chat/completions", body=body
            )
            return _make_streaming_response(result)
        else:
            result = await execute_non_stream(
                client=client, state=state, path="/chat/completions", body=body
            )
            return _make_json_response(result)
    except WaterfallExhaustedError as e:
        log.error(f"chat_completions waterfall exhausted: {e}")
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": "all upstream providers/models/keys failed",
                    "type": "waterfall_exhausted",
                    "attempts": e.attempts,
                }
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# /v1/completions  (legacy text completions)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/v1/completions")
async def completions(request: Request):
    state = _get_state(request)
    body = await _read_body(request)
    if "prompt" not in body:
        raise HTTPException(status_code=400, detail="'prompt' field is required")

    stream = _is_stream(body)
    client = get_upstream_client()

    try:
        if stream:
            result = await execute_stream(
                client=client, state=state, path="/completions", body=body
            )
            return _make_streaming_response(result)
        else:
            result = await execute_non_stream(
                client=client, state=state, path="/completions", body=body
            )
            return _make_json_response(result)
    except WaterfallExhaustedError as e:
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": "all upstream providers/models/keys failed",
                    "type": "waterfall_exhausted",
                    "attempts": e.attempts,
                }
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# /v1/embeddings
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/v1/embeddings")
async def embeddings(request: Request):
    state = _get_state(request)
    body = await _read_body(request)
    if "input" not in body:
        raise HTTPException(status_code=400, detail="'input' field is required")
    client = get_upstream_client()
    try:
        result = await execute_non_stream(
            client=client, state=state, path="/embeddings", body=body
        )
        return _make_json_response(result)
    except WaterfallExhaustedError as e:
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": "all upstream providers/models/keys failed",
                    "type": "waterfall_exhausted",
                    "attempts": e.attempts,
                }
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# /v1/models
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/v1/models")
async def list_models(request: Request):
    state = _get_state(request)
    config = state.config
    models_out: List[Dict[str, Any]] = []
    seen = set()
    for step in config.waterfall:
        for m in step.models:
            if m in seen:
                continue
            seen.add(m)
            models_out.append(
                {
                    "id": m,
                    "object": "model",
                    "created": 0,
                    "owned_by": step.provider,
                }
            )
    return JSONResponse(content={"object": "list", "data": models_out})


# ─────────────────────────────────────────────────────────────────────────────
# /v1/messages  (Anthropic-style)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/v1/messages")
async def anthropic_messages(request: Request):
    state = _get_state(request)
    body = await _read_body(request)
    requested_model = body.get("model", "")
    # Translate to OpenAI format
    openai_body = anthropic_request_to_openai(body)
    stream = _is_stream(openai_body)
    client = get_upstream_client()

    try:
        if stream:
            result = await execute_stream(
                client=client, state=state, path="/chat/completions", body=openai_body
            )
            # Wrap the OpenAI SSE stream into an Anthropic SSE stream
            anthropic_stream = translate_stream_openai_to_anthropic(
                result.chunks, requested_model=requested_model
            )
            return StreamingResponse(
                anthropic_stream,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        else:
            result = await execute_non_stream(
                client=client, state=state, path="/chat/completions", body=openai_body
            )
            # Parse the OpenAI JSON and translate to Anthropic shape
            try:
                oai_json = json.loads(result.body)
            except json.JSONDecodeError:
                return Response(
                    content=result.body,
                    status_code=result.status,
                    media_type="application/json",
                )
            anth_json = openai_response_to_anthropic(
                oai_json, requested_model=requested_model
            )
            return JSONResponse(content=anth_json)
    except WaterfallExhaustedError as e:
        return JSONResponse(
            status_code=502,
            content={
                "type": "error",
                "error": {
                    "type": "waterfall_exhausted",
                    "message": "all upstream providers/models/keys failed",
                    "attempts": e.attempts,
                },
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# Response builders
# ─────────────────────────────────────────────────────────────────────────────

def _make_json_response(result: NonStreamResult) -> Response:
    return Response(
        content=result.body,
        status_code=result.status,
        headers=_filter_response_headers(result.headers),
        media_type="application/json",
    )


def _make_streaming_response(result: StreamResult) -> StreamingResponse:
    async def gen() -> AsyncIterator[bytes]:
        try:
            async for chunk in result.chunks:
                if chunk:
                    yield chunk
        except Exception as e:
            log.error(f"stream error after opening: {e}")
            # Best-effort: emit a final error chunk in OpenAI format
            err_payload = {
                "error": {"message": f"upstream stream error: {e}", "type": "stream_error"}
            }
            yield f"data: {json.dumps(err_payload)}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"
        finally:
            await result.aclose()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )
