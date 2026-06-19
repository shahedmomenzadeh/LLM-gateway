"""
Anthropic /v1/messages <-> OpenAI /v1/chat/completions translation.

We accept Anthropic-style requests, translate them to OpenAI format, run them
through the normal waterfall, then translate the OpenAI response back to
Anthropic format. Both non-streaming and streaming (SSE) are supported.

This is a best-effort translation covering the fields that coding agents
actually send. Edge cases (rare fields, tool-use variants) may not be fully
covered — but the common path (system + user/assistant messages, text
content blocks) works correctly.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Request translation: Anthropic -> OpenAI
# ─────────────────────────────────────────────────────────────────────────────

def anthropic_request_to_openai(body: Dict[str, Any]) -> Dict[str, Any]:
    """Translate an Anthropic /v1/messages request body to OpenAI chat format.

    Anthropic shape:
      { model, system, messages: [{role, content: str | [{type:'text', text}]}],
        max_tokens, temperature, top_p, stop_sequences, stream }
    OpenAI shape:
      { model, messages: [{role, content}], max_tokens, temperature, top_p,
        stop, stream }
    """
    out: Dict[str, Any] = {}
    # model is overridden by the waterfall — but we still pass it through
    out["model"] = body.get("model", "")
    # stream flag
    if "stream" in body:
        out["stream"] = bool(body["stream"])
    # generation params
    if "max_tokens" in body:
        out["max_tokens"] = body["max_tokens"]
    if "temperature" in body:
        out["temperature"] = body["temperature"]
    if "top_p" in body:
        out["top_p"] = body["top_p"]
    if "stop_sequences" in body:
        out["stop"] = body["stop_sequences"]

    messages: List[Dict[str, Any]] = []
    # system prompt (Anthropic allows a string or a list of content blocks)
    system = body.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            # Concatenate text blocks
            text_parts = []
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            if text_parts:
                messages.append({"role": "system", "content": "\n".join(text_parts)})

    # messages
    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Flatten text blocks; ignore non-text for now (tool_use etc.)
            text_parts: List[str] = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        # Flatten tool_result content to text
                        rc = block.get("content")
                        if isinstance(rc, str):
                            text_parts.append(rc)
                        elif isinstance(rc, list):
                            for rb in rc:
                                if isinstance(rb, dict) and rb.get("type") == "text":
                                    text_parts.append(rb.get("text", ""))
                    elif block.get("type") == "tool_use":
                        # Represent tool use as a JSON snippet
                        text_parts.append(
                            json.dumps(
                                {
                                    "tool_use": {
                                        "id": block.get("id"),
                                        "name": block.get("name"),
                                        "input": block.get("input"),
                                    }
                                },
                                ensure_ascii=False,
                            )
                        )
                elif isinstance(block, str):
                    text_parts.append(block)
            messages.append({"role": role, "content": "\n".join(text_parts)})
        else:
            messages.append({"role": role, "content": str(content) if content else ""})

    out["messages"] = messages
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Response translation: OpenAI -> Anthropic (non-streaming)
# ─────────────────────────────────────────────────────────────────────────────

def openai_response_to_anthropic(
    openai_body: Dict[str, Any],
    *,
    requested_model: str,
) -> Dict[str, Any]:
    """Translate an OpenAI chat completion JSON into Anthropic /v1/messages format."""
    choices = openai_body.get("choices", [])
    choice = choices[0] if choices else {}
    message = choice.get("message", {})
    text = message.get("content") or ""

    # Token usage
    usage_oai = openai_body.get("usage", {}) or {}
    input_tokens = usage_oai.get("prompt_tokens", 0)
    output_tokens = usage_oai.get("completion_tokens", 0)

    finish_reason = choice.get("finish_reason")
    stop_reason = _map_stop_reason(finish_reason)

    return {
        "id": f"msg_{openai_body.get('id', uuid.uuid4().hex)}",
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


def _map_stop_reason(openai_finish: Optional[str]) -> Optional[str]:
    if openai_finish is None:
        return None
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "content_filter": "end_turn",
    }
    return mapping.get(openai_finish, "end_turn")


# ─────────────────────────────────────────────────────────────────────────────
# Streaming translation: OpenAI SSE -> Anthropic SSE
# ─────────────────────────────────────────────────────────────────────────────

async def translate_stream_openai_to_anthropic(
    openai_chunks: AsyncIterator[bytes],
    *,
    requested_model: str,
) -> AsyncIterator[bytes]:
    """Consume OpenAI SSE chunks and emit Anthropic SSE events.

    Anthropic event sequence:
      1. message_start   (one)
      2. content_block_start  (index 0, type text)
      3. content_block_delta  (text_delta)   ... many
      4. content_block_stop   (index 0)
      5. message_delta   (stop_reason, usage)
      6. message_stop     (one)
    """
    message_id = f"msg_{uuid.uuid4().hex}"
    # Emit message_start
    yield _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": requested_model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )
    # Emit content_block_start for index 0
    yield _sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    )

    final_stop_reason: Optional[str] = None
    final_usage: Dict[str, int] = {}

    async for raw in openai_chunks:
        # raw may contain multiple SSE events separated by blank lines
        # We need to parse them line-by-line.
        # httpx aiter_raw yields bytes; decode and split on \n
        try:
            text_chunk = raw.decode("utf-8", errors="replace")
        except Exception:
            continue
        # SSE events are separated by "\n\n"
        for event_block in text_chunk.split("\n\n"):
            event_block = event_block.strip()
            if not event_block:
                continue
            # Each block has lines like "data: {...}\n" (and maybe "event:" etc.)
            for line in event_block.splitlines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    continue
                try:
                    evt = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                # OpenAI chat completion chunk shape:
                # {choices: [{delta: {content, role}, finish_reason}], usage?}
                choices = evt.get("choices", [])
                if choices:
                    ch = choices[0]
                    delta = ch.get("delta", {}) or {}
                    content_piece = delta.get("content")
                    if content_piece:
                        yield _sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "text_delta", "text": content_piece},
                            },
                        )
                    fr = ch.get("finish_reason")
                    if fr:
                        final_stop_reason = _map_stop_reason(fr)
                if evt.get("usage"):
                    u = evt["usage"]
                    final_usage = {
                        "input_tokens": u.get("prompt_tokens", 0),
                        "output_tokens": u.get("completion_tokens", 0),
                    }

    # Close content block
    yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    # Message delta with final stop reason and usage
    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": final_stop_reason or "end_turn",
                "stop_sequence": None,
            },
            "usage": final_usage,
        },
    )
    # message_stop
    yield _sse("message_stop", {"type": "message_stop"})


def _sse(event: str, data: Dict[str, Any]) -> bytes:
    """Format a single Anthropic SSE event as bytes."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode(
        "utf-8"
    )
