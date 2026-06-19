"""
Waterfall execution engine.

Given a request body and the current state, walks the configured waterfall:
  for each cycle in 1..max_cycles:
    reset all key cooldowns
    for each step (provider, [models...]):
      for each model in step.models:
        for each key in provider.api_keys:
          if key is cooling down, skip
          try the upstream
          on success: return the result
          on UpstreamError: mark key error, mark model error, advance

If all cycles are exhausted, raises WaterfallExhaustedError.

This module exposes two entry points:
  - execute_non_stream(): returns a NonStreamResult
  - execute_stream(): returns a StreamResult (or raises)

Both use the same iteration logic; the difference is which UpstreamClient
method they call.
"""
from __future__ import annotations

import time
import uuid
from typing import AsyncIterator, Optional

from .client import NonStreamResult, StreamResult, UpstreamClient, UpstreamError
from .config import Config
from .logging_setup import get_logger, log_request_attempt
from .state import State

log = get_logger("gateway.waterfall")


class WaterfallExhaustedError(Exception):
    """All keys/models/providers/cycles exhausted."""

    def __init__(self, attempts: list[dict]) -> None:
        super().__init__(
            f"waterfall exhausted after {len(attempts)} attempts across all cycles"
        )
        self.attempts = attempts


async def _try_one(
    *,
    client: UpstreamClient,
    state: State,
    config: Config,
    provider_name: str,
    model: str,
    key_index: int,
    api_key: str,
    path: str,
    body: dict,
    stream: bool,
    request_id: str,
    cycle: int,
    step_index: int,
    model_index: int,
) -> NonStreamResult | StreamResult:
    """Try a single (provider, model, key) combination."""
    pcfg = config.providers[provider_name]
    # Inject the waterfall model into the body (user-sent model is ignored)
    body_to_send = dict(body)
    body_to_send["model"] = model

    t0 = time.perf_counter()
    status = 0
    error_msg: Optional[str] = None
    try:
        if stream:
            result = await client.post_stream(pcfg, path, body_to_send, api_key)
            status = result.status
            return result
        else:
            result = await client.post_non_stream(pcfg, path, body_to_send, api_key)
            status = result.status
            return result
    except UpstreamError as e:
        status = e.status_code or 0
        error_msg = str(e)
        # Mark the key as errored with exponential backoff (tags with request_id
        # so concurrent requests are protected but THIS request can still try
        # the same key against other models in the same cycle).
        cooldown_base = config.gateway.key_cooldown_seconds
        await state.mark_key_error(provider_name, key_index, cooldown_base, request_id)
        await state.mark_model_attempt(provider_name, model, error=True)
        await state.record_fallback()
        log.warning(
            f"upstream error provider={provider_name} model={model} key_idx={key_index} "
            f"status={status} err={error_msg} — advancing waterfall",
            extra={
                "request_id": request_id,
                "provider": provider_name,
                "model": model,
                "key_index": key_index,
                "status": status,
                "cycle": cycle,
            },
        )
        raise
    finally:
        latency_ms = (time.perf_counter() - t0) * 1000
        await log_request_attempt(
            {
                "ts": time.time(),
                "request_id": request_id,
                "provider": provider_name,
                "model": model,
                "key_index": key_index,
                "status": status,
                "latency_ms": round(latency_ms, 2),
                "error": error_msg,
                "stream": stream,
                "cycle": cycle,
                "step_index": step_index,
                "model_index": model_index,
            }
        )


async def _run_waterfall(
    *,
    client: UpstreamClient,
    state: State,
    config: Config,
    path: str,
    body: dict,
    stream: bool,
) -> NonStreamResult | StreamResult:
    """Run the full waterfall. Returns the first successful result."""
    request_id = str(uuid.uuid4())
    await state.record_request_start()

    max_cycles = config.gateway.waterfall_max_cycles
    attempts: list[dict] = []

    for cycle in range(1, max_cycles + 1):
        await state.record_cycle()
        if cycle > 1:
            log.info(
                f"starting waterfall cycle {cycle}/{max_cycles} (resetting cooldowns for this request)",
                extra={"request_id": request_id, "cycle": cycle},
            )
            # Only reset cooldowns that THIS request set — don't disrupt concurrent requests
            await state.reset_all_cooldowns(request_id)

        for step_index, step in enumerate(config.waterfall):
            provider_name = step.provider
            pcfg = config.providers.get(provider_name)
            if not pcfg:
                # Should never happen — config validation prevents this
                continue

            for model_index, model in enumerate(step.models):
                for key_index, api_key in enumerate(pcfg.api_keys):
                    # Skip keys cooled down by OTHER requests (concurrent protection).
                    # Keys cooled down by THIS request are still tried, so we honor
                    # the spec: "switch to model 2 then again use all the keys".
                    if not await state.is_key_available_for_request(
                        provider_name, key_index, request_id
                    ):
                        log.debug(
                            f"skipping key cooled down by another request "
                            f"provider={provider_name} key_idx={key_index}",
                            extra={
                                "request_id": request_id,
                                "provider": provider_name,
                                "key_index": key_index,
                            },
                        )
                        continue

                    await state.update_position(step_index, model_index, key_index)
                    try:
                        result = await _try_one(
                            client=client,
                            state=state,
                            config=config,
                            provider_name=provider_name,
                            model=model,
                            key_index=key_index,
                            api_key=api_key,
                            path=path,
                            body=body,
                            stream=stream,
                            request_id=request_id,
                            cycle=cycle,
                            step_index=step_index,
                            model_index=model_index,
                        )
                    except UpstreamError as e:
                        attempts.append(
                            {
                                "cycle": cycle,
                                "step_index": step_index,
                                "provider": provider_name,
                                "model": model,
                                "model_index": model_index,
                                "key_index": key_index,
                                "status": e.status_code,
                                "error": str(e),
                            }
                        )
                        continue

                    # Success!
                    await state.mark_key_success(provider_name, key_index)
                    await state.mark_model_attempt(provider_name, model, error=False)
                    log.info(
                        f"request succeeded provider={provider_name} model={model} "
                        f"key_idx={key_index} cycle={cycle} stream={stream}",
                        extra={
                            "request_id": request_id,
                            "provider": provider_name,
                            "model": model,
                            "key_index": key_index,
                            "cycle": cycle,
                        },
                    )
                    return result

    # All cycles exhausted
    await state.record_request_error()
    log.error(
        f"waterfall exhausted request_id={request_id} attempts={len(attempts)}",
        extra={"request_id": request_id, "attempts": len(attempts)},
    )
    raise WaterfallExhaustedError(attempts)


# ── Public API ─────────────────────────────────────────────────────────────

async def execute_non_stream(
    *,
    client: UpstreamClient,
    state: State,
    path: str,
    body: dict,
) -> NonStreamResult:
    config = state.config
    return await _run_waterfall(
        client=client,
        state=state,
        config=config,
        path=path,
        body=body,
        stream=False,
    )  # type: ignore[return-value]


async def execute_stream(
    *,
    client: UpstreamClient,
    state: State,
    path: str,
    body: dict,
) -> StreamResult:
    config = state.config
    return await _run_waterfall(
        client=client,
        state=state,
        config=config,
        path=path,
        body=body,
        stream=True,
    )  # type: ignore[return-value]
