# LLM Gateway

A FastAPI-based proxy that exposes a single OpenAI-compatible API and routes
requests through a configurable **waterfall** of providers, models, and API
keys. Built to keep coding agents (opencode, Cline, Continue, etc.) running
even when one key gets rate-limited or one provider goes down.

## Features

- **OpenAI-compatible endpoints**
  - `POST /v1/chat/completions` (streaming + non-streaming)
  - `POST /v1/completions` (legacy)
  - `POST /v1/embeddings`
  - `GET  /v1/models`
  - `POST /v1/messages` (Anthropic-style, translated to OpenAI internally)
- **Waterfall fallback** with configurable order and cycles
  - provider1 → model1 (key1, key2, …) → model2 (key1, key2, …) → provider2 → … → cycle restarts
  - `waterfall_max_cycles` controls how many full cycles to attempt
  - Per-key exponential backoff (`key_cooldown_seconds * 2^(n-1)`, capped at 60×)
  - Cooldowns reset at the start of each new cycle
- **Async throughout** — `httpx.AsyncClient` for upstream calls; no blocking I/O
- **Streaming (SSE) support** — when a stream opens successfully, the response
  is streamed back to the client. HTTP errors before the stream opens trigger
  the next key/model/provider. Mid-stream errors propagate to the client.
- **Hot-reload** — edit `config.yaml` and the proxy picks up new providers /
  keys / waterfall order without restarting. Cooldowns for unchanged keys are
  preserved.
- **Observability** — structured stdout logs + `logs/requests.jsonl` with one
  record per upstream attempt (provider, model, key index, latency, status,
  error).
- **Admin endpoints** — `GET /health`, `GET /status`, `POST /admin/reload`
- **User-sent `model` is ignored** — the waterfall decides which model is used.

## Project setup (uv)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. Install it first:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then clone and set up the project:

```bash
git clone https://github.com/shahedmomenzadeh/LLM-gateway.git
cd LLM-gateway

# Copy the example config and add your API keys
cp config.example.yaml config.yaml
$EDITOR config.yaml

# Install dependencies (creates .venv automatically)
uv sync

# Run the gateway
uv run llm-gateway
# or
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

## Configuring coding agents

Point your agent at the gateway. Examples:

**opencode / generic OpenAI client:**
```
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=dummy   # the proxy ignores this
```

**Cline / Continue:**
- Base URL: `http://localhost:8000/v1`
- API Key: any non-empty string
- Model: any string (the proxy ignores it — the waterfall picks)

**Anthropic SDK / Claude Code (Anthropic-compatible):**
```
ANTHROPIC_BASE_URL=http://localhost:8000
ANTHROPIC_API_KEY=dummy
```

## Using from WSL on Windows

If you run the gateway inside **WSL** and want to access it from **Windows** (or
other machines on your LAN), `localhost` won't work — WSL has its own network
stack. Use the WSL instance's IP address instead.

**Find your WSL IP:**

```bash
# Inside WSL
hostname -I
# e.g. 172.28.13.246
```

Then point your coding agents at that IP:

```
OPENAI_BASE_URL=http://172.28.13.246:8000/v1
OPENAI_API_KEY=dummy
```

Or for Anthropic-compatible tools:

```
ANTHROPIC_BASE_URL=http://172.28.13.246:8000
ANTHROPIC_API_KEY=dummy
```

> **Note:** The WSL IP can change on reboot. If it changes, update your
> agent's config accordingly. The gateway binds to `0.0.0.0` by default, so it
> is reachable from outside WSL as long as the firewall allows port 8000.

## Config file

See `config.yaml` for the full sample. Key sections:

```yaml
gateway:
  rate_limit_rpm: 0            # 0 = disabled
  key_cooldown_seconds: 20     # base for exponential backoff
  timeout: 60.0
  waterfall_max_cycles: 10
  # ssl_certfile: certs/cert.pem
  # ssl_keyfile:  certs/key.pem
  host: 0.0.0.0
  port: 8000

providers:
  openai:
    base_url: https://api.openai.com/v1
    timeout: 60.0
    max_retries: 1             # informational; the waterfall handles retries
    api_keys: [sk-..., sk-...]
    extra_headers: {}          # optional custom headers

waterfall:
  - provider: openai
    models: [gpt-4o, gpt-4o-mini]
  - provider: together
    models: [meta-llama/...]
```

## How the waterfall works

For every incoming request:

1. **Cycle 1** begins. All key cooldowns are cleared.
2. For each `waterfall` step (top to bottom):
   - For each `model` in the step (left to right):
     - For each `api_key` of that provider (in order):
       - If the key is in cooldown, skip.
       - Otherwise, forward the request upstream with this `model` and `key`.
       - **On HTTP 4xx/5xx** (or network/timeout error opening the request),
         mark the key as errored (exponential backoff), and try the next key.
       - **On success**, return the response to the client.
3. If all (provider, model, key) combinations in all steps are exhausted,
   start **Cycle 2**: clear cooldowns and retry the whole chain.
4. Repeat until `waterfall_max_cycles` is reached, then return `502
   waterfall_exhausted`.

**Streaming note:** once an upstream stream opens successfully (HTTP 200 +
headers received), the response is committed to that upstream. A mid-stream
failure will surface as an error chunk to the client — the gateway does NOT
retry mid-stream (because the client has already received partial output).

## Endpoints

| Method | Path                  | Description                                  |
|--------|-----------------------|----------------------------------------------|
| POST   | `/v1/chat/completions`| OpenAI chat completions                      |
| POST   | `/v1/completions`     | Legacy text completions                      |
| POST   | `/v1/embeddings`      | Embeddings (provider must support the model) |
| GET    | `/v1/models`          | List models from the waterfall config        |
| POST   | `/v1/messages`        | Anthropic-style messages (translated)        |
| GET    | `/health`             | Liveness probe                               |
| GET    | `/status`             | Full runtime state snapshot                  |
| POST   | `/admin/reload`       | Manually reload config                       |
| GET    | `/`                   | Service info                                 |

## Logs

- **stdout** — human-readable, one line per event.
- **`logs/requests.jsonl`** — one JSON line per upstream attempt:

```json
{"ts": 1718880000.0, "request_id": "abc123", "provider": "openai", "model": "gpt-4o", "key_index": 0, "status": 429, "latency_ms": 120.5, "error": "upstream returned 429", "stream": false, "cycle": 1, "step_index": 0, "model_index": 0}
```

## Hot reload

Just edit `config.yaml` and save. The watcher detects the change, validates
the new config, and atomically swaps it in. Cooldowns for keys that survived
the reload are preserved. If validation fails, the old config keeps running
and the error is logged.

You can also trigger a reload manually:

```bash
curl -X POST http://localhost:8000/admin/reload
```

## Project layout

```
LLM-gateway/
├── pyproject.toml           # project metadata + dependencies (uv)
├── config.example.yaml      # sample config — copy to config.yaml
├── config.yaml              # your config with real API keys (git-ignored)
├── main.py                  # FastAPI entry point + uvicorn launcher
├── gateway/
│   ├── __init__.py
│   ├── config.py            # Pydantic schema + YAML loader
│   ├── state.py             # async-safe runtime state (cooldowns, counters)
│   ├── client.py            # httpx async upstream client
│   ├── upstream_client.py   # singleton accessor
│   ├── waterfall.py         # waterfall execution engine
│   ├── anthropic.py         # Anthropic <-> OpenAI translation
│   ├── routes.py            # /v1/* API routes
│   ├── admin.py             # /health, /status, /admin/reload
│   ├── watcher.py           # config hot-reload watcher
│   └── logging_setup.py     # structured logs + JSONL request log
└── logs/
    └── requests.jsonl       # created at runtime (git-ignored)
```

## License

MIT.
