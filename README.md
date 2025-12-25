# Quant Strategy Lab

Lightweight MCP server and a web UI to explore simple quantitative trading strategies.

## Setup

### Using uv

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```
Editable installs keep `mcp_quant` importable without setting `PYTHONPATH`.

### Using pip

```bash
pip install -e .
```

## Run the MCP server

```bash
python -m mcp_quant.mcp_server
```

The server exposes tools:
- `list_strategies`
- `get_strategy_schema`
- `sample_price_series`
- `run_backtest`

## Run the web UI

```bash
uvicorn mcp_quant.web.app:app --reload --port 8000
```

Open `http://localhost:8000` to explore strategies and visualize price and equity curves. You can also fetch daily prices from Yahoo Finance by entering a ticker and date range.

The UI includes two tabs:
- **Quant Strategy Lab** for direct MCP-backed backtests.
- **LLM Agent** for LLM-driven tool selection and backtests.

UI assets live in `src/mcp_quant/web/templates/index.html` and `src/mcp_quant/web/static/`.

### MCP client configuration

The web UI calls the MCP server through the MCP protocol:

- Default: spawns `python -m mcp_quant.mcp_server` over stdio and keeps a persistent session.
- Optional SSE: set `MCP_SERVER_URL` to connect to an MCP server running with SSE transport.
- Optional stdio overrides: set `MCP_SERVER_COMMAND` and `MCP_SERVER_ARGS` to change how the MCP server is launched.

### LLM agent configuration

The **LLM Agent** tab uses an OpenAI-compatible chat API to decide which MCP tools to call.
You can configure LLM type, API base, model, and API key from the UI, or set defaults via environment variables:

- `LLM_API_BASE` (default: `https://api.openai.com`)
- `LLM_MODEL` (default: `gpt-4o-mini`)
- `LLM_API_KEY` (or `OPENAI_API_KEY`)
- `LLM_TIMEOUT` (default: `120`, seconds)

If you want the UI to call MCP tools directly without the LLM, select **Direct MCP tool** in the agent tab.

### Environment variables via `.env`

Create a `.env` file in the repo root:

```
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
LLM_API_BASE=https://api.openai.com
```

Load it before starting the UI:

```bash
set -a
source .env
set +a
uvicorn mcp_quant.web.app:app --reload --port 8000
```

Or pass it directly (requires `uvicorn[standard]`):

```bash
uvicorn mcp_quant.web.app:app --reload --port 8000 --env-file .env
```

## Testing

Run the test suite with the built-in unittest runner:

```bash
python3 -m unittest discover -s tests
```

Run a single test module:

```bash
python3 -m unittest tests/test_strategies.py
```

## Notes

- The backtest is intentionally simple: long/flat only, single asset, close-to-close returns.
