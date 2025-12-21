# MCP Quant Strategies

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

## Testing

Run the test suite with the built-in unittest runner:

```bash
python -m unittest discover -s tests
```

Run a single test module:

```bash
python -m unittest tests/test_strategies.py
```

## Notes

- The backtest is intentionally simple: long/flat only, single asset, close-to-close returns.
