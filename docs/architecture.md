# Architecture Diagram

```mermaid
graph TD
  user[User]
  ui[Web UI: FastAPI + HTML/JS]
  api[REST: API /api/*]
  strategies[Strategy Engine]
  data[Market Data: Yahoo Finance]
  fetcher[Price Fetcher: data.py]
  mcp[MCP Server: FastMCP]
  client[MCP Client]

  user --> ui
  ui --> api
  api --> fetcher
  fetcher --> data
  api --> strategies

  client --> mcp
  mcp --> strategies

  strategies -->|signals, metrics, equity| api
  strategies -->|tool responses| mcp
```

## Notes

- The Strategy Engine in `src/mcp_quant/strategies.py` is shared by the MCP server and the web UI.
- The web UI is served by FastAPI in `src/mcp_quant/web/app.py` and calls JSON endpoints.
- The MCP server in `src/mcp_quant/mcp_server.py` exposes tools for strategies and backtests.
- Yahoo Finance data is pulled by `src/mcp_quant/data.py` when the API receives a ticker and date range.
