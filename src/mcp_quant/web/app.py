from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mcp_quant.data import fetch_yahoo_prices
from mcp_quant.llm_agent import LLMConfigError, LLMResponseError, run_llm_agent
from mcp_quant.mcp_client import MCPClientError, mcp_client
from mcp_quant.manual_client import manual_client


app = FastAPI(title="Quant Strategy Lab")
BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "templates" / "index.html"
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.on_event("startup")
async def startup_mcp() -> None:
    await mcp_client.connect()


@app.on_event("shutdown")
async def shutdown_mcp() -> None:
    await mcp_client.close()


class BacktestRequest(BaseModel):
    strategy: str
    params: Optional[Dict[str, float]] = None
    start_cash: float = 10_000.0
    fee_bps: float = 1.0
    ticker: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class MCPToolRequest(BaseModel):
    tool_name: str
    arguments: Optional[Dict[str, Any]] = None


class AgentRequest(BaseModel):
    prompt: str
    max_steps: int = 3
    temperature: float = 0.2
    llm_type: Optional[str] = None
    llm_api_base: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return INDEX_PATH.read_text(encoding="utf-8")


@app.get("/api/strategies")
async def strategies() -> List[Dict[str, object]]:
    try:
        result = await manual_client.list_strategies()
    except MCPClientError as exc:
        raise HTTPException(status_code=502, detail=f"MCP error: {exc}") from exc
    if not isinstance(result, list):
        raise HTTPException(status_code=502, detail="Invalid MCP response for strategies")
    return result


@app.post("/api/backtest")
async def run_backtest(payload: BacktestRequest) -> Dict[str, object]:
    if payload.ticker and payload.start_date and payload.end_date:
        try:
            prices = fetch_yahoo_prices(payload.ticker, payload.start_date, payload.end_date)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Failed to fetch Yahoo Finance data") from exc
    else:
        try:
            prices = await manual_client.sample_price_series()
        except MCPClientError as exc:
            raise HTTPException(status_code=502, detail=f"MCP error: {exc}") from exc
    try:
        result = await manual_client.run_backtest(
            prices=prices,
            strategy=payload.strategy,
            params=payload.params,
            start_cash=payload.start_cash,
            fee_bps=payload.fee_bps,
        )
    except MCPClientError as exc:
        raise HTTPException(status_code=502, detail=f"MCP error: {exc}") from exc
    if not isinstance(result, dict):
        raise HTTPException(status_code=502, detail="Invalid MCP response for backtest")
    return result




@app.post("/api/agent")
async def run_agent(payload: AgentRequest) -> Dict[str, object]:
    try:
        result = await run_llm_agent(
            payload.prompt,
            max_steps=payload.max_steps,
            temperature=payload.temperature,
            llm_type=payload.llm_type,
            llm_api_base=payload.llm_api_base,
            llm_model=payload.llm_model,
            llm_api_key=payload.llm_api_key,
        )
    except LLMConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMResponseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return result
