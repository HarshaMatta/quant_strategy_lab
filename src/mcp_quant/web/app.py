from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

from mcp_quant.data import fetch_yahoo_prices
from mcp_quant.llm_agent import LLMConfigError, LLMResponseError, run_llm_agent
from mcp_quant.mcp_client import MCPClientError, mcp_client
from mcp_quant.manual_client import manual_client

logger = logging.getLogger(__name__)


# Rate Limiting


class RateLimiter:
    """Track and enforce rate limits per IP address."""

    def __init__(self, requests_per_minute: int = 30, requests_per_hour: int = 300):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests: Dict[str, List[float]] = defaultdict(list)

    def _cleanup_old_entries(self, ip: str, current_time: float) -> None:
        """Remove request timestamps older than 1 hour."""
        if ip in self.requests:
            self.requests[ip] = [
                ts for ts in self.requests[ip] if current_time - ts < 3600
            ]

    def check_rate_limit(self, ip: str) -> tuple[bool, Optional[str]]:
        """Check if IP is within rate limits.

        Returns:
            (allowed, error_message): allowed is True if under limits, False otherwise
        """
        current_time = time.time()
        self._cleanup_old_entries(ip, current_time)

        timestamps = self.requests[ip]

        # Check minute limit
        recent_minute = [ts for ts in timestamps if current_time - ts < 60]
        if len(recent_minute) >= self.requests_per_minute:
            return False, f"Rate limit exceeded: {self.requests_per_minute} requests/minute"

        # Check hour limit
        recent_hour = [ts for ts in timestamps if current_time - ts < 3600]
        if len(recent_hour) >= self.requests_per_hour:
            return False, f"Rate limit exceeded: {self.requests_per_hour} requests/hour"

        return True, None

    def record_request(self, ip: str) -> None:
        """Record a request timestamp for the IP."""
        self.requests[ip].append(time.time())

    def get_stats(self, ip: str) -> Dict[str, Any]:
        """Get rate limit stats for an IP."""
        current_time = time.time()
        self._cleanup_old_entries(ip, current_time)

        timestamps = self.requests[ip]
        recent_minute = [ts for ts in timestamps if current_time - ts < 60]
        recent_hour = [ts for ts in timestamps if current_time - ts < 3600]

        return {
            "requests_last_minute": len(recent_minute),
            "requests_last_hour": len(recent_hour),
            "limit_per_minute": self.requests_per_minute,
            "limit_per_hour": self.requests_per_hour,
            "remaining_minute": max(0, self.requests_per_minute - len(recent_minute)),
            "remaining_hour": max(0, self.requests_per_hour - len(recent_hour)),
        }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce rate limits on API requests."""

    def __init__(self, app, rate_limiter: RateLimiter):
        super().__init__(app)
        self.rate_limiter = rate_limiter

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for static files and root
        if request.url.path.startswith("/static") or request.url.path == "/":
            return await call_next(request)

        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # Check rate limit
        allowed, error_message = self.rate_limiter.check_rate_limit(client_ip)

        if not allowed:
            logger.warning(f"Rate limit exceeded for IP {client_ip}: {error_message}")
            return JSONResponse(
                status_code=429,
                content={"detail": error_message},
                headers={
                    "X-RateLimit-Limit-Minute": str(self.rate_limiter.requests_per_minute),
                    "X-RateLimit-Limit-Hour": str(self.rate_limiter.requests_per_hour),
                },
            )

        # Record the request
        self.rate_limiter.record_request(client_ip)

        # Process request
        response = await call_next(request)

        # Add rate limit headers to response
        stats = self.rate_limiter.get_stats(client_ip)
        response.headers["X-RateLimit-Limit-Minute"] = str(stats["limit_per_minute"])
        response.headers["X-RateLimit-Limit-Hour"] = str(stats["limit_per_hour"])
        response.headers["X-RateLimit-Remaining-Minute"] = str(stats["remaining_minute"])
        response.headers["X-RateLimit-Remaining-Hour"] = str(stats["remaining_hour"])

        return response


# Initialize rate limiter
rate_limiter = RateLimiter(requests_per_minute=30, requests_per_hour=300)

app = FastAPI(title="Quant Strategy Lab")

# Add rate limiting middleware
app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)
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
    strategy: str = Field(..., min_length=1, max_length=100, description="Strategy name")
    params: Optional[Dict[str, float]] = Field(None, description="Strategy parameters")
    start_cash: float = Field(10_000.0, gt=0, le=1_000_000_000, description="Starting cash")
    fee_bps: float = Field(1.0, ge=0, le=1000, description="Trading fee in basis points")
    ticker: Optional[str] = Field(None, min_length=1, max_length=10, description="Stock ticker")
    start_date: Optional[date] = Field(None, description="Start date for historical data")
    end_date: Optional[date] = Field(None, description="End date for historical data")

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        """Validate strategy name against known strategies."""
        allowed = {"sma_crossover", "rsi_reversion", "channel_breakout"}
        if v not in allowed:
            raise ValueError(f"Unknown strategy '{v}'. Allowed: {', '.join(sorted(allowed))}")
        return v

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: Optional[str]) -> Optional[str]:
        """Validate and normalize ticker symbol."""
        if v is None:
            return v
        cleaned = v.strip().upper()
        if not re.match(r"^[A-Z]{1,10}$", cleaned):
            raise ValueError(f"Invalid ticker format: '{v}'. Must be 1-10 letters.")
        return cleaned

    @field_validator("end_date")
    @classmethod
    def validate_dates(cls, v: Optional[date], info) -> Optional[date]:
        """Validate date range if both dates are provided."""
        if v is None:
            return v
        start_date = info.data.get("start_date")
        if start_date and start_date >= v:
            raise ValueError(f"start_date ({start_date}) must be before end_date ({v})")
        return v

    @field_validator("params")
    @classmethod
    def validate_params(cls, v: Optional[Dict[str, float]], info) -> Optional[Dict[str, float]]:
        """Validate parameter ranges."""
        if v is None:
            return v
        for key, value in v.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Parameter '{key}' must be numeric, got {type(value)}")
            if value < 0 or value > 10_000:
                raise ValueError(f"Parameter '{key}' value {value} out of range [0, 10000]")
        return v


class MCPToolRequest(BaseModel):
    tool_name: str
    arguments: Optional[Dict[str, Any]] = None


class AgentRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10_000, description="User prompt for the agent")
    max_steps: int = Field(3, ge=1, le=10, description="Maximum agent steps")
    temperature: float = Field(0.2, ge=0.0, le=2.0, description="LLM temperature")
    llm_type: Optional[str] = Field(None, max_length=50, description="LLM provider type")
    llm_api_base: Optional[str] = Field(None, max_length=500, description="LLM API base URL")
    llm_model: Optional[str] = Field(None, max_length=100, description="LLM model name")
    llm_api_key: Optional[str] = Field(None, max_length=500, description="LLM API key")

    @field_validator("llm_api_base")
    @classmethod
    def validate_api_base(cls, v: Optional[str]) -> Optional[str]:
        """Validate API base URL format."""
        if v is None:
            return v
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"API base URL must start with http:// or https://: '{v}'")
        if len(v) > 500:
            raise ValueError("API base URL too long")
        return v.rstrip("/")


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
        except (ConnectionError, TimeoutError) as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Yahoo Finance service unavailable for ticker '{payload.ticker}': {exc}"
            ) from exc
        except Exception as exc:
            error_type = type(exc).__name__
            logger.error(f"Unexpected error fetching Yahoo Finance data for {payload.ticker}: {exc}", exc_info=True)
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch Yahoo Finance data for ticker '{payload.ticker}': {error_type}"
            ) from exc
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


@app.get("/api/rate-limit-status")
async def rate_limit_status(request: Request) -> Dict[str, Any]:
    """Get current rate limit status for the client."""
    client_ip = request.client.host if request.client else "unknown"
    stats = rate_limiter.get_stats(client_ip)
    return {
        "ip": client_ip,
        **stats,
    }
