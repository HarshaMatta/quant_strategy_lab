from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
from typing import List
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .strategies import validate_prices


def _to_unix_day(value: date) -> int:
    return int(datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc).timestamp())


def fetch_yahoo_prices(ticker: str, start_date: date, end_date: date) -> List[float]:
    """Fetch historical price data from Yahoo Finance.

    ARCHITECTURAL NOTE - Intentional Duplication:
    This function exists in parallel with the `fetch_yahoo_prices` MCP tool in mcp_server.py.
    This is intentional and serves different use cases:

    1. **Manual Mode (this function)**: When users submit backtests directly via the web UI
       with a ticker, the FastAPI endpoint calls this function directly. This provides a
       synchronous, straightforward path without MCP overhead.

    2. **LLM Mode (MCP tool)**: When the LLM agent needs to fetch market data as part of
       an agentic workflow, it calls the `fetch_yahoo_prices` MCP tool via the MCP server.
       This allows the agent to autonomously gather data during multi-step reasoning.

    The duplication is a deliberate architectural choice to support both:
    - Direct human-driven workflows (Manual Mode)
    - LLM-driven autonomous workflows (LLM Mode with MCP tools)

    Both implementations use the same Yahoo Finance API and validation logic to ensure
    consistent behavior across modes.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        start_date: Start date for price history
        end_date: End date for price history

    Returns:
        List of adjusted close prices, validated and cleaned

    Raises:
        ValueError: If ticker is invalid, date range is invalid, or no data is returned
    """
    cleaned = (ticker or "").strip().upper()
    if not cleaned:
        raise ValueError("Ticker is required for fetching Yahoo Finance prices")
    if start_date >= end_date:
        raise ValueError(f"Start date ({start_date}) must be before end date ({end_date})")
    start_ts = _to_unix_day(start_date)
    end_ts = _to_unix_day(end_date + timedelta(days=1))
    params = urlencode(
        {
            "period1": start_ts,
            "period2": end_ts,
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        }
    )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{cleaned}?{params}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=10) as response:
        payload = json.load(response)
    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        message = error.get("description") or "Yahoo Finance error"
        raise ValueError(f"Yahoo Finance error for ticker '{cleaned}': {message}")
    results = chart.get("result")
    if not results:
        raise ValueError(f"No data returned from Yahoo Finance for ticker '{cleaned}' ({start_date} to {end_date})")
    indicators = results[0].get("indicators", {})
    adjclose = indicators.get("adjclose") or []
    if adjclose and adjclose[0].get("adjclose"):
        series = adjclose[0].get("adjclose")
    else:
        quote = indicators.get("quote") or []
        series = quote[0].get("close") if quote else []
    if not series:
        raise ValueError(f"No price data returned from Yahoo Finance for ticker '{cleaned}' ({start_date} to {end_date})")
    return validate_prices(series)
