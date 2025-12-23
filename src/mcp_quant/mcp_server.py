from __future__ import annotations

from datetime import date, timedelta
import re
from typing import Dict, List, Optional, Tuple

from mcp.server.fastmcp import FastMCP


from .data import fetch_yahoo_prices as yahoo_fetch
from .strategies import (
    StrategySpec,
    backtest,
    generate_signals,
    list_strategies as list_specs,
    sample_prices,
    validate_prices,
)


mcp = FastMCP("quant-strategies")


def _parse_iso_date(value: str) -> date | None:
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    if cleaned in ("today", "now"):
        return date.today()
    try:
        return date.fromisoformat(cleaned)
    except ValueError:
        return None


def _range_to_timedelta(value: str) -> timedelta | None:
    text = value.strip().lower()
    if not text:
        return None
    text = text.replace("one", "1")
    match = re.search(
        r"(\d+)\s*(years?|yrs?|y|months?|mos?|mo|m|weeks?|wks?|wk|w|days?|d)",
        text,
    )
    if not match:
        match = re.search(r"(\d+)(y|mo|m|w|wk|d)", text)
    if match:
        number = int(match.group(1))
        unit = match.group(2)
    elif "year" in text:
        number, unit = 1, "year"
    elif "month" in text:
        number, unit = 1, "month"
    elif "week" in text:
        number, unit = 1, "week"
    elif "day" in text:
        number, unit = 1, "day"
    else:
        return None
    if unit.startswith("y"):
        days = 365 * number
    elif unit.startswith("mo") or unit == "m":
        days = 30 * number
    elif unit.startswith("w"):
        days = 7 * number
    else:
        days = number
    return timedelta(days=days)


def _resolve_date_range(
    start_date: str | None,
    end_date: str | None,
    range_value: str | None,
) -> Tuple[date, date]:
    today = date.today()
    end = today
    if end_date:
        parsed_end = _parse_iso_date(end_date)
        if not parsed_end:
            raise ValueError("End date must be YYYY-MM-DD or 'today'")
        end = min(parsed_end, today)
    if range_value:
        delta = _range_to_timedelta(range_value)
        if not delta:
            raise ValueError("Range must look like 1y, 6mo, 30d, or 'last one year'")
        return end - delta, end
    if start_date:
        parsed_start = _parse_iso_date(start_date)
        if parsed_start:
            if parsed_start > end:
                raise ValueError("Start date must be before end date")
            return parsed_start, end
        delta = _range_to_timedelta(start_date)
        if delta:
            return end - delta, end
        raise ValueError("Start date must be YYYY-MM-DD or a relative range")
    raise ValueError("Provide start_date or range")


@mcp.tool()
def list_strategies() -> List[Dict[str, object]]:
    """Return available strategies and their default parameters."""
    specs = list_specs()
    return [
        {"name": spec.name, "description": spec.description, "params": spec.params}
        for spec in specs
    ]


@mcp.tool()
def sample_price_series(
    length: int = 240,
    start: float = 100.0,
    drift: float = 0.0005,
    vol: float = 0.01,
    seed: Optional[int] = 7,
) -> List[float]:
    """Generate a synthetic price series for quick experimentation."""
    return sample_prices(length=length, start=start, drift=drift, vol=vol, seed=seed)


@mcp.tool()
def fetch_yahoo_prices(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    range: str | None = None,
) -> List[float]:
    """Fetch daily closing prices from Yahoo Finance for a ticker and date range."""
    start, end = _resolve_date_range(start_date, end_date, range)
    return yahoo_fetch(ticker, start, end)


@mcp.tool()
def run_backtest(
    prices: List[float],
    strategy: str,
    params: Optional[Dict[str, float]] = None,
    start_cash: float = 10_000.0,
    fee_bps: float = 1.0,
) -> Dict[str, object]:
    """Run a simple long/flat backtest and return equity and metrics."""
    cleaned = validate_prices(prices)
    signals = generate_signals(cleaned, strategy, params)
    result = backtest(cleaned, signals, start_cash=start_cash, fee_bps=fee_bps)
    return {
        "prices": cleaned,
        "signals": signals,
        **result,
    }


@mcp.tool()
def get_strategy_schema(name: str) -> Dict[str, object]:
    """Return the default parameters for a named strategy."""
    specs = {spec.name: spec for spec in list_specs()}
    if name not in specs:
        raise ValueError(f"Unknown strategy: {name}")
    spec: StrategySpec = specs[name]
    return {"name": spec.name, "description": spec.description, "params": spec.params}


if __name__ == "__main__":
    mcp.run()
