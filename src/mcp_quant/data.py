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
    cleaned = (ticker or "").strip().upper()
    if not cleaned:
        raise ValueError("Ticker is required")
    if start_date >= end_date:
        raise ValueError("Start date must be before end date")
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
        raise ValueError(message)
    results = chart.get("result")
    if not results:
        raise ValueError("No data returned from Yahoo Finance")
    indicators = results[0].get("indicators", {})
    adjclose = indicators.get("adjclose") or []
    if adjclose and adjclose[0].get("adjclose"):
        series = adjclose[0].get("adjclose")
    else:
        quote = indicators.get("quote") or []
        series = quote[0].get("close") if quote else []
    if not series:
        raise ValueError("No price data returned from Yahoo Finance")
    return validate_prices(series)
