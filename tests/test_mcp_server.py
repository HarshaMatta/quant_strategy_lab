"""Unit tests for MCP server tools and utilities."""
from datetime import date, timedelta
from unittest.mock import patch

import pytest

from mcp_quant.mcp_server import (
    _parse_iso_date,
    _range_to_timedelta,
    _resolve_date_range,
    fetch_yahoo_prices,
    get_strategy_schema,
    list_strategies,
    run_backtest,
    sample_price_series,
)


# Date parsing utility tests


def test_parse_iso_date_valid():
    """Test parsing valid ISO date."""
    result = _parse_iso_date("2023-06-15")
    assert result == date(2023, 6, 15)


def test_parse_iso_date_today():
    """Test parsing 'today' keyword."""
    result = _parse_iso_date("today")
    assert result == date.today()

    result = _parse_iso_date("now")
    assert result == date.today()


def test_parse_iso_date_empty():
    """Test parsing empty string."""
    result = _parse_iso_date("")
    assert result is None

    result = _parse_iso_date("   ")
    assert result is None


def test_parse_iso_date_invalid():
    """Test parsing invalid date."""
    result = _parse_iso_date("not-a-date")
    assert result is None


def test_range_to_timedelta_years():
    """Test parsing year ranges."""
    assert _range_to_timedelta("1y") == timedelta(days=365)
    assert _range_to_timedelta("2 years") == timedelta(days=730)
    assert _range_to_timedelta("one year") == timedelta(days=365)


def test_range_to_timedelta_months():
    """Test parsing month ranges."""
    assert _range_to_timedelta("1mo") == timedelta(days=30)
    assert _range_to_timedelta("6 months") == timedelta(days=180)
    assert _range_to_timedelta("3m") == timedelta(days=90)


def test_range_to_timedelta_weeks():
    """Test parsing week ranges."""
    assert _range_to_timedelta("1w") == timedelta(days=7)
    assert _range_to_timedelta("2 weeks") == timedelta(days=14)


def test_range_to_timedelta_days():
    """Test parsing day ranges."""
    assert _range_to_timedelta("30d") == timedelta(days=30)
    assert _range_to_timedelta("90 days") == timedelta(days=90)


def test_range_to_timedelta_invalid():
    """Test invalid range format."""
    assert _range_to_timedelta("invalid") is None
    assert _range_to_timedelta("") is None


def test_resolve_date_range_with_range():
    """Test resolving date range with 'range' parameter."""
    start, end = _resolve_date_range(None, None, "1y")
    assert end == date.today()
    assert start == end - timedelta(days=365)


def test_resolve_date_range_with_start_and_end():
    """Test resolving date range with explicit start and end."""
    start, end = _resolve_date_range("2023-01-01", "2023-12-31", None)
    assert start == date(2023, 1, 1)
    assert end == date(2023, 12, 31)


def test_resolve_date_range_invalid_order():
    """Test error when start date is after end date."""
    with pytest.raises(ValueError, match="Start date must be before end date"):
        _resolve_date_range("2023-12-31", "2023-01-01", None)


def test_resolve_date_range_missing_params():
    """Test error when neither range nor start_date provided."""
    with pytest.raises(ValueError, match="Provide start_date or range"):
        _resolve_date_range(None, None, None)


def test_resolve_date_range_invalid_end():
    """Test error for invalid end date format."""
    with pytest.raises(ValueError, match="End date must be"):
        _resolve_date_range(None, "invalid-date", None)


def test_resolve_date_range_invalid_range():
    """Test error for invalid range format."""
    with pytest.raises(ValueError, match="Range must look like"):
        _resolve_date_range(None, None, "invalid-range")


# MCP tool tests


def test_list_strategies():
    """Test list_strategies tool."""
    strategies = list_strategies()

    assert isinstance(strategies, list)
    assert len(strategies) >= 3  # At least sma_crossover, rsi_reversion, channel_breakout

    # Check structure
    for strategy in strategies:
        assert "name" in strategy
        assert "description" in strategy
        assert "params" in strategy

    # Check specific strategies exist
    strategy_names = {s["name"] for s in strategies}
    assert "sma_crossover" in strategy_names
    assert "rsi_reversion" in strategy_names
    assert "channel_breakout" in strategy_names


def test_sample_price_series_default():
    """Test sample_price_series with default parameters."""
    prices = sample_price_series()

    assert isinstance(prices, list)
    assert len(prices) == 240  # Default length
    assert all(isinstance(p, (int, float)) for p in prices)
    assert prices[0] == 100.0  # Default start


def test_sample_price_series_custom():
    """Test sample_price_series with custom parameters."""
    prices = sample_price_series(length=100, start=200.0, drift=0.001, vol=0.02, seed=42)

    assert len(prices) == 100
    assert prices[0] == 200.0


def test_sample_price_series_reproducible():
    """Test that sample_price_series is reproducible with same seed."""
    prices1 = sample_price_series(seed=42)
    prices2 = sample_price_series(seed=42)

    assert prices1 == prices2


def test_fetch_yahoo_prices_tool():
    """Test fetch_yahoo_prices MCP tool (mocked)."""
    mock_prices = [100.0, 101.5, 99.8, 102.3]

    with patch("mcp_quant.mcp_server.yahoo_fetch", return_value=mock_prices):
        prices = fetch_yahoo_prices(ticker="AAPL", range="1mo")

    assert prices == mock_prices


def test_fetch_yahoo_prices_with_dates():
    """Test fetch_yahoo_prices with explicit dates (mocked)."""
    mock_prices = [150.0, 151.0, 152.0]

    with patch("mcp_quant.mcp_server.yahoo_fetch", return_value=mock_prices):
        prices = fetch_yahoo_prices(
            ticker="GOOGL",
            start_date="2023-01-01",
            end_date="2023-01-03",
        )

    assert prices == mock_prices


def test_run_backtest_tool():
    """Test run_backtest MCP tool."""
    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    result = run_backtest(
        prices=prices,
        strategy="sma_crossover",
        params={"fast_window": 2, "slow_window": 3},
        start_cash=10_000.0,
        fee_bps=1.0,
    )

    assert isinstance(result, dict)
    assert "prices" in result
    assert "signals" in result

    # Metrics are in a nested dict
    assert "metrics" in result
    assert "total_return" in result["metrics"]
    assert "sharpe" in result["metrics"]
    assert "max_drawdown" in result["metrics"]

    # Verify prices are included
    assert result["prices"] == prices


def test_run_backtest_with_default_params():
    """Test run_backtest with default parameters."""
    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    result = run_backtest(
        prices=prices,
        strategy="channel_breakout",
        params=None,  # Use defaults
    )

    assert isinstance(result, dict)
    assert "metrics" in result
    assert "total_return" in result["metrics"]


def test_run_backtest_invalid_prices():
    """Test run_backtest with invalid prices."""
    with pytest.raises(ValueError):
        run_backtest(
            prices=[100.0, None, 102.0],  # Contains None
            strategy="sma_crossover",
        )


def test_run_backtest_unknown_strategy():
    """Test run_backtest with unknown strategy."""
    with pytest.raises(ValueError, match="Unknown strategy"):
        run_backtest(
            prices=[100.0, 101.0, 102.0, 103.0, 104.0, 105.0],  # Need at least 5 prices
            strategy="nonexistent_strategy",
        )


def test_get_strategy_schema_valid():
    """Test get_strategy_schema for valid strategy."""
    schema = get_strategy_schema("sma_crossover")

    assert schema["name"] == "sma_crossover"
    assert "description" in schema
    assert "params" in schema
    assert isinstance(schema["params"], dict)


def test_get_strategy_schema_all_strategies():
    """Test get_strategy_schema for all available strategies."""
    strategies = list_strategies()

    for strategy in strategies:
        schema = get_strategy_schema(strategy["name"])
        assert schema["name"] == strategy["name"]
        assert schema["description"] == strategy["description"]


def test_get_strategy_schema_unknown():
    """Test get_strategy_schema for unknown strategy."""
    with pytest.raises(ValueError, match="Unknown strategy"):
        get_strategy_schema("nonexistent_strategy")


def test_fetch_yahoo_prices_date_parsing():
    """Test that fetch_yahoo_prices correctly parses various date formats."""
    mock_prices = [100.0, 101.0]

    with patch("mcp_quant.mcp_server.yahoo_fetch", return_value=mock_prices) as mock_fetch:
        # Test with range
        fetch_yahoo_prices(ticker="AAPL", range="1mo")
        call_args = mock_fetch.call_args
        start, end = call_args[0][1], call_args[0][2]
        assert isinstance(start, date)
        assert isinstance(end, date)
        assert end - start == timedelta(days=30)


def test_run_backtest_fee_impact():
    """Test that fees affect backtest results."""
    prices = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0]

    # Run with no fees
    result_no_fee = run_backtest(
        prices=prices,
        strategy="channel_breakout",
        params={"lookback": 2},
        fee_bps=0.0,
    )

    # Run with fees
    result_with_fee = run_backtest(
        prices=prices,
        strategy="channel_breakout",
        params={"lookback": 2},
        fee_bps=10.0,
    )

    # Check if any trades occurred (positions changed)
    positions = result_no_fee.get("positions", [])
    num_trades = sum(1 for i in range(1, len(positions)) if positions[i] != positions[i - 1])

    # With fees should have lower return if there were trades
    if num_trades > 0:
        assert result_with_fee["metrics"]["total_return"] < result_no_fee["metrics"]["total_return"]


def test_sample_price_series_no_seed():
    """Test that sample_price_series without seed produces different results."""
    prices1 = sample_price_series(seed=None)
    prices2 = sample_price_series(seed=None)

    # Should be different (very unlikely to be identical)
    assert prices1 != prices2


def test_resolve_date_range_end_date_in_future():
    """Test that end_date is capped at today."""
    future_date = (date.today() + timedelta(days=30)).isoformat()
    start, end = _resolve_date_range("2023-01-01", future_date, None)

    assert end == date.today()  # Should be capped
    assert start == date(2023, 1, 1)


def test_resolve_date_range_relative_start():
    """Test resolving with relative start date (like '30d')."""
    start, end = _resolve_date_range("30d", None, None)

    assert end == date.today()
    assert start == end - timedelta(days=30)
