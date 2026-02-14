"""Unit tests for manual mode client."""
from unittest.mock import AsyncMock, patch

import pytest

from mcp_quant.manual_client import manual_client


@pytest.mark.asyncio
async def test_list_strategies():
    """Test list_strategies wrapper."""
    expected_result = [
        {"name": "sma_crossover", "description": "SMA strategy"},
        {"name": "rsi_reversion", "description": "RSI strategy"},
    ]

    with patch("mcp_quant.manual_client.mcp_client") as mock_client:
        mock_client.call_mcp_tool = AsyncMock(return_value=expected_result)

        result = await manual_client.list_strategies()

    assert result == expected_result
    mock_client.call_mcp_tool.assert_called_once_with("list_strategies")


@pytest.mark.asyncio
async def test_sample_price_series():
    """Test sample_price_series wrapper."""
    expected_result = [100.0, 101.5, 99.8, 102.3]

    with patch("mcp_quant.manual_client.mcp_client") as mock_client:
        mock_client.call_mcp_tool = AsyncMock(return_value=expected_result)

        result = await manual_client.sample_price_series()

    assert result == expected_result
    mock_client.call_mcp_tool.assert_called_once_with("sample_price_series", {})


@pytest.mark.asyncio
async def test_run_backtest():
    """Test run_backtest wrapper."""
    prices = [100.0, 101.0, 102.0]
    strategy = "sma_crossover"
    params = {"fast_window": 10, "slow_window": 30}
    start_cash = 10_000.0
    fee_bps = 1.0

    expected_result = {
        "total_return": 0.15,
        "sharpe_ratio": 1.8,
        "max_drawdown": -0.05,
    }

    with patch("mcp_quant.manual_client.mcp_client") as mock_client:
        mock_client.call_mcp_tool = AsyncMock(return_value=expected_result)

        result = await manual_client.run_backtest(
            prices=prices,
            strategy=strategy,
            params=params,
            start_cash=start_cash,
            fee_bps=fee_bps,
        )

    assert result == expected_result
    mock_client.call_mcp_tool.assert_called_once()
    call_args = mock_client.call_mcp_tool.call_args
    assert call_args[0][0] == "run_backtest"
    assert call_args[0][1]["prices"] == prices
    assert call_args[0][1]["strategy"] == strategy
    assert call_args[0][1]["params"] == params


@pytest.mark.asyncio
async def test_run_backtest_with_none_params():
    """Test run_backtest with None params."""
    prices = [100.0, 101.0, 102.0]
    expected_result = {"total_return": 0.10}

    with patch("mcp_quant.manual_client.mcp_client") as mock_client:
        mock_client.call_mcp_tool = AsyncMock(return_value=expected_result)

        result = await manual_client.run_backtest(
            prices=prices,
            strategy="channel_breakout",
            params=None,
            start_cash=5000.0,
            fee_bps=0.5,
        )

    assert result == expected_result
    call_args = mock_client.call_mcp_tool.call_args
    assert call_args[0][1]["params"] is None
