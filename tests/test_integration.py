"""Integration tests for full workflows."""
import os
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_quant.data import fetch_yahoo_prices
from mcp_quant.llm_agent import run_llm_agent
from mcp_quant.manual_client import manual_client


@pytest.mark.asyncio
async def test_manual_mode_full_flow():
    """Test Manual Mode: list strategies → sample prices → run backtest."""
    # Mock strategies
    mock_strategies = [
        {"name": "sma_crossover", "description": "SMA strategy", "params": {}},
        {"name": "rsi_reversion", "description": "RSI strategy", "params": {}},
    ]

    # Mock price series
    mock_prices = [100.0, 101.0, 99.5, 102.0, 103.5]

    # Mock backtest result
    mock_backtest = {
        "total_return": 0.12,
        "sharpe_ratio": 1.5,
        "max_drawdown": -0.03,
        "num_trades": 5,
    }

    with patch("mcp_quant.manual_client.mcp_client") as mock_client:

        async def mock_call_tool(tool_name, args=None):
            if tool_name == "list_strategies":
                return mock_strategies
            elif tool_name == "sample_price_series":
                return mock_prices
            elif tool_name == "run_backtest":
                return mock_backtest
            raise ValueError(f"Unknown tool: {tool_name}")

        mock_client.call_mcp_tool = mock_call_tool

        # Step 1: List strategies
        strategies = await manual_client.list_strategies()
        assert len(strategies) == 2
        assert strategies[0]["name"] == "sma_crossover"

        # Step 2: Get sample prices
        prices = await manual_client.sample_price_series()
        assert len(prices) == 5
        assert prices[0] == 100.0

        # Step 3: Run backtest
        result = await manual_client.run_backtest(
            prices=prices,
            strategy="sma_crossover",
            params={"fast_window": 10, "slow_window": 30},
            start_cash=10_000.0,
            fee_bps=1.0,
        )

        assert result["total_return"] == 0.12
        assert result["sharpe_ratio"] == 1.5
        assert result["num_trades"] == 5


@pytest.mark.asyncio
async def test_manual_mode_with_yahoo_data():
    """Test Manual Mode with real Yahoo Finance data (mocked)."""
    mock_prices = [150.0, 151.5, 149.8, 152.3, 153.0]
    mock_backtest = {"total_return": 0.08}

    # Mock Yahoo Finance fetch
    with patch("mcp_quant.data.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read = MagicMock(
            return_value=b'''{
            "chart": {
                "result": [{
                    "indicators": {
                        "adjclose": [{
                            "adjclose": [150.0, 151.5, 149.8, 152.3, 153.0]
                        }]
                    }
                }]
            }
        }'''
        )
        mock_urlopen.return_value = mock_response

        prices = fetch_yahoo_prices("AAPL", date(2023, 1, 1), date(2023, 1, 5))

    assert len(prices) == 5
    assert prices[0] == 150.0

    # Run backtest with fetched prices
    with patch("mcp_quant.manual_client.mcp_client") as mock_client:
        mock_client.call_mcp_tool = AsyncMock(return_value=mock_backtest)

        result = await manual_client.run_backtest(
            prices=prices,
            strategy="channel_breakout",
            params={"lookback": 20},
            start_cash=10_000.0,
            fee_bps=1.0,
        )

    assert result["total_return"] == 0.08


@pytest.mark.asyncio
async def test_llm_mode_simple_query():
    """Test LLM Mode with simple query (skip if no API key)."""
    # Skip if no API key
    if not os.getenv("LLM_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        pytest.skip("No LLM API key available")

    llm_responses = [
        '{"tool": "list_strategies", "arguments": {}}',
        '{"final": "There are 3 strategies available: sma_crossover, rsi_reversion, and channel_breakout."}',
    ]

    mock_strategies = [
        {"name": "sma_crossover"},
        {"name": "rsi_reversion"},
        {"name": "channel_breakout"},
    ]

    with patch("mcp_quant.llm_agent._call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = llm_responses

        with patch("mcp_quant.llm_agent.mcp_client") as mock_client:
            mock_client.call_mcp_tool = AsyncMock(return_value=mock_strategies)

            result = await run_llm_agent(
                prompt="What strategies are available?",
                llm_api_key="test-key",
            )

    assert "final" in result
    assert "3 strategies" in result["final"]
    assert len(result["steps"]) == 1


@pytest.mark.asyncio
async def test_error_scenario_invalid_strategy():
    """Test error handling for invalid strategy name."""
    with patch("mcp_quant.manual_client.mcp_client") as mock_client:
        from mcp_quant.mcp_client import MCPClientError

        mock_client.call_mcp_tool = AsyncMock(
            side_effect=MCPClientError("Unknown strategy: invalid_strategy")
        )

        with pytest.raises(MCPClientError, match="Unknown strategy"):
            await manual_client.run_backtest(
                prices=[100.0, 101.0],
                strategy="invalid_strategy",
                params={},
                start_cash=10_000.0,
                fee_bps=1.0,
            )


@pytest.mark.asyncio
async def test_error_scenario_invalid_ticker():
    """Test error handling for invalid ticker in Yahoo Finance."""
    with pytest.raises(ValueError, match="Ticker is required"):
        fetch_yahoo_prices("", date(2023, 1, 1), date(2023, 12, 31))


@pytest.mark.asyncio
async def test_error_scenario_negative_cash():
    """Test error handling for negative start_cash."""
    # This would be caught by Pydantic validation in the web app
    # Here we test the underlying logic still works with any value
    with patch("mcp_quant.manual_client.mcp_client") as mock_client:
        mock_client.call_mcp_tool = AsyncMock(return_value={"total_return": 0.0})

        # Should still call the tool (validation happens at API layer)
        result = await manual_client.run_backtest(
            prices=[100.0, 101.0],
            strategy="sma_crossover",
            params={},
            start_cash=-1000.0,  # Invalid but not validated here
            fee_bps=1.0,
        )

        # The call went through (API validation would have caught this)
        assert result is not None


@pytest.mark.asyncio
async def test_llm_mode_multi_step_workflow():
    """Test LLM Mode with multi-step workflow: fetch data → run backtest."""
    llm_responses = [
        '{"tool": "fetch_yahoo_prices", "arguments": {"ticker": "AAPL", "range": "1y"}}',
        '{"tool": "run_backtest", "arguments": {"strategy": "sma_crossover", "params": {}}}',
    ]

    mock_prices = [100.0, 101.0, 102.0, 103.0]
    mock_backtest = {"total_return": 0.15, "sharpe_ratio": 1.8}

    with patch("mcp_quant.llm_agent._call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = llm_responses

        with patch("mcp_quant.llm_agent.mcp_client") as mock_client:
            call_count = 0

            async def mock_tool_call(tool_name, args):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return mock_prices
                else:
                    return mock_backtest

            mock_client.call_mcp_tool = mock_tool_call

            result = await run_llm_agent(
                prompt="Backtest SMA crossover on AAPL",
                llm_api_key="test-key",
            )

    assert len(result["steps"]) == 2
    assert result["steps"][0]["tool"] == "fetch_yahoo_prices"
    assert result["steps"][1]["tool"] == "run_backtest"
    assert result["steps"][1]["result"]["total_return"] == 0.15


@pytest.mark.asyncio
async def test_price_reuse_between_tools():
    """Test that prices are automatically reused between tool calls."""
    # Note: Agent stops after first run_backtest, so we test with get_strategy_schema instead
    llm_responses = [
        '{"tool": "sample_price_series", "arguments": {}}',
        '{"tool": "get_strategy_schema", "arguments": {"name": "sma_crossover"}}',
        '{"tool": "run_backtest", "arguments": {"strategy": "sma_crossover"}}',
    ]

    mock_prices = [100.0, 101.0, 102.0]
    mock_schema = {"name": "sma_crossover", "params": {}}
    mock_backtest = {"total_return": 0.10}

    with patch("mcp_quant.llm_agent._call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = llm_responses

        with patch("mcp_quant.llm_agent.mcp_client") as mock_client:
            call_count = 0

            async def mock_tool_call(tool_name, args):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return mock_prices
                elif call_count == 2:
                    return mock_schema
                else:
                    # Verify prices were injected
                    assert args.get("prices") == mock_prices
                    return mock_backtest

            mock_client.call_mcp_tool = mock_tool_call

            result = await run_llm_agent(
                prompt="Run backtest with schema",
                max_steps=3,
                llm_api_key="test-key",
            )

    # Agent stops after run_backtest
    assert len(result["steps"]) == 3
    # Prices should have been reused for the backtest
    assert result["steps"][2]["arguments"]["strategy"] == "sma_crossover"
