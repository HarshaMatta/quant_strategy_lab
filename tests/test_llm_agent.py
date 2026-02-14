"""Unit tests for LLM agent functionality."""
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_quant.llm_agent import (
    AgentState,
    LLMConfigError,
    LLMResponseError,
    _extract_json,
    _llm_config,
    run_llm_agent,
)


def test_extract_json_valid():
    """Test JSON extraction from valid response."""
    text = '{"tool": "list_strategies", "arguments": {}}'
    result = _extract_json(text)
    assert result == {"tool": "list_strategies", "arguments": {}}


def test_extract_json_with_code_fence():
    """Test JSON extraction from code fence."""
    text = '```json\n{"tool": "list_strategies", "arguments": {}}\n```'
    result = _extract_json(text)
    assert result == {"tool": "list_strategies", "arguments": {}}


def test_extract_json_with_surrounding_text():
    """Test JSON extraction with surrounding text."""
    text = 'Here is the response: {"final": "Done"} and that is all.'
    result = _extract_json(text)
    assert result == {"final": "Done"}


def test_extract_json_missing():
    """Test JSON extraction error when no JSON found."""
    text = "This is plain text with no JSON"
    with pytest.raises(LLMResponseError, match="did not include JSON"):
        _extract_json(text)


def test_extract_json_invalid():
    """Test JSON extraction error for invalid JSON."""
    text = '{"tool": "test", invalid}'
    with pytest.raises(LLMResponseError, match="Invalid JSON"):
        _extract_json(text)


def test_llm_config_defaults():
    """Test LLM config with defaults."""
    # Need to provide API key to avoid error
    base, model, key = _llm_config(llm_api_key="test-key")
    assert base == "https://api.openai.com"
    assert model == "gpt-4o-mini"
    assert key == "test-key"


def test_llm_config_with_type():
    """Test LLM config with specific type."""
    base, model, key = _llm_config(llm_type="groq", llm_api_key="test-key")
    assert base == "https://api.groq.com/openai"
    assert model == "llama-3.1-70b-versatile"
    assert key == "test-key"


def test_llm_config_custom_override():
    """Test LLM config with custom overrides."""
    base, model, key = _llm_config(
        llm_api_base="https://custom.api.com/",
        llm_model="custom-model",
        llm_api_key="custom-key",
    )
    assert base == "https://custom.api.com"
    assert model == "custom-model"
    assert key == "custom-key"


def test_llm_config_missing_key_openai():
    """Test LLM config error for missing OpenAI API key."""
    # Clear environment variables
    old_keys = {
        "LLM_API_KEY": os.environ.pop("LLM_API_KEY", None),
        "OPENAI_API_KEY": os.environ.pop("OPENAI_API_KEY", None),
    }
    try:
        with pytest.raises(LLMConfigError, match="Set LLM_API_KEY"):
            _llm_config()
    finally:
        # Restore environment
        for key, value in old_keys.items():
            if value is not None:
                os.environ[key] = value


def test_agent_state_add_step():
    """Test AgentState.add_step method."""
    state = AgentState()
    assert len(state.steps) == 0
    assert state.tool_call_count == 0

    state.add_step("test_tool", {"arg": "value"}, {"result": "success"})

    assert len(state.steps) == 1
    assert state.tool_call_count == 1
    assert state.steps[0]["tool"] == "test_tool"
    assert state.steps[0]["arguments"] == {"arg": "value"}


def test_agent_state_update_prices():
    """Test AgentState.update_prices method."""
    state = AgentState()
    assert state.last_prices is None

    # Update with price series
    state.update_prices([100.0, 101.5, 99.8])
    assert state.last_prices == [100.0, 101.5, 99.8]

    # Update with non-price data (should not change last_prices)
    old_prices = state.last_prices
    state.update_prices({"result": "success"})
    assert state.last_prices == old_prices


@pytest.mark.asyncio
async def test_run_llm_agent_with_final_response():
    """Test agent that returns final response without tool calls."""
    mock_response = '{"final": "The strategy list includes SMA Crossover and RSI Reversion."}'

    with patch("mcp_quant.llm_agent._call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_response

        result = await run_llm_agent(
            prompt="What strategies are available?",
            llm_api_key="test-key",
        )

    assert "final" in result
    assert "SMA Crossover" in result["final"]
    assert len(result["steps"]) == 0


@pytest.mark.asyncio
async def test_run_llm_agent_with_tool_calls():
    """Test agent that makes tool calls before final response."""
    responses = [
        '{"tool": "list_strategies", "arguments": {}}',
        '{"final": "Found 3 strategies."}',
    ]

    mock_tool_result = [
        {"name": "sma_crossover", "description": "SMA strategy"},
        {"name": "rsi_reversion", "description": "RSI strategy"},
    ]

    with patch("mcp_quant.llm_agent._call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = responses

        with patch("mcp_quant.llm_agent.mcp_client") as mock_client:
            mock_client.call_mcp_tool = AsyncMock(return_value=mock_tool_result)

            result = await run_llm_agent(
                prompt="List strategies",
                llm_api_key="test-key",
            )

    assert "final" in result
    assert "Found 3 strategies" in result["final"]
    assert len(result["steps"]) == 1
    assert result["steps"][0]["tool"] == "list_strategies"


@pytest.mark.asyncio
async def test_run_llm_agent_max_steps():
    """Test that agent respects max_steps limit."""
    # Always return tool calls, never final
    mock_response = '{"tool": "list_strategies", "arguments": {}}'

    with patch("mcp_quant.llm_agent._call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_response

        with patch("mcp_quant.llm_agent.mcp_client") as mock_client:
            mock_client.call_mcp_tool = AsyncMock(return_value=[])

            result = await run_llm_agent(
                prompt="Test max steps",
                max_steps=2,
                llm_api_key="test-key",
            )

    assert "final" in result
    assert "Max tool steps reached" in result["final"]
    assert len(result["steps"]) == 2


@pytest.mark.asyncio
async def test_run_llm_agent_reuse_prices():
    """Test that agent reuses last_prices for run_backtest."""
    responses = [
        '{"tool": "sample_price_series", "arguments": {}}',
        '{"tool": "run_backtest", "arguments": {"strategy": "sma_crossover"}}',
    ]

    mock_prices = [100.0, 101.0, 102.0]
    mock_backtest_result = {"total_return": 0.15}

    with patch("mcp_quant.llm_agent._call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = responses

        with patch("mcp_quant.llm_agent.mcp_client") as mock_client:
            call_count = 0

            async def mock_tool_call(tool_name, args):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return mock_prices
                else:
                    # Verify prices were injected
                    assert "prices" in args
                    assert args["prices"] == mock_prices
                    return mock_backtest_result

            mock_client.call_mcp_tool = mock_tool_call

            result = await run_llm_agent(
                prompt="Run backtest",
                llm_api_key="test-key",
            )

    assert len(result["steps"]) == 2
    assert result["steps"][1]["arguments"]["strategy"] == "sma_crossover"


@pytest.mark.asyncio
async def test_run_llm_agent_mcp_error():
    """Test agent handling of MCP tool errors."""
    mock_response = '{"tool": "list_strategies", "arguments": {}}'

    with patch("mcp_quant.llm_agent._call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_response

        with patch("mcp_quant.llm_agent.mcp_client") as mock_client:
            from mcp_quant.mcp_client import MCPClientError

            mock_client.call_mcp_tool = AsyncMock(side_effect=MCPClientError("Tool failed"))

            with pytest.raises(LLMResponseError, match="MCP tool error"):
                await run_llm_agent(
                    prompt="Test error",
                    llm_api_key="test-key",
                )


@pytest.mark.asyncio
async def test_run_llm_agent_missing_tool_or_final():
    """Test error when LLM response has neither tool nor final."""
    mock_response = '{"invalid": "response"}'

    with patch("mcp_quant.llm_agent._call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_response

        with pytest.raises(LLMResponseError, match="missing 'tool' or 'final' field"):
            await run_llm_agent(
                prompt="Test invalid response",
                llm_api_key="test-key",
            )
