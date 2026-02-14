"""Unit tests for MCP client connection, health checks, and retry logic."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_quant.mcp_client import MCPClient, MCPClientError, MAX_RETRIES


@pytest.fixture
def mcp_client():
    """Create a fresh MCPClient instance for each test."""
    return MCPClient()


@pytest.mark.asyncio
async def test_connect_success(mcp_client):
    """Test successful connection to MCP server."""
    with patch("mcp_quant.mcp_client.stdio_client") as mock_stdio:
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
        mock_cm.__aexit__ = AsyncMock()
        mock_stdio.return_value = mock_cm

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock()
        mock_session.initialize = AsyncMock()

        with patch("mcp_quant.mcp_client.ClientSession", return_value=mock_session):
            await mcp_client.connect()

        assert mcp_client._session is not None
        assert mcp_client._is_healthy is True
        mock_session.initialize.assert_called_once()


@pytest.mark.asyncio
async def test_connect_failure(mcp_client):
    """Test connection failure handling."""
    with patch("mcp_quant.mcp_client.stdio_client") as mock_stdio:
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(side_effect=RuntimeError("Connection failed"))
        mock_stdio.return_value = mock_cm

        with pytest.raises(MCPClientError, match="Connection failed"):
            await mcp_client.connect()

        assert mcp_client._session is None
        assert mcp_client._is_healthy is False


@pytest.mark.asyncio
async def test_connect_idempotent(mcp_client):
    """Test that multiple connect calls are idempotent."""
    with patch("mcp_quant.mcp_client.stdio_client") as mock_stdio:
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
        mock_cm.__aexit__ = AsyncMock()
        mock_stdio.return_value = mock_cm

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock()
        mock_session.initialize = AsyncMock()

        with patch("mcp_quant.mcp_client.ClientSession", return_value=mock_session):
            await mcp_client.connect()
            await mcp_client.connect()  # Second call should do nothing

        # Should only initialize once
        mock_session.initialize.assert_called_once()


@pytest.mark.asyncio
async def test_health_check(mcp_client):
    """Test health check functionality."""
    # Initially unhealthy (not connected)
    assert await mcp_client.health_check() is False

    with patch("mcp_quant.mcp_client.stdio_client") as mock_stdio:
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
        mock_cm.__aexit__ = AsyncMock()
        mock_stdio.return_value = mock_cm

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock()
        mock_session.initialize = AsyncMock()

        with patch("mcp_quant.mcp_client.ClientSession", return_value=mock_session):
            await mcp_client.connect()

    # Now healthy
    assert await mcp_client.health_check() is True

    await mcp_client.close()

    # Unhealthy after close
    assert await mcp_client.health_check() is False


@pytest.mark.asyncio
async def test_call_tool_timeout(mcp_client):
    """Test timeout handling in call_tool."""
    with patch("mcp_quant.mcp_client.stdio_client") as mock_stdio:
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
        mock_cm.__aexit__ = AsyncMock()
        mock_stdio.return_value = mock_cm

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch("mcp_quant.mcp_client.ClientSession", return_value=mock_session):
            with pytest.raises(MCPClientError, match="timed out"):
                await mcp_client.call_tool("test_tool", {})

    assert mcp_client._is_healthy is False


@pytest.mark.asyncio
async def test_call_tool_retry_logic(mcp_client):
    """Test retry logic with exponential backoff."""
    with patch("mcp_quant.mcp_client.stdio_client") as mock_stdio:
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
        mock_cm.__aexit__ = AsyncMock()
        mock_stdio.return_value = mock_cm

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock()
        mock_session.initialize = AsyncMock()

        # Fail twice, then succeed
        call_count = 0

        async def mock_call_tool(tool_name, args):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Temporary failure")
            result = MagicMock()
            result.is_error = False
            result.content = [MagicMock(text='{"result": "success"}')]
            return result

        mock_session.call_tool = mock_call_tool

        with patch("mcp_quant.mcp_client.ClientSession", return_value=mock_session):
            result = await mcp_client.call_tool("test_tool", {})

        assert result == {"result": "success"}
        assert call_count == 3  # Should have retried twice


@pytest.mark.asyncio
async def test_call_tool_max_retries_exceeded(mcp_client):
    """Test that max retries are respected."""
    with patch("mcp_quant.mcp_client.stdio_client") as mock_stdio:
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
        mock_cm.__aexit__ = AsyncMock()
        mock_stdio.return_value = mock_cm

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(side_effect=RuntimeError("Persistent failure"))

        with patch("mcp_quant.mcp_client.ClientSession", return_value=mock_session):
            with pytest.raises(MCPClientError, match="Persistent failure"):
                await mcp_client.call_tool("test_tool", {})

        # Should have attempted MAX_RETRIES times
        assert mock_session.call_tool.call_count == MAX_RETRIES


@pytest.mark.asyncio
async def test_call_tool_success(mcp_client):
    """Test successful tool call."""
    with patch("mcp_quant.mcp_client.stdio_client") as mock_stdio:
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
        mock_cm.__aexit__ = AsyncMock()
        mock_stdio.return_value = mock_cm

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock()
        mock_session.initialize = AsyncMock()

        result_mock = MagicMock()
        result_mock.is_error = False
        result_mock.content = [MagicMock(text='{"data": [1, 2, 3]}')]
        mock_session.call_tool = AsyncMock(return_value=result_mock)

        with patch("mcp_quant.mcp_client.ClientSession", return_value=mock_session):
            result = await mcp_client.call_tool("test_tool", {"arg": "value"})

        assert result == {"data": [1, 2, 3]}
        mock_session.call_tool.assert_called_once()


@pytest.mark.asyncio
async def test_call_tool_error_response(mcp_client):
    """Test handling of MCP tool error responses."""
    with patch("mcp_quant.mcp_client.stdio_client") as mock_stdio:
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
        mock_cm.__aexit__ = AsyncMock()
        mock_stdio.return_value = mock_cm

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock()
        mock_session.initialize = AsyncMock()

        result_mock = MagicMock()
        result_mock.is_error = True
        result_mock.content = [MagicMock(text="Tool execution failed")]
        mock_session.call_tool = AsyncMock(return_value=result_mock)

        with patch("mcp_quant.mcp_client.ClientSession", return_value=mock_session):
            with pytest.raises(MCPClientError, match="returned error"):
                await mcp_client.call_tool("test_tool", {})
