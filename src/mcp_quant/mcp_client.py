from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
import time
from typing import Any, Dict

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

try:
    from mcp.client.sse import sse_client
except ImportError:  # pragma: no cover - optional transport
    sse_client = None


class MCPClientError(RuntimeError):
    pass


# Connection health constants
MAX_RETRIES = 3
BASE_RETRY_DELAY = 0.5  # seconds
MAX_RETRY_DELAY = 5.0  # seconds


def _get_stdio_params() -> StdioServerParameters:
    command = os.getenv("MCP_SERVER_COMMAND") or sys.executable
    args_env = os.getenv("MCP_SERVER_ARGS")
    if args_env:
        args = shlex.split(args_env)
    else:
        args = ["-m", "mcp_quant.mcp_server"]
    return StdioServerParameters(command=command, args=args)


class MCPClient:
    def __init__(self) -> None:
        self._session: ClientSession | None = None
        self._transport_cm: Any | None = None
        self._transport: tuple[Any, Any] | None = None
        self._lock = asyncio.Lock()
        self._connect_lock = asyncio.Lock()
        self._last_connect_attempt: float = 0.0
        self._retry_delay: float = BASE_RETRY_DELAY
        self._is_healthy: bool = False

    async def connect(self) -> None:
        if self._session is not None:
            return
        async with self._connect_lock:
            if self._session is not None:
                return
            self._last_connect_attempt = time.time()
            url = os.getenv("MCP_SERVER_URL")
            if url:
                if sse_client is None:
                    raise MCPClientError("MCP_SERVER_URL is set but SSE support is unavailable")
                self._transport_cm = sse_client(url)
            else:
                params = _get_stdio_params()
                self._transport_cm = stdio_client(params)
            try:
                read, write = await self._transport_cm.__aenter__()
                self._transport = (read, write)
                self._session = ClientSession(read, write)
                enter = getattr(self._session, "__aenter__", None)
                if enter is not None:
                    await enter()
                await self._session.initialize()
                self._is_healthy = True
                self._retry_delay = BASE_RETRY_DELAY  # Reset retry delay on success
            except Exception as exc:
                self._is_healthy = False
                await self.close()
                raise MCPClientError(str(exc)) from exc

    async def close(self) -> None:
        self._is_healthy = False
        if self._session is not None:
            exit_method = getattr(self._session, "__aexit__", None)
            if exit_method is not None:
                await exit_method(None, None, None)
            else:
                close_method = getattr(self._session, "close", None)
                if close_method is not None:
                    await close_method()
            self._session = None
        if self._transport_cm is not None:
            exit_method = getattr(self._transport_cm, "__aexit__", None)
            if exit_method is not None:
                try:
                    await exit_method(None, None, None)
                except Exception:
                    pass
            self._transport_cm = None
            self._transport = None

    async def _reconnect_with_backoff(self) -> None:
        """Attempt to reconnect with exponential backoff."""
        await self.close()
        await asyncio.sleep(self._retry_delay)
        self._retry_delay = min(self._retry_delay * 2, MAX_RETRY_DELAY)
        await self.connect()

    async def health_check(self) -> bool:
        """Check if the connection is healthy."""
        if self._session is None:
            return False
        if not self._is_healthy:
            return False
        return True

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any] | None = None) -> Any:
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                await self.connect()
                async with self._lock:
                    if self._session is None:
                        raise MCPClientError("MCP session is not available")
                    try:
                        result = await asyncio.wait_for(
                            self._session.call_tool(tool_name, arguments or {}),
                            timeout=15,
                        )
                    except asyncio.TimeoutError as exc:
                        self._is_healthy = False
                        raise MCPClientError(f"MCP tool '{tool_name}' timed out after 15 seconds") from exc
                    except Exception as exc:
                        self._is_healthy = False
                        raise MCPClientError(f"MCP tool '{tool_name}' failed: {exc}") from exc
                if getattr(result, "is_error", False):
                    message = _content_to_value(getattr(result, "content", "")) or "MCP tool error"
                    raise MCPClientError(f"MCP tool '{tool_name}' returned error: {message}")
                return _content_to_value(getattr(result, "content", result))
            except MCPClientError as exc:
                last_error = exc
                if attempt < MAX_RETRIES - 1 and not self._is_healthy:
                    # Attempt reconnection with backoff
                    try:
                        await self._reconnect_with_backoff()
                    except Exception:
                        pass  # Continue to next retry
                else:
                    break
        # All retries exhausted
        if last_error:
            raise last_error
        raise MCPClientError(f"Failed to call MCP tool '{tool_name}' after {MAX_RETRIES} attempts")

    async def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any] | None = None) -> Any:
        return await self.call_tool(tool_name, arguments)

def _content_to_value(content: Any) -> Any:
    if isinstance(content, list):
        if len(content) == 1:
            return _item_to_value(content[0])
        return [_item_to_value(item) for item in content]
    return _item_to_value(content)


def _item_to_value(item: Any) -> Any:
    if isinstance(item, (dict, list, str, int, float, bool)) or item is None:
        return item
    text = getattr(item, "text", None)
    if text is not None:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    data = getattr(item, "data", None)
    if data is not None:
        return data
    return item


mcp_client = MCPClient()
