from __future__ import annotations

from typing import Any, Dict, List

from mcp_quant.mcp_client import mcp_client


class ManualModeClient:
    async def list_strategies(self) -> Any:
        return await mcp_client.call_mcp_tool("list_strategies")

    async def sample_price_series(self) -> Any:
        return await mcp_client.call_mcp_tool("sample_price_series", {})

    async def run_backtest(
        self,
        *,
        prices: List[float],
        strategy: str,
        params: Dict[str, float] | None,
        start_cash: float,
        fee_bps: float,
    ) -> Any:
        return await mcp_client.call_mcp_tool(
            "run_backtest",
            {
                "prices": prices,
                "strategy": strategy,
                "params": params,
                "start_cash": start_cash,
                "fee_bps": fee_bps,
            },
        )


manual_client = ManualModeClient()
