from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from mcp_quant.web.app import app


class WebApiTests(unittest.TestCase):
    def setUp(self) -> None:
        # Mock the MCP client to avoid connection issues
        self.mcp_patcher = patch("mcp_quant.web.app.mcp_client")
        self.mock_mcp = self.mcp_patcher.start()
        self.mock_mcp.connect = AsyncMock()
        self.mock_mcp.close = AsyncMock()

        # Mock manual_client responses
        self.manual_patcher = patch("mcp_quant.web.app.manual_client")
        self.mock_manual = self.manual_patcher.start()

        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.mcp_patcher.stop()
        self.manual_patcher.stop()

    def test_list_strategies(self) -> None:
        self.mock_manual.list_strategies = AsyncMock(return_value=[
            {"name": "sma_crossover", "description": "SMA strategy", "params": {}},
            {"name": "rsi_reversion", "description": "RSI strategy", "params": {}},
        ])

        response = self.client.get("/api/strategies")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        self.assertIn("name", data[0])
        self.assertIn("params", data[0])

    def test_run_backtest_default_data(self) -> None:
        # Mock sample_price_series
        self.mock_manual.sample_price_series = AsyncMock(return_value=[
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0
        ])

        # Mock run_backtest
        self.mock_manual.run_backtest = AsyncMock(return_value={
            "prices": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "signals": [0, 0, 1, 1, 1, 1],
            "equity_curve": [10000, 10000, 10100, 10200, 10300, 10400],
            "metrics": {"total_return": 0.04, "sharpe": 1.5, "max_drawdown": -0.01},
            "positions": [0, 0, 1, 1, 1, 1],
        })

        payload = {
            "strategy": "sma_crossover",
            "params": {"fast_window": 5, "slow_window": 15},
            "start_cash": 10000,
            "fee_bps": 0,
        }
        response = self.client.post("/api/backtest", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        for key in ("prices", "signals", "equity_curve", "metrics"):
            self.assertIn(key, data)
        self.assertEqual(len(data["prices"]), len(data["signals"]))
        self.assertEqual(len(data["prices"]), len(data["equity_curve"]))


if __name__ == "__main__":
    unittest.main()
