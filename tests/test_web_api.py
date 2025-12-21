from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from mcp_quant.web.app import app


class WebApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_list_strategies(self) -> None:
        response = self.client.get("/api/strategies")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        self.assertIn("name", data[0])
        self.assertIn("params", data[0])

    def test_run_backtest_default_data(self) -> None:
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
