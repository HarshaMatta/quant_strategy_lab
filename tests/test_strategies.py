from __future__ import annotations

import unittest

from mcp_quant import strategies


class StrategyTests(unittest.TestCase):
    def test_list_strategies(self) -> None:
        specs = strategies.list_strategies()
        self.assertTrue(specs)
        names = {spec.name for spec in specs}
        self.assertIn("sma_crossover", names)
        self.assertIn("rsi_reversion", names)
        self.assertIn("channel_breakout", names)

    def test_sample_prices_length(self) -> None:
        prices = strategies.sample_prices(length=12, seed=1)
        self.assertEqual(len(prices), 12)

    def test_generate_signals_length(self) -> None:
        prices = [100, 101, 102, 101, 100, 99, 100, 102, 103, 104, 105]
        signals = strategies.generate_signals(
            prices, "sma_crossover", {"fast_window": 3, "slow_window": 5}
        )
        self.assertEqual(len(signals), len(prices))
        self.assertTrue(all(value in (0, 1) for value in signals))

    def test_sma_rolling_values(self) -> None:
        prices = [1, 2, 3, 4, 5]
        self.assertEqual(strategies._sma(prices, 3), [None, None, 2.0, 3.0, 4.0])

    def test_sma_rejects_non_positive_window(self) -> None:
        with self.assertRaises(ValueError):
            strategies._sma([100, 101, 102], 0)

    def test_backtest_outputs(self) -> None:
        prices = [100, 102, 101, 103, 104, 106]
        signals = [0, 1, 1, 0, 1, 0]
        result = strategies.backtest(prices, signals, start_cash=1000, fee_bps=0)
        self.assertIn("equity_curve", result)
        self.assertIn("metrics", result)
        self.assertEqual(len(result["equity_curve"]), len(prices))
        self.assertIn("total_return", result["metrics"])

    def test_validate_prices_rejects_short_series(self) -> None:
        with self.assertRaises(ValueError):
            strategies.validate_prices([100, 101, None, "bad"])


if __name__ == "__main__":
    unittest.main()
