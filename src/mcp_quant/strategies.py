from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, Iterable, List

import pandas as pd


@dataclass(frozen=True)
class StrategySpec:
    name: str
    description: str
    params: Dict[str, float]


def list_strategies() -> List[StrategySpec]:
    return [
        StrategySpec(
            name="sma_crossover",
            description="Long when fast SMA is above slow SMA; flat otherwise.",
            params={"fast_window": 10, "slow_window": 30},
        ),
        StrategySpec(
            name="rsi_reversion",
            description="Long when RSI drops below oversold; exit when above overbought.",
            params={"window": 14, "oversold": 30, "overbought": 70},
        ),
        StrategySpec(
            name="channel_breakout",
            description="Long on new highs; exit on new lows over a lookback channel.",
            params={"lookback": 20},
        ),
    ]


def _sma(prices: List[float], window: int) -> List[float | None]:
    if window <= 0:
        raise ValueError("window must be positive")
    if not prices:
        return []
    sma = pd.Series(prices, dtype="float64").rolling(window=window, min_periods=window).mean()
    return [None if pd.isna(value) else float(value) for value in sma.tolist()]


def _rsi(prices: List[float], window: int) -> List[float | None]:
    if window <= 0:
        raise ValueError("window must be positive")
    rsi: List[float | None] = [None] * len(prices)
    if len(prices) < window + 1:
        return rsi
    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, window + 1):
        change = prices[i] - prices[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = sum(gains) / window
    avg_loss = sum(losses) / window
    rsi[window] = _rsi_from_avgs(avg_gain, avg_loss)
    for i in range(window + 1, len(prices)):
        change = prices[i] - prices[i - 1]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        avg_gain = (avg_gain * (window - 1) + gain) / window
        avg_loss = (avg_loss * (window - 1) + loss) / window
        rsi[i] = _rsi_from_avgs(avg_gain, avg_loss)
    return rsi


def _rsi_from_avgs(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def generate_signals(
    prices: List[float],
    strategy: str,
    params: Dict[str, float] | None = None,
) -> List[int]:
    params = params or {}
    if strategy == "sma_crossover":
        return _signals_sma(prices, params)
    if strategy == "rsi_reversion":
        return _signals_rsi(prices, params)
    if strategy == "channel_breakout":
        return _signals_channel(prices, params)
    raise ValueError(f"Unknown strategy: {strategy}")


def _signals_sma(prices: List[float], params: Dict[str, float]) -> List[int]:
    fast = int(params.get("fast_window", 10))
    slow = int(params.get("slow_window", 30))
    if fast >= slow:
        raise ValueError("fast_window must be smaller than slow_window")
    fast_sma = _sma(prices, fast)
    slow_sma = _sma(prices, slow)
    signals = [0] * len(prices)
    for i in range(len(prices)):
        if fast_sma[i] is None or slow_sma[i] is None:
            continue
        signals[i] = 1 if fast_sma[i] > slow_sma[i] else 0
    return signals


def _signals_rsi(prices: List[float], params: Dict[str, float]) -> List[int]:
    window = int(params.get("window", 14))
    oversold = float(params.get("oversold", 30))
    overbought = float(params.get("overbought", 70))
    if oversold >= overbought:
        raise ValueError("oversold must be less than overbought")
    rsi = _rsi(prices, window)
    signals = [0] * len(prices)
    in_trade = False
    for i, value in enumerate(rsi):
        if value is None:
            continue
        if not in_trade and value <= oversold:
            in_trade = True
        elif in_trade and value >= overbought:
            in_trade = False
        signals[i] = 1 if in_trade else 0
    return signals


def _signals_channel(prices: List[float], params: Dict[str, float]) -> List[int]:
    lookback = int(params.get("lookback", 20))
    if lookback <= 1:
        raise ValueError("lookback must be greater than 1")
    signals = [0] * len(prices)
    in_trade = False
    for i in range(len(prices)):
        if i < lookback:
            continue
        window = prices[i - lookback : i]
        if not window:
            continue
        high = max(window)
        low = min(window)
        if not in_trade and prices[i] > high:
            in_trade = True
        elif in_trade and prices[i] < low:
            in_trade = False
        signals[i] = 1 if in_trade else 0
    return signals


def backtest(
    prices: List[float],
    signals: List[int],
    start_cash: float = 10_000.0,
    fee_bps: float = 1.0,
) -> Dict[str, object]:
    if len(prices) != len(signals):
        raise ValueError("prices and signals must be the same length")
    if not prices:
        return {
            "equity_curve": [],
            "positions": [],
            "trades": [],
            "metrics": {},
        }
    equity = [start_cash]
    positions = [0]
    trades = []
    position = 0
    for i in range(1, len(prices)):
        desired = signals[i - 1]
        if desired != position:
            cost = equity[-1] * (fee_bps / 10_000.0)
            equity[-1] -= cost
            position = desired
            trades.append({
                "index": i,
                "price": prices[i],
                "action": "buy" if position == 1 else "sell",
                "fee": cost,
            })
        ret = (prices[i] / prices[i - 1]) - 1.0
        equity.append(equity[-1] * (1.0 + position * ret))
        positions.append(position)
    metrics = compute_metrics(equity, start_cash)
    return {
        "equity_curve": equity,
        "positions": positions,
        "trades": trades,
        "metrics": metrics,
    }


def compute_metrics(equity: List[float], start_cash: float) -> Dict[str, float]:
    if len(equity) < 2:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }
    total_return = equity[-1] / start_cash - 1.0
    periods = len(equity) - 1
    years = max(periods / 252.0, 1e-9)
    cagr = (equity[-1] / start_cash) ** (1.0 / years) - 1.0
    returns = [equity[i] / equity[i - 1] - 1.0 for i in range(1, len(equity))]
    avg = sum(returns) / len(returns)
    variance = sum((r - avg) ** 2 for r in returns) / max(len(returns) - 1, 1)
    volatility = math.sqrt(variance) * math.sqrt(252.0)
    sharpe = 0.0
    if volatility > 0:
        sharpe = (avg * 252.0) / volatility
    max_dd = _max_drawdown(equity)
    return {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


def _max_drawdown(equity: List[float]) -> float:
    peak = equity[0]
    max_dd = 0.0
    for value in equity:
        if value > peak:
            peak = value
        drawdown = (value - peak) / peak
        if drawdown < max_dd:
            max_dd = drawdown
    return abs(max_dd)


def sample_prices(
    length: int = 240,
    start: float = 100.0,
    drift: float = 0.0005,
    vol: float = 0.01,
    seed: int | None = 7,
) -> List[float]:
    rng = random.Random(seed)
    prices = [round(start, 2)]
    for _ in range(1, length):
        shock = rng.gauss(drift, vol)
        prices.append(round(prices[-1] * (1.0 + shock), 2))
    return prices


def validate_prices(prices: Iterable[float]) -> List[float]:
    cleaned: List[float] = []
    for value in prices:
        if value is None:
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        if num > 0:
            cleaned.append(round(num, 2))
    if len(cleaned) < 5:
        raise ValueError("Need at least 5 positive price points")
    return cleaned
