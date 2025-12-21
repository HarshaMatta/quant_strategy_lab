from __future__ import annotations

from typing import Dict, List, Optional

from mcp.server.fastmcp import FastMCP


from .strategies import (
    StrategySpec,
    backtest,
    generate_signals,
    list_strategies as list_specs,
    sample_prices,
    validate_prices,
)


mcp = FastMCP("quant-strategies")


@mcp.tool()
def list_strategies() -> List[Dict[str, object]]:
    """Return available strategies and their default parameters."""
    specs = list_specs()
    return [
        {"name": spec.name, "description": spec.description, "params": spec.params}
        for spec in specs
    ]


@mcp.tool()
def sample_price_series(
    length: int = 240,
    start: float = 100.0,
    drift: float = 0.0005,
    vol: float = 0.01,
    seed: Optional[int] = 7,
) -> List[float]:
    """Generate a synthetic price series for quick experimentation."""
    return sample_prices(length=length, start=start, drift=drift, vol=vol, seed=seed)


@mcp.tool()
def run_backtest(
    prices: List[float],
    strategy: str,
    params: Optional[Dict[str, float]] = None,
    start_cash: float = 10_000.0,
    fee_bps: float = 1.0,
) -> Dict[str, object]:
    """Run a simple long/flat backtest and return equity and metrics."""
    cleaned = validate_prices(prices)
    signals = generate_signals(cleaned, strategy, params)
    result = backtest(cleaned, signals, start_cash=start_cash, fee_bps=fee_bps)
    return {
        "prices": cleaned,
        "signals": signals,
        **result,
    }


@mcp.tool()
def get_strategy_schema(name: str) -> Dict[str, object]:
    """Return the default parameters for a named strategy."""
    specs = {spec.name: spec for spec in list_specs()}
    if name not in specs:
        raise ValueError(f"Unknown strategy: {name}")
    spec: StrategySpec = specs[name]
    return {"name": spec.name, "description": spec.description, "params": spec.params}


if __name__ == "__main__":
    mcp.run()
