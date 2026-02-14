"""Reusable validation utilities for user inputs."""
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Dict

from .strategies import list_strategies


class ValidationError(ValueError):
    """Custom exception for validation errors."""

    pass


def validate_ticker(ticker: str) -> str:
    """Validate and normalize ticker symbol.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Normalized uppercase ticker

    Raises:
        ValidationError: If ticker is invalid
    """
    if not ticker:
        raise ValidationError("Ticker cannot be empty")

    cleaned = ticker.strip().upper()

    # Ticker should be 1-10 uppercase letters, may include dots or hyphens
    if not re.match(r"^[A-Z]{1,10}([.-][A-Z]{1,5})?$", cleaned):
        raise ValidationError(
            f"Invalid ticker format: '{ticker}'. "
            "Must be 1-10 letters (e.g., AAPL, BRK.B, VALE3.SA)"
        )

    return cleaned


def validate_strategy_params(strategy: str, params: Dict[str, float] | None) -> Dict[str, float]:
    """Validate strategy name and parameters against known strategies.

    Args:
        strategy: Strategy name
        params: Strategy parameters (or None for defaults)

    Returns:
        Validated parameters (defaults if None provided)

    Raises:
        ValidationError: If strategy is unknown or params are invalid
    """
    # Get known strategies
    known_strategies = {spec.name: spec for spec in list_strategies()}

    if strategy not in known_strategies:
        available = ", ".join(sorted(known_strategies.keys()))
        raise ValidationError(
            f"Unknown strategy: '{strategy}'. Available strategies: {available}"
        )

    spec = known_strategies[strategy]

    # Use defaults if no params provided
    if params is None:
        return spec.params.copy()

    # Validate each parameter
    validated = {}
    for key, value in params.items():
        if not isinstance(value, (int, float)):
            raise ValidationError(
                f"Parameter '{key}' must be numeric, got {type(value).__name__}"
            )

        # General range check
        if value < 0:
            raise ValidationError(f"Parameter '{key}' must be non-negative, got {value}")

        if value > 10_000:
            raise ValidationError(
                f"Parameter '{key}' value {value} exceeds maximum (10,000)"
            )

        validated[key] = float(value)

    # Strategy-specific validation
    if strategy == "sma_crossover":
        fast = validated.get("fast_window", spec.params.get("fast_window", 0))
        slow = validated.get("slow_window", spec.params.get("slow_window", 0))

        if fast <= 0 or slow <= 0:
            raise ValidationError("SMA windows must be positive integers")

        if fast >= slow:
            raise ValidationError(
                f"fast_window ({fast}) must be less than slow_window ({slow})"
            )

    elif strategy == "rsi_reversion":
        window = validated.get("window", spec.params.get("window", 0))
        oversold = validated.get("oversold", spec.params.get("oversold", 0))
        overbought = validated.get("overbought", spec.params.get("overbought", 100))

        if window <= 0:
            raise ValidationError(f"RSI window must be positive, got {window}")

        if not (0 <= oversold <= 100):
            raise ValidationError(f"oversold must be in [0, 100], got {oversold}")

        if not (0 <= overbought <= 100):
            raise ValidationError(f"overbought must be in [0, 100], got {overbought}")

        if oversold >= overbought:
            raise ValidationError(
                f"oversold ({oversold}) must be less than overbought ({overbought})"
            )

    elif strategy == "channel_breakout":
        lookback = validated.get("lookback", spec.params.get("lookback", 0))

        if lookback <= 0:
            raise ValidationError(f"lookback must be positive, got {lookback}")

    return validated


def validate_date_range(start_date: date, end_date: date) -> None:
    """Validate date range for historical data.

    Args:
        start_date: Start date
        end_date: End date

    Raises:
        ValidationError: If date range is invalid
    """
    today = date.today()

    # Check ordering
    if start_date >= end_date:
        raise ValidationError(
            f"start_date ({start_date}) must be before end_date ({end_date})"
        )

    # Check for future dates
    if start_date > today:
        raise ValidationError(
            f"start_date ({start_date}) cannot be in the future (today: {today})"
        )

    if end_date > today:
        raise ValidationError(
            f"end_date ({end_date}) cannot be in the future (today: {today})"
        )

    # Check minimum range
    min_days = 2
    if (end_date - start_date).days < min_days:
        raise ValidationError(
            f"Date range must be at least {min_days} days, "
            f"got {(end_date - start_date).days} days"
        )

    # Check maximum range
    max_years = 20
    max_days = max_years * 365
    if (end_date - start_date).days > max_days:
        raise ValidationError(
            f"Date range cannot exceed {max_years} years ({max_days} days), "
            f"got {(end_date - start_date).days} days"
        )


def validate_backtest_params(start_cash: float, fee_bps: float) -> None:
    """Validate backtest parameters.

    Args:
        start_cash: Starting cash amount
        fee_bps: Trading fee in basis points

    Raises:
        ValidationError: If parameters are invalid
    """
    # Validate start_cash
    if start_cash <= 0:
        raise ValidationError(f"start_cash must be positive, got {start_cash}")

    if start_cash > 1_000_000_000:
        raise ValidationError(
            f"start_cash exceeds maximum (1 billion), got {start_cash}"
        )

    # Validate fee_bps
    if fee_bps < 0:
        raise ValidationError(f"fee_bps must be non-negative, got {fee_bps}")

    if fee_bps > 1000:
        raise ValidationError(
            f"fee_bps exceeds reasonable maximum (1000 bps = 10%), got {fee_bps}"
        )
