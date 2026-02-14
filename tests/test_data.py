"""Unit tests for data fetching module."""
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from mcp_quant.data import fetch_yahoo_prices


def test_fetch_yahoo_prices_invalid_ticker():
    """Test error for invalid ticker."""
    with pytest.raises(ValueError, match="Ticker is required"):
        fetch_yahoo_prices("", date(2023, 1, 1), date(2023, 12, 31))

    with pytest.raises(ValueError, match="Ticker is required"):
        fetch_yahoo_prices("   ", date(2023, 1, 1), date(2023, 12, 31))


def test_fetch_yahoo_prices_invalid_date_range():
    """Test error for invalid date range."""
    with pytest.raises(ValueError, match="Start date.*must be before end date"):
        fetch_yahoo_prices("AAPL", date(2023, 12, 31), date(2023, 1, 1))

    with pytest.raises(ValueError, match="Start date.*must be before end date"):
        fetch_yahoo_prices("AAPL", date(2023, 6, 15), date(2023, 6, 15))


def test_fetch_yahoo_prices_successful_fetch():
    """Test successful data fetch with mocked response."""
    mock_response = MagicMock()
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)

    # Need at least 5 valid prices for validation
    mock_json = {
        "chart": {
            "result": [{
                "indicators": {
                    "adjclose": [{
                        "adjclose": [100.0, 101.5, 99.8, 102.3, 103.1, 104.0]
                    }]
                }
            }]
        }
    }

    with patch("mcp_quant.data.urlopen", return_value=mock_response):
        with patch("mcp_quant.data.json.load", return_value=mock_json):
            prices = fetch_yahoo_prices("AAPL", date(2023, 1, 1), date(2023, 1, 6))

    assert len(prices) == 6
    assert prices[0] == 100.0
    assert prices[-1] == 104.0


def test_fetch_yahoo_prices_yahoo_error():
    """Test handling of Yahoo Finance API error."""
    mock_response = MagicMock()
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)
    mock_response.read = MagicMock(
        return_value=b'''{
        "chart": {
            "error": {
                "description": "Invalid ticker symbol"
            }
        }
    }'''
    )

    with patch("mcp_quant.data.urlopen", return_value=mock_response):
        with pytest.raises(ValueError, match="Yahoo Finance error.*Invalid ticker symbol"):
            fetch_yahoo_prices("INVALID", date(2023, 1, 1), date(2023, 12, 31))


def test_fetch_yahoo_prices_no_data_returned():
    """Test error when no data is returned."""
    mock_response = MagicMock()
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)
    mock_response.read = MagicMock(
        return_value=b'''{
        "chart": {
            "result": []
        }
    }'''
    )

    with patch("mcp_quant.data.urlopen", return_value=mock_response):
        with pytest.raises(ValueError, match="No data returned from Yahoo Finance"):
            fetch_yahoo_prices("TEST", date(2023, 1, 1), date(2023, 12, 31))


def test_fetch_yahoo_prices_no_price_data():
    """Test error when response contains no price series."""
    mock_response = MagicMock()
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)
    mock_response.read = MagicMock(
        return_value=b'''{
        "chart": {
            "result": [{
                "indicators": {
                    "adjclose": [],
                    "quote": []
                }
            }]
        }
    }'''
    )

    with patch("mcp_quant.data.urlopen", return_value=mock_response):
        with pytest.raises(ValueError, match="No price data returned"):
            fetch_yahoo_prices("TEST", date(2023, 1, 1), date(2023, 12, 31))


def test_fetch_yahoo_prices_ticker_normalization():
    """Test that ticker is normalized to uppercase."""
    mock_response = MagicMock()
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)

    mock_json = {
        "chart": {
            "result": [{
                "indicators": {
                    "adjclose": [{
                        "adjclose": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
                    }]
                }
            }]
        }
    }

    with patch("mcp_quant.data.urlopen", return_value=mock_response) as mock_urlopen:
        with patch("mcp_quant.data.json.load", return_value=mock_json):
            prices = fetch_yahoo_prices("aapl", date(2023, 1, 1), date(2023, 1, 6))

    # Verify the URL contains uppercase ticker
    call_args = mock_urlopen.call_args
    request = call_args[0][0]
    assert "AAPL" in request.full_url


def test_fetch_yahoo_prices_fallback_to_close():
    """Test fallback to close prices when adjclose is not available."""
    mock_response = MagicMock()
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)

    mock_json = {
        "chart": {
            "result": [{
                "indicators": {
                    "adjclose": [],
                    "quote": [{
                        "close": [50.0, 51.5, 52.0, 53.0, 54.0, 55.0]
                    }]
                }
            }]
        }
    }

    with patch("mcp_quant.data.urlopen", return_value=mock_response):
        with patch("mcp_quant.data.json.load", return_value=mock_json):
            prices = fetch_yahoo_prices("TEST", date(2023, 1, 1), date(2023, 1, 6))

    assert len(prices) == 6
    assert prices[0] == 50.0
    assert prices[-1] == 55.0
