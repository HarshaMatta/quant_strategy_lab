from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from mcp_quant.data import fetch_yahoo_prices
from mcp_quant.strategies import backtest, generate_signals, list_strategies, sample_prices


app = FastAPI(title="Quant Strategy Lab")


class BacktestRequest(BaseModel):
    strategy: str
    params: Optional[Dict[str, float]] = None
    start_cash: float = 10_000.0
    fee_bps: float = 1.0
    ticker: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return INDEX_HTML


@app.get("/api/strategies")
async def strategies() -> List[Dict[str, object]]:
    return [
        {"name": spec.name, "description": spec.description, "params": spec.params}
        for spec in list_strategies()
    ]


@app.post("/api/backtest")
async def run_backtest(payload: BacktestRequest) -> Dict[str, object]:
    if payload.ticker and payload.start_date and payload.end_date:
        try:
            prices = fetch_yahoo_prices(payload.ticker, payload.start_date, payload.end_date)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Failed to fetch Yahoo Finance data") from exc
    else:
        prices = sample_prices()
    signals = generate_signals(prices, payload.strategy, payload.params)
    result = backtest(prices, signals, start_cash=payload.start_cash, fee_bps=payload.fee_bps)
    return {
        "prices": prices,
        "signals": signals,
        **result,
    }


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Quant Strategy Lab</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet" />
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1"></script>
  <style>
    :root {
      --bg-1: #f7efe6;
      --bg-2: #e6f3ff;
      --ink: #1a1f2b;
      --muted: #5a6475;
      --accent: #c85c32;
      --accent-2: #0d6f85;
      --card: #ffffffcc;
      --stroke: #d8d0c5;
      --shadow: 0 20px 50px rgba(15, 29, 44, 0.12);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      font-family: "Space Grotesk", system-ui, sans-serif;
      color: var(--ink);
      background: radial-gradient(circle at top left, var(--bg-2), transparent 60%),
                  linear-gradient(135deg, var(--bg-1), #f6f0ea 45%, #e3f0f3);
      min-height: 100vh;
    }

    .wrap {
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 20px 60px;
    }

    header {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-bottom: 24px;
    }

    header h1 {
      font-size: clamp(28px, 4vw, 44px);
      margin: 0;
      letter-spacing: -0.02em;
    }

    header p {
      margin: 0;
      color: var(--muted);
      max-width: 620px;
      font-size: 16px;
    }

    .grid {
      display: grid;
      grid-template-columns: minmax(260px, 1fr) minmax(320px, 2fr);
      gap: 20px;
    }

    .card {
      background: var(--card);
      border: 1px solid var(--stroke);
      border-radius: 20px;
      padding: 20px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(8px);
      animation: rise 0.6s ease both;
    }

    .card:nth-child(2) {
      animation-delay: 0.1s;
    }

    @keyframes rise {
      from {
        transform: translateY(16px);
        opacity: 0;
      }
      to {
        transform: translateY(0);
        opacity: 1;
      }
    }

    label {
      display: block;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin: 14px 0 6px;
    }

    input, select, textarea, button {
      width: 100%;
      font-family: inherit;
      border-radius: 12px;
      border: 1px solid var(--stroke);
      padding: 10px 12px;
      font-size: 15px;
      background: #fff;
      color: var(--ink);
    }

    textarea {
      min-height: 120px;
      font-family: "IBM Plex Mono", ui-monospace, monospace;
      font-size: 13px;
    }

    .row {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }

    .muted {
      font-size: 12px;
      color: var(--muted);
    }

    button {
      background: var(--accent);
      color: #fff;
      font-weight: 600;
      border: none;
      cursor: pointer;
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    button:hover {
      transform: translateY(-1px);
      box-shadow: 0 10px 20px rgba(200, 92, 50, 0.25);
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
      margin-top: 12px;
    }

    .metric {
      background: #ffffff;
      border: 1px solid var(--stroke);
      border-radius: 14px;
      padding: 12px;
    }

    .metric span {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
    }

    .metric strong {
      font-size: 18px;
    }

    .chart-block {
      margin-top: 16px;
    }

    .chart-block h3 {
      margin: 4px 0 8px;
      font-size: 16px;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      background: #f2e7db;
      color: var(--accent);
      font-size: 12px;
      font-weight: 600;
    }

    .legend {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 8px;
      font-size: 12px;
    }

    .legend span {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    .dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      display: inline-block;
    }

    .dot.price { background: var(--accent-2); }
    .dot.equity { background: var(--accent); }

    @media (max-width: 900px) {
      .grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div class="badge">Quant Strategy Lab</div>
      <h1>Test momentum and mean-reversion ideas in seconds.</h1>
      <p>Pick a strategy, tune parameters, and visualize equity curves with a lightweight backtester. Fetch Yahoo Finance data or explore the synthetic series.</p>
    </header>

    <div class="grid">
      <section class="card">
        <label for="strategy">Strategy</label>
        <select id="strategy"></select>
        <div id="strategyDesc" class="muted"></div>

        <div id="paramFields"></div>

        <div class="row">
          <div>
            <label for="startCash">Start cash</label>
            <input id="startCash" type="number" value="10000" />
          </div>
          <div>
            <label for="feeBps">Fee (bps)</label>
            <input id="feeBps" type="number" step="0.1" value="1.0" />
          </div>
        </div>

        <label for="ticker">Ticker (Yahoo Finance)</label>
        <input id="ticker" list="tickerList" placeholder="AAPL" />
        <datalist id="tickerList">
          <option value="AAPL"></option>
          <option value="MSFT"></option>
          <option value="NVDA"></option>
          <option value="TSLA"></option>
          <option value="SPY"></option>
          <option value="QQQ"></option>
        </datalist>

        <div class="row">
          <div>
            <label for="startDate">Start date</label>
            <input id="startDate" type="date" />
          </div>
          <div>
            <label for="endDate">End date</label>
            <input id="endDate" type="date" />
          </div>
        </div>
        </br>
        <button id="runBtn" type="button">Run Back Test</button>
      </section>

      <section class="card">
        <div class="legend">
          <span><span class="dot price"></span> Price</span>
          <span><span class="dot equity"></span> Equity</span>
        </div>

        <div class="metrics" id="metrics"></div>

        <div class="chart-block">
          <h3>Price vs Equity</h3>
          <canvas id="chart" height="220"></canvas>
        </div>

        <div class="chart-block">
          <h3>Position</h3>
          <canvas id="positionChart" height="160"></canvas>
        </div>
      </section>
    </div>
  </div>

<script>
  const state = {
    strategies: [],
    chart: null,
    positionChart: null,
  };

  function createParamInput(key, value) {
    const wrapper = document.createElement("div");
    const label = document.createElement("label");
    label.textContent = key.replace(/_/g, " ");
    const input = document.createElement("input");
    input.type = "number";
    input.step = "0.1";
    input.value = value;
    input.dataset.paramKey = key;
    wrapper.appendChild(label);
    wrapper.appendChild(input);
    return wrapper;
  }

  function loadParams(strategy) {
    const container = document.getElementById("paramFields");
    container.innerHTML = "";
    if (!strategy) return;
    const desc = document.getElementById("strategyDesc");
    desc.textContent = strategy.description;
    Object.entries(strategy.params).forEach(([key, value]) => {
      container.appendChild(createParamInput(key, value));
    });
  }

  function getParams() {
    const params = {};
    document.querySelectorAll("#paramFields input").forEach((input) => {
      const key = input.dataset.paramKey;
      params[key] = Number(input.value);
    });
    return params;
  }

  function formatPercent(value) {
    return (value * 100).toFixed(2) + "%";
  }

  function formatCurrency(value) {
    const safeValue = Number(value);
    if (!Number.isFinite(safeValue)) return "$0.00";
    const sign = safeValue < 0 ? "-" : "";
    const absValue = Math.abs(safeValue);
    return sign + "$" + absValue.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }

  function setDefaultDates() {
    const endInput = document.getElementById("endDate");
    const startInput = document.getElementById("startDate");
    const formatDate = (value) => {
      const offsetMs = value.getTimezoneOffset() * 60 * 1000;
      return new Date(value.getTime() - offsetMs).toISOString().slice(0, 10);
    };
    if (!endInput.value) {
      const end = new Date();
      endInput.value = formatDate(end);
    }
    if (!startInput.value) {
      const start = new Date(endInput.value);
      start.setFullYear(start.getFullYear() - 1);
      startInput.value = formatDate(start);
    }
  }

  function renderMetrics(metrics, prices, equity, startCash) {
    const container = document.getElementById("metrics");
    container.innerHTML = "";
    const firstPrice = prices.length ? Number(prices[0]) : NaN;
    const lastPrice = prices.length ? Number(prices[prices.length - 1]) : NaN;
    const priceReturn = Number.isFinite(firstPrice) && Number.isFinite(lastPrice) && firstPrice > 0
      ? (lastPrice / firstPrice) - 1
      : 0;
    const actualReturn = Number(startCash) * priceReturn;
    const lastEquity = equity.length ? Number(equity[equity.length - 1]) : Number(startCash);
    const strategyReturn = lastEquity - Number(startCash);
    const items = [
      { label: "Actual return (no strategy)", value: formatCurrency(actualReturn) },
      { label: "Strategy return", value: formatCurrency(strategyReturn) },
      { label: "Total return", value: formatPercent(metrics.total_return || 0) },
      { label: "CAGR", value: formatPercent(metrics.cagr || 0) },
      { label: "Volatility", value: formatPercent(metrics.volatility || 0) },
      { label: "Sharpe", value: (metrics.sharpe || 0).toFixed(2) },
      { label: "Max drawdown", value: formatPercent(metrics.max_drawdown || 0) },
    ];
    items.forEach((item) => {
      const card = document.createElement("div");
      card.className = "metric";
      const label = document.createElement("span");
      label.textContent = item.label;
      const value = document.createElement("strong");
      value.textContent = item.value;
      card.appendChild(label);
      card.appendChild(value);
      container.appendChild(card);
    });
  }

  function renderCharts(prices, equity, positions) {
    const labels = prices.map((_, i) => i + 1);
    if (state.chart) state.chart.destroy();
    if (state.positionChart) state.positionChart.destroy();

    state.chart = new Chart(document.getElementById("chart"), {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Price",
            data: prices,
            borderColor: "#0d6f85",
            backgroundColor: "rgba(13, 111, 133, 0.12)",
            tension: 0.25,
            yAxisID: "y",
          },
          {
            label: "Equity",
            data: equity,
            borderColor: "#c85c32",
            backgroundColor: "rgba(200, 92, 50, 0.12)",
            tension: 0.25,
            yAxisID: "y1",
          },
        ]
      },
      options: {
        responsive: true,
        interaction: { mode: "index", intersect: false },
        scales: {
          y: { position: "left" },
          y1: { position: "right", grid: { drawOnChartArea: false } }
        }
      }
    });

    state.positionChart = new Chart(document.getElementById("positionChart"), {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Position",
            data: positions,
            borderColor: "#222",
            backgroundColor: "rgba(0, 0, 0, 0.1)",
            stepped: true,
            tension: 0,
          }
        ]
      },
      options: {
        scales: {
          y: { min: -0.1, max: 1.1 }
        }
      }
    });
  }

  async function init() {
    const response = await fetch("/api/strategies");
    const data = await response.json();
    state.strategies = data;
    const select = document.getElementById("strategy");
    data.forEach((strategy) => {
      const option = document.createElement("option");
      option.value = strategy.name;
      option.textContent = strategy.name.replace(/_/g, " ");
      select.appendChild(option);
    });
    select.addEventListener("change", () => {
      const chosen = state.strategies.find((s) => s.name === select.value);
      loadParams(chosen);
    });
    loadParams(data[0]);
    select.value = data[0].name;
    document.getElementById("runBtn").addEventListener("click", runBacktest);
    setDefaultDates();
    await runBacktest();
  }

  async function runBacktest() {
    const select = document.getElementById("strategy");
    const ticker = document.getElementById("ticker").value.trim();
    const startDate = document.getElementById("startDate").value;
    const endDate = document.getElementById("endDate").value;
    const payload = {
      strategy: select.value,
      params: getParams(),
      start_cash: Number(document.getElementById("startCash").value || 10000),
      fee_bps: Number(document.getElementById("feeBps").value || 1.0),
      ticker: ticker || null,
      start_date: startDate || null,
      end_date: endDate || null,
    };
    const response = await fetch("/api/backtest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      let message = "Unable to run backtest.";
      try {
        const errorData = await response.json();
        if (errorData.detail) {
          message = errorData.detail;
        }
      } catch (err) {
        message = "Unable to run backtest.";
      }
      alert(message);
      return;
    }
    const result = await response.json();
    renderMetrics(
      result.metrics || {},
      result.prices || [],
      result.equity_curve || [],
      payload.start_cash
    );
    renderCharts(result.prices || [], result.equity_curve || [], result.positions || []);
  }

  init();
</script>
</body>
</html>
"""
