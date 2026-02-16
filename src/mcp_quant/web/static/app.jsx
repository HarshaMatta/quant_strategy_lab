const { useEffect, useMemo, useRef, useState } = React;

const TYPE_OPTIONS = {
  openai: {
    label: "OpenAI",
    base: "https://api.openai.com",
    models: ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
  },
  gemini: {
    label: "Gemini",
    base: "https://generativelanguage.googleapis.com/v1beta/openai",
    models: ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"],
  },
  openrouter: {
    label: "OpenRouter",
    base: "https://openrouter.ai/api",
    models: ["openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet", "google/gemini-1.5-pro"],
  },
  groq: {
    label: "Groq",
    base: "https://api.groq.com/openai",
    models: ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
  },
  together: {
    label: "Together",
    base: "https://api.together.xyz",
    models: ["meta-llama/Llama-3.1-70B-Instruct-Turbo", "Qwen/Qwen2.5-72B-Instruct-Turbo"],
  },
  fireworks: {
    label: "Fireworks",
    base: "https://api.fireworks.ai/inference",
    models: ["accounts/fireworks/models/llama-v3p1-70b-instruct"],
  },
  deepinfra: {
    label: "DeepInfra",
    base: "https://api.deepinfra.com/v1/openai",
    models: ["meta-llama/Meta-Llama-3.1-70B-Instruct", "Qwen/Qwen2.5-72B-Instruct"],
  },
  perplexity: {
    label: "Perplexity",
    base: "https://api.perplexity.ai",
    models: ["sonar", "sonar-pro"],
  },
  mistral: {
    label: "Mistral",
    base: "https://api.mistral.ai",
    models: ["mistral-large-latest", "mistral-medium-latest"],
  },
  custom: {
    label: "Custom",
    base: "",
    models: [],
  },
};

const TYPE_ORDER = [
  "openai",
  "gemini",
  "openrouter",
  "groq",
  "together",
  "fireworks",
  "deepinfra",
  "perplexity",
  "mistral",
  "custom",
];

const PRICE_EQUITY_OPTIONS = {
  responsive: true,
  interaction: { mode: "index", intersect: false },
  scales: {
    y: { position: "left" },
    y1: { position: "right", grid: { drawOnChartArea: false } },
  },
};

const POSITION_OPTIONS = {
  scales: {
    y: { min: -0.1, max: 1.1 },
  },
};

function formatPercent(value) {
  const safeValue = Number(value);
  if (!Number.isFinite(safeValue)) return "0.00%";
  return (safeValue * 100).toFixed(2) + "%";
}

function formatCurrency(value) {
  const safeValue = Number(value);
  if (!Number.isFinite(safeValue)) return "$0.00";
  const sign = safeValue < 0 ? "-" : "";
  const absValue = Math.abs(safeValue);
  return sign + "$" + absValue.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function computeMetrics(metrics, prices, equity, startCash) {
  const firstPrice = prices.length ? Number(prices[0]) : NaN;
  const lastPrice = prices.length ? Number(prices[prices.length - 1]) : NaN;
  const priceReturn = Number.isFinite(firstPrice) && Number.isFinite(lastPrice) && firstPrice > 0
    ? (lastPrice / firstPrice) - 1
    : 0;
  const actualReturn = Number(startCash) * priceReturn;
  const lastEquity = equity.length ? Number(equity[equity.length - 1]) : Number(startCash);
  const strategyReturn = lastEquity - Number(startCash);
  const safeMetrics = metrics || {};
  return [
    { label: "Actual return", value: formatCurrency(actualReturn) },
    { label: "Strategy return", value: formatCurrency(strategyReturn) },
    { label: "Total return", value: formatPercent(safeMetrics.total_return || 0) },
    { label: "CAGR", value: formatPercent(safeMetrics.cagr || 0) },
    { label: "Volatility", value: formatPercent(safeMetrics.volatility || 0) },
    { label: "Sharpe", value: Number(safeMetrics.sharpe || 0).toFixed(2) },
    { label: "Max drawdown", value: formatPercent(safeMetrics.max_drawdown || 0) },
  ];
}

function findBacktestResult(result) {
  const steps = Array.isArray(result.steps) ? result.steps : [];
  for (let i = steps.length - 1; i >= 0; i -= 1) {
    const step = steps[i];
    if (step && step.tool === "run_backtest" && step.result) {
      const startCash = Number(step.arguments?.start_cash ?? 10000);
      return { data: step.result, startCash };
    }
  }
  return null;
}

function Legend() {
  return (
    <div className="legend">
      <span><span className="dot price"></span> Price</span>
      <span><span className="dot equity"></span> Equity</span>
    </div>
  );
}

function MetricGrid({ items }) {
  return (
    <div className="metrics">
      {items.map((item) => (
        <div key={item.label} className="metric">
          <span>{item.label}</span>
          <strong>{item.value}</strong>
        </div>
      ))}
    </div>
  );
}

function LineChart({ labels, datasets, options, height }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (chartRef.current) {
      chartRef.current.destroy();
      chartRef.current = null;
    }
    if (!labels.length || !datasets.length) return;
    const resolvedOptions = options ? JSON.parse(JSON.stringify(options)) : undefined;
    chartRef.current = new Chart(canvas, {
      type: "line",
      data: { labels, datasets },
      options: resolvedOptions,
    });
    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
        chartRef.current = null;
      }
    };
  }, [labels, datasets, options]);

  return <canvas ref={canvasRef} height={height}></canvas>;
}

function TabButton({ id, label, controlsId, active, onClick, onNavigate }) {
  return (
    <button
      className={`tab ${active ? "active" : ""}`}
      id={id}
      type="button"
      role="tab"
      aria-controls={controlsId}
      aria-selected={active}
      tabIndex={active ? 0 : -1}
      onClick={onClick}
      onKeyDown={onNavigate}
    >
      {label}
    </button>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState("lab");
  const [strategies, setStrategies] = useState([]);
  const [selectedName, setSelectedName] = useState("");
  const [params, setParams] = useState({});
  const [startCash, setStartCash] = useState(10000);
  const [feeBps, setFeeBps] = useState(1.0);
  const [ticker, setTicker] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [labMetrics, setLabMetrics] = useState({});
  const [labPrices, setLabPrices] = useState([]);
  const [labEquity, setLabEquity] = useState([]);
  const [labPositions, setLabPositions] = useState([]);
  const [labStartCash, setLabStartCash] = useState(10000);
  const [labCanRun, setLabCanRun] = useState(true);

  const [agentPrompt, setAgentPrompt] = useState("");
  const [agentSteps, setAgentSteps] = useState(3);
  const [agentTemp, setAgentTemp] = useState(0.2);
  const [llmType, setLlmType] = useState("openai");
  const [llmApiBase, setLlmApiBase] = useState(TYPE_OPTIONS.openai.base);
  const [llmModel, setLlmModel] = useState(TYPE_OPTIONS.openai.models[0]);
  const [llmApiKey, setLlmApiKey] = useState("");
  const [agentMessage, setAgentMessage] = useState("Waiting for input...");
  const [agentMetrics, setAgentMetrics] = useState({});
  const [agentPrices, setAgentPrices] = useState([]);
  const [agentEquity, setAgentEquity] = useState([]);
  const [agentPositions, setAgentPositions] = useState([]);
  const [agentStartCash, setAgentStartCash] = useState(10000);
  const [agentCanRun, setAgentCanRun] = useState(true);

  const selectedStrategy = useMemo(
    () => strategies.find((strategy) => strategy.name === selectedName) || null,
    [strategies, selectedName]
  );

  const modelOptions = useMemo(() => {
    const config = TYPE_OPTIONS[llmType] || TYPE_OPTIONS.openai;
    return config.models || [];
  }, [llmType]);

  const labMetricItems = useMemo(
    () => computeMetrics(labMetrics, labPrices, labEquity, labStartCash),
    [labMetrics, labPrices, labEquity, labStartCash]
  );

  const agentMetricItems = useMemo(
    () => computeMetrics(agentMetrics, agentPrices, agentEquity, agentStartCash),
    [agentMetrics, agentPrices, agentEquity, agentStartCash]
  );

  function enableLabRun() {
    setLabCanRun(true);
  }

  function enableAgentRun() {
    setAgentCanRun(true);
  }

  useEffect(() => {
    const config = TYPE_OPTIONS[llmType] || TYPE_OPTIONS.openai;
    if (llmType !== "custom") {
      setLlmApiBase(config.base);
    }
    if (!llmModel || (config.models.length && !config.models.includes(llmModel))) {
      setLlmModel(config.models[0] || "");
    }
  }, [llmType]);

  useEffect(() => {
    async function loadStrategies() {
      const response = await fetch("/api/strategies");
      const data = await response.json();
      setStrategies(data);
      if (data.length) {
        setSelectedName(data[0].name);
        setParams(data[0].params || {});
      }
    }
    loadStrategies();
  }, []);

  useEffect(() => {
    if (selectedStrategy) {
      setParams(selectedStrategy.params || {});
      setLabCanRun(true);
    }
  }, [selectedStrategy?.name]);

  useEffect(() => {
    if (!endDate) {
      const end = new Date();
      const offsetMs = end.getTimezoneOffset() * 60 * 1000;
      setEndDate(new Date(end.getTime() - offsetMs).toISOString().slice(0, 10));
    }
  }, [endDate]);

  useEffect(() => {
    if (!startDate) {
      const base = endDate ? new Date(endDate) : new Date();
      const start = new Date(base);
      start.setFullYear(start.getFullYear() - 1);
      const offsetMs = start.getTimezoneOffset() * 60 * 1000;
      setStartDate(new Date(start.getTime() - offsetMs).toISOString().slice(0, 10));
    }
  }, [startDate, endDate]);

  async function runBacktest() {
    if (!selectedName) return;
    const payload = {
      strategy: selectedName,
      params,
      start_cash: Number(startCash || 10000),
      fee_bps: Number(feeBps || 1.0),
      ticker: ticker.trim() || null,
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
    setLabMetrics(result.metrics || {});
    setLabPrices(result.prices || []);
    setLabEquity(result.equity_curve || []);
    setLabPositions(result.positions || []);
    setLabStartCash(Number(startCash || 10000));
    setLabCanRun(false);
  }

  async function runAgent() {
    setAgentMessage("Running...");
    if (!agentPrompt.trim()) {
      alert("Please enter a prompt.");
      setAgentMessage("Waiting for input...");
      return;
    }
    const payload = {
      prompt: agentPrompt.trim(),
      max_steps: Number(agentSteps || 3),
      temperature: Number(agentTemp || 0.2),
      llm_type: llmType || null,
      llm_api_base: llmApiBase.trim() || null,
      llm_model: llmModel.trim() || null,
      llm_api_key: llmApiKey.trim() || null,
    };
    const response = await fetch("/api/agent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      let errorMessage = "Unable to run agent.";
      try {
        const errorData = await response.json();
        if (errorData.detail) {
          errorMessage = errorData.detail;
        }
      } catch (err) {
        errorMessage = "Unable to run agent.";
      }
      setAgentMessage(errorMessage);
      return;
    }
    const result = await response.json();
    const backtest = findBacktestResult(result);
    if (backtest) {
      const prices = backtest.data.prices || [];
      const equity = backtest.data.equity_curve || [];
      const positions = backtest.data.positions || [];
      setAgentMetrics(backtest.data.metrics || {});
      setAgentPrices(prices);
      setAgentEquity(equity);
      setAgentPositions(positions);
      setAgentStartCash(backtest.startCash);
      setAgentMessage(result.final || "Backtest result loaded.");
      setAgentCanRun(false);
      return;
    }
    setAgentMessage(result.final || "No backtest result returned.");
    setAgentCanRun(true);
  }

  const tabList = ["lab", "agent"];
  const labLabels = useMemo(() => labPrices.map((_, i) => i + 1), [labPrices]);
  const agentLabels = useMemo(() => agentPrices.map((_, i) => i + 1), [agentPrices]);

  const labPriceDataset = useMemo(
    () => ({
      label: "Price",
      data: labPrices,
      borderColor: "#0d6f85",
      backgroundColor: "rgba(13, 111, 133, 0.12)",
      tension: 0.25,
      yAxisID: "y",
    }),
    [labPrices]
  );

  const labEquityDataset = useMemo(
    () => ({
      label: "Equity",
      data: labEquity,
      borderColor: "#c85c32",
      backgroundColor: "rgba(200, 92, 50, 0.12)",
      tension: 0.25,
      yAxisID: "y1",
    }),
    [labEquity]
  );

  const labPositionDataset = useMemo(
    () => ({
      label: "Position",
      data: labPositions,
      borderColor: "#222",
      backgroundColor: "rgba(0, 0, 0, 0.1)",
      stepped: true,
      tension: 0,
    }),
    [labPositions]
  );

  const labPriceEquityDatasets = useMemo(
    () => [labPriceDataset, labEquityDataset],
    [labPriceDataset, labEquityDataset]
  );

  const labPositionDatasets = useMemo(
    () => [labPositionDataset],
    [labPositionDataset]
  );

  const agentPriceDataset = useMemo(
    () => ({
      label: "Price",
      data: agentPrices,
      borderColor: "#0d6f85",
      backgroundColor: "rgba(13, 111, 133, 0.12)",
      tension: 0.25,
      yAxisID: "y",
    }),
    [agentPrices]
  );

  const agentEquityDataset = useMemo(
    () => ({
      label: "Equity",
      data: agentEquity,
      borderColor: "#c85c32",
      backgroundColor: "rgba(200, 92, 50, 0.12)",
      tension: 0.25,
      yAxisID: "y1",
    }),
    [agentEquity]
  );

  const agentPositionDataset = useMemo(
    () => ({
      label: "Position",
      data: agentPositions,
      borderColor: "#222",
      backgroundColor: "rgba(0, 0, 0, 0.1)",
      stepped: true,
      tension: 0,
    }),
    [agentPositions]
  );

  const agentPriceEquityDatasets = useMemo(
    () => [agentPriceDataset, agentEquityDataset],
    [agentPriceDataset, agentEquityDataset]
  );

  const agentPositionDatasets = useMemo(
    () => [agentPositionDataset],
    [agentPositionDataset]
  );

  return (
    <div className="wrap">
      <header>
        <h1>Quant Strategy Lab using MCP server in Manual and LLM modes</h1>
        <p>Pick a strategy, tune parameters, and visualize equity curves with a lightweight backtester with Yahoo Finance data.</p>
      </header>

      <div className="tabs" role="tablist" aria-label="App sections">
        {tabList.map((tab, index) => {
          const isActive = activeTab === tab;
          const label = tab === "lab" ? "Manual Mode" : "LLM Mode";
          return (
            <TabButton
              key={tab}
              id={`tab-btn-${tab}`}
              controlsId={`tab-${tab}`}
              label={label}
              active={isActive}
              onClick={() => setActiveTab(tab)}
              onNavigate={(event) => {
                if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
                event.preventDefault();
                const direction = event.key === "ArrowRight" ? 1 : -1;
                const nextIndex = (index + direction + tabList.length) % tabList.length;
                setActiveTab(tabList[nextIndex]);
                const nextTab = document.getElementById(`tab-btn-${tabList[nextIndex]}`);
                if (nextTab) nextTab.focus();
              }}
            />
          );
        })}
      </div>

      <div
        id="tab-lab"
        className={`tab-panel ${activeTab === "lab" ? "active" : ""}`}
        role="tabpanel"
        aria-labelledby="tab-btn-lab"
        hidden={activeTab !== "lab"}
      >
        <div className="grid">
          <section className="card">
            <label htmlFor="strategy">Strategy</label>
            <select
              id="strategy"
              value={selectedName}
              onChange={(event) => {
                setSelectedName(event.target.value);
                enableLabRun();
              }}
            >
              {strategies.map((strategy) => (
                <option key={strategy.name} value={strategy.name}>
                  {strategy.name.replace(/_/g, " ")}
                </option>
              ))}
            </select>
            <div id="strategyDesc" className="muted">
              {selectedStrategy?.description || ""}
            </div>

            <div id="paramFields">
              {Object.entries(params).map(([key, value]) => (
                <div key={key}>
                  <label htmlFor={`param-${key}`}>{key.replace(/_/g, " ")}</label>
                  <input
                    id={`param-${key}`}
                    type="number"
                    step="0.1"
                    value={value}
                    onChange={(event) => {
                      const nextValue = Number(event.target.value);
                      setParams((prev) => ({ ...prev, [key]: nextValue }));
                      enableLabRun();
                    }}
                  />
                </div>
              ))}
            </div>

            <div className="row">
              <div>
                <label htmlFor="startCash">Start cash</label>
                <input
                  id="startCash"
                  type="number"
                  value={startCash}
                  onChange={(event) => {
                    setStartCash(Number(event.target.value));
                    enableLabRun();
                  }}
                />
              </div>
              <div>
                <label htmlFor="feeBps">Fee (bps)</label>
                <input
                  id="feeBps"
                  type="number"
                  step="0.1"
                  value={feeBps}
                  onChange={(event) => {
                    setFeeBps(Number(event.target.value));
                    enableLabRun();
                  }}
                />
              </div>
            </div>

            <label htmlFor="ticker">Ticker (Yahoo Finance)</label>
            <input
              id="ticker"
              list="tickerList"
              placeholder="AAPL"
              value={ticker}
              onChange={(event) => {
                setTicker(event.target.value);
                enableLabRun();
              }}
            />
            <datalist id="tickerList">
              <option value="AAPL"></option>
              <option value="MSFT"></option>
              <option value="NVDA"></option>
              <option value="TSLA"></option>
              <option value="SPY"></option>
              <option value="QQQ"></option>
            </datalist>

            <div className="row">
              <div>
                <label htmlFor="startDate">Start date</label>
                <input
                  id="startDate"
                  type="date"
                  value={startDate}
                  onChange={(event) => {
                    setStartDate(event.target.value);
                    enableLabRun();
                  }}
                />
              </div>
              <div>
                <label htmlFor="endDate">End date</label>
                <input
                  id="endDate"
                  type="date"
                  value={endDate}
                  onChange={(event) => {
                    setEndDate(event.target.value);
                    enableLabRun();
                  }}
                />
              </div>
            </div>
            <br/>
            <button id="runBtn" type="button" onClick={runBacktest} disabled={!labCanRun}>Run backtest</button>
          </section>

          <section className="card">
            <Legend />
            <MetricGrid items={labMetricItems} />
            <div className="chart-block">
              <h3>Price vs Equity</h3>
              <LineChart
                labels={labLabels}
                datasets={labPriceEquityDatasets}
                options={PRICE_EQUITY_OPTIONS}
                height={220}
              />
            </div>

            <div className="chart-block">
              <h3>Position</h3>
              <LineChart
                labels={labLabels}
                datasets={labPositionDatasets}
                options={POSITION_OPTIONS}
                height={160}
              />
            </div>
          </section>
        </div>
      </div>

      <div
        id="tab-agent"
        className={`tab-panel ${activeTab === "agent" ? "active" : ""}`}
        role="tabpanel"
        aria-labelledby="tab-btn-agent"
        hidden={activeTab !== "agent"}
      >
        <div className="grid">
          <section className="card">
            <div id="llmControls">
              <label htmlFor="llmType">LLM type</label>
              <select
                id="llmType"
                value={llmType}
                onChange={(event) => {
                  setLlmType(event.target.value);
                  enableAgentRun();
                }}
              >
                {TYPE_ORDER.map((key) => (
                  <option key={key} value={key}>{TYPE_OPTIONS[key].label}</option>
                ))}
              </select>
              <div className="muted">Choose a provider to auto-fill defaults.</div>

              <label htmlFor="llmApiBase">LLM API base</label>
              <input
                id="llmApiBase"
                type="text"
                placeholder="https://api.openai.com"
                readOnly={llmType !== "custom"}
                value={llmApiBase}
                onChange={(event) => {
                  setLlmApiBase(event.target.value);
                  enableAgentRun();
                }}
              />

              <label htmlFor="llmModel">Model</label>
              <input
                id="llmModel"
                list="llmModelList"
                placeholder="gpt-4o-mini"
                value={llmModel}
                onChange={(event) => {
                  setLlmModel(event.target.value);
                  enableAgentRun();
                }}
              />
              <datalist id="llmModelList">
                {modelOptions.map((model) => (
                  <option key={model} value={model}></option>
                ))}
              </datalist>

              <label htmlFor="llmApiKey">API key</label>
              <input
                id="llmApiKey"
                type="password"
                placeholder="sk-..."
                value={llmApiKey}
                onChange={(event) => {
                  setLlmApiKey(event.target.value);
                  enableAgentRun();
                }}
              />

              <label htmlFor="agentPrompt">Prompt</label>
              <textarea
                id="agentPrompt"
                placeholder="Run a backtest using sma_crossover with default params."
                value={agentPrompt}
                onChange={(event) => {
                  setAgentPrompt(event.target.value);
                  enableAgentRun();
                }}
              ></textarea>
              <div className="row">
                <div>
                  <label htmlFor="agentSteps">Max steps</label>
                  <input
                    id="agentSteps"
                    type="number"
                    min="1"
                    max="6"
                    value={agentSteps}
                    onChange={(event) => {
                      setAgentSteps(Number(event.target.value));
                      enableAgentRun();
                    }}
                  />
                </div>
                <div>
                  <label htmlFor="agentTemp">Temperature</label>
                  <input
                    id="agentTemp"
                    type="number"
                    step="0.1"
                    min="0"
                    max="1.5"
                    value={agentTemp}
                    onChange={(event) => {
                      setAgentTemp(Number(event.target.value));
                      enableAgentRun();
                    }}
                  />
                </div>
              </div>
              <div className="muted">LLM settings come from server environment variables.</div>
              <br/>
            </div>

            <button id="agentRunBtn" type="button" onClick={runAgent} disabled={!agentCanRun}>Run backtest</button>
          </section>

          <section className="card">
            <Legend />

            <div id="agentMessage" className="agent-message">{agentMessage}</div>

            <MetricGrid items={agentMetricItems} />

            <div className="chart-block">
              <h3>Price vs Equity</h3>
              <LineChart
                labels={agentLabels}
                datasets={agentPriceEquityDatasets}
                options={PRICE_EQUITY_OPTIONS}
                height={220}
              />
            </div>

            <div className="chart-block">
              <h3>Position</h3>
              <LineChart
                labels={agentLabels}
                datasets={agentPositionDatasets}
                options={POSITION_OPTIONS}
                height={160}
              />
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
