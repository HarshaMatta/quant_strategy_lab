from __future__ import annotations

from dataclasses import dataclass, field
import difflib
import json
import os
import re
from datetime import date
from typing import Any, Dict, List

import httpx

from mcp_quant.mcp_client import MCPClientError, mcp_client
from mcp_quant.strategies import list_strategies as list_strategy_specs


class LLMConfigError(RuntimeError):
    pass


class LLMResponseError(RuntimeError):
    pass


SYSTEM_PROMPT = (
    "You are an MCP tool-calling agent for quantitative strategy backtests. "
    "You can call these tools:\n"
    "- list_strategies: {} arguments\n"
    "- get_strategy_schema: {\"name\": string}\n"
    "- fetch_yahoo_prices: {\"ticker\": string, \"start_date\": \"YYYY-MM-DD\", "
    "\"end_date\": \"YYYY-MM-DD\", \"range\": \"1y\"}\n"
    "- run_backtest: {\"prices\": [float], \"strategy\": string, \"params\": object, "
    "\"start_cash\": float, \"fee_bps\": float}\n\n"
    "Respond with JSON only. Choose exactly one of:\n"
    "1) {\"tool\": \"<tool_name>\", \"arguments\": { ... }}\n"
    "2) {\"final\": \"<concise answer>\"}\n\n"
    "If strategy text contains typos (e.g., smacrossover), resolve it to a canonical strategy name "
    "before calling get_strategy_schema or run_backtest.\n"
    "If the user mentions a ticker or real market data, call fetch_yahoo_prices first. "
    "Use YYYY-MM-DD for dates or a short range like 1y, 6mo, 30d. "
    "If you include range, omit start_date/end_date.\n"
    "If you need prices and they are not provided by the user, call sample_price_series first."
)


_STRATEGY_ALIASES: Dict[str, str] = {
    "sma": "sma_crossover",
    "smacrossover": "sma_crossover",
    "movingaverage": "sma_crossover",
    "moving_average": "sma_crossover",
    "rsi": "rsi_reversion",
    "channel": "channel_breakout",
    "breakout": "channel_breakout",
}


def _normalize_strategy_name(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return cleaned.strip("_")


def _compact_strategy_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _resolve_strategy_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Strategy name must be a non-empty string")

    strategy_names = [spec.name for spec in list_strategy_specs()]
    normalized_to_canonical = {_normalize_strategy_name(name): name for name in strategy_names}
    compact_to_canonical = {_compact_strategy_name(name): name for name in strategy_names}

    normalized = _normalize_strategy_name(value)
    if normalized in normalized_to_canonical:
        return normalized_to_canonical[normalized]

    alias = _STRATEGY_ALIASES.get(normalized)
    if alias in strategy_names:
        return alias

    compact = _compact_strategy_name(value)
    if compact in compact_to_canonical:
        return compact_to_canonical[compact]

    compact_aliases = {_compact_strategy_name(k): v for k, v in _STRATEGY_ALIASES.items()}
    compact_alias = compact_aliases.get(compact)
    if compact_alias in strategy_names:
        return compact_alias

    closest = difflib.get_close_matches(normalized, list(normalized_to_canonical.keys()), n=1, cutoff=0.72)
    if closest:
        return normalized_to_canonical[closest[0]]

    closest_compact = difflib.get_close_matches(compact, list(compact_to_canonical.keys()), n=1, cutoff=0.72)
    if closest_compact:
        return compact_to_canonical[closest_compact[0]]

    available = ", ".join(sorted(strategy_names))
    raise ValueError(f"Unknown strategy: {value}. Available strategies: {available}")


def _resolve_strategy_args(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(arguments, dict):
        return {}
    resolved = dict(arguments)

    if tool_name == "run_backtest":
        raw_name = resolved.get("strategy")
        if isinstance(raw_name, str) and raw_name.strip():
            try:
                resolved["strategy"] = _resolve_strategy_name(raw_name)
            except ValueError as exc:
                raise LLMResponseError(str(exc)) from exc
    elif tool_name == "get_strategy_schema":
        raw_name = resolved.get("name")
        if isinstance(raw_name, str) and raw_name.strip():
            try:
                resolved["name"] = _resolve_strategy_name(raw_name)
            except ValueError as exc:
                raise LLMResponseError(str(exc)) from exc
    return resolved


@dataclass
class AgentState:
    """Tracks the state of the LLM agent during execution."""
    messages: List[Dict[str, str]] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    last_prices: List[float] | None = None
    tool_call_count: int = 0

    def add_step(self, tool: str, arguments: Dict[str, Any], result: Any) -> None:
        """Add a tool execution step to the history."""
        self.steps.append({"tool": tool, "arguments": arguments, "result": result})
        self.tool_call_count += 1

    def update_prices(self, result: Any) -> None:
        """Update last_prices if result contains a price series."""
        if isinstance(result, list) and all(isinstance(item, (int, float)) for item in result):
            self.last_prices = result


_LLM_TYPE_DEFAULTS: Dict[str, Dict[str, str]] = {
    "openai": {"base": "https://api.openai.com", "model": "gpt-4o-mini"},
    "gemini": {"base": "https://generativelanguage.googleapis.com/v1beta/openai", "model": "gemini-1.5-pro"},
    "openrouter": {"base": "https://openrouter.ai/api", "model": "openai/gpt-4o-mini"},
    "groq": {"base": "https://api.groq.com/openai", "model": "llama-3.1-70b-versatile"},
    "together": {"base": "https://api.together.xyz", "model": "meta-llama/Llama-3.1-70B-Instruct-Turbo"},
    "fireworks": {"base": "https://api.fireworks.ai/inference", "model": "accounts/fireworks/models/llama-v3p1-70b-instruct"},
    "deepinfra": {"base": "https://api.deepinfra.com/v1/openai", "model": "meta-llama/Meta-Llama-3.1-70B-Instruct"},
    "perplexity": {"base": "https://api.perplexity.ai", "model": "sonar"},
    "mistral": {"base": "https://api.mistral.ai", "model": "mistral-large-latest"},
}


def _llm_config(
    llm_type: str | None = None,
    llm_api_base: str | None = None,
    llm_model: str | None = None,
    llm_api_key: str | None = None,
) -> tuple[str, str, str | None]:
    defaults = _LLM_TYPE_DEFAULTS.get((llm_type or "").lower(), {})
    base = (llm_api_base or defaults.get("base") or os.getenv("LLM_API_BASE") or "https://api.openai.com").rstrip("/")
    model = llm_model or defaults.get("model") or os.getenv("LLM_MODEL") or "gpt-4o-mini"
    key = llm_api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key and base.startswith("https://api.openai.com"):
        raise LLMConfigError("Set LLM_API_KEY or OPENAI_API_KEY for OpenAI API access")
    return base, model, key


def _extract_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        snippet = text[:100] + "..." if len(text) > 100 else text
        raise LLMResponseError(f"LLM response did not include JSON. Response: {snippet}")
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        json_text = cleaned[start : end + 1]
        snippet = json_text[:200] + "..." if len(json_text) > 200 else json_text
        raise LLMResponseError(f"Invalid JSON from LLM: {exc}. Text: {snippet}") from exc


async def _call_llm(
    messages: List[Dict[str, str]],
    temperature: float,
    llm_type: str | None = None,
    llm_api_base: str | None = None,
    llm_model: str | None = None,
    llm_api_key: str | None = None,
) -> str:
    base, model, key = _llm_config(
        llm_type=llm_type,
        llm_api_base=llm_api_base,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
    )
    url = f"{base}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        body = exc.response.text
        raise LLMResponseError(f"LLM HTTP error {exc.response.status_code}: {body}") from exc
    except httpx.RequestError as exc:
        raise LLMResponseError(f"LLM connection error: {exc}") from exc
    choices = data.get("choices") or []
    if not choices:
        raise LLMResponseError("LLM response missing choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not content:
        raise LLMResponseError("LLM response missing content")
    return content


async def run_llm_agent(
    prompt: str,
    max_steps: int = 3,
    temperature: float = 0.2,
    llm_type: str | None = None,
    llm_api_base: str | None = None,
    llm_model: str | None = None,
    llm_api_key: str | None = None,
) -> Dict[str, Any]:
    max_steps = max(1, min(int(max_steps), 6))
    temperature = max(0.0, min(float(temperature), 1.5))

    # Initialize agent state
    state = AgentState(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"Today's date is {date.today().isoformat()}."},
            {"role": "user", "content": prompt},
        ]
    )

    for _ in range(max_steps):
        content = await _call_llm(
            state.messages,
            temperature,
            llm_type=llm_type,
            llm_api_base=llm_api_base,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
        )
        parsed = _extract_json(content)
        if "final" in parsed:
            return {
                "final": parsed["final"],
                "steps": state.steps,
            }
        tool_name = parsed.get("tool")
        if not tool_name:
            parsed_str = json.dumps(parsed)[:200]
            raise LLMResponseError(f"LLM response missing 'tool' or 'final' field. Response: {parsed_str}")
        arguments = _resolve_strategy_args(tool_name, parsed.get("arguments") or {})
        if tool_name == "run_backtest" and "prices" not in arguments and state.last_prices is not None:
            arguments["prices"] = state.last_prices
        log_args = arguments
        if isinstance(arguments, dict) and isinstance(arguments.get("prices"), list):
            log_args = {**arguments, "prices": f"<{len(arguments['prices'])} prices>"}
        try:
            result = await mcp_client.call_mcp_tool(tool_name, arguments)
        except MCPClientError as exc:
            raise LLMResponseError(f"MCP tool error: {exc}") from exc

        # Update state
        state.add_step(tool_name, arguments, result)
        state.update_prices(result)

        if tool_name == "run_backtest":
            return {
                "final": "Tool executed.",
                "steps": state.steps,
            }
        state.messages.append({"role": "assistant", "content": content})
        state.messages.append(
            {
                "role": "user",
                "content": (
                    f"Tool result for {tool_name}: {json.dumps(result)}. "
                    "If you want to reuse the last price series, omit prices in run_backtest. "
                    "Provide the next action as JSON."
                ),
            }
        )

    return {
        "final": "Max tool steps reached without a final response.",
        "steps": state.steps,
    }
