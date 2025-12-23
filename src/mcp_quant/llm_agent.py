from __future__ import annotations

import json
import os
from datetime import date
from typing import Any, Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from mcp_quant.mcp_client import MCPClientError, mcp_client


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
    "If the user mentions a ticker or real market data, call fetch_yahoo_prices first. "
    "Use YYYY-MM-DD for dates or a short range like 1y, 6mo, 30d. "
    "If you include range, omit start_date/end_date.\n"
    "If you need prices and they are not provided by the user, call sample_price_series first."
)


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
        raise LLMResponseError("LLM response did not include JSON")
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"Invalid JSON from LLM: {exc}") from exc


def _call_llm(
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
    request = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    try:
        with urlopen(request, timeout=120) as response:
            data = json.load(response)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore") if exc.fp else ""
        raise LLMResponseError(f"LLM HTTP error {exc.code}: {body}") from exc
    except URLError as exc:
        raise LLMResponseError(f"LLM connection error: {exc.reason}") from exc
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
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Today's date is {date.today().isoformat()}."},
        {"role": "user", "content": prompt},
    ]
    steps: List[Dict[str, Any]] = []
    last_prices: List[float] | None = None

    for _ in range(max_steps):
        content = _call_llm(
            messages,
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
                "steps": steps,
            }
        tool_name = parsed.get("tool")
        if not tool_name:
            raise LLMResponseError("LLM response missing tool or final")
        arguments = parsed.get("arguments") or {}
        if tool_name == "run_backtest" and "prices" not in arguments and last_prices is not None:
            arguments["prices"] = last_prices
        log_args = arguments
        if isinstance(arguments, dict) and isinstance(arguments.get("prices"), list):
            log_args = {**arguments, "prices": f"<{len(arguments['prices'])} prices>"}
        try:
            result = await mcp_client.call_mcp_tool(tool_name, arguments)
            # print(f"Tool result for {tool_name}: {json.dumps(result)}")
        except MCPClientError as exc:
            raise LLMResponseError(f"MCP tool error: {exc}") from exc
        steps.append({"tool": tool_name, "arguments": arguments, "result": result})
        if isinstance(result, list) and all(isinstance(item, (int, float)) for item in result):
            last_prices = result
        if tool_name == "run_backtest":
            return {
                "final": "Tool executed.",
                "steps": steps,
            }
        messages.append({"role": "assistant", "content": content})
        messages.append(
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
        "steps": steps,
    }
