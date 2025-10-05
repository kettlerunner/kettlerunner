import os
from typing import Dict, Any, List
import httpx

# Minimal wrapper. Replace with official OpenAI SDK if preferred.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

def build_strategy_prompt(positions: List[dict], news: List[str]) -> str:
    return (
        "You are a cautious intraday strategy assistant.\n"
        "Given current positions and news, propose at most 3 trades with rationales, sizing, "
        "and clear entry/exit rules. Favor risk management and small size.\n"
        f"POSITIONS: {positions}\nNEWS: {news}\n"
        "Output JSON list: [{symbol, side, quantity, order_type, limit_price|null, rationale}]"
    )

def propose_trades(positions: List[dict], news: List[str]) -> List[Dict[str, Any]]:
    if not OPENAI_API_KEY:
        # Fallback demo
        return [{"symbol": "AAPL", "side": "BUY", "quantity": 1, "order_type": "MKT", "limit_price": None, "rationale": "Demo stub"}]
    prompt = build_strategy_prompt(positions, news)
    # Simple call via Responses-style (replace with your preferred endpoint if needed)
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "system", "content": "You are an expert trading assistant."},
                     {"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"}
    }
    try:
        with httpx.Client(timeout=30) as client:
            r = client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            # Extract text
            content = data["choices"][0]["message"]["content"]
    except Exception:
        return [{"symbol": "AAPL", "side": "BUY", "quantity": 1, "order_type": "MKT", "limit_price": None, "rationale": "Fallback"}]

    # Attempt to parse JSON object or list
    import json
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "trades" in parsed:
            return parsed["trades"]
        elif isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return []
