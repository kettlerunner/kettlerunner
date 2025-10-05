# trader.py
# ------------------------------------------------------------
# Multi-agent trading orchestrator using OpenAI Agents SDK.
# Strict JSON schema compatible (no dict/Any in output models).
# Tools:
#   - fetch_price_history (yfinance)
#   - calc_indicators (pandas/numpy TA)
#   - compute_position (risk-based sizing)
#   - plan_oco (typed OCO payload)
#
# Run:
#   setx OPENAI_API_KEY "sk-..."
#   pip install openai-agents pydantic pandas numpy yfinance
#   python trader.py
#
# NOTE: Educational example, NOT financial advice.

from __future__ import annotations

import os
import json
import math
import datetime as dt
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import warnings

try:
    import yfinance as yf
except ImportError:
    yf = None

from pydantic import BaseModel, Field, ConfigDict

from agents import (
    Agent,
    Runner,
    WebSearchTool,
    function_tool,
    ModelSettings,
    SQLiteSession,
)

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

# ============================
# Strict OUTPUT MODELS
# ============================

class TechnicalData(BaseModel):
    model_config = ConfigDict(extra='forbid')
    symbol: str
    last_close: float
    sma_20: float
    sma_50: float
    ema_21: float
    rsi_14: float
    macd: float
    macd_signal: float
    atr_14: float
    trend: Literal["up", "down", "sideways"]
    support_levels: List[float] = Field(default_factory=list)
    resistance_levels: List[float] = Field(default_factory=list)
    sample_window_days: int = 180
    notes: str = ""


class MarketTrends(BaseModel):
    model_config = ConfigDict(extra='forbid')
    symbol: str
    bull_bear: Literal["bullish", "bearish", "mixed"]
    summary: str
    drivers: List[str]
    tickers_mentioned: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class FactCheck(BaseModel):
    model_config = ConfigDict(extra='forbid')
    ok: bool
    issues: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)


class Strategy(BaseModel):
    model_config = ConfigDict(extra='forbid')
    approach: Literal["momentum", "mean_reversion", "breakout", "pullback", "news_event"]
    direction: Literal["long", "short"]
    entry: float
    stop: float
    take_profit: float
    max_hold_days: int = 5
    expected_risk_reward: float
    justification: str


# ---- Typed OCO payloads (no plain dicts) ----

class OCOLeg(BaseModel):
    model_config = ConfigDict(extra='forbid')
    type: Literal["LIMIT", "STOP", "STOP_LIMIT", "MARKET"]
    side: Literal["BUY", "SELL"]
    qty: int
    price: Optional[float] = None
    stopPrice: Optional[float] = None


class OCOOrderPayload(BaseModel):
    model_config = ConfigDict(extra='forbid')
    orderType: Literal["OCO"] = "OCO"
    symbol: str
    side: Literal["BUY", "SELL"]
    timeInForce: Literal["DAY", "GTC"] = "GTC"
    legs: List[OCOLeg]


class OrderPlan(BaseModel):
    model_config = ConfigDict(extra='forbid')
    symbol: str
    side: Literal["buy", "sell"]
    qty: int
    entry: float
    stop: float
    take_profit: float
    tif: Literal["DAY", "GTC"] = "GTC"
    broker_payload: OCOOrderPayload
    rationale: str
    citations: List[str] = Field(default_factory=list)


# ============================
# TOOLS
# ============================

@function_tool
def fetch_price_history(symbol: str, period_days: int = 180, interval: str = "1d") -> str:
    """
    Pull OHLCV history via yfinance. Returns a JSON array of rows:
      [{"date": ISO, "open": float, "high": float, "low": float, "close": float, "volume": float}, ...]
    """
    if yf is None:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")
    end = dt.date.today()
    start = end - dt.timedelta(days=period_days + 10)
    df = yf.download(
        symbol,
        start=start.isoformat(),
        end=end.isoformat(),
        interval=interval,
        progress=False,
        auto_adjust=True,   # ← make it explicit; silences warning
        actions=False       # we don't need dividends/splits rows for TA
    )
    if df.empty:
        raise RuntimeError(f"No data for {symbol}")
    df = df.rename(columns=str.lower).reset_index()
    out = []
    for _, r in df.iterrows():
        # pandas datetime to python datetime to ISO
        d = r["Date"].to_pydatetime() if hasattr(r["Date"], "to_pydatetime") else r["Date"]
        out.append({
            "date": d.isoformat(),
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": float(r["volume"]),
        })
    return json.dumps(out)


def _rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ema_up = up.ewm(alpha=1/period, adjust=False).mean()
    ema_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ema_up / (ema_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


@function_tool
def calc_indicators(candles_json: str) -> str:
    """
    Compute TA: last_close, SMA20/50, EMA21, RSI14, MACD(12,26,9), ATR14, crude trend & S/R.
    Input is JSON from fetch_price_history. Returns a JSON object with the fields TechnicalData needs.
    """
    candles = pd.DataFrame(json.loads(candles_json))
    close = candles["close"]
    high = candles["high"]
    low = candles["low"]

    last_close = float(close.iloc[-1])
    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_last = float(macd.iloc[-1])
    signal_last = float(signal.iloc[-1])

    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr14 = float(tr.rolling(14).mean().iloc[-1])

    trend = "up" if (sma20 > sma50 and last_close > sma20) else ("down" if (sma20 < sma50 and last_close < sma20) else "sideways")

    window = 20
    supports = candles["low"].rolling(window).min().iloc[-5:].round(2).dropna().unique().tolist()
    resistances = candles["high"].rolling(window).max().iloc[-5:].round(2).dropna().unique().tolist()

    out = {
        "last_close": last_close,
        "sma_20": sma20,
        "sma_50": sma50,
        "ema_21": ema21,
        "rsi_14": _rsi(close, 14),
        "macd": macd_last,
        "macd_signal": signal_last,
        "atr_14": atr14,
        "trend": trend,
        "support_levels": supports,
        "resistance_levels": resistances
    }
    return json.dumps(out)


@function_tool
def compute_position(entry: float,
                     stop: float,
                     direction: Literal["long", "short"],
                     account_size: float,
                     risk_per_trade: float = 0.01,
                     max_position_pct: float = 0.25) -> str:
    """
    Risk-based position sizing.
    Returns JSON:
      {"qty": int, "risk_amount": float, "per_share_risk": float, "max_cost": float, "cost": float}
    """
    if entry <= 0:
        raise ValueError("Entry must be > 0.")
    per_share_risk = abs(entry - stop)
    if per_share_risk <= 0:
        raise ValueError("Stop must differ from entry.")
    risk_amount = max(0.0, float(account_size) * float(risk_per_trade))
    qty_by_risk = int(max(1, math.floor(risk_amount / per_share_risk)))
    # Cap notional exposure
    max_cost = float(account_size) * float(max_position_pct)
    notional = qty_by_risk * (entry if direction == "long" else max(entry, 0.01))
    qty = qty_by_risk
    if notional > max_cost and entry > 0:
        qty = max(1, int(math.floor(max_cost / entry)))
    return json.dumps({
        "qty": qty,
        "risk_amount": risk_amount,
        "per_share_risk": per_share_risk,
        "max_cost": max_cost,
        "cost": qty * entry
    })


@function_tool
def plan_oco(symbol: str,
             direction: Literal["long", "short"],
             entry: float,
             stop: float,
             take_profit: float,
             qty: int,
             tif: Literal["DAY", "GTC"] = "GTC") -> str:
    """
    Build a typed OCO order payload and return it as JSON (OCOOrderPayload schema).
    """
    side_txt = "buy" if direction == "long" else "sell"
    opp_side = "SELL" if side_txt == "buy" else "BUY"
    payload = OCOOrderPayload(
        symbol=symbol,
        side=side_txt.upper(),
        timeInForce=tif,
        legs=[
            OCOLeg(type="LIMIT", side=side_txt.upper(), qty=qty, price=round(entry, 2)),
            OCOLeg(type="STOP", side=opp_side, qty=qty, stopPrice=round(stop, 2)),
            OCOLeg(type="LIMIT", side=opp_side, qty=qty, price=round(take_profit, 2)),
        ]
    )
    return payload.model_dump_json()


# ============================
# AGENTS
# ============================

technical_data_agent = Agent(
    name="Technical data fetcher",
    instructions=(
        "Given a stock symbol, call fetch_price_history then calc_indicators. "
        "Return a TechnicalData JSON. Do not guess."
    ),
    tools=[fetch_price_history, calc_indicators],
    model_settings=ModelSettings(tool_choice="required"),
    output_type=TechnicalData,
)

news_trends_agent = Agent(
    name="Market news scanner",
    instructions=(
        "Use web search to scan very recent macro and symbol-specific news. "
        "Summarize sentiment and drivers. Include at least 3 source URLs in 'sources'. "
        "Return MarketTrends."
    ),
    tools=[WebSearchTool()],
    model_settings=ModelSettings(tool_choice="required"),
    output_type=MarketTrends,
)

fact_checker_agent = Agent(
    name="Fact checker",
    instructions=(
        "Validate the key claims from the news scanner and the technical state. "
        "If something is weak, missing corroboration, or contradicted, set ok=false, "
        "list issues, and include evidence URLs. Else ok=true with supporting links."
    ),
    tools=[WebSearchTool()],
    model_settings=ModelSettings(tool_choice="required"),
    output_type=FactCheck,
)

technical_analyst_agent = Agent(
    name="Technical analyst",
    instructions=(
        "Combine TechnicalData and MarketTrends to propose a short-term Strategy (≤5 trading days). "
        "Use practical levels: "
        "  - entry near market or breakout/pullback trigger; "
        "  - stop ~1×ATR beyond invalidation; "
        "  - take_profit ~2–3× risk or at nearby resistance/support. "
        "Pick approach from ['momentum','mean_reversion','breakout','pullback','news_event'] "
        "and justify succinctly. Provide expected_risk_reward."
    ),
    output_type=Strategy,
)

aggressive_trader_agent = Agent(
    name="Aggressive trader",
    instructions=(
        "Given Strategy and account parameters (account_size, risk_per_trade, max_position_pct), "
        "FIRST call compute_position(entry, stop, direction, ...) to get 'qty'. "
        "THEN call plan_oco(...) with that qty to get an OCO payload JSON. "
        "Return an OrderPlan object: broker_payload must exactly match the JSON returned by plan_oco "
        "(i.e., parse it as the OCOOrderPayload object). Include a concise rationale and pass through "
        "citations from MarketTrends/FactCheck sources if present."
    ),
    tools=[compute_position, plan_oco],
    model_settings=ModelSettings(tool_choice="required"),
    output_type=OrderPlan,
)

# Manager pattern: orchestrator coordinates sub-agents as tools
orchestrator = Agent(
    name="Trading orchestrator",
    instructions=(
        "You coordinate specialists to analyze a stock and produce an executable OCO plan.\n"
        "Pipeline:\n"
        "1) Call technical_data_agent to get TechnicalData.\n"
        "2) Call news_trends_agent to get MarketTrends (include sources).\n"
        "3) Call fact_checker_agent — if ok=false, STOP and report issues.\n"
        "4) Call technical_analyst_agent to produce Strategy consistent with both.\n"
        "5) Call aggressive_trader_agent to size and build the OCO order.\n"
        "Return the OrderPlan from step 5."
    ),
    tools=[
        technical_data_agent.as_tool(
            tool_name="technical_data_agent",
            tool_description="Fetch price history and indicators; returns TechnicalData."
        ),
        news_trends_agent.as_tool(
            tool_name="news_trends_agent",
            tool_description="Scan recent news & macro; returns MarketTrends with 'sources'."
        ),
        fact_checker_agent.as_tool(
            tool_name="fact_checker_agent",
            tool_description="Validate claims; returns FactCheck with evidence URLs."
        ),
        technical_analyst_agent.as_tool(
            tool_name="technical_analyst_agent",
            tool_description="Form a Strategy with entry/stop/take_profit and justification."
        ),
        aggressive_trader_agent.as_tool(
            tool_name="aggressive_trader_agent",
            tool_description="Size position and build OCO payload; returns OrderPlan."
        ),
    ],
    model_settings=ModelSettings(tool_choice="required"),
    output_type=OrderPlan,
)

# ============================
# RUNNER
# ============================

def run_once(symbol: str,
             account_size: float = 25000.0,
             risk_per_trade: float = 0.01,
             max_position_pct: float = 0.25):
    """
    Single-symbol run. Uses SQLiteSession to preserve context between runs if desired.
    """
    session = SQLiteSession(f"trade-{symbol}")
    prompt = (
        f"Symbol: {symbol}\n"
        f"Account size: {account_size}\n"
        f"Risk per trade: {risk_per_trade:.3f}\n"
        f"Max position %: {max_position_pct:.2f}\n"
        f"Deliver an OrderPlan. Keep rationale tight."
    )
    result = Runner.run_sync(
        orchestrator,
        prompt,
        session=session,
        context={
            "symbol": symbol,
            "account_size": account_size,
            "risk_per_trade": risk_per_trade,
            "max_position_pct": max_position_pct,
        },
        max_turns=20,
    )
    plan: OrderPlan = result.final_output  # typed
    print("\n=== ORDER PLAN ===")
    print(plan.model_dump_json(indent=2))


if __name__ == "__main__":
    sym = os.getenv("SYMBOL", "CRM").upper()
    acct = float(os.getenv("ACCOUNT", "25000"))
    risk = float(os.getenv("RISK", "0.01"))
    maxpct = float(os.getenv("MAXPOS", "0.25"))
    run_once(sym, acct, risk, maxpct)
