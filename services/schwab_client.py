import os
from typing import List, Dict, Any
from dataclasses import dataclass

DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

@dataclass
class Position:
    symbol: str
    quantity: float
    avg_price: float
    last: float | None
    unrealized_pl: float

def get_positions() -> List[Position]:
    # TODO: Replace with real Schwab API call
    # This is a deterministic stub for UI and strategy development.
    demo = [
        Position(symbol="AAPL", quantity=10, avg_price=165.50, last=169.20, unrealized_pl=(169.20-165.50)*10),
        Position(symbol="NVDA", quantity=3, avg_price=120.00, last=118.10, unrealized_pl=(118.10-120.00)*3),
    ]
    return demo

def place_order(order: Dict[str, Any]) -> Dict[str, Any]:
    """Send an order. In DRY_RUN mode we don't call the broker."""
    if DRY_RUN:
        return {"ok": True, "dry_run": True, "order": order, "broker_order_id": None}
    # ---- PSEUDOCODE ----
    # 1) Ensure you have a valid access token (refresh if needed).
    # 2) Build the order payload per Schwab spec.
    # 3) POST to the orders endpoint for SCHWAB_ACCOUNT_ID.
    # 4) Handle errors + return broker order id.
    # --------------------
    raise NotImplementedError("Implement Schwab API order placement here.")
