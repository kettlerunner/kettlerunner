from typing import List, Dict, Any

def generate(positions: List[dict]) -> List[Dict[str, Any]]:
    """A tiny sample strategy: if NVDA is red, buy 1 share as a demo."""
    trades = []
    for p in positions:
        if p['symbol'] == 'NVDA' and p.get('last') and p['last'] < p['avg_price']:
            trades.append({
                "symbol": "NVDA", "side": "BUY", "quantity": 1, "order_type": "MKT", "limit_price": None,
                "rationale": "Demo buy-the-dip"
            })
    return trades
