import os, json
from datetime import datetime
from db import SessionLocal
from models import Trade, Order
from services import schwab_client, openai_client
from strategies import buy_dip

def run_once():
    db = SessionLocal()

    # 1) Fetch positions and stub news
    positions_dataclasses = schwab_client.get_positions()
    positions = [p.__dict__ for p in positions_dataclasses]
    news = ["FOMC minutes later today", "AI spending accelerates in cloud"]

    # 2) Get AI-proposed trades (bounded)
    ai_trades = openai_client.propose_trades(positions, news)[:3]

    # 3) Combine with one deterministic rule-based strategy
    rule_trades = buy_dip.generate(positions)
    proposals = rule_trades + ai_trades

    # 4) Place orders (or DRY_RUN)
    results = []
    for prop in proposals:
        od = Order(
            symbol=prop["symbol"],
            side=prop["side"],
            quantity=float(prop["quantity"]),
            order_type=prop.get("order_type","MKT"),
            limit_price=prop.get("limit_price")
        )
        db.add(od)
        db.commit(); db.refresh(od)

        res = schwab_client.place_order(prop)
        placed = res.get("ok", False) and not res.get("dry_run", False)
        od.placed = placed
        od.broker_order_id = res.get("broker_order_id")
        db.add(od); db.commit()

        t = Trade(symbol=prop["symbol"], side=prop["side"], quantity=float(prop["quantity"]), price=prop.get("limit_price"), status="PAPER" if res.get("dry_run") else "SENT")
        db.add(t); db.commit()

        results.append({"proposal": prop, "result": res})

    return {"time": datetime.utcnow().isoformat(), "results": results}

if __name__ == "__main__":
    out = run_once()
    print(json.dumps(out, indent=2))
