import os
from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from dotenv import load_dotenv
load_dotenv()

from .auth import router as auth_router
from services import schwab_client, openai_client
from db import init_db, SessionLocal
from models import Trade, Order

app = FastAPI(title="Kettlerunner Trading Portal")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Optional session middleware if you use app-level auth
from starlette.middleware.sessions import SessionMiddleware
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY", "dev_secret"))

@app.on_event("startup")
def on_startup():
    init_db()

app.include_router(auth_router)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/news", response_class=HTMLResponse)
def news(request: Request):
    # TODO: pull curated news (placeholder)
    items = [
        {"title": "Market opens flat", "source": "Demo Feed", "time": "09:32"},
        {"title": "Semis up on AI capex", "source": "Demo Feed", "time": "10:04"},
    ]
    return templates.TemplateResponse("news.html", {"request": request, "items": items})

@app.get("/positions", response_class=HTMLResponse)
def positions(request: Request):
    positions = schwab_client.get_positions()
    return templates.TemplateResponse("positions.html", {"request": request, "positions": positions})

@app.get("/history", response_class=HTMLResponse)
def history(request: Request):
    db = SessionLocal()
    trades = db.query(Trade).order_by(Trade.timestamp.desc()).limit(200).all()
    return templates.TemplateResponse("history.html", {"request": request, "trades": trades})

@app.get("/admin", response_class=HTMLResponse)
def admin(request: Request):
    # Protected by Nginx Basic Auth (see deploy/nginx_trader.conf)
    return templates.TemplateResponse("admin.html", {"request": request})

@app.post("/admin/run")
def admin_run_job(request: Request):
    # Trigger strategy once (synchronous)
    from scripts.run_jobs import run_once
    result = run_once()
    return {"ok": True, "result": result}
