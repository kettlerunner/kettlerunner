from fastapi import APIRouter, Request, Depends, HTTPException, Form, Response
from fastapi.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from fastapi import FastAPI
from passlib.context import CryptContext
import os

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Single-user hashed password stored in env or file for simplicity
ADMIN_USER = os.getenv("ADMIN_USER", "daniel")
ADMIN_HASH = os.getenv("ADMIN_HASH", "")  # set with: from passlib.hash import bcrypt; bcrypt.hash("yourpass")

router = APIRouter()

def is_authed(request: Request) -> bool:
    return bool(request.session.get("user") == ADMIN_USER)

def require_auth(request: Request):
    if not is_authed(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

@router.get("/login")
def login_form(request: Request):
    # Login is typically handled by Nginx Basic Auth already, this is optional app-level auth.
    return {"message": "POST username/password to /login to start a session (optional)."}

@router.post("/login")
def login(request: Request, response: Response, username: str = Form(...), password: str = Form(...)):
    if username != ADMIN_USER or not ADMIN_HASH:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # In this simple example we don't verify password if using Nginx basic auth; otherwise hash check:
    # if not pwd_context.verify(password, ADMIN_HASH): raise HTTPException(status_code=401, detail="Unauthorized")
    request.session["user"] = ADMIN_USER
    return RedirectResponse(url="/admin", status_code=303)

@router.post("/logout")
def logout(request: Request):
    request.session.clear()
    return {"ok": True}
