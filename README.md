# Kettlerunner Trading Portal (FastAPI on a DO Droplet)

A minimal, production-lean starter to:
- Serve a public site (news, open positions, trade history, portfolio perf).
- Keep private endpoints behind Nginx Basic Auth for strategy + order execution.
- Run Python strategies using OpenAI + Charles Schwab API (stubs included).
- Log trades/orders in SQLite (swap for Postgres easily).
- Deploy automatically from GitHub via Actions using SSH/rsync.
- Run scheduled jobs via systemd timer.

> ⚠️ **Risk**: This code is for educational/demo purposes. Start in **DRY_RUN** mode.
> Verify any order routing logic carefully and comply with Schwab API rules. Paper trade first.

---

## 1) Repo structure

```
app/
  main.py
  auth.py
  templates/
    base.html
    index.html
    news.html
    positions.html
    history.html
    admin.html
  static/
db.py
models.py
services/
  openai_client.py
  schwab_client.py
strategies/
  buy_dip.py
scripts/
  run_jobs.py
gunicorn_conf.py
requirements.txt
.env.example
deploy/
  bootstrap.sh
  nginx_trader.conf
  trader.service
  trader.timer
.github/
  workflows/
    deploy.yml
```

---

## 2) Droplet bootstrap (first run, on the server)

**Create a droplet** (Ubuntu 24.04 LTS, 1 vCPU / 1GB is fine to start). Point your domain A-record to the droplet IP.

SSH in as root and run:

```bash
bash -lc "sudo apt-get update && sudo apt-get install -y git"
git clone <YOUR_REPO_URL> /opt/kettlerunner && cd /opt/kettlerunner
sudo bash deploy/bootstrap.sh
```

This will:
- Create a non-root user `kettlerunner` with sudo.
- Install Python, Nginx, Certbot, Fail2ban, UFW.
- Set up a Python venv at `/opt/kettlerunner/venv` and install deps.
- Configure systemd services.
- (You still need to run certbot once to issue TLS)

### TLS with Let's Encrypt
After DNS propagates (`kettlerunner.com` → droplet IP):

```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d kettlerunner.com -d www.kettlerunner.com
sudo systemctl reload nginx
```

### Basic Auth for /admin + /jobs
Set admin password (single user) using `htpasswd`:

```bash
sudo apt-get install -y apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd daniel
# enter password
sudo nginx -t && sudo systemctl reload nginx
```

---

## 3) Environment variables

Copy `.env.example` to `.env` and fill values, then move to `/opt/kettlerunner/.env` on the server (or use `systemd` EnvironmentFile).

- `OPENAI_API_KEY`
- `SCHWAB_CLIENT_ID`, `SCHWAB_CLIENT_SECRET` (if used), `SCHWAB_REDIRECT_URI`
- `SCHWAB_ACCOUNT_ID`
- `DRY_RUN=true` (keep true until you are 100% ready)
- `ADMIN_EMAIL`

---

## 4) GitHub Actions (CI/CD)

Set repository secrets:
- `SSH_HOST` (your droplet IP or domain)
- `SSH_USER` = `kettlerunner`
- `SSH_KEY` = private key content (PEM) with access to `/home/kettlerunner/.ssh/authorized_keys`
- `SERVICE_NAME` = `trader`
- `APP_DIR` = `/opt/kettlerunner`

Push to `main`: the workflow rsyncs files, installs deps, runs DB migrate, restarts service.

---

## 5) Running + Scheduling

App service (Gunicorn): 
```bash
sudo systemctl enable --now trader.service
sudo systemctl status trader.service
```

Strategy job (systemd timer):
```bash
sudo systemctl enable --now trader.timer
sudo systemctl list-timers | grep trader
```

By default the timer runs at 09:30, 11:30, 13:30, 14:30, 15:30 America/Chicago on weekdays. Adjust in `deploy/trader.timer`.

---

## 6) Local dev

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://localhost:8000

---

## 7) Notes on Schwab API

- You must register an app, configure OAuth2/PKCE redirect, and store refresh tokens securely.
- The included `services/schwab_client.py` is a **stub**. Complete it with the official docs.
- Start with `DRY_RUN=true` — orders are logged but not sent.

---

## 8) Swap SQLite → Postgres (optional)

Install Postgres and set `DATABASE_URL=postgresql+psycopg://user:pass@localhost/dbname`. Update `db.py` and install `psycopg[binary]`.
