#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/opt/kettlerunner"
APP_USER="kettlerunner"

# Create user
if ! id -u "$APP_USER" >/dev/null 2>&1; then
  adduser --disabled-password --gecos "" "$APP_USER"
  usermod -aG sudo "$APP_USER"
fi

# Base packages
apt-get update
apt-get install -y python3 python3-venv python3-pip nginx git ufw fail2ban

# Firewall
ufw allow OpenSSH || true
ufw allow 'Nginx Full' || true
ufw --force enable || true

# Python venv
python3 -m venv "$APP_DIR/venv"
source "$APP_DIR/venv/bin/activate"
pip install --upgrade pip
pip install -r "$APP_DIR/requirements.txt"

# Nginx
cp "$APP_DIR/deploy/nginx_trader.conf" /etc/nginx/sites-available/trader.conf
ln -sf /etc/nginx/sites-available/trader.conf /etc/nginx/sites-enabled/trader.conf
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx

# Systemd services
cp "$APP_DIR/deploy/trader.service" /etc/systemd/system/trader.service
cp "$APP_DIR/deploy/trader.timer" /etc/systemd/system/trader.timer
systemctl daemon-reload
systemctl enable --now trader.service
systemctl enable --now trader.timer

chown -R "$APP_USER":"$APP_USER" "$APP_DIR"
echo "Bootstrap complete."

# Job unit
cp "/opt/kettlerunner/deploy/trader-job.service" /etc/systemd/system/trader-job.service
systemctl daemon-reload
