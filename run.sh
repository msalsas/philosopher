#!/usr/bin/env bash
# Start/stop Philosopher: the SERVER (on this machine) + the TOY (on the Pi via SSH).
#
# Usage:
#   ./run.sh start     # start server and toy (default)
#   ./run.sh stop      # stop both
#   ./run.sh status    # status of both
#   ./run.sh logs      # tail -f the server log (Ctrl-C to quit)
#   ./run.sh toylog    # last lines of the toy log (on the Pi)
#   ./run.sh install   # autostart the TOY on the Pi (systemd service). The
#                      #   server is NOT a service: start it with `start`.
#                      #   uninstall reverts it.
#
# Config (none of this is committed to git):
#   Create a `run.local.env` next to this script (see run.local.env.example)
#   or export environment variables:
#     PI_HOST        IP/host of the toy's Pi          (REQUIRED)
#     PI_USER        SSH user on the Pi               (default: $USER)
#     REMOTE_TOY_DIR toy path on the Pi, relative to the user's home
#                    (default: philosopher/philosopher-toy)
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Local config (optional, git-ignored): your own host IP/user.
[ -f "$REPO/run.local.env" ] && . "$REPO/run.local.env"

PI_HOST="${PI_HOST:-}"
PI_USER="${PI_USER:-$USER}"
REMOTE_TOY_DIR="${REMOTE_TOY_DIR:-philosopher/philosopher-toy}"
SERVER_DIR="${SERVER_DIR:-$REPO/philosopher-server}"
SERVER_LOG="${SERVER_LOG:-/tmp/philosopher-server.log}"
TOY_LOG="/tmp/toy.log"
HEALTH="http://localhost:8080/health"
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=6"
# [t]oy_client... keeps pkill -f from matching its own command line.
TOYPAT='[t]oy_client.main'

if [ -z "$PI_HOST" ]; then
  echo "ERROR: PI_HOST not set. Create run.local.env (see run.local.env.example) or: PI_HOST=<ip> ./run.sh $*" >&2
  exit 2
fi

server_up() { curl -s --max-time 3 "$HEALTH" -o /dev/null 2>/dev/null; }

start_server() {
  if server_up; then echo "[server] already running"; return; fi
  # Ensure the ML models are present (idempotent: skips fast when already cached,
  # downloads on the first run). Kokoro in particular is NOT auto-fetched at
  # startup, so without this the toy would degrade to text-only on a fresh clone.
  echo "[server] checking models (downloads only on the first run)..."
  ( cd "$SERVER_DIR" && python scripts/download_models.py ) \
    || echo "[server] ⚠ model prefetch failed; starting anyway, missing pieces degrade (see /health)"
  echo "[server] starting ($SERVER_DIR)..."
  ( cd "$SERVER_DIR" && nohup python -m philosopher.main --server >"$SERVER_LOG" 2>&1 & disown )
  for _ in $(seq 1 25); do sleep 2; server_up && { echo "[server] ready  (log: $SERVER_LOG)"; return; }; done
  echo "[server] ✗ did not respond in time — check $SERVER_LOG"
}

toy_is_service() { $SSH "$PI_USER@$PI_HOST" "systemctl is-enabled philosopher-toy.service" >/dev/null 2>&1; }

start_toy() {
  if toy_is_service; then
    $SSH "$PI_USER@$PI_HOST" "sudo systemctl restart philosopher-toy.service" >/dev/null 2>&1 \
      && echo "[toy] ✓ service (re)started on $PI_HOST" \
      || echo "[toy] ✗ failed to (re)start the service"
    return
  fi
  echo "[toy] starting on $PI_USER@$PI_HOST ..."
  $SSH "$PI_USER@$PI_HOST" "pkill -9 -f '$TOYPAT' 2>/dev/null; sleep 1" >/dev/null 2>&1
  # timeout: the process survives via nohup; ssh closes even if the child lives on.
  timeout 8 $SSH "$PI_USER@$PI_HOST" \
    "cd '$REMOTE_TOY_DIR' && nohup python3 -u -m toy_client.main >'$TOY_LOG' 2>&1 </dev/null & disown" >/dev/null 2>&1
  sleep 5
  if $SSH "$PI_USER@$PI_HOST" "pgrep -f '$TOYPAT' >/dev/null" 2>/dev/null; then
    echo "[toy] ✓ running  (log on the Pi: $TOY_LOG)"
  else
    echo "[toy] ✗ did not start — check '$TOY_LOG' on the Pi (is it powered and on the network?)"
  fi
}

stop_server() {
  # pattern without a self-match so it doesn't kill this very script
  pkill -f 'philosopher[.]main --server' 2>/dev/null && echo "[server] stopped" || echo "[server] was not running"
}

stop_toy() {
  if toy_is_service; then
    $SSH "$PI_USER@$PI_HOST" "sudo systemctl stop philosopher-toy.service" >/dev/null 2>&1 \
      && echo "[toy] service stopped" || echo "[toy] (unreachable)"
    return
  fi
  $SSH "$PI_USER@$PI_HOST" "pkill -9 -f '$TOYPAT' 2>/dev/null" >/dev/null 2>&1 \
    && echo "[toy] stopped" || echo "[toy] (unreachable or not running)"
}

status() {
  if server_up; then echo "[server] ✓ healthy ($HEALTH)"; else echo "[server] ✗ down"; fi
  if $SSH "$PI_USER@$PI_HOST" "pgrep -f '$TOYPAT' >/dev/null" 2>/dev/null; then
    echo "[toy]    ✓ running on $PI_HOST  ($($SSH "$PI_USER@$PI_HOST" 'uptime | grep -oE "up [^,]+"' 2>/dev/null))"
  else
    echo "[toy]    ✗ not running (or Pi unreachable)"
  fi
}

# --- Toy autostart (systemd on the Pi) ------------------------------------
# The SERVER is not a service: start it by hand with `./run.sh start` when you
# use the toy (so the laptop isn't loading models all the time). The TOY is a
# service because it lives inside the plush and must come up on its own on power.
# Generated with the Pi's LOCAL paths/user/python: nothing hardcoded.

install_toy() {
  local toydir="$REMOTE_TOY_DIR"
  case "$toydir" in /*) : ;; *) toydir="/home/$PI_USER/$toydir" ;; esac
  $SSH "$PI_USER@$PI_HOST" "sudo tee /etc/systemd/system/philosopher-toy.service >/dev/null" <<EOF
[Unit]
Description=Philosopher toy (the body)
After=network-online.target sound.target
Wants=network-online.target

[Service]
Type=simple
User=$PI_USER
WorkingDirectory=$toydir
ExecStart=/usr/bin/python3 -u -m toy_client.main
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
  $SSH "$PI_USER@$PI_HOST" "sudo systemctl daemon-reload && sudo systemctl enable --now philosopher-toy.service" \
    && echo "[toy] service installed and started on $PI_HOST (journalctl -u philosopher-toy -f)" \
    || echo "[toy] ✗ install failed (is the Pi powered and on the network?)"
}

case "${1:-start}" in
  start)  start_server; start_toy; echo; status ;;
  stop)   stop_toy; stop_server ;;
  status) status ;;
  logs)   echo "== tail -f $SERVER_LOG (Ctrl-C to quit) =="; tail -f "$SERVER_LOG" ;;
  toylog) $SSH "$PI_USER@$PI_HOST" "tail -30 '$TOY_LOG'" 2>/dev/null || echo "(Pi unreachable)" ;;
  install|install-toy) install_toy ;;   # autostart the toy on the Pi
  uninstall)
    $SSH "$PI_USER@$PI_HOST" "sudo systemctl disable --now philosopher-toy.service 2>/dev/null; sudo rm -f /etc/systemd/system/philosopher-toy.service; sudo systemctl daemon-reload" \
      && echo "[toy] service uninstalled" || echo "[toy] (unreachable)" ;;
  *) echo "usage: $0 [start|stop|status|logs|toylog|install|uninstall]"; exit 1 ;;
esac
