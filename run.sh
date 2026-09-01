#!/usr/bin/env bash
# Levanta/para Philosopher: el SERVIDOR (en esta máquina) + el TOY (en la Pi por SSH).
#
# Uso:
#   ./run.sh start     # arranca servidor y toy (por defecto)
#   ./run.sh stop      # para ambos
#   ./run.sh status    # estado de ambos
#   ./run.sh logs      # tail -f del log del servidor (Ctrl-C para salir)
#   ./run.sh toylog    # últimas líneas del log del toy (en la Pi)
#
# Configuración (nada de esto se sube a git):
#   Crea un fichero `run.local.env` junto a este script (ver run.local.env.example)
#   o exporta variables de entorno:
#     PI_HOST        IP/host de la Pi del toy         (OBLIGATORIO)
#     PI_USER        usuario SSH en la Pi             (por defecto: $USER)
#     REMOTE_TOY_DIR ruta del toy en la Pi, relativa al home del usuario
#                    (por defecto: philosopher/philosopher-toy)
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Config local (opcional, ignorada por git): IP/usuario propios de tu montaje.
[ -f "$REPO/run.local.env" ] && . "$REPO/run.local.env"

PI_HOST="${PI_HOST:-}"
PI_USER="${PI_USER:-$USER}"
REMOTE_TOY_DIR="${REMOTE_TOY_DIR:-philosopher/philosopher-toy}"
SERVER_DIR="${SERVER_DIR:-$REPO/philosopher-server}"
SERVER_LOG="${SERVER_LOG:-/tmp/philosopher-server.log}"
TOY_LOG="/tmp/toy.log"
HEALTH="http://localhost:8080/health"
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=6"
# [t]oy_client... evita que pkill -f se auto-elimine (su propia línea contiene el patrón).
TOYPAT='[t]oy_client.main'

if [ -z "$PI_HOST" ]; then
  echo "ERROR: falta PI_HOST. Crea run.local.env (ver run.local.env.example) o: PI_HOST=<ip> ./run.sh $*" >&2
  exit 2
fi

server_up() { curl -s --max-time 3 "$HEALTH" -o /dev/null 2>/dev/null; }

start_server() {
  if server_up; then echo "[server] ya está corriendo"; return; fi
  echo "[server] arrancando ($SERVER_DIR)..."
  ( cd "$SERVER_DIR" && nohup python -m philosopher.main --server >"$SERVER_LOG" 2>&1 & disown )
  for _ in $(seq 1 25); do sleep 2; server_up && { echo "[server] listo  (log: $SERVER_LOG)"; return; }; done
  echo "[server] ✗ no respondió a tiempo — revisa $SERVER_LOG"
}

start_toy() {
  echo "[toy] arrancando en $PI_USER@$PI_HOST ..."
  $SSH "$PI_USER@$PI_HOST" "pkill -9 -f '$TOYPAT' 2>/dev/null; sleep 1" >/dev/null 2>&1
  # timeout: el proceso queda con nohup; el ssh se cierra aunque el hijo siga vivo.
  timeout 8 $SSH "$PI_USER@$PI_HOST" \
    "cd '$REMOTE_TOY_DIR' && nohup python3 -u -m toy_client.main >'$TOY_LOG' 2>&1 </dev/null & disown" >/dev/null 2>&1
  sleep 5
  if $SSH "$PI_USER@$PI_HOST" "pgrep -f '$TOYPAT' >/dev/null" 2>/dev/null; then
    echo "[toy] ✓ corriendo  (log en la Pi: $TOY_LOG)"
  else
    echo "[toy] ✗ no arrancó — mira '$TOY_LOG' en la Pi (¿está encendida y en red?)"
  fi
}

stop_server() {
  # patrón sin auto-match para no matar este propio script
  pkill -f 'philosopher[.]main --server' 2>/dev/null && echo "[server] parado" || echo "[server] no estaba corriendo"
}

stop_toy() {
  $SSH "$PI_USER@$PI_HOST" "pkill -9 -f '$TOYPAT' 2>/dev/null" >/dev/null 2>&1 \
    && echo "[toy] parado" || echo "[toy] (no accesible o no corría)"
}

status() {
  if server_up; then echo "[server] ✓ sano ($HEALTH)"; else echo "[server] ✗ caído"; fi
  if $SSH "$PI_USER@$PI_HOST" "pgrep -f '$TOYPAT' >/dev/null" 2>/dev/null; then
    echo "[toy]    ✓ corriendo en $PI_HOST  ($($SSH "$PI_USER@$PI_HOST" 'uptime | grep -oE "up [^,]+"' 2>/dev/null))"
  else
    echo "[toy]    ✗ no corre (o Pi no accesible)"
  fi
}

case "${1:-start}" in
  start)  start_server; start_toy; echo; status ;;
  stop)   stop_toy; stop_server ;;
  status) status ;;
  logs)   echo "== tail -f $SERVER_LOG (Ctrl-C para salir) =="; tail -f "$SERVER_LOG" ;;
  toylog) $SSH "$PI_USER@$PI_HOST" "tail -30 '$TOY_LOG'" 2>/dev/null || echo "(Pi no accesible)" ;;
  *) echo "uso: $0 [start|stop|status|logs|toylog]"; exit 1 ;;
esac
