#!/usr/bin/env bash
# Start the tile explorer API and the Vite dev server.
# Ctrl+C, a terminal hangup, or the launching shell exiting stops both.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/uvicorn ]]; then
  echo "Missing explorer/.venv. From explorer/: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

if [[ ! -d web/node_modules ]]; then
  echo "Missing explorer/web/node_modules. From explorer/web: npm install" >&2
  exit 1
fi

node_works() {
  [[ -n ${1:-} && -x $1 ]] && "$1" --version >/dev/null 2>&1
}

NODE=""
if command -v node >/dev/null 2>&1 && node_works "$(command -v node)"; then
  NODE="$(command -v node)"
else
  for candidate in /tmp/node-v22.20.0-linux-x64/bin/node /usr/local/bin/node; do
    if node_works "$candidate"; then
      NODE="$candidate"
      break
    fi
  done
fi

if [[ -z $NODE ]]; then
  echo "No working node binary found. Install Node and make sure node is on PATH." >&2
  exit 1
fi
export PATH="$(dirname "$NODE"):$PATH"

API_PID=""
WEB_PID=""
WATCH_PID=""

cleanup() {
  local pid
  for pid in "$WEB_PID" "$API_PID" "$WATCH_PID"; do
    if [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup HUP INT TERM EXIT

# A killed IDE terminal can exit without SIGHUP and leave this process
# reparented to init. Stop both servers when the launching shell is gone.
if [[ $PPID -ne 1 ]]; then
  parent=$PPID
  (
    while kill -0 "$parent" 2>/dev/null; do
      sleep 1
    done
    kill -TERM $$ 2>/dev/null || true
  ) &
  WATCH_PID=$!
fi

.venv/bin/uvicorn api:app --host 127.0.0.1 --port 8000 &
API_PID=$!

(
  cd web
  npm run dev -- --host 127.0.0.1 --port 5173
) &
WEB_PID=$!

wait_http() {
  local url=$1
  local name=$2
  local pid=$3
  local i
  for i in $(seq 1 50); do
    if curl -s -o /dev/null --max-time 1 "$url"; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$name exited before it was ready." >&2
      return 1
    fi
    sleep 0.2
  done
  echo "$name did not respond at $url" >&2
  return 1
}

wait_http "http://127.0.0.1:8000/api/facets" "API" "$API_PID"
wait_http "http://127.0.0.1:5173/" "Dev server" "$WEB_PID"

echo "Explorer is at http://127.0.0.1:5173"
wait -n "$API_PID" "$WEB_PID"
