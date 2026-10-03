#!/usr/bin/env bash
# Start the tile explorer API and the Vite dev server.
# An explorer already listening on these ports is stopped first, so a second
# ./run.sh replaces the one left running in another terminal.
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

# Catalog, thumbnails, and grid. Each step is skipped when its output is already there.
# Thumbnails resume: a tile with a thumb, a mask, and a meta row is left alone.
PY=".venv/bin/python"
if [[ ! -f cache/tiles.parquet ]]; then
  echo "Building catalog (cache/tiles.parquet)..."
  "$PY" build_catalog.py
fi
"$PY" build_thumbnails.py
if [[ ! -f cache/grid.parquet || cache/tiles.parquet -nt cache/grid.parquet ]]; then
  echo "Building grid (cache/grid.parquet)..."
  "$PY" build_grid.py
fi

API_PORT=8000
WEB_PORT=5173
API_PID=""
WEB_PID=""
WATCH_PID=""

proc_cmdline() {
  local pid=$1
  [[ -r /proc/$pid/cmdline ]] || return 0
  tr '\0' ' ' < "/proc/$pid/cmdline"
}

proc_ppid() {
  local pid=$1
  [[ -r /proc/$pid/status ]] || return 0
  awk '/^PPid:/ { print $2 }' "/proc/$pid/status"
}

proc_cwd() {
  readlink -f "/proc/$1/cwd" 2>/dev/null || true
}

listening_pids() {
  local port=$1 raw
  raw="$(ss -H -lptn "sport = :${port}" 2>/dev/null || true)"
  [[ -n $raw ]] || return 0
  grep -oE 'pid=[0-9]+' <<< "$raw" | cut -d= -f2 | sort -u || true
}

port_in_use() {
  local raw
  raw="$(ss -H -lnt "sport = :${1}" 2>/dev/null || true)"
  [[ -n ${raw//[[:space:]]/} ]]
}

is_explorer_pid() {
  local pid=$1 cmd cwd
  cmd="$(proc_cmdline "$pid")"
  cwd="$(proc_cwd "$pid")"
  if [[ $cmd == *uvicorn* && $cmd == *api:app* && ( $cwd == "$ROOT" || $cmd == *"$ROOT"* ) ]]; then
    return 0
  fi
  if [[ $cwd == "$ROOT/web" || $cmd == *"$ROOT/web"* ]]; then
    if [[ $cmd == *vite* || $cmd == *npm* || $cmd == *node* ]]; then
      return 0
    fi
  fi
  return 1
}

# The run.sh that started this pid, when it is this explorer and not us.
explorer_supervisor() {
  local pid=$1 cmd cwd
  local i=0
  while [[ -n $pid && $pid != 1 && $i -lt 20 ]]; do
    if [[ $pid -eq $$ || $pid -eq ${PPID:-0} ]]; then
      return 0
    fi
    cmd="$(proc_cmdline "$pid")"
    cwd="$(proc_cwd "$pid")"
    if [[ $cmd == *run.sh* && ( $cwd == "$ROOT" || $cmd == *"$ROOT/run.sh"* ) ]]; then
      echo "$pid"
      return 0
    fi
    pid="$(proc_ppid "$pid")"
    i=$((i + 1))
  done
}

proc_tree() {
  local pid=$1 child
  echo "$pid"
  for child in $(ps -o pid= --ppid "$pid" 2>/dev/null || true); do
    proc_tree "$child"
  done
}

# True when pid is this script or something that started it.
is_our_ancestor() {
  local pid=$1 walk=${2:-$$}
  local i=0
  while [[ -n $walk && $walk -ne 1 && $i -lt 30 ]]; do
    [[ $walk -eq $pid ]] && return 0
    walk="$(proc_ppid "$walk")"
    i=$((i + 1))
  done
  return 1
}

stop_existing_explorer() {
  local port pid sup found tree_pid
  local -a listeners=() supervisors=() targets=()
  local -A seen_pid=() seen_sup=() seen_target=()

  for port in "$API_PORT" "$WEB_PORT"; do
    found=0
    while IFS= read -r pid; do
      [[ -n $pid ]] || continue
      found=1
      if ! is_explorer_pid "$pid"; then
        echo "Port ${port} is already used by pid ${pid} ($(proc_cmdline "$pid")), which is not this explorer." >&2
        exit 1
      fi
      if [[ -z ${seen_pid[$pid]:-} ]]; then
        listeners+=("$pid")
        seen_pid[$pid]=1
      fi
      sup="$(explorer_supervisor "$pid" || true)"
      if [[ -n $sup && -z ${seen_sup[$sup]:-} ]]; then
        supervisors+=("$sup")
        seen_sup[$sup]=1
      fi
    done < <(listening_pids "$port")

    if [[ $found -eq 0 ]] && port_in_use "$port"; then
      echo "Port ${port} is in use, but the owning process is not visible. Stop it, then rerun." >&2
      exit 1
    fi
  done

  if ((${#listeners[@]} == 0)); then
    return 0
  fi

  for pid in "${supervisors[@]+"${supervisors[@]}"}" "${listeners[@]}"; do
    while IFS= read -r tree_pid; do
      [[ -n $tree_pid && -z ${seen_target[$tree_pid]:-} ]] || continue
      if is_our_ancestor "$tree_pid"; then
        continue
      fi
      seen_target[$tree_pid]=1
      targets+=("$tree_pid")
    done < <(proc_tree "$pid")
  done

  if ((${#targets[@]} == 0)); then
    echo "The explorer on ports ${API_PORT} and ${WEB_PORT} is this process. Stop it, then rerun." >&2
    exit 1
  fi

  echo "Stopping the explorer already running (pid ${listeners[*]})."
  kill -TERM "${targets[@]}" 2>/dev/null || true

  local i
  for i in $(seq 1 25); do
    if ! port_in_use "$API_PORT" && ! port_in_use "$WEB_PORT"; then
      return 0
    fi
    sleep 0.2
  done

  echo "Existing explorer did not exit; forcing it to stop." >&2
  kill -KILL "${targets[@]}" 2>/dev/null || true

  for i in $(seq 1 25); do
    if ! port_in_use "$API_PORT" && ! port_in_use "$WEB_PORT"; then
      return 0
    fi
    sleep 0.2
  done

  echo "Could not free ports ${API_PORT} and ${WEB_PORT}." >&2
  exit 1
}

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

stop_existing_explorer

.venv/bin/uvicorn api:app --host 127.0.0.1 --port "$API_PORT" &
API_PID=$!

(
  cd web
  npm run dev -- --host 127.0.0.1 --port "$WEB_PORT"
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

wait_http "http://127.0.0.1:${API_PORT}/api/facets" "API" "$API_PID"
wait_http "http://127.0.0.1:${WEB_PORT}/" "Dev server" "$WEB_PID"

echo "Explorer is at http://127.0.0.1:${WEB_PORT}"
wait -n "$API_PID" "$WEB_PID"
