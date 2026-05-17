#!/usr/bin/env bash
# Bring up the local dev stack: pinchtab daemon + FastAPI backend.
# Both run in the foreground; press Ctrl-C to stop both.
#
# Prereqs:
#   - Pinchtab source cloned at ../pinchtab and built (`cd ../pinchtab && go build -o bin/pinchtab ./cmd/pinchtab`)
#   - `pip install -e ".[dev]"` done at least once
#   - .env exists (PINCHTAB_TOKEN auto-pulled from ~/.pinchtab/config.json on first run)
#
# After both are up, open http://127.0.0.1:8000/ in your browser.

set -euo pipefail

cd "$(dirname "$0")/.."

PINCHTAB_BIN="../pinchtab/bin/pinchtab"
PORT_BACKEND="${PORT_BACKEND:-8000}"

if [[ ! -x "$PINCHTAB_BIN" ]]; then
  echo "[start] pinchtab binary not found at $PINCHTAB_BIN" >&2
  echo "        cd ../pinchtab && go build -o bin/pinchtab ./cmd/pinchtab" >&2
  exit 1
fi

# Ensure .env has PINCHTAB_TOKEN. If not, pull from ~/.pinchtab/config.json
# (which pinchtab creates on first run; if it doesn't exist yet, we'll let
# pinchtab create it, then re-source).
if ! grep -q "^PINCHTAB_TOKEN=" .env 2>/dev/null; then
  if [[ -f "$HOME/.pinchtab/config.json" ]]; then
    TOK=$(python3 -c "import json; print(json.load(open('$HOME/.pinchtab/config.json'))['server']['token'])")
    echo "PINCHTAB_TOKEN=$TOK" >> .env
    echo "[start] PINCHTAB_TOKEN written to .env"
  else
    echo "[start] WARNING: no ~/.pinchtab/config.json yet — start pinchtab once first to generate it"
  fi
fi

pids=()
cleanup() {
  echo
  echo "[start] stopping…"
  for pid in "${pids[@]:-}"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo "[start] starting pinchtab daemon (guards down for dev: -y)…"
"$PINCHTAB_BIN" server -y &
pids+=($!)

# Give pinchtab a moment to bind 9867.
sleep 3

echo "[start] starting FastAPI backend on :$PORT_BACKEND…"
python -m uvicorn backend.main:app --host 127.0.0.1 --port "$PORT_BACKEND" &
pids+=($!)

# Give backend a moment.
sleep 2

echo
echo "════════════════════════════════════════════"
echo "  Dashboard:  http://127.0.0.1:$PORT_BACKEND/"
echo "  Backend:    http://127.0.0.1:$PORT_BACKEND/health"
echo "  Pinchtab:   http://127.0.0.1:9867/health"
echo "════════════════════════════════════════════"
echo
echo "Press Ctrl-C to stop everything."
wait
