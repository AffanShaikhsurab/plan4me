#!/usr/bin/env bash
# Generate .cursor/mcp.json with absolute paths for this clone.
# Usage (from repo root): ./scripts/setup_mcp.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="${ROOT}/.venv/bin/python"
OUT="${ROOT}/.cursor/mcp.json"

if [[ ! -x "${VENV_PY}" ]]; then
  echo "error: ${VENV_PY} not found. Create a venv first:"
  echo "  python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

mkdir -p "${ROOT}/.cursor"

# Escape backslashes for JSON (unlikely on Unix, but keep paths literal)
json_escape() {
  printf '%s' "$1" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read())[1:-1])'
}

CMD="$(json_escape "${VENV_PY}")"
CWD="$(json_escape "${ROOT}")"

cat > "${OUT}" <<EOF
{
  "mcpServers": {
    "plan4me": {
      "command": "${CMD}",
      "args": ["-m", "mcp_server"],
      "cwd": "${CWD}",
      "env": {
        "PYTHONPATH": "${CWD}"
      }
    }
  }
}
EOF

echo "Wrote ${OUT}"
echo "Next: reload MCP in Cursor (Settings → MCP → restart plan4me), then call health."
