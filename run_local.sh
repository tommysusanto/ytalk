#!/usr/bin/env bash
# Run yTalk from local source against the Homebrew-installed Python environment.
# Edit src/ytalk/app.py and re-run this script — no reinstall needed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BREW_PY="$(ls -d /usr/local/Cellar/ytalk/*/libexec/bin/python3.12 2>/dev/null | sort -V | tail -1)"

if [[ ! -x "$BREW_PY" ]]; then
  echo "Could not find brew-installed Python at: $BREW_PY" >&2
  echo "Run 'brew install tommysusanto/tap/ytalk' first, or edit BREW_PY in this script." >&2
  exit 1
fi

PYTHONPATH="$SCRIPT_DIR/src" exec "$BREW_PY" -m ytalk.app "$@"
