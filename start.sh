#!/usr/bin/env bash
# Claude Code API Gateway startup script for NixOS

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Find libstdc++ for NixOS
LIBSTDCPP=$(find /nix/store -name "libstdc++.so.6" -path "*gcc*" 2>/dev/null | head -1)
if [ -n "$LIBSTDCPP" ]; then
    export LD_LIBRARY_PATH="$(dirname $LIBSTDCPP):$LD_LIBRARY_PATH"
fi

# Activate virtual environment and start server
source .venv/bin/activate
exec python -m uvicorn claude_code_api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
