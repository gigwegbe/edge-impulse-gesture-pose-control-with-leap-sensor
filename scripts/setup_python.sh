#!/usr/bin/env bash
# Set up .venv with the LeapC Python bindings.
#
# macOS: recent Gemini ships a compiled module in
#   /Applications/Ultraleap Hand Tracking.app/Contents/LeapSDK/leapc_cffi
# (e.g. _leapc_cffi.cpython-312-darwin.so). The venv must use that exact
# Python version; no build needed.
#
# Linux: builds leapc_cffi from source against the installed LeapC, so any
# Python 3 works. Non-default SDK location? export LEAPSDK_INSTALL_LOCATION.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

if [ ! -d third_party/leapc-python-bindings ]; then
  mkdir -p third_party
  git clone --depth 1 https://github.com/ultraleap/leapc-python-bindings.git third_party/leapc-python-bindings
fi
BINDINGS=third_party/leapc-python-bindings

if [ "$(uname -s)" = "Darwin" ]; then
  SDK="${LEAPSDK_INSTALL_LOCATION:-/Applications/Ultraleap Hand Tracking.app/Contents/LeapSDK}"
  SO=$(ls "$SDK"/leapc_cffi/_leapc_cffi.cpython-3*-darwin.so 2>/dev/null | head -1 || true)
  if [ -z "$SO" ]; then
    echo "No compiled leapc_cffi found in $SDK/leapc_cffi. Update Ultraleap Hand Tracking." >&2
    exit 1
  fi
  TAG=$(basename "$SO" | sed -E 's/.*cpython-3([0-9]+)-darwin.so/\1/')
  PY="python3.$TAG"
  command -v "$PY" > /dev/null || { echo "Gemini's module needs $PY: brew install python@3.$TAG" >&2; exit 1; }
  echo "Using Gemini's bundled leapc_cffi with $PY"
  rm -rf .venv
  "$PY" -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -e "$BINDINGS/leapc-python-api"
else
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r "$BINDINGS/requirements.txt"
  python -m build "$BINDINGS/leapc-cffi"
  pip install "$BINDINGS"/leapc-cffi/dist/leapc_cffi-0.0.1.tar.gz
  pip install -e "$BINDINGS/leapc-python-api"
fi

pip install -r requirements.txt

echo
echo "Bindings installed. Test with:"
echo "  source .venv/bin/activate && python src/check_leap.py"
