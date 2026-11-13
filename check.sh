#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$PROJECT_DIR"
PYTHON="$PROJECT_DIR/.venv/bin/python"
"$PYTHON" -m ruff format --check gravity_studio tests
"$PYTHON" -m ruff check gravity_studio tests
"$PYTHON" -m mypy gravity_studio
QT_QPA_PLATFORM=offscreen "$PYTHON" -m pytest -q
