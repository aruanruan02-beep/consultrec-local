#!/usr/bin/env bash
set -euo pipefail

if [[ ! -x .venv/bin/uv ]]; then
  .venv/bin/python -m pip install uv
fi

.venv/bin/uv venv .pyannote-venv --python 3.10 --seed
.venv/bin/uv pip install --python .pyannote-venv/bin/python -r requirements-pyannote.txt
