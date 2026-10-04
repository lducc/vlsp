#!/usr/bin/env bash
# separate venv: vLLM needs a newer torch than the training stack
set -euxo pipefail
pip install --quiet uv ninja
uv venv .venv-vllm --python 3.12
uv pip install --python .venv-vllm/bin/python "vllm==0.30.0" --torch-backend=auto
