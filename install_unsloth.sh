#!/usr/bin/env bash
set -euxo pipefail
pip install --quiet uv
uv pip install --quiet "torch==2.8.0" "triton>=3.3.0" torchvision "bitsandbytes==0.50.2" "xformers==0.0.32.post2" \
  "unsloth==2026.9.14" "unsloth_zoo==2026.9.9" "transformers==5.5.0" "trl==0.24.0" "peft==0.21.0" \
  "accelerate==1.15.0" "datasets==4.3.0"
uv pip uninstall --quiet flash-linear-attention fla-core torchao  # torchao needs a newer torch
uv pip install --no-build-isolation "causal_conv1d==1.6.0"
