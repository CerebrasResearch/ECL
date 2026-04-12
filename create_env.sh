#!/usr/bin/env bash
# create_env.sh — Create and configure the ECL development environment.
# Usage: bash create_env.sh [--name ENV_NAME]
set -euo pipefail

ENV_NAME="${1:-ecl}"

echo "=== Creating conda environment '${ENV_NAME}' ==="
conda create -n "${ENV_NAME}" python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "=== Installing PyTorch (CUDA 12.1) ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing ECL package (editable, all extras) ==="
pip install -e ".[all]"

echo "=== Installing pre-commit hooks ==="
pre-commit install

echo ""
echo "Done. Activate with:  conda activate ${ENV_NAME}"
