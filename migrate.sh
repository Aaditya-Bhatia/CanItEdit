#!/usr/bin/env bash
# Sets up the CanItEdit benchmark repo on a new server.
# Run as: bash migrate.sh
#
# Prerequisites:
#   - conda available (or will install miniconda locally)
#   - Docker installed (for eval step)
#   - Internet access

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== CanItEdit Repo Setup ==="
echo "Repo: $REPO_DIR"
echo ""

# ---- Conda env ----
if command -v conda &>/dev/null; then
    echo ">>> conda found."
elif [ -d "/shared_workspace_mfs/aadi/miniconda3" ]; then
    eval "$(/shared_workspace_mfs/aadi/miniconda3/bin/conda shell.bash hook)"
else
    echo ">>> Installing Miniconda locally..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$REPO_DIR/.miniconda3"
    rm /tmp/miniconda.sh
    eval "$($REPO_DIR/.miniconda3/bin/conda shell.bash hook)"
fi

if ! conda env list | grep -q "canitedit"; then
    echo ">>> Creating canitedit conda env (Python 3.12)..."
    conda create -n canitedit python=3.12 -y
else
    echo ">>> canitedit env already exists."
fi
conda activate canitedit

# ---- Python deps ----
echo ">>> Installing Python dependencies..."
pip install --quiet -e "$REPO_DIR"

# Also install pyyaml (needed by run_from_config.sh's inline python)
pip install --quiet pyyaml

# ---- Docker eval image ----
if command -v docker &>/dev/null; then
    echo ">>> Pulling CanItEdit Docker eval image..."
    docker pull ghcr.io/nuprl/canitedit || echo "WARNING: Docker pull failed. Eval step needs this image."
else
    echo "WARNING: Docker not found. Install Docker for the evaluation step."
    echo "         Generation will still work without it (use --generate-only)."
fi

# ---- Create logs dir ----
mkdir -p "$REPO_DIR/logs"
mkdir -p "$REPO_DIR/runs"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Usage:"
echo "  conda activate canitedit"
echo "  ./run_from_config.sh <config.yaml> --batch-size 300"
echo ""
echo "Notes:"
echo "  - Config YAMLs go wherever you keep them (pass full path)"
echo "  - vLLM server must be running before launching benchmarks"
echo "  - Results land in ./runs/<model>-canitedit-<date>/"
echo "  - detached_eval_worker.py imports from Master_VLLM/benchmark_results.py"
echo "    If you use Docker eval, make sure that project is at the expected path"
echo "    or adjust the import in scripts/detached_eval_worker.py"
echo ""
