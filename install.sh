#!/usr/bin/env bash

set -e

# ============================================================
# MambaX environment installation
# ============================================================

ENV_NAME="mambax"
PYTHON_VERSION="3.9.18"

# Move to repository root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "========================================"
echo " MambaX Environment Installation"
echo "========================================"

# ------------------------------------------------------------
# 1. Check Conda
# ------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] Conda is not installed or not available in PATH."
    echo "Please install Miniconda or Anaconda first."
    exit 1
fi

# Enable conda activate inside shell script
source "$(conda info --base)/etc/profile.d/conda.sh"

# ------------------------------------------------------------
# 2. Create Conda environment
# ------------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[INFO] Conda environment '$ENV_NAME' already exists."
else
    echo "[INFO] Creating Conda environment '$ENV_NAME'..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"

echo "[INFO] Using Python:"
python --version

# ------------------------------------------------------------
# 3. Install PyTorch (CUDA 11.7)
# ------------------------------------------------------------
echo "[INFO] Installing PyTorch 1.13.1 + CUDA 11.7..."

python -m pip install \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# ------------------------------------------------------------
# 4. Install Python dependencies
# ------------------------------------------------------------
echo "[INFO] Installing Python dependencies..."

python -m pip install -r requirements.txt

# ------------------------------------------------------------
# 5. Build causal-conv1d
# ------------------------------------------------------------
echo "[INFO] Building causal-conv1d..."

cd "$ROOT_DIR/multimodal/causal-conv1d"

rm -rf build *.egg-info

CAUSAL_CONV1D_FORCE_BUILD=TRUE \
python -m pip install .

# ------------------------------------------------------------
# 6. Build mamba-ssm
# ------------------------------------------------------------
echo "[INFO] Building mamba-ssm..."

cd "$ROOT_DIR/multimodal/mamba"

rm -rf build *.egg-info

MAMBA_FORCE_BUILD=TRUE \
python -m pip install .

# Return to repository root
cd "$ROOT_DIR"

# ------------------------------------------------------------
# 7. Verify installation
# ------------------------------------------------------------
echo "[INFO] Verifying selective_scan..."

python -c "
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
print('selective_scan_fn: OK')
"

echo
echo "========================================"
echo " MambaX installation completed."
echo "========================================"
echo
echo "Activate the environment with:"
echo
echo "    conda activate $ENV_NAME"
echo
