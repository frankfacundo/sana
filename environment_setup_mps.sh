#!/usr/bin/env bash
set -euo pipefail

# Check if we should skip environment setup entirely.
if [ "${SKIP_ENV_SETUP:-}" = "true" ]; then
    echo "SKIP_ENV_SETUP is set to true. Skipping all environment setup steps."
    echo "Using the active Python environment. Make sure it has all required packages installed."
    exit 0
fi

if [ "$(uname -s)" != "Darwin" ]; then
    echo "Warning: environment_setup_mps.sh is intended for macOS with Apple Silicon MPS."
fi

CONDA_ENV=${1:-""}
if [ -n "$CONDA_ENV" ]; then
    # This is required to activate conda environment.
    eval "$(conda shell.bash hook)"

    if conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV"; then
        echo "Conda environment '$CONDA_ENV' already exists. Reusing it."
    else
        conda create -n "$CONDA_ENV" python=3.10.0 -y
    fi
    conda activate "$CONDA_ENV"
else
    echo "Skipping conda environment creation. Make sure you have the correct environment activated."
fi

# Keep unsupported MPS ops from hard failing when PyTorch can safely fall back to CPU.
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export DISABLE_XFORMERS="${DISABLE_XFORMERS:-1}"

if [ -n "$CONDA_ENV" ]; then
    conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK="$PYTORCH_ENABLE_MPS_FALLBACK"
    conda env config vars set DISABLE_XFORMERS="$DISABLE_XFORMERS"
fi

# update packaging tools. Keep setuptools below the version range where legacy
# packages such as mmcv 1.x can fail to import pkg_resources while building.
pip install -U pip wheel "setuptools<81"

# Install the matching macOS PyTorch wheels. Override these env vars if you need
# to test a different PyTorch release.
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.23.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.8.0}"
pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}"

# Install Sana's dependencies except CUDA-only packages. The project itself is
# then installed with --no-deps so those skipped packages are not reintroduced.
REQ_FILE="$(mktemp "${TMPDIR:-/tmp}/sana-mps-requirements.XXXXXX")"
trap 'rm -f "$REQ_FILE"' EXIT

awk '
    /^dependencies = \[/ { in_deps = 1; next }
    in_deps && /^\]/ { exit }
    in_deps {
        line = $0
        sub(/^[[:space:]]*"/, "", line)
        sub(/",[[:space:]]*$/, "", line)
        if (line != "" && line !~ /^#/) print line
    }
' pyproject.toml \
    | grep -Ev '^(torchvision|torchaudio|mmcv|xformers|triton|bitsandbytes)([<>=!~ ;]|$)' \
    > "$REQ_FILE"

# mmcv 1.x is used for Registry/Config helpers in this repo. On current pip it
# needs the active environment's setuptools/pkg_resources during build.
pip install --no-build-isolation "mmcv==1.7.2"
pip install -r "$REQ_FILE"
pip install -e . --no-deps

python - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}")
if torch.backends.mps.is_available():
    print("MPS backend is available.")
else:
    print("Warning: MPS backend is not available in this environment.")
PY
