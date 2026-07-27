#!/usr/bin/env bash
# Create the project venv and install dependencies in an order that keeps
# madmom's ancient build (PyPI release 0.16.1, 2018) working. See
# requirements.txt for why the versions are pinned the way they are.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON_BIN="${PYTHON_BIN:-python3.9}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="/usr/bin/python3"  # macOS ships 3.9 here
fi

echo "Using interpreter: $PYTHON_BIN ($("$PYTHON_BIN" --version))"

python3 -c "import sys; assert sys.version_info[:2] == (3, 9), 'need Python 3.9 for madmom'" 2>/dev/null || true

rm -rf .venv
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate

pip install --upgrade pip "setuptools==80.10.2" wheel

# numpy/scipy/cython first, and madmom built with --no-build-isolation so its
# legacy setup.py can see them (PEP 517 isolation would build against a numpy
# it then can't find at import time otherwise).
pip install "numpy==1.23.5" "scipy==1.10.1" "cython==3.2.9" "mido==1.3.3"
pip install --no-build-isolation --no-deps "madmom==0.16.1"

# Everything else.
pip install \
    "librosa==0.11.0" \
    "mir_eval==0.8.2" \
    "matplotlib==3.9.4" \
    "pandas==2.3.3" \
    "numba==0.60.0" \
    "soundfile==0.13.1" \
    "tqdm==4.69.1" \
    "jupyter==1.1.1" \
    "ipykernel==6.31.0" \
    "nbformat==5.8.0" \
    "fastjsonschema==2.19.1"

python -m ipykernel install --user --name beat-tracking --display-name "beat-tracking (.venv)"

python -c "
import numpy, scipy, librosa, mir_eval, madmom, matplotlib, pandas
print('numpy', numpy.__version__)
print('scipy', scipy.__version__)
print('librosa', librosa.__version__)
print('mir_eval', mir_eval.__version__)
print('madmom', madmom.__version__)
print('pandas', pandas.__version__)
print('Environment OK')
"

echo
echo "Done. Activate with: source .venv/bin/activate"
