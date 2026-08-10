#!/usr/bin/env bash
# One-shot setup for running slice 1 on a cloud GPU (e.g. Lightning AI L4).
#
# Run this from a Studio terminal AFTER cloning the repo:
#   git clone <repository-url>
#   cd Algoverse-VNAA
#   bash scripts/setup_gpu.sh
#
# It installs deps ON TOP OF the environment's existing (CUDA) PyTorch, logs in
# to Hugging Face, and downloads the SAD oversight data. It does NOT reinstall
# torch: the Studio ships a CUDA build, and letting pip replace it can pull a
# CPU-only wheel that silently runs everything on CPU.
set -euo pipefail

echo "== 1/4 checking torch (CUDA not required for setup; needed only for the run) =="
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
else:
    print("no CUDA yet — fine for setup; switch the Studio to the L4 GPU before the run")
PY

echo "== 2/4 installing project deps (NOT torch) =="
# Runtime deps minus torch, then the package itself without pulling deps.
pip install -q "transformers>=4.44" "datasets>=2.20" "scikit-learn>=1.5" \
    "numpy>=1.26" "tqdm>=4.66" "huggingface_hub>=0.24" "PyYAML>=6.0"
pip install -q -e . --no-deps

echo "== 3/4 Hugging Face login (needs a token with gated-repo read access) =="
if python -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
    echo "already logged in to Hugging Face"
else
    hf auth login
fi

echo "== 4/4 downloading SAD oversight data (encrypted, ~1.3 MB) =="
mkdir -p data/sad
if [ ! -f data/sad/oversight_structs.zip ]; then
    curl -sL -o data/sad/oversight_structs.zip \
        https://github.com/LRudL/sad/raw/main/sad/stages/oversight/structs.zip
fi
python -c "import zipfile; z=zipfile.ZipFile('data/sad/oversight_structs.zip'); \
z.read('batch/test_oversight.yaml', pwd=b'sadtimesforthesetimes'); print('SAD data OK')"

echo
echo "Setup done. To run slice 1 on the 8B (switch the Studio to the L4 GPU first):"
echo
echo "  python -m vnaa.run_slice1 \\"
echo "    --sad-repo data/sad/oversight_structs.zip \\"
echo "    --model meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16"
echo
