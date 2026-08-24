#!/usr/bin/env bash
set -euo pipefail

repo_root=${VNAA_REPO_ROOT:-/workspace/Algoverse-VNAA}
python_bin=${VNAA_PYTHON:-/root/vnaa-venv/bin/python}

cd "$repo_root"
export PYTHONPATH=.
export HF_HOME=${VNAA_HF_CACHE:-/workspace/hf-cache}
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

audit_dir=results/steering_priority8
direction=results/steer_Llama-3.1-8B-Instruct_direction.npz
controls="$audit_dir/steer_Llama-3.1-8B-Instruct_orthogonal_controls.npz"
logs="$audit_dir/logs"

test -f "$direction"
test -f "$controls"
mkdir -p "$logs"

date --iso-8601=seconds
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$python_bin" --version
"$python_bin" -c 'import torch, transformers; print(f"torch={torch.__version__} cuda={torch.version.cuda} transformers={transformers.__version__}")'

# Each command is resumable at the prompt level. The frozen five controls use
# alpha {-2,+2}; the learned-direction alpha=0 output is reused during analysis.
"$python_bin" -m vnaa.run_steering_priority8 generate \
  --set self_report --dtype bfloat16 --device-map auto \
  2>&1 | tee "$logs/self_report.log"

"$python_bin" -m vnaa.run_steering_priority8 generate \
  --set safety --dtype bfloat16 --device-map auto \
  2>&1 | tee "$logs/safety.log"

find "$audit_dir" -maxdepth 1 -type f ! -name SHA256SUMS.txt -print0 \
  | sort -z \
  | xargs -0 sha256sum > "$audit_dir/SHA256SUMS.txt"

touch "$audit_dir/BEHAVIOR_GENERATION_COMPLETE"
sync
echo "ALL_COMPLETE $(date --iso-8601=seconds)"
