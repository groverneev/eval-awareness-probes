#!/usr/bin/env bash
set -euo pipefail

cd /workspace/Algoverse-VNAA

export PYTHONPATH=.
export HF_HOME=/workspace/hf-cache
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

PYTHON=/root/vnaa-venv/bin/python
MANIFEST=datasets/qwen_scale/scale_ladder_v1.json
RESULTS=results/qwen_scale/v1
LOGS="$RESULTS/logs"

mkdir -p "$LOGS"

date --iso-8601=seconds
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$PYTHON" --version
"$PYTHON" -c 'import torch, transformers; print(f"torch={torch.__version__} cuda={torch.version.cuda} transformers={transformers.__version__}")'
"$PYTHON" -m vnaa.cross_model_suite \
  --manifest "$MANIFEST" --results "$RESULTS" validate

if (( $# > 0 )); then
  # A failed network or storage operation can be resumed from the first
  # incomplete checkpoint. Existing activation files are still provenance-
  # checked by the extractor before they are reused.
  MODELS=("$@")
else
  MODELS=(
    Qwen/Qwen3-0.6B
    Qwen/Qwen3-1.7B
    Qwen/Qwen3-4B
    Qwen/Qwen3-8B
    Qwen/Qwen3-14B
    Qwen/Qwen3-32B
  )
fi

for model in "${MODELS[@]}"; do
  tag=${model//\//_}
  echo "BEGIN $model $(date --iso-8601=seconds)"
  "$PYTHON" -m vnaa.cross_model_suite \
    --manifest "$MANIFEST" --results "$RESULTS" extract \
    --model "$model" --device-map auto --dtype bfloat16 \
    2>&1 | tee "$LOGS/${tag}_extract.log"
  "$PYTHON" -m vnaa.cross_model_suite \
    --manifest "$MANIFEST" --results "$RESULTS" analyze \
    --model "$model" \
    2>&1 | tee "$LOGS/${tag}_analyze.log"
  echo "END $model $(date --iso-8601=seconds)"
done

find "$RESULTS" -type f ! -name SHA256SUMS.txt -print0 \
  | sort -z \
  | xargs -0 sha256sum > "$RESULTS/SHA256SUMS.txt"

touch "$RESULTS/RUN_COMPLETE"
sync
echo "ALL_COMPLETE $(date --iso-8601=seconds)"
