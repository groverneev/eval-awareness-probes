#!/usr/bin/env bash
set -euo pipefail

cd /workspace/vnaa
export HF_HOME=/workspace/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
python_bin=/opt/vnaa-venv/bin/python

for condition in \
  target_on_policy \
  same_family_off_policy \
  cross_family_off_policy
do
  "$python_bin" -m vnaa.run_authorship_generation \
    --condition "$condition" \
    --manifest datasets/authorship/pilot_manifest_v2.json \
    --output-dir results/authorship/generations_v2 \
    --max-new-tokens 1024
done

"$python_bin" -m vnaa.build_authorship_gate \
  --manifest datasets/authorship/pilot_manifest_v2.json \
  --generation-dir results/authorship/generations_v2 \
  --output-dir results/authorship/v2
touch /workspace/vnaa/results/authorship/v2/PILOT_GENERATION_COMPLETE
