# Running slice 1 on Kaggle (8B, 2x T4)

Kaggle gives two free T4 GPUs (16 GB each). The 8B in bf16 is ~16 GB, so it does
not fit on one T4 — we shard it across both with `--device-map auto`.

## One-time Kaggle setup

1. Sign in at kaggle.com. Verify your phone (Settings -> Phone Verification;
   required to enable GPU + internet in notebooks).
2. Store your Hugging Face token as a Kaggle secret so it is not pasted in code:
   in a notebook, **Add-ons -> Secrets -> Add secret**, label `HF_TOKEN`, value =
   your HF read token (the one with gated-repo access).
3. New Notebook -> right sidebar:
   - **Accelerator: GPU T4 x2**
   - **Internet: On**

## Notebook cells

Paste each block into its own cell and run top to bottom.

**Cell 1 — get the code**
```python
!git clone <repository-url>
%cd Algoverse-VNAA
```

**Cell 2 — install deps (Kaggle already has CUDA torch; do not reinstall it)**
```python
!pip install -q "transformers>=4.44" "datasets>=2.20" "scikit-learn>=1.5" \
    "tqdm>=4.66" "huggingface_hub>=0.24" "PyYAML>=6.0" "accelerate>=0.30"
!pip install -q -e . --no-deps
```

**Cell 3 — Hugging Face login (via the Kaggle secret, not pasted in code)**
```python
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
login(UserSecretsClient().get_secret("HF_TOKEN"))
```

**Cell 4 — download the SAD oversight data (encrypted, ~1.3 MB)**
```python
!mkdir -p data/sad
!curl -sL -o data/sad/oversight_structs.zip \
  https://github.com/LRudL/sad/raw/main/sad/stages/oversight/structs.zip
```

**Cell 5 — confirm both GPUs are visible**
```python
import torch
print("cuda:", torch.cuda.is_available(), "| gpus:", torch.cuda.device_count())
```
Expect `gpus: 2`. If it says 0, the accelerator is not set to GPU T4 x2.

**Cell 6 — run slice 1 on the 8B (shards across both T4s)**
```python
!python -m vnaa.run_slice1 \
  --sad-repo data/sad/oversight_structs.zip \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dtype bfloat16 --device-map auto
```

This prints the per-layer AUROC table (33 layers, 0 = embeddings) and saves the
activations + AUROC curve under `results/`. Expect: no separation at layer 0,
high `|AUROC - 0.5|` peaking in early-to-middle layers.

**Cell 7 (optional) — pull the AUROC curve out to view/download**
```python
import json
print(open("results/slice1_Llama-3.1-8B-Instruct_auroc.json").read())
```

## Notes

- First run downloads the 8B (~16 GB) into the notebook — a few minutes.
- The quota meter runs the whole time a session is open, even idle. Stop the
  session (not just close the tab) when done.
- Do not commit anything from `results/` or the SAD data (canary rule).
