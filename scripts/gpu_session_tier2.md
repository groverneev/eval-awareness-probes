# Tier 2 GPU session runbook

Everything here is frozen code and committed inputs. The session is a matter of
running the commands in order and copying artifacts back.

## Which machine, and why

**Use the Runpod RTX 5090 (32 GB), the same setup as the slice-5 session.**

All three remaining items are dominated by *generation*, not extraction. The 8B
in bf16 is ~16 GB, so a 32 GB card holds it wholly on-GPU and generates at full
speed.

- The local 5070 Ti (16 GB) needs `device_map="auto"` and offloads the last few
  layers to system RAM. That is fine for forward passes — the slice-1
  re-extraction below was done locally in a couple of minutes — but it makes
  greedy generation several times slower, and this session is ~1,750 generations.
- Kaggle's 2×T4 shards the model across two cards. It works (slice 1 ran there)
  but it is slower still for generation and adds session-length limits.

Estimated GPU time: **~2.5 hours**, so well under $3 on a 5090 pod.

## What you need to do before the pod starts

1. **Accept the LMSYS-Chat-1M terms** at
   `https://huggingface.co/datasets/lmsys/lmsys-chat-1m` with the same HF account
   that has Llama access (`LightningPro1`). It is a gated dataset and the prep
   step will 403 without this. This is the only hard blocker.
2. Start the pod and note the SSH details **outside this repo** (no keys, hosts,
   or ports in git).

Nothing else needs your input. Everything below is deterministic.

## Already done locally, no GPU needed

Slice-1 8B activations now exist on the desktop, so the estimator and
permutation-null checks cover slice 1. It reproduced the Kaggle run exactly:
layer 0 at 0.500, best layer 4, AUROC 0.9944.

```bash
.venv/Scripts/python -m vnaa.run_slice1 --sad-repo data/sad/oversight_structs.zip --model meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --device-map auto
```

## Pod setup

**Restart the existing slice-5 pod rather than creating a new one.** The 8B
weights already sit in `/workspace/.cache/huggingface`, which is the persistent
volume, so a restart skips a ~16 GB download. Two things change on restart: the
exposed SSH port is new, and anything pip installed outside `/workspace` may be
gone, so the install is re-run below. Record the port outside this repo.

```bash
cd /workspace && export HF_HOME=/workspace/.cache/huggingface
cd vnaa && git pull            # or clone if the volume was wiped
pip install -e . && pip install accelerate
hf auth login                  # only if the token did not persist in HF_HOME
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## What runs where

Only the stages that need a GPU run on the pod. Every `analyze` is CPU-only and
runs on the desktop, where the activation caches already live. That keeps the
upload to a single file and lets each result be audited locally before it is
written up.

| stage | machine | why |
|---|---|---|
| `generate`, `reconstruct`, `readout`, `extract` | pod | needs the model |
| `label`, `analyze` | desktop | CPU only, and the `.npz` caches are here |

**Upload before starting item 2** — the one private input the pod needs:

```bash
scp -P <port> results/steer_Llama-3.1-8B-Instruct_direction.npz root@<host>:/workspace/vnaa/results/
```

It is 161 KB. Uploading the frozen direction rather than refitting it on the pod
means the pod cannot silently produce a different one. Everything else the
session reads is committed.

---

## Item 1 — silence filter on `eval_shared` (~30 min)

The proposal crosses every cell with verbalized-vs-silence-filtered, and only
the matched-coherence cell has it. `eval_shared` is the other identical-prefix
control, so it is the one worth adding.

On the pod:

```bash
python -m vnaa.run_silence_filter generate --cell eval_shared --dtype bfloat16
```

Activations already exist and are not regenerated; this is generation only, 500
replies. It is restartable — the cache is written after every pair, so a dropped
connection costs one item.

Copy back `results/s3_*_eval_shared_greedy_responses.json`, then on the desktop:

```bash
.venv/Scripts/python -m vnaa.run_silence_filter label --cell eval_shared
.venv/Scripts/python -m vnaa.run_silence_filter analyze --cell eval_shared
```

Compare against the matched-coherence cell: 192/250 silent pairs, probe
0.937 +/- 0.033, BoW 0.823, format 0.567. A much lower cap-hit count is expected
here, since `eval_shared` questions are benchmark items with short answers.

## Item 2 — steering (~1 hour)

Protocol is fixed in `datasets/steering/prespec.md`, written before any steered
output existed. Upload the frozen direction first (see above).

On the pod, in this order:

```bash
python -m vnaa.run_steering readout --dtype bfloat16
```

`readout` is the manipulation check and it **gates the other two steps**. It
sweeps the four tied candidate layers (3, 7, 10, 14) against nine coefficients
using forward passes only, ~1,300 passes, and picks the injection layer. Copy
`results/steer_*_readout.json` back and inspect it before generating anything.

If no layer moves the independent readout monotonically, **stop and report
that**. The behavioural numbers would not be interpretable, and "failed
manipulation check" is a documented outcome in the pre-spec, not something to
work around by trying more layers.

If it passes:

```bash
python -m vnaa.run_steering generate --set self_report --dtype bfloat16
python -m vnaa.run_steering generate --set safety --dtype bfloat16
```

Copy back everything matching `results/steer_*`, then on the desktop:

```bash
.venv/Scripts/python -m vnaa.run_steering analyze
```

Read the degeneracy flags before reading the outcome rates. Coefficients at
`alpha = +/-4` are expected to trip them, since the class-mean gap is 22-34% of
the mean residual norm at layers 7-14.

## Item 3 — slice 5 on LMSYS-Chat-1M (~45 min)

The proposal names LMSYS alongside WildChat and it has never been used, so
slice 5 currently rests on a single corpus.

```bash
python -m vnaa.prep_slice5 --source lmsys --n 250 --scan-limit 400000
python -m vnaa.run_slice5 reconstruct \
  --dataset data/slice5_genuine_lmsys.json \
  --output data/slice5_reconstructed_lmsys_n250.json \
  --generation-context full --dtype bfloat16
python -m vnaa.run_slice5 extract \
  --dataset data/slice5_genuine_lmsys.json \
  --reconstructions data/slice5_reconstructed_lmsys_n250.json \
  --tag reconstruction_lmsys_n250 --dtype bfloat16
```

Then copy back and analyze on the desktop:

```bash
.venv/Scripts/python -m vnaa.run_slice5 analyze \
  --dataset data/slice5_genuine_lmsys.json \
  --reconstructions data/slice5_reconstructed_lmsys_n250.json \
  --tag reconstruction_lmsys_n250
```

The protocol is byte-for-byte the one frozen for WildChat, so this is a corpus
replication and nothing else. The WildChat n=250 figures to compare against are
`final_assistant_token` 0.792 +/- 0.024 and `final_assistant_mean` 0.822 +/-
0.018.

LMSYS differs from WildChat in two ways the prep step handles: the language tag
is on the record rather than per message, and PII is replaced with placeholders
such as `NAME_1`. Redacted conversations are dropped, because those placeholders
are a surface artifact a probe or bag-of-words baseline could read as a label.

Copy back: `data/slice5_genuine_lmsys.json`,
`data/slice5_reconstructed_lmsys_n250.json`, and
`results/s5_*_reconstruction_lmsys_n250_*`.

---

## After the session

```bash
.venv/Scripts/python -m vnaa.run_estimator_check points
.venv/Scripts/python -m pytest -q
```

The estimator check picks up any new cached contrast automatically, so the
logistic estimator and the permutation null extend to the new cells with no
extra work.

Do not commit `data/` or `results/` — both are gitignored and contain real
WildChat- and LMSYS-derived conversation text. SHA-256 the artifacts after
copying back, as was done for slice 5.
