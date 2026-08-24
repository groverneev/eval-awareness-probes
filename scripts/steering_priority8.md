# Priority 8 steering audit runbook

The protocol is frozen in Amendment 2 of
`datasets/steering/prespec.md`. This is explicitly a post-result,
mentor-requested robustness audit.

## Completed audit

```powershell
.\.venv\Scripts\python.exe -m vnaa.run_steering_priority8 prepare
.\.venv\Scripts\python.exe -m vnaa.run_steering_priority8 analyze
.\.venv\Scripts\python.exe -m vnaa.run_steering_priority8 readout `
  --dtype bfloat16 --device-map auto
```

The learned-direction prompt-level result, five orthogonal-control readouts,
and all behavioral-control generations are saved under ignored
`results/steering_priority8/` artifacts. The readout run is feasible on the
local 16GB card with CPU offload; behavior generation was completed on a 24GB
RunPod RTX 4090.

## GPU execution record

The frozen behavior audit contains 840 greedy generations:

- five control directions;
- coefficients `-2` and `+2`;
- 36 self-report plus 48 safety prompts.

The zero condition is mathematically direction-independent and is reused from
the original learned-direction artifacts. Every output is saved immediately,
so an interrupted run resumes at the first missing prompt.

After the code is available on the pod, upload the two ignored inputs:

```bash
scp -P <port> \
  results/steer_Llama-3.1-8B-Instruct_direction.npz \
  root@<host>:/workspace/Algoverse-VNAA/results/

scp -P <port> \
  results/steering_priority8/steer_Llama-3.1-8B-Instruct_orthogonal_controls.npz \
  root@<host>:/workspace/Algoverse-VNAA/results/steering_priority8/
```

Run on the pod:

```bash
cd /workspace/Algoverse-VNAA
bash scripts/run_steering_priority8_pod.sh
```

The script checks its frozen inputs, records GPU/software metadata, performs
both resumable generation sets, and writes an SHA-256 inventory.

Copy these artifacts back without replacing the already completed local
control-readout file:

```bash
scp -P <port> \
  root@<host>:/workspace/Algoverse-VNAA/results/steering_priority8/steer_Llama-3.1-8B-Instruct_control_generations_self_report.json \
  results/steering_priority8/

scp -P <port> \
  root@<host>:/workspace/Algoverse-VNAA/results/steering_priority8/steer_Llama-3.1-8B-Instruct_control_generations_safety.json \
  results/steering_priority8/
```

Then run locally:

```powershell
.\.venv\Scripts\python.exe -m vnaa.run_steering_priority8 analyze
.\.venv\Scripts\python.exe -m pytest
```

The final analysis applies the original degeneration veto to each control
endpoint, reports every control direction, performs the same paired prompt
inference as for the learned direction, and evaluates the frozen specificity
criterion. All endpoints pass the quality screen. The learned direction passes
readout specificity but fails behavioral specificity because two orthogonal
controls produce larger self-report endpoint shifts.
