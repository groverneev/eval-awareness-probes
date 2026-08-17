#!/usr/bin/env bash
set -euo pipefail

cd /workspace/Algoverse-VNAA

mkdir -p results/qwen_scale/v1

watchdog_pid=unavailable
if [[ -n "${RUNPOD_POD_ID:-}" ]] \
  && [[ -n "${RUNPOD_API_KEY:-}" ]] \
  && command -v runpodctl >/dev/null; then
  nohup bash -c 'sleep 28800; runpodctl pod stop "$RUNPOD_POD_ID"' \
    > results/qwen_scale/v1/watchdog.log 2>&1 &
  watchdog_pid=$!
  echo "$watchdog_pid" > /workspace/qwen_watchdog.pid
else
  echo "pod self-stop watchdog unavailable; external API monitor required"
fi

nohup bash scripts/run_qwen_scale_pod.sh \
  > results/qwen_scale/v1/full_run.log 2>&1 &
runner_pid=$!
echo "$runner_pid" > /workspace/qwen_runner.pid

sleep 2
echo "runner=$runner_pid watchdog=$watchdog_pid"
ps -p "$runner_pid" -o pid,etime,cmd
