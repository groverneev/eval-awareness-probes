#!/usr/bin/env bash
# Fail-safe for unattended research Pods. RunPod injects a Pod-scoped API key.
set -u

delay_seconds="${1:-21600}"
sleep "$delay_seconds"
curl --fail --silent --show-error --retry 5 --request DELETE \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
  "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}"
