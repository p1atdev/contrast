#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

# SupCon was selected from validation results at epoch 80. Keep the test query
# limited to these three preselected seed checkpoints to avoid test-set tuning.
checkpoints=(
  "runs/cifar100-core-v3/20260827T074453Z-0000/checkpoints/epoch-0080.pt"
  "runs/cifar100-core-v3/20260827T091938Z-0001/checkpoints/epoch-0080.pt"
  "runs/cifar100-core-v3/20260827T105407Z-0002/checkpoints/epoch-0080.pt"
)

for checkpoint in "${checkpoints[@]}"; do
  if [[ ! -f "$checkpoint" ]]; then
    printf 'Checkpoint not found: %s\n' "$checkpoint" >&2
    exit 1
  fi
done

for checkpoint in "${checkpoints[@]}"; do
  printf 'Evaluating test split: %s\n' "$checkpoint"
  uv run contrast evaluate \
    --checkpoint "$checkpoint" \
    --queries test
done
