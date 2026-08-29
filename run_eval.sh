#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

mode="${1:-selected-test}"

case "$mode" in
  selected-test)
    # SupCon was selected from validation results at epoch 80. Keep the test
    # query limited to these preselected checkpoints to avoid test-set tuning.
    checkpoints=(
      "runs/cifar100-core-v3/20260827T074453Z-0000/checkpoints/epoch-0080.pt"
      "runs/cifar100-core-v3/20260827T091938Z-0001/checkpoints/epoch-0080.pt"
      "runs/cifar100-core-v3/20260827T105407Z-0002/checkpoints/epoch-0080.pt"
    )
    queries="test"
    extra_arguments=()
    ;;
  all-methods-pilot)
    checkpoints=(runs/cifar100-all-methods-v1-pilot/*/checkpoints/final.pt)
    if (( ${#checkpoints[@]} != 16 )); then
      printf 'Expected 16 pilot checkpoints, found %d\n' "${#checkpoints[@]}" >&2
      exit 1
    fi
    queries="eval"
    extra_arguments=(--skip-linear-probe)
    ;;
  *)
    printf 'Usage: %s [selected-test|all-methods-pilot]\n' "$0" >&2
    exit 2
    ;;
esac

for checkpoint in "${checkpoints[@]}"; do
  if [[ ! -f "$checkpoint" ]]; then
    printf 'Checkpoint not found: %s\n' "$checkpoint" >&2
    exit 1
  fi
done

for checkpoint in "${checkpoints[@]}"; do
  printf 'Evaluating %s split: %s\n' "$queries" "$checkpoint"
  uv run contrast evaluate \
    --checkpoint "$checkpoint" \
    --queries "$queries" \
    "${extra_arguments[@]}"
done
