#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

mode="${1:-pilot}"
if (( $# > 0 )); then
  shift
fi

case "$mode" in
  pilot)
    sweep="configs/sweeps/all_methods_pilot.toml"
    ;;
  full)
    sweep="configs/sweeps/all_methods.toml"
    ;;
  clip-audit)
    sweep="configs/sweeps/clip_audit.toml"
    ;;
  *)
    printf 'Usage: %s [pilot|clip-audit|full] [--dry-run]\n' "$0" >&2
    exit 2
    ;;
esac

exec uv run contrast sweep "$sweep" "$@"
