#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "usage: $0 CONFIG ARTIFACT_ROOT [starrygl arguments...]" >&2
  exit 2
fi

config=$1
artifact_root=$2
shift 2

exec torchrun \
  --nnodes="${NNODES:-1}" \
  --nproc_per_node="${NPROC_PER_NODE:-4}" \
  --node_rank="${NODE_RANK:-0}" \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29500}" \
  -m starrygl.cli.main \
  "$config" \
  --artifact-root "$artifact_root" \
  "$@"
