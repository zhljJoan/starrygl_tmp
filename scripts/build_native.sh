#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-/tmp/starrygl_native_build}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DPython3_EXECUTABLE="${PYTHON_BIN}"
cmake --build "${BUILD_DIR}" --target libstarrygl_sampler adaptive_split_cpp -j "${JOBS:-$(nproc)}"

ROOT_DIR="${ROOT_DIR}" "${PYTHON_BIN}" - <<'PY'
import importlib.util
import os
from pathlib import Path

import torch

path = Path(os.environ["ROOT_DIR"]) / "src/starrygl/native/lib/libstarrygl_sampler.so"
spec = importlib.util.spec_from_file_location("libstarrygl_sampler", path)
assert spec is not None and spec.loader is not None
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)

src = torch.tensor([0], dtype=torch.long)
dst = torch.tensor([1], dtype=torch.long)
edge_ids = torch.tensor([0], dtype=torch.long)
timestamps = torch.tensor([1], dtype=torch.long)
graph = native.get_neighbors("starrygl_native_abi_smoke", src, dst, 2, 0, edge_ids, None, None, timestamps)
native.ParallelSampler(
    graph,
    2,
    1,
    1,
    [1],
    1,
    "recent",
    0,
    torch.empty(0, dtype=torch.int32),
    torch.empty(0, dtype=torch.int32),
    torch.empty(0, dtype=torch.uint8),
    0.1,
)
print("StarryGL native sampler ABI smoke passed")
PY
