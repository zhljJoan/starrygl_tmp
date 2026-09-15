#pragma once

#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::dict speed_partition(
    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor ts,
    int64_t num_nodes,
    int64_t num_parts,
    double beta,
    double topk_ratio,
    const std::string& topk_type);

torch::Tensor assign_chunks_temporal_balance(
    torch::Tensor chunk_load,
    torch::Tensor affinity,
    int64_t world_size,
    int64_t chunks_per_rank,
    double affinity_weight,
    int64_t local_search_iters);
