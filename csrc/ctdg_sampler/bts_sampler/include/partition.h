#pragma once

#include "extension.h"

at::Tensor metis_partition(
    at::Tensor rowptr,
    at::Tensor col,
    at::optional<at::Tensor> opt_value,
    at::optional<at::Tensor> opt_vtx_w,
    at::optional<at::Tensor> opt_vtx_s,
    int64_t num_parts,
    bool recursive,
    bool min_edge_cut
);

at::Tensor metis_cache_friendly_reordering(
    at::Tensor rowptr,
    at::Tensor col,
    at::Tensor part
);

at::Tensor mt_metis_partition(
    at::Tensor rowptr,
    at::Tensor col,
    at::optional<at::Tensor> opt_value,
    at::optional<at::Tensor> opt_vtx_w,
    int64_t num_parts,
    int64_t num_workers,
    bool recursive
);

at::Tensor ldg_partition(
    at::Tensor edges,
    at::optional<at::Tensor> vertex_weights,
    at::optional<at::Tensor> initial_partition,
    int64_t num_parts,
    int64_t num_workers
);
