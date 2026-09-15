#include <torch/extension.h>

#include <parallel_hashmap/phmap.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

struct PairKey {
  int64_t value;
  int64_t ts;

  bool operator==(const PairKey& other) const {
    return value == other.value && ts == other.ts;
  }
};

struct PairHash {
  size_t operator()(const PairKey& key) const {
    const uint64_t a = static_cast<uint64_t>(key.value);
    const uint64_t b = static_cast<uint64_t>(key.ts);
    return static_cast<size_t>(a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2)));
  }
};

torch::Tensor contiguous_long_cpu(torch::Tensor x, const char* name) {
  if (x.device().is_cuda()) {
    throw std::runtime_error(std::string(name) + " must be a CPU tensor");
  }
  return x.to(torch::kLong).contiguous().view({-1});
}

}  // namespace

std::vector<torch::Tensor> stable_unique(torch::Tensor values) {
  auto vals = contiguous_long_cpu(values, "values");
  const auto n = vals.numel();
  auto* val_ptr = vals.data_ptr<int64_t>();

  phmap::flat_hash_map<int64_t, int64_t> seen;
  seen.reserve(static_cast<size_t>(n));

  std::vector<int64_t> unique;
  unique.reserve(static_cast<size_t>(n));
  auto inverse = torch::empty({n}, torch::dtype(torch::kLong).device(torch::kCPU));
  auto* inv_ptr = inverse.data_ptr<int64_t>();

  for (int64_t i = 0; i < n; ++i) {
    const int64_t value = val_ptr[i];
    auto it = seen.find(value);
    if (it == seen.end()) {
      const int64_t row = static_cast<int64_t>(unique.size());
      seen.emplace(value, row);
      unique.push_back(value);
      inv_ptr[i] = row;
    } else {
      inv_ptr[i] = it->second;
    }
  }

  auto out = torch::empty({static_cast<int64_t>(unique.size())}, torch::dtype(torch::kLong).device(torch::kCPU));
  if (!unique.empty()) {
    std::memcpy(out.data_ptr<int64_t>(), unique.data(), unique.size() * sizeof(int64_t));
  }
  return {out.to(values.scalar_type()), inverse};
}

std::vector<torch::Tensor> stable_unique_with_ts(torch::Tensor values, torch::Tensor ts) {
  auto vals = contiguous_long_cpu(values, "values");
  auto times = contiguous_long_cpu(ts, "ts");
  const auto n = vals.numel();
  if (times.numel() != n) {
    throw std::runtime_error("values and ts must have the same number of elements");
  }
  auto* val_ptr = vals.data_ptr<int64_t>();
  auto* ts_ptr = times.data_ptr<int64_t>();

  phmap::flat_hash_map<PairKey, int64_t, PairHash> seen;
  seen.reserve(static_cast<size_t>(n));

  std::vector<int64_t> unique;
  unique.reserve(static_cast<size_t>(n));
  std::vector<int64_t> first_ts;
  first_ts.reserve(static_cast<size_t>(n));
  auto inverse = torch::empty({n}, torch::dtype(torch::kLong).device(torch::kCPU));
  auto* inv_ptr = inverse.data_ptr<int64_t>();

  for (int64_t i = 0; i < n; ++i) {
    const PairKey key{val_ptr[i], ts_ptr[i]};
    auto it = seen.find(key);
    if (it == seen.end()) {
      const int64_t row = static_cast<int64_t>(unique.size());
      seen.emplace(key, row);
      unique.push_back(key.value);
      first_ts.push_back(key.ts);
      inv_ptr[i] = row;
    } else {
      inv_ptr[i] = it->second;
    }
  }

  auto uniq = torch::empty({static_cast<int64_t>(unique.size())}, torch::dtype(torch::kLong).device(torch::kCPU));
  auto first = torch::empty({static_cast<int64_t>(first_ts.size())}, torch::dtype(torch::kLong).device(torch::kCPU));
  if (!unique.empty()) {
    std::memcpy(uniq.data_ptr<int64_t>(), unique.data(), unique.size() * sizeof(int64_t));
    std::memcpy(first.data_ptr<int64_t>(), first_ts.data(), first_ts.size() * sizeof(int64_t));
  }
  return {uniq.to(values.scalar_type()), inverse, first.to(ts.scalar_type())};
}

std::vector<torch::Tensor> deduplicate_csc_edges(torch::Tensor csc_indptr, torch::Tensor edge_lids) {
  auto indptr = contiguous_long_cpu(csc_indptr, "csc_indptr");
  auto edges = contiguous_long_cpu(edge_lids, "edge_lids");
  if (indptr.numel() == 0) {
    throw std::runtime_error("csc_indptr must have at least one element");
  }
  if (indptr.data_ptr<int64_t>()[indptr.numel() - 1] != edges.numel()) {
    throw std::runtime_error("csc_indptr[-1] must match edge_lids length");
  }

  const int64_t num_dst = indptr.numel() - 1;
  auto* indptr_ptr = indptr.data_ptr<int64_t>();
  auto* edge_ptr = edges.data_ptr<int64_t>();

  phmap::flat_hash_set<PairKey, PairHash> seen;
  seen.reserve(static_cast<size_t>(edges.numel()));

  std::vector<int64_t> keep;
  keep.reserve(static_cast<size_t>(edges.numel()));
  auto new_indptr = torch::empty({num_dst + 1}, torch::dtype(torch::kLong).device(torch::kCPU));
  auto* new_ptr = new_indptr.data_ptr<int64_t>();
  new_ptr[0] = 0;

  for (int64_t col = 0; col < num_dst; ++col) {
    const int64_t begin = indptr_ptr[col];
    const int64_t end = indptr_ptr[col + 1];
    if (begin < 0 || end < begin || end > edges.numel()) {
      throw std::runtime_error("csc_indptr contains an invalid range");
    }
    for (int64_t pos = begin; pos < end; ++pos) {
      const PairKey key{col, edge_ptr[pos]};
      if (seen.insert(key).second) {
        keep.push_back(pos);
      }
    }
    new_ptr[col + 1] = static_cast<int64_t>(keep.size());
  }

  auto keep_index = torch::empty({static_cast<int64_t>(keep.size())}, torch::dtype(torch::kLong).device(torch::kCPU));
  if (!keep.empty()) {
    std::memcpy(keep_index.data_ptr<int64_t>(), keep.data(), keep.size() * sizeof(int64_t));
  }
  return {keep_index, new_indptr};
}

torch::Tensor first_ts_for_lids(torch::Tensor root_ts, torch::Tensor root_lids, int64_t size) {
  auto times = contiguous_long_cpu(root_ts, "root_ts");
  auto lids = contiguous_long_cpu(root_lids, "root_lids");
  const auto n = lids.numel();
  if (times.numel() != n) {
    throw std::runtime_error("root_ts and root_lids must have the same number of elements");
  }
  auto out = torch::zeros({size}, torch::dtype(torch::kLong).device(torch::kCPU));
  std::vector<uint8_t> filled(static_cast<size_t>(size), 0);
  auto* out_ptr = out.data_ptr<int64_t>();
  auto* ts_ptr = times.data_ptr<int64_t>();
  auto* lid_ptr = lids.data_ptr<int64_t>();
  for (int64_t i = 0; i < n; ++i) {
    const int64_t lid = lid_ptr[i];
    if (lid < 0 || lid >= size) {
      throw std::runtime_error("root_lids contains an out-of-range row");
    }
    if (!filled[static_cast<size_t>(lid)]) {
      out_ptr[lid] = ts_ptr[i];
      filled[static_cast<size_t>(lid)] = 1;
    }
  }
  return out.to(root_ts.scalar_type());
}

std::vector<torch::Tensor> build_slice_topology_impl(
    torch::Tensor eids,
    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor edge_ids,
    torch::Tensor node_to_chunk,
    torch::Tensor full_dst_ids,
    bool use_full_dst) {
  auto e = contiguous_long_cpu(eids, "eids");
  auto s_all = contiguous_long_cpu(src, "src");
  auto d_all = contiguous_long_cpu(dst, "dst");
  auto gid_all = contiguous_long_cpu(edge_ids, "edge_ids");
  auto chunks_all = contiguous_long_cpu(node_to_chunk, "node_to_chunk");
  auto full_dst = contiguous_long_cpu(full_dst_ids, "full_dst_ids");
  const auto n = e.numel();

  auto* e_ptr = e.data_ptr<int64_t>();
  auto* src_all_ptr = s_all.data_ptr<int64_t>();
  auto* dst_all_ptr = d_all.data_ptr<int64_t>();
  auto* gid_ptr = gid_all.data_ptr<int64_t>();
  auto* chunk_ptr = chunks_all.data_ptr<int64_t>();
  auto* full_ptr = full_dst.data_ptr<int64_t>();

  std::vector<int64_t> order(static_cast<size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
    const int64_t ea = e_ptr[a];
    const int64_t eb = e_ptr[b];
    const int64_t da = dst_all_ptr[ea];
    const int64_t db = dst_all_ptr[eb];
    const int64_t ca = chunk_ptr[da];
    const int64_t cb = chunk_ptr[db];
    if (ca != cb) return ca < cb;
    if (da != db) return da < db;
    return gid_ptr[ea] < gid_ptr[eb];
  });

  std::vector<int64_t> s_ordered;
  std::vector<int64_t> d_ordered;
  std::vector<int64_t> gids;
  s_ordered.reserve(static_cast<size_t>(n));
  d_ordered.reserve(static_cast<size_t>(n));
  gids.reserve(static_cast<size_t>(n));
  for (const int64_t pos : order) {
    const int64_t eid = e_ptr[pos];
    s_ordered.push_back(src_all_ptr[eid]);
    d_ordered.push_back(dst_all_ptr[eid]);
    gids.push_back(gid_ptr[eid]);
  }

  std::vector<int64_t> dst_ids;
  if (use_full_dst) {
    dst_ids.assign(full_ptr, full_ptr + full_dst.numel());
  } else {
    dst_ids = d_ordered;
    std::sort(dst_ids.begin(), dst_ids.end());
    dst_ids.erase(std::unique(dst_ids.begin(), dst_ids.end()), dst_ids.end());
  }

  phmap::flat_hash_map<int64_t, int64_t> dst_row;
  dst_row.reserve(dst_ids.size());
  for (int64_t i = 0; i < static_cast<int64_t>(dst_ids.size()); ++i) {
    dst_row.emplace(dst_ids[static_cast<size_t>(i)], i);
  }

  std::vector<int64_t> src_ids = s_ordered;
  std::sort(src_ids.begin(), src_ids.end());
  src_ids.erase(std::unique(src_ids.begin(), src_ids.end()), src_ids.end());
  src_ids.erase(
      std::remove_if(src_ids.begin(), src_ids.end(), [&](int64_t node) {
        return dst_row.find(node) != dst_row.end();
      }),
      src_ids.end());

  phmap::flat_hash_map<int64_t, int64_t> src_tail_row;
  src_tail_row.reserve(src_ids.size());
  const int64_t dst_count = static_cast<int64_t>(dst_ids.size());
  for (int64_t i = 0; i < static_cast<int64_t>(src_ids.size()); ++i) {
    src_tail_row.emplace(src_ids[static_cast<size_t>(i)], dst_count + i);
  }

  std::vector<int64_t> edge_src;
  std::vector<int64_t> edge_dst;
  std::vector<int64_t> edge_ptr(static_cast<size_t>(dst_count + 1), 0);
  edge_src.reserve(static_cast<size_t>(n));
  edge_dst.reserve(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    const int64_t sv = s_ordered[static_cast<size_t>(i)];
    const int64_t dv = d_ordered[static_cast<size_t>(i)];
    const auto dit = dst_row.find(dv);
    if (dit == dst_row.end()) {
      throw std::runtime_error("destination node is missing from dst_ids");
    }
    const int64_t drow = dit->second;
    const auto sit_dst = dst_row.find(sv);
    if (sit_dst != dst_row.end()) {
      edge_src.push_back(sit_dst->second);
    } else {
      const auto sit_tail = src_tail_row.find(sv);
      if (sit_tail == src_tail_row.end()) {
        throw std::runtime_error("source node is missing from src rows");
      }
      edge_src.push_back(sit_tail->second);
    }
    edge_dst.push_back(drow);
    edge_ptr[static_cast<size_t>(drow + 1)] += 1;
  }
  for (int64_t i = 1; i <= dst_count; ++i) {
    edge_ptr[static_cast<size_t>(i)] += edge_ptr[static_cast<size_t>(i - 1)];
  }

  std::vector<int64_t> dst_chunk;
  dst_chunk.reserve(dst_ids.size());
  for (const int64_t node : dst_ids) {
    dst_chunk.push_back(chunk_ptr[node]);
  }

  auto make_tensor = [](const std::vector<int64_t>& values) {
    auto out = torch::empty({static_cast<int64_t>(values.size())}, torch::dtype(torch::kLong).device(torch::kCPU));
    if (!values.empty()) {
      std::memcpy(out.data_ptr<int64_t>(), values.data(), values.size() * sizeof(int64_t));
    }
    return out;
  };

  return {
      make_tensor(src_ids),
      make_tensor(dst_ids),
      make_tensor(gids),
      make_tensor(edge_src),
      make_tensor(edge_dst),
      make_tensor(edge_ptr),
      make_tensor(dst_chunk),
  };
}

std::vector<torch::Tensor> build_slice_topology(
    torch::Tensor eids,
    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor edge_ids,
    torch::Tensor node_to_chunk) {
  return build_slice_topology_impl(
      eids,
      src,
      dst,
      edge_ids,
      node_to_chunk,
      torch::empty({0}, torch::dtype(torch::kLong).device(torch::kCPU)),
      false);
}

std::vector<torch::Tensor> build_slice_topology_full(
    torch::Tensor eids,
    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor edge_ids,
    torch::Tensor node_to_chunk,
    torch::Tensor full_dst_ids) {
  return build_slice_topology_impl(eids, src, dst, edge_ids, node_to_chunk, full_dst_ids, true);
}

std::vector<torch::Tensor> build_partition_topology_impl(
    torch::Tensor local_edge_ids,
    torch::Tensor time_ptr_2,
    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor edge_ids,
    torch::Tensor node_to_chunk,
    torch::Tensor full_dst_ids,
    torch::Tensor edge_weight,
    bool use_full_dst,
    bool build_gcn_norm) {
  auto local_edges = contiguous_long_cpu(local_edge_ids, "local_edge_ids");
  auto time_ptr = contiguous_long_cpu(time_ptr_2, "time_ptr_2");
  auto s_all = contiguous_long_cpu(src, "src");
  auto d_all = contiguous_long_cpu(dst, "dst");
  auto gid_all = contiguous_long_cpu(edge_ids, "edge_ids");
  auto chunks_all = contiguous_long_cpu(node_to_chunk, "node_to_chunk");
  auto full_dst = contiguous_long_cpu(full_dst_ids, "full_dst_ids");
  torch::Tensor weights;
  const bool has_weight = edge_weight.numel() > 0;
  if (has_weight) {
    if (edge_weight.device().is_cuda()) {
      throw std::runtime_error("edge_weight must be a CPU tensor");
    }
    weights = edge_weight.to(torch::kDouble).contiguous().view({-1});
  }
  if (time_ptr.numel() % 2 != 0) {
    throw std::runtime_error("time_ptr_2 must have shape [num_slices, 2]");
  }
  auto* local_ptr = local_edges.data_ptr<int64_t>();
  auto* tp = time_ptr.data_ptr<int64_t>();
  auto* src_all_ptr = s_all.data_ptr<int64_t>();
  auto* dst_all_ptr = d_all.data_ptr<int64_t>();
  auto* gid_ptr = gid_all.data_ptr<int64_t>();
  auto* weight_ptr = has_weight ? weights.data_ptr<double>() : nullptr;
  const int64_t num_slices = time_ptr.numel() / 2;

  std::vector<int64_t> sorted_edges(local_ptr, local_ptr + local_edges.numel());
  std::sort(sorted_edges.begin(), sorted_edges.end());

  phmap::flat_hash_map<int64_t, int64_t> gid_to_pos;
  gid_to_pos.reserve(static_cast<size_t>(gid_all.numel()));
  for (int64_t i = 0; i < gid_all.numel(); ++i) {
    gid_to_pos.emplace(gid_ptr[i], i);
  }

  std::vector<int64_t> src_data, src_ptr{0};
  std::vector<int64_t> dst_data, dst_ptr{0};
  std::vector<int64_t> edge_id_data, edge_id_ptr{0};
  std::vector<int64_t> event_pos_data, event_pos_ptr{0};
  std::vector<int64_t> edge_src_data, edge_src_ptr{0};
  std::vector<int64_t> edge_dst_data, edge_dst_ptr{0};
  std::vector<int64_t> edge_ptr_data, edge_ptr_ptr{0};
  std::vector<int64_t> dst_chunk_data, dst_chunk_ptr{0};
  std::vector<int64_t> combined_data, combined_ptr{0};
  std::vector<float> gcn_norm_data;
  std::vector<int64_t> gcn_norm_ptr{0};

  auto append = [](std::vector<int64_t>& data, std::vector<int64_t>& ptr, const torch::Tensor& tensor) {
    auto t = tensor.to(torch::kLong).contiguous().view({-1});
    auto* p = t.data_ptr<int64_t>();
    data.insert(data.end(), p, p + t.numel());
    ptr.push_back(static_cast<int64_t>(data.size()));
  };
  auto append_float_vec = [](std::vector<float>& data, std::vector<int64_t>& ptr, const std::vector<float>& values) {
    data.insert(data.end(), values.begin(), values.end());
    ptr.push_back(static_cast<int64_t>(data.size()));
  };

  auto norm_for_gids = [&](const torch::Tensor& gids_tensor) {
    auto gids = gids_tensor.to(torch::kLong).contiguous().view({-1});
    auto* gids_ptr = gids.data_ptr<int64_t>();
    const int64_t n = gids.numel();
    std::vector<int64_t> positions;
    positions.reserve(static_cast<size_t>(n));
    std::vector<double> weight_values;
    weight_values.reserve(static_cast<size_t>(n));
    phmap::flat_hash_map<int64_t, double> in_deg;
    phmap::flat_hash_map<int64_t, double> out_deg;
    in_deg.reserve(static_cast<size_t>(n));
    out_deg.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
      const int64_t gid = gids_ptr[i];
      const auto pos_it = gid_to_pos.find(gid);
      if (pos_it == gid_to_pos.end()) {
        throw std::runtime_error("edge gid is not present in edge_ids");
      }
      const int64_t pos = pos_it->second;
      positions.push_back(pos);
      if (has_weight && (gid < 0 || gid >= weights.numel())) {
        throw std::runtime_error("edge_weight must be indexed by global edge id");
      }
      const double w = has_weight ? weight_ptr[gid] : 1.0;
      weight_values.push_back(w);
      in_deg[dst_all_ptr[pos]] += w;
      out_deg[src_all_ptr[pos]] += w;
    }
    std::vector<float> out;
    out.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
      const int64_t pos = positions[static_cast<size_t>(i)];
      const double in_v = std::max(in_deg[dst_all_ptr[pos]], 1e-12);
      const double out_v = std::max(out_deg[src_all_ptr[pos]], 1e-12);
      const double denom = std::sqrt(in_v) * std::sqrt(out_v);
      const double value = denom == 0.0 ? 0.0 : weight_values[static_cast<size_t>(i)] / denom;
      out.push_back(static_cast<float>(std::isfinite(value) ? value : 0.0));
    }
    return out;
  };

  for (int64_t sid = 0; sid < num_slices; ++sid) {
    const int64_t begin = tp[sid * 2];
    const int64_t end = tp[sid * 2 + 1];
    auto left = std::lower_bound(sorted_edges.begin(), sorted_edges.end(), begin);
    auto right = std::lower_bound(sorted_edges.begin(), sorted_edges.end(), end);
    std::vector<int64_t> eids_vec(left, right);
    auto eids = torch::empty({static_cast<int64_t>(eids_vec.size())}, torch::dtype(torch::kLong).device(torch::kCPU));
    if (!eids_vec.empty()) {
      std::memcpy(eids.data_ptr<int64_t>(), eids_vec.data(), eids_vec.size() * sizeof(int64_t));
    }

    std::vector<torch::Tensor> topo;
    if (eids_vec.empty()) {
      auto empty = torch::empty({0}, torch::dtype(torch::kLong).device(torch::kCPU));
      auto dst_ids = use_full_dst ? full_dst.clone() : empty;
      auto edge_ptr = torch::zeros({dst_ids.numel() + 1}, torch::dtype(torch::kLong).device(torch::kCPU));
      auto dst_chunk = torch::empty({dst_ids.numel()}, torch::dtype(torch::kLong).device(torch::kCPU));
      auto* dst_id_ptr = dst_ids.data_ptr<int64_t>();
      auto* chunk_ptr = chunks_all.data_ptr<int64_t>();
      auto* out_chunk = dst_chunk.data_ptr<int64_t>();
      for (int64_t i = 0; i < dst_ids.numel(); ++i) {
        out_chunk[i] = chunk_ptr[dst_id_ptr[i]];
      }
      topo = {empty, dst_ids, empty, empty, empty, edge_ptr, dst_chunk};
    } else {
      topo = build_slice_topology_impl(eids, s_all, d_all, gid_all, chunks_all, full_dst, use_full_dst);
    }
    auto combined = torch::cat({topo[1], topo[0]}, 0);
    auto ordered_gids = topo[2].to(torch::kLong).contiguous().view({-1});
    auto* ordered_gid_ptr = ordered_gids.data_ptr<int64_t>();
    std::vector<int64_t> ordered_eids;
    ordered_eids.reserve(static_cast<size_t>(ordered_gids.numel()));
    for (int64_t i = 0; i < ordered_gids.numel(); ++i) {
      const auto pos_it = gid_to_pos.find(ordered_gid_ptr[i]);
      if (pos_it == gid_to_pos.end()) {
        throw std::runtime_error("edge gid is not present in edge_ids");
      }
      ordered_eids.push_back(pos_it->second);
    }
    auto ordered_eids_tensor = torch::empty({static_cast<int64_t>(ordered_eids.size())}, torch::dtype(torch::kLong).device(torch::kCPU));
    if (!ordered_eids.empty()) {
      std::memcpy(ordered_eids_tensor.data_ptr<int64_t>(), ordered_eids.data(), ordered_eids.size() * sizeof(int64_t));
    }
    append(src_data, src_ptr, topo[0]);
    append(dst_data, dst_ptr, topo[1]);
    append(edge_id_data, edge_id_ptr, topo[2]);
    append(event_pos_data, event_pos_ptr, ordered_eids_tensor);
    append(edge_src_data, edge_src_ptr, topo[3]);
    append(edge_dst_data, edge_dst_ptr, topo[4]);
    append(edge_ptr_data, edge_ptr_ptr, topo[5]);
    append(dst_chunk_data, dst_chunk_ptr, topo[6]);
    append(combined_data, combined_ptr, combined);
    if (build_gcn_norm) {
      append_float_vec(gcn_norm_data, gcn_norm_ptr, norm_for_gids(topo[2]));
    }
  }

  auto make_tensor = [](const std::vector<int64_t>& values) {
    auto out = torch::empty({static_cast<int64_t>(values.size())}, torch::dtype(torch::kLong).device(torch::kCPU));
    if (!values.empty()) {
      std::memcpy(out.data_ptr<int64_t>(), values.data(), values.size() * sizeof(int64_t));
    }
    return out;
  };

  auto out = std::vector<torch::Tensor>{
      make_tensor(src_data), make_tensor(src_ptr),
      make_tensor(dst_data), make_tensor(dst_ptr),
      make_tensor(edge_id_data), make_tensor(edge_id_ptr),
      make_tensor(event_pos_data), make_tensor(event_pos_ptr),
      make_tensor(edge_src_data), make_tensor(edge_src_ptr),
      make_tensor(edge_dst_data), make_tensor(edge_dst_ptr),
      make_tensor(edge_ptr_data), make_tensor(edge_ptr_ptr),
      make_tensor(dst_chunk_data), make_tensor(dst_chunk_ptr),
      make_tensor(combined_data), make_tensor(combined_ptr),
  };
  if (build_gcn_norm) {
    auto norm = torch::empty({static_cast<int64_t>(gcn_norm_data.size())}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    if (!gcn_norm_data.empty()) {
      std::memcpy(norm.data_ptr<float>(), gcn_norm_data.data(), gcn_norm_data.size() * sizeof(float));
    }
    out.push_back(norm);
    out.push_back(make_tensor(gcn_norm_ptr));
  }
  return out;
}

std::vector<torch::Tensor> build_partition_topology(
    torch::Tensor local_edge_ids,
    torch::Tensor time_ptr_2,
    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor edge_ids,
    torch::Tensor node_to_chunk) {
  return build_partition_topology_impl(
      local_edge_ids,
      time_ptr_2,
      src,
      dst,
      edge_ids,
      node_to_chunk,
      torch::empty({0}, torch::dtype(torch::kLong).device(torch::kCPU)),
      torch::empty({0}, torch::dtype(torch::kFloat32).device(torch::kCPU)),
      false,
      false);
}

std::vector<torch::Tensor> build_partition_topology_full(
    torch::Tensor local_edge_ids,
    torch::Tensor time_ptr_2,
    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor edge_ids,
    torch::Tensor node_to_chunk,
    torch::Tensor full_dst_ids) {
  return build_partition_topology_impl(
      local_edge_ids,
      time_ptr_2,
      src,
      dst,
      edge_ids,
      node_to_chunk,
      full_dst_ids,
      torch::empty({0}, torch::dtype(torch::kFloat32).device(torch::kCPU)),
      true,
      false);
}

std::vector<torch::Tensor> build_partition_topology_with_norm(
    torch::Tensor local_edge_ids,
    torch::Tensor time_ptr_2,
    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor edge_ids,
    torch::Tensor node_to_chunk,
    torch::Tensor edge_weight,
    bool build_gcn_norm) {
  return build_partition_topology_impl(
      local_edge_ids,
      time_ptr_2,
      src,
      dst,
      edge_ids,
      node_to_chunk,
      torch::empty({0}, torch::dtype(torch::kLong).device(torch::kCPU)),
      edge_weight,
      false,
      build_gcn_norm);
}

std::vector<torch::Tensor> build_partition_topology_full_with_norm(
    torch::Tensor local_edge_ids,
    torch::Tensor time_ptr_2,
    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor edge_ids,
    torch::Tensor node_to_chunk,
    torch::Tensor full_dst_ids,
    torch::Tensor edge_weight,
    bool build_gcn_norm) {
  return build_partition_topology_impl(
      local_edge_ids,
      time_ptr_2,
      src,
      dst,
      edge_ids,
      node_to_chunk,
      full_dst_ids,
      edge_weight,
      true,
      build_gcn_norm);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("stable_unique", &stable_unique, "Stable unique values with inverse (CPU)");
  m.def("stable_unique_with_ts", &stable_unique_with_ts, "Stable unique (value, ts) pairs with inverse and first ts (CPU)");
  m.def("deduplicate_csc_edges", &deduplicate_csc_edges, "Stable deduplicate CSC edges by (column, edge id) (CPU)");
  m.def("first_ts_for_lids", &first_ts_for_lids, "First timestamp per local id (CPU)");
  m.def("build_slice_topology", &build_slice_topology, "Build partition-data slice topology (CPU)");
  m.def("build_slice_topology_full", &build_slice_topology_full, "Build partition-data slice topology with full dst ids (CPU)");
  m.def("build_partition_topology", &build_partition_topology, "Build all partition-data slice topology (CPU)");
  m.def("build_partition_topology_full", &build_partition_topology_full, "Build all partition-data slice topology with full dst ids (CPU)");
  m.def("build_partition_topology_with_norm", &build_partition_topology_with_norm, "Build all partition-data slice topology with optional GCN norm (CPU)");
  m.def("build_partition_topology_full_with_norm", &build_partition_topology_full_with_norm, "Build all partition-data slice topology with full dst ids and optional GCN norm (CPU)");
}
