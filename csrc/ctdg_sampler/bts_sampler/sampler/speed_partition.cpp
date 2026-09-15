#include <speed_partition.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kEps = 1e-8;

struct SpeedNode {
    int64_t id;
    int64_t degree;
    double importance;
};

struct SpeedState {
    int64_t num_nodes;
    int64_t num_edges;
    int64_t num_parts;
    int64_t edge_cap;
    std::vector<double> time_bal;
    std::vector<std::vector<int64_t>> node_parts;
    std::vector<std::vector<uint8_t>> is_in;
    std::vector<int64_t> degree;
    std::vector<int64_t> degree_in;
    std::vector<double> importance;
    std::vector<std::vector<int64_t>> edge_ids_by_part;
    std::vector<int64_t> dropped_edges;
    std::vector<int64_t> edge_load_by_part;
    std::vector<int64_t> node_load_by_part;
    std::vector<std::vector<double>> importance_by_part;
    std::vector<std::vector<double>> last_ts_by_part;

    SpeedState(int64_t n, int64_t e, int64_t p)
        : num_nodes(n),
          num_edges(e),
          num_parts(p),
          edge_cap(static_cast<int64_t>((1.0 + 0.1) * static_cast<double>(e) / std::max<int64_t>(p, 1))),
          time_bal(p, 0.0),
          node_parts(n),
          is_in(n, std::vector<uint8_t>(p, 0)),
          degree(n, 0),
          degree_in(n, 0),
          importance(n, 0.0),
          edge_ids_by_part(p),
          edge_load_by_part(p, 0),
          node_load_by_part(p, 0),
          importance_by_part(n, std::vector<double>(p, 0.0)),
          last_ts_by_part(n, std::vector<double>(p, 0.0)) {}

    void add_edge_to_part(int64_t eid, int64_t part, double ts) {
        edge_load_by_part[part] += 1;
        edge_ids_by_part[part].push_back(eid);
        time_bal[part] = std::max(time_bal[part], ts);
    }

    void add_node_to_part(int64_t nid, int64_t part, double weight, double ts, double max_time) {
        if (!is_in[nid][part]) {
            is_in[nid][part] = 1;
            node_parts[nid].push_back(part);
            node_load_by_part[part] += 1;
        }
        degree[nid] += 1;
        importance[nid] += weight;
        const double denom = std::max(max_time, 1e-9);
        importance_by_part[nid][part] *= std::exp((last_ts_by_part[nid][part] - ts) / denom);
        importance_by_part[nid][part] += weight;
        last_ts_by_part[nid][part] = ts;
    }

    int64_t max_edge_load() const {
        return *std::max_element(edge_load_by_part.begin(), edge_load_by_part.end());
    }

    int64_t min_edge_load() const {
        return *std::min_element(edge_load_by_part.begin(), edge_load_by_part.end());
    }

    int64_t max_node_load() const {
        return *std::max_element(node_load_by_part.begin(), node_load_by_part.end());
    }

    int64_t min_node_load() const {
        return *std::min_element(node_load_by_part.begin(), node_load_by_part.end());
    }

    double max_time_load() const {
        return *std::max_element(time_bal.begin(), time_bal.end());
    }

    double min_time_load() const {
        return *std::min_element(time_bal.begin(), time_bal.end());
    }
};

bool almost_equal(double a, double b) {
    return std::abs(a - b) < kEps;
}

torch::Tensor require_1d_cpu_contiguous(torch::Tensor t, const char* name, torch::ScalarType dtype) {
    TORCH_CHECK(t.dim() == 1, name, " must be 1-D");
    TORCH_CHECK(t.device().is_cpu(), name, " must be a CPU tensor");
    if (t.scalar_type() != dtype) {
        t = t.to(dtype);
    }
    return t.contiguous();
}

torch::Tensor require_2d_cpu_contiguous(torch::Tensor t, const char* name, torch::ScalarType dtype) {
    TORCH_CHECK(t.dim() == 2, name, " must be 2-D");
    TORCH_CHECK(t.device().is_cpu(), name, " must be a CPU tensor");
    if (t.scalar_type() != dtype) {
        t = t.to(dtype);
    }
    return t.contiguous();
}

int64_t tensor_num_nodes(const int64_t* src, const int64_t* dst, int64_t num_edges, int64_t requested) {
    int64_t n = requested;
    if (n > 0) {
        return n;
    }
    int64_t max_id = -1;
    for (int64_t i = 0; i < num_edges; ++i) {
        max_id = std::max(max_id, std::max(src[i], dst[i]));
    }
    return max_id + 1;
}

template <typename Ts>
std::vector<double> copy_ts(torch::Tensor ts) {
    const auto* ptr = ts.data_ptr<Ts>();
    std::vector<double> out(ts.numel());
    for (int64_t i = 0; i < ts.numel(); ++i) {
        out[i] = static_cast<double>(ptr[i]);
    }
    return out;
}

std::vector<double> copy_timestamps(torch::Tensor ts) {
    TORCH_CHECK(ts.dim() == 1, "ts must be 1-D");
    TORCH_CHECK(ts.device().is_cpu(), "ts must be a CPU tensor");
    ts = ts.contiguous();
    if (ts.scalar_type() == torch::kFloat32) {
        return copy_ts<float>(ts);
    }
    if (ts.scalar_type() == torch::kFloat64) {
        return copy_ts<double>(ts);
    }
    ts = ts.to(torch::kFloat64).contiguous();
    return copy_ts<double>(ts);
}

void perform_step(
    int64_t u,
    int64_t v,
    int64_t eid,
    double ts,
    const std::vector<uint8_t>& is_hot,
    SpeedState& state,
    const std::vector<std::vector<int64_t>>& neighbors,
    const std::vector<std::vector<double>>& neighbor_ts,
    double max_time,
    double delta_ts,
    int64_t epsilon,
    int64_t epsilon_t) {
    std::vector<int64_t> candidates;
    double best_score = -1.0;

    if (is_hot[u] && !is_hot[v]) {
        candidates.insert(candidates.end(), state.node_parts[v].begin(), state.node_parts[v].end());
    } else if (is_hot[v] && !is_hot[u]) {
        candidates.insert(candidates.end(), state.node_parts[u].begin(), state.node_parts[u].end());
    } else if (!is_hot[u] && !is_hot[v] && !state.node_parts[u].empty() && !state.node_parts[v].empty()) {
        for (int64_t p = 0; p < state.num_parts; ++p) {
            if (state.is_in[u][p] && state.is_in[v][p]) {
                candidates.push_back(p);
            }
        }
    }

    if (candidates.empty()) {
        const int64_t max_edge_load = state.max_edge_load();
        const int64_t min_edge_load = state.min_edge_load();
        const int64_t max_node_load = state.max_node_load();
        const int64_t min_node_load = state.min_node_load();
        const double max_time_load = state.max_time_load();
        const double min_time_load = state.min_time_load();
        const double denom_time = std::max(max_time, 1e-9);

        for (int64_t p = 0; p < state.num_parts; ++p) {
            double score =
                state.importance_by_part[u][p] * std::exp((state.last_ts_by_part[u][p] - ts) / denom_time) +
                state.importance_by_part[v][p] * std::exp((state.last_ts_by_part[v][p] - ts) / denom_time) +
                1.0;

            const double node_bal =
                static_cast<double>(state.node_load_by_part[p] - min_node_load) /
                static_cast<double>(epsilon + max_node_load - min_node_load);
            const double edge_bal =
                static_cast<double>(state.edge_load_by_part[p] - min_edge_load) /
                static_cast<double>(epsilon + max_edge_load - min_edge_load);
            const double time_bal = std::exp(
                (min_time_load - state.time_bal[p]) /
                (max_time_load - min_time_load + 1.0 - static_cast<double>(epsilon_t)));

            score = score * (1.0 - node_bal) * (1.0 - edge_bal) * time_bal;
            if (score > best_score + kEps) {
                best_score = score;
                candidates.clear();
                candidates.push_back(p);
            } else if (almost_equal(score, best_score)) {
                candidates.push_back(p);
            }
        }
    }

    state.degree_in[u] += 1;
    state.degree_in[v] += 1;
    if (candidates.empty()) {
        state.dropped_edges.push_back(eid);
        return;
    }

    const int64_t part = candidates[static_cast<size_t>(eid % candidates.size())];
    constexpr double weight = 1.0;
    if (is_hot[u] || state.node_parts[u].empty() || state.node_parts[u][0] == part) {
        state.add_node_to_part(u, part, weight, ts, max_time);
    }
    if (is_hot[v] || state.node_parts[v].empty() || state.node_parts[v][0] == part) {
        state.add_node_to_part(v, part, weight, ts, max_time);
    }
    state.add_edge_to_part(eid, part, ts);
}

torch::Tensor vector_to_long_tensor(const std::vector<int64_t>& values) {
    auto out = torch::empty({static_cast<int64_t>(values.size())}, torch::dtype(torch::kInt64));
    if (!values.empty()) {
        std::copy(values.begin(), values.end(), out.data_ptr<int64_t>());
    }
    return out;
}

torch::Tensor node_parts_to_padded_tensor(
    const SpeedState& state,
    const std::vector<int64_t>& node_ids) {
    int64_t width = 0;
    for (const int64_t nid : node_ids) {
        width = std::max<int64_t>(width, static_cast<int64_t>(state.node_parts[static_cast<size_t>(nid)].size()));
    }

    auto out = torch::full(
        {static_cast<int64_t>(node_ids.size()), width},
        -1,
        torch::dtype(torch::kInt64));
    auto* ptr = out.data_ptr<int64_t>();
    for (int64_t row = 0; row < static_cast<int64_t>(node_ids.size()); ++row) {
        const auto& parts = state.node_parts[static_cast<size_t>(node_ids[static_cast<size_t>(row)])];
        for (int64_t col = 0; col < static_cast<int64_t>(parts.size()); ++col) {
            ptr[row * width + col] = parts[static_cast<size_t>(col)];
        }
    }
    return out;
}

double load_score_for_rank(
    const std::vector<double>& rank_load,
    const std::vector<double>& target,
    int64_t rank,
    int64_t num_slices) {
    double score = 0.0;
    const int64_t offset = rank * num_slices;
    for (int64_t t = 0; t < num_slices; ++t) {
        const double diff = rank_load[static_cast<size_t>(offset + t)] - target[static_cast<size_t>(t)];
        score += diff * diff;
    }
    return score / static_cast<double>(std::max<int64_t>(num_slices, 1));
}

double load_objective(
    const std::vector<double>& rank_load,
    const std::vector<double>& target,
    int64_t world_size,
    int64_t num_slices) {
    double score = 0.0;
    for (int64_t rank = 0; rank < world_size; ++rank) {
        score += load_score_for_rank(rank_load, target, rank, num_slices);
    }
    return score;
}

double average_batch_load_ratio(
    const std::vector<double>& rank_load,
    int64_t world_size,
    int64_t num_slices) {
    double total_ratio = 0.0;
    int64_t valid_slices = 0;
    for (int64_t t = 0; t < num_slices; ++t) {
        double min_load = std::numeric_limits<double>::infinity();
        double max_load = 0.0;
        for (int64_t rank = 0; rank < world_size; ++rank) {
            const double value = rank_load[static_cast<size_t>(rank * num_slices + t)];
            min_load = std::min(min_load, value);
            max_load = std::max(max_load, value);
        }
        if (max_load <= 0.0) {
            continue;
        }
        total_ratio += max_load / std::max(min_load, kEps);
        valid_slices += 1;
    }
    return total_ratio / static_cast<double>(std::max<int64_t>(valid_slices, 1));
}

double average_batch_load_ratio_after_swap(
    const std::vector<double>& rank_load,
    const double* load_ptr,
    int64_t world_size,
    int64_t num_slices,
    int64_t num_chunks,
    int64_t rank_a,
    int64_t rank_b,
    int64_t chunk_a,
    int64_t chunk_b) {
    double total_ratio = 0.0;
    int64_t valid_slices = 0;
    for (int64_t t = 0; t < num_slices; ++t) {
        double min_load = std::numeric_limits<double>::infinity();
        double max_load = 0.0;
        for (int64_t rank = 0; rank < world_size; ++rank) {
            double value = rank_load[static_cast<size_t>(rank * num_slices + t)];
            if (rank == rank_a) {
                value = value - load_ptr[t * num_chunks + chunk_a] + load_ptr[t * num_chunks + chunk_b];
            } else if (rank == rank_b) {
                value = value - load_ptr[t * num_chunks + chunk_b] + load_ptr[t * num_chunks + chunk_a];
            }
            min_load = std::min(min_load, value);
            max_load = std::max(max_load, value);
        }
        if (max_load <= 0.0) {
            continue;
        }
        total_ratio += max_load / std::max(min_load, kEps);
        valid_slices += 1;
    }
    return total_ratio / static_cast<double>(std::max<int64_t>(valid_slices, 1));
}

int64_t first_local_part_or_fallback(
    const SpeedState& state,
    int64_t nid,
    int64_t fallback) {
    if (!state.node_parts[static_cast<size_t>(nid)].empty()) {
        return state.node_parts[static_cast<size_t>(nid)][0];
    }
    return fallback;
}

torch::Tensor choose_balanced_node_master(
    const SpeedState& state,
    const int64_t* src_ptr,
    const int64_t* dst_ptr,
    const int64_t* edge_owner_ptr,
    const std::vector<uint8_t>& is_hot,
    const std::vector<int64_t>& degree) {
    const int64_t n = state.num_nodes;
    const int64_t num_parts = state.num_parts;
    auto node_master = torch::full({n}, -1, torch::dtype(torch::kInt64));
    auto* node_master_ptr = node_master.data_ptr<int64_t>();

    std::vector<double> master_load(static_cast<size_t>(num_parts), 0.0);
    std::vector<std::vector<double>> affinity(
        static_cast<size_t>(n),
        std::vector<double>(static_cast<size_t>(num_parts), 0.0));

    for (int64_t eid = 0; eid < state.num_edges; ++eid) {
        const int64_t part = edge_owner_ptr[eid];
        if (part < 0 || part >= num_parts) {
            continue;
        }
        affinity[static_cast<size_t>(src_ptr[eid])][static_cast<size_t>(part)] += 1.0;
        affinity[static_cast<size_t>(dst_ptr[eid])][static_cast<size_t>(part)] += 1.0;
    }

    std::vector<int64_t> replica_nodes;
    replica_nodes.reserve(static_cast<size_t>(n));
    for (int64_t nid = 0; nid < n; ++nid) {
        const bool replica = is_hot[static_cast<size_t>(nid)] || state.node_parts[static_cast<size_t>(nid)].size() > 1;
        if (replica) {
            replica_nodes.push_back(nid);
            continue;
        }
        const int64_t fallback = nid % std::max<int64_t>(num_parts, 1);
        const int64_t master = first_local_part_or_fallback(state, nid, fallback);
        node_master_ptr[nid] = master;
        master_load[static_cast<size_t>(master)] += static_cast<double>(std::max<int64_t>(degree[static_cast<size_t>(nid)], 1));
    }

    std::sort(replica_nodes.begin(), replica_nodes.end(), [&](int64_t a, int64_t b) {
        if (degree[static_cast<size_t>(a)] != degree[static_cast<size_t>(b)]) {
            return degree[static_cast<size_t>(a)] > degree[static_cast<size_t>(b)];
        }
        return a < b;
    });

    double total_weight = 0.0;
    for (int64_t nid = 0; nid < n; ++nid) {
        total_weight += static_cast<double>(std::max<int64_t>(degree[static_cast<size_t>(nid)], 1));
    }
    const double target_load = std::max(total_weight / static_cast<double>(std::max<int64_t>(num_parts, 1)), 1.0);

    for (int64_t nid : replica_nodes) {
        const double node_weight = static_cast<double>(std::max<int64_t>(degree[static_cast<size_t>(nid)], 1));
        double affinity_sum = 0.0;
        for (int64_t part = 0; part < num_parts; ++part) {
            affinity_sum += affinity[static_cast<size_t>(nid)][static_cast<size_t>(part)];
        }
        const double load_penalty = std::max(1.0, affinity_sum);
        int64_t best_part = 0;
        double best_score = -std::numeric_limits<double>::infinity();
        for (int64_t part = 0; part < num_parts; ++part) {
            const bool candidate = is_hot[static_cast<size_t>(nid)] || state.is_in[static_cast<size_t>(nid)][static_cast<size_t>(part)];
            if (!candidate) {
                continue;
            }
            const double score =
                affinity[static_cast<size_t>(nid)][static_cast<size_t>(part)] -
                load_penalty * (master_load[static_cast<size_t>(part)] / target_load);
            if (score > best_score + kEps || (almost_equal(score, best_score) && part < best_part)) {
                best_score = score;
                best_part = part;
            }
        }
        node_master_ptr[nid] = best_part;
        master_load[static_cast<size_t>(best_part)] += node_weight;
    }

    return node_master;
}

}  // namespace

torch::Tensor assign_chunks_temporal_balance(
    torch::Tensor chunk_load,
    torch::Tensor affinity,
    int64_t world_size,
    int64_t chunks_per_rank,
    double affinity_weight,
    int64_t local_search_iters) {
    TORCH_CHECK(world_size > 0, "world_size must be positive");
    TORCH_CHECK(chunks_per_rank > 0, "chunks_per_rank must be positive");
    chunk_load = require_2d_cpu_contiguous(chunk_load, "chunk_load", torch::kFloat64);
    affinity = require_2d_cpu_contiguous(affinity, "affinity", torch::kFloat64);
    const int64_t num_slices = chunk_load.size(0);
    const int64_t num_chunks = chunk_load.size(1);
    TORCH_CHECK(num_chunks == world_size * chunks_per_rank,
                "num_chunks must equal world_size * chunks_per_rank");
    TORCH_CHECK(affinity.size(0) == num_chunks && affinity.size(1) == num_chunks,
                "affinity shape must be [num_chunks, num_chunks]");

    const auto* load_ptr = chunk_load.data_ptr<double>();
    const auto* affinity_ptr = affinity.data_ptr<double>();
    std::vector<double> target(static_cast<size_t>(num_slices), 0.0);
    std::vector<double> scalar(static_cast<size_t>(num_chunks), 0.0);
    for (int64_t t = 0; t < num_slices; ++t) {
        double total = 0.0;
        for (int64_t c = 0; c < num_chunks; ++c) {
            const double v = load_ptr[t * num_chunks + c];
            total += v;
            scalar[static_cast<size_t>(c)] += v * v;
        }
        target[static_cast<size_t>(t)] = total / static_cast<double>(world_size);
    }

    std::vector<int64_t> order(static_cast<size_t>(num_chunks));
    for (int64_t c = 0; c < num_chunks; ++c) {
        order[static_cast<size_t>(c)] = c;
        scalar[static_cast<size_t>(c)] = std::sqrt(scalar[static_cast<size_t>(c)]);
    }
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        if (!almost_equal(scalar[static_cast<size_t>(a)], scalar[static_cast<size_t>(b)])) {
            return scalar[static_cast<size_t>(a)] > scalar[static_cast<size_t>(b)];
        }
        return a < b;
    });

    std::vector<int64_t> owner(static_cast<size_t>(num_chunks), -1);
    std::vector<int64_t> rank_count(static_cast<size_t>(world_size), 0);
    std::vector<double> rank_load(static_cast<size_t>(world_size * num_slices), 0.0);
    std::vector<int64_t> assigned;
    assigned.reserve(static_cast<size_t>(num_chunks));

    for (int64_t cid : order) {
        int64_t best_rank = -1;
        double best_score = std::numeric_limits<double>::infinity();
        for (int64_t rank = 0; rank < world_size; ++rank) {
            if (rank_count[static_cast<size_t>(rank)] >= chunks_per_rank) {
                continue;
            }
            const double before_load = load_score_for_rank(rank_load, target, rank, num_slices);
            double after_load = 0.0;
            for (int64_t t = 0; t < num_slices; ++t) {
                const double next_load =
                    rank_load[static_cast<size_t>(rank * num_slices + t)] +
                    load_ptr[t * num_chunks + cid];
                const double diff = next_load - target[static_cast<size_t>(t)];
                after_load += diff * diff;
            }
            after_load /= static_cast<double>(std::max<int64_t>(num_slices, 1));
            double locality_score = 0.0;
            for (int64_t other : assigned) {
                if (owner[static_cast<size_t>(other)] != rank) {
                    locality_score += affinity_ptr[cid * num_chunks + other];
                }
            }
            const double score = (after_load - before_load) + affinity_weight * locality_score;
            if (score < best_score - kEps ||
                (almost_equal(score, best_score) && (best_rank < 0 || rank < best_rank))) {
                best_score = score;
                best_rank = rank;
            }
        }
        TORCH_CHECK(best_rank >= 0, "failed to assign chunk under capacity constraint");
        owner[static_cast<size_t>(cid)] = best_rank;
        rank_count[static_cast<size_t>(best_rank)] += 1;
        for (int64_t t = 0; t < num_slices; ++t) {
            rank_load[static_cast<size_t>(best_rank * num_slices + t)] += load_ptr[t * num_chunks + cid];
        }
        assigned.push_back(cid);
    }

    double cut_score = 0.0;
    for (int64_t i = 0; i < num_chunks; ++i) {
        for (int64_t j = i + 1; j < num_chunks; ++j) {
            if (owner[static_cast<size_t>(i)] != owner[static_cast<size_t>(j)]) {
                cut_score += affinity_ptr[i * num_chunks + j];
            }
        }
    }
    double load_score = load_objective(rank_load, target, world_size, num_slices);
    double best_score = load_score + affinity_weight * cut_score;

    std::vector<std::vector<int64_t>> chunks_by_rank(static_cast<size_t>(world_size));
    for (int64_t c = 0; c < num_chunks; ++c) {
        chunks_by_rank[static_cast<size_t>(owner[static_cast<size_t>(c)])].push_back(c);
    }

    if (local_search_iters > 0 && num_chunks <= 256) {
        double best_ratio = average_batch_load_ratio(rank_load, world_size, num_slices);
        const int64_t ratio_passes = std::min<int64_t>(std::max<int64_t>(local_search_iters, 0), 64);
        for (int64_t pass = 0; pass < ratio_passes; ++pass) {
            int64_t best_a = -1;
            int64_t best_b = -1;
            int64_t best_rank_a = -1;
            int64_t best_rank_b = -1;
            double next_ratio = best_ratio;
            for (int64_t a = 0; a < num_chunks; ++a) {
                const int64_t rank_a = owner[static_cast<size_t>(a)];
                for (int64_t b = a + 1; b < num_chunks; ++b) {
                    const int64_t rank_b = owner[static_cast<size_t>(b)];
                    if (rank_a == rank_b) {
                        continue;
                    }
                    const double ratio = average_batch_load_ratio_after_swap(
                        rank_load, load_ptr, world_size, num_slices, num_chunks, rank_a, rank_b, a, b);
                    if (ratio < next_ratio - 1e-9 ||
                        (almost_equal(ratio, next_ratio) && (best_a < 0 || a < best_a || (a == best_a && b < best_b)))) {
                        next_ratio = ratio;
                        best_a = a;
                        best_b = b;
                        best_rank_a = rank_a;
                        best_rank_b = rank_b;
                    }
                }
            }
            if (best_a < 0 || best_b < 0) {
                break;
            }
            for (int64_t t = 0; t < num_slices; ++t) {
                rank_load[static_cast<size_t>(best_rank_a * num_slices + t)] =
                    rank_load[static_cast<size_t>(best_rank_a * num_slices + t)] -
                    load_ptr[t * num_chunks + best_a] +
                    load_ptr[t * num_chunks + best_b];
                rank_load[static_cast<size_t>(best_rank_b * num_slices + t)] =
                    rank_load[static_cast<size_t>(best_rank_b * num_slices + t)] -
                    load_ptr[t * num_chunks + best_b] +
                    load_ptr[t * num_chunks + best_a];
            }
            owner[static_cast<size_t>(best_a)] = best_rank_b;
            owner[static_cast<size_t>(best_b)] = best_rank_a;
            auto& list_a = chunks_by_rank[static_cast<size_t>(best_rank_a)];
            auto& list_b = chunks_by_rank[static_cast<size_t>(best_rank_b)];
            auto it_a = std::find(list_a.begin(), list_a.end(), best_a);
            auto it_b = std::find(list_b.begin(), list_b.end(), best_b);
            if (it_a != list_a.end()) {
                *it_a = best_b;
            }
            if (it_b != list_b.end()) {
                *it_b = best_a;
            }
            best_ratio = next_ratio;
        }
    }

    cut_score = 0.0;
    for (int64_t i = 0; i < num_chunks; ++i) {
        for (int64_t j = i + 1; j < num_chunks; ++j) {
            if (owner[static_cast<size_t>(i)] != owner[static_cast<size_t>(j)]) {
                cut_score += affinity_ptr[i * num_chunks + j];
            }
        }
    }
    load_score = load_objective(rank_load, target, world_size, num_slices);
    best_score = load_score + affinity_weight * cut_score;

    int64_t stale = 0;
    const int64_t max_stale = std::max<int64_t>(100, std::max<int64_t>(local_search_iters, 1) / 10);
    const int64_t mse_search_iters = (num_chunks <= 256) ? 0 : local_search_iters;
    for (int64_t step = 0; step < mse_search_iters; ++step) {
        const int64_t rank_a = step % world_size;
        const int64_t rank_b = (step * 7 + 1) % world_size;
        if (rank_a == rank_b ||
            chunks_by_rank[static_cast<size_t>(rank_a)].empty() ||
            chunks_by_rank[static_cast<size_t>(rank_b)].empty()) {
            continue;
        }
        auto& list_a = chunks_by_rank[static_cast<size_t>(rank_a)];
        auto& list_b = chunks_by_rank[static_cast<size_t>(rank_b)];
        const int64_t pos_a = (step * 13) % static_cast<int64_t>(list_a.size());
        const int64_t pos_b = (step * 17) % static_cast<int64_t>(list_b.size());
        const int64_t a = list_a[static_cast<size_t>(pos_a)];
        const int64_t b = list_b[static_cast<size_t>(pos_b)];

        const double before_load =
            load_score_for_rank(rank_load, target, rank_a, num_slices) +
            load_score_for_rank(rank_load, target, rank_b, num_slices);
        std::vector<double> trial_a(static_cast<size_t>(num_slices));
        std::vector<double> trial_b(static_cast<size_t>(num_slices));
        for (int64_t t = 0; t < num_slices; ++t) {
            trial_a[static_cast<size_t>(t)] =
                rank_load[static_cast<size_t>(rank_a * num_slices + t)] -
                load_ptr[t * num_chunks + a] +
                load_ptr[t * num_chunks + b];
            trial_b[static_cast<size_t>(t)] =
                rank_load[static_cast<size_t>(rank_b * num_slices + t)] -
                load_ptr[t * num_chunks + b] +
                load_ptr[t * num_chunks + a];
        }
        double after_load = 0.0;
        for (int64_t t = 0; t < num_slices; ++t) {
            const double da = trial_a[static_cast<size_t>(t)] - target[static_cast<size_t>(t)];
            const double db = trial_b[static_cast<size_t>(t)] - target[static_cast<size_t>(t)];
            after_load += da * da + db * db;
        }
        after_load /= static_cast<double>(std::max<int64_t>(num_slices, 1));

        double cut_delta = 0.0;
        for (int64_t c = 0; c < num_chunks; ++c) {
            if (c == a || c == b) {
                continue;
            }
            const int64_t rc = owner[static_cast<size_t>(c)];
            const double wa = affinity_ptr[std::min(a, c) * num_chunks + std::max(a, c)];
            const double wb = affinity_ptr[std::min(b, c) * num_chunks + std::max(b, c)];
            const bool before_a_cut = rank_a != rc;
            const bool after_a_cut = rank_b != rc;
            const bool before_b_cut = rank_b != rc;
            const bool after_b_cut = rank_a != rc;
            cut_delta += (after_a_cut ? wa : 0.0) - (before_a_cut ? wa : 0.0);
            cut_delta += (after_b_cut ? wb : 0.0) - (before_b_cut ? wb : 0.0);
        }
        const double next_score = best_score + (after_load - before_load) + affinity_weight * cut_delta;
        if (next_score < best_score - 1e-6) {
            for (int64_t t = 0; t < num_slices; ++t) {
                rank_load[static_cast<size_t>(rank_a * num_slices + t)] = trial_a[static_cast<size_t>(t)];
                rank_load[static_cast<size_t>(rank_b * num_slices + t)] = trial_b[static_cast<size_t>(t)];
            }
            owner[static_cast<size_t>(a)] = rank_b;
            owner[static_cast<size_t>(b)] = rank_a;
            list_a[static_cast<size_t>(pos_a)] = b;
            list_b[static_cast<size_t>(pos_b)] = a;
            best_score = next_score;
            stale = 0;
        } else {
            stale += 1;
            if (stale >= max_stale) {
                break;
            }
        }
    }

    return vector_to_long_tensor(owner);
}

py::dict speed_partition(
    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor ts,
    int64_t num_nodes,
    int64_t num_parts,
    double beta,
    double topk_ratio,
    const std::string& topk_type) {
    TORCH_CHECK(num_parts > 0, "num_parts must be positive");
    src = require_1d_cpu_contiguous(src, "src", torch::kInt64);
    dst = require_1d_cpu_contiguous(dst, "dst", torch::kInt64);
    TORCH_CHECK(src.numel() == dst.numel(), "src and dst must have the same length");
    std::vector<double> ts_values = copy_timestamps(ts);
    TORCH_CHECK(static_cast<int64_t>(ts_values.size()) == src.numel(), "ts must match src/dst length");

    const auto* src_ptr = src.data_ptr<int64_t>();
    const auto* dst_ptr = dst.data_ptr<int64_t>();
    const int64_t num_edges = src.numel();
    const int64_t n = tensor_num_nodes(src_ptr, dst_ptr, num_edges, num_nodes);
    TORCH_CHECK(n >= 0, "num_nodes must be non-negative");

    double max_ts = 0.0;
    for (double t : ts_values) {
        max_ts = std::max(max_ts, t);
    }
    const double raw_max_time = std::max(max_ts, 1e-9);

    std::vector<int64_t> degree(static_cast<size_t>(n), 0);
    std::vector<double> importance(static_cast<size_t>(n), 0.0);
    std::vector<std::vector<int64_t>> neighbors(static_cast<size_t>(n));
    std::vector<std::vector<double>> neighbor_ts(static_cast<size_t>(n));
    double total_delta_t = 0.0;

    for (int64_t i = 0; i < num_edges; ++i) {
        const int64_t u = src_ptr[i];
        const int64_t v = dst_ptr[i];
        TORCH_CHECK(u >= 0 && u < n && v >= 0 && v < n, "src/dst contains node id outside [0, num_nodes)");
        const double t = ts_values[static_cast<size_t>(i)];
        total_delta_t += i > 0 ? t - ts_values[static_cast<size_t>(i - 1)] : 0.0;
        neighbors[u].push_back(v);
        neighbors[v].push_back(u);
        neighbor_ts[u].push_back(t);
        neighbor_ts[v].push_back(t);
        degree[u] += 1;
        degree[v] += 1;
        importance[u] += beta * (t / raw_max_time);
        importance[v] += beta * (t / raw_max_time);
    }

    std::vector<SpeedNode> nodes;
    nodes.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        nodes.push_back(SpeedNode{i, degree[i], importance[i]});
    }

    const int64_t topk = std::clamp<int64_t>(
        static_cast<int64_t>(std::floor(topk_ratio * static_cast<double>(n))),
        0,
        n);
    const bool use_importance = topk_type == "importance";
    std::sort(nodes.begin(), nodes.end(), [&](const SpeedNode& a, const SpeedNode& b) {
        if (use_importance && !almost_equal(a.importance, b.importance)) {
            return a.importance > b.importance;
        }
        if (a.degree != b.degree) {
            return a.degree > b.degree;
        }
        return a.id < b.id;
    });

    std::vector<uint8_t> is_hot(static_cast<size_t>(n), 0);
    std::vector<int64_t> hot_node_ids;
    hot_node_ids.reserve(static_cast<size_t>(topk));
    for (int64_t i = 0; i < topk; ++i) {
        is_hot[nodes[static_cast<size_t>(i)].id] = 1;
        hot_node_ids.push_back(nodes[static_cast<size_t>(i)].id);
    }

    const double avg_delta_t = num_edges > 1 ? total_delta_t / static_cast<double>(num_edges - 1) : 0.0;
    const int64_t epsilon_t = 0;
    const int64_t epsilon = 1;
    const double delta_ts = avg_delta_t * static_cast<double>(epsilon);
    const double partition_max_time = raw_max_time / 10.0;

    SpeedState state(n, num_edges, num_parts);
    for (int64_t i = 0; i < num_edges; ++i) {
        perform_step(
            src_ptr[i],
            dst_ptr[i],
            i,
            ts_values[static_cast<size_t>(i)],
            is_hot,
            state,
            neighbors,
            neighbor_ts,
            partition_max_time,
            delta_ts,
            epsilon,
            epsilon_t);
    }

    auto edge_owner = torch::full({num_edges}, -1, torch::dtype(torch::kInt64));
    auto edge_owner_ptr = edge_owner.data_ptr<int64_t>();
    std::vector<torch::Tensor> edge_ids_by_part;
    edge_ids_by_part.reserve(static_cast<size_t>(num_parts));
    for (int64_t p = 0; p < num_parts; ++p) {
        for (int64_t eid : state.edge_ids_by_part[p]) {
            edge_owner_ptr[eid] = p;
        }
        edge_ids_by_part.push_back(vector_to_long_tensor(state.edge_ids_by_part[p]));
    }
    for (int64_t eid = 0; eid < num_edges; ++eid) {
        if (edge_owner_ptr[eid] < 0) {
            edge_owner_ptr[eid] = first_local_part_or_fallback(state, dst_ptr[eid], dst_ptr[eid] % num_parts);
        }
    }

    auto node_master = choose_balanced_node_master(state, src_ptr, dst_ptr, edge_owner_ptr, is_hot, degree);
    auto node_master_ptr = node_master.data_ptr<int64_t>();
    auto replica_mask = torch::zeros({n}, torch::dtype(torch::kBool));
    auto replica_ptr = replica_mask.data_ptr<bool>();
    std::vector<torch::Tensor> local_node_ids_by_part;
    std::vector<torch::Tensor> owned_node_ids_by_part;
    std::vector<torch::Tensor> replica_node_ids_by_part;
    std::vector<torch::Tensor> shadow_node_ids_by_part;
    local_node_ids_by_part.reserve(static_cast<size_t>(num_parts));
    owned_node_ids_by_part.reserve(static_cast<size_t>(num_parts));
    replica_node_ids_by_part.reserve(static_cast<size_t>(num_parts));
    shadow_node_ids_by_part.reserve(static_cast<size_t>(num_parts));
    std::vector<int64_t> shared_node_ids;
    for (int64_t nid = 0; nid < n; ++nid) {
        if (is_hot[static_cast<size_t>(nid)] || state.node_parts[static_cast<size_t>(nid)].size() > 1) {
            replica_ptr[nid] = true;
            shared_node_ids.push_back(nid);
        }
    }
    for (int64_t p = 0; p < num_parts; ++p) {
        std::vector<int64_t> replica_nodes;
        std::vector<int64_t> owned_nodes;
        std::vector<int64_t> shadow_nodes;
        for (int64_t nid = 0; nid < n; ++nid) {
            const bool replica = replica_ptr[nid];
            const bool held = state.is_in[static_cast<size_t>(nid)][static_cast<size_t>(p)] || replica;
            if (!held) {
                continue;
            }
            if (replica) {
                replica_nodes.push_back(nid);
            } else if (node_master_ptr[nid] == p) {
                owned_nodes.push_back(nid);
            } else {
                shadow_nodes.push_back(nid);
            }
        }
        std::vector<int64_t> local_nodes;
        local_nodes.reserve(replica_nodes.size() + owned_nodes.size() + shadow_nodes.size());
        local_nodes.insert(local_nodes.end(), replica_nodes.begin(), replica_nodes.end());
        local_nodes.insert(local_nodes.end(), owned_nodes.begin(), owned_nodes.end());
        local_nodes.insert(local_nodes.end(), shadow_nodes.begin(), shadow_nodes.end());
        local_node_ids_by_part.push_back(vector_to_long_tensor(local_nodes));
        owned_node_ids_by_part.push_back(vector_to_long_tensor(owned_nodes));
        replica_node_ids_by_part.push_back(vector_to_long_tensor(replica_nodes));
        shadow_node_ids_by_part.push_back(vector_to_long_tensor(shadow_nodes));
    }

    py::dict out;
    out["node_master"] = node_master;
    out["node_parts"] = node_master;
    out["node_owner"] = node_master;
    out["edge_owner"] = edge_owner;
    out["edge_parts"] = edge_owner;
    out["replica_mask"] = replica_mask;
    out["shared_node_ids"] = vector_to_long_tensor(shared_node_ids);
    out["shared_node_parts"] = node_parts_to_padded_tensor(state, shared_node_ids);
    out["hot_node_ids"] = vector_to_long_tensor(hot_node_ids);
    out["local_node_ids_by_part"] = local_node_ids_by_part;
    out["owned_node_ids_by_part"] = owned_node_ids_by_part;
    out["replica_node_ids_by_part"] = replica_node_ids_by_part;
    out["shadow_node_ids_by_part"] = shadow_node_ids_by_part;
    out["edge_ids_by_part"] = edge_ids_by_part;
    out["dropped_edges"] = vector_to_long_tensor(state.dropped_edges);
    return out;
}
