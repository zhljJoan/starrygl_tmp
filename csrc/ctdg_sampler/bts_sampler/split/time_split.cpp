// adaptive_split.h
#pragma once
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <numeric>

// ─────────────────────────────────────────────
// 内部辅助结构
// ─────────────────────────────────────────────
struct NodeInfo {
    std::vector<int64_t> edge_indices; // 该节点在当前窗口内的边索引
};

// ─────────────────────────────────────────────
// 陈旧度计算
// ─────────────────────────────────────────────
static double get_staleness(double tlast, double tnow, double tbatch, int dic_size,double alpha, double beta) {
    if (tnow - tlast == 0.0) return 0.0;
    double time_diff = tnow - tbatch;
    if (std::abs(time_diff) < 1e-6) time_diff = 1e-6;
    double exp_arg0 = std::clamp((tlast - tbatch) / time_diff, -100.0, 100.0);
    double dic_arg0 = std::min(1e-2 * dic_size, 100.0);
    double coeff = alpha + 2.0 * beta;
    return coeff * std::exp(exp_arg0) + coeff * std::exp(dic_arg0);
}

// ─────────────────────────────────────────────
// 聚合损失
// ─────────────────────────────────────────────
static double get_aggloss(int64_t src, int64_t dst,
                           const std::unordered_map<int64_t, NodeInfo>& dic,
                           double aggl) {
    double ans = 0.0;
    if (dic.count(src)) ans += aggl;
    if (dic.count(dst)) ans += aggl;
    return ans;
}

// ─────────────────────────────────────────────
// 计算陈旧度阈值
// ─────────────────────────────────────────────
static double get_staleness_constraint(
    const int64_t* src_ptr, const int64_t* dst_ptr, const double* ts_ptr,
    int64_t N, int64_t batch_size, double graph_feature,
    double alpha, double beta, double aggl)
{
    std::vector<int64_t> range_;
    for (int64_t i = 0; i < N; i += batch_size) range_.push_back(i);
    if (range_.back() != N) range_.push_back(N);

    // 初始化 time_dic
    std::unordered_map<int64_t, double> time_dic;
    for (int64_t i = 0; i < N; i++) {
        time_dic.emplace(src_ptr[i], ts_ptr[0]);
        time_dic.emplace(dst_ptr[i], ts_ptr[0]);
    }

    std::vector<double> staleness;
    for (int64_t bi = 0; bi + 1 < (int64_t)range_.size(); bi++) {
        int64_t seg_start = range_[bi], seg_end = range_[bi + 1];
        std::unordered_map<int64_t, NodeInfo> dic;
        std::unordered_map<int64_t, double> cache;
        double stale = 0.0;

        for (int64_t x = 0; x < seg_end - seg_start; x++) {
            int64_t idx = seg_start + x;
            int64_t s = src_ptr[idx], d = dst_ptr[idx];
            double t = ts_ptr[idx];
            cache[s] = t; cache[d] = t;

            stale += get_aggloss(s, d, dic, aggl);

            auto update_node = [&](int64_t node) {
                auto it = dic.find(node);
                if (it != dic.end()) {
                    int64_t last = it->second.edge_indices.back();
                    if (t - ts_ptr[seg_start + last] != 0.0) {
                        stale += get_staleness(ts_ptr[seg_start + last], t,
                                               time_dic[node],
                                               (int)it->second.edge_indices.size(),
                                               alpha, beta);
                        it->second.edge_indices.push_back(x);
                    }
                } else {
                    dic[node].edge_indices.push_back(x);
                }
            };

            if (!dic.count(s) && !dic.count(d)) {
                dic[s].edge_indices.push_back(x);
                dic[d].edge_indices.push_back(x);
            } else if (!dic.count(s)) {
                update_node(d);
                dic[s].edge_indices.push_back(x);
            } else if (!dic.count(d)) {
                update_node(s);
                dic[d].edge_indices.push_back(x);
            } else {
                if (s == d) {
                    update_node(s);
                } else {
                    double ts_s = ts_ptr[seg_start + dic[s].edge_indices.back()];
                    double ts_d = ts_ptr[seg_start + dic[d].edge_indices.back()];
                    bool s_stale = (t - ts_s != 0.0), d_stale = (t - ts_d != 0.0);
                    if (s_stale) {
                        stale += get_staleness(ts_s, t, time_dic[s],
                                               (int)dic[s].edge_indices.size(), alpha, beta);
                        dic[s].edge_indices.push_back(x);
                    }
                    if (d_stale) {
                        stale += get_staleness(ts_d, t, time_dic[d],
                                               (int)dic[d].edge_indices.size(), alpha, beta);
                        dic[d].edge_indices.push_back(x);
                    }
                }
            }
        }
        for (auto& [k, v] : cache) time_dic[k] = v;
        staleness.push_back(stale);
    }

    if (staleness.empty()) return 0.0;
    double mean = std::accumulate(staleness.begin(), staleness.end(), 0.0) / staleness.size();
    double maxv = *std::max_element(staleness.begin(), staleness.end());
    return mean + graph_feature * (maxv - mean);
}

// ─────────────────────────────────────────────
// 边丢弃预处理（可选）
// ─────────────────────────────────────────────
static std::vector<int64_t> preprocess_edge_dropping(
    const int64_t* src_ptr, const int64_t* dst_ptr, const double* ts_ptr,
    int64_t N, int64_t window_size, double alpha, double beta, double drop_rate = 0.8)
{
    std::unordered_map<int64_t, double> time_dic;
    for (int64_t i = 0; i < N; i++) {
        time_dic.emplace(src_ptr[i], ts_ptr[0]);
        time_dic.emplace(dst_ptr[i], ts_ptr[0]);
    }

    std::vector<double> abundant;
    abundant.reserve(N);

    std::vector<int64_t> range_;
    for (int64_t i = 0; i < N; i += window_size) range_.push_back(i);
    if (range_.back() != N) range_.push_back(N);

    for (int64_t bi = 0; bi + 1 < (int64_t)range_.size(); bi++) {
        int64_t seg_start = range_[bi], seg_end = range_[bi + 1];
        std::unordered_map<int64_t, NodeInfo> dic;
        std::unordered_map<int64_t, double> cache;

        for (int64_t x = 0; x < seg_end - seg_start; x++) {
            int64_t idx = seg_start + x;
            int64_t s = src_ptr[idx], d = dst_ptr[idx];
            double t = ts_ptr[idx];
            cache[s] = t; cache[d] = t;

            double stale = 0.0;

            // 复用同样的四分支逻辑
            auto try_append = [&](int64_t node) {
                auto it = dic.find(node);
                if (it != dic.end()) {
                    int64_t last = it->second.edge_indices.back();
                    double ts_last = ts_ptr[seg_start + last];
                    if (t - ts_last != 0.0) {
                        stale += get_staleness(ts_last, t, time_dic[node],
                                               (int)it->second.edge_indices.size(), alpha, beta);
                        it->second.edge_indices.push_back(x);
                    }
                } else {
                    dic[node].edge_indices.push_back(x);
                }
            };

            if (!dic.count(s) && !dic.count(d)) {
                dic[s].edge_indices.push_back(x);
                dic[d].edge_indices.push_back(x);
            } else if (!dic.count(s)) {
                try_append(d);
                dic[s].edge_indices.push_back(x);
            } else if (!dic.count(d)) {
                try_append(s);
                dic[d].edge_indices.push_back(x);
            } else {
                if (s == d) {
                    try_append(s);
                } else {
                    double ts_s = ts_ptr[seg_start + dic[s].edge_indices.back()];
                    double ts_d = ts_ptr[seg_start + dic[d].edge_indices.back()];
                    if (t - ts_s != 0.0) {
                        stale += get_staleness(ts_s, t, time_dic[s],
                                               (int)dic[s].edge_indices.size(), alpha, beta);
                        dic[s].edge_indices.push_back(x);
                    }
                    if (t - ts_d != 0.0) {
                        stale += get_staleness(ts_d, t, time_dic[d],
                                               (int)dic[d].edge_indices.size(), alpha, beta);
                        dic[d].edge_indices.push_back(x);
                    }
                }
            }

            // abundant = 0.5*abund - 0.5*stale，这里简化为 -stale（abund计算略重，可按需补充）
            abundant.push_back(-stale);
        }
        for (auto& [k, v] : cache) time_dic[k] = v;
    }

    // 按百分位数过滤：保留 abundant <= threshold 的边（即 stale 较小的边）
    std::vector<double> sorted_ab = abundant;
    std::sort(sorted_ab.begin(), sorted_ab.end());
    int64_t keep_count = std::max((int64_t)1, (int64_t)((1.0 - drop_rate) * N));
    double threshold = sorted_ab[keep_count - 1];

    std::vector<int64_t> keep_indices;
    keep_indices.reserve(keep_count);
    for (int64_t i = 0; i < N; i++) {
        if (abundant[i] <= threshold)
            keep_indices.push_back(i);
    }
    return keep_indices;
}

// ─────────────────────────────────────────────
// 自适应切分
// ─────────────────────────────────────────────
static std::vector<int64_t> adaptive_split_impl(
    const int64_t* src_ptr, const int64_t* dst_ptr, const double* ts_ptr,
    int64_t N, double threshold, double alpha, double beta, double aggl)
{
    std::vector<int64_t> group_index(N, 0);
    std::unordered_map<int64_t, double> time_dic;
    for (int64_t i = 0; i < N; i++) {
        time_dic.emplace(src_ptr[i], ts_ptr[0]);
        time_dic.emplace(dst_ptr[i], ts_ptr[0]);
    }

    std::vector<int64_t> start_list;
    int64_t start = 0, end = N;

    while (start < end - 1) {
        std::unordered_map<int64_t, NodeInfo> dic;
        std::unordered_map<int64_t, double> cache;
        double stale = 0.0;

        for (int64_t i = start; i < end; i++) {
            int64_t s = src_ptr[i], d = dst_ptr[i];
            double t = ts_ptr[i];
            cache[s] = t; cache[d] = t;

            stale += get_aggloss(s, d, dic, aggl);

            auto try_append = [&](int64_t node) {
                auto it = dic.find(node);
                if (it != dic.end()) {
                    int64_t last_local = it->second.edge_indices.back();
                    double ts_last = ts_ptr[last_local];
                    if (t - ts_last != 0.0) {
                        stale += get_staleness(ts_last, t, time_dic[node],
                                               (int)it->second.edge_indices.size(), alpha, beta);
                        it->second.edge_indices.push_back(i);
                    }
                } else {
                    dic[node].edge_indices.push_back(i);
                }
            };

            if (!dic.count(s) && !dic.count(d)) {
                dic[s].edge_indices.push_back(i);
                dic[d].edge_indices.push_back(i);
            } else if (!dic.count(s)) {
                try_append(d);
                dic[s].edge_indices.push_back(i);
            } else if (!dic.count(d)) {
                try_append(s);
                dic[d].edge_indices.push_back(i);
            } else {
                if (s == d) {
                    try_append(s);
                } else {
                    double ts_s = ts_ptr[dic[s].edge_indices.back()];
                    double ts_d = ts_ptr[dic[d].edge_indices.back()];
                    if (t - ts_s != 0.0) {
                        stale += get_staleness(ts_s, t, time_dic[s],
                                               (int)dic[s].edge_indices.size(), alpha, beta);
                        dic[s].edge_indices.push_back(i);
                    }
                    if (t - ts_d != 0.0) {
                        stale += get_staleness(ts_d, t, time_dic[d],
                                               (int)dic[d].edge_indices.size(), alpha, beta);
                        dic[d].edge_indices.push_back(i);
                    }
                }
            }

            if (stale > threshold || i == end - 1) {
                start_list.push_back(i);
                start = i;
                for (auto& [k, v] : cache) time_dic[k] = v;
                break;
            }
        }
    }

    // 构建 group_index
    start_list.insert(start_list.begin(), 0);
    start_list.back() = N;
    for (int64_t gi = 0; gi + 1 < (int64_t)start_list.size(); gi++) {
        for (int64_t j = start_list[gi]; j < start_list[gi + 1]; j++)
            group_index[j] = gi;
    }
    return group_index;
}

// ─────────────────────────────────────────────
// 公开接口：Tensor 输入输出
// ─────────────────────────────────────────────

struct SplitResult {
    torch::Tensor group_index;   // shape [N], int64, 每条边所属的 batch id
    torch::Tensor keep_indices;  // shape [K], int64, 保留的边索引（edge dropping 开启时）
    double drop_proportion;      // 丢弃比例
};

/**
 * @param src             [N] int64 Tensor
 * @param dst             [N] int64 Tensor
 * @param ts              [N] float64 Tensor
 * @param base_batch_size 用于估算阈值的基础 batch 大小
 * @param graph_feature   图结构特征（由调用方计算传入）
 * @param alphamemory_loss 系数
 * @param beta            message_loss 系数
 * @param aggl聚合损失系数（unique_edges / total_edges）
 * @param enable_drop     是否启用边丢弃
 * @param drop_rate       丢弃比例（enable_drop=true 时有效）
 * @param window_size     边丢弃的窗口大小
 */
SplitResult adaptive_split_tensor(
    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor ts,
    int64_t base_batch_size,
    double graph_feature,
    double alpha   = 1.0,
    double beta    = 0.5,
    double aggl    = 0.0,
    bool   enable_drop  = false,
    double drop_rate    = 0.8,
    int64_t window_size = 1000)
{
    TORCH_CHECK(src.dtype() == torch::kInt64, "src must be int64");
    TORCH_CHECK(dst.dtype() == torch::kInt64, "dst must be int64");
    TORCH_CHECK(ts.dtype()  == torch::kFloat64, "ts must be float64");
    TORCH_CHECK(src.dim() == 1 && dst.dim() == 1 && ts.dim() == 1, "inputs must be 1-D");
    TORCH_CHECK(src.size(0) == dst.size(0) && src.size(0) == ts.size(0), "size mismatch");

    // 确保连续
    src = src.contiguous(); dst = dst.contiguous(); ts = ts.contiguous();
    const int64_t* sp = src.data_ptr<int64_t>();
    const int64_t* dp = dst.data_ptr<int64_t>();
    const double*  tp = ts.data_ptr<double>();
    int64_t N = src.size(0);

    SplitResult result;
    result.drop_proportion = 0.0;

    // ── 可选：边丢弃 ──
    std::vector<int64_t> keep_idx;
    const int64_t* eff_sp = sp;
    const int64_t* eff_dp = dp;
    const double*  eff_tp = tp;
    int64_t eff_N = N;

    torch::Tensor src_filt, dst_filt, ts_filt;
    if (enable_drop) {
        keep_idx = preprocess_edge_dropping(sp, dp, tp, N, window_size, alpha, beta, drop_rate);
        int64_t K = (int64_t)keep_idx.size();
        result.keep_indices = torch::from_blob(keep_idx.data(), {K}, torch::kInt64).clone();
        result.drop_proportion = 1.0 - (double)K / N;

        // 构建过滤后的数据
        src_filt = src.index({result.keep_indices});
        dst_filt = dst.index({result.keep_indices});
        ts_filt  = ts.index({result.keep_indices});
        eff_sp = src_filt.data_ptr<int64_t>();
        eff_dp = dst_filt.data_ptr<int64_t>();
        eff_tp = ts_filt.data_ptr<double>();
        eff_N  = K;
    } else {
        result.keep_indices = torch::arange(N, torch::kInt64);
    }

    // ── 计算阈值 ──
    double threshold = get_staleness_constraint(
        eff_sp, eff_dp, eff_tp, eff_N, base_batch_size, graph_feature, alpha, beta, aggl);

    // ── 自适应切分 ──
    std::vector<int64_t> gi = adaptive_split_impl(
        eff_sp, eff_dp, eff_tp, eff_N, threshold, alpha, beta, aggl);

    result.group_index = torch::from_blob(gi.data(), {eff_N}, torch::kInt64).clone();
    return result;
}

namespace py = pybind11;

PYBIND11_MODULE(adaptive_split_cpp, m) {
    py::class_<SplitResult>(m, "SplitResult")
        .def_readwrite("group_index",    &SplitResult::group_index)
        .def_readwrite("keep_indices",   &SplitResult::keep_indices)
        .def_readwrite("drop_proportion",&SplitResult::drop_proportion);

    m.def("adaptive_split", &adaptive_split_tensor,py::arg("src"), py::arg("dst"), py::arg("ts"),
        py::arg("base_batch_size"),
        py::arg("graph_feature"),
        py::arg("alpha")       = 1.0,
        py::arg("beta")        = 0.5,
        py::arg("aggl")        = 0.0,
        py::arg("enable_drop") = false,
        py::arg("drop_rate")   = 0.8,
        py::arg("window_size") = 1000
    );
}
