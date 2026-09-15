#pragma once
#include <cstdint>
#include <head.h>
#include <neighbors.h>
# include <output.h>

struct NodeInstanceKey
{
    NodeIDType node;
    TimeStampType ts;

    bool operator==(const NodeInstanceKey& other) const
    {
        return node == other.node && ts == other.ts;
    }
};

struct NodeInstanceKeyHash
{
    size_t operator()(const NodeInstanceKey& key) const
    {
        size_t h1 = std::hash<NodeIDType>{}(key.node);
        size_t h2 = std::hash<TimeStampType>{}(key.ts);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

struct EdgeLidHash
{
    size_t operator()(const EdgeIDType& key) const
    {
        return std::hash<EdgeIDType>{}(key);
    }
};

class ParallelSampler
{
    public:
        TemporalNeighborBlock& tnb;
        NodeIDType num_nodes;
        EdgeIDType num_edges;
        int threads;
        vector<int> fanouts;
        // vector<NodeIDType> part_ptr;
        // int pid;
        vector<int>part;
        vector<int>node_part;
        vector<uint8_t>node_is_hot;
        int local_part = -1;
        int num_layers;
        string policy;
        std::vector<TemporalGraphBlock> ret;
        th::Tensor dist_nid;
        th::Tensor dist_eid;
        th::Tensor block_node_list;
        th::Tensor eid_inv;
        th::Tensor unq_id;
        th::Tensor first_block_id;
        double boundery_probility;
        double compact_total_seconds = 0.0;
        double compact_root_seconds = 0.0;
        double compact_index_seconds = 0.0;
        double compact_fill_seconds = 0.0;
        double compact_fill_cursor_seconds = 0.0;
        double compact_fill_edge_loop_seconds = 0.0;
        double compact_fill_raw_edge_rows = 0.0;
        double compact_fill_mapped_edge_rows = 0.0;
        double compact_index_edges = 0.0;
        double compact_index_unique_nodes = 0.0;
        double compact_index_frontier_nodes = 0.0;
        double compact_index_unique_edges = 0.0;
        double compact_index_last_node_hits = 0.0;
        double compact_index_hash_lookups = 0.0;
        double compact_index_hash_hits = 0.0;
        double compact_index_hash_inserts = 0.0;
        double edge_read_layout_seconds = 0.0;
        double sampler_local_nodes = 0.0;
        double sampler_remote_nodes = 0.0;
        double sampler_local_edges = 0.0;
        double sampler_remote_edges = 0.0;
        static constexpr int64_t kCompactNodePrefetchDistance = 8;
        static constexpr int64_t kCompactRawFillPrefetchDistance = 16;
        static constexpr int64_t kCompactMappedEdgePrefetchDistance = 16;
        uint64_t base_seed = 0;
        bool compact_raw_edge_ids = false;
        bool compact_node_ids = false;
        vector<unsigned int> loc_seeds;
        vector<int64_t> edge_lid_dense;
        vector<EdgeIDType> edge_lid_touched;
        vector<int64_t> frontier_pos_by_lid_dense;
        vector<int64_t> frontier_pos_by_node_dense;
        vector<int64_t> compact_frontier_lids_scratch;
        vector<int64_t> compact_next_frontier_lids_scratch;
        vector<int64_t> compact_occurrence_to_frontier_scratch;
        vector<int64_t> compact_frontier_pos_touched_scratch;
        vector<int64_t> compact_next_occurrence_to_frontier_scratch;
        vector<int64_t> compact_cursor_scratch;
        phmap::flat_hash_map<NodeInstanceKey, int64_t, NodeInstanceKeyHash> compact_local_node_index_scratch;
        phmap::flat_hash_map<NodeIDType, int64_t> compact_local_node_id_index_scratch;
        phmap::flat_hash_set<EdgeIDType, EdgeLidHash> compact_seen_edge_lids_scratch;
        vector<int64_t> edge_read_dist_index;
        vector<int64_t> edge_read_index_scratch;
        vector<int64_t> edge_read_order_scratch;
        int64_t edge_read_world_size = 0;
        ParallelSampler(TemporalNeighborBlock& _tnb, NodeIDType _num_nodes, EdgeIDType _num_edges, int _threads, 
                        vector<int>& _fanouts, int _num_layers, string _policy, int _local_part, th::Tensor _part, th::Tensor _node_part, th::Tensor _node_is_hot,double _p) :
                        tnb(_tnb), num_nodes(_num_nodes), num_edges(_num_edges), threads(_threads), 
                        fanouts(_fanouts), num_layers(_num_layers), policy(_policy), local_part(_local_part), boundery_probility(_p)
        {
            omp_set_num_threads(_threads);
            ret.clear();
            ret.resize(_num_layers);
            if(local_part != -1){
                int *part_ptr = _part.data_ptr<int>();
                part.assign(part_ptr, part_ptr + _part.numel());
                part_ptr = _node_part.data_ptr<int>();
                node_part.assign(part_ptr, part_ptr + _node_part.numel());
                if(_node_is_hot.numel() > 0){
                    uint8_t *hot_ptr = _node_is_hot.data_ptr<uint8_t>();
                    node_is_hot.assign(hot_ptr, hot_ptr + _node_is_hot.numel());
                }
            }
            for(int i = 0; i < threads; i++){
                loc_seeds.push_back(i);
            }
            if(num_edges > 0){
                edge_lid_dense.assign(static_cast<size_t>(num_edges), -1);
            }
        }

        bool is_boundary_candidate(EdgeIDType edge_lid, NodeIDType neighbor) const
        {
            if(local_part < 0 || part.empty()){
                return false;
            }
            if(edge_lid < 0 || edge_lid >= static_cast<EdgeIDType>(part.size())){
                return true;
            }
            if(part[static_cast<size_t>(edge_lid)] != local_part){
                return true;
            }
            if(node_part.empty() || neighbor < 0 || neighbor >= static_cast<NodeIDType>(node_part.size())){
                return false;
            }
            if(!node_is_hot.empty() &&
               neighbor < static_cast<NodeIDType>(node_is_hot.size()) &&
               node_is_hot[static_cast<size_t>(neighbor)]){
                return false;
            }
            return node_part[static_cast<size_t>(neighbor)] != local_part;
        }

        void reset()
        {
            ret.clear();
            ret.resize(num_layers);
        }

        void reset_profile_stats()
        {
            compact_total_seconds = 0.0;
            compact_root_seconds = 0.0;
            compact_index_seconds = 0.0;
            compact_fill_seconds = 0.0;
            compact_fill_cursor_seconds = 0.0;
            compact_fill_edge_loop_seconds = 0.0;
            compact_fill_raw_edge_rows = 0.0;
            compact_fill_mapped_edge_rows = 0.0;
            compact_index_edges = 0.0;
            compact_index_unique_nodes = 0.0;
            compact_index_frontier_nodes = 0.0;
            compact_index_unique_edges = 0.0;
            compact_index_last_node_hits = 0.0;
            compact_index_hash_lookups = 0.0;
            compact_index_hash_hits = 0.0;
            compact_index_hash_inserts = 0.0;
            edge_read_layout_seconds = 0.0;
            sampler_local_nodes = 0.0;
            sampler_remote_nodes = 0.0;
            sampler_local_edges = 0.0;
            sampler_remote_edges = 0.0;
        }

        void set_seed(uint64_t seed)
        {
            base_seed = seed;
            loc_seeds.clear();
            loc_seeds.reserve(static_cast<size_t>(threads));
            for(int i = 0; i < threads; i++){
                uint64_t value = seed + static_cast<uint64_t>(i + 1) * 0x9E3779B97F4A7C15ULL;
                loc_seeds.push_back(static_cast<unsigned int>(value ^ (value >> 32)));
            }
        }

        void set_compact_raw_edge_ids(bool enabled)
        {
            compact_raw_edge_ids = enabled;
        }

        void set_compact_node_ids(bool enabled)
        {
            compact_node_ids = enabled;
        }

        void deduplicate_compact_mfg_edges(NativeMFGBlock& mfg, bool has_delta_ts)
        {
            if(!compact_node_ids || mfg.edge_lids.size() <= 1){
                return;
            }
            const int64_t num_dst = static_cast<int64_t>(mfg.csc_indptr.size()) - 1;
            vector<int64_t> new_indptr(static_cast<size_t>(num_dst + 1), 0);
            vector<NodeIDType> new_indices;
            vector<EdgeIDType> new_edge_lids;
            vector<TimeStampType> new_delta_t;
            new_indices.reserve(mfg.csc_indices.size());
            new_edge_lids.reserve(mfg.edge_lids.size());
            if(has_delta_ts){
                new_delta_t.reserve(mfg.delta_t.size());
            }
            auto& seen = compact_seen_edge_lids_scratch;
            for(int64_t dst = 0; dst < num_dst; dst++){
                const int64_t begin = mfg.csc_indptr[static_cast<size_t>(dst)];
                const int64_t end = mfg.csc_indptr[static_cast<size_t>(dst + 1)];
                seen.clear();
                seen.reserve(static_cast<size_t>(std::max<int64_t>(8, (end - begin) * 2)));
                for(int64_t pos = begin; pos < end; pos++){
                    const EdgeIDType edge_lid = mfg.edge_lids[static_cast<size_t>(pos)];
                    auto inserted = seen.emplace(edge_lid);
                    if(!inserted.second){
                        continue;
                    }
                    new_indices.emplace_back(mfg.csc_indices[static_cast<size_t>(pos)]);
                    new_edge_lids.emplace_back(edge_lid);
                    if(has_delta_ts){
                        new_delta_t.emplace_back(mfg.delta_t[static_cast<size_t>(pos)]);
                    }
                }
                new_indptr[static_cast<size_t>(dst + 1)] = static_cast<int64_t>(new_edge_lids.size());
            }
            if(new_edge_lids.size() == mfg.edge_lids.size()){
                return;
            }
            mfg.csc_indptr.swap(new_indptr);
            mfg.csc_indices.swap(new_indices);
            mfg.edge_lids.swap(new_edge_lids);
            if(has_delta_ts){
                mfg.delta_t.swap(new_delta_t);
            }
        }

        static uint64_t splitmix64(uint64_t value)
        {
            value += 0x9E3779B97F4A7C15ULL;
            value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
            value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
            return value ^ (value >> 31);
        }

        uint64_t stateless_random_bits(int layer, int64_t root_pos, int64_t draw_pos, uint64_t salt) const
        {
            uint64_t value = base_seed;
            value ^= static_cast<uint64_t>(layer + 1) * 0xD1B54A32D192ED03ULL;
            value ^= static_cast<uint64_t>(root_pos + 1) * 0xABC98388FB8FAC03ULL;
            value ^= static_cast<uint64_t>(draw_pos + 1) * 0x8CB92BA72F3D8DD7ULL;
            value ^= salt;
            return splitmix64(value);
        }

        double stateless_uniform01(int layer, int64_t root_pos, int64_t draw_pos, uint64_t salt) const
        {
            return static_cast<double>(stateless_random_bits(layer, root_pos, draw_pos, salt) >> 11) *
                (1.0 / 9007199254740992.0);
        }

        int stateless_mod(int layer, int64_t root_pos, int64_t draw_pos, int modulo, uint64_t salt) const
        {
            return static_cast<int>(stateless_random_bits(layer, root_pos, draw_pos, salt) % static_cast<uint64_t>(modulo));
        }

        size_t compact_node_index_reserve(int64_t edge_count, int64_t layer) const
        {
            if(edge_count <= 0) return 0;
            int fanout = 0;
            if(layer >= 0 && layer < static_cast<int64_t>(fanouts.size())){
                fanout = fanouts[static_cast<size_t>(layer)];
            }
            if(fanout >= 16){
                return static_cast<size_t>(std::max<int64_t>(1, edge_count / 3));
            }
            if(fanout >= 8){
                return static_cast<size_t>(std::max<int64_t>(1, edge_count / 2));
            }
            return static_cast<size_t>(edge_count);
        }

        void neighbor_sample_from_nodes(th::Tensor nodes, optional<th::Tensor> root_ts, optional<bool> part_unique);
        void neighbor_sample_from_nodes_static(th::Tensor nodes, bool part_unique);
        void neighbor_sample_from_nodes_static_layer(th::Tensor nodes, int cur_layer, bool part_unique);
        void neighbor_sample_from_nodes_static_layer_data(const NodeIDType* nodes_data, int64_t num_input_nodes, int cur_layer, bool part_unique);
        void neighbor_sample_from_nodes_with_before(th::Tensor nodes, th::Tensor root_ts);
        void neighbor_sample_from_dynamic_nodes(th::Tensor nodes, th::Tensor root_ts);
        void neighbor_sample_from_nodes_with_before_layer(th::Tensor nodes, th::Tensor root_ts, int cur_layer);
        void neighbor_sample_from_nodes_with_before_layer_data(
            const NodeIDType* nodes_data,
            const TimeStampType* ts_data,
            int64_t num_input_nodes,
            int cur_layer
        );
        std::vector<NativeSamplingOutput> sample_dtdg_uniform(th::Tensor root_nodes, int64_t t_now, int64_t num_hist);
        template<typename T>
        void union_to_vector(vector<T> *p, vector<T> &to_vec);
        void sample_unique(     th::Tensor seed, th::Tensor seed_ts,
                                th::Tensor nid_mapper, th::Tensor eid_mapper,string out_device);
        void set_edge_read_dist_index(th::Tensor read_dist_index, int64_t world_size);
        void build_edge_read_layout(NativeSamplingOutput &out);
        void count_sampled_local_remote(const NativeSamplingOutput &out);
        NativeSamplingOutput get_sampling_output(th::Tensor root_nodes, optional<th::Tensor> root_ts);
        NativeSamplingOutput get_sampling_output_compact(th::Tensor root_nodes, optional<th::Tensor> root_ts);
        NativeSamplingOutput get_sampling_output_parallel(th::Tensor root_nodes, optional<th::Tensor> root_ts);
};

void ParallelSampler::set_edge_read_dist_index(th::Tensor read_dist_index, int64_t world_size)
{
    AT_ASSERTM(read_dist_index.is_contiguous(), "read_dist_index must be contiguous");
    AT_ASSERTM(read_dist_index.dim() == 1, "read_dist_index must be one-dimensional");
    AT_ASSERTM(read_dist_index.scalar_type() == torch::kInt64, "read_dist_index must be int64");
    int64_t *ptr = read_dist_index.data_ptr<int64_t>();
    edge_read_dist_index.assign(ptr, ptr + read_dist_index.numel());
    edge_read_world_size = world_size;
}

void ParallelSampler::build_edge_read_layout(NativeSamplingOutput &out)
{
    double t0 = omp_get_wtime();
    if(edge_read_dist_index.empty()){
        edge_read_layout_seconds += omp_get_wtime() - t0;
        return;
    }
    const int64_t n = static_cast<int64_t>(out.edge_gids.size());
    int64_t world_size = edge_read_world_size;
    if(world_size <= 0){
        for(int64_t index : edge_read_dist_index){
            int64_t part_id = (index >> 50) & 8191;
            if(part_id + 1 > world_size) world_size = part_id + 1;
        }
    }
    if(world_size <= 0){
        world_size = 1;
    }

    edge_read_index_scratch.resize(static_cast<size_t>(n));
    vector<int64_t>& read_index = edge_read_index_scratch;
    for(int64_t i = 0; i < n; i++){
        EdgeIDType gid = out.edge_gids[static_cast<size_t>(i)];
        if(gid < 0 || gid >= static_cast<EdgeIDType>(edge_read_dist_index.size())){
            out.edge_read_index.clear();
            out.edge_read_ptr.clear();
            out.compute_to_edge_feature.clear();
            edge_read_layout_seconds += omp_get_wtime() - t0;
            return;
        }
        read_index[static_cast<size_t>(i)] = edge_read_dist_index[static_cast<size_t>(gid)];
    }

    edge_read_order_scratch.resize(static_cast<size_t>(n));
    vector<int64_t>& order = edge_read_order_scratch;
    for(int64_t i = 0; i < n; i++){
        order[static_cast<size_t>(i)] = i;
    }
    sort(order.begin(), order.end(), [&](int64_t a, int64_t b){
        int64_t ia = read_index[static_cast<size_t>(a)];
        int64_t ib = read_index[static_cast<size_t>(b)];
        int64_t part_a = (ia >> 50) & 8191;
        int64_t part_b = (ib >> 50) & 8191;
        if(part_a != part_b) return part_a < part_b;
        int64_t loc_a = ia & ((1LL << 48) - 1);
        int64_t loc_b = ib & ((1LL << 48) - 1);
        if(loc_a != loc_b) return loc_a < loc_b;
        return a < b;
    });

    out.edge_read_index.resize(static_cast<size_t>(n));
    out.compute_to_edge_feature.resize(static_cast<size_t>(n));
    out.edge_read_ptr.assign(static_cast<size_t>(world_size + 1), 0);
    for(int64_t grouped_pos = 0; grouped_pos < n; grouped_pos++){
        int64_t compute_pos = order[static_cast<size_t>(grouped_pos)];
        int64_t index = read_index[static_cast<size_t>(compute_pos)];
        int64_t part_id = (index >> 50) & 8191;
        if(part_id >= 0 && part_id < world_size){
            out.edge_read_ptr[static_cast<size_t>(part_id + 1)] += 1;
        }
        out.edge_read_index[static_cast<size_t>(grouped_pos)] = index;
        out.compute_to_edge_feature[static_cast<size_t>(compute_pos)] = grouped_pos;
    }
    for(int64_t i = 0; i < world_size; i++){
        out.edge_read_ptr[static_cast<size_t>(i + 1)] += out.edge_read_ptr[static_cast<size_t>(i)];
    }
    edge_read_layout_seconds += omp_get_wtime() - t0;
}

void ParallelSampler::count_sampled_local_remote(const NativeSamplingOutput &out)
{
    if(local_part < 0 || node_part.empty() || part.empty()){
        sampler_local_nodes += static_cast<double>(out.node_gids.size());
        sampler_local_edges += static_cast<double>(out.edge_gids.size());
        return;
    }
    int64_t local_nodes = 0;
    int64_t remote_nodes = 0;
    for(NodeIDType gid : out.node_gids){
        if(gid >= 0 && gid < static_cast<NodeIDType>(node_part.size()) && node_part[static_cast<size_t>(gid)] == local_part){
            local_nodes++;
        } else {
            remote_nodes++;
        }
    }
    int64_t local_edges = 0;
    int64_t remote_edges = 0;
    for(EdgeIDType gid : out.edge_gids){
        if(gid >= 0 && gid < static_cast<EdgeIDType>(part.size()) && part[static_cast<size_t>(gid)] == local_part){
            local_edges++;
        } else {
            remote_edges++;
        }
    }
    sampler_local_nodes += static_cast<double>(local_nodes);
    sampler_remote_nodes += static_cast<double>(remote_nodes);
    sampler_local_edges += static_cast<double>(local_edges);
    sampler_remote_edges += static_cast<double>(remote_edges);
}



void ParallelSampler :: neighbor_sample_from_nodes(th::Tensor nodes, optional<th::Tensor> root_ts, optional<bool> part_unique)
{
    omp_set_num_threads(threads);
    if(policy == "weighted")
        AT_ASSERTM(tnb.weighted, "Tnb has no weight infomation!");
    else if(policy == "recent")
        AT_ASSERTM(tnb.with_timestamp, "Tnb has no timestamp infomation!");
    else if(policy == "uniform"|| policy == "dtdg_uniform" || policy =="boundery_recent_decay"||policy=="boundery_recent_uniform"||policy=="boundery_uniform");
    else{
        throw runtime_error("The policy \"" + policy + "\" is not exit!");
    }
    if(tnb.with_timestamp){
        AT_ASSERTM(tnb.with_timestamp, "Tnb has no timestamp infomation!");
        AT_ASSERTM(root_ts.has_value(), "Parameter mismatch!");
        //neighbor_sample_from_dynamic_nodes(nodes,root_ts.value());
        neighbor_sample_from_nodes_with_before(nodes, root_ts.value());
    }
    else{
        bool flag = part_unique.has_value() ? part_unique.value() : true;
        neighbor_sample_from_nodes_static(nodes, flag);
    }
}

void ParallelSampler :: neighbor_sample_from_nodes_static_layer(th::Tensor nodes, int cur_layer, bool part_unique){
    neighbor_sample_from_nodes_static_layer_data(
        get_data_ptr<NodeIDType>(nodes),
        nodes.size(0),
        cur_layer,
        part_unique
    );
}

void ParallelSampler::neighbor_sample_from_nodes_static_layer_data(
    const NodeIDType* nodes_data,
    int64_t num_input_nodes,
    int cur_layer,
    bool part_unique
){
    py::gil_scoped_release release;
    double tot_start_time = omp_get_wtime();

    TemporalGraphBlock tgb = TemporalGraphBlock();
    int fanout = fanouts[cur_layer];
    ret[cur_layer] = TemporalGraphBlock();
    vector<phmap::parallel_flat_hash_set<NodeIDType>> node_s_threads(threads);
    vector<vector<NodeIDType>> node_threads(threads);
    phmap::parallel_flat_hash_set<NodeIDType> node_s;
    vector<vector<NodeIDType>> eid_threads(threads);
    vector<vector<NodeIDType>> src_index_threads(threads);
    AT_ASSERTM(tnb.with_eid, "Tnb has no eid infomation! We need eid!");
    
    // double start_time = omp_get_wtime();
    int reserve_capacity = int(ceil(static_cast<float>(num_input_nodes) / threads)) * fanout;
    
#pragma omp parallel
{
    int tid = omp_get_thread_num();
    unsigned int loc_seed = tid;
    eid_threads[tid].reserve(reserve_capacity);
    src_index_threads[tid].reserve(reserve_capacity);
    if(!part_unique)
        node_threads[tid].reserve(reserve_capacity);
#pragma omp for schedule(static, int(ceil((static_cast<float>(num_input_nodes) / threads))))
    for(int64_t i=0; i<num_input_nodes; i++){
        NodeIDType node = nodes_data[i];
        vector<NodeIDType>& nei = tnb.neighbors[node];
        vector<EdgeIDType> edge;
        edge = tnb.eid[node];

        double s_start_time = omp_get_wtime();
        if(tnb.deg[node]>fanout){
            phmap::flat_hash_set<NodeIDType> temp_s;
            default_random_engine e(8);//(time(0));
            // uniform_int_distribution<> u(0, tnb.deg[node]-1);            
            // while(temp_s.size()!=fanout && temp_s.size()<tnb.neighbors_set[node].size()){
            for(int i=0;i<fanout;i++){
                //Ñ­»·Ñ¡Ôñfanout¸öÁÚ¾Ó
                NodeIDType indice;
                if(policy == "weighted"){//¿¼ÂÇ±ßÈ¨ÖØÐÅÏ¢
                    const vector<WeightType>& ew = tnb.edge_weight[node];
                    indice = sample_multinomial(ew, e);
                }
                else if(policy == "uniform"){//¾ùÔÈ²ÉÑù
                    // indice = u(e);
                    indice = stateless_mod(cur_layer, i, temp_s.size(), static_cast<int>(nei.size()), 0x1F123BB5ULL);
                }
                auto chosen_n_iter = nei.begin() + indice;
                auto chosen_e_iter = edge.begin() + indice;
                if(part_unique){
                    auto rst = temp_s.insert(*chosen_n_iter);
                    if(rst.second){ //²»ÖØ¸´
                        eid_threads[tid].emplace_back(*chosen_e_iter);
                        node_s_threads[tid].insert(*chosen_n_iter);
                        if(!tnb.neighbors_set.empty() && temp_s.size()<fanout && temp_s.size()<tnb.neighbors_set[node].size()) fanout++;
                    }
                }
                else{
                    eid_threads[tid].emplace_back(*chosen_e_iter);
                    node_threads[tid].emplace_back(*chosen_n_iter);
                }
            }
            if(part_unique)
                src_index_threads[tid].insert(src_index_threads[tid].end(), temp_s.size(), i);
            else
                src_index_threads[tid].insert(src_index_threads[tid].end(), fanout, i);
        }
        else{
            src_index_threads[tid].insert(src_index_threads[tid].end(), tnb.deg[node], i);
            if(part_unique)
                node_s_threads[tid].insert(nei.begin(), nei.end());
            else
                node_threads[tid].insert(node_threads[tid].end(), nei.begin(), nei.end());
            eid_threads[tid].insert(eid_threads[tid].end(),edge.begin(), edge.end());
        }
        if(tid==0)
            ret[0].sample_time += omp_get_wtime() - s_start_time;
    }
}
    // double end_time = omp_get_wtime();
    
    // cout<<"neighbor_sample_from_nodes parallel part consume: "<<end_time-start_time<<"s"<<endl;
    
    int size = 0;
    vector<int> each_begin(threads);
    for(int i = 0; i<threads; i++){
        int s = eid_threads[i].size();
        each_begin[i]=size;
        size += s;
    }
    ret[cur_layer].eid.resize(size);
    ret[cur_layer].sample_nodes.resize(size);
    ret[cur_layer].src_index.resize(size);
#pragma omp parallel for schedule(static, 1)
    for(int i = 0; i<threads; i++){
        copy(eid_threads[i].begin(), eid_threads[i].end(), ret[cur_layer].eid.begin()+each_begin[i]);
        if(!part_unique)
            copy(node_threads[i].begin(), node_threads[i].end(), ret[cur_layer].sample_nodes.begin()+each_begin[i]);
        copy(src_index_threads[i].begin(), src_index_threads[i].end(), ret[cur_layer].src_index.begin()+each_begin[i]);
    }
    if(part_unique){
        for(int i = 0; i<threads; i++)
            node_s.insert(node_s_threads[i].begin(), node_s_threads[i].end());
        ret[cur_layer].sample_nodes.assign(node_s.begin(), node_s.end());
    }

    ret[0].tot_time += omp_get_wtime() - tot_start_time;
    ret[0].sample_edge_num += ret[cur_layer].eid.size();
    py::gil_scoped_acquire acquire;
}

void ParallelSampler :: neighbor_sample_from_nodes_static(th::Tensor nodes, bool part_unique){
    auto nodes_data = get_data_ptr<NodeIDType>(nodes);
    int64_t num_input_nodes = nodes.size(0);
    for(int i=0;i<num_layers;i++){
        if(i==0) neighbor_sample_from_nodes_static_layer_data(nodes_data, num_input_nodes, i, part_unique);
        else neighbor_sample_from_nodes_static_layer_data(
            ret[i-1].sample_nodes.data(),
            static_cast<int64_t>(ret[i-1].sample_nodes.size()),
            i,
            part_unique
        );
    }
}

void ParallelSampler :: neighbor_sample_from_nodes_with_before_layer(
        th::Tensor nodes, th::Tensor root_ts, int cur_layer){
    neighbor_sample_from_nodes_with_before_layer_data(
        get_data_ptr<NodeIDType>(nodes),
        get_data_ptr<TimeStampType>(root_ts),
        nodes.size(0),
        cur_layer
    );
}

void ParallelSampler::neighbor_sample_from_nodes_with_before_layer_data(
        const NodeIDType* nodes_data,
        const TimeStampType* ts_data,
        int64_t num_input_nodes,
        int cur_layer){
    py::gil_scoped_release release;
    double tot_start_time = omp_get_wtime();
    ret[cur_layer] = TemporalGraphBlock();
    int fanout = fanouts[cur_layer];
    // HashT<pair<NodeIDType,TimeStampType> > node_s;
    vector<TemporalGraphBlock> tgb_i(threads);
    
    default_random_engine e(8);//(time(0));
    // double start_time = omp_get_wtime();
    int reserve_capacity = int(ceil(static_cast<float>(num_input_nodes) / threads)) * fanout;
#pragma omp parallel
{
    int tid = omp_get_thread_num();
    unsigned int loc_seed = tid;
    tgb_i[tid].sample_nodes.reserve(reserve_capacity);
    tgb_i[tid].sample_nodes_ts.reserve(reserve_capacity);
    tgb_i[tid].delta_ts.reserve(reserve_capacity);
    tgb_i[tid].eid.reserve(reserve_capacity);
    tgb_i[tid].src_index.reserve(reserve_capacity);
    tgb_i[tid].col.reserve(reserve_capacity);
    tgb_i[tid].sample_weight.reserve(reserve_capacity);
    vector<double> remote_prob(static_cast<size_t>(std::max(1, fanout)), 0.0);
#pragma omp for schedule(static, int(ceil((static_cast<float>(num_input_nodes) / threads))))
    for(int64_t i=0; i<num_input_nodes; i++){
        // int tid = omp_get_thread_num();
        NodeIDType node = nodes_data[i];
        TimeStampType rtts = ts_data[i];
        const vector<NodeIDType>& neighbors = tnb.neighbors[node];
        const vector<TimeStampType>& timestamps = tnb.timestamp[node];
        const vector<EdgeIDType>& edge_ids = tnb.eid[node];
        int end_index = lower_bound(timestamps.begin(), timestamps.end(), rtts)-timestamps.begin();
        int start_index = 0;
        if(policy == "dtdg_uniform"){
            start_index = end_index;
            end_index = upper_bound(timestamps.begin(), timestamps.end(), rtts)-timestamps.begin();
        }

        double s_start_time = omp_get_wtime();
        
        if(policy == "dtdg_uniform" && end_index <= start_index){
            continue;
        }
        else if(policy == "dtdg_uniform" && (end_index - start_index) <= fanout){
            tgb_i[tid].src_index.insert(tgb_i[tid].src_index.end(), end_index-start_index, i);
            tgb_i[tid].sample_nodes.insert(tgb_i[tid].sample_nodes.end(), neighbors.begin()+start_index, neighbors.begin()+end_index);
            tgb_i[tid].sample_nodes_ts.insert(tgb_i[tid].sample_nodes_ts.end(), timestamps.begin()+start_index, timestamps.begin()+end_index);
            tgb_i[tid].eid.insert(tgb_i[tid].eid.end(), edge_ids.begin()+start_index, edge_ids.begin()+end_index);
            for(int cid = start_index; cid < end_index; cid++){
                tgb_i[tid].delta_ts.emplace_back(rtts-timestamps[cid]);
                tgb_i[tid].sample_weight.emplace_back(cid-start_index);
            }
        }
        else if ((policy == "recent") || (end_index <= fanout)&& policy.substr(0,8) != "boundery" ){
            int cnt  = 0;
            for(int cid = end_index-1;cid>=0;cid--){
                cnt++;
                if(cnt>fanout)break;
            }
            int start_index = max(0, end_index-fanout);
            tgb_i[tid].src_index.insert(tgb_i[tid].src_index.end(), end_index-start_index, i);
            tgb_i[tid].sample_nodes.insert(tgb_i[tid].sample_nodes.end(), neighbors.begin()+start_index, neighbors.begin()+end_index);
            tgb_i[tid].sample_nodes_ts.insert(tgb_i[tid].sample_nodes_ts.end(), timestamps.begin()+start_index, timestamps.begin()+end_index);
            tgb_i[tid].eid.insert(tgb_i[tid].eid.end(), edge_ids.begin()+start_index, edge_ids.begin()+end_index);
            for(int cid = start_index; cid < end_index;cid++){
                tgb_i[tid].delta_ts.emplace_back(rtts-timestamps[cid]);
                tgb_i[tid].sample_weight.emplace_back(cid-start_index);
            }
        }
        else if(policy == "boundery_recent_uniform"){
            int cnt = 0;
            int cal_cnt = 0;
            for(int cid = end_index-1;cid>=0;cid--){
                cal_cnt++;
                if(cal_cnt > 2 * fanout)break;
                if(is_boundary_candidate(edge_ids[cid], neighbors[cid])){
                    double p0 = stateless_uniform01(cur_layer, i, cid, 0xB001D00DULL);
                    if(p0 > boundery_probility)continue;
                }
                tgb_i[tid].src_index.emplace_back(i);
                tgb_i[tid].sample_nodes.emplace_back(neighbors[cid]);
                tgb_i[tid].sample_nodes_ts.emplace_back(timestamps[cid]);
                tgb_i[tid].delta_ts.emplace_back(rtts-timestamps[cid]);
                tgb_i[tid].eid.emplace_back(edge_ids[cid]);
                cnt++;
                if(cnt >= fanout)break;

            }

        }
        else if(policy == "boundery_uniform"){
            int cnt = 0;
            int cal_cnt = 0;
            for(int cid = end_index-1;cid>=0;cid--){
                cal_cnt++;
                if(cal_cnt > 2 * fanout)break;
                if(is_boundary_candidate(edge_ids[cid], neighbors[cid])){
                    double p0 = stateless_uniform01(cur_layer, i, cid, 0xB0A0D00DULL);
                    if(p0 > boundery_probility)continue;
                }
                tgb_i[tid].src_index.emplace_back(i);
                tgb_i[tid].sample_nodes.emplace_back(neighbors[cid]);
                tgb_i[tid].sample_nodes_ts.emplace_back(timestamps[cid]);
                tgb_i[tid].delta_ts.emplace_back(rtts-timestamps[cid]);
                tgb_i[tid].eid.emplace_back(edge_ids[cid]);
                cnt++;
                if(cnt >= fanout)break;
            }
        }
        else if(policy == "boundery_recent_decay"){
            int cnt = 0;
            int cal_cnt = 0;
            double sum_p = 0;
            int sum_1 = 0;
            TimeStampType delta = end_index-1>=0?(rtts - timestamps[end_index-1])*fanout:0;
            if(static_cast<int>(remote_prob.size()) < fanout){
                remote_prob.resize(static_cast<size_t>(fanout), 0.0);
            }
            for(int cid = end_index-1;cid>=0;cid--){
                cal_cnt++;
                if(cal_cnt>fanout)break;
                if(is_boundary_candidate(edge_ids[cid], neighbors[cid])){
                    double ep = exp((double)(timestamps[cid]-rtts)/(delta));
                    sum_p+=ep;remote_prob[static_cast<size_t>(cal_cnt-1)]=ep;
                    sum_1++;
                }
            }
            if(sum_p<1e-6)sum_p=1;
            cal_cnt = 0;
            for(int cid = end_index-1;cid>=0;cid--){
                cal_cnt++;
                if(cal_cnt > 2 * fanout)break;
                if(is_boundary_candidate(edge_ids[cid], neighbors[cid])){
                    if(cal_cnt<=fanout){
                        double p0 = stateless_uniform01(cur_layer, i, cid, 0xDECADA7AULL);
                        double ep = boundery_probility*remote_prob[static_cast<size_t>(cal_cnt-1)]/sum_p*sum_1;
                        if(p0 > ep)continue;
                        //tgb_i[tid].sample_weight.emplace_back((float)ep);
                    }
                    else continue;
                    //cout<<"in"<<endl;
                    
                }
                else{
                    //tgb_i[tid].sample_weight.emplace_back((float)1.0);
                }
                tgb_i[tid].src_index.emplace_back(i);
                tgb_i[tid].sample_nodes.emplace_back(neighbors[cid]);
                tgb_i[tid].sample_nodes_ts.emplace_back(timestamps[cid]);
                tgb_i[tid].delta_ts.emplace_back(rtts-timestamps[cid]);
                tgb_i[tid].eid.emplace_back(edge_ids[cid]);
                cnt++;
                if(cnt >= fanout)break;
            }
        }
        else{
            //¿ÉÑ¡ÁÚ¾Ó±ß´óÓÚÉÈ³öµÄ»°ÐèÒªËæ»úÑ¡Ôñfanout¸öÁÚ¾Ó
            tgb_i[tid].src_index.insert(tgb_i[tid].src_index.end(), fanout, i);
            uniform_int_distribution<> u(start_index, end_index-1);
            //cout<<end_index<<endl;
            // cout<<"start:"<<start_index<<" end:"<<end_index<<endl;
            for(int draw=0; draw<fanout; draw++){
                int cid;
                if(policy == "uniform")
                    // cid = u(e);
                    cid = stateless_mod(cur_layer, i, draw, end_index, 0xA11CE001ULL);
                else if(policy == "dtdg_uniform")
                    cid = start_index + stateless_mod(cur_layer, i, draw, end_index - start_index, 0xD7D60001ULL);
                else if(policy == "weighted"){
                    const vector<WeightType>& ew = tnb.edge_weight[node];
                    cid = sample_multinomial(ew, e);
                }
                tgb_i[tid].sample_nodes.emplace_back(neighbors[cid]);
                tgb_i[tid].sample_nodes_ts.emplace_back(timestamps[cid]);
                tgb_i[tid].delta_ts.emplace_back(rtts-timestamps[cid]);
                tgb_i[tid].eid.emplace_back(edge_ids[cid]);
            }
        }
        if(tid==0)
            ret[0].sample_time += omp_get_wtime() - s_start_time;
    }
}
    // double end_time = omp_get_wtime();
    // cout<<"neighbor_sample_from_nodes parallel part consume: "<<end_time-start_time<<"s"<<endl;
    
    // start_time = omp_get_wtime();

    int size = 0;
    vector<int> each_begin(threads);
    for(int i = 0; i<threads; i++){
        int s = tgb_i[i].eid.size();
        each_begin[i]=size;
        size += s;
    }
    //if(policy == "boundery_recent_decay")
    ret[cur_layer].sample_weight.resize(size);
    ret[cur_layer].eid.resize(size);
    ret[cur_layer].src_index.resize(size);
    ret[cur_layer].delta_ts.resize(size);
    ret[cur_layer].sample_nodes.resize(size);
    ret[cur_layer].sample_nodes_ts.resize(size);

#pragma omp parallel for schedule(static, 1)
    for(int i = 0; i<threads; i++){
        //if(policy == "boundery_recent_decay")
        copy(tgb_i[i].sample_weight.begin(), tgb_i[i].sample_weight.end(), ret[cur_layer].sample_weight.begin()+each_begin[i]);
        copy(tgb_i[i].eid.begin(), tgb_i[i].eid.end(), ret[cur_layer].eid.begin()+each_begin[i]);
        copy(tgb_i[i].src_index.begin(), tgb_i[i].src_index.end(), ret[cur_layer].src_index.begin()+each_begin[i]);
        copy(tgb_i[i].delta_ts.begin(), tgb_i[i].delta_ts.end(), ret[cur_layer].delta_ts.begin()+each_begin[i]);
        copy(tgb_i[i].sample_nodes.begin(), tgb_i[i].sample_nodes.end(), ret[cur_layer].sample_nodes.begin()+each_begin[i]);
        copy(tgb_i[i].sample_nodes_ts.begin(), tgb_i[i].sample_nodes_ts.end(), ret[cur_layer].sample_nodes_ts.begin()+each_begin[i]);
    }

    // end_time = omp_get_wtime();
    // cout<<"end union consume: "<<end_time-start_time<<"s"<<endl;
    ret[0].tot_time += omp_get_wtime() - tot_start_time;
    ret[0].sample_edge_num += ret[cur_layer].eid.size();
    py::gil_scoped_acquire acquire;
}

void ParallelSampler :: neighbor_sample_from_nodes_with_before(th::Tensor nodes, th::Tensor root_ts){
    auto nodes_data = get_data_ptr<NodeIDType>(nodes);
    auto ts_data = get_data_ptr<TimeStampType>(root_ts);
    int64_t num_input_nodes = nodes.size(0);
    for(int i=0;i<num_layers;i++){
        if(i==0) {
            neighbor_sample_from_nodes_with_before_layer_data(nodes_data, ts_data, num_input_nodes, i);
        }
        else {
            neighbor_sample_from_nodes_with_before_layer_data(
                ret[i-1].sample_nodes.data(),
                ret[i-1].sample_nodes_ts.data(),
                static_cast<int64_t>(ret[i-1].sample_nodes.size()),
                i
            );
        }
    }
}

std::vector<NativeSamplingOutput> ParallelSampler::sample_dtdg_uniform(th::Tensor root_nodes, int64_t t_now, int64_t num_hist)
{
    AT_ASSERTM(tnb.with_timestamp, "DTDG uniform sampling requires timestamped/time-slice edges");
    AT_ASSERTM(root_nodes.is_contiguous(), "root_nodes must be contiguous");
    AT_ASSERTM(root_nodes.dim() == 1, "root_nodes must be one-dimensional");
    AT_ASSERTM(num_hist >= 1, "num_hist must be >= 1");

    std::vector<NativeSamplingOutput> outputs;
    outputs.reserve(static_cast<size_t>(num_hist));
    const std::string old_policy = policy;
    policy = "dtdg_uniform";

    int64_t start_slice = t_now - num_hist + 1;
    for(int64_t slice = start_slice; slice <= t_now; slice++){
        reset();
        th::Tensor root_ts = th::full(
            {root_nodes.size(0)},
            slice,
            th::TensorOptions().dtype(th::kInt64).device(root_nodes.device())
        ).contiguous();
        neighbor_sample_from_nodes_with_before(root_nodes, root_ts);
        NativeSamplingOutput out = get_sampling_output(root_nodes, root_ts);
        outputs.emplace_back(std::move(out));
    }

    policy = old_policy;
    return outputs;
}


NativeSamplingOutput ParallelSampler::get_sampling_output(th::Tensor root_nodes, optional<th::Tensor> root_ts)
{
    AT_ASSERTM(root_nodes.is_contiguous(), "root_nodes must be contiguous");
    AT_ASSERTM(root_nodes.dim() == 1, "root_nodes must be one-dimensional");
    if(root_ts.has_value()){
        AT_ASSERTM(root_ts.value().is_contiguous(), "root_ts must be contiguous");
        AT_ASSERTM(root_ts.value().dim() == 1, "root_ts must be one-dimensional");
        AT_ASSERTM(root_ts.value().size(0) == root_nodes.size(0), "root_ts size must match root_nodes");
    }

    py::gil_scoped_release release;

    NativeSamplingOutput out;
    out.mfgs.resize(ret.size());
    out.node_layer_ptr.reserve(ret.size() + 2);
    out.edge_layer_ptr.reserve(ret.size() + 1);
    out.node_layer_ptr.emplace_back(0);
    out.edge_layer_ptr.emplace_back(0);

    size_t estimated_nodes = static_cast<size_t>(root_nodes.size(0));
    size_t estimated_edges = 0;
    for(const TemporalGraphBlock& block : ret){
        estimated_nodes += block.sample_nodes.size();
        estimated_edges += block.eid.size();
    }
    out.node_gids.reserve(estimated_nodes);
    out.node_ts.reserve(estimated_nodes);
    out.edge_gids.reserve(estimated_edges);
    out.edge_ts.reserve(estimated_edges);

    phmap::flat_hash_map<NodeInstanceKey, int64_t, NodeInstanceKeyHash> node_lid;
    node_lid.reserve(estimated_nodes);
    edge_lid_touched.clear();
    edge_lid_touched.reserve(estimated_edges);

    auto add_node = [&](NodeIDType gid, TimeStampType ts) -> int64_t {
        NodeInstanceKey key{gid, ts};
        auto it = node_lid.find(key);
        if(it != node_lid.end()) return it->second;
        int64_t lid = static_cast<int64_t>(out.node_gids.size());
        node_lid.emplace(key, lid);
        out.node_gids.emplace_back(gid);
        out.node_ts.emplace_back(ts);
        return lid;
    };

    auto add_edge = [&](EdgeIDType gid, TimeStampType ts) -> int64_t {
        AT_ASSERTM(gid >= 0 && gid < num_edges, "edge gid out of range");
        int64_t& cached = edge_lid_dense[static_cast<size_t>(gid)];
        if(cached >= 0) return cached;
        int64_t lid = static_cast<int64_t>(out.edge_gids.size());
        cached = lid;
        edge_lid_touched.emplace_back(gid);
        out.edge_gids.emplace_back(gid);
        out.edge_ts.emplace_back(ts);
        return lid;
    };

    auto root_nodes_data = get_data_ptr<NodeIDType>(root_nodes);
    TimeStampType* root_ts_data = nullptr;
    if(root_ts.has_value()){
        root_ts_data = get_data_ptr<TimeStampType>(root_ts.value());
    }

    out.root_gids.reserve(root_nodes.size(0));
    out.root_ts.reserve(root_nodes.size(0));
    out.root_lids.reserve(root_nodes.size(0));
    vector<int64_t> frontier_lids;
    frontier_lids.reserve(root_nodes.size(0));

    for(int64_t i = 0; i < root_nodes.size(0); i++){
        NodeIDType gid = root_nodes_data[i];
        TimeStampType ts = root_ts_data == nullptr ? 0 : root_ts_data[i];
        int64_t lid = add_node(gid, ts);
        out.root_gids.emplace_back(gid);
        out.root_ts.emplace_back(ts);
        out.root_lids.emplace_back(lid);
        frontier_lids.emplace_back(lid);
    }
    out.node_layer_ptr.emplace_back(static_cast<int64_t>(out.node_gids.size()));

    for(int64_t layer = 0; layer < static_cast<int64_t>(ret.size()); layer++){
        TemporalGraphBlock& block = ret[layer];
        NativeMFGBlock mfg;
        mfg.layer = layer;
        mfg.dst_begin = layer == 0 ? 0 : out.node_layer_ptr[layer];
        mfg.dst_end = out.node_layer_ptr[layer + 1];

        const int64_t num_dst = static_cast<int64_t>(frontier_lids.size());
        mfg.dst_lids.assign(frontier_lids.begin(), frontier_lids.end());

        mfg.csc_indptr.assign(num_dst + 1, 0);
        const int64_t edge_count = static_cast<int64_t>(block.src_index.size());
        for(int64_t j = 0; j < edge_count; j++){
            int64_t dst_pos = block.src_index[j];
            if(dst_pos >= 0 && dst_pos < num_dst){
                mfg.csc_indptr[dst_pos + 1] += 1;
            }
        }
        for(int64_t i = 0; i < num_dst; i++){
            mfg.csc_indptr[i + 1] += mfg.csc_indptr[i];
        }

        vector<int64_t> cursor = mfg.csc_indptr;
        mfg.csc_indices.resize(edge_count);
        mfg.edge_lids.resize(edge_count);
        const bool has_delta_ts = !block.delta_ts.empty();
        if(has_delta_ts){
            mfg.delta_t.resize(edge_count);
        }
        vector<int64_t> next_frontier_lids(static_cast<size_t>(block.sample_nodes.size()), -1);

        phmap::flat_hash_map<NodeInstanceKey, int64_t, NodeInstanceKeyHash> local_node_index;
        phmap::flat_hash_map<EdgeIDType, int64_t> local_edge_index;
        vector<NodeInstanceKey> local_nodes;
        vector<EdgeIDType> local_edges;
        vector<TimeStampType> local_edge_ts;
        vector<int64_t> edge_to_local_node(static_cast<size_t>(edge_count), -1);
        vector<int64_t> edge_to_local_edge(static_cast<size_t>(edge_count), -1);
        local_node_index.reserve(static_cast<size_t>(edge_count));
        local_edge_index.reserve(static_cast<size_t>(edge_count));
        local_nodes.reserve(static_cast<size_t>(edge_count));
        local_edges.reserve(static_cast<size_t>(edge_count));
        local_edge_ts.reserve(static_cast<size_t>(edge_count));

        for(int64_t j = 0; j < edge_count; j++){
            const NodeInstanceKey node_key{
                block.sample_nodes[j],
                block.sample_nodes_ts.empty() ? 0 : block.sample_nodes_ts[j],
            };
            auto node_it = local_node_index.find(node_key);
            if(node_it == local_node_index.end()){
                int64_t idx = static_cast<int64_t>(local_nodes.size());
                local_node_index.emplace(node_key, idx);
                local_nodes.emplace_back(node_key);
                edge_to_local_node[static_cast<size_t>(j)] = idx;
            } else {
                edge_to_local_node[static_cast<size_t>(j)] = node_it->second;
            }

            const EdgeIDType edge_gid = block.eid[j];
            auto edge_it = local_edge_index.find(edge_gid);
            if(edge_it == local_edge_index.end()){
                int64_t idx = static_cast<int64_t>(local_edges.size());
                local_edge_index.emplace(edge_gid, idx);
                local_edges.emplace_back(edge_gid);
                local_edge_ts.emplace_back(node_key.ts);
                edge_to_local_edge[static_cast<size_t>(j)] = idx;
            } else {
                edge_to_local_edge[static_cast<size_t>(j)] = edge_it->second;
            }
        }

        vector<int64_t> local_node_lids(local_nodes.size(), -1);
        for(int64_t i = 0; i < static_cast<int64_t>(local_nodes.size()); i++){
            local_node_lids[static_cast<size_t>(i)] = add_node(local_nodes[static_cast<size_t>(i)].node, local_nodes[static_cast<size_t>(i)].ts);
        }
        vector<int64_t> local_edge_lids(local_edges.size(), -1);
        for(int64_t i = 0; i < static_cast<int64_t>(local_edges.size()); i++){
            local_edge_lids[static_cast<size_t>(i)] = add_edge(local_edges[static_cast<size_t>(i)], local_edge_ts[static_cast<size_t>(i)]);
        }

        for(int64_t j = 0; j < edge_count; j++){
            int64_t src_lid = local_node_lids[static_cast<size_t>(edge_to_local_node[static_cast<size_t>(j)])];
            next_frontier_lids[static_cast<size_t>(j)] = src_lid;
            int64_t dst_pos = block.src_index[j];
            if(dst_pos < 0 || dst_pos >= num_dst) continue;
            int64_t offset = cursor[dst_pos]++;
            mfg.csc_indices[offset] = src_lid;
            mfg.edge_lids[offset] = local_edge_lids[static_cast<size_t>(edge_to_local_edge[static_cast<size_t>(j)])];
            if(has_delta_ts){
                mfg.delta_t[offset] = block.delta_ts[j];
            }
        }

        mfg.src_begin = 0;
        mfg.src_end = static_cast<int64_t>(out.node_gids.size());
        mfg.src_lids.resize(mfg.src_end - mfg.src_begin);
        std::iota(mfg.src_lids.begin(), mfg.src_lids.end(), mfg.src_begin);
        out.mfgs[layer] = std::move(mfg);
        out.node_layer_ptr.emplace_back(static_cast<int64_t>(out.node_gids.size()));
        out.edge_layer_ptr.emplace_back(static_cast<int64_t>(out.edge_gids.size()));
        frontier_lids.swap(next_frontier_lids);
    }
    count_sampled_local_remote(out);
    build_edge_read_layout(out);
    for(EdgeIDType gid : edge_lid_touched){
        edge_lid_dense[static_cast<size_t>(gid)] = -1;
    }
    return out;
}



NativeSamplingOutput ParallelSampler::get_sampling_output_compact(th::Tensor root_nodes, optional<th::Tensor> root_ts)
{
    AT_ASSERTM(root_nodes.is_contiguous(), "root_nodes must be contiguous");
    AT_ASSERTM(root_nodes.dim() == 1, "root_nodes must be one-dimensional");
    if(root_ts.has_value()){
        AT_ASSERTM(root_ts.value().is_contiguous(), "root_ts must be contiguous");
        AT_ASSERTM(root_ts.value().dim() == 1, "root_ts must be one-dimensional");
        AT_ASSERTM(root_ts.value().size(0) == root_nodes.size(0), "root_ts size must match root_nodes");
    }

    py::gil_scoped_release release;
    double total_start_time = omp_get_wtime();

    NativeSamplingOutput out;
    out.mfgs.resize(ret.size());
    out.node_layer_ptr.reserve(ret.size() + 2);
    out.edge_layer_ptr.reserve(ret.size() + 1);
    out.node_layer_ptr.emplace_back(0);
    out.edge_layer_ptr.emplace_back(0);

    size_t estimated_nodes = static_cast<size_t>(root_nodes.size(0));
    size_t estimated_edges = 0;
    for(const TemporalGraphBlock& block : ret){
        estimated_nodes += block.sample_nodes.size();
        estimated_edges += block.eid.size();
    }
    out.node_gids.reserve(estimated_nodes);
    out.node_ts.reserve(estimated_nodes);
    out.edge_gids.reserve(estimated_edges);
    out.edge_ts.reserve(estimated_edges);
    out.root_gids.resize(static_cast<size_t>(root_nodes.size(0)));
    out.root_ts.resize(static_cast<size_t>(root_nodes.size(0)));
    out.root_lids.resize(static_cast<size_t>(root_nodes.size(0)));

    phmap::flat_hash_map<NodeInstanceKey, int64_t, NodeInstanceKeyHash> node_lid;
    node_lid.reserve(estimated_nodes);
    edge_lid_touched.clear();
    edge_lid_touched.reserve(estimated_edges);

    auto add_node = [&](NodeIDType gid, TimeStampType ts) -> int64_t {
        NodeInstanceKey key{gid, ts};
        auto it = node_lid.find(key);
        if(it != node_lid.end()) return it->second;
        int64_t lid = static_cast<int64_t>(out.node_gids.size());
        node_lid.emplace(key, lid);
        out.node_gids.emplace_back(gid);
        out.node_ts.emplace_back(ts);
        return lid;
    };

    auto add_edge = [&](EdgeIDType gid, TimeStampType ts) -> int64_t {
        AT_ASSERTM(gid >= 0 && gid < num_edges, "edge gid out of range");
        int64_t& cached = edge_lid_dense[static_cast<size_t>(gid)];
        if(cached >= 0) return cached;
        int64_t lid = static_cast<int64_t>(out.edge_gids.size());
        cached = lid;
        edge_lid_touched.emplace_back(gid);
        out.edge_gids.emplace_back(gid);
        out.edge_ts.emplace_back(ts);
        return lid;
    };

    double root_start_time = omp_get_wtime();
    auto root_nodes_data = get_data_ptr<NodeIDType>(root_nodes);
    TimeStampType* root_ts_data = nullptr;
    if(root_ts.has_value()){
        root_ts_data = get_data_ptr<TimeStampType>(root_ts.value());
    }

    compact_frontier_lids_scratch.clear();
    compact_frontier_lids_scratch.reserve(root_nodes.size(0));
    compact_occurrence_to_frontier_scratch.resize(static_cast<size_t>(root_nodes.size(0)));
    compact_frontier_pos_touched_scratch.clear();
    compact_frontier_pos_touched_scratch.reserve(root_nodes.size(0));
    vector<int64_t>& frontier_lids = compact_frontier_lids_scratch;
    vector<int64_t>& occurrence_to_frontier = compact_occurrence_to_frontier_scratch;
    vector<int64_t>& frontier_pos_touched = compact_frontier_pos_touched_scratch;
    if(!compact_node_ids && frontier_pos_by_lid_dense.size() < estimated_nodes){
        frontier_pos_by_lid_dense.resize(estimated_nodes, -1);
    }
    if(compact_node_ids && frontier_pos_by_node_dense.size() < static_cast<size_t>(num_nodes)){
        frontier_pos_by_node_dense.resize(static_cast<size_t>(num_nodes), -1);
    }
    for(int64_t i = 0; i < root_nodes.size(0); i++){
        NodeIDType gid = root_nodes_data[i];
        TimeStampType ts = root_ts_data == nullptr ? 0 : root_ts_data[i];
        int64_t lid = add_node(gid, ts);
        out.root_gids[static_cast<size_t>(i)] = gid;
        out.root_ts[static_cast<size_t>(i)] = ts;
        out.root_lids[static_cast<size_t>(i)] = lid;
        if(compact_node_ids){
            AT_ASSERTM(gid >= 0 && gid < num_nodes, "root node gid out of range");
            int64_t& frontier_pos_ref = frontier_pos_by_node_dense[static_cast<size_t>(gid)];
            if(frontier_pos_ref < 0){
                int64_t frontier_pos = static_cast<int64_t>(frontier_lids.size());
                frontier_pos_ref = frontier_pos;
                frontier_pos_touched.emplace_back(gid);
                frontier_lids.emplace_back(lid);
                occurrence_to_frontier[static_cast<size_t>(i)] = frontier_pos;
            }
            else{
                occurrence_to_frontier[static_cast<size_t>(i)] = frontier_pos_ref;
            }
            continue;
        }
        int64_t& frontier_pos_ref = frontier_pos_by_lid_dense[static_cast<size_t>(lid)];
        if(frontier_pos_ref < 0){
            int64_t frontier_pos = static_cast<int64_t>(frontier_lids.size());
            frontier_pos_ref = frontier_pos;
            frontier_pos_touched.emplace_back(lid);
            frontier_lids.emplace_back(lid);
            occurrence_to_frontier[static_cast<size_t>(i)] = frontier_pos;
        }
        else{
            occurrence_to_frontier[static_cast<size_t>(i)] = frontier_pos_ref;
        }
    }
    if(compact_node_ids){
        for(int64_t gid : frontier_pos_touched){
            frontier_pos_by_node_dense[static_cast<size_t>(gid)] = -1;
        }
    } else {
        for(int64_t lid : frontier_pos_touched){
            frontier_pos_by_lid_dense[static_cast<size_t>(lid)] = -1;
        }
    }
    frontier_pos_touched.clear();
    out.node_layer_ptr.emplace_back(static_cast<int64_t>(out.node_gids.size()));
    compact_root_seconds += omp_get_wtime() - root_start_time;

    for(int64_t layer = 0; layer < static_cast<int64_t>(ret.size()); layer++){
        TemporalGraphBlock& block = ret[layer];
        NativeMFGBlock mfg;
        mfg.layer = layer;
        mfg.dst_begin = layer == 0 ? 0 : out.node_layer_ptr[layer];
        mfg.dst_end = out.node_layer_ptr[layer + 1];
        const bool has_next_layer = layer + 1 < static_cast<int64_t>(ret.size());

        const int64_t num_dst = static_cast<int64_t>(frontier_lids.size());
        const int64_t edge_count = static_cast<int64_t>(block.src_index.size());
        mfg.dst_lids.assign(frontier_lids.begin(), frontier_lids.end());
        mfg.src_lids.assign(frontier_lids.begin(), frontier_lids.end());
        mfg.csc_indptr.assign(num_dst + 1, 0);
        mfg.csc_indices.resize(edge_count);
        mfg.edge_lids.resize(edge_count);
        const bool has_delta_ts = !block.delta_ts.empty();
        if(has_delta_ts){
            mfg.delta_t.resize(edge_count);
        }

        double index_start_time = omp_get_wtime();
        compact_next_frontier_lids_scratch.clear();
        if(has_next_layer){
            compact_next_frontier_lids_scratch.reserve(static_cast<size_t>(block.sample_nodes.size()));
        }
        vector<int64_t>& next_frontier_lids = compact_next_frontier_lids_scratch;
        compact_next_occurrence_to_frontier_scratch.resize(static_cast<size_t>(edge_count));
        vector<int64_t>& next_occurrence_to_frontier = compact_next_occurrence_to_frontier_scratch;
        compact_local_node_index_scratch.clear();
        compact_local_node_id_index_scratch.clear();
        if(compact_node_ids){
            compact_local_node_id_index_scratch.reserve(compact_node_index_reserve(edge_count, layer));
        } else {
            compact_local_node_index_scratch.reserve(compact_node_index_reserve(edge_count, layer));
        }
        auto& local_node_index = compact_local_node_index_scratch;
        auto& local_node_id_index = compact_local_node_id_index_scratch;
        const bool has_sample_node_ts = !block.sample_nodes_ts.empty();
        NodeInstanceKey last_node_key{};
        NodeIDType last_node_id = 0;
        int64_t last_src_idx = -1;
        int64_t last_src_lid = -1;
        bool has_last_node = false;
        const size_t edge_touched_before = edge_lid_touched.size();

        for(int64_t j = 0; j < edge_count; j++){
            const int64_t prefetch_j = j + kCompactNodePrefetchDistance;
            if(prefetch_j < edge_count){
                const TimeStampType prefetch_ts = has_sample_node_ts ? block.sample_nodes_ts[prefetch_j] : 0;
                if(compact_node_ids){
                    local_node_id_index.prefetch(block.sample_nodes[prefetch_j]);
                } else {
                    local_node_index.prefetch(NodeInstanceKey{block.sample_nodes[prefetch_j], prefetch_ts});
                }
            }
            int64_t dst_occurrence = block.src_index[j];
            if(dst_occurrence >= 0 && dst_occurrence < static_cast<int64_t>(occurrence_to_frontier.size())){
                int64_t dst_pos = occurrence_to_frontier[static_cast<size_t>(dst_occurrence)];
                if(dst_pos >= 0 && dst_pos < num_dst){
                    mfg.csc_indptr[dst_pos + 1] += 1;
                }
            }

            const TimeStampType src_ts = has_sample_node_ts ? block.sample_nodes_ts[j] : 0;
            const NodeInstanceKey node_key{block.sample_nodes[j], src_ts};
            int64_t src_idx = -1;
            int64_t src_lid = -1;
            const bool last_node_match = compact_node_ids
                ? (has_last_node && node_key.node == last_node_id)
                : (has_last_node && node_key == last_node_key);
            if(last_node_match){
                compact_index_last_node_hits += 1.0;
                src_idx = last_src_idx;
                src_lid = last_src_lid;
            } else {
                compact_index_hash_lookups += 1.0;
                if(compact_node_ids){
                    auto node_res = local_node_id_index.emplace(node_key.node, -1);
                    src_idx = node_res.first->second;
                    if(node_res.second){
                        compact_index_hash_inserts += 1.0;
                        src_lid = add_node(node_key.node, node_key.ts);
                        src_idx = static_cast<int64_t>(mfg.src_lids.size());
                        node_res.first->second = src_idx;
                        mfg.src_lids.emplace_back(src_lid);
                        if(has_next_layer){
                            next_frontier_lids.emplace_back(src_lid);
                        }
                    } else {
                        compact_index_hash_hits += 1.0;
                        src_lid = mfg.src_lids[static_cast<size_t>(src_idx)];
                    }
                } else {
                    auto node_res = local_node_index.emplace(node_key, -1);
                    src_idx = node_res.first->second;
                    if(node_res.second){
                        compact_index_hash_inserts += 1.0;
                        src_lid = add_node(node_key.node, node_key.ts);
                        src_idx = static_cast<int64_t>(mfg.src_lids.size());
                        node_res.first->second = src_idx;
                        mfg.src_lids.emplace_back(src_lid);
                        if(has_next_layer){
                            next_frontier_lids.emplace_back(src_lid);
                        }
                    } else {
                        compact_index_hash_hits += 1.0;
                        src_lid = mfg.src_lids[static_cast<size_t>(src_idx)];
                    }
                }
                last_node_key = node_key;
                last_node_id = node_key.node;
                last_src_idx = src_idx;
                last_src_lid = src_lid;
                has_last_node = true;
            }
            const int64_t next_frontier_pos = src_idx - num_dst;
            next_occurrence_to_frontier[static_cast<size_t>(j)] = next_frontier_pos;
        }
        for(int64_t i = 0; i < num_dst; i++){
            mfg.csc_indptr[i + 1] += mfg.csc_indptr[i];
        }
        compact_index_edges += static_cast<double>(edge_count);
        const size_t unique_node_count = compact_node_ids ? local_node_id_index.size() : local_node_index.size();
        compact_index_unique_nodes += static_cast<double>(unique_node_count);
        compact_index_frontier_nodes += static_cast<double>(has_next_layer ? next_frontier_lids.size() : unique_node_count);
        compact_index_seconds += omp_get_wtime() - index_start_time;

        double fill_start_time = omp_get_wtime();
        double cursor_start_time = omp_get_wtime();
        compact_cursor_scratch = mfg.csc_indptr;
        compact_fill_cursor_seconds += omp_get_wtime() - cursor_start_time;
        vector<int64_t>& cursor = compact_cursor_scratch;
        double edge_loop_start_time = omp_get_wtime();
        if(compact_raw_edge_ids){
            compact_fill_raw_edge_rows += static_cast<double>(edge_count);
            for(int64_t j = 0; j < edge_count; j++){
                const int64_t prefetch_j = j + kCompactRawFillPrefetchDistance;
                if(prefetch_j < edge_count){
#if defined(__GNUC__)
                    __builtin_prefetch(static_cast<const void*>(&block.src_index[static_cast<size_t>(prefetch_j)]));
                    __builtin_prefetch(static_cast<const void*>(&block.eid[static_cast<size_t>(prefetch_j)]));
                    __builtin_prefetch(static_cast<const void*>(&next_occurrence_to_frontier[static_cast<size_t>(prefetch_j)]));
#endif
                    const int64_t prefetch_dst_occurrence = block.src_index[static_cast<size_t>(prefetch_j)];
                    if(prefetch_dst_occurrence >= 0 &&
                        prefetch_dst_occurrence < static_cast<int64_t>(occurrence_to_frontier.size())){
#if defined(__GNUC__)
                        __builtin_prefetch(static_cast<const void*>(
                            &occurrence_to_frontier[static_cast<size_t>(prefetch_dst_occurrence)]));
#endif
                        const int64_t prefetch_dst_pos =
                            occurrence_to_frontier[static_cast<size_t>(prefetch_dst_occurrence)];
                        if(prefetch_dst_pos >= 0 && prefetch_dst_pos < num_dst){
#if defined(__GNUC__)
                            __builtin_prefetch(static_cast<const void*>(&cursor[static_cast<size_t>(prefetch_dst_pos)]));
#endif
                        }
                    }
                }
                int64_t dst_occurrence = block.src_index[j];
                if(dst_occurrence < 0 || dst_occurrence >= static_cast<int64_t>(occurrence_to_frontier.size())){
                    continue;
                }
                int64_t dst_pos = occurrence_to_frontier[static_cast<size_t>(dst_occurrence)];
                if(dst_pos < 0 || dst_pos >= num_dst) continue;
                int64_t offset = cursor[dst_pos]++;
                mfg.csc_indices[offset] = next_occurrence_to_frontier[static_cast<size_t>(j)] + num_dst;
                mfg.edge_lids[offset] = static_cast<int64_t>(block.eid[j]);
                if(has_delta_ts){
                    mfg.delta_t[offset] = block.delta_ts[j];
                }
            }
        } else {
            compact_fill_mapped_edge_rows += static_cast<double>(edge_count);
            EdgeIDType last_edge_gid = 0;
            int64_t last_edge_lid = -1;
            bool has_last_edge = false;
            for(int64_t j = 0; j < edge_count; j++){
                const int64_t prefetch_j = j + kCompactMappedEdgePrefetchDistance;
                if(prefetch_j < edge_count){
                    const EdgeIDType prefetch_edge_gid = block.eid[prefetch_j];
                    if(prefetch_edge_gid >= 0 && prefetch_edge_gid < num_edges){
#if defined(__GNUC__)
                        __builtin_prefetch(static_cast<const void*>(&edge_lid_dense[static_cast<size_t>(prefetch_edge_gid)]));
#endif
                    }
                }
                int64_t dst_occurrence = block.src_index[j];
                if(dst_occurrence < 0 || dst_occurrence >= static_cast<int64_t>(occurrence_to_frontier.size())){
                    continue;
                }
                int64_t dst_pos = occurrence_to_frontier[static_cast<size_t>(dst_occurrence)];
                if(dst_pos < 0 || dst_pos >= num_dst) continue;
                int64_t offset = cursor[dst_pos]++;
                mfg.csc_indices[offset] = next_occurrence_to_frontier[static_cast<size_t>(j)] + num_dst;
                const EdgeIDType edge_gid = block.eid[j];
                int64_t edge_lid_val = -1;
                if(has_last_edge && edge_gid == last_edge_gid){
                    edge_lid_val = last_edge_lid;
                } else {
                    const TimeStampType edge_ts = has_sample_node_ts ? block.sample_nodes_ts[j] : 0;
                    edge_lid_val = add_edge(edge_gid, edge_ts);
                    last_edge_gid = edge_gid;
                    last_edge_lid = edge_lid_val;
                    has_last_edge = true;
                }
                mfg.edge_lids[offset] = edge_lid_val;
                if(has_delta_ts){
                    mfg.delta_t[offset] = block.delta_ts[j];
                }
            }
        }
        compact_fill_edge_loop_seconds += omp_get_wtime() - edge_loop_start_time;
        compact_index_unique_edges += static_cast<double>(edge_lid_touched.size() - edge_touched_before);
        deduplicate_compact_mfg_edges(mfg, has_delta_ts);
        compact_fill_seconds += omp_get_wtime() - fill_start_time;

        mfg.src_begin = 0;
        mfg.src_end = static_cast<int64_t>(out.node_gids.size());
        out.mfgs[layer] = std::move(mfg);
        out.node_layer_ptr.emplace_back(static_cast<int64_t>(out.node_gids.size()));
        out.edge_layer_ptr.emplace_back(static_cast<int64_t>(out.edge_gids.size()));
        if(has_next_layer){
            frontier_lids.swap(next_frontier_lids);
            occurrence_to_frontier.swap(next_occurrence_to_frontier);
        }
    }

    compact_total_seconds += omp_get_wtime() - total_start_time;
    count_sampled_local_remote(out);
    build_edge_read_layout(out);
    for(EdgeIDType gid : edge_lid_touched){
        edge_lid_dense[static_cast<size_t>(gid)] = -1;
    }
    return out;
}


NativeSamplingOutput ParallelSampler::get_sampling_output_parallel(th::Tensor root_nodes, optional<th::Tensor> root_ts)
{
    AT_ASSERTM(root_nodes.is_contiguous(), "root_nodes must be contiguous");
    AT_ASSERTM(root_nodes.dim() == 1, "root_nodes must be one-dimensional");
    if(root_ts.has_value()){
        AT_ASSERTM(root_ts.value().is_contiguous(), "root_ts must be contiguous");
        AT_ASSERTM(root_ts.value().dim() == 1, "root_ts must be one-dimensional");
        AT_ASSERTM(root_ts.value().size(0) == root_nodes.size(0), "root_ts size must match root_nodes");
    }

    py::gil_scoped_release release;

    struct FirstEdgeOccurrence
    {
        uint64_t pos;
        TimeStampType ts;
    };
    struct NodeOccurrenceEntry
    {
        NodeInstanceKey key;
        uint64_t pos;
    };
    struct EdgeOccurrenceEntry
    {
        EdgeIDType gid;
        uint64_t pos;
        TimeStampType ts;
    };

    auto update_node_first = [](phmap::flat_hash_map<NodeInstanceKey, uint64_t, NodeInstanceKeyHash>& first,
                                const NodeInstanceKey& key,
                                uint64_t pos) {
        auto it = first.find(key);
        if(it == first.end()){
            first.emplace(key, pos);
        }
        else if(pos < it->second){
            it->second = pos;
        }
    };
    auto update_edge_first = [](phmap::flat_hash_map<EdgeIDType, FirstEdgeOccurrence>& first,
                                EdgeIDType gid,
                                TimeStampType ts,
                                uint64_t pos) {
        auto it = first.find(gid);
        if(it == first.end()){
            first.emplace(gid, FirstEdgeOccurrence{pos, ts});
        }
        else if(pos < it->second.pos){
            it->second = FirstEdgeOccurrence{pos, ts};
        }
    };

    const int nthreads = std::max(1, threads);
    const int64_t num_roots = root_nodes.size(0);
    const int64_t num_layers_i64 = static_cast<int64_t>(ret.size());
    vector<uint64_t> layer_dst_base(ret.size(), 0);
    vector<uint64_t> layer_edge_base(ret.size(), 0);
    vector<uint64_t> layer_end_pos(ret.size(), static_cast<uint64_t>(num_roots));
    uint64_t next_pos = static_cast<uint64_t>(num_roots);
    size_t estimated_nodes = static_cast<size_t>(num_roots);
    size_t estimated_edges = 0;
    for(int64_t layer = 0; layer < num_layers_i64; layer++){
        const int64_t num_dst = layer == 0 ? num_roots : static_cast<int64_t>(ret[layer - 1].sample_nodes.size());
        layer_dst_base[layer] = next_pos;
        next_pos += static_cast<uint64_t>(num_dst);
        layer_edge_base[layer] = next_pos;
        next_pos += static_cast<uint64_t>(ret[layer].src_index.size());
        layer_end_pos[layer] = next_pos;
        estimated_nodes += ret[layer].sample_nodes.size();
        estimated_edges += ret[layer].eid.size();
    }

    auto root_nodes_data = get_data_ptr<NodeIDType>(root_nodes);
    TimeStampType* root_ts_data = nullptr;
    if(root_ts.has_value()){
        root_ts_data = get_data_ptr<TimeStampType>(root_ts.value());
    }

    phmap::flat_hash_map<NodeInstanceKey, uint64_t, NodeInstanceKeyHash> global_node_first;
    phmap::flat_hash_map<EdgeIDType, FirstEdgeOccurrence> global_edge_first;
    global_node_first.reserve(estimated_nodes);
    global_edge_first.reserve(estimated_edges);

    for(int64_t i = 0; i < num_roots; i++){
        TimeStampType ts = root_ts_data == nullptr ? 0 : root_ts_data[i];
        update_node_first(global_node_first, NodeInstanceKey{root_nodes_data[i], ts}, static_cast<uint64_t>(i));
    }

    vector<phmap::flat_hash_map<NodeInstanceKey, uint64_t, NodeInstanceKeyHash>> local_node_first(nthreads);
    vector<phmap::flat_hash_map<EdgeIDType, FirstEdgeOccurrence>> local_edge_first(nthreads);
    for(int tid = 0; tid < nthreads; tid++){
        local_node_first[tid].reserve(estimated_nodes / nthreads + 1);
        local_edge_first[tid].reserve(estimated_edges / nthreads + 1);
    }

#pragma omp parallel num_threads(nthreads)
    {
        const int tid = omp_get_thread_num();
        auto& node_first = local_node_first[tid];
        auto& edge_first = local_edge_first[tid];
#pragma omp for schedule(static)
        for(int64_t layer = 0; layer < num_layers_i64; layer++){
            if(layer > 0){
                TemporalGraphBlock& prev = ret[layer - 1];
                const uint64_t dst_base = layer_dst_base[layer];
                for(int64_t i = 0; i < static_cast<int64_t>(prev.sample_nodes.size()); i++){
                    TimeStampType ts = prev.sample_nodes_ts.empty() ? 0 : prev.sample_nodes_ts[i];
                    update_node_first(node_first, NodeInstanceKey{prev.sample_nodes[i], ts}, dst_base + static_cast<uint64_t>(i));
                }
            }

            TemporalGraphBlock& block = ret[layer];
            const int64_t num_dst = layer == 0 ? num_roots : static_cast<int64_t>(ret[layer - 1].sample_nodes.size());
            const uint64_t edge_base = layer_edge_base[layer];
            const int64_t edge_count = static_cast<int64_t>(block.src_index.size());
            for(int64_t j = 0; j < edge_count; j++){
                const int64_t dst_pos = block.src_index[j];
                if(dst_pos < 0 || dst_pos >= num_dst) continue;
                const TimeStampType src_ts = block.sample_nodes_ts.empty() ? 0 : block.sample_nodes_ts[j];
                const uint64_t pos = edge_base + static_cast<uint64_t>(j);
                update_node_first(node_first, NodeInstanceKey{block.sample_nodes[j], src_ts}, pos);
                update_edge_first(edge_first, block.eid[j], src_ts, pos);
            }
        }
    }

    for(int tid = 0; tid < nthreads; tid++){
        for(const auto& item : local_node_first[tid]){
            update_node_first(global_node_first, item.first, item.second);
        }
        for(const auto& item : local_edge_first[tid]){
            update_edge_first(global_edge_first, item.first, item.second.ts, item.second.pos);
        }
    }

    vector<NodeOccurrenceEntry> node_entries;
    node_entries.reserve(global_node_first.size());
    for(const auto& item : global_node_first){
        node_entries.push_back(NodeOccurrenceEntry{item.first, item.second});
    }
    std::sort(node_entries.begin(), node_entries.end(), [](const NodeOccurrenceEntry& a, const NodeOccurrenceEntry& b) {
        if(a.pos != b.pos) return a.pos < b.pos;
        if(a.key.node != b.key.node) return a.key.node < b.key.node;
        return a.key.ts < b.key.ts;
    });

    vector<EdgeOccurrenceEntry> edge_entries;
    edge_entries.reserve(global_edge_first.size());
    for(const auto& item : global_edge_first){
        edge_entries.push_back(EdgeOccurrenceEntry{item.first, item.second.pos, item.second.ts});
    }
    std::sort(edge_entries.begin(), edge_entries.end(), [](const EdgeOccurrenceEntry& a, const EdgeOccurrenceEntry& b) {
        if(a.pos != b.pos) return a.pos < b.pos;
        return a.gid < b.gid;
    });

    NativeSamplingOutput out;
    out.mfgs.resize(ret.size());
    out.node_gids.resize(node_entries.size());
    out.node_ts.resize(node_entries.size());
    out.edge_gids.resize(edge_entries.size());
    out.edge_ts.resize(edge_entries.size());
    out.node_layer_ptr.reserve(ret.size() + 2);
    out.edge_layer_ptr.reserve(ret.size() + 1);
    out.root_gids.reserve(num_roots);
    out.root_ts.reserve(num_roots);
    out.root_lids.reserve(num_roots);

    phmap::flat_hash_map<NodeInstanceKey, int64_t, NodeInstanceKeyHash> node_lid;
    phmap::flat_hash_map<EdgeIDType, int64_t> edge_lid;
    node_lid.reserve(node_entries.size());
    edge_lid.reserve(edge_entries.size());
    for(int64_t i = 0; i < static_cast<int64_t>(node_entries.size()); i++){
        const NodeInstanceKey& key = node_entries[i].key;
        node_lid.emplace(key, i);
        out.node_gids[i] = key.node;
        out.node_ts[i] = key.ts;
    }
    for(int64_t i = 0; i < static_cast<int64_t>(edge_entries.size()); i++){
        edge_lid.emplace(edge_entries[i].gid, i);
        out.edge_gids[i] = edge_entries[i].gid;
        out.edge_ts[i] = edge_entries[i].ts;
    }

    for(int64_t i = 0; i < num_roots; i++){
        const TimeStampType ts = root_ts_data == nullptr ? 0 : root_ts_data[i];
        const NodeIDType gid = root_nodes_data[i];
        out.root_gids.emplace_back(gid);
        out.root_ts.emplace_back(ts);
        out.root_lids.emplace_back(node_lid.at(NodeInstanceKey{gid, ts}));
    }

    out.node_layer_ptr.emplace_back(0);
    int64_t node_cursor = 0;
    auto append_node_layer_ptr = [&](uint64_t end_pos) {
        while(node_cursor < static_cast<int64_t>(node_entries.size()) && node_entries[node_cursor].pos < end_pos){
            node_cursor++;
        }
        out.node_layer_ptr.emplace_back(node_cursor);
    };
    append_node_layer_ptr(static_cast<uint64_t>(num_roots));
    for(int64_t layer = 0; layer < num_layers_i64; layer++){
        append_node_layer_ptr(layer_end_pos[layer]);
    }

    out.edge_layer_ptr.emplace_back(0);
    int64_t edge_cursor = 0;
    for(int64_t layer = 0; layer < num_layers_i64; layer++){
        while(edge_cursor < static_cast<int64_t>(edge_entries.size()) && edge_entries[edge_cursor].pos < layer_end_pos[layer]){
            edge_cursor++;
        }
        out.edge_layer_ptr.emplace_back(edge_cursor);
    }

    for(int64_t layer = 0; layer < num_layers_i64; layer++){
        TemporalGraphBlock& block = ret[layer];
        NativeMFGBlock mfg;
        mfg.layer = layer;
        mfg.dst_begin = layer == 0 ? 0 : out.node_layer_ptr[layer];
        mfg.dst_end = out.node_layer_ptr[layer + 1];

        const vector<NodeIDType>* frontier_nodes_ptr = nullptr;
        const vector<TimeStampType>* frontier_ts_ptr = nullptr;
        vector<NodeIDType> root_frontier_nodes;
        vector<TimeStampType> root_frontier_ts;
        if(layer == 0){
            root_frontier_nodes.assign(root_nodes_data, root_nodes_data + num_roots);
            root_frontier_ts.reserve(num_roots);
            for(int64_t i = 0; i < num_roots; i++){
                root_frontier_ts.emplace_back(root_ts_data == nullptr ? 0 : root_ts_data[i]);
            }
            frontier_nodes_ptr = &root_frontier_nodes;
            frontier_ts_ptr = &root_frontier_ts;
        }
        else{
            frontier_nodes_ptr = &ret[layer - 1].sample_nodes;
            frontier_ts_ptr = &ret[layer - 1].sample_nodes_ts;
        }
        const vector<NodeIDType>& frontier_nodes = *frontier_nodes_ptr;
        const vector<TimeStampType>& frontier_ts = *frontier_ts_ptr;
        const int64_t num_dst = static_cast<int64_t>(frontier_nodes.size());

        mfg.dst_lids.resize(num_dst);
#pragma omp parallel for num_threads(nthreads) schedule(static)
        for(int64_t i = 0; i < num_dst; i++){
            const TimeStampType ts = frontier_ts.empty() ? 0 : frontier_ts[i];
            mfg.dst_lids[i] = node_lid.at(NodeInstanceKey{frontier_nodes[i], ts});
        }

        const int64_t edge_count = static_cast<int64_t>(block.src_index.size());
        mfg.csc_indptr.assign(num_dst + 1, 0);
        vector<vector<int64_t>> thread_dst_counts(nthreads, vector<int64_t>(num_dst, 0));
#pragma omp parallel num_threads(nthreads)
        {
            const int tid = omp_get_thread_num();
            const int64_t begin = edge_count * tid / nthreads;
            const int64_t end = edge_count * (tid + 1) / nthreads;
            for(int64_t j = begin; j < end; j++){
                const int64_t dst_pos = block.src_index[j];
                if(dst_pos >= 0 && dst_pos < num_dst){
                    thread_dst_counts[tid][dst_pos] += 1;
                }
            }
        }
        for(int64_t dst = 0; dst < num_dst; dst++){
            int64_t count = 0;
            for(int tid = 0; tid < nthreads; tid++){
                count += thread_dst_counts[tid][dst];
            }
            mfg.csc_indptr[dst + 1] = count;
        }
        for(int64_t dst = 0; dst < num_dst; dst++){
            mfg.csc_indptr[dst + 1] += mfg.csc_indptr[dst];
        }
        for(int64_t dst = 0; dst < num_dst; dst++){
            int64_t base = mfg.csc_indptr[dst];
            for(int tid = 0; tid < nthreads; tid++){
                const int64_t count = thread_dst_counts[tid][dst];
                thread_dst_counts[tid][dst] = base;
                base += count;
            }
        }

        mfg.csc_indices.resize(edge_count);
        mfg.edge_lids.resize(edge_count);
        const bool has_delta_ts = !block.delta_ts.empty();
        if(has_delta_ts){
            mfg.delta_t.resize(edge_count);
        }
#pragma omp parallel num_threads(nthreads)
        {
            const int tid = omp_get_thread_num();
            const int64_t begin = edge_count * tid / nthreads;
            const int64_t end = edge_count * (tid + 1) / nthreads;
            for(int64_t j = begin; j < end; j++){
                const int64_t dst_pos = block.src_index[j];
                if(dst_pos < 0 || dst_pos >= num_dst) continue;
                const TimeStampType src_ts = block.sample_nodes_ts.empty() ? 0 : block.sample_nodes_ts[j];
                const int64_t offset = thread_dst_counts[tid][dst_pos]++;
                mfg.csc_indices[offset] = node_lid.at(NodeInstanceKey{block.sample_nodes[j], src_ts});
                mfg.edge_lids[offset] = edge_lid.at(block.eid[j]);
                if(has_delta_ts){
                    mfg.delta_t[offset] = block.delta_ts[j];
                }
            }
        }

        mfg.src_begin = 0;
        mfg.src_end = static_cast<int64_t>(out.node_gids.size());
        mfg.src_lids.resize(mfg.src_end - mfg.src_begin);
#pragma omp parallel for num_threads(nthreads) schedule(static)
        for(int64_t lid = mfg.src_begin; lid < mfg.src_end; lid++){
            mfg.src_lids[lid - mfg.src_begin] = lid;
        }
        out.mfgs[layer] = std::move(mfg);
    }

    count_sampled_local_remote(out);
    build_edge_read_layout(out);
    return out;
}

template <typename T>
void ParallelSampler::union_to_vector(vector<T> *p, vector<T> &to_vec){
    int sz = 0;
    for(int i=0 ;i<threads; i++){
        sz+=p[i].size();
    }
    to_vec.resize(sz);
    sz = 0;
    for(int i=0;i<threads;i++){
        copy(p[i].begin(),p[i].end(),to_vec.begin()+sz);
        sz+=p[i].size();
    }
}


void ParallelSampler::sample_unique(th::Tensor seed, th::Tensor seed_ts,
                                th::Tensor nid_mapper, th::Tensor eid_mapper,string out_device){
    th::Device device(torch::kCPU);
    if(out_device == "cpu"){}
    else device = th::Device(torch::kCUDA, out_device[0]-'0');
    vector<th::Tensor> eid_all_vec;
    vector<th::Tensor> dst_all_vec;  
    vector<th::Tensor> dst_ts_all_vec;  
    int hop = ret.size();
    vector<int> len(hop);
    for(int l = 0; l<hop;l++){
        int llen = ret[l].eid.end()-ret[l].eid.begin();
        len[l] = llen;
        eid_all_vec.emplace_back(torch::from_blob(ret[l].eid.data(), llen, torch::kInt64));
        dst_all_vec.emplace_back(torch::from_blob(ret[l].sample_nodes.data(), llen, torch::kInt64));
        dst_ts_all_vec.emplace_back(torch::from_blob(ret[l].sample_nodes_ts.data(),llen, torch::kFloat32));
    }
    th::Tensor dst = th::cat(dst_all_vec,0);
    th::Tensor dst_ts = th::cat(dst_ts_all_vec,0);
    th::Tensor eid_tensor = th::cat(eid_all_vec,0).to(eid_mapper.device());
    dist_eid = eid_mapper.index_select(0,eid_tensor).to(device);
    auto result0 = th::unique_dim(dist_eid,0,false,true,false);
    dist_eid = std::get<0>(result0);
    eid_inv = std::get<1>(result0);
    th::Tensor src_node = dst.to(nid_mapper.device());
    th::Tensor src_ts = th::cat({seed_ts.to(device),dst_ts.to(device)});
    th::Tensor nid_tensor = th::cat({seed.to(nid_mapper.device()),src_node});
    dist_nid = nid_mapper.index_select(0,nid_tensor).to(device);
    auto result1 = th::unique_dim(dist_nid,0,false,true,false);
    dist_nid = std::get<0>(result1);
    th::Tensor nid_inv = std::get<1>(result1);
    auto result2 = th::unique_dim(th::stack({nid_inv,src_ts.to(nid_inv.dtype())}),1,false,true,false);
    block_node_list = std::get<0>(result2);
    unq_id = std::get<1>(result2);
    th::Tensor array = th::arange(unq_id.size(0),torch::TensorOptions().dtype(unq_id.dtype()).device(device));
    th::Tensor first_index = th::empty(block_node_list.size(1),torch::TensorOptions().dtype(unq_id.dtype()).device(device));
    first_index = th::scatter_reduce(first_index, 0, unq_id, array, "amin",false);
    th::Tensor first_mask = th::zeros(unq_id.size(0),torch::TensorOptions().dtype(th::kBool).device(device));
    first_mask.index_fill_(0, first_index, 1);
    first_index = unq_id.masked_select(first_mask);
    first_block_id = th::empty(first_mask.size(0),torch::TensorOptions().dtype(unq_id.dtype()).device(device));
    array = th::arange(first_index.size(0),torch::TensorOptions().dtype(unq_id.dtype()).device(device));
    first_block_id.index_copy_(0,first_index,array);
    first_block_id = first_block_id.index_select(0,unq_id).contiguous();
}
