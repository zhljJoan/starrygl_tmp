#pragma once
#include <head.h>
#include <neighbors.h>
# include <output.h>

class ParallelTppRComputer
{
    public:
        TemporalNeighborBlock& tnb;
        NodeIDType num_nodes;
        EdgeIDType num_edges;
        int threads;
        int fanout;//k, width
        int num_layers;//depth
        int num_tpprs;//n_tpprs
        vector<float> alpha_list;
        vector<float> beta_list;
        // string policy;
        PPRListListDictType PPR_list;
        PPRListListDictType val_PPR_list;
        NormListType norm_list;
        NormListType val_norm_list;
        vector<vector<TemporalGraphBlock>> ret;

        ParallelTppRComputer(TemporalNeighborBlock& _tnb, NodeIDType _num_nodes, EdgeIDType _num_edges, int _threads, 
                        int _fanout, int _num_layers, int _num_tpprs, vector<float>& _alpha_list, vector<float>& _beta_list) :
                        tnb(_tnb), num_nodes(_num_nodes), num_edges(_num_edges), threads(_threads), 
                        fanout(_fanout), num_layers(_num_layers), num_tpprs(_num_tpprs), alpha_list(_alpha_list), beta_list(_beta_list)
        {
            omp_set_num_threads(_threads);
            ret.clear();
            ret = vector<vector<TemporalGraphBlock>>(_num_tpprs, vector<TemporalGraphBlock>());
        }

        void reset_ret() {
            for (int i = 0; i < num_tpprs; ++i) {
                ret[i].clear(); // 清空每个内部的 vector
            }
        }

        void reset_ret_i(int tppr_id) {
            ret[tppr_id].clear(); // 清空 tppr_id 处的 vector
        }
        
        void reset_tppr(){
            PPR_list = PPRListListDictType(num_tpprs, PPRListDictType(num_nodes));
            norm_list = NormListType(num_tpprs, vector<double>(num_nodes, 0.0));
        }

        void reset_val_tppr(){
            val_PPR_list = PPRListListDictType(num_tpprs, PPRListDictType(num_nodes));
            val_norm_list = NormListType(num_tpprs, vector<double>(num_nodes, 0.0));
        }
        
        py::tuple backup_tppr(){
            return py::make_tuple(this->PPR_list, this->norm_list);
        }

        void restore_tppr(PPRListListDictType& input_PPR_list, NormListType& input_norm_list){
            this->PPR_list = input_PPR_list;
            this->norm_list = input_norm_list;
        }

        void restore_val_tppr(PPRListListDictType& input_PPR_list, NormListType& input_norm_list){
            this->val_PPR_list = input_PPR_list;
            this->val_norm_list = input_norm_list;
        }

        PPRDictType compute_s1_s2(NodeIDType s1, NodeIDType s2, int tppr_id, EdgeIDType eid, TimeStampType ts);
        void get_pruned_topk(th::Tensor src_nodes, th::Tensor root_ts, int tppr_id);
        void extract_streaming_tppr(PPRDictType tppr_dict, TimeStampType current_ts, int index0, int position);
        void streaming_topk(th::Tensor src_nodes, th::Tensor root_ts, th::Tensor eids);
        void single_streaming_topk(th::Tensor src_nodes, th::Tensor root_ts, th::Tensor eids, int tppr_id);
        void streaming_topk_no_fake(th::Tensor src_nodes, th::Tensor root_ts, th::Tensor eids);
        void compute_val_tppr(th::Tensor src_nodes, th::Tensor dst_nodes, th::Tensor root_ts, th::Tensor eids);
};

PPRDictType ParallelTppRComputer :: compute_s1_s2(NodeIDType s1, NodeIDType s2, int tppr_id, EdgeIDType eid, TimeStampType ts){
    int alpha = alpha_list[tppr_id], beta = beta_list[tppr_id];
    vector<double> norm_list = this->norm_list[tppr_id];
    PPRListDictType PPR_list = this->PPR_list[tppr_id];
    PPRDictType t_s1_PPR= PPRDictType();
    PPRDictType updated_tppr= PPRDictType();
    float scala_s1, scala_s2;
    /***************s1 side*******************/
    if(norm_list[s1]==0){
        scala_s2 = 1-alpha;
    }
    else{
        t_s1_PPR = PPR_list[s1];
        double last_norm = norm_list[s1], new_norm;
        new_norm = last_norm*beta+beta;
        scala_s1 = last_norm/new_norm*beta;
        scala_s2 = beta/new_norm*(1-alpha);
        for (const auto& pair : t_s1_PPR)
            t_s1_PPR[pair.first] = pair.second*scala_s1;
    }
    /**************s2 side*******************/
    if(norm_list[s1]==0){
        t_s1_PPR[make_tuple(eid, s2, ts)] = alpha!=0 ? scala_s2*alpha : scala_s2;
    }
    else{
        PPRDictType s2_PPR= PPR_list[s2];
        for (const auto& pair : s2_PPR){
            if(t_s1_PPR.count(pair.first)==1)
                t_s1_PPR[pair.first] += pair.second*scala_s2;
            else
                t_s1_PPR[pair.first] = pair.second*scala_s2;
        }
        t_s1_PPR[make_tuple(eid, s2, ts)] = alpha!=0 ? scala_s2*alpha : scala_s2;
    }
    /*********exract the top-k items ********/
    int tppr_size = t_s1_PPR.size();
    if(tppr_size<=this->fanout)
        updated_tppr = t_s1_PPR;
    else{
        std::vector<std::pair<PPRKeyType, PPRValueType>> pairs;
        pairs.reserve(t_s1_PPR.size());
        // 提取键值对到 pairs 向量
        for (const auto& pair : t_s1_PPR) {
            pairs.emplace_back(pair.first, pair.second);
        }
        // 使用并行部分排序来获得前 this->fanout 个元素
        std::partial_sort(pairs.begin(), pairs.begin() + this->fanout, pairs.end(),
                        [](const auto& a, const auto& b) { return a.second > b.second; });
        // 将部分排序后的键值对添加到 updated_tppr
        for (size_t i = 0; i < this->fanout; ++i) {
            const auto& pair = pairs[i];
            updated_tppr[pair.first] = pair.second;
        }
    }
    return updated_tppr;
}

void ParallelTppRComputer :: get_pruned_topk(th::Tensor src_nodes, th::Tensor root_ts, int tppr_id){
    auto src_nodes_data = get_data_ptr<NodeIDType>(src_nodes);
    auto ts_data = get_data_ptr<TimeStampType>(root_ts);
    int64_t n_edges = src_nodes.size(0);
    float alpha = alpha_list[tppr_id], beta = beta_list[tppr_id];
    this->reset_ret_i(tppr_id);
    for(int i=0;i<n_edges;i++)
    {
        NodeIDType target_node =  src_nodes_data[i];
        TimeStampType target_timestamp =  ts_data[i];
        PPRDictType tppr_dict;
        /*******get dictionary of neighbors*********************/
        vector<tuple<NodeIDType, TimeStampType, PPRValueType>> query_list;
        query_list.push_back(make_tuple(target_node,  target_timestamp, 1.0));
        for(int depth=0;depth<this->num_layers;depth++)
        {
            vector<tuple<NodeIDType, TimeStampType, PPRValueType>> new_query_list;
            /*******traverse the query list*********************/
            for(int j=0;j<query_list.size();j++)
            {
                NodeIDType query_node = get<0>(query_list[j]);
                NodeIDType query_ts = get<1>(query_list[j]);
                NodeIDType query_weight = get<2>(query_list[j]);

                int end_index = lower_bound(tnb.timestamp[query_node].begin(), tnb.timestamp[query_node].end(), query_ts)-tnb.timestamp[query_node].begin();
                int n_ngh = end_index;
                if(n_ngh==0) continue;
                else
                {
                    double norm = beta/(1-beta)*(1-pow(beta, n_ngh));
                    double weight = alpha!=0 && depth==0 ? query_weight*(1-alpha)*beta/norm*alpha : query_weight*(1-alpha)*beta/norm;

                    for(int z=0;z<min(this->fanout, n_ngh);z++){
                        EdgeIDType eid = tnb.eid[query_node][end_index-z-1];
                        NodeIDType node = tnb.neighbors[query_node][end_index-z-1];
                        // the timestamp here is a neighbor timestamp, 
                        // so that it is indeed a temporal random walk
                        TimeStampType timestamp =  tnb.timestamp[query_node][end_index-z-1];
                        PPRKeyType state = make_tuple(eid, node, timestamp);
                        // update dict
                        if(tppr_dict.count(state)==1)
                            tppr_dict[state] = tppr_dict[state]+weight;
                        else
                            tppr_dict[state] = weight;
                        // update query list
                        tuple<NodeIDType, TimeStampType, PPRValueType> new_query = make_tuple(node, timestamp, weight);
                        new_query_list.push_back(new_query);
                        // update weight
                        weight = weight*beta;
                    }
                }
            }
            if(new_query_list.empty()) break;
            else query_list = new_query_list;
        }
        /*****sort and get the top-k neighbors********/
        int tppr_size = tppr_dict.size();
        if(tppr_size==0) continue;

        TimeStampType current_timestamp = ts_data[i];

        PPRDictType updated_tppr= PPRDictType();            
        if(tppr_size<=this->fanout)
            updated_tppr = tppr_dict;
        else
        {
            std::vector<std::pair<PPRKeyType, PPRValueType>> pairs;
            pairs.reserve(tppr_dict.size());
            // 提取键值对到 pairs 向量
            for (const auto& pair : tppr_dict) {
                pairs.emplace_back(pair.first, pair.second);
            }
            // 使用并行部分排序来获得前 this->fanout 个元素
            std::partial_sort(pairs.begin(), pairs.begin() + this->fanout, pairs.end(),
                            [](const auto& a, const auto& b) { return a.second > b.second; });
            // 将部分排序后的键值对添加到 updated_tppr
            for (size_t i = 0; i < this->fanout; ++i) {
                const auto& pair = pairs[i];
                updated_tppr[pair.first] = pair.second;
            }
        }
        // this->PPR_list[tppr_id][target_node] = updated_tppr;
        extract_streaming_tppr(updated_tppr, current_timestamp, tppr_id, i);
    }
}

// category=0-src  category=1-dst category=2-fake
void ParallelTppRComputer :: extract_streaming_tppr(PPRDictType tppr_dict, TimeStampType current_ts, int index0, int position){
    ret[index0][position] = TemporalGraphBlock();
    if(!tppr_dict.empty()){
        ret[index0][position].sample_nodes.resize(this->fanout);
        ret[index0][position].eid.resize(this->fanout);
        ret[index0][position].sample_nodes_ts.resize(this->fanout);
        ret[index0][position].e_weights.resize(this->fanout);
        ret[index0][position].delta_ts.resize(this->fanout);
        int j=0;
        for (const auto& pair : tppr_dict){
            auto tuple = pair.first;
            auto weight = pair.second;
            EdgeIDType eid = get<0>(tuple);
            NodeIDType dst = get<1>(tuple);
            TimeStampType ets = get<2>(tuple);

            ret[index0][position].sample_nodes[j]=dst;
            ret[index0][position].eid[j]=eid;
            ret[index0][position].sample_nodes_ts[j]=ets;
            ret[index0][position].e_weights[j]=weight;
            ret[index0][position].delta_ts[j]=current_ts-ets;

            j++;
        }
    }
}

void ParallelTppRComputer :: streaming_topk(th::Tensor src_nodes, th::Tensor root_ts, th::Tensor eids){
    auto src_nodes_data = get_data_ptr<NodeIDType>(src_nodes);
    auto ts_data = get_data_ptr<TimeStampType>(root_ts);
    auto eids_data = get_data_ptr<EdgeIDType>(eids);
    int n_nodes = src_nodes.size(0);
    int n_edges = num_nodes/3;
    this->reset_ret();
    for(int index0=0;index0<num_tpprs;index0++){
        int alpha = alpha_list[index0], beta = beta_list[index0];
        ret[index0].resize(n_nodes);
        vector<double>& norm_list = this->norm_list[index0];
        PPRListDictType& PPR_list = this->PPR_list[index0];
        for(int i=0; i<n_edges; i++){
            NodeIDType src = src_nodes_data[i];
            NodeIDType dst = src_nodes_data[i+n_edges];
            NodeIDType fake = src_nodes_data[i+(n_edges<<1)];
            TimeStampType ts = ts_data[i];
            EdgeIDType eid = eids_data[i];

            /******first extract the top-k neighbors and fill the list******/
            extract_streaming_tppr(PPR_list[src], ts, index0, i);
            extract_streaming_tppr(PPR_list[dst], ts, index0, i+n_edges);
            extract_streaming_tppr(PPR_list[fake], ts, index0, i+(n_edges<<1));

            /******then update the PPR values here**************************/
            PPR_list[src] = compute_s1_s2(src, dst, index0, eid, ts);
            norm_list[src] = norm_list[src]*beta+beta;
            if(src!=dst){
                PPR_list[dst] = compute_s1_s2(dst, src, index0, eid, ts);
                norm_list[dst] = norm_list[dst]*beta+beta;
            }
        }
    }
}

void ParallelTppRComputer :: single_streaming_topk(th::Tensor src_nodes, th::Tensor root_ts, th::Tensor eids, int tppr_id){
    auto src_nodes_data = get_data_ptr<NodeIDType>(src_nodes);
    auto ts_data = get_data_ptr<TimeStampType>(root_ts);
    auto eids_data = get_data_ptr<EdgeIDType>(eids);
    int n_nodes = src_nodes.size(0);
    int n_edges = num_nodes/3;
    this->reset_ret_i(tppr_id);
    int alpha = alpha_list[tppr_id], beta = beta_list[tppr_id];
    ret[tppr_id].resize(n_nodes);
    vector<double>& norm_list = this->norm_list[tppr_id];
    PPRListDictType& PPR_list = this->PPR_list[tppr_id];
    for(int i=0; i<n_edges; i++){
        NodeIDType src = src_nodes_data[i];
        NodeIDType dst = src_nodes_data[i+n_edges];
        NodeIDType fake = src_nodes_data[i+(n_edges<<1)];
        TimeStampType ts = ts_data[i];
        EdgeIDType eid = eids_data[i];

        /******first extract the top-k neighbors and fill the list******/
        extract_streaming_tppr(PPR_list[src], ts, tppr_id, i);
        extract_streaming_tppr(PPR_list[dst], ts, tppr_id, i+n_edges);
        extract_streaming_tppr(PPR_list[fake], ts, tppr_id, i+(n_edges<<1));

        /******then update the PPR values here**************************/
        PPR_list[src] = compute_s1_s2(src, dst, tppr_id, eid, ts);
        norm_list[src] = norm_list[src]*beta+beta;
        if(src!=dst){
            PPR_list[dst] = compute_s1_s2(dst, src, tppr_id, eid, ts);
            norm_list[dst] = norm_list[dst]*beta+beta;
        }
    }
}

void ParallelTppRComputer :: streaming_topk_no_fake(th::Tensor src_nodes, th::Tensor root_ts, th::Tensor eids){
    auto src_nodes_data = get_data_ptr<NodeIDType>(src_nodes);
    auto ts_data = get_data_ptr<TimeStampType>(root_ts);
    auto eids_data = get_data_ptr<EdgeIDType>(eids);
    int n_nodes = src_nodes.size(0);
    int n_edges = num_nodes/2;
    this->reset_ret();
    for(int index0=0;index0<num_tpprs;index0++){
        int alpha = alpha_list[index0], beta = beta_list[index0];
        ret[index0].resize(n_nodes);
        vector<double>& norm_list = this->norm_list[index0];
        PPRListDictType& PPR_list = this->PPR_list[index0];
        for(int i=0; i<n_edges; i++){
            NodeIDType src = src_nodes_data[i];
            NodeIDType dst = src_nodes_data[i+n_edges];
            TimeStampType ts = ts_data[i];
            EdgeIDType eid = eids_data[i];

            /******first extract the top-k neighbors and fill the list******/
            extract_streaming_tppr(PPR_list[src], ts, index0, i);
            extract_streaming_tppr(PPR_list[dst], ts, index0, i+n_edges);

            /******then update the PPR values here**************************/
            PPR_list[src] = compute_s1_s2(src, dst, index0, eid, ts);
            norm_list[src] = norm_list[src]*beta+beta;
            if(src!=dst){
                PPR_list[dst] = compute_s1_s2(dst, src, index0, eid, ts);
                norm_list[dst] = norm_list[dst]*beta+beta;
            }
        }
    }
}

void ParallelTppRComputer :: compute_val_tppr(th::Tensor src_nodes, th::Tensor dst_nodes, th::Tensor root_ts, th::Tensor eids){
    auto src_nodes_data = get_data_ptr<NodeIDType>(src_nodes);
    auto dst_nodes_data = get_data_ptr<NodeIDType>(dst_nodes);
    auto ts_data = get_data_ptr<TimeStampType>(root_ts);
    auto eids_data = get_data_ptr<EdgeIDType>(eids);
    int n_edges = src_nodes.size(0);
    for(int index0=0;index0<num_tpprs;index0++){
        int alpha = alpha_list[index0], beta = beta_list[index0];
        vector<double>& norm_list = this->norm_list[index0];
        PPRListDictType& PPR_list = this->PPR_list[index0];
        for(int i=0; i<n_edges; i++){
            NodeIDType src = src_nodes_data[i];
            NodeIDType dst = dst_nodes_data[i];
            TimeStampType ts = ts_data[i];
            EdgeIDType eid = eids_data[i];
            PPR_list[src] = compute_s1_s2(src, dst, index0, eid, ts);
            norm_list[src] = norm_list[src]*beta+beta;
            if(src!=dst){
                PPR_list[dst] = compute_s1_s2(dst, src, index0, eid, ts);
                norm_list[dst] = norm_list[dst]*beta+beta;                
            }
        }
    }
    this->val_norm_list.assign(this->norm_list.begin(), this->norm_list.end());
    this->val_PPR_list.assign(this->PPR_list.begin(), this->PPR_list.end());
}