#pragma once
#include<head.h>

/* 
 * NeighborSampler Utils
 */
class TemporalNeighborBlock
{
    public:
        vector<vector<NodeIDType>> neighbors;
        vector<vector<TimeStampType>> timestamp;
        vector<vector<EdgeIDType>> eid;
        vector<vector<EdgeIDType>> dist_eid;
        vector<vector<WeightType>> edge_weight;
        vector<phmap::parallel_flat_hash_map<NodeIDType, int64_t>> inverted_index;
        vector<int64_t> deg;
        vector<phmap::parallel_flat_hash_set<NodeIDType>> neighbors_set;
        vector<int16_t> part; 
        bool with_eid = false;
        bool with_dist_eid = false;
        bool weighted = false;
        bool with_timestamp = false;
        bool with_historical = false;
        
        TemporalNeighborBlock(){}
        // TemporalNeighborBlock(const TemporalNeighborBlock &tnb);
        TemporalNeighborBlock(vector<vector<NodeIDType>>& neighbors, 
                              vector<int64_t> &deg):
                              neighbors(neighbors), deg(deg){}
        TemporalNeighborBlock(vector<vector<NodeIDType>>& neighbors, 
                              vector<vector<WeightType>>& edge_weight, 
                              vector<vector<EdgeIDType>>& eid,
                              vector<vector<EdgeIDType>>& dist_eid,
                              vector<int64_t> &deg):
                              neighbors(neighbors), edge_weight(edge_weight),eid(eid),dist_eid(dist_eid),deg(deg)
                              { this->with_eid=true;
                                this->weighted=true; }
        TemporalNeighborBlock(vector<vector<NodeIDType>>& neighbors, 
                              vector<vector<WeightType>>& edge_weight,
                              vector<vector<TimeStampType>>& timestamp,
                              vector<vector<EdgeIDType>>& eid, 
                              vector<vector<EdgeIDType>>& dist_eid, 
                              vector<int64_t> &deg):
                              neighbors(neighbors), edge_weight(edge_weight), timestamp(timestamp),eid(eid),dist_eid(dist_eid),deg(deg)
                              { this->with_eid=true;
                                this->weighted=true; 
                                this->with_timestamp=true;}
        
        py::array get_node_neighbor(NodeIDType node_id){
            return vec2npy(neighbors[node_id]);
        }
        py::array get_node_neighbor_timestamp(NodeIDType node_id){
            return vec2npy(timestamp[node_id]);
        }
        int64_t get_node_deg(NodeIDType node_id){
            return deg[node_id];
        }
        bool empty(){
            return this->deg.empty();
        }
        void update_edge_weight(th::Tensor row_or_eid, th::Tensor col, th::Tensor edge_weight);
        void update_node_weight(th::Tensor nid, th::Tensor node_weight);
        void update_all_node_weight(th::Tensor node_weight);
        int64_t update_neighbors_with_time(th::Tensor row, th::Tensor col, th::Tensor time, th::Tensor eid, int is_distinct, std::optional<th::Tensor> edge_weight);

        
        std::string serialize() const {
            std::ostringstream oss;
            oss << with_eid << " " << with_dist_eid<<" "<<weighted << " " << with_timestamp << " ";

            auto serializeVecVec = [&oss](const auto& vecVec) {
                for (const auto& vec : vecVec) {
                    oss << vec.size() << " ";
                    for (const auto& elem : vec) {
                        oss << elem << " ";
                    }
                }
                oss << "|"; 
            };

            serializeVecVec(neighbors);
            serializeVecVec(timestamp);
            serializeVecVec(eid);
            serializeVecVec(dist_eid);
            serializeVecVec(edge_weight);

            oss << deg.size() << " ";
            for (const auto& d : deg) {
                oss << d << " ";
            }
            oss << "|";

            for (const auto& map : inverted_index) {
                oss << map.size() << " ";
                for (const auto& [key, value] : map) {
                    oss << key << " " << value << " ";
                }
            }
            oss << "|";

            for (const auto& set : neighbors_set) {
                oss << set.size() << " ";
                for (const auto& elem : set) {
                    oss << elem << " ";
                }
            }
            oss << "|";

            return oss.str();
        }

        static TemporalNeighborBlock deserialize(const std::string& s) {
            std::istringstream iss(s);
            TemporalNeighborBlock tnb;


            iss >> tnb.with_eid >>tnb.with_dist_eid>> tnb.weighted >> tnb.with_timestamp;

            auto deserializeVecLong = [&iss](vector<vector<int64_t>>& vecVec) {
                std::string segment;
                std::getline(iss, segment, '|');
                std::istringstream vec_iss(segment);
                while (!vec_iss.eof()) {
                    size_t vec_size;
                    vec_iss >> vec_size;
                    if (vec_iss.eof()) break; 
                    vector<int64_t> vec(vec_size);
                    for (size_t i = 0; i < vec_size; ++i) {
                        vec_iss >> vec[i];
                    }
                    vecVec.push_back(vec);
                }
            };

            
            auto deserializeVecFloat = [&iss](vector<vector<float>>& vecVec) {
                std::string segment;
                std::getline(iss, segment, '|');
                std::istringstream vec_iss(segment);
                while (!vec_iss.eof()) {
                    size_t vec_size;
                    vec_iss >> vec_size;
                    if (vec_iss.eof()) break;  // ��ֹ����Ŀհ�
                    vector<float> vec(vec_size);
                    for (size_t i = 0; i < vec_size; ++i) {
                        vec_iss >> vec[i];
                    }
                    vecVec.push_back(vec);
                }
            };

            deserializeVecLong(tnb.neighbors);
            deserializeVecLong(tnb.timestamp);
            //deserializeVecLong(tnb.timestamp);
            deserializeVecLong(tnb.eid);
            deserializeVecLong(tnb.dist_eid);
            deserializeVecFloat(tnb.edge_weight);

            std::string segment;
            // �����л� vector<int64_t> ���ͳ�Ա
            segment="";
            std::getline(iss, segment, '|');
            std::istringstream vec_iss(segment);
            size_t vec_size;
            vec_iss >> vec_size;
            tnb.deg.resize(vec_size);
            for (size_t i = 0; i < vec_size; ++i) {
                vec_iss >> tnb.deg[i];
            }

            // �����л� inverted_index
            segment="";
            std::getline(iss, segment, '|');
            std::istringstream map_iss(segment);
            while (!map_iss.eof()) {
                size_t map_size;
                map_iss >> map_size;
                if (map_iss.eof()) break;
                phmap::parallel_flat_hash_map<NodeIDType, int64_t> map;
                for (size_t i = 0; i < map_size; ++i) {
                    NodeIDType key;
                    int64_t value;
                    map_iss >> key >> value;
                    map[key] = value;
                }
                tnb.inverted_index.push_back(map);
            }

            // �����л� neighbors_set
            std::getline(iss, segment, '|');
            std::istringstream set_iss(segment);
            while (!set_iss.eof()) {
                size_t set_size;
                set_iss >> set_size;
                if (set_iss.eof()) break;
                phmap::parallel_flat_hash_set<NodeIDType> set;
                for (size_t i = 0; i < set_size; ++i) {
                    NodeIDType elem;
                    set_iss >> elem;
                    set.insert(elem);
                }
                tnb.neighbors_set.push_back(set);
            }

            return tnb;
        }
};

TemporalNeighborBlock& get_neighbors(
        string graph_name, th::Tensor row, th::Tensor col, int64_t num_nodes, int is_distinct, optional<th::Tensor> eid, optional<th::Tensor> dist_eid, optional<th::Tensor> edge_weight, optional<th::Tensor> time)
{   //row��col��time��time�������У���ʱ����ĵ�ʱ������
    auto src = get_data_ptr<NodeIDType>(row);
    auto dst = get_data_ptr<NodeIDType>(col);
    EdgeIDType* eid_ptr = eid ? get_data_ptr<EdgeIDType>(eid.value()) : nullptr;
    EdgeIDType* dist_eid_ptr = dist_eid ? get_data_ptr<EdgeIDType>(dist_eid.value()) : nullptr;
    WeightType* ew = edge_weight ? get_data_ptr<WeightType>(edge_weight.value()) : nullptr;
    TimeStampType* t = time ? get_data_ptr<TimeStampType>(time.value()) : nullptr;
    int64_t edge_num = row.size(0);
    static phmap::parallel_flat_hash_map<string, TemporalNeighborBlock> tnb_map;
    if(tnb_map.count(graph_name)==1)
        return tnb_map[graph_name];
    tnb_map[graph_name] = TemporalNeighborBlock();
    TemporalNeighborBlock& tnb = tnb_map[graph_name];

    double start_time = omp_get_wtime();
    //��ʼ��
    tnb.neighbors.resize(num_nodes);
    tnb.deg.resize(num_nodes, 0);
        //��ʼ��optional���
    tnb.with_eid = eid.has_value();
    tnb.with_dist_eid = dist_eid.has_value();
    tnb.weighted = edge_weight.has_value();
    tnb.with_timestamp = time.has_value();
    if (tnb.with_eid) tnb.eid.resize(num_nodes);
    if (tnb.with_dist_eid) tnb.dist_eid.resize(num_nodes);
    if (tnb.weighted) {
        tnb.edge_weight.resize(num_nodes);
        tnb.inverted_index.resize(num_nodes);
    }
    if (tnb.with_timestamp) tnb.timestamp.resize(num_nodes);
        
    //����, �����ж��Ƴ�ѭ���Ż�ִ��Ч��
    for(int64_t i=0; i<edge_num; i++){
        //����ڵ��ھ�
        tnb.neighbors[dst[i]].emplace_back(src[i]);
    }
    //�����eid������
    if(tnb.with_eid)
        for(int64_t i=0; i<edge_num; i++){
            tnb.eid[dst[i]].emplace_back(eid_ptr[i]);
        }
    if(tnb.with_dist_eid)
        for(int64_t i=0; i<edge_num;i++){
            tnb.dist_eid[dst[i]].emplace_back(dist_eid_ptr[i]);
        }
    //�����Ȩ����Ϣ������ڵ����ھӱߵ�Ȩ�غͷ�������
    if(tnb.weighted)
        for(int64_t i=0; i<edge_num; i++){
            tnb.edge_weight[dst[i]].emplace_back(ew[i]);
            if(tnb.with_eid) tnb.inverted_index[dst[i]][eid_ptr[i]]=tnb.neighbors[dst[i]].size()-1;
            else tnb.inverted_index[dst[i]][src[i]]=tnb.neighbors[dst[i]].size()-1;
        }
    //�����ʱ����Ϣ������ڵ����ھӱߵ�ʱ��
    if(tnb.with_timestamp)
        for(int64_t i=0; i<edge_num; i++){
            tnb.timestamp[dst[i]].emplace_back(t[i]);
        }
        
    if(is_distinct){
        for(int64_t i=0; i<num_nodes; i++){
            //�ռ�ȥ���ھ�
            phmap::parallel_flat_hash_set<NodeIDType> temp_s;
            temp_s.insert(tnb.neighbors[i].begin(), tnb.neighbors[i].end());
            tnb.neighbors_set.emplace_back(temp_s);
        }
    }
    for(int64_t i=0; i<num_nodes; i++){
        //�ռ����߽ڵ��
        tnb.deg[i] = tnb.neighbors[i].size();
    }
    double end_time = omp_get_wtime();
    cout<<"get_neighbors consume: "<<end_time-start_time<<"s"<<endl;
    return tnb;
}

void TemporalNeighborBlock::update_edge_weight(
        th::Tensor row_or_eid, th::Tensor col, th::Tensor edge_weight){
    AT_ASSERTM(this->weighted, "This Graph has no edge weight infomation");
    auto dst = get_data_ptr<NodeIDType>(col);
    WeightType* ew = get_data_ptr<WeightType>(edge_weight);
    NodeIDType* src;
    EdgeIDType* eid_ptr;
    if(this->with_eid) eid_ptr = get_data_ptr<EdgeIDType>(row_or_eid);
    else src = get_data_ptr<NodeIDType>(row_or_eid);
    
    int64_t edge_num = col.size(0);

    for(int64_t i=0; i<edge_num; i++){
        //�޸Ľڵ����ھӱߵ�Ȩ��
		int index;
        if(this->with_eid){
            AT_ASSERTM(this->inverted_index[dst[i]].count(eid_ptr[i])==1, "Unexist Eid --> Col: "+to_string(eid_ptr[i])+"-->"+to_string(dst[i]));
            index = this->inverted_index[dst[i]][eid_ptr[i]];
        }
        else{
            AT_ASSERTM(this->inverted_index[dst[i]].count(src[i])==1, "Unexist Edge Index: "+to_string(src[i])+", "+to_string(dst[i]));
            index = this->inverted_index[dst[i]][src[i]];
        }
        this->edge_weight[dst[i]][index] = ew[i];
    }
}

void TemporalNeighborBlock:: update_node_weight(th::Tensor nid, th::Tensor node_weight){
    AT_ASSERTM(this->weighted, "This Graph has no edge weight infomation");
    auto dst = get_data_ptr<NodeIDType>(nid);
    WeightType* nw = get_data_ptr<WeightType>(node_weight);
    
    int64_t node_num = nid.size(0);

    for(int64_t i=0; i<node_num; i++){
        //�޸Ľڵ����ھӱߵ�Ȩ��
        AT_ASSERTM(dst[i]<this->deg.size(), "Unexist Node Index: "+to_string(dst[i]));
        if(this->inverted_index[dst[i]].empty())
            return;
        for(auto index : this->inverted_index[dst[i]]){
            this->edge_weight[dst[i]][index.second] = nw[i];
        }
    }
}

void TemporalNeighborBlock:: update_all_node_weight(th::Tensor node_weight){
    AT_ASSERTM(this->weighted, "This Graph has no edge weight infomation");
    WeightType* nw = get_data_ptr<WeightType>(node_weight);
    int64_t node_num = node_weight.size(0);
    AT_ASSERTM(node_num==this->neighbors.size(), "The tensor node_weight size is not suitable node number.");

    for(int64_t i=0; i<node_num; i++){
        //�޸Ľڵ����ھӱߵ�Ȩ��
        for(int j=0; j<this->neighbors[i].size();j++){
            this->edge_weight[i][j] = nw[this->neighbors[i][j]];
        }
    }
}

int64_t TemporalNeighborBlock::update_neighbors_with_time(
        th::Tensor row, th::Tensor col, th::Tensor time,th::Tensor eid, int is_distinct, std::optional<th::Tensor> edge_weight){
        //row��col��time��time�������У���ʱ����ĵ�ʱ������
    AT_ASSERTM(this->empty(), "Empty TemporalNeighborBlock, please use get_neighbors_with_time");
    AT_ASSERTM(this->with_timestamp == true, "This Graph has no time infomation!");
    auto src = get_data_ptr<NodeIDType>(row);
    auto dst = get_data_ptr<NodeIDType>(col);
    auto eid_ptr = get_data_ptr<EdgeIDType>(eid);
    auto t = get_data_ptr<TimeStampType>(time);
    WeightType* ew = edge_weight ? get_data_ptr<WeightType>(edge_weight.value()) : nullptr;
    int64_t edge_num = row.size(0);
    int64_t num_nodes = this->neighbors.size();

    //����optional��ֵ
    if(edge_weight.has_value()){
        AT_ASSERTM(this->weighted == true, "This Graph has no edge weight");
    }
    if(this->weighted){
        AT_ASSERTM(edge_weight.has_value(), "This Graph need edge weight");
    }

    // double start_time = omp_get_wtime();
    if(is_distinct){
        for(int64_t i=0; i<edge_num; i++){
            //������½ڵ�
            if(dst[i]>=num_nodes){
                num_nodes = dst[i]+1;
                this->neighbors.resize(num_nodes);
                this->deg.resize(num_nodes, 0);
                this->eid.resize(num_nodes);
                this->timestamp.resize(num_nodes);
                    //��ʼ��optional���
                if (this->weighted) {
                    this->edge_weight.resize(num_nodes);
                    this->inverted_index.resize(num_nodes);
                }
            }
            this->neighbors[dst[i]].emplace_back(src[i]);
            this->eid[dst[i]].emplace_back(eid_ptr[i]);
            this->timestamp[dst[i]].emplace_back(t[i]);
            if(this->weighted){
                this->edge_weight[dst[i]].emplace_back(ew[i]);
                if(this->with_eid) this->inverted_index[dst[i]][eid_ptr[i]]=this->neighbors[dst[i]].size()-1;
                else this->inverted_index[dst[i]][src[i]]=this->neighbors[dst[i]].size()-1;
            }
            
            this->neighbors_set[dst[i]].insert(src[i]);
            this->deg[dst[i]]=this->neighbors_set[dst[i]].size();
        }
    }
    else{
        for(int64_t i=0; i<edge_num; i++){
            this->neighbors[dst[i]].emplace_back(src[i]);
            this->eid[dst[i]].emplace_back(eid_ptr[i]);
            this->timestamp[dst[i]].emplace_back(t[i]);
            if(this->weighted){
                this->edge_weight[dst[i]].emplace_back(ew[i]);
                this->inverted_index[dst[i]][src[i]]=this->neighbors[dst[i]].size()-1;
            }

            this->deg[dst[i]]=this->neighbors[dst[i]].size();
        }
    }
    return num_nodes;
}