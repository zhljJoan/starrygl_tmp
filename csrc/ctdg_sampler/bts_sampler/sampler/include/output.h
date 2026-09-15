#pragma once
#include <head.h>

class TemporalGraphBlock
{
    public:
        vector<NodeIDType> row;
        vector<NodeIDType> col;
        vector<EdgeIDType> eid;
        vector<TimeStampType> delta_ts;
        vector<int64_t> src_index;
        vector<NodeIDType> sample_nodes;
        vector<TimeStampType> sample_nodes_ts;
        vector<float> sample_weight; 
        vector<WeightType> e_weights;
        double sample_time = 0;
        double tot_time = 0;
        int64_t sample_edge_num = 0;

        TemporalGraphBlock(){}
        // TemporalGraphBlock(const TemporalGraphBlock &tgb);
        TemporalGraphBlock(vector<NodeIDType> &_row, vector<NodeIDType> &_col,
                           vector<NodeIDType> &_sample_nodes):
                           row(_row), col(_col), sample_nodes(_sample_nodes){}
        TemporalGraphBlock(vector<NodeIDType> &_row, vector<NodeIDType> &_col,
                           vector<NodeIDType> &_sample_nodes,
                           vector<TimeStampType> &_sample_nodes_ts):
                           row(_row), col(_col), sample_nodes(_sample_nodes),
                           sample_nodes_ts(_sample_nodes_ts){}
};

class NativeMFGBlock
{
    public:
        int64_t layer = 0;
        vector<NodeIDType> dst_lids;
        vector<NodeIDType> src_lids;
        vector<int64_t> csc_indptr;
        vector<NodeIDType> csc_indices;
        vector<EdgeIDType> edge_lids;
        vector<TimeStampType> delta_t;
        int64_t src_begin = 0;
        int64_t src_end = 0;
        int64_t dst_begin = 0;
        int64_t dst_end = 0;

        NativeMFGBlock(){}
};

class NativeSamplingOutput
{
    public:
        vector<NativeMFGBlock> mfgs;
        vector<NodeIDType> node_gids;
        vector<TimeStampType> node_ts;
        vector<int64_t> node_layer_ptr;
        vector<NodeIDType> root_gids;
        vector<TimeStampType> root_ts;
        vector<int64_t> root_lids;
        vector<EdgeIDType> edge_gids;
        vector<TimeStampType> edge_ts;
        vector<int64_t> edge_layer_ptr;
        vector<int64_t> edge_read_index;
        vector<int64_t> edge_read_ptr;
        vector<int64_t> compute_to_edge_feature;

        NativeSamplingOutput(){}
};
