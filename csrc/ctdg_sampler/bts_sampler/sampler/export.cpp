#include<head.h>
#include <sampler.h>
#include <tppr.h>
#include <output.h>
#include <neighbors.h>
#include <temporal_utils.h>
#include <speed_partition.h>


/*------------Python Bind--------------------------------------------------------------*/
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m
    .def("speed_partition",
        &speed_partition,
        py::arg("src"),
        py::arg("dst"),
        py::arg("ts"),
        py::arg("num_nodes"),
        py::arg("num_parts"),
        py::arg("beta") = 1.0,
        py::arg("topk_ratio") = 0.01,
        py::arg("topk_type") = "decay",
        py::return_value_policy::move)
    .def("assign_chunks_temporal_balance",
        &assign_chunks_temporal_balance,
        py::arg("chunk_load"),
        py::arg("affinity"),
        py::arg("world_size"),
        py::arg("chunks_per_rank"),
        py::arg("affinity_weight") = 0.05,
        py::arg("local_search_iters") = 2000)
    .def("get_neighbors", 
        &get_neighbors, 
        py::return_value_policy::reference)    
    .def("heads_unique", 
        &heads_unique, 
        py::return_value_policy::reference)
    .def("divide_nodes_to_part", 
        &divide_nodes_to_part, 
        py::return_value_policy::reference)
    .def("sparse_get_index", 
        &sparse_get_index, 
        py::return_value_policy::reference)
    .def("get_norm_temporal",
        &get_norm_temporal, 
        py::return_value_policy::reference
    );

    py::class_<TemporalGraphBlock>(m, "TemporalGraphBlock")
        .def(py::init<vector<NodeIDType> &, vector<NodeIDType> &,
                      vector<NodeIDType> &>())
        .def("row", [](const TemporalGraphBlock &tgb) { return vecToTensor<NodeIDType>(tgb.row); })
        .def("col", [](const TemporalGraphBlock &tgb) { return vecToTensor<NodeIDType>(tgb.col); })
        .def("eid", [](const TemporalGraphBlock &tgb) { return vecToTensor<EdgeIDType>(tgb.eid); })
        .def("delta_ts", [](const TemporalGraphBlock &tgb) { return vecToTensor<TimeStampType>(tgb.delta_ts); })
        .def("src_index", [](const TemporalGraphBlock &tgb) { return vecToTensor<EdgeIDType>(tgb.src_index); })
        .def("sample_nodes", [](const TemporalGraphBlock &tgb) { return vecToTensor<NodeIDType>(tgb.sample_nodes); })
        .def("sample_nodes_ts", [](const TemporalGraphBlock &tgb) { return vecToTensor<TimeStampType>(tgb.sample_nodes_ts); })
        .def("sample_weight",[](const TemporalGraphBlock &tgb){
            return vecToTensor<float>(tgb.sample_weight);
        })
        .def_readonly("sample_time", &TemporalGraphBlock::sample_time, py::return_value_policy::reference)
        .def_readonly("tot_time", &TemporalGraphBlock::tot_time, py::return_value_policy::reference)
        .def_readonly("sample_edge_num", &TemporalGraphBlock::sample_edge_num, py::return_value_policy::reference);

    py::class_<NativeMFGBlock>(m, "NativeMFGBlock")
        .def("dst_lids", [](const NativeMFGBlock &mfg) { return vecToTensor<NodeIDType>(mfg.dst_lids).clone(); })
        .def("src_lids", [](const NativeMFGBlock &mfg) { return vecToTensor<NodeIDType>(mfg.src_lids).clone(); })
        .def("csc_indptr", [](const NativeMFGBlock &mfg) { return vecToTensor<int64_t>(mfg.csc_indptr).clone(); })
        .def("csc_indices", [](const NativeMFGBlock &mfg) { return vecToTensor<NodeIDType>(mfg.csc_indices).clone(); })
        .def("edge_lids", [](const NativeMFGBlock &mfg) { return vecToTensor<EdgeIDType>(mfg.edge_lids).clone(); })
        .def("delta_t", [](const NativeMFGBlock &mfg) { return vecToTensor<TimeStampType>(mfg.delta_t).clone(); })
        .def_readonly("layer", &NativeMFGBlock::layer, py::return_value_policy::reference)
        .def_readonly("src_begin", &NativeMFGBlock::src_begin, py::return_value_policy::reference)
        .def_readonly("src_end", &NativeMFGBlock::src_end, py::return_value_policy::reference)
        .def_readonly("dst_begin", &NativeMFGBlock::dst_begin, py::return_value_policy::reference)
        .def_readonly("dst_end", &NativeMFGBlock::dst_end, py::return_value_policy::reference);

    py::class_<NativeSamplingOutput>(m, "NativeSamplingOutput")
        .def_readonly("mfgs", &NativeSamplingOutput::mfgs, py::return_value_policy::reference)
        .def("node_gids", [](const NativeSamplingOutput &out) { return vecToTensor<NodeIDType>(out.node_gids).clone(); })
        .def("node_ts", [](const NativeSamplingOutput &out) { return vecToTensor<TimeStampType>(out.node_ts).clone(); })
        .def("node_layer_ptr", [](const NativeSamplingOutput &out) { return vecToTensor<int64_t>(out.node_layer_ptr).clone(); })
        .def("root_gids", [](const NativeSamplingOutput &out) { return vecToTensor<NodeIDType>(out.root_gids).clone(); })
        .def("root_ts", [](const NativeSamplingOutput &out) { return vecToTensor<TimeStampType>(out.root_ts).clone(); })
        .def("root_lids", [](const NativeSamplingOutput &out) { return vecToTensor<int64_t>(out.root_lids).clone(); })
        .def("edge_gids", [](const NativeSamplingOutput &out) { return vecToTensor<EdgeIDType>(out.edge_gids).clone(); })
        .def("edge_ts", [](const NativeSamplingOutput &out) { return vecToTensor<TimeStampType>(out.edge_ts).clone(); })
        .def("edge_layer_ptr", [](const NativeSamplingOutput &out) { return vecToTensor<int64_t>(out.edge_layer_ptr).clone(); })
        .def("edge_read_index", [](const NativeSamplingOutput &out) { return vecToTensor<int64_t>(out.edge_read_index).clone(); })
        .def("edge_read_ptr", [](const NativeSamplingOutput &out) { return vecToTensor<int64_t>(out.edge_read_ptr).clone(); })
        .def("compute_to_edge_feature", [](const NativeSamplingOutput &out) { return vecToTensor<int64_t>(out.compute_to_edge_feature).clone(); });

    py::class_<TemporalNeighborBlock>(m, "TemporalNeighborBlock")
        .def(py::init<vector<vector<NodeIDType>>&, 
                      vector<int64_t> &>())
        .def(py::pickle(
            [](const TemporalNeighborBlock& tnb) { return tnb.serialize(); },
            [](const std::string& s) { return TemporalNeighborBlock::deserialize(s); }
        ))
        .def("update_neighbors_with_time", 
            &TemporalNeighborBlock::update_neighbors_with_time)
        .def("update_edge_weight", 
            &TemporalNeighborBlock::update_edge_weight)
        .def("update_node_weight", 
            &TemporalNeighborBlock::update_node_weight)
        .def("update_all_node_weight", 
            &TemporalNeighborBlock::update_all_node_weight)            
        // .def("get_node_neighbor",&TemporalNeighborBlock::get_node_neighbor)
        // .def("get_node_deg", &TemporalNeighborBlock::get_node_deg)
        .def_readonly("neighbors", &TemporalNeighborBlock::neighbors, py::return_value_policy::reference)
        .def_readonly("timestamp", &TemporalNeighborBlock::timestamp, py::return_value_policy::reference)
        .def_readonly("edge_weight", &TemporalNeighborBlock::edge_weight, py::return_value_policy::reference)
        .def_readonly("eid", &TemporalNeighborBlock::eid, py::return_value_policy::reference)
        .def_readonly("deg", &TemporalNeighborBlock::deg, py::return_value_policy::reference)
        .def_readonly("with_eid", &TemporalNeighborBlock::with_eid, py::return_value_policy::reference)
        .def_readonly("with_timestamp", &TemporalNeighborBlock::with_timestamp, py::return_value_policy::reference)
        .def_readonly("weighted", &TemporalNeighborBlock::weighted, py::return_value_policy::reference);

    py::class_<ParallelSampler>(m, "ParallelSampler")
        .def(py::init<TemporalNeighborBlock &, NodeIDType, EdgeIDType, int,
                      vector<int>&, int, string, int, th::Tensor &,th::Tensor &,th::Tensor &,double>())
        .def_readonly("ret", &ParallelSampler::ret, py::return_value_policy::reference)
        .def("neighbor_sample_from_nodes", &ParallelSampler::neighbor_sample_from_nodes)
        .def("sample_dtdg_uniform", &ParallelSampler::sample_dtdg_uniform)
        .def("reset", &ParallelSampler::reset)
        .def("set_seed", &ParallelSampler::set_seed)
        .def("set_compact_raw_edge_ids", &ParallelSampler::set_compact_raw_edge_ids)
        .def("set_compact_node_ids", &ParallelSampler::set_compact_node_ids)
        .def("reset_profile_stats", &ParallelSampler::reset_profile_stats)
        .def("get_ret", [](const ParallelSampler &ps) { return ps.ret; })
        .def("get_sampling_output", &ParallelSampler::get_sampling_output)
        .def("get_sampling_output_compact", &ParallelSampler::get_sampling_output_compact)
        .def("get_sampling_output_parallel", &ParallelSampler::get_sampling_output_parallel)
        .def("set_edge_read_dist_index", &ParallelSampler::set_edge_read_dist_index)
        .def("sample_unique", &ParallelSampler::sample_unique)
        .def_readonly("dist_nid",&ParallelSampler::dist_nid,py::return_value_policy::reference)
        .def_readonly("dist_eid",&ParallelSampler::dist_eid,py::return_value_policy::reference)
        .def_readonly("block_node_list",&ParallelSampler::block_node_list,py::return_value_policy::reference)
        .def_readonly("eid_inv",&ParallelSampler::eid_inv,py::return_value_policy::reference)
        .def_readonly("unq_id",&ParallelSampler::unq_id,py::return_value_policy::reference)
        .def_readonly("first_block_id",&ParallelSampler::first_block_id,py::return_value_policy::reference)
        .def_readonly("compact_total_seconds", &ParallelSampler::compact_total_seconds)
        .def_readonly("compact_root_seconds", &ParallelSampler::compact_root_seconds)
        .def_readonly("compact_index_seconds", &ParallelSampler::compact_index_seconds)
        .def_readonly("compact_fill_seconds", &ParallelSampler::compact_fill_seconds)
        .def_readonly("compact_fill_cursor_seconds", &ParallelSampler::compact_fill_cursor_seconds)
        .def_readonly("compact_fill_edge_loop_seconds", &ParallelSampler::compact_fill_edge_loop_seconds)
        .def_readonly("compact_fill_raw_edge_rows", &ParallelSampler::compact_fill_raw_edge_rows)
        .def_readonly("compact_fill_mapped_edge_rows", &ParallelSampler::compact_fill_mapped_edge_rows)
        .def_readonly("compact_index_edges", &ParallelSampler::compact_index_edges)
        .def_readonly("compact_index_unique_nodes", &ParallelSampler::compact_index_unique_nodes)
        .def_readonly("compact_index_frontier_nodes", &ParallelSampler::compact_index_frontier_nodes)
        .def_readonly("compact_index_unique_edges", &ParallelSampler::compact_index_unique_edges)
        .def_readonly("compact_index_last_node_hits", &ParallelSampler::compact_index_last_node_hits)
        .def_readonly("compact_index_hash_lookups", &ParallelSampler::compact_index_hash_lookups)
        .def_readonly("compact_index_hash_hits", &ParallelSampler::compact_index_hash_hits)
        .def_readonly("compact_index_hash_inserts", &ParallelSampler::compact_index_hash_inserts)
        .def_readonly("sampler_local_nodes", &ParallelSampler::sampler_local_nodes)
        .def_readonly("sampler_remote_nodes", &ParallelSampler::sampler_remote_nodes)
        .def_readonly("sampler_local_edges", &ParallelSampler::sampler_local_edges)
        .def_readonly("sampler_remote_edges", &ParallelSampler::sampler_remote_edges)
        .def_property_readonly("compact_node_prefetch_distance", [](const ParallelSampler &) {
            return ParallelSampler::kCompactNodePrefetchDistance;
        })
        .def_property_readonly("compact_raw_fill_prefetch_distance", [](const ParallelSampler &) {
            return ParallelSampler::kCompactRawFillPrefetchDistance;
        })
        .def_property_readonly("compact_mapped_edge_prefetch_distance", [](const ParallelSampler &) {
            return ParallelSampler::kCompactMappedEdgePrefetchDistance;
        })
        .def_readonly("edge_read_layout_seconds", &ParallelSampler::edge_read_layout_seconds);

    py::class_<ParallelTppRComputer>(m, "ParallelTppRComputer")
        .def(py::init<TemporalNeighborBlock &, NodeIDType, EdgeIDType, int,
                      int, int, int, vector<float>&, vector<float>& >())
        .def_readonly("ret", &ParallelTppRComputer::ret, py::return_value_policy::reference)
        .def("reset_ret", &ParallelTppRComputer::reset_ret)
        .def("reset_tppr", &ParallelTppRComputer::reset_tppr)
        .def("reset_val_tppr", &ParallelTppRComputer::reset_val_tppr)
        .def("backup_tppr", &ParallelTppRComputer::backup_tppr)
        .def("restore_tppr", &ParallelTppRComputer::restore_tppr)
        .def("restore_val_tppr", &ParallelTppRComputer::restore_val_tppr)
        .def("get_pruned_topk", &ParallelTppRComputer::get_pruned_topk)
        .def("extract_streaming_tppr", &ParallelTppRComputer::extract_streaming_tppr)
        .def("streaming_topk", &ParallelTppRComputer::streaming_topk)
        .def("single_streaming_topk", &ParallelTppRComputer::single_streaming_topk)
        .def("streaming_topk_no_fake", &ParallelTppRComputer::streaming_topk_no_fake)
        .def("compute_val_tppr", &ParallelTppRComputer::compute_val_tppr)
        .def("get_ret", [](const ParallelTppRComputer &ps) { return ps.ret; });

}
