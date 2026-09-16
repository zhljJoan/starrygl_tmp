from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from starrygl.batch import Batch
from starrygl.prepare import load_graph_data
from starrygl.prepare.snapshot_csc import build_snapshot_csc_views
from starrygl.prepare.temporal_csr import build_temporal_csr_view
from starrygl.runtime.dataloader.blocks import event_rows_to_graph_block, snapshot_row_to_graph_block
from starrygl.view import GraphBlock, graph_block_from_coo


def _time(fn, repeats: int):
    values, result = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        values.append(time.perf_counter() - start)
    return result, {"median_seconds": statistics.median(values), "samples_seconds": values}


def _event_recent(view, roots, fanout):
    rows = torch.arange(view["src"].numel())
    rows = rows[torch.isin(view["src"], roots)]
    order = torch.argsort(view["src"].index_select(0, rows), stable=True)
    rows = rows.index_select(0, order)
    keys = view["src"].index_select(0, rows)
    counts = torch.bincount(keys, minlength=int(view["src"].max()) + 1)
    starts = counts.cumsum(0) - counts
    positions = torch.arange(rows.numel()) - torch.repeat_interleave(starts, counts)
    keep = positions >= torch.repeat_interleave((counts - fanout).clamp_min(0), counts)
    return event_rows_to_graph_block(view, rows[keep], extra_nodes=roots)


def _tcsr_recent(view, roots, fanout):
    starts = view["indptr"].index_select(0, roots)
    ends = view["indptr"].index_select(0, roots + 1)
    lengths = (ends - starts).clamp_max(fanout)
    offsets = torch.arange(fanout).expand(roots.numel(), -1)
    positions = ends[:, None] - lengths[:, None] + offsets
    valid = offsets < lengths[:, None]
    positions = positions[valid]
    rows = view["edge_order"].index_select(0, positions)
    return event_rows_to_graph_block(view, rows, extra_nodes=roots)


def _snapshot_event(graph, begin, end):
    rows = torch.arange(begin, end)
    return graph_block_from_coo(src=graph.src[rows], dst=graph.dst[rows],
        edge_ids=graph.edge_ids[rows], num_nodes=graph.num_nodes, format="csc", materialize_coo=False)


def _snapshot_tcsr(view, edge_to_position, begin, end, num_nodes):
    positions = torch.sort(edge_to_position[begin:end]).values
    rows = view["edge_order"].index_select(0, positions)
    dst = view["src"].index_select(0, rows)
    counts = torch.bincount(dst, minlength=num_nodes)
    indptr = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    nodes = torch.arange(num_nodes)
    return GraphBlock(src_nodes=nodes, dst_nodes=nodes,
        edge_ids=view["edge_ids"].index_select(0, rows), format="csc", indptr=indptr,
        indices=view["indices"].index_select(0, positions), num_src=num_nodes, num_dst=num_nodes,
        edata={"ts": view["ts"].index_select(0, rows)}, exec_mode="LOCAL_FULL")


def _edge_ids(batch):
    block = batch.graph if batch.graph is not None else batch.blocks[0][0]
    return torch.unique(block.edge_ids.long(), sorted=True)


def run(args):
    if args.synthetic_edges:
        generator = torch.Generator().manual_seed(20260915)
        num_nodes = max(2, int(args.synthetic_edges ** 0.5))
        graph = load_graph_data({"src": torch.randint(num_nodes, (args.synthetic_edges,), generator=generator),
            "dst": torch.randint(num_nodes, (args.synthetic_edges,), generator=generator),
            "ts": torch.arange(args.synthetic_edges), "num_nodes": num_nodes})
    else:
        graph = load_graph_data(args.data)
    if args.max_edges and graph.src.numel() > args.max_edges:
        keep = slice(0, args.max_edges)
        graph = type(graph)(**{**vars(graph), "src": graph.src[keep], "dst": graph.dst[keep],
            "ts": graph.ts[keep], "edge_ids": graph.edge_ids[keep], "time_ptr_2": None})
    nodes, edges = graph.num_nodes, graph.src.numel()
    ptr = graph.time_ptr_2
    if ptr is None:
        step = max(1, edges // args.snapshots)
        starts = torch.arange(0, edges, step)[:args.snapshots]
        ptr = torch.stack((starts, torch.cat((starts[1:], torch.tensor([edges])))), 1)
    node_master = torch.zeros(nodes, dtype=torch.long)
    edge_dist = torch.arange(edges, dtype=torch.long)
    node_chunk = torch.zeros(nodes, dtype=torch.long)
    edge_chunk = torch.zeros(edges, dtype=torch.long)
    hot = torch.empty(0, dtype=torch.long)
    hot_mask = torch.zeros(nodes, dtype=torch.bool)

    t0 = time.perf_counter()
    tcsr = build_temporal_csr_view(src=graph.src, dst=graph.dst, ts=graph.ts,
        edge_ids=graph.edge_ids, node_dist_index=torch.arange(nodes), edge_dist_index=edge_dist,
        node_to_chunk=node_chunk, edge_chunk=edge_chunk, time_ptr_2=ptr, num_nodes=nodes,
        bidirectional=False, shared=False, node_is_hot=hot_mask)
    tcsr_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    snapshot_tcsr = build_temporal_csr_view(src=graph.dst, dst=graph.src, ts=graph.ts,
        edge_ids=graph.edge_ids, node_dist_index=torch.arange(nodes), edge_dist_index=edge_dist,
        node_to_chunk=node_chunk, edge_chunk=edge_chunk, time_ptr_2=ptr, num_nodes=nodes,
        bidirectional=False, shared=False, node_is_hot=hot_mask)
    edge_to_position = torch.empty(edges, dtype=torch.long)
    edge_to_position[snapshot_tcsr["edge_order"]] = torch.arange(edges)
    snapshot_tcsr_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    snapshots = build_snapshot_csc_views(src=graph.src, dst=graph.dst, ts=graph.ts,
        edge_ids=graph.edge_ids, edge_dist_index=edge_dist, node_master=node_master,
        hot_node_ids=hot, node_is_hot=hot_mask, node_to_chunk=node_chunk, time_ptr_2=ptr,
        num_nodes=nodes, world_size=1)[0]["slices"]
    snapshot_build = time.perf_counter() - t0

    roots = torch.unique(graph.src[-args.batch_size:])
    event_view = {"src": graph.src, "dst": graph.dst, "edge_ids": graph.edge_ids, "ts": graph.ts}
    event_block, event_time = _time(lambda: _event_recent(event_view, roots, args.fanout), args.repeats)
    tcsr_block, tcsr_time = _time(lambda: _tcsr_recent(tcsr, roots, args.fanout), args.repeats)
    event_batch = Batch(mode="event", blocks=((event_block,),))
    tcsr_batch = Batch(mode="event", blocks=((tcsr_block,),))
    if not torch.equal(_edge_ids(event_batch), _edge_ids(tcsr_batch)):
        raise RuntimeError("Event and T-CSR history queries produced different edges")

    sid = len(snapshots) - 1 if args.snapshot < 0 else min(args.snapshot, len(snapshots) - 1)
    begin, end = map(int, ptr[sid].tolist())
    snapshot_nodes = torch.cat((graph.src[begin:end], graph.dst[begin:end]))
    event_snap, event_snap_time = _time(lambda: _snapshot_event(graph, begin, end), args.repeats)
    tcsr_snap, tcsr_snap_time = _time(
        lambda: _snapshot_tcsr(snapshot_tcsr, edge_to_position, begin, end, nodes), args.repeats)
    csc_snap, csc_time = _time(lambda: snapshot_row_to_graph_block(snapshots[sid]), args.repeats)
    snapshot_batches = [Batch(mode="snapshot", graph=b) for b in (event_snap, tcsr_snap, csc_snap)]
    expected = _edge_ids(snapshot_batches[0])
    if any(not torch.equal(expected, _edge_ids(batch)) for batch in snapshot_batches[1:]):
        counts = [int(_edge_ids(batch).numel()) for batch in snapshot_batches]
        raise RuntimeError(f"snapshot paths produced different edges: event/tcsr/csc={counts}")

    result = {"data": "synthetic" if args.synthetic_edges else str(args.data), "num_nodes": nodes, "num_edges": int(edges),
        "query": {"batch_roots": int(roots.numel()), "fanout": args.fanout,
                  "snapshot": sid, "snapshot_edges": end - begin,
                  "snapshot_nodes": int(torch.unique(snapshot_nodes).numel())},
        "build_seconds": {"event": 0.0, "temporal_csr": tcsr_build,
                          "snapshot_temporal_csr": snapshot_tcsr_build,
                          "snapshot_csc": snapshot_build},
        "model_ready_seconds": {"history_event": event_time, "history_tcsr": tcsr_time,
            "snapshot_event": event_snap_time, "snapshot_tcsr": tcsr_snap_time,
            "snapshot_csc": csc_time}}
    result["amortized_seconds"] = {
        str(n): {"history_event": n * event_time["median_seconds"],
                 "history_tcsr": tcsr_build + n * tcsr_time["median_seconds"],
                 "snapshot_event": n * event_snap_time["median_seconds"],
                 "snapshot_tcsr": snapshot_tcsr_build + n * tcsr_snap_time["median_seconds"],
                 "snapshot_csc": snapshot_build + n * csc_time["median_seconds"]}
        for n in (1, 5, 10, 50, 100)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--synthetic-edges", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8000)
    parser.add_argument("--fanout", type=int, default=10)
    parser.add_argument("--snapshots", type=int, default=16)
    parser.add_argument("--snapshot", type=int, default=-1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-edges", type=int)
    args = parser.parse_args()
    if args.data is None and not args.synthetic_edges:
        parser.error("one of --data or --synthetic-edges is required")
    run(args)


if __name__ == "__main__":
    main()
