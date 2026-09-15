from __future__ import annotations

import torch

from starrygl.runtime.comm import Route, all_to_all_counts, distributed
from starrygl.store.remote_fetch import submit_owner_request
from starrygl.store.snapshot_history import SnapshotHistory


def bind_snapshot_history(runtime, store, hot, *, window_size: int = 1) -> None:
    """Bind causal history and owner-boundary subscribers once."""
    manager = getattr(runtime, "memory_manager", runtime)
    device = manager.values.device
    slices = store.graph.snapshot_csc_view.get("slices", ())
    if not slices:
        raise ValueError("snapshot history requires prepared snapshot CSC views")
    if hot.numel():
        raise ValueError("snapshot history uses owner-only computation, without hot replicas")
    owned = (manager.row_map >= 0).nonzero(as_tuple=True)[0]
    runtime.snapshot_owned_count = int(owned.numel())
    present = torch.zeros(store.graph.num_nodes, dtype=torch.bool, device="cpu")
    present.index_fill_(0, owned.cpu(), True)
    for row in slices:
        present.index_fill_(0, row["src_nodes"].to(device="cpu", dtype=torch.long), True)
    nodes = present.nonzero(as_tuple=True)[0].to(device)
    runtime.snapshot_history = SnapshotHistory(nodes, store.graph.num_nodes, window_size, manager.values.shape[-1])
    runtime.snapshot_shared_history = SnapshotHistory(hot, store.graph.num_nodes, window_size, manager.values.shape[-1])
    runtime.snapshot_exact = not bool(getattr(runtime, "bounded_stale_reads", False))
    runtime.snapshot_version = 0
    runtime.snapshot_send_nodes = nodes.new_empty(0)
    runtime.snapshot_recv_nodes = nodes.new_empty(0)
    runtime.snapshot_route = None
    runtime.snapshot_boundary_nodes = nodes.new_empty(0)
    if distributed() and not runtime.snapshot_exact:
        remote = nodes[(manager.row_map[nodes] < 0) & ~torch.isin(nodes, hot)]
        request = submit_owner_request(remote, manager.node_dist_index, scheduler=runtime.comm,
                                       name="snapshot_boundary_subscribers")
        runtime.snapshot_send_nodes = request.recv_nodes
        runtime.snapshot_recv_nodes = remote[request.order]
        runtime.snapshot_route = Route(tuple(request.recv_counts.tolist()), tuple(request.send_counts.tolist()))
        runtime.snapshot_boundary_nodes, runtime.snapshot_send_inverse = torch.unique(
            runtime.snapshot_send_nodes, sorted=True, return_inverse=True)
        runtime.snapshot_boundary_rows = torch.arange(runtime.snapshot_boundary_nodes.numel(), device=device)
        runtime.snapshot_send_ranks = torch.repeat_interleave(
            torch.arange(runtime.comm.world_size, device=device), request.recv_counts.to(device),
            output_size=runtime.snapshot_send_nodes.numel())
    runtime.skip_remote_non_hot_owner_commit = True


def hydrate_snapshot_history(batch, runtime) -> None:
    history = runtime.snapshot_history
    manager = getattr(runtime, "memory_manager", runtime)
    exact = bool(getattr(runtime, "snapshot_exact", False))
    blocks = tuple(window[-1] for window in batch.iter_blocks())
    if len(blocks) > history.window_size:
        raise ValueError("Batch exceeds the configured snapshot history window_size")
    if getattr(runtime, "snapshot_route", None) is not None:
        if int(blocks[-1].cache["snapshot_id"]) != runtime.snapshot_version:
            raise ValueError("bounded owner history requires consecutive batches; warm earlier splits before replay")
        if blocks[-1].dst_nodes.numel() != runtime.snapshot_owned_count:
            raise ValueError("bounded owner history requires a complete newest owner snapshot")
        # A single batch-boundary await enforces the read-age bound; no snapshot fetch.
        finish_snapshot_pushes(runtime, ready_only=False)
    history.advance(int(blocks[0].cache["snapshot_id"]))
    shared = runtime.snapshot_shared_history
    shared.advance(int(blocks[0].cache["snapshot_id"]))
    packets = tuple(history.read(block.src_nodes, int(block.cache["snapshot_id"])) for block in blocks)
    hot_rows = tuple((shared.row_map[block.dst_nodes] >= 0).nonzero(as_tuple=True)[0] for block in blocks)
    shared_packets = tuple(shared.read(block.dst_nodes[rows], int(block.cache["snapshot_id"]) + 1)
                           for block, rows in zip(blocks, hot_rows))
    cold_rows = tuple(((manager.row_map[block.src_nodes] < 0)
                       & (shared.row_map[block.src_nodes] < 0)).nonzero(as_tuple=True)[0] for block in blocks)
    if exact:
        cold_rows = tuple(rows[:0] for rows in cold_rows)
    last = packets[-1]
    ids = blocks[-1].src_nodes
    state = dict(batch.state)
    state.update({
        "neighbor_recurrent_snapshots": packets,
        "neighbor_recurrent_shared_snapshots": shared_packets,
        "neighbor_recurrent_shared_dst_rows": hot_rows,
        "neighbor_recurrent_cold_src_rows": cold_rows,
        "neighbor_recurrent": last[:, :history.dim],
        "neighbor_recurrent_ts": last[:, -1],
        "neighbor_recurrent_node_ids": ids,
        "neighbor_recurrent_shared_mask": manager.row_map[ids] < 0,
        "neighbor_recurrent_exact": exact,
    })
    batch.state = state


def commit_snapshot_history(runtime, delta) -> None:
    history = runtime.snapshot_history
    for nodes, version, values in delta.metadata.get("snapshot_states", ()):
        history.update(nodes, version, values)
        runtime.snapshot_version = version


def launch_snapshot_push(runtime) -> None:
    route = runtime.snapshot_route
    if route is None:
        return
    # Keep at most two immutable in-flight packets. This may expose a wait when
    # the network cannot keep pace; it never changes collective launch order.
    if len(runtime.pending_snapshot_pushes) >= 2:
        runtime._count("snapshot_boundary_backpressure_waits", 1)
        _finish_snapshot_push(runtime, runtime.pending_snapshot_pushes.pop(0))
    values = runtime.snapshot_history.read(runtime.snapshot_boundary_nodes, runtime.snapshot_version)
    keep = torch.ones(values.shape[0], dtype=torch.bool, device=values.device)
    change_filter = runtime.change_filter
    if change_filter is not None:
        rows = runtime.snapshot_boundary_rows
        state = values[:, :runtime.snapshot_history.dim]
        if change_filter.min_cosine_distance is not None:
            keep = change_filter.allow(rows, values=state)
        else:
            reference = change_filter.historical.to(state).index_select(0, rows)
            keep = change_filter.allow(rows, state - reference)
        change_filter.update(rows, values=state, keep=keep)
    runtime._count("snapshot_boundary_candidate_rows", values.shape[0])
    runtime._count("snapshot_boundary_published_rows", int(keep.sum().item()))
    selected = keep[runtime.snapshot_send_inverse].nonzero(as_tuple=True)[0]
    nodes = runtime.snapshot_send_nodes[selected]
    values = values[runtime.snapshot_send_inverse[selected]]
    send_counts = torch.bincount(runtime.snapshot_send_ranks[selected], minlength=runtime.comm.world_size).cpu()
    recv_counts = all_to_all_counts(send_counts, group=runtime.comm.group)
    route = Route(tuple(send_counts.tolist()), tuple(recv_counts.tolist()))
    runtime._count("snapshot_boundary_pushes", 1)
    runtime._count("snapshot_boundary_send_rows", nodes.numel())
    runtime._count("snapshot_boundary_send_bytes", values.numel() * values.element_size() + nodes.numel() * nodes.element_size())
    runtime.pending_snapshot_pushes.append((
        runtime.comm.launch_push(route, nodes, name="snapshot_boundary:node_ids"),
        runtime.comm.launch_push(route, values, name="snapshot_boundary:owner"),
    ))


def finish_snapshot_pushes(runtime, *, ready_only: bool) -> None:
    waiting = []
    for handle in runtime.pending_snapshot_pushes:
        if not ready_only or all(part.ready() for part in handle):
            _finish_snapshot_push(runtime, handle)
        else:
            waiting.append(handle)
    runtime.pending_snapshot_pushes = waiting


def _finish_snapshot_push(runtime, handle) -> None:
    nodes, packets = (runtime.comm.finish_push(part) for part in handle)
    runtime.snapshot_history.install(nodes, packets)


def install_shared_history(runtime, nodes, packets) -> None:
    # Received hot observations never overwrite locally computed history.
    runtime.snapshot_shared_history.install(nodes, packets)
