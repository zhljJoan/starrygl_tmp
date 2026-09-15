"""Small row/route invariants for explicit snapshot device materialization."""

import os

import pytest
import torch

from starrygl.runtime.snapshot.cache import _snapshot_graph_blob
from starrygl.runtime.snapshot.rows import (
    _apply_chunk_limit_to_snapshot_row,
    _move_snapshot_row,
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda":
        if os.environ.get("STARRYGL_TEST_CUDA_MATERIALIZE") != "1":
            pytest.skip("set STARRYGL_TEST_CUDA_MATERIALIZE=1 for CUDA route checks")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
    return torch.device(request.param)


def _row(chunks=(0, 1, 0, 2)):
    return {
        "snapshot_id": 0,
        "src_nodes": torch.tensor([11, 7, 19, 2, 40, 50]),
        "dst_nodes": torch.tensor([11, 7, 19, 2]),
        "indptr": torch.tensor([0, 2, 4, 6, 8]),
        "indices": torch.tensor([0, 4, 1, 2, 2, 5, 3, 0]),
        "edge_ids": torch.arange(8),
        "edge_feature_ids": torch.arange(8) + 100,
        "edge_gcn_norm": torch.arange(1, 9, dtype=torch.float32) / 10,
        "self_gcn_norm": torch.tensor([0.2, 0.3, 0.4, 0.5]),
        "node_chunk": torch.tensor([*chunks, 0, 0]),
        "src_feature_row": torch.arange(6),
        "node_data": {"x": torch.arange(12, dtype=torch.float32).reshape(6, 2)},
        "_chunk_base": 0,
        "route": {
            "send_sizes": [1, 2],
            "recv_sizes": [1, 1],
            "send_index": torch.tensor([0, 3, 1]),
            "recv_src_row": torch.tensor([4, 5]),
        },
    }


def _edges(row):
    dst = torch.repeat_interleave(row["dst_nodes"], row["indptr"].diff())
    src = row["src_nodes"].index_select(0, row["indices"])
    triples = torch.stack((row["edge_ids"], src, dst), dim=1)
    return triples.index_select(0, row["edge_ids"].argsort()).cpu()


def _blob(row, device, order=(2, 0, 1)):
    cache = {}
    kwargs = dict(sid=0, row=row, chunk_order=torch.tensor(order),
                  device=device, cache_row_on_device=True)
    blob = _snapshot_graph_blob(cache, **kwargs)
    assert _snapshot_graph_blob(cache, **kwargs) is blob
    return blob.row


def test_move_row_places_route_indices_but_preserves_host_sizes(device):
    original = _row()
    moved = _move_snapshot_row(original, device)
    for key in ("src_nodes", "dst_nodes", "indptr", "indices", "edge_ids"):
        assert moved[key].device.type == device.type
        torch.testing.assert_close(moved[key].cpu(), original[key], rtol=0, atol=0)
    assert moved["node_data"]["x"].device.type == device.type
    for key in ("send_index", "recv_src_row"):
        assert moved["route"][key].device.type == device.type
        assert original["route"][key].device.type == "cpu"
        torch.testing.assert_close(moved["route"][key].cpu(), original["route"][key])
    for key in ("send_sizes", "recv_sizes"):
        assert isinstance(moved["route"][key], (tuple, list))
        assert all(isinstance(count, int) for count in moved["route"][key])
        assert list(moved["route"][key]) == original["route"][key]


def test_full_reorder_preserves_edges_features_and_forward_backward_send_rows(device):
    original = _row()
    packed = _blob(original, device)
    assert packed["dst_nodes"].device.type == device.type
    torch.testing.assert_close(packed["dst_nodes"].cpu(), torch.tensor([7, 2, 11, 19]))
    torch.testing.assert_close(_edges(packed), _edges(original))
    send = packed["route"]["send_index"]
    assert send.device.type == device.type
    torch.testing.assert_close(packed["dst_nodes"][send].cpu(),
                               original["dst_nodes"][original["route"]["send_index"]])
    recv = packed["route"]["recv_src_row"]
    assert recv.device.type == device.type
    torch.testing.assert_close(packed["src_nodes"][recv].cpu(), torch.tensor([40, 50]))
    torch.testing.assert_close(packed["edge_gcn_norm"].cpu(),
                               original["edge_gcn_norm"][packed["edge_ids"].cpu()])
    grad = torch.zeros(4, device=device)
    grad.index_add_(0, send, torch.tensor([2., 3., 5.], device=device))
    by_id = dict(zip(packed["dst_nodes"].cpu().tolist(), grad.cpu().tolist()))
    assert by_id == {11: 2., 7: 5., 19: 0., 2: 3.}
    assert original["route"]["send_index"].tolist() == [0, 3, 1]


def test_strict_partial_is_local_and_matches_cpu_layout(device):
    packed = _blob(_row(), device)
    partial = _apply_chunk_limit_to_snapshot_row(packed, 2)
    reference = _apply_chunk_limit_to_snapshot_row(_blob(_row(), torch.device("cpu")), 2)
    assert partial.get("route") is None and bool(partial["chunk_limited"])
    torch.testing.assert_close(partial["src_nodes"].cpu(), torch.tensor([7, 2]))
    torch.testing.assert_close(_edges(partial), _edges(reference))
    torch.testing.assert_close(partial["edge_ids"].cpu(), torch.tensor([2, 6]))


def test_empty_partial_drops_route_and_marks_local(device):
    empty = _apply_chunk_limit_to_snapshot_row(_blob(_row(), device), 0)
    assert empty.get("route") is None
    assert bool(empty["chunk_limited"])
    assert empty["src_nodes"].numel() == empty["dst_nodes"].numel() == 0
    assert empty["edge_ids"].numel() == 0
    assert empty["edge_feature_ids"].numel() == 0
    assert empty["indptr"].cpu().tolist() == [0]


def test_partial_with_all_owners_still_removes_remote_neighbors(device):
    # Global J=3, but this rank has no owner in chunk 2. Limit 2 is still partial.
    packed = _blob(_row(chunks=(0, 1, 0, 1)), device, order=(0, 1, 2))
    partial = _apply_chunk_limit_to_snapshot_row(packed, 2)
    assert partial.get("route") is None
    assert bool(partial["chunk_limited"])
    assert partial["src_nodes"].numel() == partial["dst_nodes"].numel() == 4
    assert partial["edge_ids"].numel() == 6
    assert set(partial["edge_ids"].cpu().tolist()) == {0, 2, 3, 4, 6, 7}
    assert int(partial["indices"].max()) < 4


def _empty_owner_row():
    row = _row()
    for key in ("src_nodes", "dst_nodes", "indices", "edge_ids", "edge_feature_ids",
                "edge_gcn_norm", "self_gcn_norm", "node_chunk", "src_feature_row"):
        row[key] = row[key].new_empty((0,))
    row["indptr"] = torch.zeros(1, dtype=torch.long)
    row["node_data"] = {"x": torch.empty(0, 2)}
    row["route"] = {"send_sizes": [0, 0], "recv_sizes": [0, 0],
                    "send_index": torch.empty(0, dtype=torch.long), "recv_src_row": torch.empty(0, dtype=torch.long)}
    return row


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_known_global_j_preserves_s1_full_but_not_strict_partial(device, packed, empty):
    row = _empty_owner_row() if empty else _row(chunks=(0, 1, 0, 1))
    order = torch.arange(3)
    row = _blob(row, device, order=(0, 1, 2)) if packed else _move_snapshot_row(row, device)
    order = None if packed else order
    full = _apply_chunk_limit_to_snapshot_row(row, 3, chunk_order=order)
    assert full is row and full.get("route") is not None
    assert not bool(full.get("chunk_limited", False))
    assert full["src_nodes"].numel() == (0 if empty else 6)
    partial = _apply_chunk_limit_to_snapshot_row(row, 2, chunk_order=order)
    assert partial.get("route") is None and bool(partial["chunk_limited"])
    assert partial["src_nodes"].numel() == (0 if empty else 4)
    assert partial["edge_ids"].numel() == (0 if empty else 6)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_unknown_global_j_never_infers_full_from_local_max(device, packed, empty):
    row = _empty_owner_row() if empty else _row(chunks=(0, 1, 0, 1))
    row = _move_snapshot_row(row, device)
    if packed:
        row = _snapshot_graph_blob({}, sid=0, row=row, chunk_order=None,
                                   device=device, cache_row_on_device=True).row
    partial = _apply_chunk_limit_to_snapshot_row(row, 128)
    assert partial.get("route") is None and bool(partial["chunk_limited"])
    assert partial["src_nodes"].numel() == (0 if empty else 4)
    assert _apply_chunk_limit_to_snapshot_row(row, -1) is row
