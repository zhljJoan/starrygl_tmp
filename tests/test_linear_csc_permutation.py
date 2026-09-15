"""A CSC column permutation concatenates segments without an edge-key sort."""
from unittest.mock import patch

import pytest
import torch

from starrygl.runtime.snapshot.rows import _reorder_snapshot_row_for_chunk_prefix


@pytest.mark.parametrize("counts", [(2, 0, 3, 1), (0, 0, 0, 0), (1, 1, 1, 1)])
@pytest.mark.parametrize("priority", [None, (2, 0, 3, 1)])
def test_column_permutation_preserves_every_edge_and_route(counts, priority):
    counts = torch.tensor(counts)
    indptr = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    edge_count = int(indptr[-1])
    src_nodes = torch.tensor([12, 30, 8, 40, 99, 70])
    node_data = torch.arange(12).reshape(6, 2).float()
    row = dict(snapshot_id=4, src_nodes=src_nodes, dst_nodes=src_nodes[:4],
        node_chunk=torch.tensor([0, 1, 2, 3, -1, -1]), indptr=indptr,
        indices=torch.tensor([4, 0, 5, 2, 2, 1])[:edge_count],
        edge_ids=torch.arange(edge_count) + 100, edge_feature_ids=torch.arange(edge_count) * 2,
        ts=torch.arange(edge_count).float() / 3, edge_gcn_norm=torch.arange(edge_count).float() + .25,
        self_gcn_norm=torch.arange(4).float() + .5, src_feature_row=torch.arange(6) + 20,
        node_data={"x": node_data}, src_data_row=torch.arange(6),
        route=dict(send_index=torch.tensor([0, 3, 0]), recv_index=torch.tensor([4, 5]),
                   send_sizes=(1, 2), recv_sizes=(1, 1)))
    order = None if priority is None else torch.tensor(priority)
    permutation = torch.arange(4) if order is None else order.argsort(stable=True)
    edge_parent = torch.cat([torch.arange(int(indptr[d]), int(indptr[d + 1])) for d in permutation])
    original_argsort = torch.argsort

    def node_sort_only(value, *args, **kwargs):
        assert value.numel() <= 4, "column permutation must not sort one key per edge"
        return original_argsort(value, *args, **kwargs)

    with patch("torch.argsort", side_effect=node_sort_only):
        output = _reorder_snapshot_row_for_chunk_prefix(dict(row), chunk_order=order)
    torch.testing.assert_close(output["_edge_parent_row"], edge_parent, rtol=0, atol=0)
    torch.testing.assert_close(output["dst_nodes"], row["dst_nodes"][permutation], rtol=0, atol=0)
    torch.testing.assert_close(output["indptr"][1:] - output["indptr"][:-1], counts[permutation])
    torch.testing.assert_close(output["src_nodes"][output["indices"]],
                               src_nodes[row["indices"][edge_parent]], rtol=0, atol=0)
    for key in ("edge_ids", "edge_feature_ids", "ts", "edge_gcn_norm"):
        torch.testing.assert_close(output[key], row[key][edge_parent], rtol=0, atol=0)
    torch.testing.assert_close(output["self_gcn_norm"], row["self_gcn_norm"][permutation], rtol=0, atol=0)
    torch.testing.assert_close(output["dst_nodes"][output["route"]["send_index"]],
                               row["dst_nodes"][row["route"]["send_index"]], rtol=0, atol=0)
    for key in ("recv_index", "send_sizes", "recv_sizes"):
        assert output["route"][key] is row["route"][key]
    expected_rows = torch.cat((permutation, torch.tensor([4, 5])))
    torch.testing.assert_close(output["src_feature_row"], row["src_feature_row"][expected_rows])
    torch.testing.assert_close(output["node_data"]["x"][output["src_data_row"]], node_data[expected_rows])
