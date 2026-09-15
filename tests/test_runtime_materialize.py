import torch

from starrygl.runtime.dataloader.materialize import materialized_batch
from starrygl.runtime.snapshot.rows import (
    _apply_chunk_limit_to_snapshot_row,
    _reorder_snapshot_row_for_chunk_prefix,
)
from starrygl.view import GraphBlock


def _block() -> GraphBlock:
    return GraphBlock(
        src_nodes=torch.tensor([0, 1]),
        dst_nodes=torch.tensor([1]),
        edge_ids=torch.tensor([0]),
        format="coo",
        row=torch.tensor([0]),
        col=torch.tensor([0]),
        num_src=2,
        num_dst=1,
    )


def test_materialized_batch_contains_only_model_inputs() -> None:
    graph = _block()

    batch = materialized_batch(
        mode="event",
        features={
            "x": (torch.ones(2, 3),),
            "pos_edge_feat": (torch.ones(1, 2),),
        },
        targets={"task": object(), "runtime": {}},
        state={"node_memory": torch.zeros(2, 4)},
        blocks=((graph,),),
        graph=graph,
    )

    assert not hasattr(batch, "meta")
    assert batch.features["x"][0].shape == (2, 3)
    assert batch.state["node_memory"].shape == (2, 4)


def test_snapshot_chunk_order_is_chunk_to_priority_like_flare() -> None:
    row = {
        "src_nodes": torch.tensor([0, 1, 2]),
        "dst_nodes": torch.tensor([0, 1, 2]),
        "edge_ids": torch.tensor([0, 1, 2, 3]),
        "indptr": torch.tensor([0, 1, 3, 4]),
        "indices": torch.tensor([0, 1, 2, 2]),
        "node_chunk": torch.tensor([0, 1, 2]),
        "src_feature_row": torch.arange(3),
        "route": {
            "send_sizes": [1, 1],
            "recv_sizes": [0, 0],
            "send_index": torch.tensor([0, 2]),
        },
    }

    packed = _reorder_snapshot_row_for_chunk_prefix(row, chunk_order=torch.tensor([2, 0, 1]))
    prefix = _apply_chunk_limit_to_snapshot_row(packed, 1)

    assert packed["dst_nodes"].tolist() == [1, 2, 0]
    assert packed["route"]["send_index"].tolist() == [2, 1]
    assert "route" not in prefix
    assert prefix["dst_nodes"].tolist() == [1]
    assert prefix["src_nodes"].tolist() == [1]
    assert prefix["edge_ids"].tolist() == [1]
    assert prefix["indices"].tolist() == [0]
