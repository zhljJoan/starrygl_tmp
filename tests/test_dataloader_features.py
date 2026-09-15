from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

from starrygl.batch import Batch, EventRows
from starrygl.runtime.comm import CommScheduler
from starrygl.runtime.dataloader.features import launch_batch_features
from starrygl.runtime.dataloader.pipeline import finish_pending_feature_fetches
from starrygl.store import FeatureManager, GraphStore, LabelStore, StoreBundle
from starrygl.view import GraphBlock


def _block(edge_id: int, *, nodes: torch.Tensor | None = None) -> GraphBlock:
    nodes = torch.tensor([0, 1]) if nodes is None else nodes.long()
    edge_ids = (
        torch.tensor([edge_id], dtype=torch.long)
        if int(nodes.numel())
        else nodes.new_empty(0)
    )
    block = GraphBlock(
        src_nodes=nodes,
        dst_nodes=nodes,
        edge_ids=edge_ids,
        format="coo",
        row=edge_ids.new_zeros(edge_ids.shape),
        col=edge_ids.new_zeros(edge_ids.shape),
        num_src=int(nodes.numel()),
        num_dst=int(nodes.numel()),
    )
    block.cache.update(snapshot_id=0, edge_feature_ids=edge_ids)
    return block


@pytest.mark.parametrize(
    ("mode", "sampling_policy"),
    (
        ("event", "full"),
        ("event", "neighbor"),
        ("snapshot", "full"),
        ("snapshot", "neighbor"),
    ),
)
def test_all_batch_modes_use_one_feature_launch(mode: str, sampling_policy: str) -> None:
    store = StoreBundle(
        graph=GraphStore(num_nodes=3),
        features=FeatureManager(
            node_features={"x": torch.tensor([[1.0], [2.0], [3.0]])},
            edge_features={"edge": torch.tensor([[10.0], [11.0]])},
        ),
        labels=LabelStore(),
    )
    blocks = (_block(0),) if sampling_policy == "full" else (_block(0), _block(1))
    targets = {}
    if mode == "event":
        targets["events"] = EventRows(
            src=torch.tensor([0]),
            dst=torch.tensor([1]),
            edge_ids=torch.tensor([1]),
        )
    batch = Batch(
        mode=mode,
        blocks=(blocks,),
        graph=blocks[-1],
        targets=targets,
    )

    batch, pending_nodes, pending_edges = launch_batch_features(
        batch,
        store,
        comm=CommScheduler(),
        sampler_options={"defer_node_feature_finish": True},
        feature_node_ids=(torch.tensor([0, 1]),),
        edge_ids=(torch.tensor([0, 1]),),
    )
    batch = finish_pending_feature_fetches(batch, pending_nodes, pending_edges)

    assert torch.equal(batch.features["x"][0], torch.tensor([[1.0], [2.0]]))
    assert all("edge_feat" in block.edata for block in blocks)
    if mode == "event":
        assert torch.equal(batch.features["pos_edge_feat"][0], torch.tensor([[11.0]]))
    else:
        assert torch.equal(batch.features["edge"][0], torch.tensor([[10.0]]))


def test_event_feature_fetch_compacts_duplicate_sampler_rows() -> None:
    store = StoreBundle(
        graph=GraphStore(num_nodes=2),
        features=FeatureManager(node_features={"x": torch.tensor([[1.0], [2.0]])}),
        labels=LabelStore(),
    )
    nodes = torch.tensor([0, 1, 0])
    block = _block(0, nodes=nodes)
    batch = Batch(mode="event", blocks=((block,),), graph=block)

    batch, pending_nodes, pending_edges = launch_batch_features(
        batch,
        store,
        comm=CommScheduler(),
        sampler_options={"defer_node_feature_finish": True},
        feature_node_ids=(nodes,),
        edge_ids=(block.edge_ids,),
    )

    assert pending_nodes[0]["pending"].compact is not None
    batch = finish_pending_feature_fetches(batch, pending_nodes, pending_edges)
    assert torch.equal(batch.features["x"][0], torch.tensor([[1.0], [2.0], [1.0]]))


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) < 2,
    reason="requires torchrun with at least two ranks",
)
def test_empty_rank_enters_the_same_feature_collective() -> None:
    created_group = not dist.is_initialized()
    if created_group:
        dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        assert dist.get_world_size() == 2
        dist_index = torch.tensor([0, (1 << 48)], dtype=torch.long)
        row_map = (
            torch.tensor([0, -1], dtype=torch.long)
            if rank == 0
            else torch.tensor([-1, 0], dtype=torch.long)
        )
        store = StoreBundle(
            graph=GraphStore(
                num_nodes=2,
                rank=rank,
                prepare={"meta": {"world_size": 2}, "partition": {"node_dist_index": dist_index}},
            ),
            features=FeatureManager(
                node_features={"x": torch.tensor([[float(rank)]])},
                node_row_map=row_map,
            ),
            labels=LabelStore(),
        )
        node_ids = torch.tensor([0, 1]) if rank == 0 else torch.empty(0, dtype=torch.long)
        block = _block(0, nodes=node_ids)
        batch = Batch(
            mode="event",
            blocks=((block,),),
            graph=block,
        )

        batch, pending_nodes, pending_edges = launch_batch_features(
            batch,
            store,
            comm=CommScheduler(),
            sampler_options={"defer_node_feature_finish": True},
            feature_node_ids=(node_ids,),
            edge_ids=(block.edge_ids,),
        )
        batch = finish_pending_feature_fetches(batch, pending_nodes, pending_edges)

        expected = torch.tensor([[0.0], [1.0]]) if rank == 0 else torch.empty((0, 1))
        assert torch.equal(batch.features["x"][0], expected)
        dist.barrier()
    finally:
        if created_group and dist.is_initialized():
            dist.destroy_process_group()
