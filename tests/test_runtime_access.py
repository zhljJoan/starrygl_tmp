from __future__ import annotations

import torch

from starrygl.batch import Batch
from starrygl.runtime.exchange import PendingNodeFeatureFetch
from starrygl.runtime.dataloader.pipeline import finish_pending_feature_fetches
from starrygl.runtime.state.access import hydrate_state
from starrygl.store.state import StateManager
from starrygl.view import GraphBlock


def _batch() -> Batch:
    empty = torch.empty(0, dtype=torch.long)
    block = GraphBlock(
        src_nodes=torch.tensor([2, 1, 2]),
        dst_nodes=torch.tensor([2, 1]),
        edge_ids=empty,
        format="coo",
        row=empty,
        col=empty,
        num_src=3,
        num_dst=2,
    )
    return Batch(mode="event", graph=block)


def _state_manager() -> StateManager:
    return StateManager(
        values=torch.tensor([[10.0], [20.0], [30.0]]),
        timestamps=torch.tensor([1.0, 2.0, 3.0]),
        kind="node_memory",
    )


def test_ready_batch_finishes_features_without_access_ticket() -> None:
    pending = PendingNodeFeatureFetch(
        keys=("node",),
        out={"node": torch.tensor([[2.0], [1.0], [2.0]])},
    )
    batch = finish_pending_feature_fetches(_batch(), ({"pending": pending},))

    assert torch.equal(batch.features["node"], torch.tensor([[2.0], [1.0], [2.0]]))


def test_exact_state_hydration_does_not_require_access_ticket() -> None:
    hydrated = hydrate_state(_batch(), _state_manager())

    assert torch.equal(hydrated.state["node_memory_node_ids"], torch.tensor([1, 2]))
    assert torch.equal(hydrated.state["node_memory"], torch.tensor([[20.0], [30.0]]))
