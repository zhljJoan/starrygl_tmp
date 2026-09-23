from __future__ import annotations

import torch

from starrygl.runtime.exchange import (
    finish_node_feature_fetch,
    launch_edge_feature_fetch,
    launch_node_feature_fetch,
    materialize_node_features,
)
from starrygl.store import FeatureManager, GraphStore, LabelStore, StoreBundle


def _store() -> StoreBundle:
    return StoreBundle(
        graph=GraphStore(num_nodes=4),
        features=FeatureManager(
            node_features={
                "x": torch.tensor(
                    [
                        [0.0, 1.0],
                        [2.0, 3.0],
                        [4.0, 5.0],
                        [6.0, 7.0],
                    ]
                )
            },
            edge_features={"edge": torch.tensor([[10.0], [11.0], [12.0], [13.0]])},
        ),
        labels=LabelStore(),
    )


def test_local_node_fetch_compacts_and_restores_duplicate_ids() -> None:
    values, remote = materialize_node_features(
        _store(),
        torch.tensor([2, 1, 2]),
        assume_unique=False,
    )

    assert remote is False
    assert torch.equal(
        values["x"],
        torch.tensor(
            [
                [4.0, 5.0],
                [2.0, 3.0],
                [4.0, 5.0],
            ]
        ),
    )


def test_empty_node_fetch_returns_shaped_feature_tensor() -> None:
    values, remote = materialize_node_features(
        _store(),
        torch.empty(0, dtype=torch.long),
        assume_unique=True,
    )

    assert remote is False
    assert values["x"].shape == (0, 2)


def test_replicated_identity_edge_fetch_skips_partition_scatter() -> None:
    store = _store()
    store.features.edge_row_map = torch.arange(4)
    store.features.edge_row_map_is_identity = True
    store.features.edge_features_replicated = True

    pending = launch_edge_feature_fetch(
        store, torch.tensor([3, 1]), assume_unique=True, defer_local_read=True,
    )

    assert pending.local_read_kind == "edge_ids"
    assert pending.local_pos is None
    assert torch.equal(finish_node_feature_fetch(pending)["edge"], torch.tensor([[13.0], [11.0]]))
