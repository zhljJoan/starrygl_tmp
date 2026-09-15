"""Prepared CSC access needs no edge summary; feature reads still use blocks."""
from unittest.mock import patch

import pytest
import torch

from starrygl.runtime.comm import CommScheduler
from starrygl.runtime.dataloader.materialize import materialize_accessed_window
from starrygl.runtime.dataloader.pipeline import finish_batch, launch_batch
from starrygl.runtime.snapshot.materialize import access_snapshot_window
from starrygl.store import FeatureManager, GraphStore, LabelStore, StoreBundle


@pytest.mark.parametrize("limit", [-1, 1])
@pytest.mark.parametrize("edge_features", [False, True])
def test_prepared_snapshot_skips_summary_and_preserves_features_targets(limit, edge_features):
    physical_ids = (torch.tensor([4, 1, 4, 2]), torch.tensor([5, 3, 5, 0]))
    slices = [dict(snapshot_id=sid, src_nodes=torch.arange(3), dst_nodes=torch.arange(3),
                   edge_ids=ids + 100, edge_feature_ids=ids,
                   indptr=torch.tensor([0, 1, 3, 4]), indices=torch.tensor([0, 1, 2, 2]),
                   node_chunk=torch.tensor([0, 0, 1]), src_feature_row=torch.arange(3))
              for sid, ids in enumerate(physical_ids)]
    edge_values = torch.arange(12).reshape(6, 2).float()
    store = StoreBundle(
        graph=GraphStore(num_nodes=3, prepare={"partition": {}, "snapshot_csc_views": []}),
        features=FeatureManager(node_features={"x": torch.arange(6).reshape(3, 2).float()},
                                edge_features={"edge": edge_values} if edge_features else {}),
        labels=LabelStore(task_kind="node", task_ptr=torch.tensor([0, 2, 4]),
                          task_payload={"node_ids": torch.tensor([1, 0, 1, 0]),
                                        "label": torch.tensor([[1.], [2.], [3.], [4.]])}),
    )
    # No edge scan/unique belongs to this accessor, including without features.
    with patch("torch.unique", side_effect=AssertionError("unused accessor edge scan")):
        accessed = access_snapshot_window(
            store, slices, split="train", window_id=1, input_window=range(2),
            chunk_limits=(limit, limit), sampling_policy="full",
            sampler_options={"snapshot_reverse_direction": False},
            entry_cache={}, blob_cache={},
        )
    assert accessed[4] == ()
    batch = materialize_accessed_window(accessed, mode="snapshot", num_layers=2)
    launched = launch_batch(batch, store=store, comm=CommScheduler(), options={}, device="cpu",
                            feature_node_ids=accessed[3], edge_ids=accessed[4])
    batch = finish_batch(*launched, device="cpu")
    target = batch.targets["task"]
    torch.testing.assert_close(target.node_ids, torch.tensor([1, 0]))
    torch.testing.assert_close(target.target_route.target_rows, torch.tensor([1, 0]))
    torch.testing.assert_close(target.label, torch.tensor([[3.], [4.]]))
    for sid, (block,) in enumerate(batch.blocks):
        ids = physical_ids[sid] if limit < 0 else physical_ids[sid][:2]
        torch.testing.assert_close(block.edge_ids, ids + 100)
        torch.testing.assert_close(block.cache["edge_feature_ids"], ids)
        torch.testing.assert_close(batch.features["x"][sid],
                                   store.features.node_features["x"][block.src_nodes])
        if edge_features:
            torch.testing.assert_close(block.edata["edge_feat"], edge_values[ids])
            torch.testing.assert_close(batch.features["edge"][sid], edge_values[ids])
        else:
            assert "edge_feat" not in block.edata and "edge" not in batch.features
