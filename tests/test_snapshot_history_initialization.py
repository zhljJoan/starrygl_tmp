"""History binding retains exact union semantics without staging every row."""
from types import SimpleNamespace

import pytest
import torch

from starrygl.runtime.memory.snapshot import bind_snapshot_history
from starrygl.store import StateManager


@pytest.mark.parametrize("owned_ids,sources", [
    ([4, 0], [[4, 0, 6, 6], [0, 4, 6], [4, 0, 9, 9]]),
    ([], [[], [8, 3, 8]]),
    ([7], [[], []]),
    ([], [[], []]),
])
def test_binding_matches_old_sorted_union_and_state_roundtrip(owned_ids, sources):
    owned = torch.tensor(owned_ids, dtype=torch.long)
    rows = [{"src_nodes": torch.tensor(ids, dtype=torch.long)} for ids in sources]
    row_map = torch.full((12,), -1, dtype=torch.long)
    row_map[owned] = torch.arange(owned.numel())
    runtime = StateManager(values=torch.zeros(owned.numel(), 2), row_map=row_map)
    store = SimpleNamespace(graph=SimpleNamespace(num_nodes=12, snapshot_csc_view={"slices": rows}))
    expected = torch.unique(torch.cat((owned, owned[:0], *(row["src_nodes"] for row in rows))))
    bind_snapshot_history(runtime, store, owned[:0], window_size=3)
    history = runtime.snapshot_history
    torch.testing.assert_close(history.node_ids, expected, rtol=0, atol=0)
    torch.testing.assert_close(history.row_map[expected], torch.arange(expected.numel()), rtol=0, atol=0)
    assert runtime.snapshot_owned_count == owned.numel()
    assert runtime.snapshot_exact and runtime.snapshot_route is None
    assert history.packets.shape == (4, expected.numel(), 6)
    assert runtime.snapshot_shared_history.row_map.shape == (12,)
    values = torch.arange(expected.numel() * 2).reshape(-1, 2).float()
    history.update(expected, 1, values)
    torch.testing.assert_close(history.read(expected, 1)[:, :2], values, rtol=0, atol=0)


def test_binding_still_rejects_hot_replicas():
    runtime = StateManager(values=torch.zeros(1, 2), row_map=torch.tensor([0]))
    store = SimpleNamespace(graph=SimpleNamespace(num_nodes=1, snapshot_csc_view={"slices": [{"src_nodes": torch.tensor([0])}]}))
    with pytest.raises(ValueError, match="without hot replicas"):
        bind_snapshot_history(runtime, store, torch.tensor([0]), window_size=2)
