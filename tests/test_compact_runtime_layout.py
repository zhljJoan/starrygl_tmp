from __future__ import annotations

from starrygl.utils.index import compact_node_time

from types import SimpleNamespace

import torch

from starrygl.batch import Batch
from starrygl.runtime.state.access import (
    _store_hydrated_read,
    finish_hydrate_state,
    submit_hydrate_state,
)
from starrygl.model.tgn import _state_tensor, _state_value_for_layout
from starrygl.store.state import StateManager, StateRead
from starrygl.utils.index import compact_lookup_rows
from starrygl.view import GraphBlock


class _BatchState:
    def __init__(self) -> None:
        self.state: dict[str, torch.Tensor] = {}


def test_compact_lookup_rows_supports_sparse_large_ids_and_duplicates() -> None:
    base = 1 << 60
    source = torch.tensor([base + 9, base + 3, base + 9, base + 5])
    query = torch.tensor([base + 3, base + 9, base + 7])

    rows = compact_lookup_rows(source, query)

    assert torch.equal(rows, torch.tensor([1, 2, -1]))


def test_state_query_compaction_preserves_large_integer_node_time_keys() -> None:
    base = 1 << 60
    nodes = torch.tensor([base + 1, base + 2, base + 1, base + 1])
    timestamps = torch.tensor([1.0, 1.0, 1.0, 2.0])

    compact, inverse = compact_node_time(nodes, timestamps)

    assert torch.equal(compact, torch.tensor([base + 1, base + 1, base + 2]))
    assert torch.equal(inverse, torch.tensor([0, 2, 0, 1]))


def test_node_memory_stays_compact_until_model_layout_is_requested() -> None:
    state: dict[str, torch.Tensor] = {}
    nodes = torch.tensor([9, 10, 9])
    compact_nodes = torch.tensor([9, 10])
    inverse = torch.tensor([0, 1, 0])
    read = StateRead(
        node_ids=compact_nodes,
        values=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        timestamps=torch.tensor([7.0, 8.0]),
    )

    _store_hydrated_read(
        state,
        kind="node_memory",
        read=read,
        nodes=nodes,
        compact_nodes=compact_nodes,
        inverse=inverse,
    )

    assert state["node_memory"].shape == (2, 2)
    assert torch.equal(state["node_memory_node_ids"], compact_nodes)
    assert torch.equal(state["node_memory_layout_inverse"], inverse)
    assert "node_memory_version" not in state

    batch = _BatchState()
    batch.state = state
    like = torch.zeros(3, 2)
    expanded = _state_tensor(batch, "node_memory", like, 2)
    timestamps = _state_value_for_layout(batch, "node_memory_ts", 3)

    assert torch.equal(expanded, torch.tensor([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]]))
    assert timestamps is not None
    assert torch.equal(timestamps, torch.tensor([7.0, 8.0, 7.0]))


def test_non_memory_recurrent_state_keeps_existing_expanded_layout() -> None:
    state: dict[str, torch.Tensor] = {}
    nodes = torch.tensor([9, 10, 9])
    compact_nodes = torch.tensor([9, 10])
    inverse = torch.tensor([0, 1, 0])
    read = StateRead(
        node_ids=compact_nodes,
        values=torch.tensor([[1.0], [2.0]]),
    )

    _store_hydrated_read(
        state,
        kind="neighbor_recurrent",
        read=read,
        nodes=nodes,
        compact_nodes=compact_nodes,
        inverse=inverse,
    )

    assert torch.equal(state["neighbor_recurrent"], torch.tensor([[1.0], [2.0], [1.0]]))
    assert "neighbor_recurrent_layout_inverse" not in state


def test_shared_recurrent_state_exposes_generic_compensation_inputs() -> None:
    state: dict[str, torch.Tensor] = {}
    nodes = torch.tensor([1, 2, 1])
    compact_nodes = torch.tensor([1, 2])
    shared = StateManager(values=torch.zeros(1, 2), row_map=torch.tensor([-1, 0, -1]))
    read = StateRead(
        node_ids=compact_nodes,
        values=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        timestamps=torch.tensor([7.0, 8.0]),
    )

    _store_hydrated_read(
        state,
        kind="neighbor_recurrent",
        read=read,
        nodes=nodes,
        compact_nodes=compact_nodes,
        inverse=torch.tensor([0, 1, 0]),
        manager=SimpleNamespace(shared_manager=shared),
    )

    assert torch.equal(state["neighbor_recurrent_shared_mask"], torch.tensor([True, False, True]))
    assert torch.equal(state["neighbor_recurrent_shared_rows"], torch.tensor([0, -1, 0]))
    assert torch.equal(state["neighbor_recurrent_historical"], state["neighbor_recurrent"])
    assert torch.equal(state["neighbor_recurrent_historical_ts"], torch.tensor([7.0, 8.0, 7.0]))
    assert "neighbor_recurrent_version" not in state
    assert "neighbor_recurrent_historical_version" not in state


def test_model_state_initialization_is_internal_to_hydration() -> None:
    state: dict[str, torch.Tensor] = {}
    nodes = torch.tensor([0])

    _store_hydrated_read(
        state,
        kind="model_recurrent",
        read=StateRead(nodes, torch.zeros(1, 3)),
        nodes=nodes,
        compact_nodes=nodes,
        inverse=None,
        manager=SimpleNamespace(commit_count=0),
    )
    assert "model_recurrent" not in state

    _store_hydrated_read(
        state,
        kind="model_recurrent",
        read=StateRead(nodes, torch.ones(1, 3)),
        nodes=nodes,
        compact_nodes=nodes,
        inverse=None,
        manager=SimpleNamespace(commit_count=1),
    )
    assert torch.equal(state["model_recurrent"], torch.ones(1, 3))
    assert "model_recurrent_version" not in state


def test_owner_hot_state_is_not_marked_for_compensation() -> None:
    state: dict[str, torch.Tensor] = {}
    shared = StateManager(values=torch.zeros(1, 2), row_map=torch.tensor([0]))
    read = StateRead(
        node_ids=torch.tensor([0]),
        values=torch.ones(1, 2),
        metadata={
            "shared_mask": torch.tensor([False]),
            "shared_rows": torch.tensor([-1]),
        },
    )

    _store_hydrated_read(
        state,
        kind="neighbor_recurrent",
        read=read,
        nodes=torch.tensor([0]),
        compact_nodes=torch.tensor([0]),
        inverse=None,
        manager=SimpleNamespace(shared_manager=shared),
    )

    assert "neighbor_recurrent_historical" not in state


def test_submit_and_finish_hydrate_state_keep_compact_memory() -> None:
    nodes = torch.tensor([2, 1, 2])
    empty = torch.empty(0, dtype=torch.long)
    block = GraphBlock(
        src_nodes=nodes,
        dst_nodes=torch.tensor([2, 1]),
        edge_ids=empty,
        format="coo",
        row=empty,
        col=empty,
        num_src=3,
        num_dst=2,
    )
    batch = Batch(mode="event", graph=block)
    manager = StateManager(
        values=torch.tensor([[10.0], [20.0], [30.0]]),
        timestamps=torch.tensor([1.0, 2.0, 3.0]),
        kind="node_memory",
    )

    pending = submit_hydrate_state(batch, manager)
    hydrated = finish_hydrate_state(batch, pending)

    assert len(pending) == 1
    assert torch.equal(hydrated.state["node_memory_node_ids"], torch.tensor([1, 2]))
    assert torch.equal(hydrated.state["node_memory"], torch.tensor([[20.0], [30.0]]))
    assert torch.equal(hydrated.state["node_memory_layout_inverse"], torch.tensor([1, 0, 1]))
