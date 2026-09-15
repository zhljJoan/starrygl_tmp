from __future__ import annotations

from itertools import product
from pathlib import Path

import pytest

import starrygl as sg


EVENT_DATASETS = (
    ("lastfm", 1000),
    ("wikitalk", 3000),
    ("stackoverflow", 3000),
    ("gdelt", 3000),
)
EVENT_MODELS = (
    ("jodie", "coupled", 1, (20,)),
    ("tgat", "decoupled", 2, (10, 10)),
    ("tgn", "coupled", 1, (20,)),
    ("apan", "coupled", 1, (20,)),
)
SNAPSHOT_DATASETS = (
    ("rec-amazon", ("node_regression", "edge_prediction")),
    ("soc-youtube", ("node_regression", "edge_prediction")),
    ("soc-flickr", ("node_regression", "edge_prediction")),
    ("soc-bitcoin", ("node_regression",)),
)
SNAPSHOT_MODELS = (
    ("tgcn", "decoupled", "node_recurrent", 2),
    ("mpnn_lstm", "decoupled", "node_recurrent", 2),
    ("evolve_gcn", "coupled", "model_recurrent", 1),
)


EVENT_CASES = tuple(
    (dataset, batch_size, model, coupling, layers, fanouts, task)
    for (dataset, batch_size), (model, coupling, layers, fanouts) in product(EVENT_DATASETS, EVENT_MODELS)
    for task in (("edge_prediction", "node_classification") if dataset == "gdelt" else ("edge_prediction",))
)
SNAPSHOT_CASES = tuple(
    (dataset, model, coupling, state_kind, layers, task)
    for dataset, tasks in SNAPSHOT_DATASETS
    for model, coupling, state_kind, layers in SNAPSHOT_MODELS
    for task in tasks
)


@pytest.mark.parametrize(
    "dataset,batch_size,model,coupling,layers,fanouts,task",
    EVENT_CASES,
)
def test_paper_event_experiment_matrix_lowers_to_one_spine(
    dataset, batch_size, model, coupling, layers, fanouts, task
) -> None:
    trainer = sg.from_config(_event_config(dataset, batch_size, model, coupling, layers, fanouts, task))

    assert trainer.plan.execution_spine == "temporal_sampling"
    assert trainer.plan.coupling == coupling
    assert trainer.plan.view.required_layouts == ("event_view", "temporal_csr")
    assert trainer.preprocess_config["split_ratios"] == [0.7, 0.15, 0.15]
    assert trainer.train_config["batch_size"] == batch_size
    assert trainer.runtime_config["sampling"]["neighbor"]["fanouts"] == list(fanouts)
    expected_state = set() if model == "tgat" else {"node_memory", "mailbox"}
    assert {dependency.kind for dependency in trainer.plan.state_dependencies} == expected_state


@pytest.mark.parametrize(
    "dataset,model,coupling,state_kind,layers,task",
    SNAPSHOT_CASES,
)
def test_paper_snapshot_experiment_matrix_lowers_chunk_decay(
    dataset, model, coupling, state_kind, layers, task
) -> None:
    trainer = sg.from_config(_snapshot_config(dataset, model, coupling, state_kind, layers, task))

    assert trainer.plan.execution_spine == "snapshot_full_graph"
    assert trainer.plan.coupling == coupling
    assert trainer.plan.window_policy == "chunk_decay"
    assert trainer.plan.view.required_layouts == ("snapshot_csc",)
    assert [dependency.kind for dependency in trainer.plan.state_dependencies] == [state_kind]
    assert trainer.preprocess_config["split_ratios"] == [0.4, 0.0, 0.6]
    assert tuple(trainer._chunk_decay(None)) == (59, 28, 13)


def test_paper_configs_cover_representative_paths_and_coupled_dtdg() -> None:
    root = Path(__file__).parents[1] / "configs" / "paper"
    trainers = {path.name: sg.from_config(path) for path in root.glob("*.json")}

    assert set(trainers) == {
        "event_gdelt_tgat_edge_prediction.json",
        "event_gdelt_tgn_edge_prediction.json",
        "snapshot_soc_bitcoin_gconv_gru_coupled_node_regression.json",
        "snapshot_soc_bitcoin_tgcn_node_regression.json",
    }
    coupled = trainers["snapshot_soc_bitcoin_gconv_gru_coupled_node_regression.json"]
    dependency = coupled.plan.state_dependencies[0]
    assert coupled.plan.coupling == "coupled"
    assert coupled.plan.cache_policy == "shared_hot"
    assert dependency.kind == "neighbor_recurrent"
    assert dependency.stage == "before_gcn"
    assert dependency.freshness_policy == "bounded_stale"

    tgn = trainers["event_gdelt_tgn_edge_prediction.json"]
    assert tgn.graph["source"].endswith("/starrygl/GDELT")
    assert (tgn.model_config["time_dim"], tgn.model_config["num_heads"]) == (100, 2)
    assert (tgn.train_config["epochs"], tgn.train_config["lr"]) == (50, 0.0004)

    tgat = trainers["event_gdelt_tgat_edge_prediction.json"]
    assert (tgat.model_config["time_dim"], tgat.model_config["num_heads"]) == (100, 2)
    assert (tgat.train_config["epochs"], tgat.train_config["lr"]) == (10, 0.0001)

    tgcn = trainers["snapshot_soc_bitcoin_tgcn_node_regression.json"]
    assert (tgcn.model_config["hidden_dim"], tgcn.train_config["epochs"], tgcn.train_config["lr"]) == (2, 1000, 0.001)


def _event_config(dataset, batch_size, model, coupling, layers, fanouts, task):
    state = "stateless" if model == "tgat" else "persistent"
    task_config = {"name": task}
    if task == "edge_prediction":
        task_config["services"] = {"negative_sampling": {"ratio": 1, "mode": "dst"}}
    return {
        "data": {"source": f"paper://{dataset}", "temporal_representation": "event_stream"},
        "backbone": {
            "name": model,
            "num_layers": layers,
            "temporal_representation": "event_stream",
            "spatial_aggregation": "sampled_neighbor",
            "coupling": coupling,
            "state": state,
        },
        "task": task_config,
        "runtime": {
            "temporal_state": {
                "consistency": "exact" if state == "stateless" else "bounded_stale",
                "max_staleness": 0 if state == "stateless" else 1,
            },
            "sampling": {
                "mode": "neighbor",
                "window": {"policy": "event_window"},
                "neighbor": {
                    "fanouts": list(fanouts),
                    "boundary_sampling": {"enabled": False},
                },
            },
            "preprocess": {
                "chunks_per_rank": 128,
                "hot_node_ratio": 0.1,
                "time_split": "batch",
                "target_batch_size": batch_size,
                "split_ratios": [0.7, 0.15, 0.15],
            },
            "train": {"batch_size": batch_size},
        },
    }


def _snapshot_config(dataset, model, coupling, state_kind, layers, task):
    task_config = {"name": task}
    if task == "edge_prediction":
        task_config["services"] = {"negative_sampling": {"ratio": 1, "mode": "dst"}}
    return {
        "data": {"source": f"paper://{dataset}", "temporal_representation": "snapshot_sequence"},
        "backbone": {
            "name": model,
            "num_layers": layers,
            "temporal_representation": "snapshot_sequence",
            "spatial_aggregation": "full_neighbor",
            "coupling": coupling,
            "state": "snapshot_recurrent",
            "state_kind": state_kind,
        },
        "task": task_config,
        "runtime": {
            "temporal_state": {"consistency": "exact", "max_staleness": 0},
            "sampling": {
                "mode": "full",
                "window": {
                    "policy": "chunk_decay",
                    "snaps_count": 4,
                    "chunk_decay": "auto:0.1",
                    "num_full_snapshots": 1,
                    "chunk_order": "rand",
                },
            },
            "preprocess": {
                "chunks_per_rank": 128,
                "hot_node_ratio": 0.1,
                "split_ratios": [0.4, 0.0, 0.6],
            },
        },
    }
