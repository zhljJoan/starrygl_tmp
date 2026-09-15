import pytest

from starrygl.runtime.builders import build_model_from_config, build_task_from_config
from starrygl.spec import normalize_task


def test_runtime_builders_keep_task_loss_and_state_policies_distinct() -> None:
    task = build_task_from_config({"name": "edge_prediction", "loss": "softmax"})
    model = build_model_from_config(
        {"name": "tgn", "in_dim": 3, "hidden_dim": 4},
        temporal_state={
            "filter": {"enabled": True},
            "smooth_aggregation": {"enabled": False},
        },
    )
    smooth_model = build_model_from_config(
        {"name": "tgn", "in_dim": 3, "hidden_dim": 4},
        temporal_state={
            "consistency": "bounded_stale",
            "filter": {"enabled": False},
            "smooth_aggregation": {"enabled": True, "gamma_init": 0.25},
        },
    )
    exact_model = build_model_from_config(
        {"name": "tgn", "in_dim": 3, "hidden_dim": 4},
        temporal_state={
            "consistency": "exact",
            "filter": {"enabled": True},
            "smooth_aggregation": {"enabled": True, "gamma_init": 0.25},
        },
    )

    assert task.loss == "softmax"
    assert model.memory.increment is None
    assert smooth_model.memory.increment is not None
    assert smooth_model.memory.gamma.item() == 0.25
    assert exact_model.memory.increment is None


def test_removed_edge_label_prediction_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown task"):
        normalize_task({"name": "edge_label_prediction"})


def test_task_owner_is_derived_from_task_kind() -> None:
    assert normalize_task({"name": "edge_prediction"})["ownership"] == "edge"
    assert normalize_task({"name": "node_regression"})["ownership"] == "node"
    with pytest.raises(ValueError, match="must not be configured"):
        normalize_task({"name": "node_regression", "ownership": "node"})


def test_runtime_task_keeps_canonical_task_name() -> None:
    assert build_task_from_config({"name": "node_classification"}).name == "node_classification"
    assert build_task_from_config({"name": "node_regression"}).name == "node_regression"


def test_runtime_builder_constructs_coupled_gconv_gru_without_runtime_policy_arguments() -> None:
    model = build_model_from_config(
        {"name": "gconv_gru", "in_dim": 3, "hidden_dim": 4, "out_dim": 2},
        temporal_state={
            "consistency": "bounded_stale",
            "max_staleness": 3,
            "smooth_aggregation": {"enabled": True, "gamma_init": 0.25},
        },
    )

    assert model.cell.state_key == "neighbor_recurrent"
    assert model.state_shapes == {"neighbor_recurrent": (4,)}
    assert model.increment is not None
    assert model.gamma.item() == 0.25

    exact = build_model_from_config(
        {"name": "gconv_gru", "in_dim": 3, "hidden_dim": 4, "out_dim": 2},
        temporal_state={
            "consistency": "exact",
            "max_staleness": 0,
            "smooth_aggregation": {"enabled": True, "gamma_init": 0.25},
        },
    )
    assert exact.gamma is None
