from __future__ import annotations

from typing import Any, Mapping

from starrygl.model import (
    APANModel,
    DCRNNModel,
    EvolveGCNModel,
    GConvGRUModel,
    JODIEModel,
    MPNNLSTMModel,
    StarryModel,
    TGATModel,
    TGCNModel,
    TGNModel,
)
from starrygl.store import StoreBundle
from starrygl.task import EdgePredictionTask, NodePredictionTask, StarryTask


def build_model_from_config(
    config: Mapping[str, Any],
    *,
    store: StoreBundle | None = None,
    temporal_state: Mapping[str, Any] | None = None,
    training: Mapping[str, Any] | None = None,
) -> StarryModel:
    data = dict(config)
    name = str(data.pop("name", "")).strip().lower().replace("-", "_")
    if not name:
        raise ValueError("model config requires name")
    for key in _BACKBONE_SEMANTIC_KEYS:
        data.pop(key, None)
    _lower_temporal_state(data, name=name, temporal_state=temporal_state, store=store)
    _lower_training(data, name=name, training=training)
    in_dim = int(data.pop("in_dim")) if "in_dim" in data else _infer_input_dim(store)
    hidden_dim = int(data.pop("hidden_dim", 16))
    out_dim = int(data.pop("out_dim", 2))
    if name in {"tgat", "tgn", "jodie", "apan"}:
        data.setdefault("edge_dim", _infer_edge_dim(store))
    models = {
        "tgat": TGATModel,
        "tgn": TGNModel,
        "tgcn": TGCNModel,
        "gconv_gru": GConvGRUModel,
        "dcrnn": DCRNNModel,
        "mpnn_lstm": MPNNLSTMModel,
        "evolve_gcn": EvolveGCNModel,
        "evolvegcn": EvolveGCNModel,
        "jodie": JODIEModel,
        "apan": APANModel,
    }
    try:
        model = models[name]
    except KeyError as exc:
        raise ValueError(f"unsupported model: {name!r}") from exc
    return model(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, **data)


def maybe_build_model_from_config(
    config: Mapping[str, Any],
    *,
    temporal_state: Mapping[str, Any] | None = None,
    training: Mapping[str, Any] | None = None,
) -> StarryModel | None:
    return (
        build_model_from_config(config, temporal_state=temporal_state, training=training)
        if "name" in config and "in_dim" in config
        else None
    )


def build_task_from_config(config: Mapping[str, Any]) -> StarryTask:
    name = str(config.get("name", "edge_prediction")).strip().lower()
    train_loss_mode = str(config.get("train_loss_mode", "last_only"))
    if name == "edge_prediction":
        if train_loss_mode != "last_only":
            raise ValueError("window_mean requires a snapshot node task")
        return EdgePredictionTask(loss=str(config.get("loss", "bce")))
    if name in {"node_prediction", "node_classification", "node_regression"}:
        return NodePredictionTask(name=name, loss=str(config.get("loss",
            "mse" if name == "node_regression" else "cross_entropy")), train_loss_mode=train_loss_mode)
    raise ValueError(f"unsupported task: {name!r}")


def _infer_input_dim(store: StoreBundle | None) -> int:
    if store is None:
        raise ValueError("model config requires in_dim when no store is available")
    feature = store.features.node_features.get("x")
    return 0 if feature is None else int(feature.shape[-1]) if feature.dim() > 1 else 1


def _infer_edge_dim(store: StoreBundle | None) -> int:
    if store is None:
        return 0
    feature = store.features.edge_features.get("edge")
    return 0 if feature is None else int(feature.shape[-1]) if feature.dim() > 1 else 1


def _lower_temporal_state(
    data: dict[str, Any],
    *,
    name: str,
    temporal_state: Mapping[str, Any] | None,
    store: StoreBundle | None,
) -> None:
    state = temporal_state or {}
    if name in {"tgn", "jodie", "apan"}:
        smooth_cfg = state.get("smooth_aggregation", {})
        enabled = (
            str(state.get("consistency", "exact")) == "bounded_stale"
            and isinstance(smooth_cfg, Mapping)
            and bool(smooth_cfg.get("enabled", False))
        )
        data.setdefault("state_compensation", enabled)
        if enabled and "gamma_init" in smooth_cfg:
            data.setdefault("gamma_init", float(smooth_cfg["gamma_init"]))
        if name == "tgn" and state.get("update") is not None:
            data.setdefault("memory_update", str(state["update"]))
        if data.get("state_compensation") and "compensation_num_rows" not in data:
            hot = None if store is None else store.graph.partition.get("hot_node_ids")
            data["compensation_num_rows"] = max(1, int(hot.numel()) if hasattr(hot, "numel") else 1)
    if name in {"gconv_gru", "dcrnn"}:
        prediction = state.get("boundary_prediction", {})
        enabled = (
            str(state.get("consistency", "exact")) == "bounded_stale"
            and bool(data.get("state_extrapolation", True))
            and bool(prediction.get("learnable", True))
        )
        data["gamma_boundary_init"] = float(prediction.get("gamma_init", 2.1972245773362196)) if enabled else None
    if name in {"tgcn", "mpnn_lstm", "evolvegcn", "evolve_gcn"}:
        data.setdefault("persist_state", True)


def _lower_training(data: dict[str, Any], *, name: str, training: Mapping[str, Any] | None) -> None:
    config = training or {}
    if name in {"tgat", "tgn", "jodie", "apan"} and "dropout" in config:
        data.setdefault("dropout", float(config["dropout"]))
    if name in {"tgat", "tgn", "apan"} and "att_dropout" in config:
        data.setdefault("att_dropout", float(config["att_dropout"]))


_BACKBONE_SEMANTIC_KEYS = {
    "aggregate_key",
    "compensation_num_rows",
    "coupling",
    "gamma_init",
    "reads_neighbor_state",
    "requires_temporal_state",
    "spatial_aggregation",
    "state",
    "state_compensation",
    "state_key",
    "state_kind",
    "temporal_representation",
}


__all__ = ["build_model_from_config", "build_task_from_config", "maybe_build_model_from_config"]
