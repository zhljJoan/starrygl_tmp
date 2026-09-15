from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any, Literal, Mapping, Union


TemporalMode = Literal["event", "snapshot"]
StateMode = Literal["persistent", "stateless", "snapshot_recurrent"]
ScopeMode = Literal["sampled", "full_graph"]
ConsistencyMode = Literal["exact", "bounded_stale"]
ApproximationMode = Literal["none"]
TemporalRepresentation = Literal["event_stream", "snapshot_sequence"]
SpatialAggregation = Literal["sampled_neighbor", "full_neighbor"]
CouplingMode = Literal["coupled", "decoupled"]
TemporalStateUpdate = Literal["gru", "rnn", "transformer"]
StateKind = Literal["node_memory", "mailbox", "node_recurrent", "neighbor_recurrent", "model_recurrent"]


DEFAULT_RUNTIME_CONFIG: dict[str, Any] = {
    "device": "cuda",
    "access_pipeline": True,
    "train_compute_metrics": False,
    "temporal_state": {
        "update": "gru",
        "consistency": "bounded_stale",
        "max_staleness": 1,
        "filter": {
            "enabled": True,
            "min_change_norm": 0.0,
            "min_cosine_distance": 0.3,
            "max_skip": 10,
        },
        "smooth_aggregation": {
            "enabled": True,
            "gamma_init": 0.5,
        },
        "boundary_prediction": {"learnable": True, "gamma_init": 2.1972245773362196},
    },
    "train": {
        "batch_size": 3000,
        "epochs": 50,
        "optimizer": "adam",
        "lr": 0.0004,
        "weight_decay": 0.0,
        "dropout": 0.2,
        "att_dropout": 0.2,
        "max_batches_per_epoch": None,
    },
    "sampling": {
        "mode": None,
        "window": {
            "policy": None,
            "snaps_count": 8,
            "chunk_decay": "half",
            "num_full_snapshots": 2,
            "chunk_order": "rand",
        },
        "neighbor": {
            "fanouts": [20],
            "policy": "recent",
            "seed": 0,
            "workers": 32,
            "boundary_sampling": {
                "enabled": True,
                "uniform_policy": "boundary_uniform",
                "recent_policy": "boundary_decay_sampling",
                "probability": 0.1,
            },
        },
    },
    "preprocess": {
        "num_parts": 1,
        "chunks_per_rank": 1,
        "hot_node_ratio": 0.1,
        "feature_layout": "separate",
    },
}


@dataclass(frozen=True)
class StarrySpec:
    temporal: TemporalMode
    state: StateMode
    scope: ScopeMode
    consistency: ConsistencyMode = "exact"
    max_staleness: int = 0
    approximation: ApproximationMode = "none"

    def __post_init__(self) -> None:
        if self.max_staleness < 0:
            raise ValueError("max_staleness must be >= 0")


@dataclass(frozen=True)
class DataSource:
    source: str | None = None
    temporal_representation: TemporalRepresentation | None = None
    name: str | None = None


@dataclass(frozen=True)
class ModelBackbone:
    name: str
    temporal_representation: TemporalRepresentation | None = None
    spatial_aggregation: SpatialAggregation | None = None
    coupling: CouplingMode | None = None
    state: StateMode | None = None
    state_kind: StateKind | None = None
    state_key: str | None = None
    aggregate_key: str | None = None
    requires_temporal_state: bool | None = None


@dataclass(frozen=True)
class TaskSegment:
    name: str
    negative_sampler: NegativeSampler | None = None


@dataclass(frozen=True)
class NegativeSampler:
    """Minimal negative-sampling declaration.

    The sampling implementation is a task service and is intentionally not part
    of this first contract extraction.
    """

    ratio: int = 1
    mode: Literal["dst", "src_dst"] = "dst"
    policy: str = "random"
    sample_kwargs: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.ratio < 1:
            raise ValueError("negative sampling ratio must be >= 1")


@dataclass(frozen=True)
class EdgePrediction:
    negative_sampler: NegativeSampler | None = None
    ownership: Literal["edge"] = "edge"
    name: str = "edge_prediction"


@dataclass(frozen=True)
class NodeClassification:
    ownership: Literal["node"] = "node"
    name: str = "node_prediction"


@dataclass(frozen=True)
class NodeRegression:
    ownership: Literal["node"] = "node"
    name: str = "node_regression"


TaskDeclaration = Union[EdgePrediction, NodeClassification, NodeRegression, TaskSegment, Mapping[str, Any], str]


def normalize_spec(value: StarrySpec | Mapping[str, Any] | None) -> StarrySpec:
    if isinstance(value, StarrySpec):
        return value
    data = {} if value is None else dict(value)
    temporal = str(data.get("temporal", "event")).strip().lower()
    state = str(data.get("state", "stateless")).strip().lower()
    scope = str(data.get("scope", "sampled")).strip().lower()
    return StarrySpec(
        temporal=_checked(temporal, {"event", "snapshot"}, "temporal"),
        state=_checked(state, {"persistent", "stateless", "snapshot_recurrent"}, "state"),
        scope=_checked(scope, {"sampled", "full_graph"}, "scope"),
        consistency=_checked(
            str(data.get("consistency", "exact")).strip().lower(),
            {"exact", "bounded_stale"},
            "consistency",
        ),
        max_staleness=int(data.get("max_staleness", 0)),
        approximation=_checked(
            str(data.get("approximation", "none")).strip().lower(),
            {"none"},
            "approximation",
        ),
    )


def normalize_temporal_state(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    data = dict(value)
    if "exact" in data:
        raise ValueError("temporal_state uses consistency, not exact")
    if "approximation" in data:
        approximation = str(data.pop("approximation")).strip().lower()
        if approximation != "none":
            raise ValueError("temporal_state uses filter and smooth_aggregation, not approximation")
    consistency = _checked(str(data.get("consistency", "exact")).strip().lower(), {"exact", "bounded_stale"}, "consistency")
    max_staleness = int(data.get("max_staleness", 0 if consistency == "exact" else 1))
    result = {
        "consistency": consistency,
        "max_staleness": max_staleness,
    }
    update = data.get("update")
    if update is not None:
        result["update"] = _checked(str(update).strip().lower(), {"gru", "rnn", "transformer"}, "update")
    filter_cfg = data.get("filter")
    if filter_cfg is not None:
        if not isinstance(filter_cfg, Mapping):
            raise TypeError("temporal_state.filter must be a mapping")
        result["filter"] = dict(filter_cfg)
    smooth_cfg = data.get("smooth_aggregation")
    if smooth_cfg is not None:
        if not isinstance(smooth_cfg, Mapping):
            raise TypeError("temporal_state.smooth_aggregation must be a mapping")
        if "historical_mix" in smooth_cfg:
            raise ValueError("temporal_state.smooth_aggregation uses gamma_init, not historical_mix")
        result["smooth_aggregation"] = dict(smooth_cfg)
    boundary_cfg = data.get("boundary_prediction")
    if boundary_cfg is not None:
        if not isinstance(boundary_cfg, Mapping):
            raise TypeError("temporal_state.boundary_prediction must be a mapping")
        result["boundary_prediction"] = dict(boundary_cfg)
    return result


def normalize_runtime(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate runtime sections and merge the canonical public defaults."""

    if value is not None and not isinstance(value, Mapping):
        raise TypeError("runtime must be a mapping")
    provided = {} if value is None else dict(value)
    for name in ("temporal_state", "train", "sampling", "preprocess"):
        section = provided.get(name)
        if section is not None and not isinstance(section, Mapping):
            raise TypeError(f"runtime.{name} must be a mapping")

    result = _merge_mapping(DEFAULT_RUNTIME_CONFIG, provided)
    temporal_state = _merge_mapping(
        DEFAULT_RUNTIME_CONFIG["temporal_state"],
        normalize_temporal_state(provided.get("temporal_state")),
    )
    result["temporal_state"] = temporal_state
    _validate_runtime(result)
    return result


def normalize_data_source(value: DataSource | Mapping[str, Any] | str | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        return {"source": value}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, DataSource):
        result = {}
        if value.source is not None:
            result["source"] = value.source
        if value.name is not None:
            result["name"] = value.name
        if value.temporal_representation is not None:
            result["temporal_representation"] = value.temporal_representation
        return result
    raise TypeError("data_source must be a DataSource, mapping, string, or None")


def normalize_backbone(value: ModelBackbone | Mapping[str, Any] | str | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        return {"name": value}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, ModelBackbone):
        result = {"name": value.name}
        if value.temporal_representation is not None:
            result["temporal_representation"] = value.temporal_representation
        if value.spatial_aggregation is not None:
            result["spatial_aggregation"] = value.spatial_aggregation
        if value.coupling is not None:
            result["coupling"] = value.coupling
        if value.state is not None:
            result["state"] = value.state
        if value.state_kind is not None:
            result["state_kind"] = value.state_kind
        if value.state_key is not None:
            result["state_key"] = value.state_key
        if value.aggregate_key is not None:
            result["aggregate_key"] = value.aggregate_key
        if value.requires_temporal_state is not None:
            result["requires_temporal_state"] = value.requires_temporal_state
        return result
    raise TypeError("backbone must be a ModelBackbone, mapping, string, or None")


def normalize_task(task: TaskDeclaration | None) -> dict[str, Any]:
    if task is None:
        result: dict[str, Any] = {"name": "edge_prediction", "ownership": "edge"}
    elif isinstance(task, str):
        result = {"name": task}
    elif isinstance(task, Mapping):
        result = dict(task)
        if "ownership" in result:
            raise ValueError("task ownership is derived from task.name and must not be configured")
    elif isinstance(task, EdgePrediction):
        result = {"name": task.name, "ownership": task.ownership}
        if task.negative_sampler is not None:
            result["services"] = _negative_sampling_services(task.negative_sampler)
    elif isinstance(task, TaskSegment):
        result = {"name": task.name}
        if task.negative_sampler is not None:
            result["services"] = _negative_sampling_services(task.negative_sampler)
    elif isinstance(task, (NodeClassification, NodeRegression)):
        result = {"name": task.name, "ownership": task.ownership}
    else:
        raise TypeError("task must be a task declaration, mapping, string, or None")
    result["name"] = _checked(
        str(result.get("name", "")).strip().lower(),
        {"edge_prediction", "node_classification", "node_prediction", "node_regression"},
        "task",
    )
    expected_owner = "edge" if result["name"] == "edge_prediction" else "node"
    declared_owner = result.get("ownership")
    if declared_owner is not None and str(declared_owner).strip().lower() != expected_owner:
        raise ValueError(f"task {result['name']!r} must use ownership={expected_owner!r}")
    result["ownership"] = expected_owner
    if "train_loss_mode" in result:
        result["train_loss_mode"] = _checked(str(result["train_loss_mode"]),
            {"last_only", "window_mean"}, "task.train_loss_mode")
        if expected_owner != "node" and result["train_loss_mode"] != "last_only":
            raise ValueError("window_mean requires a snapshot node task")
    _validate_task(result)
    return result


def _negative_sampling_services(sampler: NegativeSampler) -> dict[str, Any]:
    return {
        "negative_sampling": {
            "ratio": sampler.ratio,
            "mode": sampler.mode,
            "policy": sampler.policy,
            "sample_kwargs": dict(sampler.sample_kwargs or {}),
        }
    }


def _validate_task(task: Mapping[str, Any]) -> None:
    services = task.get("services", {})
    if not isinstance(services, Mapping):
        raise TypeError("task.services must be a mapping")
    sampling = services.get("negative_sampling")
    if sampling is None:
        return
    if not isinstance(sampling, Mapping):
        raise TypeError("task.services.negative_sampling must be a mapping")
    if int(sampling.get("ratio", 1)) < 1:
        raise ValueError("negative sampling ratio must be >= 1")
    _checked(str(sampling.get("mode", "dst")).strip().lower(), {"dst", "src_dst"}, "negative sampling mode")
    for phase in ("train", "eval", "test"):
        config = sampling.get(phase)
        if config is None:
            continue
        if not isinstance(config, Mapping):
            raise TypeError(f"task.services.negative_sampling.{phase} must be a mapping")
        _checked(str(config.get("policy", "random")).strip().lower(), {"random"}, f"{phase} negative policy")
    train = sampling.get("train", {})
    if isinstance(train, Mapping):
        local = float(train.get("local_probability", 0.9))
        remote = float(train.get("remote_probability", 0.1))
        if local < 0 or remote < 0 or abs(local + remote - 1.0) > 1e-6:
            raise ValueError("negative train probabilities must be non-negative and sum to 1")
    if not isinstance(sampling.get("sample_kwargs", {}), Mapping):
        raise TypeError("task.services.negative_sampling.sample_kwargs must be a mapping")


def _checked(value: str, choices: set[str], field_name: str) -> Any:
    if value not in choices:
        raise ValueError(f"unknown {field_name}: {value!r}")
    return value


def _merge_mapping(defaults: Mapping[str, Any], provided: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(defaults))
    for key, value in provided.items():
        current = result.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            result[key] = _merge_mapping(current, value)
        else:
            result[key] = deepcopy(value)
    return result


def _validate_runtime(runtime: Mapping[str, Any]) -> None:
    temporal_state = runtime["temporal_state"]
    if temporal_state["consistency"] == "exact" and temporal_state["max_staleness"] != 0:
        raise ValueError("exact temporal state requires max_staleness=0")
    if temporal_state["consistency"] == "bounded_stale" and int(temporal_state["max_staleness"]) < 1:
        raise ValueError("bounded_stale temporal state requires max_staleness >= 1")
    if int(runtime["train"]["batch_size"]) <= 0:
        raise ValueError("runtime.train.batch_size must be positive")
    if int(runtime["train"]["epochs"]) <= 0:
        raise ValueError("runtime.train.epochs must be positive")
    if int(runtime["sampling"]["window"]["snaps_count"]) <= 0:
        raise ValueError("runtime.sampling.window.snaps_count must be positive")
    if runtime["sampling"]["mode"] not in {None, "full", "neighbor"}:
        raise ValueError("runtime.sampling.mode must be null, full, or neighbor")
    if runtime["sampling"]["window"]["policy"] not in {None, "event_window", "full_snapshot", "chunk_decay"}:
        raise ValueError(
            "runtime.sampling.window.policy must be null, event_window, full_snapshot, or chunk_decay"
        )
    boundary = runtime["sampling"]["neighbor"].get("boundary_sampling", {})
    if not isinstance(boundary, Mapping):
        raise TypeError("runtime.sampling.neighbor.boundary_sampling must be a mapping")
    probability = float(boundary.get("probability", 0.1))
    if not 0.0 <= probability <= 1.0:
        raise ValueError("boundary sampling probability must be in [0, 1]")
    if int(runtime["preprocess"]["num_parts"]) <= 0:
        raise ValueError("runtime.preprocess.num_parts must be positive")
    if int(runtime["preprocess"]["chunks_per_rank"]) <= 0:
        raise ValueError("runtime.preprocess.chunks_per_rank must be positive")
    hot_node_ratio = float(runtime["preprocess"]["hot_node_ratio"])
    if not 0.0 <= hot_node_ratio <= 1.0:
        raise ValueError("runtime.preprocess.hot_node_ratio must be in [0, 1]")
    if runtime["preprocess"]["feature_layout"] not in {"separate", "snapshot_csc"}:
        raise ValueError("runtime.preprocess.feature_layout must be separate or snapshot_csc")


def derive_spec(
    *,
    data: Mapping[str, Any],
    backbone: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> StarrySpec:
    """Derive canonical temporal/state/scope semantics from public sections."""

    name = str(backbone.get("name", "")).strip().lower().replace("-", "_")
    persistent = name in {"tgn", "jodie", "apan"}
    recurrent = name in {"gconv_gru", "dcrnn", "tgcn", "mpnn_lstm", "mpnnlstm", "evolvegcn", "evolve_gcn"}
    temporal_value = data.get("temporal_representation") or backbone.get("temporal_representation")
    temporal = {"event_stream": "event", "snapshot_sequence": "snapshot"}.get(str(temporal_value or "").lower())
    temporal = temporal or ("snapshot" if recurrent else "event")
    scope = {"sampled_neighbor": "sampled", "full_neighbor": "full_graph"}.get(
        str(backbone.get("spatial_aggregation", "")).lower()
    )
    scope = scope or ("full_graph" if recurrent else "sampled")
    state = str(backbone.get("state", "")).strip().lower()
    requires = backbone.get("requires_temporal_state")
    if requires is None and backbone.get("state_key") is not None:
        requires = str(backbone["state_key"]).strip() == "s"
    if not state:
        if requires is False or name == "tgat":
            state = "stateless"
        elif temporal == "snapshot" and (requires is True or recurrent):
            state = "snapshot_recurrent"
        elif requires is True or persistent or str(backbone.get("coupling", "")).lower() == "coupled":
            state = "persistent"
        else:
            state = "stateless"
    values: dict[str, Any] = {"temporal": temporal, "state": state, "scope": scope}
    temporal_state = runtime.get("temporal_state", {})
    if isinstance(temporal_state, Mapping):
        values.update({key: temporal_state[key] for key in ("consistency", "max_staleness") if key in temporal_state})
    return normalize_spec(values)


__all__ = [
    "ApproximationMode",
    "ConsistencyMode",
    "CouplingMode",
    "DEFAULT_RUNTIME_CONFIG",
    "DataSource",
    "derive_spec",
    "EdgePrediction",
    "NegativeSampler",
    "NodeClassification",
    "NodeRegression",
    "ScopeMode",
    "StateMode",
    "StateKind",
    "ModelBackbone",
    "SpatialAggregation",
    "TaskSegment",
    "TaskDeclaration",
    "TemporalMode",
    "TemporalRepresentation",
    "TemporalStateUpdate",
    "normalize_backbone",
    "normalize_data_source",
    "normalize_runtime",
    "normalize_task",
]
