from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from torch import nn

from starrygl.plan import ChunkBindingPlanner
from starrygl.runtime.trainer import Trainer, maybe_build_model_from_config
from starrygl.spec import (
    DataSource,
    ModelBackbone,
    TaskSegment,
    normalize_backbone,
    normalize_data_source,
    normalize_runtime,
    normalize_task,
)


def compile(
    *,
    data_source: DataSource | Mapping[str, Any] | str | None = None,
    backbone: ModelBackbone | Mapping[str, Any] | nn.Module | str | None = None,
    task_segment: TaskSegment | Mapping[str, Any] | str | None = None,
    artifact_root: str | Path | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> Trainer:
    """Compile canonical semantic segments into an executable trainer.

    Python callers use `data_source`, `backbone`, `task_segment`, and
    `runtime`. Config files use the same `data`, `backbone`, `task`, and
    `runtime` sections.
    """

    runtime_cfg = normalize_runtime(runtime)
    graph_cfg = normalize_data_source(data_source)
    task_cfg = normalize_task(task_segment)
    model_cfg, model_obj = _normalize_backbone(backbone, runtime=runtime_cfg, task=task_cfg)
    plan = ChunkBindingPlanner().compile(
        data=graph_cfg,
        backbone=model_cfg,
        task=task_cfg,
        runtime=runtime_cfg,
    )
    trainer_runtime = dict(runtime_cfg)
    train_cfg = trainer_runtime.pop("train")
    preprocess_cfg = trainer_runtime.pop("preprocess")
    return Trainer(
        graph=graph_cfg,
        model_config=model_cfg,
        model=model_obj,
        task=task_cfg,
        spec=plan.spec,
        plan=plan,
        artifact_root=artifact_root,
        train_config=train_cfg,
        runtime_config=trainer_runtime,
        preprocess_config=preprocess_cfg,
    )


def from_config(
    config_or_path: Mapping[str, Any] | str | Path,
    *,
    artifact_root: str | Path | None = None,
) -> Trainer:
    cfg = load_config(config_or_path)
    reject_config_aliases(cfg)
    config_artifact_root = cfg.get("artifact_root")
    if config_artifact_root is not None and not isinstance(config_artifact_root, (str, Path)):
        raise TypeError("top-level artifact_root must be a string path")
    model = section(cfg, "backbone")
    runtime = section(cfg, "runtime")
    if "artifact_root" in runtime:
        raise ValueError("artifact_root belongs at config top level, not under runtime")
    return compile(
        data_source=section(cfg, "data"),
        backbone=model,
        task_segment=section(cfg, "task"),
        artifact_root=artifact_root if artifact_root is not None else config_artifact_root,
        runtime=runtime,
    )


def load_config(config_or_path: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config_or_path, Mapping):
        return _expand_config(dict(config_or_path))
    path = Path(config_or_path).expanduser()
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, Mapping):
        raise ValueError("config must be a JSON object")
    return _expand_config(dict(data))


def section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"config section {name!r} must be a mapping")
    return dict(value)


def reject_config_aliases(config: Mapping[str, Any]) -> None:
    legacy = sorted(
        {
            "execution",
            "gnn",
            "graph",
            "memory",
            "model",
            "preprocess",
            "sampling",
            "spec",
            "temporal_state",
            "train",
        }.intersection(config)
    )
    if legacy:
        raise ValueError(
            f"config uses legacy section name(s): {', '.join(legacy)}; "
            "use data, backbone, task, and runtime"
        )


def _expand_config(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_config(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand_config(item) for item in value)
    if isinstance(value, Mapping):
        return {key: _expand_config(item) for key, item in value.items()}
    return value


def _normalize_backbone(
    backbone: ModelBackbone | Mapping[str, Any] | nn.Module | str | None,
    *,
    runtime: Mapping[str, Any],
    task: Mapping[str, Any],
) -> tuple[dict[str, Any], Any]:
    if backbone is None:
        return {}, None
    if isinstance(backbone, nn.Module):
        semantics = getattr(backbone, "backbone_semantics", None)
        if semantics is None:
            return {"name": type(backbone).__name__}, backbone
        config = normalize_backbone(semantics)
        config.setdefault("name", type(backbone).__name__)
        return config, backbone
    config = normalize_backbone(backbone)
    if str(task.get("ownership", "")) == "node" and "node_output_dim" not in config:
        return config, None
    temporal_state = runtime.get("temporal_state")
    return config, maybe_build_model_from_config(
        config,
        temporal_state=temporal_state if isinstance(temporal_state, Mapping) else None,
        training=runtime.get("train") if isinstance(runtime.get("train"), Mapping) else None,
    )


__all__ = ["compile", "from_config"]
