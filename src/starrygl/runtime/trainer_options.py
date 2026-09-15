from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from starrygl.model import StarryModel
from starrygl.partition import PartitionPlan
from starrygl.plan import ChunkBindingPlanner
from starrygl.runtime.builders import build_model_from_config
from starrygl.batch import BatchMode, SamplingPolicy, WindowPolicy
from starrygl.store import StoreBundle, load_starrygl_store
from starrygl.runtime.state.build import build_state_managers, model_device


class TrainerOptions:
    def _prepare_source(self, graph: Any | None) -> Any:
        if graph is not None:
            return graph
        if self.graph:
            return dict(self.graph)
        raise ValueError("Trainer.prepare requires data.source or a graph argument")

    def _store(
        self,
        store: StoreBundle | None,
        *,
        artifact_root: str | Path | None,
        rank: int,
        map_location: str | torch.device,
        mmap: bool,
    ) -> StoreBundle:
        if store is not None:
            return self._bind_store_plan(store)
        root = artifact_root if artifact_root is not None else self.artifact_root
        if root is None:
            raise ValueError("Trainer.fit/evaluate/predict requires a StoreBundle or artifact_root")
        key = (str(Path(root).expanduser().resolve()), int(rank), str(map_location), bool(mmap))
        if self._runtime_store_key == key and self._runtime_store is not None:
            return self._runtime_store
        loaded = load_starrygl_store(root, rank=int(rank), map_location=map_location, mmap=bool(mmap))
        self._validate_artifact_store(loaded)
        self._runtime_store = self._bind_store_plan(loaded)
        self._runtime_store_key = key
        return self._runtime_store

    def _bind_store_plan(self, store: StoreBundle) -> StoreBundle:
        partition = store.graph.partition
        if partition:
            self.plan = ChunkBindingPlanner().compile(
                data=self.graph,
                backbone=self.model_config,
                task=self.task,
                runtime=self.runtime_config,
                partition_plan=PartitionPlan.from_artifact(partition),
            )
        return store

    def _model(self, model: StarryModel | None, store: StoreBundle) -> StarryModel:
        if model is not None:
            return model
        if isinstance(self.model, StarryModel):
            return self.model
        if isinstance(self.model, nn.Module):
            return self.model
        temporal_state = self.runtime_config.get("temporal_state")
        self.model = build_model_from_config(
            self._model_config_for_store(store),
            store=store,
            temporal_state=temporal_state if isinstance(temporal_state, Mapping) else None,
            training=self.train_config,
        )
        return self.model

    def _model_config_for_store(self, store: StoreBundle) -> Mapping[str, Any]:
        config = dict(self.model_config)
        task_name = str(self.task.get("name", "")).strip().lower()
        if task_name not in {"node_prediction", "node_classification", "node_regression"}:
            return config
        if "node_output_dim" not in config:
            task_loss = str(self.task.get("loss", "mse" if task_name == "node_regression" else "cross_entropy")).strip().lower()
            output_dim = _infer_node_output_dim(store, loss=task_loss)
            if output_dim is not None:
                config["node_output_dim"] = int(output_dim)
        return config

    def _optimizer(self, model: StarryModel) -> torch.optim.Optimizer:
        name = str(self.train_config.get("optimizer", "adam")).lower()
        lr = float(self.train_config.get("lr", 1e-3))
        weight_decay = float(self.train_config.get("weight_decay", 0.0))
        if name == "sgd":
            return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        if name == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        if name != "adam":
            raise ValueError(f"unsupported optimizer: {name!r}")
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def _state_manager(
        self,
        manager: object | Mapping[str, object] | None,
        *,
        model: StarryModel,
        store: StoreBundle,
        device: str | torch.device | None,
        comm: object,
        window_policy: WindowPolicy | None = None,
        num_full_snapshots: int | None = None,
        chunk_decay: Sequence[int] | torch.Tensor | None = None,
    ) -> object | Mapping[str, object] | None:
        if manager is not None or not self.plan.state_dependencies:
            return manager
        target = model_device(model, device)
        decay = self._chunk_decay(chunk_decay) if self._window_policy(window_policy) == "chunk_decay" else None
        window_size = max(1, self._num_full_snapshots(num_full_snapshots))
        if decay is not None:
            window_size += int((torch.as_tensor(decay) >= 0).sum().item())
        key = (id(model), id(store), str(target), self.spec.consistency, self.spec.max_staleness, window_size)
        if self._runtime_state_key != key:
            self._runtime_state_manager = build_state_managers(
                model=model,
                store=store,
                kinds=tuple(dep.kind for dep in self.plan.state_dependencies),
                temporal_state=self.runtime_config.get("temporal_state", {}),
                device=target,
                comm=comm,
                window_size=window_size,
            )
            self._runtime_state_key = key
        return self._runtime_state_manager

    def _batch_mode(self, mode: BatchMode | None) -> BatchMode:
        if mode is not None:
            return mode
        return "snapshot" if self.spec.temporal == "snapshot" else "event"

    def _device(self, value: str | torch.device | None) -> torch.device | None:
        configured = value if value is not None else self.runtime_config.get("device")
        return None if configured is None else torch.device(configured)

    def _sampling_config(self) -> Mapping[str, Any]:
        sampling = self.runtime_config.get("sampling")
        return sampling if isinstance(sampling, Mapping) else {}

    def _sampling_window(self) -> Mapping[str, Any]:
        window = self._sampling_config().get("window")
        return window if isinstance(window, Mapping) else {}

    def _sampling_neighbor(self) -> Mapping[str, Any]:
        neighbor = self._sampling_config().get("neighbor")
        return neighbor if isinstance(neighbor, Mapping) else {}

    def _window_policy(self, value: WindowPolicy | None) -> WindowPolicy:
        if value is not None:
            return value
        return str(self.plan.window_policy)  # type: ignore[return-value]

    def _sampling_policy(self, value: SamplingPolicy | None) -> SamplingPolicy:
        if value is not None:
            return value
        return str(self.plan.sampling_policy)  # type: ignore[return-value]

    def _chunk_decay(self, value: Sequence[int] | torch.Tensor | str | None) -> Sequence[int] | torch.Tensor | None:
        raw = value if value is not None else self._sampling_window().get("chunk_decay")
        if raw is None:
            return None
        if isinstance(raw, str):
            return _parse_chunk_decay(
                raw,
                chunk_count=int(self.preprocess_config.get("chunks_per_rank", 1)),
                snaps_count=self._snaps_count(),
                fulls_count=self._num_full_snapshots(None),
            )
        return raw

    def _num_layers(self, value: int | None) -> int:
        if value is not None:
            return int(value)
        if "num_layers" in self.model_config:
            return int(self.model_config["num_layers"])
        return 1

    def _num_full_snapshots(self, value: int | None) -> int:
        if value is not None:
            return int(value)
        return int(self._sampling_window().get("num_full_snapshots", 1))

    def _snaps_count(self) -> int:
        return int(self._sampling_window().get("snaps_count", self._num_full_snapshots(None)))

    def _fanouts(self, value: Sequence[int] | torch.Tensor | None) -> Sequence[int] | torch.Tensor | None:
        if value is not None:
            return value
        neighbor = self._sampling_neighbor()
        if "fanouts" in neighbor:
            return tuple(int(v) for v in neighbor["fanouts"])
        return None

    def _sampler_options(
        self,
        value: Mapping[str, Any] | None,
        *,
        fanouts: Sequence[int] | torch.Tensor | None = None,
        train: bool | None = None,
        split: str | None = None,
    ) -> Mapping[str, Any]:
        options: dict[str, Any] = self._negative_sampler_options(train=train, split=split)
        sampling = self._sampling_config()
        if isinstance(sampling, Mapping):
            options.update({key: value for key, value in sampling.items() if key not in {"mode", "window", "neighbor"}})
        neighbor = self._sampling_neighbor()
        if "policy" in neighbor:
            options["policy"] = neighbor["policy"]
        if "workers" in neighbor:
            options["workers"] = neighbor["workers"]
        if "seed" in neighbor:
            options["seed"] = neighbor["seed"]
        boundary = neighbor.get("boundary_sampling", {})
        if isinstance(boundary, Mapping) and bool(boundary.get("enabled", False)):
            policy = str(options.get("policy", "recent")).strip().lower()
            if policy in {"uniform", "random"}:
                options["policy"] = boundary.get("uniform_policy", "boundary_uniform")
            elif policy in {"recent", "latest"}:
                options["policy"] = boundary.get("recent_policy", "boundary_decay_sampling")
            if str(options.get("policy", "")).startswith(("boundary_", "boundery_")):
                options["probability"] = float(boundary.get("probability", 0.1))
        window = self._sampling_window()
        if "chunk_order" in window:
            options["chunk_order"] = window["chunk_order"]
        for key in (
            "profile_runtime",
            "access_pipeline",
            "defer_node_feature_finish",
            "defer_feature_launch",
            "snapshot_materialize_on_device",
            "feature_cache_on_device",
            "snapshot_row_cache_on_device",
            "snapshot_precompute_edge_rows",
            "snapshot_sparse_gcn",
            "snapshot_dgl_gcn",
            "rolling_snapshot_cache",
            "full_snapshot_chunk_limit",
        ):
            if key in self.runtime_config:
                options[key.removeprefix("sampler_")] = self.runtime_config[key]
        if "feature_cache_on_device" not in options:
            feature_device = str(self.runtime_config.get("feature_device", "")).strip().lower()
            if feature_device in {"cuda", "gpu"} or feature_device.startswith("cuda:"):
                options["feature_cache_on_device"] = True
        if "chunks_per_rank" not in options:
            options["chunks_per_rank"] = int(self.preprocess_config.get("chunks_per_rank", 1))
        if value is not None:
            options.update(dict(value))
        maximum = self.train_config.get("max_batches_per_epoch")
        if maximum is not None:
            options.setdefault("max_batches_per_epoch", int(maximum))
        resolved_fanouts = self._fanouts(fanouts)
        if resolved_fanouts is not None and "fanouts" not in options:
            options["fanouts"] = tuple(int(v) for v in resolved_fanouts)
        return options

    def _negative_sampler_options(self, *, train: bool | None, split: str | None) -> dict[str, Any]:
        if train is None or str(self.task.get("name", "")).strip().lower() != "edge_prediction":
            return {}
        services = self.task.get("services", {})
        sampling = services.get("negative_sampling", {}) if isinstance(services, Mapping) else {}
        sampling = sampling if isinstance(sampling, Mapping) else {}
        phase = "train" if train else "test" if str(split).lower() == "test" else "eval"
        phase_config = sampling.get(phase, {})
        phase_config = phase_config if isinstance(phase_config, Mapping) else {}
        options = dict(sampling.get("sample_kwargs", {}))
        options["negative_mode"] = str(sampling.get("mode", "dst"))
        if phase == "train":
            options["negative_local_prob"] = float(phase_config.get("local_probability", 0.9))
            options["negative_global_prob"] = float(phase_config.get("remote_probability", 0.1))
            options["negative_global_pool"] = "remote_dst"
        else:
            options.update(negative_local_prob=0.0, negative_global_prob=1.0, negative_global_pool="global_dst")
            options["eval_negative_dst_pool"] = "global_dst"
        return options

    def _num_negatives(self, value: int | None) -> int:
        if value is not None:
            return int(value)
        services = self.task.get("services", {})
        service = services.get("negative_sampling", {}) if isinstance(services, Mapping) else {}
        if service is None:
            return 0
        if isinstance(service, Mapping) and "ratio" in service:
            return int(service["ratio"])
        return 1 if str(self.task.get("name", "")) == "edge_prediction" else 0

    def _gradient_sync(self) -> str | None:
        value = self.runtime_config.get("gradient_sync")
        if value is None:
            return None
        return str(value)

    def _dist_backend(self, device: str | torch.device | None) -> str | None:
        value = self.runtime_config.get("dist_backend")
        if value is not None:
            return str(value)
        if device is not None and torch.device(device).type == "cpu":
            return "gloo"
        return "nccl" if torch.cuda.is_available() else "gloo"

    def _include_static_one_hop_default(self, *, has_snapshot_view: bool) -> bool:
        return bool(self.preprocess_config.get("include_static_one_hop", has_snapshot_view))

    def _config_split_ratios(self) -> tuple[float, float, float]:
        if "split_ratios" not in self.preprocess_config and (
            "train_ratio" in self.preprocess_config or "val_ratio" in self.preprocess_config
        ):
            train = float(self.preprocess_config.get("train_ratio", 1.0))
            val = float(self.preprocess_config.get("val_ratio", 0.0))
            return (train, val, max(0.0, 1.0 - train - val))
        value = self.preprocess_config.get("split_ratios", (1.0, 0.0, 0.0))
        if isinstance(value, Mapping):
            return (
                float(value.get("train", 1.0)),
                float(value.get("val", 0.0)),
                float(value.get("test", 0.0)),
            )
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError("preprocess.split_ratios must be a 3-value sequence or train/val/test mapping")
        return (float(value[0]), float(value[1]), float(value[2]))


def _parse_chunk_decay(
    pattern: str,
    *,
    chunk_count: int,
    snaps_count: int,
    fulls_count: int,
) -> tuple[int, ...]:
    if chunk_count < 1 or fulls_count < 1:
        raise ValueError("chunks_per_rank and num_full_snapshots must be >= 1")
    if fulls_count > snaps_count:
        raise ValueError("num_full_snapshots must be <= snaps_count")
    decay_count = snaps_count - fulls_count
    if decay_count <= 0:
        return ()
    text = str(pattern).strip()
    mode = text.split(":", 1)[0].lower()
    if mode in {"auto", "half"}:
        if mode == "auto" and ":" not in text:
            raise ValueError("auto chunk_decay must use auto:<ratio>")
        ratio = float(text.split(":", 1)[1]) if mode == "auto" else 0.5
        alpha = max(0.0, min(ratio, 1.0)) ** (1.0 / decay_count) if mode == "auto" else ratio
        return tuple(max(0, min(round(chunk_count * alpha**i), chunk_count)) for i in range(1, decay_count + 1))
    values = [
        max(0, min(round(chunk_count * number), chunk_count))
        if 0.0 <= number <= 1.0 and ("." in item or "e" in item.lower())
        else max(0, min(int(round(number)), chunk_count))
        for item in filter(None, (part.strip() for part in text.split(",")))
        for number in (float(item),)
    ]
    if len(values) != decay_count:
        raise ValueError(f"chunk_decay must provide {decay_count} values")
    if any(left < right for left, right in zip(values, values[1:])):
        raise ValueError("chunk_decay values must be non-increasing")
    return tuple(values)


def _infer_node_output_dim(store: StoreBundle | None, *, loss: str = "cross_entropy") -> int | None:
    value = None if store is None else store.labels.node_label
    if not isinstance(value, torch.Tensor) or not value.numel():
        return None
    if str(loss).lower() == "mse":
        if store is not None and store.labels.node_label_temporal and value.dim() == 2:
            return 1
        return int(value.shape[-1]) if value.dim() > 1 else 1
    if store is not None and store.labels.node_label_temporal and value.dim() == 2:
        return int(value.long().amax().item()) + 1
    return int(value.shape[-1]) if value.dim() > 1 else int(value.long().amax().item()) + 1

__all__ = ["TrainerOptions"]
