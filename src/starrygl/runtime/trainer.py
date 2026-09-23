from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import torch

from starrygl.model import ModelOutput
from starrygl.plan import ChunkBindingPlanner, ExecutionPlan
from starrygl.prepare import PREPARE_FORMAT, PrepareConfig, PreparedViews, load_graph_data, materialize_graph_data, partition_graph_data
from starrygl.runtime.builders import build_model_from_config, build_task_from_config, maybe_build_model_from_config
from starrygl.runtime.loop import EpochResult
from starrygl.spec import StarrySpec
from starrygl.store import build_label_shards, build_static_feature_shards, write_prepare_artifacts
from starrygl.store.artifact import artifact_fingerprint
from starrygl.utils import DistributedContext


from starrygl.runtime.train import evaluate as evaluate_trainer
from starrygl.runtime.train import fit as fit_trainer
from starrygl.runtime.train import predict as predict_trainer
from starrygl.runtime.trainer_options import TrainerOptions

@dataclass
class Trainer(TrainerOptions):
    """Compiled StarryGL declaration and prepare/train/evaluate facade."""

    graph: Mapping[str, Any]
    model_config: Mapping[str, Any]
    model: Any
    task: Mapping[str, Any]
    spec: StarrySpec
    plan: ExecutionPlan
    artifact_root: str | Path | None = None
    train_config: Mapping[str, Any] = field(default_factory=dict)
    runtime_config: Mapping[str, Any] = field(default_factory=dict)
    preprocess_config: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _runtime_store: Any = field(default=None, init=False, repr=False)
    _runtime_store_key: tuple[Any, ...] | None = field(default=None, init=False, repr=False)
    _runtime_state_manager: Any = field(default=None, init=False, repr=False)
    _runtime_state_key: tuple[Any, ...] | None = field(default=None, init=False, repr=False)

    def prepare(
        self,
        *,
        graph: Any | None = None,
        world_size: int | None = None,
        chunks_per_rank: int | None = None,
        node_master: Any | None = None,
        edge_master: Any | None = None,
        hot_node_ids: Any | None = None,
        node_to_chunk: Any | None = None,
        save: bool = False,
        artifact_root: str | Path | None = None,
        include_static_one_hop: bool | None = None,
    ) -> PreparedViews:
        """Prepare graph views through the target StarryGL prepare path."""

        data = self._prepare_source(graph)
        node_master, edge_master, hot_node_ids, node_to_chunk = self._partition_inputs(
            node_master, edge_master, hot_node_ids, node_to_chunk
        )
        cfg = self._prepare_config(world_size=world_size, chunks_per_rank=chunks_per_rank)
        graph_data = _with_config_random_features(load_graph_data(data))
        partition_plan = partition_graph_data(
            graph_data,
            config=cfg,
            node_master=node_master,
            edge_master=edge_master,
            hot_node_ids=hot_node_ids,
            node_to_chunk=node_to_chunk,
        )
        self.plan = ChunkBindingPlanner().compile(
            data=self.graph,
            backbone=self.model_config,
            task=self.task,
            runtime=self.runtime_config,
            partition_plan=partition_plan,
        )
        prepared = materialize_graph_data(
            graph_data,
            config=cfg,
            partition_plan=partition_plan,
            view_plan=self.plan.view,
        )
        self._runtime_store = None
        self._runtime_store_key = None
        self._runtime_state_manager = None
        self._runtime_state_key = None
        if save:
            root = artifact_root if artifact_root is not None else self.artifact_root
            if root is None:
                raise ValueError("Trainer.prepare(save=True) requires artifact_root")
            include_one_hop = (
                self._include_static_one_hop_default(has_snapshot_view=self.plan.view.requires("snapshot_csc"))
                if include_static_one_hop is None
                else bool(include_static_one_hop)
            )
            prepared.meta["artifact_signature"] = self._artifact_signature(
                data=data,
                config=cfg,
                node_master=node_master,
                edge_master=edge_master,
                hot_node_ids=hot_node_ids,
                include_static_one_hop=include_one_hop,
                node_to_chunk=node_to_chunk,
            )
            feature_shards = build_static_feature_shards(
                prepared=prepared,
                node_feat=graph_data.node_feat,
                edge_feat=graph_data.edge_feat,
                include_static_one_hop=include_one_hop,
                replicate_node_features=bool(
                    self.preprocess_config.get("replicate_node_features", False)
                ),
            )
            label_shards = build_label_shards(
                prepared=prepared,
                node_label=graph_data.node_label,
                node_label_nodes=graph_data.node_label_nodes,
                node_label_ts=graph_data.node_label_ts,
                node_label_split=graph_data.node_label_split,
                node_label_temporal=graph_data.node_label_temporal,
                node_label_horizon=graph_data.node_label_horizon,
                edge_label=graph_data.edge_label,
                task=self.task,
                temporal=self.spec.temporal,
                src=graph_data.src,
                dst=graph_data.dst,
                ts=graph_data.ts,
                edge_ids=graph_data.edge_ids,
            )
            write_prepare_artifacts(
                root,
                prepared=prepared,
                feature_shards=feature_shards,
                label_shards=label_shards,
                feature_layout=str(self.preprocess_config.get("feature_layout", "separate")),
            )
        return prepared

    def prepare_artifacts(
        self,
        *,
        artifact_root: str | Path | None = None,
        world_size: int | None = None,
        force: bool = False,
    ) -> Path:
        world_size = int(self.preprocess_config.get("num_parts", 1) if world_size is None else world_size)
        root = Path(artifact_root if artifact_root is not None else self.artifact_root or "artifacts").resolve()
        if not bool(force) and self._artifacts_ready(root, world_size=world_size):
            return root
        self.prepare(world_size=world_size, save=True, artifact_root=root)
        return root

    def _prepare_config(
        self,
        *,
        world_size: int | None,
        chunks_per_rank: int | None = None,
    ) -> PrepareConfig:
        time_split = str(
            self.preprocess_config.get(
                "time_split",
                "batch" if self.spec.temporal == "event" else "equal_edges",
            )
        )
        target_batch_size = self.preprocess_config.get("target_batch_size")
        if target_batch_size is None and time_split in {"batch", "adaptive_batch"}:
            target_batch_size = self.train_config.get("batch_size")
        return PrepareConfig(
            world_size=int(world_size if world_size is not None else self.preprocess_config.get("num_parts", 1)),
            chunks_per_rank=int(
                chunks_per_rank if chunks_per_rank is not None else self.preprocess_config.get("chunks_per_rank", 1)
            ),
            time_split=time_split,
            target_batch_size=None if target_batch_size is None else int(target_batch_size),
            partition_backend=_prepare_partition_backend(
                str(self.preprocess_config.get("partition_backend", "speed_partition"))
            ),
            speed_partition_beta=float(self.preprocess_config.get("speed_partition_beta", 0.1)),
            speed_partition_topk_ratio=float(self.preprocess_config.get("hot_node_ratio", 0.1)),
            speed_partition_topk_type=str(self.preprocess_config.get("speed_partition_topk_type", "degree")),
            split_ratios=self._config_split_ratios(),
            include_state_write_routes=bool(self.preprocess_config.get("include_state_write_routes", True)),
            profile_prepare=bool(self.preprocess_config.get("profile_prepare", False)),
        )

    def _partition_inputs(self, node_master, edge_master, hot_node_ids, node_to_chunk=None):
        values = (node_master, edge_master, hot_node_ids, node_to_chunk)
        names = ("node_master_source", "edge_master_source", "hot_node_ids_source", "node_to_chunk_source")
        return tuple(
            value if value is not None else _load_tensor_config(self.preprocess_config.get(name), name)
            for value, name in zip(values, names)
        )

    def _artifact_signature(
        self,
        *,
        data: Any,
        config: PrepareConfig,
        node_master: Any,
        edge_master: Any,
        hot_node_ids: Any,
        include_static_one_hop: bool,
        node_to_chunk: Any = None,
    ) -> str:
        prepare = {key: value for key, value in vars(config).items() if key != "profile_prepare"}
        partition_inputs = (node_master, edge_master, hot_node_ids)
        if node_to_chunk is not None:
            partition_inputs += (node_to_chunk,)
        signature = {
            "format": PREPARE_FORMAT,
            "data": data,
            "prepare": prepare,
            "view": self.plan.view.as_dict(),
            "task": self.task.get("name"),
            "temporal": self.spec.temporal,
            "partition_inputs": partition_inputs,
            "feature_layout": str(self.preprocess_config.get("feature_layout", "separate")),
            "include_static_one_hop": bool(include_static_one_hop),
        }
        if self.preprocess_config.get("replicate_node_features", False):
            signature["replicate_node_features"] = True
        return artifact_fingerprint(signature)

    def _validate_artifact_store(self, store) -> None:
        meta = store.graph.prepare.get("meta", {}) if store.graph.prepare is not None else {}
        actual = meta.get("artifact_signature")
        if not isinstance(actual, str):
            raise ValueError("prepared artifacts have no signature; rebuild them with Trainer.prepare_artifacts()")
        world_size = (
            int(torch.distributed.get_world_size())
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else int(self.preprocess_config.get("num_parts", 1))
        )
        node_master, edge_master, hot_node_ids, node_to_chunk = self._partition_inputs(None, None, None)
        expected = self._artifact_signature(
            data=self._prepare_source(None),
            config=self._prepare_config(world_size=world_size),
            node_master=node_master,
            edge_master=edge_master,
            hot_node_ids=hot_node_ids,
            node_to_chunk=node_to_chunk,
            include_static_one_hop=self._include_static_one_hop_default(
                has_snapshot_view=self.plan.view.requires("snapshot_csc")
            ),
        )
        if actual != expected:
            raise ValueError("prepared artifacts do not match this data/preprocess/view configuration")

    def _artifacts_ready(self, root: str | Path, *, world_size: int = 1) -> bool:
        path = Path(root)
        prepare_path = path / "prepare.pt"
        if not prepare_path.exists():
            return False
        feature_layout = str(self.preprocess_config.get("feature_layout", "separate"))
        prepare = torch.load(prepare_path, map_location="cpu", weights_only=False)
        meta = prepare.get("meta", {})
        if meta.get("feature_layout", "separate") != feature_layout:
            return False
        if int(meta.get("world_size", 1)) != int(world_size):
            return False
        node_master, edge_master, hot_node_ids, node_to_chunk = self._partition_inputs(None, None, None)
        signature = self._artifact_signature(
            data=self._prepare_source(None),
            config=self._prepare_config(world_size=world_size),
            node_master=node_master,
            edge_master=edge_master,
            hot_node_ids=hot_node_ids,
            node_to_chunk=node_to_chunk,
            include_static_one_hop=self._include_static_one_hop_default(
                has_snapshot_view=self.plan.view.requires("snapshot_csc")
            ),
        )
        if meta.get("artifact_signature") != signature:
            return False
        for rank in range(int(world_size)):
            if not (path / f"graph_{rank:03d}.pt").exists():
                return False
            if not (path / f"feature_{rank:03d}.pt").exists():
                return False
            if not (path / f"label_{rank:03d}.pt").exists():
                return False
        return True

    def fit(self, **kwargs: Any) -> list[EpochResult]:
        """Train through the compiled execution plan."""
        return fit_trainer(self, **kwargs)

    def evaluate(self, **kwargs: Any) -> EpochResult:
        """Evaluate through the same batch and state pipeline."""
        return evaluate_trainer(self, **kwargs)

    def predict(self, **kwargs: Any) -> list[ModelOutput]:
        """Enumerate compiled batches and return model outputs."""
        return predict_trainer(self, **kwargs)

    def fit_distributed(
        self,
        *,
        prepare: bool = False,
        auto_prepare: bool = True,
        artifact_root: str | Path | None = None,
        device: str | torch.device | None = None,
        shutdown: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Launch distributed training for the current process."""
        context, store = _distributed_setup(
            self, "fit_distributed", prepare, auto_prepare, artifact_root, device, kwargs
        )
        rows: list[Mapping[str, Any]] = []
        user_callback = kwargs.pop("epoch_callback", None)

        def callback(row: Mapping[str, Any]) -> None:
            rows.append(dict(row))
            if user_callback is not None:
                user_callback(row)

        model = self._model(None, store).to(context.device)
        epochs = self.fit(
            store=store,
            model=model,
            task=build_task_from_config(self.task),
            optimizer=self._optimizer(model),
            rank=context.rank,
            device=context.device,
            epoch_callback=callback,
            **kwargs,
        )
        return _finish_distributed(
            {
                "rank": context.rank,
                "world_size": context.world_size,
                "epochs": [_epoch_result_dict(epoch) for epoch in epochs],
                "epoch_seconds": [float(row.get("train_seconds", 0.0)) for row in rows],
                "epoch_rows": [_jsonable_epoch_row(row) for row in rows],
                "plan": self.plan.as_dict(),
            },
            context,
            shutdown,
        )

    def eval_distributed(
        self,
        *,
        prepare: bool = False,
        auto_prepare: bool = True,
        artifact_root: str | Path | None = None,
        device: str | torch.device | None = None,
        shutdown: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Launch distributed evaluation for the current process."""
        context, store = _distributed_setup(
            self, "eval_distributed", prepare, auto_prepare, artifact_root, device, kwargs
        )
        result = self.evaluate(
            store=store,
            model=self._model(None, store).to(context.device),
            task=build_task_from_config(self.task),
            rank=context.rank,
            device=context.device,
            **kwargs,
        )
        return _finish_distributed(
            {
                "rank": context.rank,
                "world_size": context.world_size,
                "eval": _epoch_result_dict(result),
                "plan": self.plan.as_dict(),
            },
            context,
            shutdown,
        )

    def to_config(self) -> dict[str, Any]:
        runtime = dict(self.runtime_config)
        runtime["train"] = dict(self.train_config)
        runtime["preprocess"] = dict(self.preprocess_config)
        config: dict[str, Any] = {
            "data": dict(self.graph),
            "backbone": dict(self.model_config),
            "task": {key: value for key, value in self.task.items() if key != "ownership"},
            "runtime": runtime,
        }
        if self.artifact_root is not None:
            config["artifact_root"] = str(self.artifact_root)
        return config


def _distributed_setup(self, operation, prepare, auto_prepare, artifact_root, device, kwargs):
    device = self._device(device)
    context = DistributedContext.init(backend=self._dist_backend(device), device=device)
    root = artifact_root if artifact_root is not None else self.artifact_root
    if root is None:
        raise ValueError(f"{operation} requires artifact_root")
    if context.rank == 0 and (
        prepare or (auto_prepare and not self._artifacts_ready(root, world_size=context.world_size))
    ):
        self.prepare(save=True, artifact_root=root, world_size=context.world_size)
    context.barrier()
    store = self._store(
        None,
        artifact_root=root,
        rank=context.rank,
        map_location="cpu",
        mmap=bool(kwargs.pop("mmap", False)),
    )
    return context, store


def _finish_distributed(
    result: dict[str, Any],
    context: DistributedContext,
    shutdown: bool,
) -> dict[str, Any]:
    context.barrier()
    if shutdown:
        context.shutdown()
    return result


def _epoch_result_dict(result: EpochResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "loss": float(result.loss),
        "steps": int(result.steps),
        "metrics": {key: float(value) for key, value in result.metrics.items()},
    }


def _jsonable_epoch_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "epoch": int(row.get("epoch", 0)),
        "train_seconds": float(row.get("train_seconds", 0.0)),
        "train_seconds_cumulative": float(row.get("train_seconds_cumulative", 0.0)),
        "train": _epoch_result_dict(row.get("train")),
    }


def _with_config_random_features(graph_data):
    meta = graph_data.meta if isinstance(graph_data.meta, Mapping) else {}
    node_feat, edge_feat = graph_data.node_feat, graph_data.edge_feat
    if node_feat is None and int(meta.get("random_node_feat_dim", 0) or 0) > 0:
        generator = torch.Generator().manual_seed(int(meta.get("random_node_feat_seed", 0) or 0))
        node_feat = torch.randn(graph_data.num_nodes, int(meta["random_node_feat_dim"]), generator=generator)
    if edge_feat is None and int(meta.get("random_edge_feat_dim", 0) or 0) > 0:
        generator = torch.Generator().manual_seed(int(meta.get("random_edge_feat_seed", 0) or 0))
        edge_feat = torch.randn(graph_data.edge_ids.numel(), int(meta["random_edge_feat_dim"]), generator=generator)
    return graph_data if node_feat is graph_data.node_feat and edge_feat is graph_data.edge_feat else replace(graph_data, node_feat=node_feat, edge_feat=edge_feat)


def _prepare_partition_backend(value: str) -> str:
    name = str(value).strip().lower().replace("-", "_")
    if name in {"speed", "node_id_chunk_balance", "chunk_balance", "metis", "metis_chunk"}:
        return "speed_partition"
    return "round_robin" if name == "roundrobin" else name


def _load_tensor_config(value: Any, name: str) -> torch.Tensor | None:
    if value is None or torch.is_tensor(value):
        return value
    tensor = torch.load(Path(str(value)).expanduser(), map_location="cpu", weights_only=False)
    if not torch.is_tensor(tensor):
        raise ValueError(f"{name} must point to a tensor payload")
    return tensor



__all__ = ["Trainer", "build_model_from_config", "build_task_from_config", "maybe_build_model_from_config"]
