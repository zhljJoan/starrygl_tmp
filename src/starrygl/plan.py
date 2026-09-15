from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from starrygl.partition import PartitionPlan
from starrygl.spec import StarrySpec, derive_spec


FEATURE_KINDS = frozenset({"x", "edge_feat"})
STATE_KINDS = frozenset({"node_memory", "mailbox", "node_recurrent", "neighbor_recurrent", "model_recurrent"})
TASK_KINDS = frozenset({"endpoint_embedding", "label", "negative_target"})


@dataclass(frozen=True)
class AwaitDependency:
    name: str
    kind: str
    stage: str
    freshness_policy: str
    cache_policy: str
    wait_policy: str
    fulfillment: str
    owner_policy: str
    max_staleness: int = 0
    approximation: str = "none"

    def explain(self) -> str:
        return f"AwaitDependency({self.as_dict()!r})"

    def as_dict(self) -> dict[str, Any]:
        return dict(vars(self))


def lower_dependencies(
    *,
    spec: StarrySpec,
    task: Mapping[str, Any],
    cache_policy: str,
    wait_policy: str,
    owner_policy: str,
    state_kind: str,
) -> tuple[AwaitDependency, ...]:
    exact = dict(freshness_policy="exact", cache_policy="none", wait_policy=wait_policy)
    dependencies = [
        _dependency("x", "before_gcn", owner_policy="node_owner", fulfillment="collective_epoch", **exact)
    ]
    if is_edge_task(task):
        dependencies.append(
            _dependency("edge_feat", "before_message", owner_policy="edge_owner", fulfillment="collective_epoch", **exact)
        )

    freshness = spec.consistency
    state = dict(
        freshness_policy=freshness,
        cache_policy=cache_policy,
        wait_policy=wait_policy,
        fulfillment=_fulfillment(cache_policy=cache_policy, freshness_policy=freshness),
        owner_policy="node_owner",
        max_staleness=int(spec.max_staleness),
        approximation=spec.approximation,
    )
    if spec.state == "persistent":
        dependencies.extend(
            (_dependency("node_memory", "before_gcn", **state), _dependency("mailbox", "before_rnn", **state))
        )
    elif spec.state == "snapshot_recurrent":
        stage = "before_gcn" if state_kind in {"neighbor_recurrent", "model_recurrent"} else "before_rnn"
        dependencies.append(_dependency(state_kind, stage, **state))

    if is_edge_task(task):
        dependencies.append(
            _dependency(
                "endpoint_embedding",
                "before_task",
                owner_policy=owner_policy,
                fulfillment="collective_epoch",
                **exact,
            )
        )
        if _has_negative_sampling(task):
            dependencies.append(
                _dependency(
                    "negative_target",
                    "before_sample",
                    owner_policy=owner_policy,
                    fulfillment="collective_epoch",
                    **exact,
                )
            )
    dependencies.append(
        _dependency("label", "before_loss", owner_policy=owner_policy, fulfillment="collective_epoch", **exact)
    )
    return tuple(dependencies)


def is_edge_task(task: Mapping[str, Any]) -> bool:
    return str(task.get("name", "")).strip().lower() == "edge_prediction"


def _dependency(kind: str, stage: str, **policy: Any) -> AwaitDependency:
    return AwaitDependency(name=kind, kind=kind, stage=stage, **policy)


def _has_negative_sampling(task: Mapping[str, Any]) -> bool:
    services = task.get("services")
    if isinstance(services, Mapping) and "negative_sampling" in services:
        return services["negative_sampling"] is not None
    name = str(task.get("name", "")).strip().lower()
    return name == "edge_prediction" or any(
        key in task for key in ("negative_sampling", "negative_sampler", "negative_target")
    )


def _fulfillment(*, cache_policy: str, freshness_policy: str) -> str:
    if cache_policy == "shared_hot":
        return "shared_hot_cache"
    if cache_policy == "local":
        return "local_cache"
    return "collective_epoch" if freshness_policy == "exact" else "owner_fetch"


@dataclass(frozen=True)
class ViewPlan:
    """Physical graph layouts required by one semantic execution path."""

    kind: str
    required_layouts: tuple[str, ...]
    temporal_csr_bidirectional: bool = True
    temporal_csr_shared: bool = False

    def requires(self, layout: str) -> bool:
        return layout in self.required_layouts

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "required_layouts": list(self.required_layouts),
            "temporal_csr_bidirectional": self.temporal_csr_bidirectional,
            "temporal_csr_shared": self.temporal_csr_shared,
        }


@dataclass(frozen=True)
class ExecutionPlan:
    spec: StarrySpec
    partition_plan: PartitionPlan | None
    view: ViewPlan
    window_policy: str
    sampling_policy: str
    schedule_policy: str
    cache_policy: str
    wait_policy: str
    comm_mode: str
    owner_policy: str
    task: str
    coupling: str
    execution_order: tuple[str, ...]
    state_commit_policy: str
    await_dependencies: tuple[AwaitDependency, ...] = ()
    model_state_key: str | None = None
    model_aggregate_key: str | None = None
    requires_temporal_state: bool = False

    @property
    def storage_view(self) -> str:
        if self.view.kind == "snapshot" and self.sampling_policy == "neighbor":
            return "temporal_sampling_view"
        return {
            "temporal_sampling": "temporal_sampling_view",
            "snapshot": "snapshot_block_view",
            "event": "event_view",
        }[self.view.kind]

    @property
    def execution_spine(self) -> str:
        if self.spec.temporal == "snapshot":
            return "sampled_snapshot" if self.sampling_policy == "neighbor" else "snapshot_full_graph"
        return "temporal_sampling" if self.sampling_policy == "neighbor" else "event"

    @property
    def dependency_sources(self) -> tuple[str, ...]:
        return tuple(dep.kind for dep in self.await_dependencies)

    @property
    def feature_dependencies(self) -> tuple[AwaitDependency, ...]:
        return tuple(dep for dep in self.await_dependencies if dep.kind in FEATURE_KINDS)

    @property
    def state_dependencies(self) -> tuple[AwaitDependency, ...]:
        return tuple(dep for dep in self.await_dependencies if dep.kind in STATE_KINDS)

    @property
    def task_dependencies(self) -> tuple[AwaitDependency, ...]:
        return tuple(dep for dep in self.await_dependencies if dep.kind in TASK_KINDS)

    @property
    def temporal_representation(self) -> str:
        return "event_stream" if self.spec.temporal == "event" else "snapshot_sequence"

    @property
    def spatial_aggregation(self) -> str:
        return "sampled_neighbor" if self.spec.scope == "sampled" else "full_neighbor"

    def explain(self) -> str:
        fields = self.as_dict()
        fields["partition_plan"] = "bound" if self.partition_plan is not None else "unbound"
        if (self.spec.temporal == "snapshot" and self.coupling == "coupled"
                and self.spec.consistency == "bounded_stale" and not self.view.requires("temporal_csr")
                and any(dep.kind == "neighbor_recurrent" for dep in self.state_dependencies)):
            fields["snapshot_boundary"] = {
                "storage": "sliding_window_slots",
                "retention": "window_outputs_plus_predecessor",
                "compute": "owner_only",
                "increment": "cumulative_mean",
                "prediction": "remote_cache_plus_sigmoid_gamma_boundary_times_age_times_mean_increment",
                "ablation": "state_extrapolation_false=cache_only; boundary_prediction.learnable_false=fixed_one",
                "publish": "filtered_owner_boundary_all_to_all_once_per_batch; empty_payloads_participate",
                "freshness": "causal_age_at_most_K; consecutive_complete_latest_snapshots; previous_batch_push_awaited",
            }
        scope = "    execution_order_scope='semantic_lowering',\n"
        return "ExecutionPlan(\n" + scope + "".join(f"    {name}={value!r},\n" for name, value in fields.items()) + ")"

    def as_dict(self) -> dict[str, Any]:
        return {
            "spec": {
                "temporal": self.spec.temporal,
                "state": self.spec.state,
                "scope": self.spec.scope,
                "consistency": self.spec.consistency,
                "max_staleness": self.spec.max_staleness,
            },
            "partition_plan": None if self.partition_plan is None else self.partition_plan.as_dict(),
            "view": self.view.as_dict(),
            "storage_view": self.storage_view,
            "window_policy": self.window_policy,
            "sampling_policy": self.sampling_policy,
            "schedule_policy": self.schedule_policy,
            "cache_policy": self.cache_policy,
            "wait_policy": self.wait_policy,
            "comm_mode": self.comm_mode,
            "owner_policy": self.owner_policy,
            "temporal_representation": self.temporal_representation,
            "spatial_aggregation": self.spatial_aggregation,
            "dependency_sources": list(self.dependency_sources),
            "task": self.task,
            "execution_spine": self.execution_spine,
            "coupling": self.coupling,
            "execution_order": list(self.execution_order),
            "state_commit_policy": self.state_commit_policy,
            "model_state_key": self.model_state_key,
            "model_aggregate_key": self.model_aggregate_key,
            "requires_temporal_state": self.requires_temporal_state,
            "await_dependencies": [dep.as_dict() for dep in self.await_dependencies],
        }


class ChunkBindingPlanner:
    """Bind semantic declarations to the first observable execution plan."""

    def compile(
        self,
        *,
        data: Mapping[str, Any],
        backbone: Mapping[str, Any],
        task: Mapping[str, Any],
        runtime: Mapping[str, Any],
        partition_plan: PartitionPlan | None = None,
    ) -> ExecutionPlan:
        spec = derive_spec(data=data, backbone=backbone, runtime=runtime)
        model = backbone
        sampling_policy = _sampling_policy(spec=spec, runtime=runtime)
        window_policy = _window_policy(spec=spec, runtime=runtime, sampling_policy=sampling_policy)
        schedule_policy = (
            "spatial" if spec.state == "persistent"
            else "spatiotemporal_pipeline" if spec.temporal == "snapshot" and spec.scope == "full_graph"
            else "temporal_pipeline"
        )
        state_kind = _state_kind(spec, model)
        cache_policy = _cache_policy(
            spec=spec,
            state_kind=state_kind,
            runtime=runtime,
            partition_plan=partition_plan,
        )
        wait_policy = "block"
        owner_policy = "edge_owner" if is_edge_task(task) else "node_owner"
        coupling = _coupling(spec, model, state_kind=state_kind)
        requires_state = model.get("requires_temporal_state")
        state_key = model.get("state_key")
        if requires_state is None:
            requires_state = spec.state != "stateless" or state_key == "s"
        return ExecutionPlan(
            spec=spec,
            partition_plan=partition_plan,
            view=_view_plan(spec=spec, runtime=runtime, sampling_policy=sampling_policy,
                            diffusion=model.get("name") == "dcrnn"),
            window_policy=window_policy,
            sampling_policy=sampling_policy,
            schedule_policy=schedule_policy,
            cache_policy=cache_policy,
            wait_policy=wait_policy,
            comm_mode="collective_epoch",
            owner_policy=owner_policy,
            task=str(task.get("name", "")),
            coupling=coupling,
            execution_order=_execution_order(spec, coupling, sampling_policy),
            state_commit_policy=_state_commit_policy(spec),
            await_dependencies=lower_dependencies(
                spec=spec,
                task=task,
                cache_policy=cache_policy,
                wait_policy=wait_policy,
                owner_policy=owner_policy,
                state_kind=state_kind,
            ),
            model_state_key=state_key if state_key is not None else ("s" if requires_state else None),
            model_aggregate_key=model.get("aggregate_key", "h"),
            requires_temporal_state=bool(requires_state),
        )


def _view_plan(*, spec: StarrySpec, runtime: Mapping[str, Any], sampling_policy: str,
               diffusion: bool = False) -> ViewPlan:
    if spec.temporal == "snapshot":
        if sampling_policy == "neighbor":
            sampling = runtime.get("sampling", {})
            neighbor = sampling.get("neighbor", {}) if isinstance(sampling, Mapping) else {}
            return ViewPlan(
                kind="snapshot",
                required_layouts=("snapshot_csc", "temporal_csr"),
                temporal_csr_bidirectional=bool(neighbor.get("bidirectional", True)) if isinstance(neighbor, Mapping) else True,
                temporal_csr_shared=bool(neighbor.get("shared", False)) if isinstance(neighbor, Mapping) else False,
            )
        return ViewPlan(
            kind="snapshot",
            required_layouts=("snapshot_csc",) + (("snapshot_diffusion",) if diffusion else ()),
        )
    if sampling_policy == "neighbor":
        sampling = runtime.get("sampling", {})
        neighbor = sampling.get("neighbor", {}) if isinstance(sampling, Mapping) else {}
        return ViewPlan(
            kind="temporal_sampling",
            required_layouts=("event_view", "temporal_csr"),
            temporal_csr_bidirectional=bool(neighbor.get("bidirectional", True)) if isinstance(neighbor, Mapping) else True,
            temporal_csr_shared=bool(neighbor.get("shared", False)) if isinstance(neighbor, Mapping) else False,
        )
    return ViewPlan(kind="event", required_layouts=("event_view",))

def _coupling(
    spec: StarrySpec,
    model: Mapping[str, Any] | None = None,
    *,
    state_kind: str | None = None,
) -> str:
    model = {} if model is None else model
    explicit = str(model.get("coupling", "")).strip().lower()
    if explicit in {"coupled", "decoupled"}:
        return explicit
    if spec.state == "persistent":
        return "coupled"
    if spec.state == "snapshot_recurrent":
        if state_kind in {"neighbor_recurrent", "model_recurrent"}:
            return "coupled"
        reads_neighbors = model.get("reads_neighbor_state")
        if reads_neighbors is None:
            reads_neighbors = str(model.get("name", "")).strip().lower().replace("-", "_") in {"gconv_gru", "dcrnn"}
        return "coupled" if reads_neighbors else "decoupled"
    return "decoupled"


def _state_kind(spec: StarrySpec, model: Mapping[str, Any]) -> str:
    explicit = str(model.get("state_kind", "")).strip().lower()
    if explicit:
        if explicit not in STATE_KINDS:
            raise ValueError(f"unknown state_kind: {explicit!r}")
        return explicit
    if spec.state == "persistent":
        return "node_memory"
    name = str(model.get("name", "")).strip().lower().replace("-", "_")
    if name in {"evolvegcn", "evolve_gcn"}:
        return "model_recurrent"
    if (
        name in {"gconv_gru", "dcrnn"}
        or bool(model.get("reads_neighbor_state", False))
        or str(model.get("coupling", "")).strip().lower() == "coupled"
    ):
        return "neighbor_recurrent"
    return "node_recurrent"


def _cache_policy(
    *,
    spec: StarrySpec,
    state_kind: str,
    runtime: Mapping[str, Any],
    partition_plan: PartitionPlan | None,
) -> str:
    if spec.consistency == "exact":
        return "none"
    if spec.temporal == "snapshot" and spec.scope == "full_graph":
        return "local"
    if partition_plan is not None:
        has_shared = partition_plan.shared_nodes is not None and int(partition_plan.shared_nodes.numel()) > 0
        return "shared_hot" if state_kind in {"node_memory", "neighbor_recurrent"} and has_shared else "local"
    preprocess = runtime.get("preprocess", {})
    hot_ratio = float(preprocess.get("hot_node_ratio", 0.0)) if isinstance(preprocess, Mapping) else 0.0
    return "shared_hot" if state_kind in {"node_memory", "neighbor_recurrent"} and hot_ratio > 0 else "local"


def _execution_order(spec: StarrySpec, coupling: str, sampling_policy: str) -> tuple[str, ...]:
    steps = ["select_input_window", "build_task_target"]
    if sampling_policy == "neighbor":
        steps.extend(("sample_native_blocks", "materialize_temporal_sampling_view"))
    elif spec.temporal == "snapshot":
        steps.append("materialize_snapshot_block_view")
    else:
        steps.append("materialize_event_view")
    steps.append("attach_task_route")

    if spec.state != "stateless":
        steps.append("fetch_state_with_consistency")
    if spec.state == "snapshot_recurrent" and coupling == "decoupled" and sampling_policy == "full":
        steps.extend(("execute_gcn_windows", "scan_recurrent_state"))
    else:
        steps.append("execute_model")
    if spec.state == "snapshot_recurrent":
        steps.append("carry_window_state")
    steps.append("compute_task_loss")
    if spec.state == "persistent":
        steps.append("commit_state_to_master")
    return tuple(steps)


def _state_commit_policy(spec: StarrySpec) -> str:
    if spec.state == "persistent":
        if spec.consistency == "bounded_stale":
            return "master_commit_with_bounded_stale_reads"
        return "master_commit_after_model"
    if spec.state == "snapshot_recurrent":
        return "carry_recurrent_window_state"
    return "none"


def _window_policy(*, spec: StarrySpec, runtime: Mapping[str, Any], sampling_policy: str) -> str:
    if spec.temporal == "event":
        return "event_window"
    sampling = runtime.get("sampling", {})
    window = sampling.get("window", {}) if isinstance(sampling, Mapping) else {}
    configured = window.get("policy") if isinstance(window, Mapping) else None
    if sampling_policy == "neighbor":
        if configured not in {None, "full_snapshot"}:
            raise ValueError("Snapshot neighbor sampling requires window.policy='full_snapshot' or null")
        return "full_snapshot"
    if configured == "event_window":
        raise ValueError("Snapshot full execution requires window.policy='chunk_decay', 'full_snapshot', or null")
    return "chunk_decay" if configured is None else str(configured)


def _sampling_policy(*, spec: StarrySpec, runtime: Mapping[str, Any]) -> str:
    sampling = runtime.get("sampling", {})
    configured = sampling.get("mode") if isinstance(sampling, Mapping) else None
    if configured is not None:
        return str(configured)
    return "neighbor" if spec.temporal == "event" else "full"


__all__ = ["AwaitDependency", "ChunkBindingPlanner", "ExecutionPlan", "ViewPlan"]
