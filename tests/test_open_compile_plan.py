from itertools import product

import pytest
import starrygl as sg
from starrygl.plan import ChunkBindingPlanner
from starrygl.spec import normalize_backbone, normalize_data_source, normalize_runtime, normalize_task


def _internal_plan(*, data, backbone, task, runtime=None):
    return ChunkBindingPlanner().compile(
        data=normalize_data_source(data),
        backbone=normalize_backbone(backbone),
        task=normalize_task(task),
        runtime=normalize_runtime(runtime),
    )


def test_event_sampled_compile_exposes_observable_plan() -> None:
    trainer = sg.compile(
        data_source={"source": "wiki"},
        backbone={"name": "tgn"},
        task_segment=sg.EdgePrediction(),
    )

    plan = trainer.plan
    explain = plan.explain()

    assert isinstance(plan, sg.ExecutionPlan)
    assert plan.execution_spine == "temporal_sampling"
    assert plan.window_policy == "event_window"
    assert plan.sampling_policy == "neighbor"
    assert plan.storage_view == "temporal_sampling_view"
    assert plan.temporal_representation == "event_stream"
    assert plan.spatial_aggregation == "sampled_neighbor"
    assert plan.owner_policy == "edge_owner"
    assert plan.execution_order[:3] == (
        "select_input_window",
        "build_task_target",
        "sample_native_blocks",
    )
    assert "attach_task_route" in plan.execution_order
    assert plan.execution_order.index("compute_task_loss") < plan.execution_order.index("commit_state_to_master")
    assert plan.dependency_sources == ("x", "edge_feat", "node_memory", "mailbox", "endpoint_embedding", "negative_target", "label")
    assert "execution_order_scope='semantic_lowering'" in explain
    assert "await_dependencies=" in explain


def test_plan_explain_keeps_internal_runtime_strategy_names_private() -> None:
    plan = _internal_plan(
        data={"source": "wiki"},
        backbone={"name": "tgn"},
        task=sg.EdgePrediction(),
        runtime={"temporal_state": {"consistency": "bounded_stale", "max_staleness": 1}},
    )

    explain = plan.explain()

    assert plan.spec.consistency == "bounded_stale"
    assert plan.spec.approximation == "none"
    assert plan.dependency_sources == ("x", "edge_feat", "node_memory", "mailbox", "endpoint_embedding", "negative_target", "label")
    assert "stale_increment" not in explain
    assert "timestamp_increment" not in explain
    assert "layerwise" not in explain
    assert "memshare" not in explain


def test_exact_and_approximate_state_dependencies_lower_to_public_consistency() -> None:
    exact = sg.compile(
        data_source={"source": "wiki"},
        backbone={"name": "tgn"},
        task_segment=sg.EdgePrediction(),
        runtime={"temporal_state": {"consistency": "exact"}},
    )
    approximate = sg.compile(
        data_source={"source": "wiki"},
        backbone={"name": "tgn"},
        task_segment=sg.EdgePrediction(),
        runtime={"temporal_state": {"consistency": "bounded_stale", "max_staleness": 2}},
    )

    exact_deps = {dep.kind: dep for dep in exact.plan.await_dependencies}
    approximate_deps = {dep.kind: dep for dep in approximate.plan.await_dependencies}

    assert exact.plan.spec.consistency == "exact"
    assert exact.plan.spec.max_staleness == 0
    assert exact_deps["node_memory"].freshness_policy == "exact"
    assert exact_deps["mailbox"].freshness_policy == "exact"
    assert exact_deps["x"].freshness_policy == "exact"
    assert exact_deps["edge_feat"].freshness_policy == "exact"
    assert exact_deps["endpoint_embedding"].freshness_policy == "exact"
    assert exact_deps["label"].freshness_policy == "exact"
    assert exact_deps["node_memory"].stage == "before_gcn"
    assert exact_deps["mailbox"].stage == "before_rnn"

    assert approximate.plan.spec.consistency == "bounded_stale"
    assert approximate.plan.spec.max_staleness == 2
    assert approximate.plan.spec.approximation == "none"
    assert approximate_deps["node_memory"].freshness_policy == "bounded_stale"
    assert approximate_deps["mailbox"].freshness_policy == "bounded_stale"
    assert approximate_deps["x"].freshness_policy == "exact"
    assert approximate_deps["edge_feat"].freshness_policy == "exact"
    assert approximate_deps["endpoint_embedding"].freshness_policy == "exact"
    assert approximate_deps["label"].freshness_policy == "exact"
    assert approximate_deps["node_memory"].max_staleness == 2
    assert approximate_deps["mailbox"].max_staleness == 2
    assert approximate_deps["node_memory"].approximation == "none"
    assert approximate_deps["x"].max_staleness == 0
    assert approximate_deps["endpoint_embedding"].max_staleness == 0
    assert approximate_deps["label"].max_staleness == 0


def test_dcrnn_stale_candidate_input_reuses_neighbor_dependency_contract() -> None:
    stale = sg.compile(
        data_source={"source": "snapshots", "temporal_representation": "snapshot_sequence"},
        backbone={"name": "dcrnn"},
        task_segment={"name": "node_regression"},
        runtime={"temporal_state": {"consistency": "bounded_stale", "max_staleness": 1}},
    )
    dependencies = {dependency.name: dependency for dependency in stale.plan.state_dependencies}
    candidate = dependencies["neighbor_recurrent.candidate_input"]
    assert candidate.kind == "neighbor_recurrent"
    assert candidate.stage == "before_candidate"
    assert candidate.cache_policy == stale.plan.cache_policy


def test_backbone_semantics_require_s_without_exposing_read_stage() -> None:
    temporal = sg.compile(
        data_source={"source": "custom"},
        backbone=sg.ModelBackbone(
            name="custom_temporal",
            state_key="s",
            aggregate_key="h",
            requires_temporal_state=True,
        ),
        task_segment=sg.EdgePrediction(),
    )
    stateless = sg.compile(
        data_source={"source": "custom"},
        backbone=sg.ModelBackbone(
            name="custom_stateless",
            aggregate_key="h",
            requires_temporal_state=False,
        ),
        task_segment=sg.EdgePrediction(),
    )

    temporal_explain = temporal.plan.explain()

    assert temporal.spec.state == "persistent"
    assert temporal.plan.model_state_key == "s"
    assert temporal.plan.model_aggregate_key == "h"
    assert temporal.plan.requires_temporal_state is True
    assert {dep.kind for dep in temporal.plan.state_dependencies} == {"node_memory", "mailbox"}
    assert "model_state_key='s'" in temporal_explain
    assert "model_aggregate_key='h'" in temporal_explain
    assert "requires_temporal_state=True" in temporal_explain
    assert "state_read_stage" not in temporal_explain

    assert stateless.spec.state == "stateless"
    assert stateless.plan.model_state_key is None
    assert stateless.plan.model_aggregate_key == "h"
    assert stateless.plan.requires_temporal_state is False
    assert stateless.plan.state_dependencies == ()


def test_model_name_defaults_are_backbone_semantics_not_runtime_strategy() -> None:
    tgat = sg.compile(data_source={"source": "wiki"}, backbone={"name": "tgat"}, task_segment=sg.EdgePrediction())
    tgn = sg.compile(data_source={"source": "wiki"}, backbone={"name": "tgn"}, task_segment=sg.EdgePrediction())
    apan = sg.compile(data_source={"source": "wiki"}, backbone={"name": "apan"}, task_segment=sg.EdgePrediction())
    gconv_gru = sg.compile(data_source={"source": "snapshots"}, backbone={"name": "gconv_gru"}, task_segment=sg.EdgePrediction())
    tgcn = sg.compile(data_source={"source": "snapshots"}, backbone={"name": "tgcn"}, task_segment=sg.NodeRegression())
    mpnn = sg.compile(data_source={"source": "snapshots"}, backbone={"name": "mpnn_lstm"}, task_segment=sg.NodeRegression())
    evolve = sg.compile(data_source={"source": "snapshots"}, backbone={"name": "evolvegcn"}, task_segment=sg.NodeRegression())

    assert tgat.spec.state == "stateless"
    assert tgat.plan.requires_temporal_state is False
    assert tgat.plan.state_dependencies == ()

    assert tgn.spec.state == "persistent"
    assert {dep.kind for dep in tgn.plan.state_dependencies} == {"node_memory", "mailbox"}
    assert apan.spec.state == "persistent"
    assert {dep.kind for dep in apan.plan.state_dependencies} == {"node_memory", "mailbox"}

    assert gconv_gru.spec.temporal == "snapshot"
    assert gconv_gru.plan.coupling == "coupled"
    assert [dep.kind for dep in gconv_gru.plan.state_dependencies] == ["neighbor_recurrent"]

    assert tgcn.spec.temporal == "snapshot"
    assert tgcn.plan.coupling == "decoupled"
    assert [dep.kind for dep in tgcn.plan.state_dependencies] == ["node_recurrent"]
    assert mpnn.spec.temporal == "snapshot"
    assert mpnn.plan.coupling == "decoupled"
    assert [dep.kind for dep in mpnn.plan.state_dependencies] == ["node_recurrent"]
    assert evolve.plan.coupling == "coupled"
    assert [dep.kind for dep in evolve.plan.state_dependencies] == ["model_recurrent"]
    assert evolve.plan.state_dependencies[0].stage == "before_gcn"


def test_snapshot_full_graph_compile_exposes_batch_graph_and_recurrent_await() -> None:
    trainer = sg.compile(
        data_source={"source": "snapshots"},
        backbone={"name": "tgcn", "spatial_aggregation": "full_neighbor"},
        task_segment=sg.NodeRegression(),
        runtime={"temporal_state": {"consistency": "bounded_stale", "max_staleness": 2}},
    )

    plan = trainer.plan

    assert plan.execution_spine == "snapshot_full_graph"
    assert plan.window_policy == "chunk_decay"
    assert plan.sampling_policy == "full"
    assert plan.storage_view == "snapshot_block_view"
    assert plan.temporal_representation == "snapshot_sequence"
    assert plan.spatial_aggregation == "full_neighbor"
    assert plan.owner_policy == "node_owner"
    assert plan.cache_policy == "local"
    assert plan.spec.consistency == "bounded_stale"
    assert plan.wait_policy == "block"
    assert plan.spec.max_staleness == 2
    assert plan.execution_order[:2] == ("select_input_window", "build_task_target")
    assert "materialize_snapshot_block_view" in plan.execution_order
    assert len(plan.await_dependencies) == 3
    assert plan.dependency_sources == ("x", "node_recurrent", "label")
    dep = {dep.kind: dep for dep in plan.await_dependencies}["node_recurrent"]
    assert dep.kind == "node_recurrent"
    assert dep.stage == "before_rnn"
    assert dep.owner_policy == "node_owner"
    assert dep.cache_policy == "local"
    assert dep.freshness_policy == "bounded_stale"
    assert dep.wait_policy == "block"
    assert dep.fulfillment == "local_cache"


def test_sampled_snapshot_compile_uses_blocks_not_full_graph_view() -> None:
    trainer = sg.compile(
        data_source={"source": "snapshots"},
        backbone={"name": "gconv_gru", "spatial_aggregation": "sampled_neighbor"},
        task_segment=sg.EdgePrediction(),
        runtime={"sampling": {"mode": "neighbor"}},
    )

    plan = trainer.plan

    assert plan.execution_spine == "sampled_snapshot"
    assert plan.window_policy == "full_snapshot"
    assert plan.sampling_policy == "neighbor"
    assert plan.storage_view == "temporal_sampling_view"
    assert plan.temporal_representation == "snapshot_sequence"
    assert plan.spatial_aggregation == "sampled_neighbor"
    assert plan.coupling == "coupled"
    assert plan.execution_order[:3] == (
        "select_input_window",
        "build_task_target",
        "sample_native_blocks",
    )
    assert "sample_native_blocks" in plan.execution_order
    assert "materialize_temporal_sampling_view" in plan.execution_order
    assert "materialize_snapshot_block_view" not in plan.execution_order
    assert plan.dependency_sources == ("x", "edge_feat", "neighbor_recurrent", "endpoint_embedding", "negative_target", "label")
    assert {dep.fulfillment for dep in plan.await_dependencies} == {"collective_epoch", "shared_hot_cache"}


def test_canonical_runtime_consistency_lowers_state_await() -> None:
    plan = _internal_plan(
        data={"source": "snapshots"},
        backbone={"name": "gconv_gru", "spatial_aggregation": "full_neighbor"},
        task=sg.NodeRegression(),
        runtime={"temporal_state": {"consistency": "bounded_stale", "max_staleness": 1}},
    )

    assert plan.cache_policy == "shared_hot"
    assert plan.spec.consistency == "bounded_stale"
    assert plan.wait_policy == "block"
    assert plan.spec.max_staleness == 1
    assert plan.spec.approximation == "none"
    state_dep = plan.state_dependencies[0]
    assert state_dep.cache_policy == "shared_hot"
    assert state_dep.freshness_policy == "bounded_stale"
    assert state_dep.wait_policy == "block"
    assert state_dep.fulfillment == "shared_hot_cache"
    assert {dep.freshness_policy for dep in plan.feature_dependencies} == {"exact"}
    assert {dep.freshness_policy for dep in plan.task_dependencies} == {"exact"}


def test_plan_covers_paper_method_design_dimensions() -> None:
    expected_view = {
        ("event_stream", "sampled_neighbor"): "temporal_sampling_view",
        ("event_stream", "full_neighbor"): "event_view",
        ("snapshot_sequence", "sampled_neighbor"): "temporal_sampling_view",
        ("snapshot_sequence", "full_neighbor"): "snapshot_block_view",
    }
    for temporal, spatial, coupling in product(
        ("event_stream", "snapshot_sequence"),
        ("sampled_neighbor", "full_neighbor"),
        ("decoupled", "coupled"),
    ):
        plan = _internal_plan(
            data={"source": "graph", "temporal_representation": temporal},
            backbone={
                "name": "custom",
                "spatial_aggregation": spatial,
                "coupling": coupling,
                "requires_temporal_state": True,
                "state_key": "s",
            },
            task=sg.NodeRegression(),
            runtime={"sampling": {"mode": "neighbor" if spatial == "sampled_neighbor" else "full"}},
        )

        assert plan.temporal_representation == temporal
        assert plan.spatial_aggregation == spatial
        assert plan.coupling == coupling
        assert plan.storage_view == expected_view[(temporal, spatial)]
        assert "fetch_state_with_consistency" in plan.execution_order
        if temporal == "event_stream":
            assert plan.state_commit_policy != "none"
            assert "commit_state_to_master" in plan.execution_order
        else:
            expected_state = "neighbor_recurrent" if coupling == "coupled" else "node_recurrent"
            assert plan.state_dependencies[0].kind == expected_state
            assert "carry_window_state" in plan.execution_order


def test_snapshot_neighbor_rejects_chunk_decay_window() -> None:
    with pytest.raises(ValueError, match="requires window.policy='full_snapshot' or null"):
        sg.compile(
            data_source={"source": "snapshots"},
            backbone={"name": "gconv_gru"},
            task_segment=sg.NodeRegression(),
            runtime={
                "sampling": {
                    "mode": "neighbor",
                    "window": {"policy": "chunk_decay"},
                }
            },
        )
