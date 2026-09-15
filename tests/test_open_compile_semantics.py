import json
from pathlib import Path
from types import SimpleNamespace

import starrygl as sg
import torch
from starrygl.config_example import CONFIG_EXAMPLE
from starrygl.spec import DEFAULT_RUNTIME_CONFIG
import starrygl.api as public_api
import starrygl.plan as public_plan
import starrygl.spec as public_spec
from starrygl.runtime.dataloader.loader import _input_window
from starrygl.store import FeatureManager, GraphStore, LabelStore, StoreBundle


def test_public_api_uses_top_level_implementations() -> None:
    assert sg.compile is public_api.compile
    assert sg.from_config is public_api.from_config
    assert sg.DataSource is public_spec.DataSource
    assert sg.ModelBackbone is public_spec.ModelBackbone
    assert sg.TaskSegment is public_spec.TaskSegment
    assert sg.ExecutionPlan is public_plan.ExecutionPlan


def test_trainer_runtime_responsibilities_stay_split() -> None:
    from starrygl.runtime.trainer import Trainer

    assert Trainer.__module__ == "starrygl.runtime.trainer"
    assert Trainer.prepare.__module__ == "starrygl.runtime.trainer"
    assert Trainer.fit.__module__ == "starrygl.runtime.trainer"
    assert Trainer._window_policy.__module__ == "starrygl.runtime.trainer_options"


def test_distributed_context_initializes_local_process(monkeypatch) -> None:
    import starrygl.utils.context as context_module

    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(context_module.dist, "is_initialized", lambda: False)

    context = context_module.DistributedContext.init(backend="gloo", device="cpu")
    assert (context.rank, context.world_size, context.local_rank, context.device.type) == (0, 1, 0, "cpu")
    assert context_module.DistributedContext.get_default_context() is context
    assert context.get_ranks_by_host() == (0,)
    assert context.get_hybrid_matrix().tolist() == [[0]]
    context.shutdown()


def test_python_main_runs_explicit_task_sequence(monkeypatch) -> None:
    from importlib import import_module

    app = import_module("starrygl.cli.main")

    calls = []

    class FakeTrainer:
        def prepare_artifacts(self, **kwargs):
            calls.append(("prepare", kwargs))

        def fit(self):
            calls.append(("fit", {}))

        def evaluate(self):
            calls.append(("evaluate", {}))

        def predict(self):
            calls.append(("predict", {}))

    monkeypatch.setattr(app, "from_config", lambda config, artifact_root=None: FakeTrainer())

    result = app.main(["config.json", "--predict"])

    assert result == 0
    assert [name for name, _ in calls] == ["prepare", "fit", "evaluate", "predict"]


def test_canonical_package_layout() -> None:
    import starrygl.batch as batch
    import starrygl.cli as cli
    import starrygl.model as model
    import starrygl.native as native
    import starrygl.runtime as runtime
    import starrygl.runtime.event as runtime_event
    import starrygl.runtime.memory as runtime_memory
    import starrygl.runtime.dataloader as runtime_dataloader
    import starrygl.runtime.sample as runtime_sample
    import starrygl.runtime.dataloader.pipeline as runtime_access
    import starrygl.runtime.snapshot as runtime_snapshot
    import starrygl.runtime.state as runtime_state
    import starrygl.runtime.comm as runtime_comm
    import starrygl.runtime.trainer as runtime_trainer
    import starrygl.store as store
    import starrygl.task as task
    import starrygl.utils as utils
    import starrygl.view as view
    import starrygl.batch as batch_source

    assert runtime.Trainer is runtime_trainer.Trainer
    assert runtime_access.__file__.replace("\\", "/").endswith("starrygl/runtime/dataloader/pipeline.py")
    assert runtime_dataloader.__file__.replace("\\", "/").endswith("starrygl/runtime/dataloader/__init__.py")
    assert runtime_event.__file__.replace("\\", "/").endswith("starrygl/runtime/event/__init__.py")
    assert runtime_snapshot.__file__.replace("\\", "/").endswith("starrygl/runtime/snapshot/__init__.py")
    assert runtime_sample.__file__.replace("\\", "/").endswith("starrygl/runtime/sample/__init__.py")
    assert runtime_memory.__file__.replace("\\", "/").endswith("starrygl/runtime/memory/__init__.py")
    assert runtime_state.__file__.replace("\\", "/").endswith("starrygl/runtime/state/__init__.py")
    assert runtime_comm.__file__.replace("\\", "/").endswith("starrygl/runtime/comm.py")
    assert model.StarryModel is sg.StarryModel
    assert store.GraphStore is sg.GraphStore
    assert task.StarryTask is sg.StarryTask
    assert view.GraphBlock is sg.GraphBlock
    assert native.NativeTemporalSampler is sg.NativeTemporalSampler
    assert utils.DistributedContext.__module__ == "starrygl.utils.context"
    assert utils.RouteBook is sg.RouteBook
    assert callable(cli.main)
    assert batch.Batch is batch_source.Batch


def test_from_config_expands_environment_variables(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("STARRYGL_DATA_ROOT", "/public/data")
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "data": {"source": "${STARRYGL_DATA_ROOT}/TGL-DATA/WIKI"},
                "backbone": {"name": "tgn"},
                "task": {"name": "edge_prediction"},
            }
        )
    )

    trainer = sg.from_config(path)

    assert trainer.graph["source"] == "/public/data/TGL-DATA/WIKI"


def test_from_config_translates_canonical_data_backbone_task_sections() -> None:
    trainer = sg.from_config(
        {
            "data": {
                "source": "snapshots",
                "temporal_representation": "snapshot_sequence",
            },
            "backbone": {
                "name": "tgcn",
                "spatial_aggregation": "full_neighbor",
            },
            "task": {"name": "node_regression"},
        }
    )

    assert trainer.graph["source"] == "snapshots"
    assert trainer.model_config["name"] == "tgcn"
    assert trainer.task["name"] == "node_regression"
    assert trainer.plan.execution_spine == "snapshot_full_graph"


def test_bundled_configs_compile_through_the_canonical_entry() -> None:
    config_root = Path(__file__).parents[1] / "configs"

    plans = {path.name: sg.from_config(path).plan.execution_spine for path in config_root.glob("*.json")}

    assert plans == {
        "ctdg_tgat_node_prediction.json": "temporal_sampling",
        "ctdg_tgn_edge_prediction.json": "temporal_sampling",
        "dtdg_rec_amazon_tgcn_node_regression.json": "snapshot_full_graph",
        "dtdg_soc_bitcoin_gconv_gru_edge_prediction.json": "snapshot_full_graph",
    }


def test_compile_defaults_match_the_documented_runtime_example() -> None:
    trainer = sg.compile(
        data_source={"source": "wiki"},
        backbone={"name": "tgn"},
        task_segment=sg.EdgePrediction(),
    )

    assert CONFIG_EXAMPLE["runtime"] == DEFAULT_RUNTIME_CONFIG
    assert trainer.spec.consistency == "bounded_stale"
    assert trainer.spec.max_staleness == 1
    assert trainer.runtime_config["access_pipeline"] is True
    assert trainer.runtime_config["train_compute_metrics"] is False
    assert trainer.runtime_config["temporal_state"] == DEFAULT_RUNTIME_CONFIG["temporal_state"]
    assert trainer.runtime_config["sampling"] == DEFAULT_RUNTIME_CONFIG["sampling"]
    assert trainer.train_config == DEFAULT_RUNTIME_CONFIG["train"]
    assert trainer.preprocess_config == DEFAULT_RUNTIME_CONFIG["preprocess"]


def test_temporal_state_defaults_lower_once_at_model_construction() -> None:
    trainer = sg.compile(
        data_source={"source": "wiki"},
        backbone={"name": "tgn", "in_dim": 2, "hidden_dim": 4, "out_dim": 4},
        task_segment=sg.EdgePrediction(),
    )

    assert trainer.model.memory.memory_update == "gru"
    assert trainer.model.memory.increment is not None
    assert isinstance(trainer.model.memory.gamma, torch.nn.Parameter)
    assert float(trainer.model.memory.gamma.detach()) == 0.5
    trainer.model.memory.increment.update(torch.tensor([2]), torch.ones(1, 4))
    assert trainer.model.memory.increment.count.shape[0] == 3
    assert "memory_filter" not in trainer.to_config()["backbone"]
    assert "historical_mix" not in trainer.to_config()["backbone"]


def test_node_model_waits_for_store_to_infer_class_count() -> None:
    trainer = sg.compile(
        data_source={"source": "labels"},
        backbone={"name": "tgn", "in_dim": 2, "hidden_dim": 4, "out_dim": 4},
        task_segment=sg.NodeClassification(),
    )
    store = StoreBundle(
        graph=GraphStore(num_nodes=3),
        features=FeatureManager(node_features={"x": torch.ones(3, 2)}),
        labels=LabelStore(node_label=torch.tensor([0, 2, 1])),
    )

    assert trainer.model is None
    model = trainer._model(None, store)
    assert model.node_head is not None
    assert model.node_head.out_features == 3


def test_backbone_semantic_fields_are_not_model_constructor_kwargs() -> None:
    trainer = sg.compile(
        data_source={"source": "wiki"},
        backbone={
            "name": "tgn",
            "in_dim": 2,
            "hidden_dim": 4,
            "out_dim": 4,
            "coupling": "coupled",
            "state_key": "s",
            "aggregate_key": "h",
        },
        task_segment=sg.EdgePrediction(),
    )

    assert isinstance(trainer.model, sg.TGNModel)


def test_loading_prepared_store_binds_partition_to_execution_plan() -> None:
    trainer = sg.compile(
        data_source={"source": "wiki"},
        backbone={"name": "tgn"},
        task_segment=sg.EdgePrediction(),
    )
    owner = torch.tensor([0, 1, 0], dtype=torch.long)
    location = torch.tensor([0, 0, 1], dtype=torch.long)
    store = StoreBundle(
        graph=GraphStore(
            num_nodes=3,
            prepare={
                "partition": {
                    "node_dist_index": (owner << 48) | location,
                    "edge_dist_index": torch.tensor([0, 1], dtype=torch.long) << 48,
                    "hot_node_ids": torch.tensor([2]),
                    "node_to_chunk": torch.tensor([0, 0, 1]),
                    "edge_chunk": torch.tensor([0, 1]),
                }
            },
        ),
        features=FeatureManager(node_features={"x": torch.zeros(3, 1)}),
        labels=LabelStore(),
    )

    trainer._store(store, artifact_root=None, rank=0, map_location="cpu", mmap=False)

    bound = trainer.plan.partition_plan
    assert bound is not None
    assert torch.equal(bound.node_master, owner)
    assert torch.equal(bound.edge_master, torch.tensor([0, 1]))
    assert torch.equal(bound.shared_nodes, torch.tensor([2]))
    assert trainer.plan.cache_policy == "shared_hot"
    assert "partition_plan='bound'" in trainer.plan.explain()


def test_from_config_keeps_sampling_window_and_neighbor_structure() -> None:
    trainer = sg.from_config(
        {
            "data": {"source": "wiki", "temporal_representation": "event_stream"},
            "backbone": {"name": "tgn"},
            "task": {"name": "edge_prediction"},
            "runtime": {
                "sampling": {
                    "mode": "neighbor",
                    "window": {
                        "policy": "chunk_decay",
                        "snaps_count": 3,
                        "chunk_decay": [1, 2],
                        "num_full_snapshots": 1,
                        "chunk_order": "rand",
                    },
                    "neighbor": {"fanouts": [15, 10], "policy": "recent", "seed": 7, "workers": 0},
                },
            },
        }
    )

    sampling = trainer.runtime_config["sampling"]

    assert sampling["mode"] == "neighbor"
    assert sampling["window"]["policy"] == "chunk_decay"
    assert sampling["window"]["snaps_count"] == 3
    assert sampling["window"]["chunk_decay"] == [1, 2]
    assert sampling["window"]["num_full_snapshots"] == 1
    assert sampling["window"]["chunk_order"] == "rand"
    assert sampling["neighbor"]["fanouts"] == [15, 10]
    assert sampling["neighbor"]["policy"] == "recent"
    assert sampling["neighbor"]["seed"] == 7
    assert sampling["neighbor"]["workers"] == 0
    assert "semantics" not in sampling
    assert trainer._window_policy(None) == "event_window"
    assert trainer._sampling_policy(None) == "neighbor"
    assert trainer._chunk_decay(None) == [1, 2]
    assert trainer._snaps_count() == 3
    assert trainer._num_full_snapshots(None) == 1
    assert trainer._fanouts(None) == (15, 10)
    sampler_options = trainer._sampler_options(None)
    assert sampler_options["policy"] == "boundary_decay_sampling"
    assert sampler_options["probability"] == 0.1
    assert sampler_options["seed"] == 7
    assert sampler_options["workers"] == 0
    assert sampler_options["chunk_order"] == "rand"


def test_runtime_sampling_window_chunk_decay_string_uses_existing_parser() -> None:
    trainer = sg.from_config(
        {
            "data": {"source": "snapshots", "temporal_representation": "snapshot_sequence"},
            "backbone": {"name": "tgcn"},
            "task": {"name": "node_regression"},
            "runtime": {
                "preprocess": {"chunks_per_rank": 8},
                "sampling": {
                    "window": {
                        "policy": "chunk_decay",
                        "snaps_count": 3,
                        "chunk_decay": "half",
                        "num_full_snapshots": 1,
                    },
                },
            },
        }
    )

    assert trainer._window_policy(None) == "chunk_decay"
    assert trainer._sampling_policy(None) == "full"
    assert trainer._chunk_decay(None) == (4, 2)


def test_input_window_derives_chunk_decay_without_batch_plan() -> None:
    snapshot_ids, chunk_limits = _input_window(
        window_id=2,
        split_start=0,
        window_policy="chunk_decay",
        chunk_decay=(1,),
        num_full_snapshots=1,
    )

    assert list(snapshot_ids) == [1, 2]
    assert chunk_limits == (1, -1)


def test_sampled_snapshot_plan_requires_snapshot_and_temporal_layouts() -> None:
    trainer = sg.from_config(
        {
            "data": {
                "source": "snapshots",
                "temporal_representation": "snapshot_sequence",
            },
            "backbone": {
                "name": "tgcn",
                "spatial_aggregation": "sampled_neighbor",
            },
            "task": {"name": "node_regression"},
            "runtime": {"sampling": {"mode": "neighbor"}},
        }
    )

    assert trainer.plan.view.required_layouts == ("snapshot_csc", "temporal_csr")


def test_compile_accepts_data_source_backbone_task_segment_semantics() -> None:
    trainer = sg.compile(
        data_source=sg.DataSource(
            source="wiki",
            temporal_representation="event_stream",
        ),
        backbone=sg.ModelBackbone(
            name="tgn",
            temporal_representation="event_stream",
            spatial_aggregation="sampled_neighbor",
            coupling="coupled",
        ),
        task_segment=sg.TaskSegment(
            name="edge_prediction",
            negative_sampler=sg.NegativeSampler(ratio=1),
        ),
    )

    plan = trainer.plan

    assert trainer.graph["source"] == "wiki"
    assert trainer.model_config["name"] == "tgn"
    assert trainer.task["name"] == "edge_prediction"
    assert plan.storage_view == "temporal_sampling_view"
    assert plan.temporal_representation == "event_stream"
    assert plan.spatial_aggregation == "sampled_neighbor"
    assert plan.coupling == "coupled"
    assert plan.dependency_sources == (
        "x",
        "edge_feat",
        "node_memory",
        "mailbox",
        "endpoint_embedding",
        "negative_target",
        "label",
    )
    assert plan.as_dict()["storage_view"] == "temporal_sampling_view"


def test_compile_preserves_temporal_state_semantics_metadata() -> None:
    trainer = sg.compile(
        data_source={"source": "wiki"},
        backbone={"name": "tgn"},
        task_segment=sg.EdgePrediction(),
        runtime={
            "temporal_state": {
                "update": "gru",
                "consistency": "bounded_stale",
                "max_staleness": 2,
                "filter": {"enabled": True, "min_change_norm": 0.0, "max_skip": 10},
                "smooth_aggregation": {"enabled": True, "gamma_init": 0.5},
            }
        },
    )

    temporal_state = trainer.runtime_config["temporal_state"]

    assert temporal_state["update"] == "gru"
    assert temporal_state["consistency"] == "bounded_stale"
    assert temporal_state["max_staleness"] == 2
    assert temporal_state["filter"]["enabled"] is True
    assert temporal_state["smooth_aggregation"]["gamma_init"] == 0.5
    assert "approximation" not in temporal_state
    assert trainer.plan.spec.consistency == "bounded_stale"
    assert {dep.freshness_policy for dep in trainer.plan.feature_dependencies} == {"exact"}
    assert {dep.freshness_policy for dep in trainer.plan.task_dependencies} == {"exact"}


def test_canonical_config_lowers_into_runtime_options() -> None:
    trainer = sg.from_config(
        {
            "data": {"source": "events"},
            "backbone": {"name": "tgn", "in_dim": 2, "hidden_dim": 4, "out_dim": 4},
            "task": {
                "name": "edge_prediction",
                "services": {
                    "negative_sampling": {
                        "ratio": 2,
                        "mode": "src_dst",
                        "train": {"policy": "random", "local_probability": 0.7, "remote_probability": 0.3},
                        "eval": {"policy": "random"},
                        "test": {"policy": "random"},
                        "sample_kwargs": {"deduplicate_roots": True},
                    }
                },
            },
            "runtime": {
                "device": "cpu",
                "train": {"batch_size": 17, "max_batches_per_epoch": 3, "dropout": 0.25, "att_dropout": 0.15},
                "sampling": {"neighbor": {"policy": "uniform"}},
            },
        }
    )

    train = trainer._sampler_options(None, train=True, split="train")
    evaluate = trainer._sampler_options(None, train=False, split="val")

    assert trainer._device(None) == torch.device("cpu")
    assert trainer._num_negatives(None) == 2
    prepare = trainer._prepare_config(world_size=1)
    assert prepare.time_split == "batch"
    assert prepare.target_batch_size == 17
    assert "edge_batch_size" not in train
    assert train["max_batches_per_epoch"] == 3
    assert train["policy"] == "boundary_uniform"
    assert train["probability"] == 0.1
    assert train["negative_mode"] == "src_dst"
    assert train["negative_local_prob"] == 0.7
    assert train["negative_global_prob"] == 0.3
    assert train["deduplicate_roots"] is True
    assert evaluate["negative_local_prob"] == 0.0
    assert evaluate["negative_global_prob"] == 1.0
    assert trainer.model.layers[0].dropout.p == 0.25
    assert trainer.model.layers[0].att_dropout.p == 0.15


def test_negative_sampling_can_be_disabled_explicitly() -> None:
    trainer = sg.compile(
        data_source={"source": "events"},
        backbone={"name": "tgn"},
        task_segment={"name": "edge_prediction", "services": {"negative_sampling": None}},
    )

    assert "negative_target" not in trainer.plan.dependency_sources
    assert trainer._num_negatives(None) == 0


def test_compile_rejects_mixed_old_and_new_semantic_arguments() -> None:
    try:
        sg.compile(graph={"source": "wiki"})
    except TypeError as exc:
        assert "graph" in str(exc)
    else:
        raise AssertionError("legacy graph argument was accepted")


def test_compile_rejects_spec_argument() -> None:
    try:
        sg.compile(data_source={"source": "wiki"}, backbone={"name": "tgn"}, spec={"consistency": "exact"})
    except TypeError as exc:
        assert "spec" in str(exc)
    else:
        raise AssertionError("legacy spec argument was accepted")


def test_compile_rejects_temporal_state_approximation() -> None:
    try:
        sg.compile(
            data_source={"source": "wiki"},
            backbone={"name": "tgn"},
            task_segment=sg.EdgePrediction(),
            runtime={"temporal_state": {"consistency": "bounded_stale", "approximation": "time_decay"}},
        )
    except ValueError as exc:
        assert "filter and smooth_aggregation" in str(exc)
    else:
        raise AssertionError("temporal_state approximation was accepted")


def test_from_config_rejects_legacy_section_names() -> None:
    try:
        sg.from_config(
            {
                "graph": {"source": "legacy"},
                "backbone": {"name": "tgn"},
                "task": {"name": "edge_prediction"},
            }
        )
    except ValueError as exc:
        assert "legacy section name" in str(exc)
        assert "graph" in str(exc)
    else:
        raise AssertionError("legacy graph section was accepted")


def test_from_config_rejects_execution_section() -> None:
    try:
        sg.from_config(
            {
                "data": {"source": "wiki"},
                "backbone": {"name": "tgn"},
                "task": {"name": "edge_prediction"},
                "execution": {"exact": True},
            }
        )
    except ValueError as exc:
        assert "execution" in str(exc)
        assert "runtime" in str(exc)
    else:
        raise AssertionError("execution section was accepted")


def test_from_config_rejects_top_level_runtime_subsections() -> None:
    for name in ("temporal_state", "sampling", "train", "preprocess"):
        try:
            sg.from_config(
                {
                    "data": {"source": "wiki"},
                    "backbone": {"name": "tgn"},
                    "task": {"name": "edge_prediction"},
                    name: {},
                }
            )
        except ValueError as exc:
            assert name in str(exc)
            assert "runtime" in str(exc)
        else:
            raise AssertionError(f"top-level {name} section was accepted")


def test_from_config_rejects_spec_section() -> None:
    try:
        sg.from_config(
            {
                "data": {"source": "wiki"},
                "backbone": {"name": "tgn"},
                "task": {"name": "edge_prediction"},
                "spec": {"consistency": "exact"},
            }
        )
    except ValueError as exc:
        assert "spec" in str(exc)
        assert "runtime" in str(exc)
    else:
        raise AssertionError("spec section was accepted")
