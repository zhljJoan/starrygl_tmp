from __future__ import annotations

from pathlib import Path
from typing import Any

from starrygl.api import compile, from_config
from starrygl.spec import DataSource, ModelBackbone, NegativeSampler, TaskSegment


# Minimal config surface accepted by from_config().
# Required top-level sections: data, backbone, task.
# Optional top-level sections: runtime, artifact_root.
CONFIG_EXAMPLE: dict[str, Any] = {
    # Required. Declares where graph/event/snapshot input data comes from.
    "data": {
        # Required for file-backed datasets. Path or dataset URI understood by
        # the loader/preprocess path. Candidates: local path string, dataset
        # URI string, or other loader-supported source string.
        "source": "/path/to/dataset",
        # Optional. event_stream means temporal events; snapshot_sequence means
        # discrete graph snapshots. If omitted, compile() may infer it from the
        # backbone name. Candidates: event_stream, snapshot_sequence.
        "temporal_representation": "event_stream",
        # Optional. Human-readable dataset name used by callers that need one.
        # Candidates: any short string such as wiki, gdelt, reddit.
        "name": "wiki",
    },
    # Required. Declares the model backbone semantics, not a runtime pipeline.
    "backbone": {
        # Required. Built-in model name or model selector understood by runtime
        # construction. Current example candidates: tgn, jodie, apan, tgat,
        # gconv_gru, tgcn, mpnn_lstm, evolvegcn.
        "name": "tgn",
        # Optional. Usually matches data.temporal_representation. Use this only
        # when the backbone semantics need to be explicit. Candidates:
        # event_stream, snapshot_sequence.
        "temporal_representation": "event_stream",
        # Optional. sampled_neighbor lowers to sampled blocks; full_neighbor
        # lowers to full snapshot/block execution. Candidates:
        # sampled_neighbor, full_neighbor.
        "spatial_aggregation": "sampled_neighbor",
        # Optional. coupled means current aggregation reads previous-window
        # neighbor state; decoupled means only same-node recurrent state is
        # needed after aggregation. Candidates: coupled, decoupled.
        "coupling": "coupled",
        # Optional. Persistent event memory/state. Usually inferred for TGN,
        # JODIE, and APAN. Candidates: persistent, stateless,
        # snapshot_recurrent.
        "state": "persistent",
        # Optional. Model semantic key for temporal state reads. The default
        # convention is s for state. Candidates: any model-supported key;
        # current convention: s.
        "state_key": "s",
        # Optional. Model semantic key for aggregation features. The default
        # convention is h for hidden/embedding features. Candidates: any
        # model-supported key; current convention: h.
        "aggregate_key": "h",
        # Optional. Forces temporal-state inference for custom backbones.
        # Candidates: true, false.
        "requires_temporal_state": True,
    },
    # Required. Declares supervision semantics.
    "task": {
        # Required. Current candidates: edge_prediction,
        # node_classification, node_prediction, node_regression. Target and
        # output ownership is derived from this task kind.
        "name": "edge_prediction",
        # Optional. Task services. Negative sampling is only needed by link or
        # edge prediction tasks that train with negative endpoints. Candidates:
        # currently negative_sampling.
        "services": {
            "negative_sampling": {
                # Optional. Number of negatives per positive sample.
                # Candidates: positive integer.
                "ratio": 1,
                # Optional. dst keeps positive src and samples negative dst;
                # src_dst samples both negative src and negative dst.
                # Candidates: dst, src_dst.
                "mode": "dst",
                # Optional. Train-time negative endpoint policy and source
                # mix. Candidates: policy=random; probabilities are floats in
                # [0, 1] that sum to 1.
                "train": {
                    "policy": "random",
                    "local_probability": 0.9,
                    "remote_probability": 0.1,
                },
                # Optional. Evaluation-time negative endpoint policy.
                # Candidates: random.
                "eval": {
                    "policy": "random",
                },
                # Optional. Test-time negative endpoint policy.
                # Candidates: random.
                "test": {
                    "policy": "random",
                },
                # Optional. Extra keyword arguments passed to the sampler
                # service when a concrete implementation supports them.
                # Candidates: empty object or implementation-specific fields.
                "sample_kwargs": {},
            }
        },
    },
    # Optional. Runtime placement, execution, sampling, train, and preprocess
    # options. Everything except data/backbone/task belongs here.
    "runtime": {
        # Optional. Device string. Candidates: cpu, cuda, cuda:0, cuda:1, or
        # any torch-supported device string.
        "device": "cuda",
        # Optional. Safe two-slot access double buffer. The current Batch
        # computes while the next Batch fetches features and permitted
        # bounded-stale state. Exact state remains ordered in execution.
        # Candidates: true, false. Default: true.
        "access_pipeline": True,
        # Optional. Training keeps loss only; evaluation still computes task
        # metrics. Enable only when per-batch training metrics are required.
        # Candidates: true, false. Default: false.
        "train_compute_metrics": False,
        # Optional. User-visible temporal-state semantics. This section
        # expresses exact versus stale temporal state reads.
        "temporal_state": {
            # Optional. Temporal state update operator used by memory/state
            # models. Candidates: gru, rnn, transformer.
            "update": "gru",
            # Optional. Candidates: exact, bounded_stale.
            "consistency": "bounded_stale",
            # Optional. Candidates: integer >= 0.
            "max_staleness": 1,
            # Optional. Filters low-value shared state refreshes/commits.
            "filter": {
                # Optional. Candidates: true, false.
                "enabled": True,
                # Optional. Candidates: float >= 0.
                "min_change_norm": 0.0,
                # Optional. Candidates: null or float >= 0.
                "min_cosine_distance": 0.3,
                # Optional. Candidates: integer >= 0.
                "max_skip": 10,
            },
            # Optional. Smooth stale reads with historical state and accumulated
            # increments. Gamma is learnable; gamma_init is only initialization.
            "smooth_aggregation": {
                # Optional. Candidates: true, false.
                "enabled": True,
                # Optional. Candidates: float in [0, 1].
                "gamma_init": 0.5,
            },
        },
        # Optional. Training loop parameters.
        "train": {
            # Optional. Candidates: positive integer.
            "batch_size": 3000,
            # Optional. Candidates: positive integer.
            "epochs": 50,
            # Optional. Candidates: adam, adamw, sgd.
            "optimizer": "adam",
            # Optional. Candidates: positive float.
            "lr": 0.0004,
            # Optional. Candidates: float >= 0.
            "weight_decay": 0.0,
            # Optional. Candidates: float in [0, 1].
            "dropout": 0.2,
            # Optional. Candidates: float in [0, 1].
            "att_dropout": 0.2,
            # Optional. Positive integer limits each epoch for debugging;
            # null executes every materialized batch.
            "max_batches_per_epoch": None,
        },
        # Optional. Sampling semantics.
        "sampling": {
            # Optional. Graph selection inside each Event/Snapshot window.
            # null defaults Event input to neighbor and Snapshot input to full.
            # Candidates: null, full, neighbor.
            "mode": None,
            # Optional. Window construction, independent of sampling.mode.
            "window": {
                # Optional. null derives event_window for Event input and
                # chunk_decay for full Snapshot input. Explicit Snapshot
                # neighbor sampling uses full_snapshot. Candidates: null,
                # event_window, full_snapshot, chunk_decay.
                "policy": None,
                # Optional. Number of snapshots in one window. Required for
                # string chunk_decay schedules such as half or auto:<ratio>.
                # Candidates: positive integer.
                "snaps_count": 8,
                # Optional. Candidates: list[int], tensor-like sequence, or
                # implementation-supported string schedule.
                "chunk_decay": "half",
                # Optional. Candidates: positive integer.
                "num_full_snapshots": 2,
                # Optional. Chunk visit order for chunk-decay windows.
                # Candidates: rand; future implementations may add loss or
                # perturbation-based orders.
                "chunk_order": "rand",
            },
            # Optional. Neighbor sampler used inside either window mode when
            # sampling.mode=neighbor.
            "neighbor": {
                # Optional. Candidates: list of positive integers; -1 may be
                # reserved by a sampler implementation for all.
                "fanouts": [20],
                # Optional. Candidates: recent, uniform, native, or
                # implementation-supported policy string. When
                # boundary_sampling is enabled, uniform maps to
                # boundary_uniform and recent maps to boundary_decay_sampling.
                "policy": "recent",
                # Optional. Candidates: integer.
                "seed": 0,
                # Optional. Candidates: integer >= 0.
                "workers": 32,
                # Optional. Boundary-aware remote neighbor sampling. This
                # keeps the public spelling as boundary even if a source
                # implementation used a legacy misspelling internally.
                "boundary_sampling": {
                    # Optional. Candidates: true, false.
                    "enabled": True,
                    # Optional. Policy used when neighbor.policy=uniform.
                    # Candidates: boundary_uniform.
                    "uniform_policy": "boundary_uniform",
                    # Optional. Policy used when neighbor.policy=recent.
                    # Candidates: boundary_decay_sampling.
                    "recent_policy": "boundary_decay_sampling",
                    # Optional. Probability for including boundary samples.
                    # Candidates: float in [0, 1].
                    "probability": 0.1,
                },
            },
        },
        # Optional. Data preparation options.
        "preprocess": {
            # Optional. Candidates: positive integer.
            "num_parts": 1,
            # Optional. Number of chunks to split within each partition after
            # the owner partition is assigned. Candidates: positive integer.
            "chunks_per_rank": 1,
            # Optional. Ratio of hot/shared nodes selected for the shared-hot
            # cache plane. Candidates: float in [0, 1].
            "hot_node_ratio": 0.1,
            # Optional. separate keeps all features in feature_XXX.pt;
            # snapshot_csc packs only temporal node features [S, N, F] by
            # Snapshot-CSC source rows as data + ptr. Static node and edge
            # features stay in feature_XXX.pt. Candidates: separate,
            # snapshot_csc.
            "feature_layout": "separate",
        },
    },
    # Optional. Top-level output directory for prepared artifacts/checkpoints.
    # Candidates: local path string.
    "artifact_root": "/path/to/artifacts",
}


# Approximate state-read example. Do not also add a memory, spec, or execution
# section; exact versus stale is expressed only through
# temporal_state.consistency.
BOUNDED_STALE_CONFIG_EXAMPLE: dict[str, Any] = {
    **CONFIG_EXAMPLE,
    "runtime": {
        **CONFIG_EXAMPLE["runtime"],
        "temporal_state": {
            # Candidates: gru, rnn, transformer.
            "update": "gru",
            # Candidates: exact, bounded_stale.
            "consistency": "bounded_stale",
            # Candidates: integer >= 1 for bounded_stale.
            "max_staleness": 2,
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
        },
    },
}


def compile_example(*, artifact_root: str | Path | None = None):
    return compile(
        # Same meaning as CONFIG_EXAMPLE["data"].
        data_source=DataSource(
            source="/path/to/dataset",
            temporal_representation="event_stream",
            name="wiki",
        ),
        # Same meaning as CONFIG_EXAMPLE["backbone"].
        backbone=ModelBackbone(
            name="tgn",
            temporal_representation="event_stream",
            spatial_aggregation="sampled_neighbor",
            coupling="coupled",
            state="persistent",
            state_key="s",
            aggregate_key="h",
            requires_temporal_state=True,
        ),
        # Same meaning as CONFIG_EXAMPLE["task"].
        task_segment=TaskSegment(
            name="edge_prediction",
            negative_sampler=NegativeSampler(
                ratio=1,
                mode="dst",
                policy="random",
                sample_kwargs={},
            ),
        ),
        # Optional. Overrides CONFIG_EXAMPLE["artifact_root"] style output.
        artifact_root=artifact_root,
        # Optional. Same meaning as CONFIG_EXAMPLE["runtime"].
        runtime=CONFIG_EXAMPLE["runtime"],
    )


def from_config_example(*, artifact_root: str | Path | None = None):
    return from_config(CONFIG_EXAMPLE, artifact_root=artifact_root)


__all__ = [
    "BOUNDED_STALE_CONFIG_EXAMPLE",
    "CONFIG_EXAMPLE",
    "compile_example",
    "from_config_example",
]
