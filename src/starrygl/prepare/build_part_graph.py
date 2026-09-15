from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from starrygl.partition import PartitionPlan
from starrygl.plan import ViewPlan
from starrygl.prepare.event import build_event_view_for_rank
from starrygl.prepare.partition_index import build_partition_indices
from starrygl.prepare.split import (
    concat_time_ptr_2,
    resolve_split_ranges,
    resolve_split_time_ptr_2,
    resolve_time_ptr_2,
    split_masks as build_split_masks,
)
from starrygl.prepare.snapshot_csc import build_snapshot_csc_views
from starrygl.prepare.temporal_csr import build_temporal_csr_view
from starrygl.utils.route import build_state_write_mask

PREPARE_FORMAT = "starrygl_prepare_v2"


@dataclass(frozen=True)
class PrepareConfig:
    world_size: int
    chunks_per_rank: int = 1
    num_time_slices: int | None = None
    time_ptr_2: Tensor | None = None
    time_split: str = "equal_edges"
    target_batch_size: int | None = None
    partition_backend: str = "speed_partition"
    speed_partition_beta: float = 0.1
    speed_partition_topk_ratio: float = 0.01
    speed_partition_topk_type: str = "degree"
    split_ratios: tuple[float, float, float] = (1.0, 0.0, 0.0)
    include_state_write_routes: bool = True
    profile_prepare: bool = False

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError("world_size must be >= 1")
        if self.chunks_per_rank < 1:
            raise ValueError("chunks_per_rank must be >= 1")
        if self.time_split not in {"equal_edges", "batch", "adaptive_batch"}:
            raise ValueError("time_split must be one of: equal_edges, batch, adaptive_batch")
        if self.time_split in {"batch", "adaptive_batch"}:
            if self.target_batch_size is None or int(self.target_batch_size) < 1:
                raise ValueError("target_batch_size must be >= 1 for batch time splitting")
        if self.partition_backend not in {"speed_partition", "round_robin"}:
            raise ValueError("partition_backend must be speed_partition or round_robin")
        if len(self.split_ratios) != 3:
            raise ValueError("split_ratios must have three values: train, val, test")
        if any(float(v) < 0 for v in self.split_ratios):
            raise ValueError("split_ratios must be non-negative")
        if sum(float(v) for v in self.split_ratios) <= 0:
            raise ValueError("split_ratios must contain at least one positive value")


@dataclass(frozen=True)
class PreparedViews:
    partition: dict[str, Tensor]
    time_ptr_2: Tensor
    event_views: list[dict[str, Any]]
    temporal_csr_view: dict[str, Any]
    snapshot_csc_views: list[dict[str, Any]]
    split_masks: dict[str, Tensor] = field(default_factory=dict)
    split_time_ptr_2: dict[str, Tensor] = field(default_factory=dict)
    split_ranges: dict[str, tuple[int, int]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": PREPARE_FORMAT,
            "partition": self.partition,
            "time_ptr_2": self.time_ptr_2,
            "event_views": self.event_views,
            "temporal_csr_view": self.temporal_csr_view,
            "snapshot_csc_views": self.snapshot_csc_views,
            "split_masks": self.split_masks,
            "split_time_ptr_2": self.split_time_ptr_2,
            "split_ranges": dict(self.split_ranges),
            "meta": dict(self.meta),
        }


def materialize_graph_views(
    *,
    src: Tensor,
    dst: Tensor,
    ts: Tensor | None = None,
    num_nodes: int | None = None,
    config: PrepareConfig,
    edge_ids: Tensor | None = None,
    edge_weight: Tensor | None = None,
    split_labels: Tensor | None = None,
    partition_plan: PartitionPlan,
    view_plan: ViewPlan,
) -> PreparedViews:
    """Materialize only the layouts requested by ``view_plan``."""

    src = src.long().cpu().contiguous()
    dst = dst.long().cpu().contiguous()
    ts = None if ts is None else ts.cpu().contiguous()
    edge_count = int(src.numel())
    if int(dst.numel()) != edge_count:
        raise ValueError("src and dst must have the same length")
    if edge_ids is None:
        edge_ids = torch.arange(edge_count, dtype=torch.long)
    else:
        edge_ids = edge_ids.long().cpu().contiguous()
    if int(edge_ids.numel()) != edge_count:
        raise ValueError("edge_ids must have one value per edge")
    if edge_weight is not None:
        edge_weight = edge_weight.float().cpu().reshape(-1).contiguous()
        if int(edge_weight.numel()) != edge_count:
            raise ValueError("edge_weight must have one value per edge")
    if split_labels is not None:
        split_labels = split_labels.long().cpu().contiguous()
        if int(split_labels.numel()) != edge_count:
            raise ValueError("split_labels must have one value per edge")
    if num_nodes is None:
        max_node = int(torch.cat((src, dst)).max().item()) if edge_count else -1
        num_nodes = max_node + 1
    num_nodes = int(num_nodes)
    profile = bool(config.profile_prepare)
    stage_time = time.perf_counter()

    def mark(stage: str) -> None:
        nonlocal stage_time
        if not profile:
            return
        now = time.perf_counter()
        print(f"[prepare] {stage}: {now - stage_time:.3f}s", flush=True)
        stage_time = now

    split_ranges = resolve_split_ranges(edge_count=edge_count, ts=ts, config=config, split_labels=split_labels)
    split_masks = build_split_masks(edge_count=edge_count, split_ranges=split_ranges, split_labels=split_labels)
    split_time_ptr_2 = resolve_split_time_ptr_2(
        edge_count=edge_count,
        ts=ts,
        config=config,
        split_ranges=split_ranges,
    )
    time_ptr_2 = (
        resolve_time_ptr_2(edge_count=edge_count, ts=ts, config=config)
        if config.time_ptr_2 is not None
        else concat_time_ptr_2(split_time_ptr_2)
    )
    mark("time_split")
    node_master = partition_plan.node_master
    edge_master = partition_plan.edge_master
    hot_node_ids = partition_plan.shared_nodes
    if node_master is None or edge_master is None or hot_node_ids is None:
        raise ValueError("PartitionPlan must define node_master, edge_master, and shared_nodes")
    mark("partition")
    partition_indices = build_partition_indices(plan=partition_plan, num_nodes=num_nodes)
    node_is_hot = partition_indices.node_is_hot
    node_to_chunk = partition_indices.node_to_chunk
    edge_chunk = partition_indices.edge_chunk
    node_dist_index = partition_indices.node_dist_index
    edge_dist_index = partition_indices.edge_dist_index
    mark("chunk_and_dist_index")
    partition = {
        "node_dist_index": node_dist_index,
        "edge_dist_index": edge_dist_index,
        "hot_node_ids": hot_node_ids,
        "hot_node_owner_index": node_dist_index.index_select(0, hot_node_ids) if int(hot_node_ids.numel()) else torch.empty(0, dtype=torch.long),
        "hot_count": torch.tensor(int(hot_node_ids.numel()), dtype=torch.long),
        "node_is_hot": node_is_hot,
        "node_to_chunk": node_to_chunk,
        "edge_chunk": edge_chunk,
    }
    if view_plan.requires("event_view"):
        state_write_mask = build_state_write_mask(
            src=src,
            dst=dst,
            time_ptr_2=time_ptr_2,
            num_nodes=num_nodes,
        )
        mark("state_write_mask")
        global_dst_pool = torch.unique(dst, sorted=True) if edge_count else torch.empty(0, dtype=torch.long)
        mark("global_dst_pool")
        event_views = [
            build_event_view_for_rank(
                rank=rank,
                src=src,
                dst=dst,
                ts=ts,
                edge_master=edge_master,
                time_ptr_2=time_ptr_2,
                split_time_ptr_2=split_time_ptr_2,
                state_write_mask=state_write_mask,
                node_dist_index=node_dist_index,
                node_is_hot=node_is_hot,
                hot_node_ids=hot_node_ids,
                global_dst_pool=global_dst_pool,
                world_size=int(config.world_size),
                include_state_write_routes=bool(config.include_state_write_routes),
            )
            for rank in range(int(config.world_size))
        ]
    else:
        event_views = []
    mark("event_views")
    if view_plan.requires("temporal_csr"):
        temporal_csr_view = build_temporal_csr_view(
            src=src,
            dst=dst,
            ts=ts,
            edge_ids=edge_ids,
            node_dist_index=node_dist_index,
            edge_dist_index=edge_dist_index,
            node_to_chunk=node_to_chunk,
            edge_chunk=edge_chunk,
            time_ptr_2=time_ptr_2,
            num_nodes=num_nodes,
            bidirectional=bool(view_plan.temporal_csr_bidirectional),
            shared=bool(view_plan.temporal_csr_shared),
            node_is_hot=node_is_hot,
        )
    else:
        temporal_csr_view = {}
    mark("temporal_csr_view")
    if view_plan.requires("snapshot_csc"):
        snapshot_csc_views = build_snapshot_csc_views(
            src=src,
            dst=dst,
            ts=ts,
            edge_ids=edge_ids,
            edge_weight=edge_weight,
            edge_dist_index=edge_dist_index,
            node_master=node_master,
            hot_node_ids=hot_node_ids,
            node_is_hot=node_is_hot,
            node_to_chunk=node_to_chunk,
            time_ptr_2=time_ptr_2,
            num_nodes=num_nodes,
            world_size=int(config.world_size),
            diffusion=view_plan.requires("snapshot_diffusion"),
            hot_compute=view_plan.requires("snapshot_hot_compute"),
        )
    else:
        snapshot_csc_views = []
    mark("snapshot_csc_views")
    return PreparedViews(
        partition=partition,
        time_ptr_2=time_ptr_2,
        event_views=event_views,
        temporal_csr_view=temporal_csr_view,
        snapshot_csc_views=snapshot_csc_views,
        split_masks=split_masks,
        split_time_ptr_2=split_time_ptr_2,
        split_ranges=split_ranges,
        meta={
            "world_size": int(config.world_size),
            "chunks_per_rank": int(config.chunks_per_rank),
            "num_nodes": num_nodes,
            "num_edges": edge_count,
            "time_split": config.time_split,
            "target_batch_size": config.target_batch_size,
            "split_ratios": tuple(float(v) for v in config.split_ratios),
            "split_source": "labels" if split_labels is not None else "ratios",
            "split_ranges": dict(split_ranges),
            "partition_backend": config.partition_backend,
            "speed_partition_beta": float(config.speed_partition_beta),
            "speed_partition_topk_ratio": float(config.speed_partition_topk_ratio),
            "speed_partition_topk_type": str(config.speed_partition_topk_type),
            "view_plan": view_plan.as_dict(),
        },
    )


__all__ = ["PREPARE_FORMAT", "PrepareConfig", "PreparedViews", "materialize_graph_views"]
