from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from starrygl.view import GraphBlock

from .sampling_output import (
    _infer_num_nodes,
    _int_part,
    _uint8_mask,
    batch_from_native_sampling_output,
    blocks_from_native_sampling_output,
    graph_block_from_native_mfg,
)

class NativeSamplingUnavailable(RuntimeError):
    pass


@dataclass
class NativeTemporalSampler:
    native_sampler: Any
    output: str = "compact"
    materialize_col: bool = True
    edge_id_map: Tensor | None = None
    edge_feature_id_map: Tensor | None = None
    deduplicate_edges: bool = True
    deduplicate_nodes: bool = False
    approximate_node_compaction: bool = False
    native_deduplicates_nodes: bool = False
    profile_stats: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_graph(
        cls,
        graph: Mapping[str, Any],
        *,
        fanouts: Sequence[int],
        num_layers: int | None = None,
        workers: int = 1,
        policy: str = "recent",
        local_part: int = 0,
        edge_part: Tensor | None = None,
        node_part: Tensor | None = None,
        node_is_hot: Tensor | None = None,
        probability: float = 1.0,
        graph_name: str = "starrygl_events",
        output: str = "compact",
        add_reverse_edges: bool = False,
        materialize_col: bool = True,
        deduplicate_edges: bool = True,
        deduplicate_nodes: bool = False,
        approximate_node_compaction: bool = False,
        seed: int | None = None,
    ) -> "NativeTemporalSampler":
        if bool(deduplicate_nodes) and not bool(approximate_node_compaction):
            raise ValueError(
                "deduplicate_nodes compacts sampled rows by node id and can merge "
                "different timestamps for the same node. Pass "
                "approximate_node_compaction=True only for models that explicitly "
                "accept node-id-level sampled MFG approximation."
            )
        src = torch.as_tensor(graph["src"], dtype=torch.long).cpu().contiguous()
        dst = torch.as_tensor(graph["dst"], dtype=torch.long).cpu().contiguous()
        edge_id_map = torch.as_tensor(graph.get("edge_ids", torch.arange(int(src.numel()))), dtype=torch.long).cpu().contiguous()
        feature_ids = graph.get("edge_feature_ids")
        edge_feature_id_map = torch.as_tensor(
            torch.arange(int(src.numel())) if feature_ids is None else feature_ids,
            dtype=torch.long,
        ).cpu().contiguous()
        edge_ids = torch.arange(int(src.numel()), dtype=torch.long)
        if int(edge_id_map.numel()) != int(src.numel()) or int(edge_feature_id_map.numel()) != int(src.numel()):
            raise ValueError("edge_ids and edge_feature_ids must have one value per sampler edge")
        ts = graph.get("timestamps", graph.get("ts"))
        timestamps = None if ts is None else torch.as_tensor(ts, dtype=torch.long).cpu().contiguous()
        num_nodes = int(graph.get("num_nodes", _infer_num_nodes(src, dst)))
        if add_reverse_edges and int(src.numel()):
            src, dst = torch.cat((src, dst), dim=0).contiguous(), torch.cat((dst, src), dim=0).contiguous()
            edge_ids = torch.cat((edge_ids, edge_ids), dim=0).contiguous()
            edge_id_map = torch.cat((edge_id_map, edge_id_map), dim=0).contiguous()
            edge_feature_id_map = torch.cat((edge_feature_id_map, edge_feature_id_map), dim=0).contiguous()
            timestamps = None if timestamps is None else torch.cat((timestamps, timestamps), dim=0).contiguous()
            edge_part = None if edge_part is None else torch.cat((edge_part, edge_part), dim=0).contiguous()

        mod = _load_native_sampler_module()
        temporal_graph = mod.get_neighbors(
            str(graph_name),
            src,
            dst,
            int(num_nodes),
            0,
            edge_ids,
            None,
            None,
            None if timestamps is None else timestamps,
        )
        sampler = mod.ParallelSampler(
            temporal_graph,
            int(num_nodes),
            int(src.numel()),
            max(1, int(workers)),
            [int(fanout) for fanout in fanouts],
            int(len(fanouts) if num_layers is None else num_layers),
            _normalize_native_policy(policy),
            int(local_part),
            _int_part(edge_part, int(src.numel())),
            _int_part(node_part, int(num_nodes)),
            _uint8_mask(node_is_hot, int(num_nodes)),
            float(probability),
        )
        if seed is not None:
            set_seed = getattr(sampler, "set_seed", None)
            if callable(set_seed):
                set_seed(int(seed))
        native_deduplicates_nodes = False
        set_compact_node_ids = getattr(sampler, "set_compact_node_ids", None)
        if bool(approximate_node_compaction) and callable(set_compact_node_ids):
            set_compact_node_ids(bool(deduplicate_nodes))
            native_deduplicates_nodes = bool(deduplicate_nodes)
        return cls(
            native_sampler=sampler,
            output=str(output),
            materialize_col=bool(materialize_col),
            edge_id_map=edge_id_map,
            edge_feature_id_map=edge_feature_id_map,
            deduplicate_edges=bool(deduplicate_edges) and not native_deduplicates_nodes,
            deduplicate_nodes=bool(deduplicate_nodes) and not native_deduplicates_nodes,
            approximate_node_compaction=bool(approximate_node_compaction),
            native_deduplicates_nodes=native_deduplicates_nodes,
        )

    def sample_blocks(self, root_nodes: Tensor, root_ts: Tensor | None = None) -> tuple[GraphBlock, ...]:
        roots = root_nodes.long().cpu().contiguous()
        timestamps = None if root_ts is None else root_ts.to(dtype=torch.long).cpu().contiguous()
        counter_before = _native_profile_counter_values(self.native_sampler)
        total_start = time.perf_counter()
        start = total_start
        self.native_sampler.neighbor_sample_from_nodes(roots, timestamps, None)
        neighbor_seconds = time.perf_counter() - start
        start = time.perf_counter()
        output = self._sampling_output(roots, timestamps)
        output_seconds = time.perf_counter() - start
        start = time.perf_counter()
        blocks = blocks_from_native_sampling_output(
            output,
            materialize_col=self.materialize_col,
            edge_id_map=self.edge_id_map,
            edge_feature_id_map=self.edge_feature_id_map,
            deduplicate_edges=self.deduplicate_edges,
            deduplicate_nodes=self.deduplicate_nodes,
        )
        block_build_seconds = time.perf_counter() - start
        total_seconds = time.perf_counter() - total_start
        sampled_edges = sum(int(block.edge_ids.numel()) for block in blocks)
        counter_after = _native_profile_counter_values(self.native_sampler)
        counter_delta = {
            name: float(value) - float(counter_before.get(name, 0.0))
            for name, value in counter_after.items()
        }
        last = {
            "native_root_count": float(roots.numel()),
            "native_root_ts_count": 0.0 if timestamps is None else float(timestamps.numel()),
            "native_neighbor_sample_seconds": float(neighbor_seconds),
            "native_sampling_output_seconds": float(output_seconds),
            "native_block_build_seconds": float(block_build_seconds),
            "native_sample_total_seconds": float(total_seconds),
            "native_sample_blocks": float(len(blocks)),
            "native_sample_edges": float(sampled_edges),
            "sampler_materialize_col_enabled": 1.0 if self.materialize_col else 0.0,
        }
        for name, value in counter_delta.items():
            if value == 0:
                continue
            last[f"native_counter_{name}"] = float(value)
        self.profile_stats.update({f"last_{key}": value for key, value in last.items()})
        for key, value in last.items():
            self.profile_stats[key] = self.profile_stats.get(key, 0.0) + float(value)
        self.profile_stats["sampler_blocks"] = self.profile_stats.get("sampler_blocks", 0.0) + float(len(blocks))
        self.profile_stats["sampler_edges"] = self.profile_stats.get("sampler_edges", 0.0) + float(sampled_edges)
        return blocks

    def _sampling_output(self, root_nodes: Tensor, root_ts: Tensor | None) -> Any:
        if self.output == "parallel" and hasattr(self.native_sampler, "get_sampling_output_parallel"):
            return self.native_sampler.get_sampling_output_parallel(root_nodes, root_ts)
        if self.output in {"compact", "compact_raw_edges"} and hasattr(self.native_sampler, "get_sampling_output_compact"):
            set_raw_edges = getattr(self.native_sampler, "set_compact_raw_edge_ids", None)
            if callable(set_raw_edges):
                set_raw_edges(self.output == "compact_raw_edges")
            return self.native_sampler.get_sampling_output_compact(root_nodes, root_ts)
        if hasattr(self.native_sampler, "get_sampling_output"):
            return self.native_sampler.get_sampling_output(root_nodes, root_ts)
        raise NativeSamplingUnavailable("native sampler does not expose a sampling output method")

    def reset_profile_stats(self) -> None:
        self.profile_stats.clear()
        for name in _native_profile_counter_names(self.native_sampler):
            try:
                setattr(self.native_sampler, name, 0.0)
            except Exception:
                pass

    def pop_profile_stats(self) -> dict[str, float]:
        out = {key: float(value) for key, value in self.profile_stats.items()}
        self.profile_stats.clear()
        for name in _native_profile_counter_names(self.native_sampler):
            try:
                out[name] = float(getattr(self.native_sampler, name))
            except Exception:
                continue
        out.setdefault("sampler_materialize_col_enabled", 0.0)
        return out

def _normalize_native_policy(policy: str) -> str:
    value = str(policy).strip().lower()
    aliases = {
        "latest": "recent",
        "random": "uniform",
        "boundary_recent": "boundary_recent_uniform",
        "boundary_recent_sample": "boundary_recent_uniform",
        "boundary_uniform": "boundery_uniform",
        "boundary_uniform_sampling": "boundery_uniform",
        "boundary_decay": "boundary_recent_decay",
        "boundary_decay_sampling": "boundary_recent_decay",
        "boundery_recent": "boundery_recent_uniform",
        "boundery_recent_sample": "boundery_recent_uniform",
        "boundery_uniform": "boundery_uniform",
        "boundery_uniform_sampling": "boundery_uniform",
        "boundery_decay": "boundery_recent_decay",
        "boundery_decay_sampling": "boundery_recent_decay",
    }
    value = aliases.get(value, value)
    if value.startswith("boundary_"):
        return "boundery_" + value[len("boundary_"):]
    return value


def _native_profile_counter_names(native_sampler: Any) -> tuple[str, ...]:
    names = []
    for name in dir(native_sampler):
        if name.startswith("_"):
            continue
        if not (
            name.endswith("_seconds")
            or name.endswith("_rows")
            or name.endswith("_edges")
            or name.endswith("_nodes")
            or name.startswith("compact_")
            or name.startswith("sampler_")
            or name in {"neighbor_sample_from_nodes_seconds", "get_sampling_output_seconds"}
        ):
            continue
        try:
            value = getattr(native_sampler, name)
        except Exception:
            continue
        if isinstance(value, (int, float)):
            names.append(str(name))
    return tuple(sorted(set(names)))


def _native_profile_counter_values(native_sampler: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in _native_profile_counter_names(native_sampler):
        try:
            value = getattr(native_sampler, name)
        except Exception:
            continue
        if isinstance(value, (int, float)):
            out[str(name)] = float(value)
    return out


def _load_native_sampler_module():
    try:
        import starrygl.native.lib.libstarrygl_sampler as mod
        return mod
    except ImportError as exc:
        raise NativeSamplingUnavailable(
            "StarryGL native sampler is unavailable; run "
            "`bash scripts/build_native.sh` from the source tree"
        ) from exc

__all__ = [
    "NativeSamplingUnavailable",
    "NativeTemporalSampler",
    "batch_from_native_sampling_output",
    "blocks_from_native_sampling_output",
    "graph_block_from_native_mfg",
]
