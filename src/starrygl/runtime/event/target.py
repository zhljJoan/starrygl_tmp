from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor

from starrygl.store import StoreBundle
from starrygl.task import NegativeSamplePool, TaskTarget
from starrygl.utils.index import compact_lookup_rows, compact_node_time
from starrygl.utils.route import dist_part


def target_extra_nodes(target: TaskTarget) -> Tensor | None:
    pieces = []
    if target.target_kind == "node":
        nodes = target.node_ids if target.node_ids is not None else target.target_ids
        if nodes is not None and int(nodes.numel()):
            pieces.append(nodes.long())
    for nodes in (target.neg_src, target.neg_dst):
        if nodes is not None and int(nodes.numel()):
            pieces.append(nodes.long())
    return torch.cat(pieces) if pieces else None


def train_positive_rows(
    store: StoreBundle,
    target_src: Tensor,
    target_dst: Tensor,
    *,
    split: str,
    sampler_options: Mapping[str, Any] | None,
    generator: object | None,
) -> Tensor:
    rows = torch.arange(int(target_src.numel()), device=target_src.device)
    prob = float((sampler_options or {}).get("remote_positive_keep_prob", 1.0))
    if str(split) != "train" or prob >= 1.0 or not int(target_src.numel()):
        return rows
    index = store.graph.partition.get("node_dist_index") if store.graph.prepare is not None else None
    if not isinstance(index, Tensor):
        return rows
    src_rank = dist_part(index.long().index_select(0, target_src.to(device=index.device).long()))
    dst_rank = dist_part(index.long().index_select(0, target_dst.to(device=index.device).long()))
    remote = src_rank != dst_rank
    hot = hot_node_ids(store)
    if int(hot.numel()):
        remote &= ~(membership_mask(target_src, hot.to(device=target_src.device)).to(remote.device) |
                    membership_mask(target_dst, hot.to(device=target_dst.device)).to(remote.device))
    if not bool(remote.any().item()):
        return rows
    keep = ~remote
    remote_rows = remote.nonzero(as_tuple=True)[0]
    if prob > 0.0 and int(remote_rows.numel()):
        sampled = torch.rand(
            int(remote_rows.numel()),
            device=remote_rows.device,
            generator=compatible_generator(remote_rows, generator),
        ) < prob
        keep.index_copy_(0, remote_rows, sampled)
    return keep.nonzero(as_tuple=True)[0]


def negative_dst_pool(view: Mapping[str, Any], options: Mapping[str, Any] | None = None) -> NegativeSamplePool:
    options = options or {}
    local = view.get("dst_pool")
    if local is None:
        dst = view.get("dst")
        local = (
            torch.unique(dst.long(), sorted=True)
            if dst is not None and int(dst.numel())
            else torch.empty(0, dtype=torch.long)
        )
    global_ = view.get("global_dst_pool")
    if global_ is None or not int(global_.numel()):
        global_ = local
    hot = view.get("hot_node_ids")
    if isinstance(hot, Tensor) and int(hot.numel()):
        local = merge_local_hot_dst_pool(local.long(), global_.long(), hot.long())
    if str(options.get("negative_global_pool", "")).strip().lower() in {"remote", "remote_dst", "nonlocal_dst"}:
        global_ = cached_remote_dst_pool(view, local.long(), global_.long())
        if not int(global_.numel()):
            global_ = local
    local_prob = float(options.get("negative_local_prob", 1.0))
    global_prob = float(options.get("negative_global_prob", 0.0))
    local_weight = global_weight = 1.0
    p_local, p_global = _negative_mix_prob(local_prob, global_prob)
    correction = str(options.get("negative_weight_correction", "none"))
    if correction in {"inverse_probability", "inverse_prob", "unbiased"}:
        local_weight = 1.0 / max(p_local, 1e-12) if p_local else 0.0
        global_weight = 1.0 / max(p_global, 1e-12) if p_global else 0.0
    elif correction == "memshare" and p_global:
        global_weight = p_local / p_global * max(1.0, float(global_.numel())) / max(1.0, float(local.numel()))
    return NegativeSamplePool(
        mode=str(options.get("negative_mode", "dst")),
        local_node_ids=local.long(),
        global_node_ids=global_.long(),
        local_src_ids=local.long(),
        global_src_ids=global_.long(),
        local_dst_ids=local.long(),
        global_dst_ids=global_.long(),
        local_prob=local_prob,
        global_prob=global_prob,
        local_loss_weight=local_weight,
        global_loss_weight=global_weight,
    )


def compatible_generator(like: Tensor | None, generator: object | None) -> torch.Generator | None:
    if not isinstance(generator, torch.Generator) or like is None:
        return None
    try:
        return generator if str(generator.device) == str(like.device) else None
    except Exception:
        return None


def deduplicate_roots(nodes: Tensor, ts: Tensor | None, *, enabled: bool) -> tuple[Tensor, Tensor | None]:
    if not enabled or int(nodes.numel()) <= 1:
        return nodes.long(), ts
    node_ids = nodes.long()
    pos = torch.arange(int(node_ids.numel()), device=node_ids.device)
    compact_nodes, inverse = compact_node_time(node_ids, ts)
    first = torch.full(
        (int(compact_nodes.numel()),),
        int(node_ids.numel()),
        dtype=torch.long,
        device=node_ids.device,
    )
    first.scatter_reduce_(0, inverse, pos, reduce="amin", include_self=True)
    rows = (first.index_select(0, inverse) == pos).nonzero(as_tuple=True)[0]
    return node_ids.index_select(0, rows), None if ts is None else ts.index_select(0, rows.to(device=ts.device))


def cached_remote_dst_pool(view: Mapping[str, Any], local: Tensor, global_: Tensor) -> Tensor:
    if not int(local.numel()) or not int(global_.numel()):
        return global_.long()
    cached = view.get("_dst_pool_remote")
    if isinstance(cached, Tensor):
        return cached
    keep = compact_lookup_rows(local.long().cpu(), global_.long().cpu()) < 0
    remote = global_.long().index_select(0, keep.to(global_.device).nonzero(as_tuple=True)[0])
    if isinstance(view, dict):
        view["_dst_pool_remote"] = remote
    return remote


def hot_node_ids(store: StoreBundle) -> Tensor:
    if store.graph.prepare is None:
        return torch.empty(0, dtype=torch.long)
    hot = store.graph.partition.get("hot_node_ids")
    return hot.long() if isinstance(hot, Tensor) else torch.empty(0, dtype=torch.long)


def merge_local_hot_dst_pool(local: Tensor, global_: Tensor, hot: Tensor) -> Tensor:
    if not int(hot.numel()):
        return local.long()
    if int(global_.numel()):
        hot = hot[membership_mask(hot, global_)]
    return torch.unique(torch.cat((local.long(), hot.to(local.device))), sorted=True)


def membership_mask(values: Tensor, candidates: Tensor) -> Tensor:
    if not int(values.numel()) or not int(candidates.numel()):
        return torch.zeros(int(values.numel()), dtype=torch.bool, device=values.device)
    candidates = torch.unique(candidates.long().to(values.device), sorted=True)
    idx = torch.searchsorted(candidates, values.long())
    valid = idx < int(candidates.numel())
    return valid & (candidates.index_select(0, idx.clamp_max(int(candidates.numel()) - 1)) == values.long())


def _negative_mix_prob(local: float, global_: float) -> tuple[float, float]:
    local, global_ = max(0.0, local), max(0.0, global_)
    total = local + global_
    return (local / total, global_ / total) if total else (1.0, 0.0)
