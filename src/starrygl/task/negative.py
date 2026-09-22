from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor

from .target import NegativeSamplePool, TaskTarget


def snapshot_negative_dst_pool(
    *,
    local_dst_ids: Tensor,
    global_dst_ids: Tensor,
    split: str,
    options,
) -> NegativeSamplePool:
    if str(split).strip().lower() == "train":
        local_prob = float(options.get("negative_local_prob", 0.9))
        global_prob = float(options.get("negative_global_prob", 0.1))
    elif str(options.get("eval_negative_dst_pool", options.get("negative_eval_dst_pool", "global_dst"))).lower() in {"local", "local_dst"}:
        local_prob, global_prob = 1.0, 0.0
    else:
        local_prob, global_prob = 0.0, 1.0
    total = local_prob + global_prob
    local_prob, global_prob = ((1.0, 0.0) if total <= 0 else (local_prob / total, global_prob / total))
    return NegativeSamplePool(
        mode=str(options.get("negative_mode", "dst")),
        local_node_ids=local_dst_ids,
        global_node_ids=global_dst_ids,
        local_src_ids=local_dst_ids,
        global_src_ids=global_dst_ids,
        local_dst_ids=local_dst_ids,
        global_dst_ids=global_dst_ids,
        local_prob=local_prob,
        global_prob=global_prob,
    )


def materialize_negative_samples(
    target: TaskTarget,
    *,
    num_negatives: int,
    generator: torch.Generator | None = None,
) -> TaskTarget:
    if int(num_negatives) < 1:
        return replace(target, neg_src=_empty_like_ids(target.target_ids), neg_dst=_empty_like_ids(target.target_ids), neg_loss_weight=None)
    pool = target.negative_pool
    if pool is None:
        raise ValueError("TaskTarget requires negative_pool")
    if pool.mode not in {"dst", "src_dst"}:
        raise ValueError(f"unsupported negative sampling mode: {pool.mode!r}")
    if target.pos_src is None:
        raise ValueError("dst negative sampling requires pos_src")
    neg_dst, neg_weight = _sample_dst_pool(
        pool,
        count=int(target.pos_src.numel()) * int(num_negatives),
        like=target.pos_src,
        generator=generator,
    )
    neg_src = None
    if pool.mode == "src_dst":
        neg_src, _ = _sample_src_pool(
            pool,
            count=int(target.pos_src.numel()) * int(num_negatives),
            like=target.pos_src,
            generator=generator,
        )
    return replace(target, neg_src=neg_src, neg_dst=neg_dst, neg_loss_weight=neg_weight)


def _sample_dst_pool(
    pool: NegativeSamplePool,
    *,
    count: int,
    like: Tensor,
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor]:
    local = _dst_candidates(pool.local_dst_ids, pool.local_node_ids, like)
    global_ = _dst_candidates(pool.global_dst_ids, pool.global_node_ids, like)
    values, weight = _sample_mixed(pool, local=local, global_=global_, count=count, generator=generator)
    if pool.loss_weight_fn is not None:
        weight = _weight_tensor(pool.loss_weight_fn(values, pool), count=count)
    return values, weight


def _sample_src_pool(
    pool: NegativeSamplePool,
    *,
    count: int,
    like: Tensor,
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor]:
    local = _dst_candidates(pool.local_src_ids, pool.local_node_ids, like)
    global_ = _dst_candidates(pool.global_src_ids, pool.global_node_ids, like)
    return _sample_mixed(pool, local=local, global_=global_, count=count, generator=generator)


def _sample_mixed(
    pool: NegativeSamplePool,
    *,
    local: Tensor,
    global_: Tensor,
    count: int,
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor]:
    if int(local.numel()) == 0 and int(global_.numel()) == 0:
        raise ValueError("negative pool is empty")
    if int(global_.numel()) == 0 or float(pool.global_prob) <= 0.0:
        return _draw(local, count=count, weight=pool.local_loss_weight, generator=generator)
    if int(local.numel()) == 0 or float(pool.local_prob) <= 0.0:
        return _draw(global_, count=count, weight=pool.global_loss_weight, generator=generator)

    global_mask = torch.rand(count, generator=generator) < _global_probability(pool)
    out = torch.empty(count, dtype=local.dtype)
    weight = _weight_tensor(pool.local_loss_weight, count=count)
    local_count = int((~global_mask).sum().item())
    global_count = int(global_mask.sum().item())
    if local_count:
        out[~global_mask], weight[~global_mask] = _draw(local, count=local_count, weight=pool.local_loss_weight, generator=generator)
    if global_count:
        global_values, global_weight = _draw(global_, count=global_count, weight=pool.global_loss_weight, generator=generator)
        out[global_mask] = global_values
        weight[global_mask] = global_weight
    return out, weight


def _dst_candidates(primary: Tensor | None, fallback: Tensor | None, like: Tensor) -> Tensor:
    values = primary if primary is not None else fallback
    if values is None:
        return like.new_empty((0,))
    return values.to(dtype=like.dtype, device=like.device).flatten()


def _draw(candidates: Tensor, *, count: int, weight: float | Tensor, generator: torch.Generator | None) -> tuple[Tensor, Tensor]:
    if int(candidates.numel()) == 0:
        raise ValueError("negative pool is empty")
    index = torch.randint(int(candidates.numel()), (int(count),), generator=generator, device=candidates.device)
    return candidates.index_select(0, index), _weight_tensor(weight, count=count)


def _weight_tensor(weight: float | Tensor, *, count: int) -> Tensor:
    if isinstance(weight, Tensor):
        if int(weight.numel()) == 1:
            return weight.reshape(1).expand(int(count)).clone()
        if int(weight.numel()) == int(count):
            return weight.clone()
        raise ValueError("negative pool loss weight must be scalar or match sample count")
    return torch.full((int(count),), float(weight), dtype=torch.float32)


def _global_probability(pool: NegativeSamplePool) -> float:
    total = max(0.0, float(pool.local_prob)) + max(0.0, float(pool.global_prob))
    if total == 0.0:
        return 0.0
    return max(0.0, float(pool.global_prob)) / total


def _empty_like_ids(ids: Tensor) -> Tensor:
    return ids.new_empty((0,))


__all__ = [
    "materialize_negative_samples",
]
