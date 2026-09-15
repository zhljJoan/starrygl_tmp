from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from starrygl.view import GraphBlock
from starrygl.view.base import mfg_chain_matches


def empty_sample_block(root_nodes: Tensor) -> GraphBlock:
    empty = root_nodes.new_empty((0,))
    roots = root_nodes.long()
    return GraphBlock(
        src_nodes=roots,
        dst_nodes=roots,
        edge_ids=empty,
        format="csc",
        indptr=torch.zeros(int(roots.numel()) + 1, dtype=torch.long, device=roots.device),
        indices=empty,
        row=empty,
        col=empty,
        num_src=int(roots.numel()),
        num_dst=int(roots.numel()),
        exec_mode="LOCAL_SAMPLE",
    )


def normalize_mfg_blocks(blocks: Sequence[GraphBlock]) -> tuple[GraphBlock, ...]:
    return drop_trailing_empty_blocks(ordered_mfg_blocks(tuple(blocks)))


def ordered_mfg_blocks(blocks: tuple[GraphBlock, ...]) -> tuple[GraphBlock, ...]:
    if len(blocks) <= 1 or mfg_chain_matches(blocks):
        return blocks
    reversed_blocks = tuple(reversed(blocks))
    if mfg_chain_matches(reversed_blocks):
        return reversed_blocks
    return blocks


def drop_trailing_empty_blocks(blocks: tuple[GraphBlock, ...]) -> tuple[GraphBlock, ...]:
    keep = len(blocks)
    while keep > 1:
        block = blocks[keep - 1]
        if int(block.dst_nodes.numel()) != 0 or int(block.edge_ids.numel()) != 0:
            break
        keep -= 1
    return blocks[:keep]


def native_feature_block(blocks: Sequence[GraphBlock]) -> GraphBlock:
    return blocks[0]


def native_target_block(blocks: Sequence[GraphBlock]) -> GraphBlock:
    return blocks[-1] if mfg_chain_matches(blocks) else blocks[0]


def sampled_feature_graph(blocks: Sequence[GraphBlock]) -> GraphBlock:
    if len(blocks) <= 1:
        return native_feature_block(blocks)
    pieces = [block.src_nodes.long() for block in blocks if int(block.src_nodes.numel()) > 0]
    if not pieces:
        return native_feature_block(blocks)
    nodes = torch.unique(torch.cat(pieces, dim=0), sorted=True)
    empty = nodes.new_empty((0,))
    return GraphBlock(
        src_nodes=nodes,
        dst_nodes=nodes,
        edge_ids=empty,
        format="native",
        row=empty,
        col=empty,
        edge_index=torch.empty((2, 0), dtype=torch.long, device=nodes.device),
        num_src=int(nodes.numel()),
        num_dst=int(nodes.numel()),
        exec_mode="LOCAL_SAMPLE_FEATURE_TABLE",
    )


def sampled_edge_ids(blocks: Sequence[GraphBlock]) -> Tensor:
    values = [
        block.cache.get("edge_feature_ids", block.edge_ids).long()
        for block in blocks
        if int(block.edge_ids.numel())
    ]
    if not values:
        return torch.empty(0, dtype=torch.long)
    return torch.unique(torch.cat(values, dim=0), sorted=True)


def unique_mfg_edge_ids(mfgs: Sequence[Sequence[GraphBlock]]) -> Tensor | None:
    pieces = [
        block.cache.get("edge_feature_ids", block.edge_ids).long()
        for window in mfgs
        for block in window
        if int(block.edge_ids.numel()) > 0
    ]
    if not pieces:
        return None
    return torch.unique(torch.cat(pieces, dim=0), sorted=True)


__all__ = [
    "empty_sample_block",
    "native_feature_block",
    "native_target_block",
    "normalize_mfg_blocks",
    "sampled_edge_ids",
    "sampled_feature_graph",
    "unique_mfg_edge_ids",
]
