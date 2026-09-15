from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from starrygl.utils.index import compact_lookup_rows
from starrygl.view import GraphBlock

from ..graph_conv import edge_rows


class TimeEncode(nn.Module):
    """Cosine time encoder used by temporal attention and memory updates."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.w = nn.Linear(1, self.dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.w.bias.data.zero_()
        weight = 1.0 / torch.pow(10.0, torch.linspace(0, 9, self.dim))
        self.w.weight.data.copy_(weight.reshape(self.dim, 1))

    def forward(self, t: Tensor) -> Tensor:
        return torch.cos(self.w(t.float().reshape(-1, 1)))


class IdentityNormLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(dim))

    def forward(self, h: Tensor) -> Tensor:
        return self.norm(h)


class JODIETimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.time_emb = _NormalLinear(1, int(dim))

    def forward(self, h: Tensor, memory_ts: Tensor, ts: Tensor) -> Tensor:
        time_diff = (ts.to(device=h.device, dtype=h.dtype) - memory_ts.to(device=h.device, dtype=h.dtype)) / (
            ts.to(device=h.device, dtype=h.dtype) + 1
        )
        return h * (1 + self.time_emb(time_diff.reshape(-1, 1)))


class _NormalLinear(nn.Linear):
    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)


class StateIncrementEstimator(nn.Module):
    """Running increment estimator for shared-hot temporal state."""

    def __init__(self, num_rows: int, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.register_buffer("count", torch.zeros(int(num_rows), 1), persistent=False)
        self.register_buffer("increment", torch.zeros(int(num_rows), self.dim), persistent=False)

    def estimate(self, rows: Tensor) -> Tensor:
        rows = rows.long().to(device=self.increment.device)
        self._ensure_rows(rows)
        return self.increment.index_select(0, rows) / self.count.index_select(0, rows).clamp_min(1)

    @torch.no_grad()
    def update(self, rows: Tensor, change: Tensor) -> None:
        if int(rows.numel()) == 0:
            return
        rows = rows.long().to(device=self.increment.device)
        self._ensure_rows(rows)
        ones = torch.ones(int(rows.numel()), 1, dtype=self.count.dtype, device=self.count.device)
        self.count.index_add_(0, rows, ones)
        self.increment.index_add_(0, rows, change.detach().to(device=self.increment.device, dtype=self.increment.dtype))

    @torch.no_grad()
    def clear(self) -> None:
        self.count.zero_()
        self.increment.zero_()

    def _ensure_rows(self, rows: Tensor) -> None:
        if not int(rows.numel()):
            return
        extra = int(rows.max().item()) + 1 - int(self.count.shape[0])
        if extra > 0:
            self.count = torch.cat((self.count, self.count.new_zeros((extra, 1))))
            self.increment = torch.cat((self.increment, self.increment.new_zeros((extra, self.dim))))


class TGNMemoryUpdater(nn.Module):
    def __init__(
        self,
        *,
        memory_dim: int,
        message_dim: int,
        time_dim: int,
        node_dim: int,
        combine_node_feature: bool,
        memory_update: str,
        state_compensation: bool,
        compensation_num_rows: int,
        gamma_init: float,
        mailbox_size: int = 1,
        transformer_heads: int = 2,
        dropout: float = 0.0,
        att_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.memory_dim = int(memory_dim)
        self.message_dim = int(message_dim)
        self.time_dim = int(time_dim)
        self.combine_node_feature = bool(combine_node_feature)
        self.memory_update = str(memory_update)
        self.mailbox_size = max(1, int(mailbox_size))
        self.transformer_heads = max(1, int(transformer_heads))
        self.time_enc = TimeEncode(self.time_dim) if self.time_dim > 0 else None
        self.slot_message_dim = max(1, self.message_dim // self.mailbox_size)
        update_dim = self.message_dim + self.time_dim
        if self.memory_update == "gru":
            self.updater: nn.Module = nn.GRUCell(update_dim, self.memory_dim)
        elif self.memory_update == "rnn":
            self.updater = nn.RNNCell(update_dim, self.memory_dim)
        elif self.memory_update == "transformer":
            if self.memory_dim % self.transformer_heads != 0:
                raise ValueError("memory_dim must be divisible by transformer_heads")
            self.w_q = nn.Linear(self.memory_dim, self.memory_dim)
            self.w_k = nn.Linear(self.slot_message_dim + self.time_dim, self.memory_dim)
            self.w_v = nn.Linear(self.slot_message_dim + self.time_dim, self.memory_dim)
            self.att_act = nn.LeakyReLU(0.2)
            self.layer_norm = nn.LayerNorm(self.memory_dim)
            self.mlp = nn.Linear(self.memory_dim, self.memory_dim)
            self.dropout = nn.Dropout(float(dropout))
            self.att_dropout = nn.Dropout(float(att_dropout))
            self.updater = nn.Identity()
        else:
            raise ValueError(f"unsupported TGN memory_update: {memory_update!r}")
        self.node_feat_map = nn.Linear(int(node_dim), self.memory_dim) if int(node_dim) != self.memory_dim else None
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init))) if state_compensation else None
        self.increment = StateIncrementEstimator(int(compensation_num_rows), self.memory_dim) if state_compensation else None

    def forward(
        self,
        *,
        node_feat: Tensor,
        memory: Tensor,
        memory_ts: Tensor,
        node_ts: Tensor,
        mem_input: Tensor,
        historical_memory: Tensor | None,
        shared_mask: Tensor | None,
        shared_rows: Tensor | None,
        mailbox_ts: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        if self.memory_update == "transformer":
            raw_memory = self._transformer_update(
                memory=memory,
                memory_ts=memory_ts,
                node_ts=node_ts,
                mem_input=mem_input,
                mailbox_ts=mailbox_ts,
            )
        else:
            if self.time_enc is not None:
                mem_input = torch.cat((mem_input, self.time_enc(node_ts - memory_ts)), dim=-1)
            raw_memory = self.updater(mem_input, memory)
        updated_memory = raw_memory
        aux: dict[str, Tensor] = {
            "memory_raw_update": raw_memory,
            "memory_prev": memory,
            "memory_node_ts": node_ts,
        }
        if self.increment is not None and shared_mask is not None and shared_rows is not None and bool(shared_mask.any().item()):
            historical = memory if historical_memory is None else historical_memory.to(device=memory.device, dtype=memory.dtype)
            shared_pos = shared_mask.nonzero(as_tuple=True)[0]
            shared_rows = shared_rows.to(device=memory.device).long()
            predicted = _normalize_transition(
                historical.index_select(0, shared_pos)
                + self.increment.estimate(shared_rows).to(device=memory.device, dtype=memory.dtype)
            )
            gamma = torch.sigmoid(self.gamma).to(device=memory.device, dtype=memory.dtype)
            mixed = gamma * raw_memory.index_select(0, shared_pos) + (1 - gamma) * predicted
            updated_memory = raw_memory.clone()
            updated_memory.index_copy_(0, shared_pos, mixed)
            change = mixed.detach() - historical.index_select(0, shared_pos).detach()
            self.increment.update(shared_rows, change)
            aux.update(
                {
                    "state_compensation_rows": shared_rows.detach(),
                    "state_compensation_pos": shared_pos.detach(),
                    "state_compensation_change": change.detach(),
                    "state_compensation_prediction": predicted.detach(),
                }
            )
        if not self.combine_node_feature:
            h = updated_memory
        elif self.node_feat_map is None:
            h = updated_memory + node_feat
        else:
            h = updated_memory + self.node_feat_map(node_feat)
        aux["memory_updated"] = updated_memory
        return updated_memory, h, aux

    def clear_compensation(self) -> None:
        if self.increment is not None:
            self.increment.clear()

    def _transformer_update(
        self,
        *,
        memory: Tensor,
        memory_ts: Tensor,
        node_ts: Tensor,
        mem_input: Tensor,
        mailbox_ts: Tensor | None,
    ) -> Tensor:
        slots = _fit_mailbox_slots(mem_input, mailbox_size=self.mailbox_size, slot_dim=self.slot_message_dim)
        if self.time_enc is not None:
            if mailbox_ts is None:
                mailbox_ts = memory_ts[:, None].expand(-1, self.mailbox_size)
            mailbox_ts = mailbox_ts.to(device=node_ts.device, dtype=node_ts.dtype).reshape(-1, self.mailbox_size)
            time_feat = self.time_enc(node_ts[:, None] - mailbox_ts).reshape(
                int(node_ts.shape[0]), self.mailbox_size, self.time_dim
            )
            slots = torch.cat((slots, time_feat), dim=-1)
        batch = int(slots.shape[0])
        head_dim = self.memory_dim // self.transformer_heads
        q = self.w_q(memory).reshape(batch, self.transformer_heads, head_dim).unsqueeze(1)
        k = self.w_k(slots).reshape(batch, self.mailbox_size, self.transformer_heads, head_dim)
        v = self.w_v(slots).reshape(batch, self.mailbox_size, self.transformer_heads, head_dim)
        score = self.att_act((q * k).sum(dim=-1))
        att = torch.softmax(score, dim=1)
        context = (self.att_dropout(att).unsqueeze(-1) * v).sum(dim=1).reshape(batch, self.memory_dim)
        out = self.layer_norm(memory + context)
        return torch.relu(self.dropout(self.mlp(out)))


def _fit_mailbox_slots(value: Tensor, *, mailbox_size: int, slot_dim: int) -> Tensor:
    width = int(mailbox_size) * int(slot_dim)
    if int(value.shape[1]) > width:
        value = value[:, :width]
    elif int(value.shape[1]) < width:
        value = torch.cat((value, value.new_zeros((int(value.shape[0]), width - int(value.shape[1])))), dim=1)
    return value.reshape(int(value.shape[0]), int(mailbox_size), int(slot_dim))


def _normalize_transition(value: Tensor) -> Tensor:
    if int(value.numel()) == 0:
        return value
    shifted = value - value.min()
    scale = shifted.max()
    normalized = 2 * shifted / scale.clamp_min(torch.finfo(value.dtype).tiny) - 1
    return torch.where(scale > 0, normalized, value)


class TemporalTransformerAttentionLayer(nn.Module):
    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        time_dim: int,
        num_heads: int,
        out_dim: int,
        dropout: float,
        att_dropout: float,
        score_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.time_dim = int(time_dim)
        self.num_heads = int(num_heads)
        self.out_dim = int(out_dim)
        self.score_scale = score_scale
        if self.out_dim % self.num_heads != 0:
            raise ValueError("out_dim must be divisible by num_heads")
        self.time_enc = TimeEncode(self.time_dim) if self.time_dim > 0 else None
        self.w_q = nn.Linear(self.node_dim + self.time_dim, self.out_dim)
        self.w_k = nn.Linear(self.node_dim + self.edge_dim + self.time_dim, self.out_dim)
        self.w_v = nn.Linear(self.node_dim + self.edge_dim + self.time_dim, self.out_dim)
        self.w_out = nn.Linear(self.node_dim + self.out_dim, self.out_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.att_dropout = nn.Dropout(float(att_dropout))
        self.att_act = nn.LeakyReLU(0.2)
        self.layer_norm = nn.LayerNorm(self.out_dim)

    def forward(self, block: GraphBlock, h_src: Tensor, edge_feat: Tensor, edge_dt: Tensor) -> Tensor:
        src, dst = edge_rows(block, device=h_src.device)
        num_dst = int(block.num_dst or block.dst_nodes.numel())
        if int(src.numel()) == 0:
            return h_src.new_zeros((num_dst, self.out_dim))
        if self.time_enc is None:
            src_time = h_src.new_empty((int(src.numel()), 0))
            dst_time = h_src.new_empty((num_dst, 0))
        else:
            src_time = self.time_enc(edge_dt.to(device=h_src.device, dtype=h_src.dtype))
            dst_time = self.time_enc(h_src.new_zeros((num_dst,)))
        dst_rows = _dst_rows_in_src(block, device=h_src.device)
        dst_h = h_src.index_select(0, dst_rows) if dst_rows is not None else h_src[:num_dst]
        q_base = torch.cat((dst_h, dst_time), dim=-1)
        kv_base = torch.cat((h_src.index_select(0, src), edge_feat, src_time), dim=-1)
        head_dim = self.out_dim // self.num_heads
        q = self.w_q(q_base).index_select(0, dst).reshape(-1, self.num_heads, head_dim)
        k = self.w_k(kv_base).reshape(-1, self.num_heads, head_dim)
        v = self.w_v(kv_base).reshape(-1, self.num_heads, head_dim)
        score = (q * k).sum(dim=-1)
        if self.score_scale is None:
            score = score / math.sqrt(float(head_dim))
        elif self.score_scale != 1.0:
            score = score * self.score_scale
        score = self.att_act(score)
        score = score.nan_to_num(0.0, posinf=1e4, neginf=-1e4)
        att = grouped_softmax(score, dst, num_dst)
        att = att.nan_to_num(0.0)
        msg = (self.att_dropout(att).unsqueeze(-1) * v).reshape(-1, self.out_dim)
        agg = h_src.new_zeros((num_dst, self.out_dim))
        agg.index_add_(0, dst, msg)
        out = self.w_out(torch.cat((dst_h, agg), dim=-1))
        out = self.dropout(torch.relu(out))
        return self.layer_norm(out).nan_to_num(0.0)


def grouped_softmax(score: Tensor, group: Tensor, num_groups: int) -> Tensor:
    max_score = score.new_full((int(num_groups), int(score.shape[1])), -torch.inf)
    max_score.scatter_reduce_(0, group[:, None].expand(-1, int(score.shape[1])), score, reduce="amax", include_self=True)
    exp = torch.exp(score - max_score.index_select(0, group))
    denom = score.new_zeros((int(num_groups), int(score.shape[1])))
    denom.index_add_(0, group, exp)
    return exp / denom.index_select(0, group).clamp_min(1e-12)


def _dst_rows_in_src(block: GraphBlock, *, device: torch.device) -> Tensor | None:
    num_dst = int(block.num_dst or block.dst_nodes.numel())
    if int(block.src_nodes.numel()) >= num_dst and torch.equal(
        block.src_nodes[:num_dst].to(device=block.dst_nodes.device).long(),
        block.dst_nodes[:num_dst].long(),
    ):
        return None
    src_nodes = block.src_nodes.to(device=device).long()
    dst_nodes = block.dst_nodes.to(device=device).long()
    if int(src_nodes.numel()) == 0 or int(dst_nodes.numel()) == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    return compact_lookup_rows(src_nodes, dst_nodes).clamp_min(0)


__all__ = [
    "IdentityNormLayer",
    "JODIETimeEmbedding",
    "StateIncrementEstimator",
    "TGNMemoryUpdater",
    "TemporalTransformerAttentionLayer",
    "TimeEncode",
    "grouped_softmax",
]
