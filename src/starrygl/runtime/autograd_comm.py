from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.autograd import Function


class _PullContext:
    def __init__(self, route: Any, x_shape: tuple[int, ...], group: dist.ProcessGroup | None) -> None:
        self.route = route
        self.x_shape = x_shape
        self.group = group
        self.forward_task: tuple[Tensor, Tensor, Tensor, dist.Work] | Tensor | None = None
        self.backward_task: tuple[Tensor, Tensor, Tensor, dist.Work] | None = None

    def forward_send(self, x: Tensor) -> None:
        send_index = self.route.send_index
        if send_index is None or not _distributed():
            self.forward_task = x
            return
        send = x.index_select(0, send_index.to(device=x.device).long()).contiguous()
        recv = x.new_empty((self.route.recv_len, *x.shape[1:]))
        work = dist.all_to_all_single(
            recv,
            send,
            self.route.recv_sizes,
            self.route.send_sizes,
            group=self.group,
            async_op=True,
        )
        self.forward_task = (x, recv, send, work)

    def forward_recv(self) -> Tensor:
        task = self.forward_task
        assert task is not None
        self.forward_task = None
        if isinstance(task, Tensor):
            return task
        x, recv, _send, work = task
        work.wait()
        rows = self.route.recv_index
        if rows is None:
            return torch.cat((x, recv), dim=0)
        rows = rows.to(device=recv.device).long()
        size = int(self.route.output_len) if self.route.output_len is not None else max(
            int(x.shape[0]),
            int(rows.max().item()) + 1 if int(rows.numel()) else 0,
        )
        out = x.new_zeros((size, *x.shape[1:]))
        if int(x.shape[0]):
            out[: int(x.shape[0])] = x
        if int(rows.numel()):
            out.index_copy_(0, rows, recv)
        return out

    def backward_send(self, grad_output: Tensor) -> None:
        send_index = self.route.send_index
        if send_index is None or not _distributed():
            self.backward_task = (
                grad_output,
                grad_output.new_empty((0, *grad_output.shape[1:])),
                grad_output.new_empty((0, *grad_output.shape[1:])),
                _CompletedWork(),
            )
            return
        grad_x = grad_output.new_zeros(self.x_shape)
        local_rows = min(int(self.x_shape[0]), int(grad_output.shape[0]))
        if local_rows:
            grad_x[:local_rows].copy_(grad_output[:local_rows])
        recv_index = self.route.recv_index
        remote_grad = (
            grad_output[int(self.x_shape[0]) :].contiguous()
            if recv_index is None
            else grad_output.index_select(0, recv_index.to(device=grad_output.device).long()).contiguous()
        )
        returned = grad_output.new_empty((self.route.send_len, *grad_output.shape[1:]))
        work = dist.all_to_all_single(
            returned,
            remote_grad,
            self.route.send_sizes,
            self.route.recv_sizes,
            group=self.group,
            async_op=True,
        )
        self.backward_task = (grad_x, returned, remote_grad, work)

    def backward_recv(self) -> Tensor:
        task = self.backward_task
        assert task is not None
        self.backward_task = None
        grad_x, returned, _remote_grad, work = task
        work.wait()
        send_index = self.route.send_index
        if send_index is not None and int(send_index.numel()) and int(returned.numel()):
            grad_x.index_add_(0, send_index.to(device=grad_x.device).long(), returned)
        return grad_x


class _CompletedWork:
    def wait(self) -> None:
        return None


class _PullSend(Function):
    @staticmethod
    def forward(ctx, x: Tensor, route: Any, group: object):
        pull = _PullContext(route, tuple(x.shape), group)
        pull.forward_send(x)
        key = torch.empty(0, dtype=torch.float32, device=x.device)
        key._starrygl_pull_ctx = pull
        ctx.pull = pull
        return key

    @staticmethod
    def backward(ctx, _grad_key: Tensor):
        return ctx.pull.backward_recv(), None, None


class _PullRecv(Function):
    @staticmethod
    def forward(ctx, key: Tensor):
        pull = key._starrygl_pull_ctx
        ctx.pull = pull
        return pull.forward_recv()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        ctx.pull.backward_send(grad_output)
        return torch.empty(0, dtype=torch.float32, device=grad_output.device)


@dataclass
class AutogradRoutePullHandle:
    key: Tensor

    def wait(self) -> Tensor:
        return _PullRecv.apply(self.key)

    result = wait

    def ready(self) -> bool:
        return False

    def wait_on_stream(self, stream: torch.cuda.Stream | None = None) -> Tensor:
        del stream
        return self.wait()

    async def async_wait(self, stream: torch.cuda.Stream | None = None) -> Tensor:
        del stream
        await asyncio.sleep(0.0)
        return self.wait()

    def __await__(self):
        return self.async_wait().__await__()


def autograd_pull_async(route: Any, x: Tensor, *, group: dist.ProcessGroup | None = None) -> AutogradRoutePullHandle:
    return AutogradRoutePullHandle(_PullSend.apply(x, route, group))


def autograd_pull(route: Any, x: Tensor, *, group: dist.ProcessGroup | None = None) -> Tensor:
    if route.send_index is None or not _distributed():
        return x
    return autograd_pull_async(route, x, group=group).wait()


class _Push(Function):
    @staticmethod
    def forward(ctx, x: Tensor, send_index: Tensor, send_sizes: Tensor, recv_sizes: Tensor, group: object):
        send_index = send_index.long().to(device=x.device)
        send = x.index_select(0, send_index).contiguous()
        send_sizes_tuple = tuple(int(v) for v in send_sizes.cpu().tolist())
        recv_sizes_tuple = tuple(int(v) for v in recv_sizes.cpu().tolist())
        recv = x.new_empty((sum(recv_sizes_tuple), *x.shape[1:]))
        dist.all_to_all_single(recv, send, recv_sizes_tuple, send_sizes_tuple, group=group)
        ctx.group = group
        ctx.x_shape = tuple(x.shape)
        ctx.save_for_backward(send_index, send_sizes.cpu(), recv_sizes.cpu())
        return recv

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        send_index, send_sizes, recv_sizes = ctx.saved_tensors
        send_sizes_tuple = tuple(int(v) for v in send_sizes.tolist())
        recv_sizes_tuple = tuple(int(v) for v in recv_sizes.tolist())
        returned = grad_output.new_empty((sum(send_sizes_tuple), *grad_output.shape[1:]))
        dist.all_to_all_single(returned, grad_output.contiguous(), send_sizes_tuple, recv_sizes_tuple, group=ctx.group)
        grad_x = grad_output.new_zeros(ctx.x_shape)
        if int(send_index.numel()):
            grad_x.index_add_(0, send_index.to(device=grad_x.device).long(), returned)
        return grad_x, None, None, None, None


def autograd_push(route: Any, x: Tensor, *, group: dist.ProcessGroup | None = None) -> Tensor:
    if route.send_index is None or not _distributed():
        return x.contiguous()
    send_sizes = torch.tensor(route.send_sizes, dtype=torch.long)
    recv_sizes = torch.tensor(route.recv_sizes, dtype=torch.long)
    return _Push.apply(x, route.send_index, send_sizes, recv_sizes, group)


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and int(dist.get_world_size()) > 1


__all__ = ["AutogradRoutePullHandle", "autograd_pull", "autograd_pull_async", "autograd_push"]
