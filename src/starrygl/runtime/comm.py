from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
import torch
import torch.distributed as dist
from torch import Tensor

from .autograd_comm import AutogradRoutePullHandle, autograd_pull_async, autograd_push


@dataclass(frozen=True)
class Route:
    send_sizes: tuple[int, ...]
    recv_sizes: tuple[int, ...]
    send_index: Tensor | None = None
    recv_index: Tensor | None = None
    output_len: int | None = None

    @property
    def send_len(self) -> int:
        return int(sum(self.send_sizes))

    @property
    def recv_len(self) -> int:
        return int(sum(self.recv_sizes))

    @property
    def world_size(self) -> int:
        return len(self.send_sizes)

    def to(self, *args, **kwargs) -> Route:
        return Route(
            send_sizes=self.send_sizes,
            recv_sizes=self.recv_sizes,
            send_index=None if self.send_index is None else self.send_index.to(*args, **kwargs),
            recv_index=None if self.recv_index is None else self.recv_index.to(*args, **kwargs),
            output_len=self.output_len,
        )

@dataclass
class RouteHandle:
    route: Route
    local: Tensor
    recv_buffer: Tensor
    work: dist.Work | None
    op: str
    name: str
    event: torch.cuda.Event | None = None
    send_buffer: Tensor | None = None

    def wait(self) -> Tensor:
        if self.work is not None:
            self.work.wait()
        return self._result()

    def ready(self) -> bool:
        if self.event is not None:
            return bool(self.event.query())
        if self.work is None:
            return True
        completed = getattr(self.work, "is_completed", None)
        return bool(completed()) if callable(completed) else False

    def wait_on_stream(self, stream: torch.cuda.Stream | None = None) -> Tensor:
        if self.event is not None and stream is not None:
            stream.wait_event(self.event)
        if self.work is not None:
            self.work.wait()
        return self._result()

    async def async_wait(self, stream: torch.cuda.Stream | None = None) -> Tensor:
        await asyncio.sleep(0.0)
        return self.wait_on_stream(stream)

    def _result(self) -> Tensor:
        if self.op == "pull":
            if self.route.recv_index is None:
                return torch.cat((self.local, self.recv_buffer), dim=0)
            rows = self.route.recv_index.to(device=self.recv_buffer.device).long()
            size = int(self.route.output_len) if self.route.output_len is not None else max(
                int(self.local.shape[0]),
                int(rows.max().item()) + 1 if int(rows.numel()) else 0,
            )
            out = self.local.new_zeros((size, *self.local.shape[1:]))
            if int(self.local.shape[0]) > 0:
                out[: int(self.local.shape[0])] = self.local
            if int(rows.numel()) > 0:
                out.index_copy_(0, rows, self.recv_buffer)
            return out
        if self.op == "push":
            return self.recv_buffer
        raise RuntimeError(f"unknown route handle op: {self.op!r}")


@dataclass
class AllGatherHandle:
    recv_counts: Tensor
    recv_buffer: Tensor | list[Tensor]
    work: dist.Work | None
    name: str
    max_len: int
    output_device: torch.device
    send_buffer: Tensor | None = None

    def wait(self) -> Tensor:
        if self.work is not None:
            self.work.wait()
        return self._result()

    def ready(self) -> bool:
        if self.work is None:
            return True
        completed = getattr(self.work, "is_completed", None)
        return bool(completed()) if callable(completed) else False

    def _result(self) -> Tensor:
        recv_buffer = (
            torch.cat(self.recv_buffer, dim=0)
            if isinstance(self.recv_buffer, list)
            else self.recv_buffer
        )
        if int(self.recv_counts.numel()) == 1:
            count = int(self.recv_counts[0].item())
            return recv_buffer[:count].to(device=self.output_device)
        parts = []
        offset = 0
        for count in self.recv_counts.tolist():
            count = int(count)
            if count:
                parts.append(recv_buffer[offset : offset + count])
            offset += int(self.max_len)
        if not parts:
            shape = (0, *recv_buffer.shape[1:])
            return recv_buffer.new_empty(shape).to(device=self.output_device)
        return torch.cat(parts, dim=0).to(device=self.output_device)


class CommScheduler:
    """Shared process-group and stream context for route communication.

    The runtime call sites own execution order.  This object only launches and
    finishes communication; it is not a stage dispatcher or a slot plan.
    """

    def __init__(self, *, group: dist.ProcessGroup | None = None) -> None:
        self.group = group
        self._streams: dict[str, torch.cuda.Stream] = {}

    @property
    def world_size(self) -> int:
        if distributed():
            return int(dist.get_world_size(group=self.group))
        return 1

    def stream(self, device: torch.device | str) -> torch.cuda.Stream | None:
        target = torch.device(device)
        if target.type != "cuda" or not torch.cuda.is_available():
            return None
        key = str(target)
        if key not in self._streams:
            self._streams[key] = torch.cuda.Stream(device=target)
        return self._streams[key]

    def launch_pull(
        self,
        route: Route,
        x: Tensor,
        *,
        name: str = "pull",
        stream: torch.cuda.Stream | None = None,
    ) -> RouteHandle:
        if route.send_index is None:
            return RouteHandle(route=route, local=x, recv_buffer=x.new_empty((0, *x.shape[1:])), work=None, op="pull", name=name)
        if stream is not None and x.device.type == "cuda":
            with torch.cuda.stream(stream):
                send = x.index_select(0, route.send_index.to(device=x.device).long()).contiguous()
                recv = x.new_empty((route.recv_len, *x.shape[1:]))
                work = self._all_to_all(recv, send, route.recv_sizes, route.send_sizes)
                event = torch.cuda.Event()
                event.record(stream)
            return RouteHandle(route=route, local=x, recv_buffer=recv, work=work, op="pull", name=name, event=event, send_buffer=send)
        send = x.index_select(0, route.send_index.to(device=x.device).long()).contiguous()
        recv = x.new_empty((route.recv_len, *x.shape[1:]))
        if self._needs_cuda_collective(send):
            recv.copy_(self._all_to_all_sync_cuda(send, route.recv_sizes, route.send_sizes))
            return RouteHandle(route=route, local=x, recv_buffer=recv, work=None, op="pull", name=name)
        work = self._all_to_all(recv, send, route.recv_sizes, route.send_sizes)
        return RouteHandle(route=route, local=x, recv_buffer=recv, work=work, op="pull", name=name, send_buffer=send)

    def finish_pull(self, handle: RouteHandle) -> Tensor:
        if handle.op != "pull":
            raise RuntimeError("finish_pull received a non-pull handle")
        return handle.wait()

    def launch_autograd_pull(
        self,
        route: Route,
        x: Tensor,
        *,
        name: str = "autograd_pull",
    ) -> AutogradRoutePullHandle:
        return autograd_pull_async(route, x, group=self.group)

    def launch_push(
        self,
        route: Route,
        x: Tensor,
        *,
        name: str = "push",
    ) -> RouteHandle:
        if route.send_index is None:
            send = x.contiguous()
        else:
            send = x.index_select(0, route.send_index.to(device=x.device).long()).contiguous()
        recv = x.new_empty((route.recv_len, *x.shape[1:]))
        if self._needs_cuda_collective(send):
            return self._launch_cpu_nccl_push(route, send, x, name=name)
        work = self._all_to_all(recv, send, route.recv_sizes, route.send_sizes)
        return RouteHandle(route=route, local=x, recv_buffer=recv, work=work, op="push", name=name, send_buffer=send)

    def finish_push(self, handle: RouteHandle) -> Tensor:
        if handle.op != "push":
            raise RuntimeError("finish_push received a non-push handle")
        return handle.wait()

    def autograd_push(
        self,
        route: Route,
        x: Tensor,
        *,
        name: str = "autograd_push",
    ) -> Tensor:
        return autograd_push(route, x, group=self.group)

    def all_reduce(
        self,
        x: Tensor,
        *,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
        name: str = "all_reduce",
    ) -> Tensor:
        """Run one in-place reduction on the bound process group."""

        if distributed():
            dist.all_reduce(x, op=op, group=self.group)
        return x

    def launch_all_gather(
        self,
        x: Tensor,
        *,
        name: str = "all_gather",
    ) -> AllGatherHandle | RouteHandle:
        """Launch a variable-length all-gather without duplicating payloads."""

        if not distributed():
            return self._local_all_gather(x, name)
        send_len = int(x.shape[0])
        recv_counts = _all_gather_counts(send_len, group=self.group)
        return self._launch_all_gather_with_counts(x, recv_counts, name=name)

    def launch_all_gather_with_counts(
        self,
        x: Tensor,
        recv_counts: Tensor,
        *,
        name: str = "all_gather",
    ) -> AllGatherHandle | RouteHandle:
        if not distributed():
            return self._local_all_gather(x, name)
        return self._launch_all_gather_with_counts(x, recv_counts, name=name)

    @staticmethod
    def _local_all_gather(x: Tensor, name: str) -> AllGatherHandle:
        return AllGatherHandle(
            recv_counts=torch.tensor([int(x.shape[0])], dtype=torch.long),
            recv_buffer=x.contiguous(), work=None, name=name,
            max_len=int(x.shape[0]), output_device=x.device,
        )

    def _launch_all_gather_with_counts(
        self,
        x: Tensor,
        recv_counts: Tensor,
        *,
        name: str,
    ) -> AllGatherHandle:
        send_len = int(x.shape[0])
        max_len = int(recv_counts.max().item()) if int(recv_counts.numel()) else send_len
        backend = str(dist.get_backend(group=self.group)).lower()
        output_device = x.device
        send_src = x.contiguous()
        if backend == "nccl" and not send_src.is_cuda:
            send_src = send_src.to(device=local_cuda_device(), non_blocking=True)
        if max_len == 0:
            recv = send_src.new_empty((0, *send_src.shape[1:]))
            return AllGatherHandle(
                recv_counts=recv_counts,
                recv_buffer=recv,
                work=None,
                name=name,
                max_len=0,
                output_device=output_device,
            )
        if send_len < max_len:
            padded = send_src.new_empty((max_len, *send_src.shape[1:]))
            if send_len:
                padded[:send_len].copy_(send_src)
            if max_len > send_len:
                padded[send_len:].zero_()
            send = padded.contiguous()
        else:
            send = send_src
        if backend == "gloo":
            recv = [torch.empty_like(send) for _ in range(self.world_size)]
            work = dist.all_gather(recv, send, group=self.group, async_op=True)
        else:
            recv = send.new_empty((self.world_size * max_len, *send.shape[1:]))
            work = dist.all_gather_into_tensor(recv, send, group=self.group, async_op=True)
        return AllGatherHandle(
            recv_counts=recv_counts,
            recv_buffer=recv,
            work=work,
            name=name,
            max_len=max_len,
            output_device=output_device,
            send_buffer=send,
        )

    def finish_all_gather(self, handle: AllGatherHandle | RouteHandle) -> Tensor:
        if isinstance(handle, AllGatherHandle):
            return handle.wait()
        return self.finish_push(handle)

    def _all_to_all(
        self,
        recv: Tensor,
        send: Tensor,
        recv_sizes: tuple[int, ...],
        send_sizes: tuple[int, ...],
    ) -> dist.Work | None:
        if not distributed():
            if int(send.numel()) and recv.shape == send.shape:
                recv.copy_(send)
            return None
        return dist.all_to_all_single(
            recv,
            send,
            output_split_sizes=[int(v) for v in recv_sizes],
            input_split_sizes=[int(v) for v in send_sizes],
            group=self.group,
            async_op=True,
        )

    def _needs_cuda_collective(self, tensor: Tensor) -> bool:
        return distributed() and tensor.device.type == "cpu" and torch.cuda.is_available() and str(dist.get_backend(group=self.group)).lower() == "nccl"

    def _all_to_all_sync_cuda(
        self,
        send: Tensor,
        recv_sizes: tuple[int, ...],
        send_sizes: tuple[int, ...],
    ) -> Tensor:
        device = local_cuda_device()
        send_cuda = send.to(device=device, non_blocking=True).contiguous()
        recv_cuda = torch.empty((int(sum(recv_sizes)), *send.shape[1:]), dtype=send.dtype, device=device)
        work = dist.all_to_all_single(
            recv_cuda,
            send_cuda,
            output_split_sizes=[int(v) for v in recv_sizes],
            input_split_sizes=[int(v) for v in send_sizes],
            group=self.group,
            async_op=True,
        )
        work.wait()
        return recv_cuda.to(device=send.device)

    def _launch_cpu_nccl_push(self, route: Route, send: Tensor, local: Tensor, *, name: str) -> RouteHandle:
        device = local_cuda_device()
        send_cuda = send.to(device=device, non_blocking=True).contiguous()
        recv_cuda = torch.empty((route.recv_len, *send.shape[1:]), dtype=send.dtype, device=device)
        work = dist.all_to_all_single(
            recv_cuda,
            send_cuda,
            output_split_sizes=[int(v) for v in route.recv_sizes],
            input_split_sizes=[int(v) for v in route.send_sizes],
            group=self.group,
            async_op=True,
        )
        return RouteHandle(route=route, local=local, recv_buffer=recv_cuda, work=work, op="push", name=name, send_buffer=send_cuda)


def distributed(*, group: dist.ProcessGroup | None = None) -> bool:
    return dist.is_available() and dist.is_initialized() and int(dist.get_world_size(group=group)) > 1


def all_to_all_counts(send_counts: Tensor, *, group: dist.ProcessGroup | None = None) -> Tensor:
    if not distributed(group=group):
        return send_counts.clone()
    if torch.cuda.is_available() and str(dist.get_backend(group=group)).lower() == "nccl":
        device = local_cuda_device()
        send = send_counts.to(device=device)
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=group)
        return recv.cpu()
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts, group=group)
    return recv_counts


def _all_gather_counts(send_len: int, *, group: dist.ProcessGroup | None = None) -> Tensor:
    if not distributed(group=group):
        return torch.tensor([int(send_len)], dtype=torch.long)
    world_size = int(dist.get_world_size(group=group))
    if torch.cuda.is_available() and str(dist.get_backend(group=group)).lower() == "nccl":
        device = local_cuda_device()
        send = torch.tensor([int(send_len)], dtype=torch.long, device=device)
        recv = torch.empty((world_size,), dtype=torch.long, device=device)
        dist.all_gather_into_tensor(recv, send, group=group)
        return recv.cpu()
    send = torch.tensor([int(send_len)], dtype=torch.long)
    recv_list = [torch.empty_like(send) for _ in range(world_size)]
    dist.all_gather(recv_list, send, group=group)
    return torch.cat(recv_list, dim=0)


def collective_needed(local_needed: bool, *, group: dist.ProcessGroup | None = None) -> bool:
    if not distributed(group=group):
        return bool(local_needed)
    device = local_cuda_device() if str(dist.get_backend(group=group)).lower() == "nccl" else torch.device("cpu")
    flag = torch.tensor([int(bool(local_needed))], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
    return bool(flag.item())


def reduce_int(
    value: int,
    *,
    reduction: str = "max",
    device: torch.device | None = None,
    group: dist.ProcessGroup | None = None,
) -> int:
    value = max(0, int(value))
    if not distributed(group=group):
        return value
    target = device
    if target is None or (str(dist.get_backend(group=group)).lower() == "nccl" and target.type != "cuda"):
        target = local_cuda_device() if str(dist.get_backend(group=group)).lower() == "nccl" else torch.device("cpu")
    op = dist.ReduceOp.MIN if str(reduction).lower() == "min" else dist.ReduceOp.MAX
    tensor = torch.tensor([value], dtype=torch.long, device=target)
    dist.all_reduce(tensor, op=op, group=group)
    return int(tensor.item())


def local_cuda_device() -> torch.device:
    index = int(os.environ.get("LOCAL_RANK", torch.cuda.current_device()))
    torch.cuda.set_device(index)
    return torch.device("cuda", index)


__all__ = [
    "AutogradRoutePullHandle",
    "CommScheduler",
    "Route",
    "RouteHandle",
    "all_to_all_counts",
    "collective_needed",
    "distributed",
    "local_cuda_device",
    "reduce_int",
]
