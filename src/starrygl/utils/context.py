from __future__ import annotations

import logging
import os
import socket
from contextlib import contextmanager
from math import floor
from typing import Any

import torch
import torch.distributed as dist


class DistributedContext:
    """Distributed identity, process groups, topology, and device context."""

    @classmethod
    def init(
        cls,
        backend: str | None = None,
        use_gpu: bool | None = None,
        memory_group_num: int = 1,
        device: str | torch.device | None = None,
    ) -> "DistributedContext":
        if cls.is_initialized():
            return cls.get_default_context()
        has_environment = _rank("RANK", "OMPI_COMM_WORLD_RANK") is not None
        if has_environment and not dist.is_available():
            raise RuntimeError("torch.distributed is unavailable")
        if dist.is_available() and not dist.is_initialized() and has_environment:
            rank = _required_rank("RANK", "OMPI_COMM_WORLD_RANK")
            world_size = _required_rank("WORLD_SIZE", "OMPI_COMM_WORLD_SIZE")
            if "RANK" in os.environ:
                dist.init_process_group(backend=backend or _default_backend())
            else:
                dist.init_process_group(
                    backend=backend or _default_backend(),
                    init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
                    rank=rank,
                    world_size=world_size,
                )
        initialized = dist.is_available() and dist.is_initialized()
        rank = int(dist.get_rank()) if initialized else 0
        world_size = int(dist.get_world_size()) if initialized else 1
        local_rank = int(os.getenv("LOCAL_RANK") or os.getenv("OMPI_COMM_WORLD_LOCAL_RANK") or rank)
        local_size = os.getenv("LOCAL_SIZE") or os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE")
        if initialized and "LOCAL_RANK" not in os.environ and "OMPI_COMM_WORLD_LOCAL_RANK" not in os.environ:
            logging.warning("LOCAL_RANK is not set; using global rank")
        context = cls(
            backend=backend or (dist.get_backend() if initialized else _default_backend()),
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            local_size=None if local_size is None else int(local_size),
            memory_group_num=memory_group_num,
            device=device,
            use_gpu=use_gpu,
        )
        cls._instance_ = context
        return context

    @classmethod
    def get_default_context(cls) -> "DistributedContext":
        if not cls.is_initialized():
            raise RuntimeError("call DistributedContext.init first")
        return cls._instance_

    @classmethod
    def is_initialized(cls) -> bool:
        return getattr(cls, "_instance_", None) is not None

    def __init__(
        self,
        *,
        backend: str,
        rank: int,
        world_size: int,
        local_rank: int,
        local_size: int | None = None,
        memory_group_num: int = 1,
        device: str | torch.device | None = None,
        use_gpu: bool | None = None,
    ) -> None:
        gpu = (
            bool(use_gpu)
            if use_gpu is not None
            else torch.device(device).type == "cuda"
            if device is not None
            else backend.lower() in {"nccl", "mpi"} and torch.cuda.is_available()
        )
        self._device = torch.device(device) if device is not None else torch.device(f"cuda:{local_rank}" if gpu else "cpu")
        if self._device.type == "cuda":
            self._device = torch.device("cuda", local_rank) if self._device.index is None else self._device
            torch.cuda.set_device(self._device)
        self._rank = int(rank)
        self._world_size = int(world_size)
        self._local_rank = int(local_rank)
        self._hostname = socket.gethostname()
        initialized = dist.is_available() and dist.is_initialized()
        self._group = dist.GroupMember.WORLD if initialized else None
        self._cpu_group = (
            self._group
            if initialized and dist.get_backend() == "gloo"
            else dist.new_group(backend="gloo")
            if initialized
            else None
        )

        if initialized:
            local_rank_max = torch.tensor([local_rank], device=self._device)
            dist.all_reduce(local_rank_max, op=dist.ReduceOp.MAX)
            self._local_size = max(int(local_rank_max) + 1, int(local_size or 0))
            hosts: list[Any] = [None] * world_size
            dist.all_gather_object(hosts, (self.hostname, self.local_rank), group=self._cpu_group)
            self._rank_to_host = tuple(hosts)
        else:
            self._local_size = int(local_size or 1)
            self._rank_to_host = ((self.hostname, self.local_rank),)
        host_names = sorted({host for host, _ in self._rank_to_host})
        self._host_index = {host: index for index, host in enumerate(host_names)}

        if memory_group_num < 1 or world_size % memory_group_num:
            raise ValueError("memory_group_num must divide world_size")
        self.memory_group_num = int(memory_group_num)
        self.memory_group_size = floor(world_size / memory_group_num)
        self.memory_group = rank // self.memory_group_size
        self.memory_group_rank = rank % self.memory_group_size
        start = self.memory_group * self.memory_group_size
        ranks = list(range(start, start + self.memory_group_size))
        self.memory_gloo_group = dist.new_group(ranks=ranks, backend="gloo") if initialized else None
        self.memory_nccl_group = (
            dist.new_group(ranks=ranks, backend="nccl")
            if initialized and self.device.type == "cuda"
            else None
        )

    def shutdown(self) -> None:
        groups = (self.memory_nccl_group, self.memory_gloo_group, self._cpu_group)
        seen: set[int] = set()
        for group in groups:
            if group is not None and group is not self._group and id(group) not in seen:
                dist.destroy_process_group(group)
                seen.add(id(group))
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group(self._group)
        type(self)._instance_ = None

    def barrier(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def size(self) -> int:
        return self._world_size

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def local_rank(self) -> int:
        return self._local_rank

    @property
    def local_size(self) -> int:
        if self.world_size % self._local_size:
            raise RuntimeError("world_size must be divisible by local_size")
        return self._local_size

    @property
    def cross_rank(self) -> int:
        return self.world_size % self.local_size

    @property
    def cross_size(self) -> int:
        return self.world_size // self.local_size

    @property
    def device(self) -> torch.device:
        return self._device

    def get_device(self) -> torch.device:
        return self.device

    @property
    def hostname(self) -> str:
        return self._hostname

    @property
    def rank_to_host(self):
        return self._rank_to_host

    @property
    def host_index(self):
        return self._host_index

    @property
    def group(self) -> dist.ProcessGroup | None:
        return self._group

    @property
    def cpu_group(self) -> dist.ProcessGroup | None:
        return self._cpu_group

    @property
    def gloo_group(self) -> dist.ProcessGroup | None:
        return self._cpu_group

    def get_ranks_by_host(self, hostname: str | None = None) -> tuple[int, ...]:
        name = hostname or self.hostname
        return tuple(rank for rank, (host, _) in enumerate(self.rank_to_host) if host == name)

    def get_ranks_by_local(self, local_rank: int | None = None) -> tuple[int, ...]:
        local = self.local_rank if local_rank is None else int(local_rank)
        ranks = [
            (rank, host)
            for rank, (host, rank_local) in enumerate(self.rank_to_host)
            if int(rank_local) == local
        ]
        return tuple(rank for rank, _ in sorted(ranks, key=lambda item: self.host_index[item[1]]))

    def get_hybrid_matrix(self) -> torch.Tensor:
        hosts = sorted(self.host_index, key=self.host_index.get)
        return torch.tensor([self.get_ranks_by_host(host) for host in hosts])

    def new_hybrid_subgroups(
        self,
        matrix: torch.Tensor | None = None,
        backend: Any = None,
    ) -> tuple[dist.ProcessGroup, dist.ProcessGroup]:
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("distributed process group is not initialized")
        matrix = self.get_hybrid_matrix() if matrix is None else matrix
        rows = [row for row in matrix.tolist() if self.rank in row]
        cols = [col for col in matrix.t().tolist() if self.rank in col]
        if len(rows) != 1 or len(cols) != 1:
            raise RuntimeError("rank must appear once in each hybrid axis")
        return (
            dist.new_group(rows[0], backend=backend, use_local_synchronization=True),
            dist.new_group(cols[0], backend=backend, use_local_synchronization=True),
        )

    @contextmanager
    def use_stream(self, stream: torch.cuda.Stream, with_event: bool = True):
        event = torch.cuda.Event() if with_event else None
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            yield event
            if event is not None:
                event.record()

    def all_reduce_mean_scalar(self, value: float, count: int = 1) -> float:
        if not (dist.is_available() and dist.is_initialized()):
            return float(value) / max(1e-10, int(count))
        values: list[Any] = [None] * self.world_size
        dist.all_gather_object(values, (value, count), group=self.cpu_group)
        return sum(item for item, _ in values) / max(1e-10, sum(size for _, size in values))


def _rank(primary: str, fallback: str) -> int | None:
    value = os.getenv(primary) or os.getenv(fallback)
    return None if value is None else int(value)


def _required_rank(primary: str, fallback: str) -> int:
    value = _rank(primary, fallback)
    if value is None:
        raise RuntimeError(f"{primary} is not set")
    return value


def _default_backend() -> str:
    return "nccl" if torch.cuda.is_available() else "gloo"


__all__ = ["DistributedContext"]
