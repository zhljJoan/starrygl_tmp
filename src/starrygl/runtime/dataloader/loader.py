from __future__ import annotations

from contextlib import nullcontext
from itertools import islice
from queue import Empty, Full, Queue
from threading import Event, Semaphore, Thread
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from starrygl.batch import Batch, BatchMode, SamplingPolicy, WindowPolicy
from starrygl.runtime.comm import CommScheduler
from starrygl.store import StoreBundle
from starrygl.store.graph import split_window_range

from .materialize import AccessedWindow, materialize_accessed_window
from .pipeline import finish_batch, launch_batch, record_batch_stream


WindowInput = tuple[int, range, tuple[int, ...]]
ReadyBatch = tuple[Batch, torch.cuda.Event | None]
GraphAccessor = Callable[..., AccessedWindow]


class DataLoader:
    """Bound three-stage loader shared by Event and Snapshot execution.

    Static graph access, native sampling and communication resources are bound
    at construction. Iteration only advances prepared window rows through the
    depth-one access and ready queues.
    """

    def __init__(
        self,
        store: StoreBundle,
        *,
        mode: BatchMode,
        split: str,
        window_policy: WindowPolicy,
        sampling_policy: SamplingPolicy,
        chunk_decay: Sequence[int] | Tensor | None,
        num_full_snapshots: int,
        num_layers: int,
        fanouts: Sequence[int] | Tensor | None,
        sampler_options: Mapping[str, Any],
        num_negatives: int,
        generator: object | None,
        comm: CommScheduler,
        device: str | torch.device | None,
        prefetch_state: Callable | None,
        enabled: bool = True,
        skip: int = 0,
        maximum: int = 0,
    ) -> None:
        self.store = store
        self.mode = mode
        self.split = str(split)
        self.window_policy = window_policy
        self.sampling_policy = sampling_policy
        self.chunk_decay = (
            _normalize_chunk_decay(chunk_decay)
            if window_policy == "chunk_decay"
            else ()
        )
        self.num_full_snapshots = int(num_full_snapshots)
        self.num_layers = int(num_layers)
        self.options = dict(sampler_options)
        if self.options.get("snapshot_materialize_on_device") is True and (
            mode != "snapshot" or sampling_policy != "full"
            or window_policy not in {"full_snapshot", "chunk_decay"}
            or store.labels.task_kind != "node"
        ):
            raise ValueError("snapshot_materialize_on_device requires full/chunk snapshot node prediction")
        self.num_negatives = int(num_negatives)
        self.generator = generator
        self.comm = comm
        self.device = None if device is None else torch.device(device)
        self.prefetch_state = prefetch_state
        self.enabled = bool(enabled)
        self.skip = max(0, int(skip))
        self.maximum = max(0, int(maximum))
        self._entry_cache: dict = {}
        self._blob_cache: dict = {}
        self.options["_reuse_static_snapshot_graph"] = bool(
            self.options.get("_reuse_static_snapshot_graph", False)
            and self.options.get("rolling_snapshot_cache", True)
            and mode == "snapshot" and sampling_policy == "full"
            and window_policy in {"full_snapshot", "chunk_decay"}
            and (self.options.get("full_snapshot_chunk_limit") is None
                 or int(self.options["full_snapshot_chunk_limit"]) < 0)
        )
        self._graph_cache = {} if self.options["_reuse_static_snapshot_graph"] else None
        self.window_ids, self.access = _bind_source(
            store,
            mode=mode,
            split=self.split,
            sampling_policy=sampling_policy,
            fanouts=fanouts,
            num_layers=self.num_layers,
            options=self.options,
            num_negatives=self.num_negatives,
            generator=generator,
            entry_cache=self._entry_cache,
            blob_cache=self._blob_cache,
        )
        self.prefetch_stream = None
        if self.device is not None and self.device.type == "cuda" and torch.cuda.is_available():
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            streams = store.graph.runtime_cache.setdefault("loader_streams", {})
            if index not in streams:
                streams[index] = torch.cuda.Stream(device=index)
            self.prefetch_stream = streams[index]
        self.pin_memory = bool(self.prefetch_stream) and bool(
            self.options.get("pin_memory", True)
        )
        if self.prefetch_stream is not None and self.options.get("snapshot_materialize_on_device") is True:
            self.prefetch_stream.wait_stream(torch.cuda.current_stream(device=self.device))

    def __len__(self) -> int:
        count = max(0, len(self.window_ids) - self.skip)
        return min(count, self.maximum) if self.maximum else count

    def __iter__(self):
        self._entry_cache.clear()
        self._blob_cache.clear()
        if self._graph_cache is not None:
            self._graph_cache.clear()
        accessed = iter(self._accessed())
        if not self.enabled:
            for item in accessed:
                yield self._wait_ready(self._finish(self._launch(item)))
            return

        request_queue: Queue = Queue(maxsize=1)
        ready_queue: Queue = Queue(maxsize=1)
        request_slot = Semaphore(1)
        ready_slot = Semaphore(1)
        stop = Event()
        sentinel = object()

        def source_worker() -> None:
            try:
                while _acquire(request_slot, stop):
                    try:
                        item = next(accessed)
                    except StopIteration:
                        request_slot.release()
                        _put(request_queue, sentinel, stop)
                        return
                    if not _put(request_queue, item, stop):
                        request_slot.release()
                        return
            except BaseException as exc:
                _put(request_queue, exc, stop)

        def stage_worker() -> None:
            try:
                while _acquire(ready_slot, stop):
                    item = _get(request_queue, stop)
                    if item is sentinel:
                        ready_slot.release()
                        _put(ready_queue, sentinel, stop)
                        return
                    if isinstance(item, BaseException):
                        raise item
                    request_slot.release()
                    launched = self._launch(item)
                    if not _put(ready_queue, self._finish(launched), stop):
                        ready_slot.release()
                        return
            except BaseException as exc:
                _put(ready_queue, exc, stop)

        workers = (
            Thread(target=source_worker, name="starrygl-data", daemon=True),
            Thread(target=stage_worker, name="starrygl-prefetch", daemon=True),
        )
        for worker in workers:
            worker.start()

        try:
            while True:
                item = _get(ready_queue, stop)
                if item is sentinel:
                    return
                if isinstance(item, BaseException):
                    raise item
                ready_slot.release()
                yield self._wait_ready(item)
        finally:
            stop.set()
            request_slot.release()
            ready_slot.release()
            for queue in (request_queue, ready_queue):
                try:
                    queue.put_nowait(sentinel)
                except Full:
                    pass
            for worker in workers:
                worker.join()

    def _accessed(self) -> Iterable[AccessedWindow]:
        windows: Iterable[WindowInput] = (
            (
                window_id,
                *_input_window(
                    window_id=window_id,
                    split_start=self.window_ids.start,
                    window_policy=self.window_policy,
                    chunk_decay=self.chunk_decay,
                    num_full_snapshots=self.num_full_snapshots,
                    full_snapshot_chunk_limit=self.options.get(
                        "full_snapshot_chunk_limit"
                    ),
                ),
            )
            for window_id in self.window_ids
        )
        if self.skip:
            windows = islice(windows, self.skip, None)
        if self.maximum:
            windows = islice(windows, self.maximum)
        if bool(self.options.get("profile_runtime", False)):
            print(
                f"[starrygl.profile] {self.mode}_windows "
                f"rank={int(self.store.graph.rank)} split={self.split} "
                f"windows={len(self)}",
                flush=True,
            )
        for window_id, input_window, limits in windows:
            context = (
                torch.cuda.stream(self.prefetch_stream)
                if self.prefetch_stream is not None
                and self.options.get("snapshot_materialize_on_device") is True
                else nullcontext()
            )
            with context:
                item = self.access(window_id=window_id, input_window=input_window, chunk_limits=limits)
            yield item

    def _launch(self, item: AccessedWindow):
        context = (
            torch.cuda.stream(self.prefetch_stream)
            if self.prefetch_stream is not None
            else nullcontext()
        )
        with context:
            _, _, _, feature_node_ids, edge_ids = item
            return launch_batch(
                materialize_accessed_window(
                    item,
                    mode=self.mode,
                    num_layers=self.num_layers,
                ),
                store=self.store,
                comm=self.comm,
                options=self.options,
                device=self.device,
                prefetch_state=self.prefetch_state,
                feature_node_ids=feature_node_ids,
                edge_ids=edge_ids,
                non_blocking=self.prefetch_stream is not None,
                pin_memory=self.pin_memory,
                graph_cache=self._graph_cache,
            )

    def _finish(self, launched) -> ReadyBatch:
        context = (
            torch.cuda.stream(self.prefetch_stream)
            if self.prefetch_stream is not None
            else nullcontext()
        )
        with context:
            batch, pending_nodes, pending_edges, pending_state = launched
            batch = finish_batch(
                batch,
                pending_nodes,
                pending_edges,
                pending_state,
                device=self.device,
                non_blocking=self.prefetch_stream is not None,
                pin_memory=self.pin_memory,
            )
            event = None
            if self.prefetch_stream is not None:
                event = torch.cuda.Event()
                event.record(self.prefetch_stream)
        return batch, event

    def _wait_ready(self, ready: ReadyBatch) -> Batch:
        batch, event = ready
        if event is not None:
            stream = torch.cuda.current_stream(device=self.device)
            stream.wait_event(event)
            record_batch_stream(batch, stream)
        return batch


def _bind_source(
    store: StoreBundle,
    *,
    mode: BatchMode,
    split: str,
    sampling_policy: SamplingPolicy,
    fanouts: Sequence[int] | Tensor | None,
    num_layers: int,
    options: Mapping[str, Any],
    num_negatives: int,
    generator: object | None,
    entry_cache: dict,
    blob_cache: dict,
) -> tuple[range, GraphAccessor]:
    native_sampler = None
    if sampling_policy == "neighbor":
        from ..sample import build_native_sampler

        native_sampler = build_native_sampler(
            store,
            store.graph.temporal_csr_view,
            fanouts=fanouts,
            num_layers=num_layers,
            options=options,
            snapshot=mode == "snapshot",
        )

    if mode == "event":
        from ..event.materialize import access_event_window, event_window_ids

        window_ids = event_window_ids(
            store,
            split,
            drop_last=bool(options.get("drop_last", False)),
        )
        return window_ids, lambda **window: access_event_window(
            store,
            store.graph.event_view,
            split=split,
            sampling_policy=sampling_policy,
            native_sampler=native_sampler,
            sampler_options=options,
            num_negatives=num_negatives,
            generator=generator,
            **window,
        )

    from ..snapshot.materialize import access_snapshot_window

    window_ids = split_window_range(store.graph.split_time_ptr_2, split)
    rolling = sampling_policy != "neighbor" and bool(
        options.get("rolling_snapshot_cache", True)
    )
    return window_ids, lambda **window: access_snapshot_window(
        store,
        store.graph.snapshot_csc_view.get("slices", ()),
        split=split,
        sampling_policy=sampling_policy,
        native_sampler=native_sampler,
        sampler_options=options,
        num_negatives=num_negatives,
        generator=generator,
        entry_cache=entry_cache if rolling else None,
        blob_cache=blob_cache if rolling else None,
        **window,
    )


def _acquire(slot: Semaphore, stop: Event) -> bool:
    while not stop.is_set():
        if slot.acquire(timeout=0.05):
            return True
    return False


def _put(queue: Queue, item: Any, stop: Event) -> bool:
    while not stop.is_set():
        try:
            queue.put(item, timeout=0.05)
            return True
        except Full:
            pass
    return False


def _get(queue: Queue, stop: Event) -> Any:
    while not stop.is_set():
        try:
            return queue.get(timeout=0.05)
        except Empty:
            pass
    raise RuntimeError("DataLoader stopped before the pipeline drained")


def _normalize_chunk_decay(
    chunk_decay: Sequence[int] | Tensor | None,
) -> tuple[int, ...]:
    if chunk_decay is None:
        return ()
    values = torch.as_tensor(chunk_decay, dtype=torch.long).flatten()
    return tuple(int(value) for value in values[values >= 0].tolist())


def _input_window(
    *,
    window_id: int,
    split_start: int,
    window_policy: WindowPolicy,
    chunk_decay: Sequence[int],
    num_full_snapshots: int,
    full_snapshot_chunk_limit: Any = None,
) -> tuple[range, tuple[int, ...]]:
    if window_policy == "event_window":
        return range(int(window_id), int(window_id) + 1), (-1,)
    full_begin = max(
        int(split_start),
        int(window_id) - max(1, int(num_full_snapshots)) + 1,
    )
    begin = max(int(split_start), full_begin - len(chunk_decay))
    decay_count = full_begin - begin
    decay_limits = (
        tuple(reversed(chunk_decay))[-decay_count:] if decay_count else ()
    )
    full_limit = (
        -1
        if full_snapshot_chunk_limit is None
        else max(-1, int(full_snapshot_chunk_limit))
    )
    full_limits = (full_limit,) * (int(window_id) - full_begin + 1)
    return range(begin, int(window_id) + 1), decay_limits + full_limits


def with_materialize_device(
    options: Mapping[str, Any] | None,
    device: str | torch.device | None,
) -> Mapping[str, Any] | None:
    if device is None:
        return options
    out = dict(options or {})
    if out.get("snapshot_materialize_on_device") is True:
        out.setdefault("_materialize_device", torch.device(device))
    return out


__all__ = ["BatchMode", "DataLoader", "with_materialize_device"]
