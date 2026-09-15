"""Opt-in placement preserves the common epoch and communication path."""
import os
from contextlib import contextmanager
from datetime import timedelta
from threading import local
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import starrygl as sg
from starrygl.prepare.snapshot_csc import build_snapshot_csc_views
from starrygl.runtime.comm import CommScheduler
from starrygl.runtime.dataloader.loader import DataLoader, with_materialize_device
from starrygl.runtime.epoch import with_model_snapshot_options
from starrygl.runtime.loop import run_epoch
from starrygl.runtime.snapshot import materialize
from starrygl.store import FeatureManager, GraphStore, LabelStore, StoreBundle
from test_flare_snapshot_hotpath import _chunk_loader


def test_explicit_option_and_model_boundary():
    for option in ({}, {"snapshot_materialize_on_device": False}):
        assert "_materialize_device" not in with_materialize_device(option, "cuda")
    for cls in (sg.TGCNModel, sg.MPNNLSTMModel, sg.EvolveGCNModel):
        option = with_model_snapshot_options({"snapshot_materialize_on_device": True}, cls(1, 2, 1))
        assert not option["snapshot_reverse_direction"]
        assert with_materialize_device(option, "cuda")["_materialize_device"].type == "cuda"
    class CustomTGCN(sg.TGCNModel):
        pass
    for cls in (sg.GConvGRUModel, sg.DCRNNModel, CustomTGCN):
        with pytest.raises(ValueError, match="snapshot_materialize_on_device requires built-in"):
            with_model_snapshot_options({"snapshot_materialize_on_device": True}, cls(1, 2, 1))
    with pytest.raises(ValueError):
        with_model_snapshot_options({"snapshot_materialize_on_device": True, "snapshot_reverse_direction": True}, sg.TGCNModel(1, 2, 1))


@pytest.mark.parametrize("mode,sampling,window,task", [
    ("event", "full", "event_window", "node"),
    ("snapshot", "neighbor", "full_snapshot", "node"),
    ("snapshot", "full", "full_snapshot", "edge"),
])
def test_loader_rejects_unsupported_paths(mode, sampling, window, task):
    with pytest.raises(ValueError, match="full/chunk snapshot node prediction"):
        DataLoader(SimpleNamespace(labels=SimpleNamespace(task_kind=task)), mode=mode, split="train",
            window_policy=window, sampling_policy=sampling, chunk_decay=None, num_full_snapshots=1,
            num_layers=1, fanouts=None, sampler_options={"snapshot_materialize_on_device": True},
            num_negatives=0, generator=None, comm=CommScheduler(), device="cpu", prefetch_state=None)


@pytest.mark.parametrize("pipeline", [False, True])
def test_source_uses_existing_stream_in_both_iteration_modes(monkeypatch, pipeline):
    loader = _chunk_loader(sg.TGCNModel(1, 2, 1), pipeline=pipeline)
    loader.options["snapshot_materialize_on_device"] = True
    sentinel, current, visits = object(), local(), []
    loader.prefetch_stream = sentinel
    original = loader.access
    @contextmanager
    def stream(value):
        assert value is sentinel
        current.active = True
        try:
            yield
        finally:
            current.active = False
    def access(**kwargs):
        assert current.active
        visits.append(kwargs["window_id"])
        return original(**kwargs)
    monkeypatch.setattr(torch.cuda, "stream", stream)
    loader.access = access
    loader._launch = loader._finish = loader._wait_ready = lambda item: item
    assert len(list(loader)) == len(loader) == len(visits)


def _store(rank=0, world=1):
    n, t = 8, 3
    src = torch.arange(n).repeat_interleave(2)
    dst = (src + torch.tensor([1, 4]).repeat(n)) % n
    e = src.numel()
    ptr = torch.arange(t)[:, None] * e + torch.tensor([0, e])
    owner = torch.arange(n) // (n // world)
    views = build_snapshot_csc_views(src=src.repeat(t), dst=dst.repeat(t),
        ts=torch.arange(t).repeat_interleave(e).float(), edge_ids=torch.arange(t * e),
        edge_dist_index=torch.arange(t * e), node_master=owner,
        hot_node_ids=torch.empty(0, dtype=torch.long), node_is_hot=torch.zeros(n, dtype=torch.bool),
        node_to_chunk=torch.arange(n) // 2, time_ptr_2=ptr, num_nodes=n, world_size=world)
    owned = (owner == rank).nonzero(as_tuple=True)[0]
    index = (owner << 48) | (torch.arange(n) % (n // world))
    return StoreBundle(graph=GraphStore(num_nodes=n, rank=rank, prepare={
        "meta": {"world_size": world}, "partition": {"node_dist_index": index},
        "time_ptr_2": ptr, "snapshot_csc_views": views}),
        features=FeatureManager(node_features={"x": torch.arange(t * n).reshape(t, n, 1).float() / 10}),
        labels=LabelStore(task_kind="node", task_ptr=torch.arange(t + 1) * owned.numel(),
            task_payload={"node_ids": owned.repeat(t), "label": owned.repeat(t).float() / 8}))


class _RecordAdam(torch.optim.Adam):
    def __init__(self, parameters):
        super().__init__(parameters, lr=.001)
        self.gradients = []
    def step(self, closure=None):
        self.gradients.append(tuple(p.grad.detach().clone() for group in self.param_groups
                                    for p in group["params"] if p.grad is not None))
        return super().step(closure)


class _RecordComm(CommScheduler):
    def __init__(self):
        super().__init__()
        self.calls = []
    def launch_autograd_pull(self, route, x, **kwargs):
        self.calls.append((kwargs.get("name"), route.send_sizes, route.recv_sizes))
        return super().launch_autograd_pull(route, x, **kwargs)


def _run_case(model_cls, *, device, placement, pipeline=True, rank=0, world=1,
              inspect_layout=True, train_loss_mode="window_mean"):
    torch.manual_seed(23)
    layers = 1 if model_cls is sg.EvolveGCNModel else 2
    model = model_cls(1, 3, 1, num_layers=layers).to(device)
    optimizer, comm, store = _RecordAdam(model.parameters()), _RecordComm(), _store(rank, world)
    store.features.to(device)
    task = sg.NodePredictionTask(name="node_regression", loss="mse", train_loss_mode=train_loss_mode)
    outputs, layouts, access_devices = [], [], []
    current_batch = None
    original_entries = materialize._materialize_snapshot_entries
    def entries(*args, **kwargs):
        if placement and torch.device(device).type == "cuda":
            assert torch.cuda.current_stream(device).cuda_stream != torch.cuda.default_stream(device).cuda_stream
        result = original_entries(*args, **kwargs)
        access_devices.extend(entry.graph.dst_nodes.device.type for entry in result)
        return result
    def batch_check(batch):
        layout = []
        for slot, (block,) in enumerate(batch.blocks):
            target = (batch.targets["window_tasks"][slot] if "window_tasks" in batch.targets
                      else batch.targets["task"] if slot == len(batch.blocks) - 1 else None)
            supervision = None
            if target is not None:
                assert target.target_ids.device == target.label.device == block.dst_nodes.device
                assert target.node_ids.device == block.dst_nodes.device
                assert target.target_ts is None or target.target_ts.device == block.dst_nodes.device
                torch.testing.assert_close(block.dst_nodes[target.target_route.target_rows], target.target_ids)
                target_order = target.target_ids.argsort(stable=True)
                supervision = (target.target_ids[target_order].cpu().tolist(), target.label[target_order].cpu().tolist())
            partial = bool(block.cache["chunk_limited"])
            if partial:
                assert block.route is None and block.src_nodes.numel() == block.dst_nodes.numel()
            layout.append((partial, block.dst_nodes.sort().values.cpu().tolist(), supervision))
        layouts.append(layout)
    def capture_batch(batch):
        nonlocal current_batch
        current_batch = batch
        if inspect_layout:
            batch_check(batch)
    def capture_output(output):
        # Device materialization may permute physical owner rows. Keep identities
        # beside predictions; align only after the epoch, including overlap tests.
        outputs.append((current_batch.blocks[-1][-1].dst_nodes.detach().clone(),
                        output.logits.detach().clone()))
    with patch.object(materialize, "_materialize_snapshot_entries", side_effect=entries):
        result = run_epoch(store=store, model=model, task=task, mode="snapshot", training=True,
            optimizer=optimizer, window_policy="chunk_decay", sampling_policy="full", chunk_decay=(1, 1),
            num_full_snapshots=1, num_layers=layers,
            sampler_options={"snapshot_materialize_on_device": placement, "snapshot_dgl_gcn": True,
                "access_pipeline": pipeline, "rolling_snapshot_cache": True,
                "chunk_order": torch.arange(4 // world).flip(0)},
            comm=comm, device=device, gradient_sync="all_reduce" if world > 1 else None,
            output_callback=capture_output,
            batch_callback=capture_batch)
    assert result.steps == 3 and torch.isfinite(torch.tensor(result.loss))
    assert access_devices and set(access_devices) == ({torch.device(device).type} if placement else {"cpu"})
    return result, outputs, optimizer.gradients, layouts, comm.calls


def _compare(before, after, *, exact=False):
    assert before[0].steps == after[0].steps
    assert before[0].loss == pytest.approx(after[0].loss, rel=0 if exact else 1e-5, abs=0 if exact else 1e-6)
    assert before[3:] == after[3:]
    for (a_ids, a), (b_ids, b) in zip(before[1], after[1]):
        a_order, b_order = a_ids.argsort(stable=True), b_ids.argsort(stable=True)
        torch.testing.assert_close(a_ids[a_order], b_ids[b_order], rtol=0, atol=0)
        torch.testing.assert_close(a[a_order], b[b_order], rtol=0 if exact else 1e-5, atol=0 if exact else 1e-6)
    assert len(before[2]) == len(after[2])
    for step_a, step_b in zip(before[2], after[2]):
        assert len(step_a) == len(step_b)
        for a, b in zip(step_a, step_b):
            torch.testing.assert_close(a, b, rtol=0 if exact else 1e-5, atol=0 if exact else 1e-6)


@pytest.mark.parametrize("model_cls", [sg.TGCNModel, sg.MPNNLSTMModel, sg.EvolveGCNModel])
@pytest.mark.parametrize("train_loss_mode", ["last_only", "window_mean"])
def test_cpu_opt_in_preserves_epoch_math(model_cls, train_loss_mode):
    # Last-only retains physical order. Window-mean also permutes full rows, so
    # parameter-gradient reductions use the existing CUDA/NCCL tolerance.
    _compare(_run_case(model_cls, device="cpu", placement=False, train_loss_mode=train_loss_mode),
             _run_case(model_cls, device="cpu", placement=True, train_loss_mode=train_loss_mode),
             exact=train_loss_mode == "last_only")


@pytest.mark.skipif(os.environ.get("STARRYGL_TEST_CUDA_MATERIALIZE") != "1", reason="explicit CUDA opt-in required")
@pytest.mark.parametrize("pipeline", [False, True])
def test_cuda_placement_stream_and_math(pipeline):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    _compare(_run_case(sg.TGCNModel, device="cuda:0", placement=False, pipeline=pipeline),
             _run_case(sg.TGCNModel, device="cuda:0", placement=True, pipeline=pipeline))


@pytest.mark.skipif(os.environ.get("STARRYGL_TEST_CUDA_MATERIALIZE") != "1", reason="explicit CUDA opt-in required")
def test_cuda_overlap_without_host_inspection_callback():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    _compare(_run_case(sg.TGCNModel, device="cuda:0", placement=False, inspect_layout=False),
             _run_case(sg.TGCNModel, device="cuda:0", placement=True, inspect_layout=False))


def _distributed_worker(rank, rendezvous, backend):
    torch.set_num_threads(1)
    device = f"cuda:{rank}" if backend == "nccl" else "cpu"
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(backend, init_method="file://" + rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=45))
    try:
        before = _run_case(sg.TGCNModel, device=device, placement=False, rank=rank, world=2)
        dist.barrier()
        after = _run_case(sg.TGCNModel, device=device, placement=True, rank=rank, world=2)
        _compare(before, after)
        assert before[4], "two-layer TGCN must communicate full-route embeddings"
        assert any(partial for batch in after[3] for partial, _, _ in batch)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_two_rank_cpu_fixture_preserves_routes_and_gradients(tmp_path):
    mp.spawn(_distributed_worker, args=(str(tmp_path / "rendezvous"), "gloo"), nprocs=2, join=True)


@pytest.mark.skipif(os.environ.get("STARRYGL_TEST_CUDA_MATERIALIZE") != "1", reason="explicit CUDA opt-in required")
def test_two_rank_nccl_placement_preserves_forward_loss_gradients(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices required")
    mp.spawn(_distributed_worker, args=(str(tmp_path / "rendezvous"), "nccl"), nprocs=2, join=True)
