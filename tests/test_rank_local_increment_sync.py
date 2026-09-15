"""Epoch synchronization must not broadcast rank-local increment observations."""
from datetime import timedelta
import os

import pytest
import torch
import torch.distributed as dist
from torch import nn

from starrygl.model.layers.temporal import StateIncrementEstimator
from starrygl.runtime import epoch


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(0.5))
        self.register_buffer("persistent", torch.tensor([2.0]))
        self.local = StateIncrementEstimator(1, 2)

    def get_extra_state(self):
        return {"format": "test"}

    def set_extra_state(self, state):
        assert state == {"format": "test"}


def test_increment_checkpoint_and_device_contract():
    model = _Model().to(dtype=torch.float64)
    model.local.update(torch.tensor([0, 0]), torch.tensor([[1., 3.], [3., 5.]]))
    torch.testing.assert_close(model.local.estimate(torch.tensor([0])), torch.tensor([[2., 4.]], dtype=torch.float64))
    assert model.local.count.dtype == model.local.increment.dtype == torch.float64
    assert set(model.state_dict()) == {"gamma", "persistent", "_extra_state"}
    restored = _Model().to(dtype=torch.float64)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.local.count.sum() == 0
    torch.testing.assert_close(restored.gamma, model.gamma)
    model.local.clear()
    assert model.local.count.sum() == model.local.increment.sum() == 0


def test_sync_only_parameters_and_persistent_tensor_buffers(monkeypatch):
    model = _Model()
    model.local.update(torch.tensor([0]), torch.tensor([[3., 4.]]))
    before = (model.local.count.clone(), model.local.increment.clone())
    seen = []
    monkeypatch.setattr(epoch, "_distributed_sync", lambda mode: True)

    def broadcast(value, src):
        assert isinstance(value, torch.Tensor) and src == 0
        seen.append(value)
        value.fill_(9)

    monkeypatch.setattr(dist, "broadcast", broadcast)
    epoch.sync_model_parameters(model, "all_reduce")
    assert len(seen) == 2
    assert model.gamma.item() == model.persistent.item() == 9
    torch.testing.assert_close(model.local.count, before[0], rtol=0, atol=0)
    torch.testing.assert_close(model.local.increment, before[1], rtol=0, atol=0)


def test_two_rank_different_increment_shapes_at_next_epoch():
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("run with torch.distributed.run --nproc_per_node=2")
    cuda = os.environ.get("STARRYGL_TEST_NCCL") == "1"
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"])) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    created = not dist.is_initialized()
    if created:
        dist.init_process_group("nccl" if cuda else "gloo", timeout=timedelta(seconds=30))
    try:
        rank = dist.get_rank()
        model = _Model().to(device)
        with torch.no_grad():
            model.gamma.fill_(rank + 0.5)
            model.persistent.fill_(rank + 2.)
        epoch.sync_model_parameters(model, "all_reduce")
        assert model.gamma.item() == 0.5 and model.persistent.item() == 2.
        model.local.update(torch.tensor([rank * 7], device=device),
                           torch.tensor([[rank + 1., rank + 2.]], device=device))
        before = (model.local.count.clone(), model.local.increment.clone())
        with torch.no_grad():
            model.gamma.fill_(rank + 10.)
            model.persistent.fill_(rank + 20.)
        epoch.sync_model_parameters(model, "all_reduce")
        assert model.gamma.item() == 10. and model.persistent.item() == 20.
        assert model.local.count.shape[0] == (1 if rank == 0 else 8)
        torch.testing.assert_close(model.local.count, before[0], rtol=0, atol=0)
        torch.testing.assert_close(model.local.increment, before[1], rtol=0, atol=0)
        assert set(model.state_dict()) == {"gamma", "persistent", "_extra_state"}
    finally:
        if created:
            dist.destroy_process_group()
