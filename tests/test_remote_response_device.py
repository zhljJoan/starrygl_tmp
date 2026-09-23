import pytest
import torch

from starrygl.store.remote_fetch import _restore_order


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_owner_response_restores_cpu_route_on_payload_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    order = torch.tensor([2, 0, 1])
    value = torch.tensor([[30.0], [10.0], [20.0]], device=device)
    torch.testing.assert_close(_restore_order(value, order), value.new_tensor([[10.0], [20.0], [30.0]]))
    assert _restore_order(value[:0], order[:0]).shape == (0, 1)
