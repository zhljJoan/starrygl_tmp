import pytest
import torch

from starrygl.runtime.comm import CommScheduler
from starrygl.store.remote_fetch import _restore_order, submit_owner_request


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_owner_response_restores_cpu_route_on_payload_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    order = torch.tensor([2, 0, 1])
    value = torch.tensor([[30.0], [10.0], [20.0]], device=device)
    torch.testing.assert_close(_restore_order(value, order), value.new_tensor([[10.0], [20.0], [30.0]]))
    assert _restore_order(value[:0], order[:0]).shape == (0, 1)


def test_fixed_owner_request_carries_counts_and_ids_in_one_packet():
    request = submit_owner_request(
        torch.tensor([2, 1]),
        torch.empty(0, dtype=torch.long),
        scheduler=CommScheduler(),
        name="test",
        order=torch.tensor([1, 0]),
        send_counts=torch.tensor([2]),
        packet_capacity=3,
    )

    torch.testing.assert_close(request.recv_counts, torch.tensor([2]))
    torch.testing.assert_close(request.recv_nodes, torch.tensor([1, 2]))
