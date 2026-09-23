import pytest
import torch

from starrygl.runtime.comm import CommScheduler
from starrygl.store.remote_fetch import (
    _restore_order,
    finish_owner_request,
    launch_owner_request,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_owner_response_restores_cpu_route_on_payload_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    order = torch.tensor([2, 0, 1])
    value = torch.tensor([[30.0], [10.0], [20.0]], device=device)
    torch.testing.assert_close(_restore_order(value, order), value.new_tensor([[10.0], [20.0], [30.0]]))
    assert _restore_order(value[:0], order[:0]).shape == (0, 1)


def test_owner_request_exposes_count_handle_before_node_exchange():
    nodes = torch.tensor([2, 1])
    pending = launch_owner_request(
        nodes,
        torch.empty(0, dtype=torch.long),
        scheduler=CommScheduler(),
        name="test",
        order=torch.tensor([1, 0]),
        send_counts=torch.tensor([2]),
    )

    assert pending.count_handle.name == "test:counts"
    request = finish_owner_request(pending)
    torch.testing.assert_close(request.node_ids, nodes)
    torch.testing.assert_close(request.recv_nodes, torch.tensor([1, 2]))
