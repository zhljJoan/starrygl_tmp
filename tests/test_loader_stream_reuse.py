import os

import pytest
import torch

from starrygl.runtime.comm import CommScheduler
from starrygl.runtime.dataloader.loader import DataLoader
from test_snapshot_device_materialization import _store


def _loader(store, device):
    return DataLoader(store, mode='snapshot', split='train', window_policy='full_snapshot',
        sampling_policy='full', chunk_decay=None, num_full_snapshots=1, num_layers=1,
        fanouts=None, sampler_options={}, num_negatives=0, generator=None,
        comm=CommScheduler(), device=device, prefetch_state=None)


def test_cpu_loader_does_not_allocate_cuda_stream():
    store = _store()
    assert _loader(store, 'cpu').prefetch_stream is None
    assert 'loader_streams' not in store.graph.runtime_cache


@pytest.mark.skipif(os.environ.get('STARRYGL_TEST_CUDA_MATERIALIZE') != '1', reason='explicit CUDA opt-in required')
def test_cuda_loader_reuses_store_device_stream_without_cross_device_aliasing():
    store = _store()
    with torch.cuda.device(0):
        first = _loader(store, 'cuda')
        assert _loader(store, 'cuda:0').prefetch_stream is first.prefetch_stream
        assert _loader(store, 'cpu').prefetch_stream is None
        assert _loader(_store(), 'cuda:0').prefetch_stream is not first.prefetch_stream
        if torch.cuda.device_count() > 1:
            other = _loader(store, 'cuda:1')
            assert other.prefetch_stream is not first.prefetch_stream
            assert other.prefetch_stream.device == torch.device('cuda:1')
        assert _loader(store, 'cuda:0').prefetch_stream is first.prefetch_stream
        assert first.prefetch_stream.device == torch.device('cuda:0')
