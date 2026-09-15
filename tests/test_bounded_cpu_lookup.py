from types import SimpleNamespace

import pytest
import torch

from starrygl.utils.index import compact_lookup_rows


def _expected(source, query):
    last = {key: row for row, key in enumerate(source.long().tolist())}
    return torch.tensor([last.get(key, -1) for key in query.long().tolist()])


@pytest.mark.parametrize("keys,queries", [
    ([1, 1, 2], [1, 2, 3, -1]),
    ([-5, -1, -5, 0], [-5, -4, -1, 0, 5]),
    ([-2**63, -2**63 + 2], [-2**63, -2**63 + 1, -2**63 + 2, 2**63 - 1]),
    ([2**63 - 3, 2**63 - 1], [-2**63, 2**63 - 3, 2**63 - 2, 2**63 - 1]),
])
def test_dense_last_source_row_and_extreme_offsets(keys, queries, monkeypatch):
    source = torch.tensor(keys).repeat(128)
    query = torch.tensor(queries)

    def no_sort(*args, **kwargs):
        raise AssertionError("dense CPU lookup must not sort")

    monkeypatch.setattr(torch, "argsort", no_sort)
    assert torch.equal(compact_lookup_rows(source, query), _expected(source, query))


@pytest.mark.parametrize("source_dtype", [torch.int32, torch.int64, torch.float64])
@pytest.mark.parametrize("query_dtype", [torch.int32, torch.int64, torch.float64])
def test_dense_dtype_conversion_and_noncontiguous_inputs(source_dtype, query_dtype):
    source = torch.tensor([-3.9, 99, 2.2, 99, -3.1, 99], dtype=source_dtype).repeat(128)[::2]
    query = torch.tensor([-3.5, 88, 2.1, 88, 8.9, 88], dtype=query_dtype)[::2]
    actual = compact_lookup_rows(source, query)
    assert actual.dtype == torch.long
    assert torch.equal(actual, _expected(source, query))


@pytest.mark.parametrize("source,query", [([], [1, -1]), ([1, 2], []), ([], [])])
def test_empty_inputs(source, query):
    source, query = torch.tensor(source), torch.tensor(query)
    assert torch.equal(compact_lookup_rows(source, query), _expected(source, query))


@pytest.mark.parametrize("keys", [[-2**63, 2**63 - 1], [0, 2**62]])
def test_sparse_huge_range_uses_sort_without_inverse(keys, monkeypatch):
    source = torch.tensor(keys).repeat(128)
    query = torch.tensor([*keys, -1, 1])

    def no_inverse(*args, **kwargs):
        raise AssertionError("sparse IDs must not allocate a global-range inverse")

    monkeypatch.setattr(torch, "full", no_inverse)
    assert torch.equal(compact_lookup_rows(source, query), _expected(source, query))


def test_dense_budget_includes_query_rows(monkeypatch):
    source = torch.arange(256) * 10
    query = torch.arange(3000)

    def no_sort(*args, **kwargs):
        raise AssertionError("the query budget admits this dense CPU span")

    monkeypatch.setattr(torch, "argsort", no_sort)
    assert torch.equal(compact_lookup_rows(source, query), _expected(source, query))


def test_dense_scratch_stays_on_cpu_with_another_default_device():
    source = torch.tensor([3, -1, 3]).repeat(128)
    query = torch.tensor([3, -1, 9])
    expected = _expected(source, query)
    with torch.device("meta"):
        actual = compact_lookup_rows(source, query)
    assert actual.device.type == "cpu"
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("original_cuda", [False, True])
def test_small_or_original_cuda_source_retains_sort(original_cuda, monkeypatch):
    source = torch.tensor([3, 1, 3]).repeat(128 if original_cuda else 1)
    query = torch.tensor([3, 9, 1])
    original = source
    if original_cuda:
        # Exercise device dispatch without allocating GPU memory. The original
        # helper transfers a source to the query device before sorting it.
        original = SimpleNamespace(device=torch.device("cuda"), to=lambda **kwargs: source)

    def no_dense_reduction(*args, **kwargs):
        raise AssertionError("small/original CUDA source must retain sort/search")

    monkeypatch.setattr(torch, "aminmax", no_dense_reduction)
    assert torch.equal(compact_lookup_rows(original, query), _expected(source, query))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("source_device,query_device", [("cuda", "cuda"), ("cpu", "cuda"), ("cuda", "cpu")])
def test_actual_gpu_inputs_retain_sort(source_device, query_device, monkeypatch):
    source = torch.tensor([3, -1, 3]).repeat(128).to(source_device)
    query = torch.tensor([3, -1, 9]).to(query_device)

    def no_cpu_reduction(*args, **kwargs):
        raise AssertionError("any GPU input must retain sort/search")

    monkeypatch.setattr(torch, "aminmax", no_cpu_reduction)
    assert torch.equal(compact_lookup_rows(source, query).cpu(), torch.tensor([383, 382, -1]))
