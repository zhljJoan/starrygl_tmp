import pytest
import torch

from starrygl.batch import Batch
from starrygl.model._graph_ops import edge_endpoint_embeddings
from starrygl.runtime.event.target import deduplicate_roots
from starrygl.task.target import attach_target_route, build_task_target, sampling_roots_from_target
from starrygl.utils.index import temporal_lookup_rows
from starrygl.view import GraphBlock


@pytest.mark.parametrize('mode', ['dst', 'src_dst'])
@pytest.mark.parametrize('lazy', [False, True])
def test_endpoint_rows_keep_cutoffs_and_negative_ratio(mode, lazy):
    target = build_task_target(
        target_kind='edge', target_ids=torch.tensor([0, 1]),
        pos_src=torch.tensor([9, 9]), pos_dst=torch.tensor([8, 8]),
        target_ts=torch.tensor([1.25, 1.75]),
        neg_src=torch.tensor([9, 9, 9, 9]) if mode == 'src_dst' else None,
        neg_dst=torch.tensor([7, 7, 7, 7]),
    )
    roots = sampling_roots_from_target(target)
    nodes, ts = deduplicate_roots(roots.node_ids, roots.ts, enabled=True)
    # Permute MFG rows to ensure the route is based on identity, not position.
    nodes, ts = nodes.flip(0), ts.flip(0)
    graph = GraphBlock(format='csc', src_nodes=nodes, dst_nodes=nodes, edge_ids=torch.empty(0, dtype=torch.long),
                       srcdata={'ts': ts}, dstdata={'ts': ts})
    target = attach_target_route(graph, target, lazy=lazy)
    values = (nodes.float() * 10 + ts).reshape(-1, 1).requires_grad_()
    batch = Batch(mode='event', graph=graph, targets={'task': target})
    out = edge_endpoint_embeddings(batch, graph, values)
    torch.testing.assert_close(out['pos_src'][:, 0], torch.tensor([91.25, 91.75]))
    torch.testing.assert_close(out['pos_dst'][:, 0], torch.tensor([81.25, 81.75]))
    torch.testing.assert_close(out['neg_dst'][:, 0], torch.tensor([71.25, 71.25, 71.75, 71.75]))
    out['pos_src'][0].sum().backward()
    assert values.grad[:, 0].nonzero().flatten().tolist() == target.target_route.pos_src_rows[:1].tolist()


def test_node_target_missing_cutoff_and_large_ids():
    nodes = torch.tensor([2**54, 2**54 + 1, 2**54])
    ts = torch.tensor([1.25, 1.25, 1.75], dtype=torch.float64)
    rows = temporal_lookup_rows(nodes, ts, nodes, torch.tensor([1.75, 1.25, 2.0], dtype=torch.float64))
    assert rows.tolist() == [2, 1, -1]
    target = build_task_target(target_kind='node', target_ids=nodes[:1], target_ts=ts[2:])
    graph = GraphBlock(format='csc', src_nodes=nodes, dst_nodes=nodes, edge_ids=torch.empty(0, dtype=torch.long),
                       dstdata={'ts': ts})
    assert attach_target_route(graph, target).target_route.target_rows.tolist() == [2]
    empty = temporal_lookup_rows(nodes[:0], ts[:0], nodes[:0], ts[:0])
    assert empty.numel() == 0


def test_temporal_join_empty_and_duplicate_rows():
    nodes, ts = torch.tensor([3, 3]), torch.tensor([2.0, 2.0])
    assert temporal_lookup_rows(nodes, ts, nodes[:1], ts[:1]).tolist() == [1]
    assert temporal_lookup_rows(nodes[:0], ts[:0], nodes, ts).tolist() == [-1, -1]
    assert temporal_lookup_rows(nodes, ts, nodes[:0], ts[:0]).numel() == 0
    graph = GraphBlock(format='csc', src_nodes=nodes[:0], dst_nodes=nodes[:0],
                       edge_ids=nodes[:0], dstdata={'ts': ts[:0]})
    target = build_task_target(target_kind='edge', target_ids=nodes[:0], target_ts=ts[:0],
                               pos_src=nodes[:0], pos_dst=nodes[:0], neg_dst=nodes[:0])
    target = attach_target_route(graph, target, lazy=True)
    batch = Batch(mode='event', graph=graph, targets={'task': target})
    out = edge_endpoint_embeddings(batch, graph, torch.empty(0, 2))
    assert all(value.shape == (0, 2) for value in out.values())
