import torch

from starrygl.prepare import load_graph_data
from starrygl.tools.convert_to_starrygl_format import (
    _pack_web_windows,
    _read_numeric_rows,
    _write_flat_temporal_tensors,
    convert_tgl,
)


def test_tgl_conversion_preserves_time_splits_features_and_node_labels(tmp_path) -> None:
    source = tmp_path / "raw"
    output = tmp_path / "converted"
    source.mkdir()
    (source / "edges.csv").write_text(
        ",src,dst,time,ext_roll\n0,1,2,20,1\n1,0,1,10,0\n",
        encoding="utf-8",
    )
    (source / "labels.csv").write_text(
        ",node,time,label,ext_roll\n0,0,9,2,0\n1,2,19,1,1\n",
        encoding="utf-8",
    )
    torch.save(torch.arange(6, dtype=torch.float32).reshape(3, 2), source / "node_features.pt")
    torch.save(torch.tensor([[20.0], [10.0]]), source / "edge_features.pt")

    convert_tgl(source, out_dir=output)
    graph = load_graph_data(output)

    assert graph.ts.tolist() == [10.0, 20.0]
    assert graph.split_labels.tolist() == [0, 1]
    assert graph.edge_ids.tolist() == [1, 0]
    assert graph.node_feat.tolist() == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
    assert graph.edge_feat.tolist() == [[10.0], [20.0]]
    assert graph.node_label_nodes.tolist() == [0, 2]
    assert graph.node_label_ts.tolist() == [9.0, 19.0]
    assert graph.node_label_split.tolist() == [0, 1]


def test_tgl_loads_memshare_learned_node_features(tmp_path) -> None:
    torch.save({
        "src": torch.tensor([0]),
        "dst": torch.tensor([1]),
    }, tmp_path / "graph.pt")
    expected = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    torch.save(expected, tmp_path / "learned_node_feats.pt")

    assert torch.equal(load_graph_data(tmp_path).node_feat, expected)


def test_dtdg_numeric_reader_preserves_headerless_first_row(tmp_path) -> None:
    plain = tmp_path / "edges.txt"
    plain.write_text("1 2 3\n4 5 6\n", encoding="utf-8")
    headed = tmp_path / "edges.csv"
    headed.write_text("src,dst,time\n1,2,3\n4,5,6\n", encoding="utf-8")

    expected = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert _read_numeric_rows(plain, skiprows=0, sep=r"\s+").tolist() == expected
    assert _read_numeric_rows(headed, skiprows=0, sep=",").tolist() == expected


def test_dtdg_conversion_records_next_snapshot_label_horizon(tmp_path) -> None:
    flat = _pack_web_windows(
        torch.tensor([0, 1, 0, 1]),
        torch.tensor([1, 0, 1, 0]),
        torch.arange(4, dtype=torch.float32),
        torch.ones(4),
        torch.tensor([[0, 2], [2, 4]]),
        num_nodes=2,
    )

    _write_flat_temporal_tensors(flat, out_dir=tmp_path)
    graph = load_graph_data(tmp_path)

    assert graph.node_label_horizon == 1
    assert torch.isnan(graph.node_label[-1]).all()
