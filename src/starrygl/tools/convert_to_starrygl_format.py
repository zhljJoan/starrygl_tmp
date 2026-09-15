from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from starrygl.prepare import load_graph_data


@dataclass(frozen=True)
class WebSplitConfig:
    name: str
    window: int = 200
    batch_size: int | None = None
    use_batch_split: bool = False
    edge_file: str | None = None
    skiprows: int = 2
    sep: str = r"\s+"
    skiptime: int = 1
    lags: int = 30


@dataclass(frozen=True)
class FlatTemporalTensors:
    src: Tensor
    dst: Tensor
    ts: Tensor
    edge_ids: Tensor
    time_ptr_2: Tensor
    num_nodes: int
    edge_weight: Tensor | None = None
    node_feat: Tensor | None = None
    node_label: Tensor | None = None


WEB_SPLIT_CONFIGS: dict[str, WebSplitConfig] = {
    "ia-slashdot-reply-dir": WebSplitConfig("ia-slashdot-reply-dir", window=200, skiprows=2, sep=r"\s+", skiptime=1, lags=30),
    "WikiTalk": WebSplitConfig("WikiTalk", batch_size=3000, use_batch_split=True, edge_file="WikiTalk/edges.csv", skiprows=0, sep=",", skiptime=1),
    "rec-amazon-ratings": WebSplitConfig("rec-amazon-ratings", window=100, skiprows=2, sep=r"\s+", skiptime=1, lags=30),
    "rec-amz-Books": WebSplitConfig("rec-amz-Books", window=100, skiprows=0, sep=",", skiptime=0, lags=0),
    "soc-bitcoin": WebSplitConfig("soc-bitcoin", window=100, skiprows=0, sep=r"\s+", skiptime=1, lags=10),
    "soc-flickr-growth": WebSplitConfig("soc-flickr-growth", window=100, skiprows=1, sep=r"\s+", skiptime=1, lags=30),
    "soc-youtube-growth": WebSplitConfig("soc-youtube-growth", window=100, skiprows=2, sep=r"\s+", skiptime=1, lags=20),
    "stackexch": WebSplitConfig("stackexch", window=200, skiprows=2, sep=r"\s+", skiptime=1, lags=30),
}


def convert_to_starrygl_format(
    source: str | Path,
    out: str | Path,
    *,
    kind: str = "auto",
    lags: int = 1,
    dataset: str | None = None,
) -> dict[str, Any]:
    source_path = Path(source).expanduser()
    out_dir = Path(out).expanduser()
    resolved_kind = _detect_kind(source_path) if kind == "auto" else kind
    if resolved_kind in {"tgl", "tgm"}:
        return convert_tgl(source_path, out_dir=out_dir)
    if resolved_kind == "dtdg":
        return convert_dtdg(source_path, out_dir=out_dir, lags=lags, dataset=dataset)
    raise ValueError(f"unsupported conversion kind: {resolved_kind}")


def convert_tgl(source: Path, *, out_dir: Path) -> dict[str, Any]:
    graph = load_graph_data(source)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_out = {
        "src": graph.src,
        "dst": graph.dst,
        "ts": graph.ts,
        "edge_ids": graph.edge_ids,
        "num_nodes": graph.num_nodes,
    }
    if graph.split_labels is not None:
        graph_out["split_labels"] = graph.split_labels
    label_events = _load_node_label_events(source)
    if label_events is not None:
        graph_out.update(label_events)
    torch.save(graph_out, out_dir / "graph.pt")
    _save_tensor(out_dir / "node_feat.pt", graph.node_feat)
    _save_tensor(out_dir / "edge_feat.pt", graph.edge_feat)
    _save_tensor(out_dir / "node_label.pt", graph.node_label if graph.node_label is not None and label_events is None else None)
    _save_tensor(out_dir / "edge_label.pt", graph.edge_label)
    return {"out": str(out_dir), "kind": "tgl", "num_edges": int(graph.src.numel())}


def convert_dtdg(source: Path, *, out_dir: Path, lags: int, dataset: str | None = None) -> dict[str, Any]:
    if source.suffix in {".pt", ".pth"}:
        raise ValueError("DTDG conversion expects raw edge data, not preprocessed .pt/.pth snapshot datasets")
    flat = _flat_temporal_from_web_edges(source, dataset=dataset)
    _write_flat_temporal_tensors(flat, out_dir=out_dir)
    return {"out": str(out_dir), "kind": "dtdg", "num_edges": int(flat.src.numel())}


def _flat_temporal_from_web_edges(source: Path, *, dataset: str | None) -> FlatTemporalTensors:
    config = _web_config(source, dataset)
    data = _read_numeric_rows(_web_edge_path(source, config), skiprows=config.skiprows, sep=config.sep)
    src, dst, weight, ts = _web_uvwt(data)
    order = torch.argsort(ts, stable=True)
    src = src.index_select(0, order)
    dst = dst.index_select(0, order)
    weight = weight.index_select(0, order)
    ts = ts.index_select(0, order)
    src, dst, num_nodes = _remap_nodes(src, dst)
    windows = _batch_windows(ts, config) if config.use_batch_split else _masked_windows(ts, config)
    return _pack_web_windows(src, dst, ts, weight, windows, num_nodes=num_nodes)


def _write_flat_temporal_tensors(flat: FlatTemporalTensors, *, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    graph = {
        "src": flat.src,
        "dst": flat.dst,
        "ts": flat.ts,
        "edge_ids": flat.edge_ids,
        "num_nodes": flat.num_nodes,
        "time_ptr_2": flat.time_ptr_2,
    }
    if flat.node_label is not None:
        graph["node_label_horizon"] = 1
    torch.save(graph, out_dir / "graph.pt")
    if flat.edge_weight is not None:
        torch.save(flat.edge_weight.reshape(-1, 1).contiguous(), out_dir / "edge_feat.pt")
        torch.save(flat.edge_weight.contiguous(), out_dir / "edge_label.pt")
    _save_tensor(out_dir / "node_feat.pt", flat.node_feat)
    _save_tensor(out_dir / "node_label.pt", flat.node_label)


def _web_config(source: Path, dataset: str | None) -> WebSplitConfig:
    if dataset is not None:
        if dataset not in WEB_SPLIT_CONFIGS:
            raise ValueError(f"unknown web dtdg dataset {dataset!r}; supported: {sorted(WEB_SPLIT_CONFIGS)}")
        return WEB_SPLIT_CONFIGS[dataset]
    for name in (source.stem, source.name, source.parent.name):
        if name in WEB_SPLIT_CONFIGS:
            return WEB_SPLIT_CONFIGS[name]
    if source.is_file():
        return WebSplitConfig(source.stem, skiprows=0, sep="," if source.suffix == ".csv" else r"\s+", skiptime=0, lags=1)
    raise ValueError("cannot infer web dtdg dataset; pass --dataset")


def _web_edge_path(source: Path, config: WebSplitConfig) -> Path:
    if source.is_file():
        return source
    candidates = [
        *( [source / config.edge_file] if config.edge_file else [] ),
        source / config.name / f"{config.name}.edges",
        source / config.name / "edges.csv",
        source / f"{config.name}.edges",
        source / "edges.csv",
        source / "edges.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(source.glob("*.edges"))
    if matches:
        return matches[0]
    raise ValueError(f"cannot find web dtdg edge file under {source}")


def _read_numeric_rows(path: Path, *, skiprows: int, sep: str) -> Tensor:
    first_row = None
    has_header = False
    with path.open("r", encoding="utf-8", newline="") as f:
        for line_no, line in enumerate(f):
            if line_no < skiprows:
                continue
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "%")):
                continue
            first_row = line_no
            parts = stripped.split(",") if sep == "," else stripped.split()
            try:
                [float(part) for part in parts if part.strip()]
            except ValueError:
                has_header = True
            break
    if first_row is None:
        raise ValueError(f"no numeric edge rows found in {path}")
    frame = pd.read_csv(path, skiprows=first_row, sep=sep, header=0 if has_header else None)
    if frame.empty:
        raise ValueError(f"no numeric edge rows found in {path}")
    return torch.as_tensor(frame.to_numpy(dtype="float64", copy=False), dtype=torch.float64).contiguous()


def _web_uvwt(data: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    width = int(data.size(1))
    if width == 3:
        return data[:, 0].long(), data[:, 1].long(), torch.ones(data.size(0), dtype=torch.float32), data[:, 2].long()
    if width == 4:
        return data[:, 0].long(), data[:, 1].long(), data[:, 2].float(), data[:, 3].long()
    if width == 5:
        return data[:, 1].long(), data[:, 2].long(), torch.ones(data.size(0), dtype=torch.float32), data[:, 3].long()
    raise ValueError("web dtdg rows must have 3, 4, or 5 columns")


def _batch_windows(ts: Tensor, config: WebSplitConfig) -> Tensor:
    if not config.batch_size:
        raise ValueError("batch_size is required for batch split")
    uniq = torch.unique(ts, sorted=True)
    begin = 0
    if 0 < config.skiptime < int(uniq.numel()):
        begin = int(torch.searchsorted(ts, uniq[config.skiptime], right=False).item())
    rows = [(left, min(left + int(config.batch_size), int(ts.numel()))) for left in range(begin, int(ts.numel()), int(config.batch_size))]
    return torch.tensor(rows, dtype=torch.long) if rows else torch.empty((0, 2), dtype=torch.long)


def _masked_windows(ts: Tensor, config: WebSplitConfig) -> Tensor:
    uniq = torch.unique(ts, sorted=True)[config.skiptime:]
    if int(uniq.numel()) == 0:
        return torch.empty((0, 2), dtype=torch.long)
    start, end = float(uniq[0].item()), float(uniq[-1].item())
    step = (end - start) / float(config.window)
    rows: list[tuple[int, int]] = []
    ts_float = ts.float().contiguous()
    for i in range(max(0, config.window - config.lags)):
        left = int(torch.searchsorted(ts_float, torch.tensor(start + i * step, dtype=ts_float.dtype), right=False).item())
        right = int(torch.searchsorted(ts_float, torch.tensor(start + (i + config.lags + 1) * step, dtype=ts_float.dtype), right=False).item())
        if right - left > 50:
            rows.append((left, right))
    return torch.tensor(rows, dtype=torch.long) if rows else torch.empty((0, 2), dtype=torch.long)


def _pack_web_windows(src: Tensor, dst: Tensor, ts: Tensor, weight: Tensor, windows: Tensor, *, num_nodes: int) -> FlatTemporalTensors:
    feats: list[Tensor] = []
    labels: list[Tensor] = []
    for begin, end in windows.tolist():
        begin = int(begin)
        end = int(end)
        src_win = src[begin:end].long()
        dst_win = dst[begin:end].long()
        weight_win = weight[begin:end].to(dtype=torch.long).to(dtype=torch.float32).contiguous()
        in_deg = torch.zeros(num_nodes, dtype=torch.float32)
        out_deg = torch.zeros(num_nodes, dtype=torch.float32)
        in_deg.scatter_add_(0, dst_win, weight_win)
        out_deg.scatter_add_(0, src_win, weight_win)
        feats.append(torch.stack((in_deg, out_deg), dim=1))
        labels.append(torch.log(in_deg + 1))
    if not feats:
        node_feat = None
        node_label = None
    else:
        node_feat = torch.stack(feats, dim=0).contiguous()
        node_label = torch.stack(labels[1:] + [torch.full_like(labels[0], float("nan"))], dim=0).contiguous()
    return FlatTemporalTensors(
        src=src.long().contiguous(),
        dst=dst.long().contiguous(),
        ts=ts.float().contiguous(),
        edge_ids=torch.arange(int(src.numel()), dtype=torch.long),
        time_ptr_2=windows.long().contiguous(),
        num_nodes=num_nodes,
        edge_weight=weight.to(dtype=torch.long).to(dtype=torch.float32).contiguous(),
        node_feat=node_feat,
        node_label=node_label,
    )


def _load_node_label_events(source: Path) -> dict[str, Tensor] | None:
    path = source / "labels.csv" if source.is_dir() else None
    if path is None or not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty:
        return None
    cols = {str(name).strip().lower(): name for name in frame.columns}
    node_col = cols.get("node", cols.get("nid", cols.get("id")))
    label_col = cols.get("label", cols.get("y"))
    ts_col = cols.get("time", cols.get("ts"))
    split_col = cols.get("ext_roll", cols.get("split", cols.get("role", cols.get("int_roll"))))
    if node_col is None or label_col is None:
        return None
    out = {
        "node_label_nodes": torch.as_tensor(frame[node_col].to_numpy(copy=False), dtype=torch.long).contiguous(),
        "node_label": torch.as_tensor(frame[label_col].to_numpy(copy=False), dtype=torch.long).contiguous(),
    }
    if ts_col is not None:
        out["node_label_ts"] = torch.as_tensor(frame[ts_col].to_numpy(copy=False), dtype=torch.float32).contiguous()
    if split_col is not None:
        out["node_label_split"] = torch.as_tensor(frame[split_col].to_numpy(copy=False), dtype=torch.long).clamp(0, 2).to(torch.uint8)
    return out


def _save_tensor(path: Path, value: Tensor | None) -> None:
    if value is not None:
        torch.save(torch.as_tensor(value).cpu().contiguous(), path)


def _detect_kind(source: Path) -> str:
    if source.suffix in {".pt", ".pth"}:
        raise ValueError("auto conversion does not accept preprocessed .pt/.pth inputs; pass a raw edge file or directory")
    return "tgl"


def _remap_nodes(src: Tensor, dst: Tensor) -> tuple[Tensor, Tensor, int]:
    _, inverse = torch.unique(torch.cat((src.long(), dst.long())), sorted=True, return_inverse=True)
    n_src = int(src.numel())
    return inverse[:n_src].long().contiguous(), inverse[n_src:].long().contiguous(), int(inverse.max().item()) + 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert TGL/TGM/DTDG datasets to StarryGL unified graph format.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--kind", choices=("auto", "tgl", "tgm", "dtdg"), default="auto")
    parser.add_argument("--lags", type=int, default=1)
    parser.add_argument("--dataset", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = convert_to_starrygl_format(args.source, args.out, kind=args.kind, lags=args.lags, dataset=args.dataset)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
