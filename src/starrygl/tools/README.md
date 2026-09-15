# StarryGL Tools

This directory contains standalone migration/conversion tools. They are kept
inside `src/starrygl/tools/` so new StarryGL utilities do not get mixed with the
large legacy top-level `tools/` directory.

## Convert To Unified Graph Format

Entry function:

```python
from starrygl.tools.convert_to_starrygl_format import convert_to_starrygl_format

convert_to_starrygl_format(
    source="/path/to/raw/dataset",
    out="/path/to/unified_dataset",
    kind="auto",   # auto | tgl | tgm | dtdg
    lags=1,        # DTDG time_ptr_2 lag over generated snapshots
    dataset=None,  # optional Flare WebDataLoader dataset name
)
```

CLI:

```bash
PYTHONPATH=src python -m starrygl.tools.convert_to_starrygl_format \
  --source /path/to/raw/dataset \
  --out /path/to/unified_dataset \
  --kind auto
```

To convert TGL/TGM/raw web edges into DTDG snapshots, call the same entry with
`kind="dtdg"`. This follows the FlareDTDG `WebDataLoader` split semantics:

```bash
PYTHONPATH=src python -m starrygl.tools.convert_to_starrygl_format \
  --source "${STARRYGL_DATA_ROOT}/rec-amazon-ratings/rec-amazon-ratings.edges" \
  --out "${STARRYGL_DATA_ROOT}/starrygl/rec-amazon-ratings" \
  --kind dtdg \
  --dataset rec-amazon-ratings
```

The currently mirrored WebDataLoader dataset names are:

```text
WikiTalk
ia-slashdot-reply-dir
rec-amazon-ratings
rec-amz-Books
soc-bitcoin
soc-flickr-growth
soc-youtube-growth
stackexch
```

`WikiTalk` uses FlareDTDG's batch split setting. The other web datasets use the
window/lags masked snapshot split. `kind="dtdg"` expects raw edge data; it does
not accept existing FlareDTDG `.pth` snapshot datasets.

DTDG output is not written in FlareDTDG's `dataset: list[dict]` format and does
not store a list of snapshot `edge_index` tensors. The StarryGL directory keeps
one flattened edge array in `graph.pt`; snapshots or lag windows are recovered
only from `graph.pt["time_ptr_2"]`. Node features and labels are written as
tensor sidecars:

```text
node_feat.pt   FloatTensor[num_snapshots, num_nodes, feature_dim]
node_label.pt  FloatTensor[num_snapshots, num_nodes, ...]
```

For Flare-style next-snapshot labels, the last snapshot has no next label and
is filled with `NaN` so the time dimension stays aligned with `node_feat.pt`.

Output directory:

```text
unified_dataset/
  graph.pt
  node_feat.pt     optional
  edge_feat.pt     optional
  node_label.pt    optional
  edge_label.pt    optional
```

`graph.pt` contains:

```python
{
    "src": LongTensor[num_edges],
    "dst": LongTensor[num_edges],
    "ts": FloatTensor[num_edges],
    "edge_ids": LongTensor[num_edges],
    "num_nodes": int,

    # DTDG/snapshot conversion only:
    "time_ptr_2": LongTensor[num_windows, 2],
}
```

TGL/TGM event conversion does not write `time_ptr_2`; event-window generation
belongs to `prepare`. DTDG conversion writes `time_ptr_2` to preserve snapshot
or lag-window ranges. No `snapshot_ptr` and no `meta.json` are written.

The converted directory can be passed directly to the current user-facing entry:

```python
import starrygl as sg

trainer = sg.compile(
    data_source={"source": "/path/to/unified_dataset"},
    backbone={"name": "tgcn", "temporal_representation": "snapshot_sequence"},
    task_segment=sg.NodeRegression(),
    runtime={"preprocess": {"num_parts": 4, "chunks_per_rank": 32}},
)

prepared = trainer.prepare()
```
