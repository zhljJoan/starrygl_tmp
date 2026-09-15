# Open Source Layout

`src/starrygl` has one implementation path for each responsibility. Removed
transition namespaces are not compatibility APIs.

| Directory | Responsibility |
| --- | --- |
| `api.py`, `spec.py`, `plan.py` | Public declarations, config loading, semantic lowering, and plan observability |
| `partition/`, `prepare/` | Ownership planning and Event/TemporalCSR/SnapshotCSC artifact construction |
| `store/`, `view/`, `batch/` | Stored tensors, graph views, and the unified model-facing `Batch` |
| `runtime/event/`, `runtime/snapshot/` | Mode-specific window planning, targets, features, and graph materialization |
| `runtime/sample/` | Native sampling, `GraphBlock` construction, bounded loader queues, and Fetch/Await pipeline |
| `runtime/state/`, `runtime/memory/` | Generic state lifecycle; concrete memory/mailbox, shared-hot, and historical-cache policy |
| remaining `runtime/` modules | Shared communication, epoch loop, trainer, and config lowering |
| `model/` | `StarryModel`, shared layers, and the supported model implementations |
| `task/` | Targets, negative sampling, loss, and metrics |
| `native/` | Native temporal sampling and compact sampler outputs |
| `cli/` | Command entry point; dataset loading belongs to `prepare/` |
| `utils/` | Distributed process context and shared tensor routing utilities |

The public entry is:

```python
trainer = sg.compile(
    data_source=sg.DataSource(...),
    backbone=sg.ModelBackbone(...),
    task_segment=sg.TaskSegment(...),
    runtime={...},
)
```

`trainer.plan` is the global semantic-to-physical plan. Runtime iterates the
prepared global `time_ptr_2` row range directly; it does not create another
per-window plan object.

The tree intentionally has no `interface/`, `executor/`, `model/backbone/`,
`store/core/`, `task/segment/`, or similar re-export hierarchy.
