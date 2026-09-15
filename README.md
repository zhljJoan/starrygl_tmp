# StarryGL

StarryGL is a distributed dynamic graph neural network framework organized
around the Spatio-Temporal Chunk (STC) abstraction.

This open-source staging tree is arranged to match the paper method:

- STC indexed data organization
- Multi-view chunk store
- Coupling-aware distributed executor
- Generalized execution interface
- Model backbones and task segments as composable semantics

## Source Layout

```text
src/starrygl/
  api.py       semantic compile and config entry
  spec.py      canonical declaration normalization
  plan.py      observable semantic-to-physical lowering
  partition/   ownership and chunk planning
  prepare/     Event, TemporalCSR, and SnapshotCSC construction
  store/       graph, feature, state, and artifact storage
  view/        Event and Snapshot graph views
  batch/       unified model-facing Batch
  runtime/     sampling, Fetch/Await, communication, and epoch execution
  model/       DGNN backbones
  task/        task segments, supervision, negative sampling, metrics
  native/      compact temporal sampler bridge
  cli/         one command-line entry
  utils/       shared utilities
```

The intended public entry is:

```python
import starrygl as sg

trainer = sg.compile(
    data_source=...,
    backbone=...,
    task_segment=...,
)

print(trainer.plan.explain())
```

Raw DTDG `.edges` files are converted once before training. Bundled Snapshot
configs read `${STARRYGL_DATA_ROOT}/starrygl/<dataset>`; create that directory
with `python -m starrygl.tools.convert_to_starrygl_format` as documented in
`src/starrygl/tools/README.md`.

Legacy migration code from `src/atc_starrygl_lib` is intentionally not copied
into this tree.

## Install

```bash
pip install -e .
bash scripts/build_native.sh
```

The native build ends with a real sampler-constructor smoke check so Python and
the C++ extension cannot silently use different constructor ABIs.
