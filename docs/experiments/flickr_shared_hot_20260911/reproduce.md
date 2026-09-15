# Reproduce the Flickr experiment

The experiment uses the working tree at `/home/zlj/starrygl-undate/starrygl-open`.
There is no usable Git checkout metadata; the 103 production Python modules are
identified by per-file SHA256 in every run manifest and archived in
`source_snapshot.zip`. `environment.json` records the software and four A40 GPUs.
Historical results are not used.

Run from the package directory with the `tgnn_3.10` environment and `PYTHONPATH=src`.
The original conversion entry point is:

```bash
python -m starrygl.tools.convert_to_starrygl_format \
  --source /mnt/data/zlj/starrygl-data/raw/soc-flickr-growth/soc-flickr-growth.edges \
  --out /tmp/starrygl_open_flickr_ablation_20260911/data \
  --kind dtdg --dataset soc-flickr-growth
```

The dataset-specific converter uses its existing sliding windows (100 temporal
bins, 30-bin lookback, first unique time skipped), producing 70 snapshots. Inputs
are current-window in/out degree. Labels are next-window `log(1 + in_degree)`.
This is temporal Flickr node regression, not the unrelated static Flickr
classification task. Nodes are remapped once across the complete input graph.

Prepare each model using the package CLI; DCRNN needs bidirectional layouts:

```bash
python -m starrygl.cli.coupled_ablation \
  --data /tmp/starrygl_open_flickr_ablation_20260911/data \
  --artifact-root /tmp/starrygl_open_flickr_ablation_20260911/prepared_dcrnn_2rank \
  --output /tmp/flickr_prepare_unused \
  --model dcrnn --world-size 2 --prepare-only
```

For GConvGRU, substitute `--model gconv_gru` and its separate artifact root.
Prepared tensors occupy about 30 GiB for DCRNN and 20 GiB for GConvGRU. Existing
prepared artifacts are reusable. In this run GConvGRU artifacts reside under
`/home/zlj/starrygl-undate/.experiment_artifacts/flickr_gconv_gru_2rank`, with the
`prepared_gconv_gru_2rank` path above implemented as a symlink.

The exact DCRNN command is:

```bash
env CUDA_VISIBLE_DEVICES=0,1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src \
  torchrun --standalone --nproc_per_node=2 -m starrygl.cli.coupled_ablation \
  --data /tmp/starrygl_open_flickr_ablation_20260911/data \
  --artifact-root /tmp/starrygl_open_flickr_ablation_20260911/prepared_dcrnn_2rank \
  --output /tmp/flickr_reproduction/dcrnn_exact_s42 \
  --model dcrnn --policy exact --epochs 100 --eval-every 5
```

Use a fresh output directory for every run; the CLI refuses to overwrite an
existing `epochs.jsonl`. The main arms use these substitutions:

| Model | Policy | Compensation | GPUs in the original run |
|---|---|---|---|
| dcrnn | exact | off | 0,1 |
| dcrnn | bounded_stale | off | 2,3 |
| gconv_gru | exact | off | 0,1 |
| gconv_gru | bounded_stale | `--compensate` | 2,3 |

Defaults shared by all arms: seed 42, hidden width 8, Adam lr 0.001, hot-node
ratio 0.1, max_staleness 1, cosine refresh threshold 0.3, full snapshots,
access_pipeline disabled. All nodes have owner-only output/loss responsibility.
The GConvGRU compensation flag produces zero increment updates in this layout;
the two additional switch-check arms were stopped after their first four losses
matched exactly and are marked as diagnostics.

Training uses 28 windows; validation and test use 14 and 28. The last window of
each split advances state but cannot predict across a split boundary, giving
27/13/27 supervised windows. Every validation/test replay resets state and warms
all earlier splits with the evaluated weights. Best checkpoint selection uses
only validation under the operational read policy. Approximate checkpoints are
tested with both the operational policy and exact state reads.

The main jobs ran concurrently; their recorded time is not an isolated hardware
benchmark. `timing.csv` compares equal epoch budgets. `time_to_target.csv` uses
first observed validation threshold crossings. Setup, checkpoint writes, final
testing and training state resets are outside the cumulative train/eval timer
sums. A performance attribution experiment would additionally need isolated,
order-balanced execution on the same devices and communication profiling. GConvGRU shared timing also includes the enabled
compensation path even though its increment is zero; the two initial diagnostic
jobs briefly shared these devices as well.

Validation artifacts: `end_to_end_parity.json`, the per-run read-age audits in
`raw/`, and the source hashes. The implementation check was 272 tests passed,
15 skipped; two-rank DCRNN output/gradient and serialized-layout parity also
passed. Default-suite skips do not replace the separate distributed checks.
