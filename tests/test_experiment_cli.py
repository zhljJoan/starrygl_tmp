import json
import os
import time

import pytest
import torch
import torch.distributed as dist

from starrygl.cli.coupled_ablation import main
from starrygl.cli.experiment_metrics import elapsed_rankmax, synchronized_start
from starrygl.runtime.comm import CommScheduler


@pytest.mark.parametrize("model", ["tgcn", "tgat"])
def test_config_experiment_records_quality_and_workload(tmp_path, monkeypatch, model):
    data = tmp_path / "data"
    data.mkdir()
    snapshot = model == "tgcn"
    torch.save(dict(src=torch.tensor([0, 1, 2]).repeat(12),
                    dst=torch.tensor([1, 2, 0]).repeat(12),
                    ts=torch.arange(12).repeat_interleave(3).float(),
                    edge_ids=torch.arange(36), num_nodes=3,
                    time_ptr_2=torch.arange(12).unsqueeze(1)*3 + torch.tensor([0, 3]),
                    node_label_horizon=1 if snapshot else 0), data / "graph.pt")
    torch.save(torch.ones(12, 3, 2) if snapshot else torch.ones(3, 2), data / "node_feat.pt")
    if snapshot:
        torch.save(torch.ones(12, 3), data / "node_label.pt")
    config = {
        "data": {"source": str(data), "temporal_representation": "snapshot_sequence" if snapshot else "event_stream"},
        "backbone": {"name": model, "in_dim": 2, "hidden_dim": 2, "out_dim": 1,
                     "num_layers": 2, "spatial_aggregation": "full_neighbor" if snapshot else "sampled_neighbor"},
        "task": {"name": "node_regression" if snapshot else "edge_prediction"},
        "runtime": {"temporal_state": {"consistency": "exact"}, "train_compute_metrics": False,
            "sampling": {"mode": "full" if snapshot else "neighbor",
                "window": {"policy": "full_snapshot" if snapshot else "event_window",
                           "num_full_snapshots": 3 if snapshot else 1},
                "neighbor": {"fanouts": [2, 2], "workers": 1, "policy": "uniform", "seed": 42}},
            "preprocess": {"num_parts": 1, "chunks_per_rank": 1, "split_ratios": [0.4, 0.2, 0.4]}}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    output = tmp_path / "run"
    args = ["--config", str(path), "--artifact-root", str(tmp_path / "prepared"),
            "--output", str(output), "--world-size", "1", "--epochs", "1",
            "--device", "cpu", "--no-audit", "--target-batch-size", "5"]
    main(args + ["--prepare-only"])
    main(args)
    result = json.loads((output / "result.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert "increment_term_enabled" not in manifest
    assert manifest["compile"]["runtime"]["preprocess"]["target_batch_size"] == 5
    epoch = json.loads((output / "epochs.jsonl").read_text())
    assert result["workload"]["train_prepared_targets"] > 0
    assert result["test_loss"] >= 0 and result["best_epoch"] == 1
    assert epoch["train_seconds"] > 0 and epoch["train_targets_per_second"] > 0
    assert epoch["train_metrics"] == {}
    if not snapshot:
        assert "ap" in result["test_metrics"] and "train_mse" not in epoch
    repeated = tmp_path / "repeated"
    if snapshot:
        monkeypatch.setenv("STARRYGL_DISABLE_LAYERWISE_DAG", "1")
    main([str(repeated) if value == str(output) else value for value in args])
    first = torch.load(output / "best_rank0.pt", weights_only=False)["model"]
    second = torch.load(repeated / "best_rank0.pt", weights_only=False)["model"]
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 2, reason="two torchrun ranks")
def test_epoch_time_uses_slowest_rank():
    dist.init_process_group("gloo")
    try:
        comm = CommScheduler()
        start = synchronized_start("cpu", comm)
        time.sleep(0.1 if dist.get_rank() else 0.01)
        elapsed = elapsed_rankmax(start, "cpu", comm)
        values = [None, None]
        dist.all_gather_object(values, elapsed)
        assert values[0] == values[1] and elapsed >= 0.09
    finally:
        dist.destroy_process_group()
