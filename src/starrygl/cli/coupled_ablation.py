"""Convergence experiments on exact or owner-boundary snapshot cache reads."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import random

import numpy as np
import torch
import torch.distributed as dist

import starrygl as sg
from starrygl.runtime.builders import build_task_from_config
from starrygl.runtime.comm import CommScheduler
from starrygl.runtime.loop import run_epoch
from starrygl.runtime.state.build import build_state_managers
from starrygl.runtime.snapshot.layerwise import get_layerwise_profile
from starrygl.runtime.train import _reset_epoch_state
from .experiment_metrics import elapsed_rankmax, global_counters, prepared_workload, synchronized_start


class StateReadAudit:
    """Observe hydrated values without changing reads or adding window collectives."""

    def __init__(self, windows, device, comm, *, hot_nodes=None, num_nodes=0, window_size=1):
        self.counts = torch.zeros(5, dtype=torch.float64, device=device)
        self.histogram = torch.zeros(windows + 1, dtype=torch.float64, device=device)
        self.history_counts = torch.zeros(6, dtype=torch.float64, device=device)
        self.slot_counts = torch.zeros(window_size, 3, dtype=torch.float64, device=device)
        self.slot_ages = torch.zeros(window_size, windows + 1, dtype=torch.float64, device=device)
        self.hot = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        if hot_nodes is not None:
            self.hot[hot_nodes.to(device)] = True
        self.comm = comm

    def reset(self):
        self.counts.zero_()
        self.histogram.zero_()
        self.history_counts.zero_()
        self.slot_counts.zero_()
        self.slot_ages.zero_()

    @torch.no_grad()
    def __call__(self, batch):
        timestamps = batch.state["neighbor_recurrent_ts"]
        mask = batch.state.get("neighbor_recurrent_shared_mask", torch.zeros_like(timestamps, dtype=torch.bool))
        block = tuple(batch.iter_blocks())[-1][-1]
        # State produced by snapshot s commits version s+1; initial state is 0.
        lag = int(block.cache["snapshot_id"]) - timestamps
        self.counts += torch.stack((timestamps.new_tensor(timestamps.numel()), mask.sum(),
                                    (mask & (timestamps == 0)).sum(), (lag < 0).sum(),
                                    ((~mask) & (lag > 0)).sum())).to(self.counts)
        ages = lag[mask].clamp(0, self.histogram.numel() - 1).long()
        self.histogram.index_add_(0, ages, torch.ones_like(ages, dtype=self.histogram.dtype))
        packets = batch.state.get("neighbor_recurrent_snapshots")
        if packets is not None:
            hot = mask & self.hot[block.src_nodes]
            cold = mask & ~self.hot[block.src_nodes]
            dim = (packets[-1].shape[1] - 2) // 2
            nonzero = packets[-1][:, dim:2 * dim].abs().sum(1) > 0
            self.history_counts += torch.stack((hot.sum(), cold.sum(), lag[hot].sum(),
                lag[cold].sum(), (mask & (lag > 0) & nonzero).sum(),
                timestamps.new_tensor(timestamps.numel()))).to(self.history_counts)
        if packets is not None:
            windows = tuple(batch.iter_blocks())
            offset = self.slot_counts.shape[0] - len(windows)
            for slot, (blocks, packet, cold_rows) in enumerate(zip(
                    windows, packets, batch.state["neighbor_recurrent_cold_src_rows"])):
                packet = packet[cold_rows]
                dim = (packet.shape[1] - 2) // 2
                age = int(blocks[-1].cache["snapshot_id"]) - packet[:, -1]
                nonzero = packet[:, dim:2 * dim].abs().sum(1) > 0
                self.slot_counts[offset + slot] += torch.stack((
                    (packet[:, -1] == 0).sum(), (age < 0).sum(),
                    ((age > 0) & nonzero).sum())).to(self.slot_counts)
                ages = age.clamp(0, self.slot_ages.shape[-1] - 1).long()
                self.slot_ages[offset + slot].index_add_(0, ages, torch.ones_like(ages, dtype=torch.float64))

    def report(self):
        packed = torch.cat((self.counts, self.histogram, self.history_counts,
                            self.slot_counts.flatten(), self.slot_ages.flatten()))
        if dist.is_initialized():
            self.comm.all_reduce(packed, name="state_read_audit")
        rows, shared, initial, future, owner_stale = packed[:5].tolist()
        hist = packed[5:5 + self.histogram.numel()].tolist()
        end = 5 + self.histogram.numel()
        hot, cold, hot_lag, cold_lag, increment_rows, history_rows = packed[end:end + 6].tolist()
        slot_counts = packed[end + 6:end + 6 + self.slot_counts.numel()].view_as(self.slot_counts).tolist()
        slot_ages = packed[-self.slot_ages.numel():].view_as(self.slot_ages).tolist()
        slots = [dict(slot=slot, rows=int(sum(slot_hist)), initial_rows=int(slot_initial),
            future_rows=int(slot_future), nonzero_extrapolation_rows=int(nonzero),
            mean_lag=sum(i*n for i, n in enumerate(slot_hist)) / max(1, sum(slot_hist)),
            lag_histogram={str(i): int(n) for i, n in enumerate(slot_hist) if n})
            for slot, ((slot_initial, slot_future, nonzero), slot_hist) in enumerate(zip(slot_counts, slot_ages))]
        return {"read_rows": int(rows), "shared_rows": int(shared),
                "cold_history_inputs": slots,
                "shared_fraction": shared / max(1, rows), "shared_initial_rows": int(initial),
                "future_rows": int(future), "owner_stale_rows": int(owner_stale),
                "shared_lag_histogram": {str(i): int(n) for i, n in enumerate(hist) if n},
                "shared_mean_lag": sum(i*n for i, n in enumerate(hist)) / max(1, shared),
                "history_rows": int(history_rows), "history_remote_hot_rows": int(hot),
                "history_remote_cold_rows": int(cold),
                "history_hot_mean_lag": hot_lag / max(1, hot),
                "history_cold_mean_lag": cold_lag / max(1, cold),
                "stale_remote_nonzero_increment_rows": int(increment_rows)}


def main(argv=None):
    run_start = time.perf_counter()
    started_at = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--config", type=Path, help="regular StarryGL config for component experiments")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("dcrnn", "gconv_gru"), default="gconv_gru")
    parser.add_argument("--policy", choices=("exact", "bounded_stale"), default="bounded_stale")
    parser.add_argument("--max-staleness", type=int, default=1)
    parser.add_argument("--hot-ratio", type=float, default=0.1)
    parser.add_argument("--boundary-prediction", choices=("cache", "mean", "learnable"), default="learnable")
    parser.add_argument("--alpha", type=float, default=0.3, help="publish when 1-cosine_similarity exceeds alpha")
    parser.add_argument("--gamma-boundary-init", type=float, default=2.1972245773362196, help="boundary increment logit; sigmoid(logit) defaults to 0.9")
    parser.add_argument("--no-audit", action="store_true", help="exclude state-read and launch profiling from clean timing")
    parser.add_argument("--skip-test", action="store_true", help="validation-only exploratory run; leave test set untouched")
    parser.add_argument("--train-only", action="store_true", help="training timing only; no checkpoint selection or evaluation")
    parser.add_argument("--num-full-snapshots", type=int, default=1)
    parser.add_argument("--access-pipeline", action="store_true")
    parser.add_argument("--no-rolling-snapshot-cache", action="store_true", help="disable static sliding-window graph reuse for a matched layout ablation")
    parser.add_argument("--world-size", type=int, default=2, help="partition count for prepare-only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    os.environ["STARRYGL_PROFILE_LAYERWISE"] = "0" if args.no_audit else "1"
    if min(args.epochs, args.eval_every, args.hidden_dim, args.world_size, args.max_staleness,
           args.num_full_snapshots) < 1:
        parser.error("counts must be positive")
    if args.data is None and args.config is None:
        parser.error("--data or --config is required")
    if not 0 <= args.hot_ratio <= 1:
        parser.error("hot-ratio must be in [0,1]")
    if not 0 <= args.alpha <= 2 or not torch.isfinite(torch.tensor(args.gamma_boundary_init)):
        parser.error("alpha must be in [0,2] and gamma-init must be finite")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    world = args.world_size if args.prepare_only else int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    device = torch.device(args.device)
    if not args.prepare_only:
        if device.type == "cuda":
            device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
            torch.cuda.set_device(device)
        if world > 1 and not dist.is_initialized():
            dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
        if args.config is None and args.policy == "bounded_stale" and world < 2:
            parser.error("actual owner-boundary reads require at least two torchrun ranks")
    temporal_state = {"consistency": args.policy,
                      "max_staleness": args.max_staleness if args.policy == "bounded_stale" else 0,
                      "filter": {"enabled": True, "min_cosine_distance": args.alpha, "max_skip": args.max_staleness},
                      "boundary_prediction": {"learnable": args.boundary_prediction == "learnable", "gamma_init": args.gamma_boundary_init}}
    config = {
        "data_source": {"source": str(args.data.resolve()) if args.data else "", "temporal_representation": "snapshot_sequence"},
        "backbone": {"name": args.model, "in_dim": 2, "hidden_dim": args.hidden_dim, "out_dim": 1,
                     "spatial_aggregation": "full_neighbor", "state_kind": "neighbor_recurrent",
                     "state_extrapolation": args.boundary_prediction != "cache"},
        "task_segment": {"name": "node_regression", "loss": "mse"},
        "runtime": {"temporal_state": temporal_state,
                    "sampling": {"mode": "full", "window": {"policy": "full_snapshot", "num_full_snapshots": args.num_full_snapshots}},
                    "preprocess": {"num_parts": world, "chunks_per_rank": 1, "hot_node_ratio": args.hot_ratio,
                                   "split_ratios": [0.4, 0.2, 0.4], "include_static_one_hop": True}},
    }
    trainer = sg.from_config(args.config) if args.config else sg.compile(**config)
    if args.config:
        config = json.loads(args.config.read_text())
        args.model = trainer.model_config["name"]
        args.num_full_snapshots = trainer._num_full_snapshots(None)
        temporal_state = trainer.runtime_config.get("temporal_state", {"consistency": "exact"})
        args.policy = temporal_state.get("consistency", "exact")
    coupled = any(dep.kind == "neighbor_recurrent" for dep in trainer.plan.state_dependencies)
    trainer.artifact_root = args.artifact_root
    if args.prepare_only:
        if rank == 0:
            trainer.prepare_artifacts(world_size=world)
            print(json.dumps({"prepared": str(args.artifact_root), "world_size": world}), flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "epochs.jsonl"
    if metrics_path.exists():
        raise FileExistsError(f"refusing to overwrite an existing run: {metrics_path}")
    store = trainer._store(None, artifact_root=args.artifact_root, rank=rank, map_location="cpu", mmap=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    model = trainer._model(None, store).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    comm = CommScheduler()
    managers = trainer._state_manager(None, model=model, store=store, device=device, comm=comm) or {}
    exact = (build_state_managers(model=model, store=store, kinds=("neighbor_recurrent",),
                                 temporal_state={"consistency": "exact"}, device=device, comm=comm)
             if coupled and args.policy != "exact" and args.num_full_snapshots == 1 else managers)
    audit = StateReadAudit(len(store.graph.time_ptr_2), device, comm,
                          hot_nodes=None, num_nodes=store.graph.num_nodes,
                          window_size=args.num_full_snapshots)
    gamma_grad = torch.zeros(2, device=device)
    if getattr(model, "gamma_boundary", None) is not None:
        def record_gamma_grad(grad):
            gamma_grad.add_(torch.stack((grad.detach().abs().sum(), grad.new_ones(()))))
        model.gamma_boundary.register_hook(record_gamma_grad)
    options = trainer._sampler_options(None, train=True, split="train") if args.config else {
        "access_pipeline": args.access_pipeline, "snapshot_dgl_gcn": True,
        "rolling_snapshot_cache": not args.no_rolling_snapshot_cache, "snapshot_reverse_direction": False}
    options["progress_every"] = 0 if args.no_audit else 5
    common = dict(store=store, model=model, task=build_task_from_config(trainer.task),
                  mode=trainer._batch_mode(None), window_policy=trainer._window_policy(None),
                  sampling_policy=trainer._sampling_policy(None), chunk_decay=trainer._chunk_decay(None),
                  fanouts=trainer._fanouts(None), num_negatives=trainer._num_negatives(None),
                  num_full_snapshots=args.num_full_snapshots, num_layers=trainer._num_layers(None), device=device, comm=comm,
                  gradient_sync="all_reduce" if world > 1 else None, batch_callback=None if args.no_audit else audit,
                  sampler_options=options)
    if not coupled:
        common["batch_callback"] = None
    workload = prepared_workload(store, device, comm)
    source_root = Path(sg.__file__).parent
    if rank == 0:
        hashes = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(source_root.rglob("*.py"))}
        manifest = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "compile": config, "starrygl_source": str(source_root), "source_hashes": hashes,
                    "num_nodes": store.graph.num_nodes, "num_edges": store.graph.num_edges,
                    "hot_nodes": int(store.graph.partition["hot_node_ids"].numel()), "world_size": world,
                    "split_windows": {k: len(v) for k, v in store.graph.split_time_ptr_2.items()},
                    "validation_policy": "actual policy replay; select by validation only",
                    "exact_checkpoint_replay": args.policy == "exact" or args.num_full_snapshots == 1,
                    "state_policy": "current StateManager/AsyncMemoryCommitter, no injected delay",
                    "started_at_unix": started_at,
                    "history_window_size": args.num_full_snapshots,
                    "rolling_snapshot_graph_reuse": not args.no_rolling_snapshot_cache,
                    "boundary_prediction_formula": "remote: cache+beta*age*mean; beta=0/1/sigmoid(gamma_boundary)",
                    "increment_term_enabled": args.boundary_prediction != "cache",
                    "snapshot_compute": "owner_only; no hot all_gather or local/shared blend",
                    "audit_enabled": not args.no_audit,
                    "alpha_semantics": "cosine distance, not cosine similarity",
                    "slot_audit_semantics": "all remote boundary targets=snapshot_id; oldest-to-newest, short windows right-aligned",
                    "shared_mask_semantics": "nonowner boundary history rows only",
                    "max_staleness_semantics": "read age<=K for consecutive complete latest snapshots after batch push completion"}
        manifest.update(workload=workload, resolved_sampler_options=options,
                        timing="barrier then CUDA-complete epoch wall time, MAX over all ranks",
                        evaluation_negative_seed=20260915,
                        layerwise_disabled=os.environ.get("STARRYGL_DISABLE_LAYERWISE_DAG", "0"),
                        library_versions={"torch": torch.__version__, "numpy": np.__version__})
        if not coupled:
            for key in ("boundary_prediction_formula", "increment_term_enabled", "snapshot_compute",
                        "alpha_semantics", "slot_audit_semantics", "shared_mask_semantics",
                        "max_staleness_semantics", "exact_checkpoint_replay", "state_policy"):
                manifest.pop(key)
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (args.output / "plan.txt").write_text(str(trainer.plan.explain()) + "\n")

    def replay_coupled(split, state):
        _reset_epoch_state(state, model)
        for earlier in ("train", "val", "test"):
            if earlier == split:
                break
            run_epoch(**common, state_manager=state, training=False, split=earlier,
                      commit_state=True, compute_metrics=False)
        audit.reset()
        result = run_epoch(**common, state_manager=state, training=False, split=split, commit_state=True)
        return result, {} if args.no_audit else audit.report()

    def score(split, state):
        # Evaluation does not consume training dropout/chunk/negative RNG state.
        devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(20260915)
            if coupled:
                return replay_coupled(split, state)
            result = trainer.evaluate(store=store, model=model, state_manager=state,
                split=split, device=device, comm=comm,
                sampler_options={"seed": 20260915,
                    "policy": trainer._sampling_neighbor().get("policy", "recent"), "probability": 1.0},
                generator=torch.Generator().manual_seed(20260915))
            return result, {}

    best_mse, best_epoch = float("inf"), 0
    setup_seconds = time.perf_counter() - run_start
    cumulative_train = cumulative_eval = 0.0
    checkpoint_path = args.output / f"best_rank{rank}.pt"
    for epoch in range(1, args.epochs + 1):
        _reset_epoch_state(managers, model)
        audit.reset()
        gamma_grad.zero_()
        runtime = managers.get("neighbor_recurrent")
        if hasattr(runtime, "_profile_counters"):
            runtime._profile_counters.clear()
        get_layerwise_profile(reset=True)
        start = synchronized_start(device, comm)
        train = run_epoch(**common, state_manager=managers, training=True, split="train", optimizer=optimizer,
                          generator=torch.Generator().manual_seed(args.seed * 10000 + epoch))
        elapsed = elapsed_rankmax(start, device, comm)
        cumulative_train += elapsed
        row = {"epoch": epoch, "train_mse": train.loss, "train_steps": train.steps,
               "train_seconds": elapsed, "train_seconds_cumulative": cumulative_train,
               "state_reads": {} if args.no_audit or not coupled else audit.report(), "layerwise_profile_rank0": get_layerwise_profile(),
               "communication_counts_rank0": runtime.profile_counters() if hasattr(runtime, "profile_counters") else {}}
        row.update(train_metrics=dict(train.metrics), train_loss=train.loss,
                   train_targets_per_second=workload["train_prepared_targets"] / elapsed,
                   train_windows_per_second=workload["train_windows"] / elapsed,
                   communication_counts_global=global_counters(runtime, device, comm))
        if not coupled and trainer.task["name"] == "edge_prediction":
            del row["train_mse"]
        history = getattr(runtime, "snapshot_history", None)
        if history is not None:
            row["history_observations_rank0"] = int(history.packets[:, :, -2].amax(0).sum().item())
        if getattr(model, "gamma_boundary", None) is not None:
            row["gamma_boundary"] = float(model.gamma_boundary.detach())
            row["gamma_boundary_effective"] = float(torch.sigmoid(model.gamma_boundary.detach()))
            row["gamma_boundary_gradient_abs_sum_rank0"], row["gamma_boundary_gradient_calls_rank0"] = gamma_grad.tolist()
        if not torch.isfinite(torch.tensor(train.loss)):
            raise RuntimeError(f"nonfinite training loss at epoch {epoch}")
        if not args.train_only and (epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs):
            start = synchronized_start(device, comm)
            val, val_reads = score("val", managers)
            elapsed = elapsed_rankmax(start, device, comm)
            cumulative_eval += elapsed
            row.update(val_mse=val.loss, val_steps=val.steps, val_state_reads=val_reads, eval_seconds=elapsed)
            row.update(val_loss=val.loss, val_metrics=dict(val.metrics))
            if trainer.task["name"] == "edge_prediction":
                del row["val_mse"]
            if val.steps == 0 or not torch.isfinite(torch.tensor(val.loss)):
                raise RuntimeError("validation must have finite, nonempty supervision")
            if val.loss < best_mse:
                best_mse, best_epoch = val.loss, epoch
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_mse": best_mse}, checkpoint_path)
        row.update(best_val_mse=None if args.train_only else best_mse, best_epoch=best_epoch, eval_seconds_cumulative=cumulative_eval,
                   run_seconds_cumulative=time.perf_counter() - run_start)
        if trainer.task["name"] == "edge_prediction":
            row["best_val_loss"] = row.pop("best_val_mse")
        if rank == 0:
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    if not args.train_only:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False)["model"])
    start = synchronized_start(device, comm)
    if args.skip_test or args.train_only:
        operational = test = None
        operational_reads = exact_reads = {}
    else:
        operational, operational_reads = score("test", managers)
        if coupled and args.policy != "exact" and args.num_full_snapshots == 1:
            test, exact_reads = score("test", exact)
        else:
            test, exact_reads = operational, operational_reads
    test_seconds = elapsed_rankmax(start, device, comm)
    peak_memory = torch.tensor(
        [torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)]
        if device.type == "cuda" else [0, 0], device=device, dtype=torch.long)
    if dist.is_initialized():
        comm.all_reduce(peak_memory, op=dist.ReduceOp.MAX, name="experiment_peak_memory")
    if rank == 0:
        result = {"best_epoch": best_epoch, "best_val_mse": None if args.train_only else best_mse,
                  "test_mse_exact": test.loss if test is not None and (args.policy == "exact" or args.num_full_snapshots == 1) else None,
                  "test_mse_operational": None if operational is None else operational.loss,
                  "test_steps": 0 if test is None else test.steps, "train_seconds": cumulative_train,
                  "test_skipped": args.skip_test or args.train_only,
                  "validation_seconds": cumulative_eval, "test_seconds": test_seconds,
                  "setup_seconds": setup_seconds, "run_seconds_total": time.perf_counter() - run_start,
                  "run_seconds_to_last_epoch": row["run_seconds_cumulative"],
                  "peak_allocated_bytes_rankmax": int(peak_memory[0]),
                  "peak_reserved_bytes_rankmax": int(peak_memory[1]),
                  "test_state_reads": operational_reads,
                  "exact_test_state_reads": exact_reads if args.policy == "exact" or args.num_full_snapshots == 1 else None}
        result.update(test_loss=None if operational is None else operational.loss,
                      test_metrics={} if operational is None else dict(operational.metrics),
                      workload=workload, checkpoint_selection="minimum validation loss")
        if trainer.task["name"] == "edge_prediction":
            result["best_val_loss"] = result.pop("best_val_mse")
            result.pop("test_mse_exact")
            result.pop("test_mse_operational")
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
