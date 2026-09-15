from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from starrygl.api import from_config


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    trainer = from_config(args.config, artifact_root=args.artifact_root)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        return _run_distributed(trainer, args, world_size=world_size)
    if not args.skip_prepare:
        trainer.prepare_artifacts(world_size=args.world_size, force=args.force_prepare)
    if not args.skip_train:
        trainer.fit()
    if not args.skip_evaluate:
        trainer.evaluate()
    if args.predict:
        trainer.predict()
    return 0


def _run_distributed(trainer, args: argparse.Namespace, *, world_size: int) -> int:
    if args.world_size is not None and int(args.world_size) != int(world_size):
        raise ValueError("--world-size must match torchrun WORLD_SIZE")
    if args.predict:
        raise NotImplementedError("distributed CLI prediction is not implemented")
    run_train = not args.skip_train
    run_eval = not args.skip_evaluate
    if not run_train and not run_eval:
        raise ValueError("distributed CLI requires training or evaluation")

    auto_prepare = not args.skip_prepare
    force_prepare = bool(args.force_prepare and auto_prepare)
    if run_train:
        trainer.fit_distributed(
            prepare=force_prepare,
            auto_prepare=auto_prepare,
            shutdown=not run_eval,
        )
        auto_prepare = False
        force_prepare = False
    if run_eval:
        trainer.eval_distributed(
            prepare=force_prepare,
            auto_prepare=auto_prepare,
            shutdown=True,
        )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="starrygl")
    parser.add_argument("config", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--predict", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
