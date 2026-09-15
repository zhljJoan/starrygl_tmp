from __future__ import annotations

from importlib import import_module


class _Trainer:
    def __init__(self) -> None:
        self.calls = []

    def fit_distributed(self, **kwargs) -> None:
        self.calls.append(("fit", kwargs))

    def eval_distributed(self, **kwargs) -> None:
        self.calls.append(("eval", kwargs))


def test_torchrun_cli_uses_distributed_trainer_methods(monkeypatch) -> None:
    cli = import_module("starrygl.cli.main")
    trainer = _Trainer()
    monkeypatch.setattr(cli, "from_config", lambda *args, **kwargs: trainer)
    monkeypatch.setenv("WORLD_SIZE", "4")

    assert cli.main(["config.json", "--artifact-root", "artifacts"]) == 0
    assert trainer.calls == [
        ("fit", {"prepare": False, "auto_prepare": True, "shutdown": False}),
        ("eval", {"prepare": False, "auto_prepare": False, "shutdown": True}),
    ]


def test_torchrun_cli_respects_skip_prepare_and_skip_evaluate(monkeypatch) -> None:
    cli = import_module("starrygl.cli.main")
    trainer = _Trainer()
    monkeypatch.setattr(cli, "from_config", lambda *args, **kwargs: trainer)
    monkeypatch.setenv("WORLD_SIZE", "2")

    assert cli.main(["config.json", "--skip-prepare", "--skip-evaluate"]) == 0
    assert trainer.calls == [
        ("fit", {"prepare": False, "auto_prepare": False, "shutdown": True}),
    ]
