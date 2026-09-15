import json

import starrygl as sg


def test_open_compile_and_config_entrypoints_roundtrip(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "data": {"source": "events", "temporal_representation": "event_stream"},
                "backbone": {"name": "tgn", "in_dim": 2, "hidden_dim": 4, "out_dim": 4},
                "task": {"name": "edge_prediction"},
            }
        ),
        encoding="utf-8",
    )

    trainer = sg.from_config(path)

    assert isinstance(trainer.model, sg.TGNModel)
    assert trainer.plan.storage_view == "temporal_sampling_view"
    assert trainer.plan.view.required_layouts == ("event_view", "temporal_csr")

    exported = trainer.to_config()
    assert "ownership" not in exported["task"]
    restored = sg.from_config(exported)

    assert set(exported) == {"data", "backbone", "task", "runtime"}
    assert restored.graph == trainer.graph
    assert restored.model_config == trainer.model_config
    assert restored.task == trainer.task
    assert restored.runtime_config == trainer.runtime_config
    assert restored.train_config == trainer.train_config
    assert restored.preprocess_config == trainer.preprocess_config


def test_from_config_lowers_temporal_state_without_exposing_internal_strategy(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "data": {"source": "events", "temporal_representation": "event_stream"},
                "backbone": {"name": "tgn", "coupling": "coupled"},
                "task": {"name": "edge_prediction"},
                "runtime": {"temporal_state": {"consistency": "bounded_stale", "max_staleness": 2}},
            }
        ),
        encoding="utf-8",
    )

    trainer = sg.from_config(path)
    deps = {dep.kind: dep for dep in trainer.plan.await_dependencies}
    explain = trainer.plan.explain()

    assert trainer.spec.consistency == "bounded_stale"
    assert trainer.spec.max_staleness == 2
    assert trainer.spec.approximation == "none"
    assert deps["node_memory"].freshness_policy == "bounded_stale"
    assert deps["mailbox"].freshness_policy == "bounded_stale"
    assert deps["x"].freshness_policy == "exact"
    assert deps["edge_feat"].freshness_policy == "exact"
    assert deps["endpoint_embedding"].freshness_policy == "exact"
    assert deps["label"].freshness_policy == "exact"
    assert "stale_increment" not in explain
    assert "layerwise" not in explain


def test_from_config_rejects_legacy_semantic_sections() -> None:
    try:
        sg.from_config(
            {"model": {"name": "tgn"}, "task": {"name": "edge_prediction"}, "execution": {"exact": False}}
        )
    except ValueError as exc:
        assert "legacy section name" in str(exc)
    else:
        raise AssertionError("legacy semantic sections were accepted")
