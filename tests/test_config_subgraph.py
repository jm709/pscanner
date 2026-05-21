"""Defaults + validation for SubgraphTradeCollectorConfig and SubgraphCopyEvaluatorConfig."""

from __future__ import annotations

from pscanner.config import Config, SubgraphCopyEvaluatorConfig, SubgraphTradeCollectorConfig


def test_subgraph_collector_defaults() -> None:
    cfg = SubgraphTradeCollectorConfig()
    assert cfg.enabled is False
    assert cfg.subgraph_id == "B9mm21DKCex8ka4g8cteQU4NQqtviwmcTjQAYLbzQ1eR"
    assert cfg.poll_interval_seconds == 10.0
    assert cfg.rpm == 60
    assert cfg.page_size == 1000
    assert cfg.cold_start_lookback_seconds == 0
    assert cfg.indexer_lag_warn_seconds == 60
    assert cfg.indexer_lag_error_seconds == 600


def test_subgraph_copy_evaluator_defaults() -> None:
    cfg = SubgraphCopyEvaluatorConfig()
    assert cfg.enabled is False
    assert cfg.position_fraction == 0.005
    assert cfg.min_multiplier == 0.10


def test_root_config_wires_subgraph_sections() -> None:
    root = Config()
    assert isinstance(root.subgraph_trades, SubgraphTradeCollectorConfig)
    assert isinstance(
        root.paper_trading.evaluators.subgraph_copy, SubgraphCopyEvaluatorConfig
    )
