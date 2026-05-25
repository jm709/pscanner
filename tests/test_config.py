"""Tests for ``pscanner.config`` typed sections."""

from __future__ import annotations

from pathlib import Path

from pscanner.config import (
    Config,
    EvaluatorsConfig,
    GateModelConfig,
    GateModelMarketFilterConfig,
    PaperTradingConfig,
    SubgraphCopyEvaluatorConfig,
)


def test_paper_trading_defaults() -> None:
    cfg = PaperTradingConfig()
    assert cfg.enabled is False  # opt-in
    assert cfg.starting_bankroll_usd == 1000.0
    assert cfg.min_position_cost_usd == 0.50
    assert cfg.resolver_scan_interval_seconds == 300.0


def test_paper_trading_attached_to_root_config() -> None:
    cfg = Config()
    assert isinstance(cfg.paper_trading, PaperTradingConfig)
    assert cfg.paper_trading.enabled is False


def test_evaluators_config_defaults() -> None:
    cfg = EvaluatorsConfig()
    assert cfg.subgraph_copy == SubgraphCopyEvaluatorConfig()

    sc = SubgraphCopyEvaluatorConfig()
    assert sc.enabled is False
    assert sc.position_fraction == 0.005
    assert sc.min_multiplier == 0.10

    root = Config()
    assert root.paper_trading.evaluators == cfg


def test_paper_trading_config_no_longer_has_position_fraction() -> None:
    """The old ``position_fraction`` and ``min_weighted_edge`` fields must
    stay off ``PaperTradingConfig`` — sizing is per-evaluator."""
    cfg = PaperTradingConfig()
    assert not hasattr(cfg, "position_fraction")
    assert not hasattr(cfg, "min_weighted_edge")


def test_gate_model_config_defaults() -> None:
    cfg = GateModelConfig()
    assert cfg.enabled is False
    assert cfg.artifact_dir == Path("models/current")
    assert cfg.min_pred == 0.5
    assert cfg.min_edge_pct == 0.05
    assert cfg.accepted_categories is None
    assert cfg.queue_max_size == 1024
    assert cfg.platform == "polymarket"


def test_gate_model_market_filter_defaults() -> None:
    cfg = GateModelMarketFilterConfig()
    assert cfg.enabled is False
    assert cfg.accepted_categories == ("esports",)
    assert cfg.min_volume_24h_usd == 100_000
    assert cfg.max_markets == 50
    assert cfg.poll_interval_seconds == 60


def test_root_config_aggregates_gate_sections() -> None:
    cfg = Config()
    assert isinstance(cfg.gate_model, GateModelConfig)
    assert isinstance(cfg.gate_model_market_filter, GateModelMarketFilterConfig)
