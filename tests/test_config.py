"""Tests for ``pscanner.config`` typed sections."""

from __future__ import annotations

from pathlib import Path

from pscanner.config import (
    ClusterConfig,
    Config,
    EvaluatorsConfig,
    GateModelConfig,
    GateModelEvaluatorConfig,
    GateModelMarketFilterConfig,
    MispricingEvaluatorConfig,
    MoveAttributionConfig,
    MoveAttributionEvaluatorConfig,
    PaperTradingConfig,
    SmartMoneyEvaluatorConfig,
    VelocityEvaluatorConfig,
    WorkerSinkConfig,
)


def test_move_attribution_defaults() -> None:
    cfg = MoveAttributionConfig()
    assert cfg.enabled is True
    assert cfg.trigger_detectors == ("velocity", "convergence")
    assert cfg.lookback_seconds_baseline == 86400
    assert cfg.backwalk_multiplier == 3.0
    assert cfg.backwalk_check_window_seconds == 300
    assert cfg.max_backwalk_seconds == 7200
    assert cfg.burst_bucket_seconds == 60
    assert cfg.min_burst_wallets == 4
    assert cfg.max_burst_size_cv == 0.4
    assert cfg.max_burst_hits_per_alert == 5
    assert cfg.max_contributors_per_burst == 50


def test_move_attribution_attached_to_root_config() -> None:
    cfg = Config()
    assert isinstance(cfg.move_attribution, MoveAttributionConfig)
    assert cfg.move_attribution.enabled is True


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


def test_cluster_max_co_trade_group_size_default() -> None:
    cfg = ClusterConfig()
    assert cfg.max_co_trade_group_size == 100


def test_worker_sink_config_defaults() -> None:
    cfg = WorkerSinkConfig()
    assert cfg.velocity_maxsize == 4096
    assert cfg.stats_interval_seconds == 60

    root = Config()
    assert root.worker_sink == cfg


def test_evaluators_config_defaults() -> None:
    cfg = EvaluatorsConfig()
    assert cfg.smart_money == SmartMoneyEvaluatorConfig()
    assert cfg.move_attribution == MoveAttributionEvaluatorConfig()
    assert cfg.velocity == VelocityEvaluatorConfig()
    assert cfg.mispricing == MispricingEvaluatorConfig()

    sm = SmartMoneyEvaluatorConfig()
    assert sm.enabled is True
    assert sm.position_fraction == 0.01
    assert sm.min_weighted_edge == 0.0

    ma = MoveAttributionEvaluatorConfig()
    assert ma.enabled is True
    assert ma.position_fraction == 0.01
    assert ma.min_severity == "med"
    assert ma.min_wallets == 3

    v = VelocityEvaluatorConfig()
    assert v.enabled is True
    assert v.position_fraction == 0.0025
    assert v.min_severity == "high"
    assert v.allow_consolidation is False

    m = MispricingEvaluatorConfig()
    assert m.enabled is True
    assert m.position_fraction == 0.01
    assert m.min_edge_dollars == 0.05

    root = Config()
    assert root.paper_trading.evaluators == cfg


def test_paper_trading_config_no_longer_has_position_fraction() -> None:
    """The old `position_fraction` and `min_weighted_edge` fields are removed
    from PaperTradingConfig — they live under evaluators.smart_money now."""
    cfg = PaperTradingConfig()
    assert not hasattr(cfg, "position_fraction"), (
        "position_fraction must move to evaluators.smart_money.position_fraction"
    )
    assert not hasattr(cfg, "min_weighted_edge"), (
        "min_weighted_edge must move to evaluators.smart_money.min_weighted_edge"
    )


def test_config_default_monotone_section() -> None:
    """Monotone defaults match the documented values."""
    cfg = Config()
    assert cfg.monotone.enabled is True
    assert cfg.monotone.scan_interval_seconds == 300
    assert cfg.monotone.min_violation == 0.02
    assert cfg.monotone.min_event_liquidity_usd == 10000.0
    assert cfg.monotone.min_market_liquidity_usd == 100.0
    assert cfg.monotone.max_market_count == 12


def test_gate_model_config_defaults() -> None:
    cfg = GateModelConfig()
    assert cfg.enabled is False
    assert cfg.artifact_dir == Path("models/current")
    assert cfg.min_pred == 0.7
    assert cfg.min_edge_pct == 0.01
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


def test_gate_model_evaluator_config_defaults() -> None:
    cfg = GateModelEvaluatorConfig()
    assert cfg.enabled is False
    assert cfg.min_edge_pct == 0.01
    assert cfg.position_fraction == 0.005


def test_evaluators_config_aggregates_gate_model() -> None:
    cfg = EvaluatorsConfig()
    assert isinstance(cfg.gate_model, GateModelEvaluatorConfig)
    assert cfg.gate_model.enabled is False
