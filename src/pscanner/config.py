"""Configuration loading for pscanner.

Loads ``./config.toml`` (override via ``PSCANNER_CONFIG`` env var) into a
typed pydantic model. Defaults match ``config.toml.example`` so the daemon
runs out-of-the-box if no config file is present.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_DEFAULT_CONFIG_PATH = Path("./config.toml")
_CONFIG_ENV_VAR = "PSCANNER_CONFIG"


class _Section(BaseModel):
    """Base for config sections — forbids unknown keys to catch typos early."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ScannerConfig(_Section):
    """Top-level scanner runtime settings."""

    db_path: Path = Field(default=Path("./data/pscanner.sqlite3"))
    log_level: str = Field(default="INFO")


class RatelimitConfig(_Section):
    """Per-host request rate limits."""

    gamma_rpm: int = 50
    data_rpm: int = 50


class PositionsConfig(_Section):
    """Cadence + toggles for the position-snapshot collector."""

    enabled: bool = True
    snapshot_interval_seconds: float = 300.0


class ActivityConfig(_Section):
    """Cadence + toggles for the activity-stream collector."""

    enabled: bool = True
    poll_interval_seconds: float = 300.0
    activity_page_limit: int = 200
    max_pages: int = 10
    dup_lookback: int = 50


class MarketsConfig(_Section):
    """Cadence + toggles for the market-snapshot collector."""

    enabled: bool = True
    snapshot_interval_seconds: float = 300.0
    snapshot_max: int = 5000


class EventsConfig(_Section):
    """Cadence + toggles for the event-snapshot collector."""

    enabled: bool = True
    snapshot_interval_seconds: float = 900.0
    snapshot_max: int = 2000


class SubgraphTradeCollectorConfig(_Section):
    """Tunables for the live SubgraphTradeCollector (#152).

    Polls the Polymarket V2 subgraph for trades by watchlisted wallets and
    emits ``subgraph_copy`` alerts. Coexists with the ``/activity``-based
    ``TradeCollector`` — both run independently.
    """

    enabled: bool = False
    subgraph_id: str = "B9mm21DKCex8ka4g8cteQU4NQqtviwmcTjQAYLbzQ1eR"
    poll_interval_seconds: float = 10.0
    rpm: int = 60
    page_size: int = 1000
    cold_start_lookback_seconds: int = 0
    """Seconds before ``now()`` to start from on first daemon boot when
    ``subgraph_watch_state`` is empty. ``0`` ignores history."""
    indexer_lag_warn_seconds: int = 60
    indexer_lag_error_seconds: int = 600


class SubgraphCopyEvaluatorConfig(_Section):
    """Tunables for the subgraph-copy paper-trading evaluator (#152).

    Sizes each copy at ``bankroll * position_fraction * multiplier`` where
    ``multiplier`` decays as a wallet's share of total subgraph_copy trades
    exceeds ``1.0 / active_watchlist_size``, floored at ``min_multiplier``.
    """

    enabled: bool = False
    position_fraction: float = 0.005
    min_multiplier: float = 0.10


class EvaluatorsConfig(_Section):
    """Container for the per-source evaluator configs.

    Disabling a source via its ``enabled`` flag prevents that Evaluator
    from being constructed at scheduler boot — no detector code path
    changes; the alert is simply not handled by anyone.
    """

    subgraph_copy: SubgraphCopyEvaluatorConfig = Field(
        default_factory=SubgraphCopyEvaluatorConfig,
    )


class PaperTradingConfig(_Section):
    """Thresholds + cadence for the paper-trading subsystem.

    Off by default. When enabled, PaperTrader subscribes to AlertSink and
    fans every alert through the evaluators list to mirror trades onto a
    virtual bankroll. PaperResolver runs as a periodic detector that books
    PnL when the underlying market resolves. State lives in ``paper_trades``.

    Per-source tunables (enabled, position_fraction, quality gates) live
    under ``evaluators.<source>``.
    """

    enabled: bool = False
    starting_bankroll_usd: float = 1000.0
    min_position_cost_usd: float = 0.50
    resolver_scan_interval_seconds: float = 300.0
    replay_lookback_seconds: int = 0
    """On boot, replay alerts emitted in the last N seconds that don't yet have
    a paper_trades entry through the evaluator pipeline. ``0`` disables the
    replay (default). Set to e.g. ``900`` (15 minutes) to recover from a
    daemon restart without losing in-flight alerts. See issue #105.
    """
    evaluators: EvaluatorsConfig = Field(default_factory=EvaluatorsConfig)


class Config(BaseModel):
    """Root pscanner config aggregating every section."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scanner: ScannerConfig = Field(default_factory=ScannerConfig)
    ratelimit: RatelimitConfig = Field(default_factory=RatelimitConfig)
    positions: PositionsConfig = Field(default_factory=PositionsConfig)
    activity: ActivityConfig = Field(default_factory=ActivityConfig)
    markets: MarketsConfig = Field(default_factory=MarketsConfig)
    events: EventsConfig = Field(default_factory=EventsConfig)
    subgraph_trades: SubgraphTradeCollectorConfig = Field(
        default_factory=SubgraphTradeCollectorConfig,
    )
    paper_trading: PaperTradingConfig = Field(default_factory=PaperTradingConfig)

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        """Load a Config from disk, falling back to defaults when absent.

        Resolution order: explicit ``path`` arg > ``PSCANNER_CONFIG`` env var >
        ``./config.toml``. A missing file is not an error: the returned Config
        uses the model defaults (which match ``config.toml.example``).

        Args:
            path: Optional explicit path to the TOML config file.

        Returns:
            A fully-validated, frozen ``Config`` instance.

        Raises:
            ValueError: If the file exists but cannot be parsed or validated.
        """
        resolved = _resolve_config_path(path)
        if resolved is None or not resolved.exists():
            return cls()
        try:
            raw = _read_toml(resolved)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            msg = f"failed to read config file at {resolved}: {exc}"
            raise ValueError(msg) from exc
        return cls.model_validate(raw)


def _resolve_config_path(explicit: Path | None) -> Path | None:
    """Resolve the config-file path using the documented precedence."""
    if explicit is not None:
        return explicit
    env_value = os.environ.get(_CONFIG_ENV_VAR)
    if env_value:
        return Path(env_value)
    return _DEFAULT_CONFIG_PATH


def _read_toml(path: Path) -> dict[str, Any]:
    """Read and parse a TOML file from disk."""
    with path.open("rb") as handle:
        return tomllib.load(handle)
