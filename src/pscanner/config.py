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


class SmartMoneyConfig(_Section):
    """Thresholds for the smart-money detector.

    ``category_min_edge`` keys (``thesis``, ``sports``, ``esports``) are the
    minimum ``mean_edge`` values per category applied at READ time when the
    convergence detector queries roster slices. Sports gets a higher bar
    because Vegas-tight markets are harder to beat.
    """

    enabled: bool = True
    leaderboard_top_n: int = 200
    min_resolved_positions: int = 20
    min_edge: float = 0.05
    min_excess_pnl_usd: float = 1000.0
    refresh_interval_seconds: int = 3600
    position_poll_interval_seconds: int = 300
    new_position_min_usd: float = 1000.0
    prewarm_event_tag_cache: bool = True
    category_min_edge: dict[str, float] = Field(
        default_factory=lambda: {"thesis": 0.05, "sports": 0.10, "esports": 0.05},
    )


class MispricingConfig(_Section):
    """Thresholds for the mispricing detector.

    ``alert_max_deviation`` is the upper cap on ``|Σ - 1|`` that still emits an
    alert: events with deviation above this value are silently captured into
    ``event_outcome_sum_history`` (forming a research dataset for high-Σ
    checkbox/multi-outcome layouts) but produce no alert. Alerts fire only when
    ``sum_deviation_threshold < |Σ - 1| <= alert_max_deviation``.

    ``min_market_liquidity_usd`` filters per-market noise: when > 0, any event
    containing a market with ``liquidity`` below this threshold (or NULL) is
    skipped entirely. Defaults to 0.0 (no filter) for backward compatibility.
    """

    enabled: bool = True
    scan_interval_seconds: int = 300
    sum_deviation_threshold: float = 0.03
    alert_max_deviation: float = 0.5
    min_event_liquidity_usd: float = 10000.0
    min_market_liquidity_usd: float = 0.0
    excluded_tags: tuple[str, ...] = ("Sports", "Esports")


class ConvergenceConfig(_Section):
    """Thresholds + window settings for the convergence detector.

    Convergence fires when at least ``convergence_min_wallets`` smart wallets
    in the same category enter the same condition_id within the configured
    window. Window length varies by category: thesis events stay active for
    weeks so we use a 48h window, sports markets resolve in hours so we use
    6h, esports lands in between at 24h.
    """

    enabled: bool = True
    convergence_min_wallets: int = 2
    window_seconds_thesis: int = 48 * 3600
    window_seconds_sports: int = 6 * 3600
    window_seconds_esports: int = 24 * 3600

    def window_seconds_for(self, category: str) -> int:
        """Return the configured window (seconds) for ``category``.

        Falls back to the thesis window for unknown categories. Categories
        not in the {thesis, sports, esports} set should not occur in practice
        but are tolerated to keep the detector resilient to schema drift.
        """
        return {
            "thesis": self.window_seconds_thesis,
            "sports": self.window_seconds_sports,
            "esports": self.window_seconds_esports,
        }.get(category, self.window_seconds_thesis)


class WhalesConfig(_Section):
    """Thresholds for the whale detector."""

    enabled: bool = True
    new_account_max_age_days: int = 30
    new_account_max_trades: int = 50
    small_market_max_liquidity_usd: float = 50000.0
    big_bet_min_pct_of_liquidity: float = 0.05
    big_bet_min_usd: float = 2000.0
    ws_resubscribe_interval_seconds: int = 1800
    subscription_max_markets: int = 2000
    subscription_min_volume_usd: float = 100.0


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


class TicksConfig(_Section):
    """Cadence + scope for the WS-driven market-tick collector.

    The collector subscribes to the union of (assets held by watched wallets)
    and (markets above ``tick_volume_floor_usd``), capped at ``max_assets``,
    and persists one row per asset per ``tick_interval_seconds``.
    """

    enabled: bool = True
    tick_interval_seconds: float = 30.0
    subscription_refresh_seconds: float = 300.0
    tick_volume_floor_usd: float = 10000.0
    max_assets: int = 1000


class VelocityConfig(_Section):
    """Thresholds + cadence for the price-velocity detector.

    Polls the tick collector for recent mid-price history and alerts when
    ``(end - start) / start`` over ``velocity_window_seconds`` exceeds
    ``velocity_threshold_pct`` in either direction.
    """

    enabled: bool = True
    velocity_threshold_pct: float = 0.05
    velocity_window_seconds: int = 60
    poll_interval_seconds: float = 5.0


class Config(BaseModel):
    """Root pscanner config aggregating every section."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scanner: ScannerConfig = Field(default_factory=ScannerConfig)
    smart_money: SmartMoneyConfig = Field(default_factory=SmartMoneyConfig)
    mispricing: MispricingConfig = Field(default_factory=MispricingConfig)
    whales: WhalesConfig = Field(default_factory=WhalesConfig)
    convergence: ConvergenceConfig = Field(default_factory=ConvergenceConfig)
    ratelimit: RatelimitConfig = Field(default_factory=RatelimitConfig)
    positions: PositionsConfig = Field(default_factory=PositionsConfig)
    activity: ActivityConfig = Field(default_factory=ActivityConfig)
    markets: MarketsConfig = Field(default_factory=MarketsConfig)
    events: EventsConfig = Field(default_factory=EventsConfig)
    ticks: TicksConfig = Field(default_factory=TicksConfig)
    velocity: VelocityConfig = Field(default_factory=VelocityConfig)

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
