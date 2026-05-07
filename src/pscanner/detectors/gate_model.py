"""Gate-model detector (#79).

Scores every observed trade on top-volume open markets in
``accepted_categories`` through a loaded XGBoost gate model. Emits
``gate_buy`` alerts when the model's predicted P(win) clears the
configured floor AND exceeds the implied probability paid AND the
trade's market category is in ``accepted_categories``.

Loads ``model.json`` + ``preprocessor.json`` once at construction. Hot
reload is deferred (v2 — see RFC #77 Q3). The artifact format is the
one written by ``pscanner ml train`` and consumed by
``scripts/analyze_model.py``.

This file ships the skeleton only — pre-screen, scoring, and alert
emission land in plan tasks 4-6.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import structlog
import xgboost as xgb

from pscanner.config import GateModelConfig
from pscanner.detectors.trade_driven import TradeDrivenDetector
from pscanner.ml.preprocessing import OneHotEncoder

if TYPE_CHECKING:
    from pscanner.daemon.live_history import LiveHistoryProvider
    from pscanner.store.repo import AlertsRepo, WalletTrade

_LOG = structlog.get_logger(__name__)


class GateModelDetector(TradeDrivenDetector):
    """Score trades against a loaded XGBoost gate model and emit alerts."""

    name = "gate_model"

    def __init__(
        self,
        *,
        config: GateModelConfig,
        provider: LiveHistoryProvider,
        alerts_repo: AlertsRepo,
    ) -> None:
        """Load model + preprocessor artifacts and configure category filter.

        Args:
            config: GateModelConfig with artifact_dir, min_pred, and optional
                accepted_categories override.
            provider: LiveHistoryProvider for per-wallet and per-market state.
            alerts_repo: AlertsRepo for deduplication and persistence.
        """
        super().__init__()
        self._config = config
        self._provider = provider
        self._alerts_repo = alerts_repo
        artifact_dir = config.artifact_dir
        self._booster = xgb.Booster()
        self._booster.load_model(str(artifact_dir / "model.json"))
        payload = json.loads((artifact_dir / "preprocessor.json").read_text())
        self._encoder = OneHotEncoder.from_json({"levels": payload["encoder"]["levels"]})
        cfg_categories = config.accepted_categories
        if cfg_categories is None:
            cfg_categories = tuple(payload.get("accepted_categories") or ())
        self.accepted_categories: tuple[str, ...] = cfg_categories
        self._model_version = hashlib.sha256(
            (artifact_dir / "model.json").read_bytes()
        ).hexdigest()[:16]
        _LOG.info(
            "gate_model.loaded",
            artifact_dir=str(artifact_dir),
            accepted_categories=list(self.accepted_categories),
            model_version=self._model_version,
        )

    def _should_score(self, trade: WalletTrade) -> bool:
        """Cheap pre-filter that doesn't require model inference.

        Only BUY trades are scored — SELLs are accumulator-only and never
        produce a ``gate_buy`` alert. The YES/NO outcome side check runs
        downstream via :class:`MarketCacheRepo` (Task 7); a missing or
        non-binary asset returns ``""`` from outcome resolution and is
        skipped at the scoring stage.
        """
        return trade.side == "BUY"

    async def evaluate(self, trade: WalletTrade) -> None:
        """Stub for now — scoring + emit land in Tasks 5-6."""
        del trade
