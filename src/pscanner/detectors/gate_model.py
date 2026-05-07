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
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import time
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import structlog
import xgboost as xgb

from pscanner.alerts.models import Alert
from pscanner.alerts.sink import AlertSink
from pscanner.config import GateModelConfig
from pscanner.corpus.features import (
    FeatureRow,
    Trade,
    compute_features,
)
from pscanner.detectors.trade_driven import TradeDrivenDetector
from pscanner.ml.preprocessing import CARRIER_COLS, OneHotEncoder

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
        self._feature_cols: tuple[str, ...] = self._derive_feature_cols()
        self._model_version = hashlib.sha256(
            (artifact_dir / "model.json").read_bytes()
        ).hexdigest()[:16]
        self._queue: asyncio.Queue[WalletTrade] = asyncio.Queue(
            maxsize=config.queue_max_size,
        )
        _LOG.info(
            "gate_model.loaded",
            artifact_dir=str(artifact_dir),
            accepted_categories=list(self.accepted_categories),
            model_version=self._model_version,
        )

    def _derive_feature_cols(self) -> tuple[str, ...]:
        """Compute the column order the booster was trained on.

        The training pipeline calls
        ``pscanner.ml.streaming._derive_feature_names`` to set the ordering;
        we replicate it here without needing the corpus DB. ``FeatureRow``
        is the schema source of truth.
        """
        levels = self._encoder.levels
        excluded = {*CARRIER_COLS, "label_won"}
        non_cat = [f.name for f in dataclasses.fields(FeatureRow) if f.name not in levels]
        indicators = [f"{col}__{lvl}" for col, lvls in levels.items() for lvl in lvls]
        return tuple(c for c in [*non_cat, *indicators] if c not in excluded)

    def _should_score(self, trade: WalletTrade) -> bool:
        """Cheap pre-filter that doesn't require model inference.

        Only BUY trades are scored — SELLs are accumulator-only and never
        produce a ``gate_buy`` alert. The YES/NO outcome side check runs
        downstream via :class:`MarketCacheRepo` (Task 7); a missing or
        non-binary asset returns ``""`` from outcome resolution and is
        skipped at the scoring stage.
        """
        return trade.side == "BUY"

    def handle_trade_sync(self, trade: WalletTrade) -> None:
        """Enqueue for scoring; drop if queue is full or trade fails pre-screen."""
        if not self._should_score(trade):
            return
        try:
            self._queue.put_nowait(trade)
        except asyncio.QueueFull:
            _LOG.warning(
                "gate_model.queue_full",
                detector=self.name,
                tx=trade.transaction_hash,
                queue_max=self._config.queue_max_size,
            )

    async def run(self, sink: AlertSink) -> None:
        """Worker loop — drains the queue and scores each trade."""
        if self._sink is None:
            self._sink = sink
        while True:
            trade = await self._queue.get()
            try:
                await self.evaluate(trade)
            except Exception:  # one bad trade can't kill the loop
                _LOG.exception(
                    "gate_model.evaluate_failed",
                    tx=trade.transaction_hash,
                )
            finally:
                self._queue.task_done()

    async def evaluate(self, trade: WalletTrade) -> None:
        """Score the trade and emit a gate_buy alert if all gates pass."""
        if not self._should_score(trade):
            return
        outcome_side = self._resolve_outcome_side(trade)
        if outcome_side not in ("YES", "NO"):
            return
        try:
            metadata = self._provider.market_metadata(trade.condition_id)
        except KeyError:
            _LOG.debug(
                "gate_model.no_metadata",
                condition_id=trade.condition_id,
            )
            return
        feature_trade = Trade(
            tx_hash=trade.transaction_hash,
            asset_id=trade.asset_id,
            wallet_address=trade.wallet,
            condition_id=trade.condition_id,
            outcome_side=outcome_side,
            bs="BUY",
            price=trade.price,
            size=trade.size,
            notional_usd=trade.usd_value,
            ts=trade.timestamp,
            category=metadata.category,
        )
        features = compute_features(feature_trade, self._provider)
        if self.accepted_categories and features.market_category not in self.accepted_categories:
            return
        pred = self._predict_one(features)
        implied = features.implied_prob_at_buy
        edge = pred - implied
        if pred < self._config.min_pred:
            return
        if edge < self._config.min_edge_pct:
            return
        # Fold trade into provider state AFTER feature compute (parity with
        # offline build-features ordering).
        self._provider.observe(feature_trade)
        await self._emit_alert(trade, features, pred=pred, edge=edge)

    def _resolve_outcome_side(self, trade: WalletTrade) -> str:
        """Map ``WalletTrade.asset_id`` -> ``"YES"`` / ``"NO"``.

        Stub for now — Task 7 wires :class:`MarketCacheRepo`. Tests
        monkeypatch this method to return a fixed value.
        """
        del trade
        return ""

    def _predict_one(self, features: FeatureRow) -> float:
        """Run the booster on a single feature row.

        Builds a 1-row Polars DataFrame, applies the encoder, projects to
        the trained column order, and predicts. Tests monkeypatch this for
        deterministic outputs.
        """
        row_dict = dataclasses.asdict(features)
        df = pl.DataFrame([row_dict])
        encoded = self._encoder.transform(df)
        # Add any missing one-hot columns as zeros (unseen categorical level
        # at train time). Reorder to match the trained column ordering.
        present = set(encoded.columns)
        for col in self._feature_cols:
            if col not in present:
                encoded = encoded.with_columns(pl.lit(0).cast(pl.Int8).alias(col))
        x = encoded.select(list(self._feature_cols)).to_numpy().astype(np.float32)
        dmat = xgb.DMatrix(np.ascontiguousarray(x))
        return float(self._booster.predict(dmat)[0])

    async def _emit_alert(
        self,
        trade: WalletTrade,
        features: FeatureRow,
        *,
        pred: float,
        edge: float,
    ) -> None:
        if self._sink is None:
            _LOG.warning("gate_model.no_sink", tx=trade.transaction_hash)
            return
        alert = Alert(
            detector="gate_buy",
            alert_key=f"gate:{trade.transaction_hash}:{features.side}",
            severity="med",
            title=f"gate_buy {features.side} on {trade.condition_id}",
            body={
                "wallet": trade.wallet,
                "condition_id": str(trade.condition_id),
                "side": features.side,
                "implied_prob_at_buy": float(features.implied_prob_at_buy),
                "pred": float(pred),
                "edge": float(edge),
                "top_category": features.market_category,
                "model_version": self._model_version,
                "trade_ts": trade.timestamp,
                "bet_size_usd": float(trade.usd_value),
            },
            created_at=int(time.time()),
        )
        await self._sink.emit(alert)
