"""Unit tests for GateModelDetector (#79)."""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb
from structlog.testing import capture_logs

from pscanner.alerts.sink import AlertSink
from pscanner.config import GateModelConfig
from pscanner.corpus.features import FeatureRow, MarketMetadata
from pscanner.daemon.live_history import LiveHistoryProvider
from pscanner.detectors.gate_model import GateModelDetector
from pscanner.ml.preprocessing import LEAKAGE_COLS, OneHotEncoder
from pscanner.ml.streaming import _derive_feature_names
from pscanner.poly.ids import AssetId, ConditionId, EventId, MarketId
from pscanner.store.db import init_db
from pscanner.store.repo import AlertsRepo, CachedMarket, MarketCacheRepo, WalletTrade


def _train_dummy_model(
    out_dir: Path, *, accepted_categories: tuple[str, ...] = ("esports",)
) -> None:
    """Train a 1-feature stub model and persist artifacts in the layout the detector expects."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=(200, 1))
    y = (x[:, 0] > 0.5).astype(int)
    booster = xgb.train(
        params={
            "objective": "binary:logistic",
            "max_depth": 2,
            "tree_method": "hist",
            "verbosity": 0,
        },
        dtrain=xgb.DMatrix(x, label=y, feature_names=["x0"]),
        num_boost_round=5,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_dir / "model.json"))
    (out_dir / "preprocessor.json").write_text(
        json.dumps(
            {
                "version": 2,
                "leakage_cols": [],
                "carrier_cols": [],
                "encoder": {"levels": {}},
                "accepted_categories": list(accepted_categories),
                "platform": "polymarket",
            }
        )
    )


def _new_db() -> sqlite3.Connection:
    return init_db(Path(":memory:"))


def test_detector_loads_model_and_accepted_categories(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "model"
    _train_dummy_model(artifact_dir)
    cfg = GateModelConfig(enabled=True, artifact_dir=artifact_dir, min_pred=0.7)
    conn = _new_db()
    try:
        provider = LiveHistoryProvider(conn=conn, metadata={})
        detector = GateModelDetector(
            config=cfg,
            provider=provider,
            alerts_repo=AlertsRepo(conn),
        )
    finally:
        conn.close()
    assert detector.name == "gate_model"
    assert detector.accepted_categories == ("esports",)


def test_detector_overrides_accepted_categories_from_config(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "model"
    _train_dummy_model(artifact_dir)
    cfg = GateModelConfig(
        enabled=True,
        artifact_dir=artifact_dir,
        accepted_categories=("sports", "esports"),
    )
    conn = _new_db()
    try:
        provider = LiveHistoryProvider(conn=conn, metadata={})
        detector = GateModelDetector(config=cfg, provider=provider, alerts_repo=AlertsRepo(conn))
    finally:
        conn.close()
    assert detector.accepted_categories == ("sports", "esports")


def _make_wallet_trade(
    *,
    side: str = "BUY",
    wallet: str = "0xabc",
    condition_id: str = "0xc1",
    asset_id: str = "0xa1",
    price: float = 0.42,
    size: float = 100.0,
    usd_value: float = 42.0,
    timestamp: int = 1_700_000_000,
) -> WalletTrade:
    return WalletTrade(
        transaction_hash=f"tx{timestamp}",
        asset_id=AssetId(asset_id),
        side=side,
        wallet=wallet,
        condition_id=ConditionId(condition_id),
        size=size,
        price=price,
        usd_value=usd_value,
        status="filled",
        source="market_scoped",
        timestamp=timestamp,
        recorded_at=timestamp + 1,
    )


def test_pre_screen_skips_sell_trade(tmp_path: Path) -> None:
    conn = _new_db()
    try:
        artifact_dir = tmp_path / "model"
        _train_dummy_model(artifact_dir)
        provider = LiveHistoryProvider(conn=conn, metadata={})
        detector = GateModelDetector(
            config=GateModelConfig(enabled=True, artifact_dir=artifact_dir),
            provider=provider,
            alerts_repo=AlertsRepo(conn),
        )
        trade = _make_wallet_trade(side="SELL")
        assert detector._should_score(trade) is False
    finally:
        conn.close()


def test_pre_screen_accepts_buy(tmp_path: Path) -> None:
    conn = _new_db()
    try:
        artifact_dir = tmp_path / "model"
        _train_dummy_model(artifact_dir)
        provider = LiveHistoryProvider(conn=conn, metadata={})
        detector = GateModelDetector(
            config=GateModelConfig(enabled=True, artifact_dir=artifact_dir),
            provider=provider,
            alerts_repo=AlertsRepo(conn),
        )
        trade = _make_wallet_trade(side="BUY")
        assert detector._should_score(trade) is True
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_queue_full_drops_with_warning(tmp_path: Path) -> None:
    conn = _new_db()
    try:
        artifact_dir = tmp_path / "model"
        _train_dummy_model(artifact_dir)
        cfg = GateModelConfig(
            enabled=True,
            artifact_dir=artifact_dir,
            queue_max_size=2,
        )
        provider = LiveHistoryProvider(conn=conn, metadata={})
        detector = GateModelDetector(config=cfg, provider=provider, alerts_repo=AlertsRepo(conn))
        # Don't start the worker — queue fills up.
        with capture_logs() as logs:
            for i in range(5):
                detector.handle_trade_sync(_make_wallet_trade(timestamp=1_700_000_000 + i))
        events = [le["event"] for le in logs]
    finally:
        conn.close()
    assert any(e == "gate_model.queue_full" for e in events)


@pytest.mark.asyncio
async def test_queue_drops_sell_trades_before_enqueue(tmp_path: Path) -> None:
    """SELLs should never enter the queue (pre-screen happens before enqueue)."""
    conn = _new_db()
    try:
        artifact_dir = tmp_path / "model"
        _train_dummy_model(artifact_dir)
        cfg = GateModelConfig(enabled=True, artifact_dir=artifact_dir, queue_max_size=1)
        provider = LiveHistoryProvider(conn=conn, metadata={})
        detector = GateModelDetector(config=cfg, provider=provider, alerts_repo=AlertsRepo(conn))
        # 10 SELLs into a queue with capacity 1 — none should drop because
        # all are filtered out before enqueue.
        with capture_logs() as logs:
            for i in range(10):
                detector.handle_trade_sync(
                    _make_wallet_trade(side="SELL", timestamp=1_700_000_000 + i)
                )
        events = [le["event"] for le in logs]
    finally:
        conn.close()
    assert "gate_model.queue_full" not in events


@pytest.mark.asyncio
async def test_evaluate_emits_alert_when_gates_pass(tmp_path: Path) -> None:
    conn = _new_db()
    try:
        artifact_dir = tmp_path / "model"
        _train_dummy_model(artifact_dir)
        metadata = {
            "0xc1": MarketMetadata(
                condition_id="0xc1",
                category="esports",
                closed_at=1_700_100_000,
                opened_at=1_699_900_000,
            )
        }
        cfg = GateModelConfig(enabled=True, artifact_dir=artifact_dir, min_pred=0.5)
        provider = LiveHistoryProvider(conn=conn, metadata=metadata)
        alerts_repo = AlertsRepo(conn)
        detector = GateModelDetector(config=cfg, provider=provider, alerts_repo=alerts_repo)
        # Replace prediction + outcome side with deterministic stubs.
        detector._predict_one = lambda _: 0.85  # type: ignore[method-assign,assignment]  # ty:ignore[invalid-assignment]
        detector._resolve_outcome_side = lambda _trade: "YES"  # type: ignore[method-assign,assignment]  # ty:ignore[invalid-assignment]
        sink = AlertSink(alerts_repo=alerts_repo)
        detector._sink = sink
        trade = _make_wallet_trade(condition_id="0xc1", price=0.40)
        await detector.evaluate(trade)
        recent = alerts_repo.recent(detector="gate_buy", limit=10)
    finally:
        conn.close()
    assert len(recent) == 1
    body = recent[0].body
    assert isinstance(body, dict)
    assert body["condition_id"] == "0xc1"
    assert body["pred"] == pytest.approx(0.85)
    assert body["implied_prob_at_buy"] == pytest.approx(0.40)
    assert body["edge"] == pytest.approx(0.85 - 0.40)


@pytest.mark.asyncio
async def test_evaluate_skips_when_pred_below_floor(tmp_path: Path) -> None:
    conn = _new_db()
    try:
        artifact_dir = tmp_path / "model"
        _train_dummy_model(artifact_dir)
        metadata = {
            "0xc1": MarketMetadata(
                condition_id="0xc1",
                category="esports",
                closed_at=1_700_100_000,
                opened_at=1_699_900_000,
            )
        }
        cfg = GateModelConfig(enabled=True, artifact_dir=artifact_dir, min_pred=0.7)
        provider = LiveHistoryProvider(conn=conn, metadata=metadata)
        alerts_repo = AlertsRepo(conn)
        detector = GateModelDetector(config=cfg, provider=provider, alerts_repo=alerts_repo)
        detector._predict_one = lambda _: 0.30  # type: ignore[method-assign,assignment]  # ty:ignore[invalid-assignment]
        detector._resolve_outcome_side = lambda _trade: "YES"  # type: ignore[method-assign,assignment]  # ty:ignore[invalid-assignment]
        detector._sink = AlertSink(alerts_repo=alerts_repo)
        trade = _make_wallet_trade(condition_id="0xc1", price=0.20)
        await detector.evaluate(trade)
        recent = alerts_repo.recent(detector="gate_buy", limit=10)
    finally:
        conn.close()
    assert recent == []


@pytest.mark.asyncio
async def test_evaluate_skips_when_category_not_accepted(tmp_path: Path) -> None:
    conn = _new_db()
    try:
        artifact_dir = tmp_path / "model"
        _train_dummy_model(artifact_dir)
        metadata = {
            "0xc1": MarketMetadata(
                condition_id="0xc1",
                category="politics",  # NOT in accepted_categories=("esports",)
                closed_at=1_700_100_000,
                opened_at=1_699_900_000,
            )
        }
        cfg = GateModelConfig(enabled=True, artifact_dir=artifact_dir, min_pred=0.5)
        provider = LiveHistoryProvider(conn=conn, metadata=metadata)
        alerts_repo = AlertsRepo(conn)
        detector = GateModelDetector(config=cfg, provider=provider, alerts_repo=alerts_repo)
        detector._predict_one = lambda _: 0.85  # type: ignore[method-assign,assignment]  # ty:ignore[invalid-assignment]
        detector._resolve_outcome_side = lambda _trade: "YES"  # type: ignore[method-assign,assignment]  # ty:ignore[invalid-assignment]
        detector._sink = AlertSink(alerts_repo=alerts_repo)
        trade = _make_wallet_trade(condition_id="0xc1", price=0.40)
        await detector.evaluate(trade)
        recent = alerts_repo.recent(detector="gate_buy", limit=10)
    finally:
        conn.close()
    assert recent == []


@pytest.mark.asyncio
async def test_evaluate_accepts_multi_label_market_via_intersection(tmp_path: Path) -> None:
    """Differentiating case: primary ``market_category`` is NOT in
    ``accepted_categories``, but a secondary category in ``market_categories``
    IS. Set-intersection accepts; legacy single-string membership rejects.
    """
    conn = _new_db()
    try:
        artifact_dir = tmp_path / "model"
        _train_dummy_model(artifact_dir, accepted_categories=("elections",))
        metadata = {
            "0xc1": MarketMetadata(
                condition_id="0xc1",
                category="macro",  # primary — NOT in accepted
                closed_at=1_700_100_000,
                opened_at=1_699_900_000,
                categories=("macro", "elections"),  # contains elections — IS in accepted
            )
        }
        cfg = GateModelConfig(enabled=True, artifact_dir=artifact_dir, min_pred=0.5)
        provider = LiveHistoryProvider(conn=conn, metadata=metadata)
        alerts_repo = AlertsRepo(conn)
        detector = GateModelDetector(config=cfg, provider=provider, alerts_repo=alerts_repo)
        detector._predict_one = lambda _: 0.85  # type: ignore[method-assign,assignment]  # ty:ignore[invalid-assignment]
        detector._resolve_outcome_side = lambda _trade: "YES"  # type: ignore[method-assign,assignment]  # ty:ignore[invalid-assignment]
        detector._sink = AlertSink(alerts_repo=alerts_repo)
        trade = _make_wallet_trade(condition_id="0xc1", price=0.40)
        await detector.evaluate(trade)
        recent = alerts_repo.recent(detector="gate_buy", limit=10)
    finally:
        conn.close()
    assert len(recent) == 1


@pytest.mark.asyncio
async def test_evaluate_falls_back_to_primary_category_when_categories_empty(
    tmp_path: Path,
) -> None:
    """A market with ``MarketMetadata.categories=()`` (un-backfilled corpus
    row, pending #121) still gates correctly because ``compute_features``
    falls back to ``(meta.category,)`` for ``features.market_categories``.
    Decision C from #119 — preserves backward compatibility before #121's
    gamma backfill completes.
    """
    conn = _new_db()
    try:
        artifact_dir = tmp_path / "model"
        _train_dummy_model(artifact_dir, accepted_categories=("esports",))
        metadata = {
            "0xc1": MarketMetadata(
                condition_id="0xc1",
                category="esports",  # primary string
                closed_at=1_700_100_000,
                opened_at=1_699_900_000,
                # categories=() (default — un-backfilled)
            )
        }
        cfg = GateModelConfig(enabled=True, artifact_dir=artifact_dir, min_pred=0.5)
        provider = LiveHistoryProvider(conn=conn, metadata=metadata)
        alerts_repo = AlertsRepo(conn)
        detector = GateModelDetector(config=cfg, provider=provider, alerts_repo=alerts_repo)
        detector._predict_one = lambda _: 0.85  # type: ignore[method-assign,assignment]  # ty:ignore[invalid-assignment]
        detector._resolve_outcome_side = lambda _trade: "YES"  # type: ignore[method-assign,assignment]  # ty:ignore[invalid-assignment]
        detector._sink = AlertSink(alerts_repo=alerts_repo)
        trade = _make_wallet_trade(condition_id="0xc1", price=0.40)
        await detector.evaluate(trade)
        recent = alerts_repo.recent(detector="gate_buy", limit=10)
    finally:
        conn.close()
    assert len(recent) == 1


@pytest.mark.asyncio
async def test_evaluate_rejects_when_no_category_set_member_accepted(
    tmp_path: Path,
) -> None:
    """A multi-label market with no category in ``accepted_categories`` is rejected.

    Sanity guard: even with set semantics, a disjoint set produces zero alerts.
    """
    conn = _new_db()
    try:
        artifact_dir = tmp_path / "model"
        _train_dummy_model(artifact_dir, accepted_categories=("esports",))
        metadata = {
            "0xc1": MarketMetadata(
                condition_id="0xc1",
                category="macro",
                closed_at=1_700_100_000,
                opened_at=1_699_900_000,
                categories=("macro", "elections"),
            )
        }
        cfg = GateModelConfig(enabled=True, artifact_dir=artifact_dir, min_pred=0.5)
        provider = LiveHistoryProvider(conn=conn, metadata=metadata)
        alerts_repo = AlertsRepo(conn)
        detector = GateModelDetector(config=cfg, provider=provider, alerts_repo=alerts_repo)
        detector._predict_one = lambda _: 0.85  # type: ignore[method-assign,assignment]  # ty:ignore[invalid-assignment]
        detector._resolve_outcome_side = lambda _trade: "YES"  # type: ignore[method-assign,assignment]  # ty:ignore[invalid-assignment]
        detector._sink = AlertSink(alerts_repo=alerts_repo)
        trade = _make_wallet_trade(condition_id="0xc1", price=0.40)
        await detector.evaluate(trade)
        recent = alerts_repo.recent(detector="gate_buy", limit=10)
    finally:
        conn.close()
    assert recent == []


def test_resolve_outcome_side_returns_empty_without_market_cache(tmp_path: Path) -> None:
    conn = _new_db()
    try:
        artifact_dir = tmp_path / "model"
        _train_dummy_model(artifact_dir)
        provider = LiveHistoryProvider(conn=conn, metadata={})
        detector = GateModelDetector(
            config=GateModelConfig(enabled=True, artifact_dir=artifact_dir),
            provider=provider,
            alerts_repo=AlertsRepo(conn),
        )
        trade = _make_wallet_trade(condition_id="0xc1", asset_id="0xa1")
        assert detector._resolve_outcome_side(trade) == ""
    finally:
        conn.close()


def test_resolve_outcome_side_via_market_cache(tmp_path: Path) -> None:
    conn = _new_db()
    try:
        artifact_dir = tmp_path / "model"
        _train_dummy_model(artifact_dir)
        market_cache = MarketCacheRepo(conn)
        cached = CachedMarket(
            market_id=MarketId("m1"),
            event_id=EventId("e1"),
            title="t",
            liquidity_usd=1.0,
            volume_usd=1.0,
            outcome_prices=[0.5, 0.5],
            active=True,
            cached_at=1_700_000_000,
            condition_id=ConditionId("0xc1"),
            event_slug=None,
            outcomes=["Yes", "No"],
            asset_ids=[AssetId("0xa1"), AssetId("0xa2")],
        )
        market_cache.upsert(cached)
        provider = LiveHistoryProvider(conn=conn, metadata={})
        detector = GateModelDetector(
            config=GateModelConfig(enabled=True, artifact_dir=artifact_dir),
            provider=provider,
            alerts_repo=AlertsRepo(conn),
            market_cache=market_cache,
        )
        yes_trade = _make_wallet_trade(condition_id="0xc1", asset_id="0xa1")
        no_trade = _make_wallet_trade(condition_id="0xc1", asset_id="0xa2")
        unknown = _make_wallet_trade(condition_id="0xc1", asset_id="0xother")
        yes_side = detector._resolve_outcome_side(yes_trade)
        no_side = detector._resolve_outcome_side(no_trade)
        unknown_side = detector._resolve_outcome_side(unknown)
    finally:
        conn.close()
    assert yes_side == "YES"
    assert no_side == "NO"
    assert unknown_side == ""


def test_resolve_outcome_logs_when_no_market_cache(tmp_path: Path) -> None:
    conn = _new_db()
    try:
        artifact_dir = tmp_path / "model"
        _train_dummy_model(artifact_dir)
        provider = LiveHistoryProvider(conn=conn, metadata={})
        detector = GateModelDetector(
            config=GateModelConfig(enabled=True, artifact_dir=artifact_dir),
            provider=provider,
            alerts_repo=AlertsRepo(conn),
        )
        trade = _make_wallet_trade(condition_id="0xc1", asset_id="0xa1")
        with capture_logs() as logs:
            assert detector._resolve_outcome_side(trade) == ""
    finally:
        conn.close()
    matches = [log for log in logs if log["event"] == "gate_model.no_market_cache"]
    assert len(matches) == 1
    assert matches[0]["condition_id"] == "0xc1"


def test_resolve_outcome_logs_when_market_not_cached(tmp_path: Path) -> None:
    conn = _new_db()
    try:
        artifact_dir = tmp_path / "model"
        _train_dummy_model(artifact_dir)
        market_cache = MarketCacheRepo(conn)  # empty
        provider = LiveHistoryProvider(conn=conn, metadata={})
        detector = GateModelDetector(
            config=GateModelConfig(enabled=True, artifact_dir=artifact_dir),
            provider=provider,
            alerts_repo=AlertsRepo(conn),
            market_cache=market_cache,
        )
        trade = _make_wallet_trade(condition_id="0xc1", asset_id="0xa1")
        with capture_logs() as logs:
            assert detector._resolve_outcome_side(trade) == ""
    finally:
        conn.close()
    matches = [log for log in logs if log["event"] == "gate_model.market_not_cached"]
    assert len(matches) == 1
    assert matches[0]["condition_id"] == "0xc1"


@pytest.mark.asyncio
async def test_evaluate_logs_when_outcome_unresolved(tmp_path: Path) -> None:
    """`evaluate()` emits gate_model.skip_unresolved_outcome on the early-return."""
    conn = _new_db()
    try:
        artifact_dir = tmp_path / "model"
        _train_dummy_model(artifact_dir)
        provider = LiveHistoryProvider(conn=conn, metadata={})
        detector = GateModelDetector(
            config=GateModelConfig(enabled=True, artifact_dir=artifact_dir),
            provider=provider,
            alerts_repo=AlertsRepo(conn),
        )  # no market_cache => _resolve_outcome_side returns ""
        trade = _make_wallet_trade(condition_id="0xc1")
        with capture_logs() as logs:
            await detector.evaluate(trade)
    finally:
        conn.close()
    events = [log["event"] for log in logs]
    assert "gate_model.skip_unresolved_outcome" in events


def test_resolve_outcome_logs_when_outcome_not_binary(tmp_path: Path) -> None:
    conn = _new_db()
    try:
        artifact_dir = tmp_path / "model"
        _train_dummy_model(artifact_dir)
        market_cache = MarketCacheRepo(conn)
        cached = CachedMarket(
            market_id=MarketId("m1"),
            event_id=EventId("e1"),
            title="t",
            liquidity_usd=1.0,
            volume_usd=1.0,
            outcome_prices=[0.5, 0.5],
            active=True,
            cached_at=1_700_000_000,
            condition_id=ConditionId("0xc1"),
            event_slug=None,
            outcomes=["Trump", "Biden"],  # neither YES nor NO
            asset_ids=[AssetId("0xa1"), AssetId("0xa2")],
        )
        market_cache.upsert(cached)
        provider = LiveHistoryProvider(conn=conn, metadata={})
        detector = GateModelDetector(
            config=GateModelConfig(enabled=True, artifact_dir=artifact_dir),
            provider=provider,
            alerts_repo=AlertsRepo(conn),
            market_cache=market_cache,
        )
        trade = _make_wallet_trade(condition_id="0xc1", asset_id="0xa1")
        with capture_logs() as logs:
            assert detector._resolve_outcome_side(trade) == ""
    finally:
        conn.close()
    matches = [log for log in logs if log["event"] == "gate_model.outcome_not_binary"]
    assert len(matches) == 1
    assert matches[0]["outcome"] == "Trump"
    assert matches[0]["outcome_normalized"] == "TRUMP"


def test_resolve_outcome_logs_when_asset_id_not_found(tmp_path: Path) -> None:
    conn = _new_db()
    try:
        artifact_dir = tmp_path / "model"
        _train_dummy_model(artifact_dir)
        market_cache = MarketCacheRepo(conn)
        cached = CachedMarket(
            market_id=MarketId("m1"),
            event_id=EventId("e1"),
            title="t",
            liquidity_usd=1.0,
            volume_usd=1.0,
            outcome_prices=[0.5, 0.5],
            active=True,
            cached_at=1_700_000_000,
            condition_id=ConditionId("0xc1"),
            event_slug=None,
            outcomes=["Yes", "No"],
            asset_ids=[AssetId("0xa1"), AssetId("0xa2")],
        )
        market_cache.upsert(cached)
        provider = LiveHistoryProvider(conn=conn, metadata={})
        detector = GateModelDetector(
            config=GateModelConfig(enabled=True, artifact_dir=artifact_dir),
            provider=provider,
            alerts_repo=AlertsRepo(conn),
            market_cache=market_cache,
        )
        trade = _make_wallet_trade(condition_id="0xc1", asset_id="0xother")
        with capture_logs() as logs:
            assert detector._resolve_outcome_side(trade) == ""
    finally:
        conn.close()
    matches = [log for log in logs if log["event"] == "gate_model.asset_id_not_found"]
    assert len(matches) == 1
    assert matches[0]["asset_id"] == "0xother"


def test_feature_cols_parity_against_training_derive_feature_names(tmp_path: Path) -> None:
    """Detector's ``_feature_cols`` must equal the training pipeline's order.

    The training pipeline computes feature_names via
    ``pscanner.ml.streaming._derive_feature_names`` from ``kept_cols`` (the
    PRAGMA-derived list of ``training_examples`` columns minus
    ``_NEVER_LOAD_COLS``). The detector replicates this analytically from
    ``FeatureRow`` fields. Any drift would feed the booster an off-by-one
    DMatrix at inference and silently mis-align every prediction —
    xgboost's QuantileDMatrix doesn't carry feature_names so column-index
    matching is silent on mismatch.
    """
    artifact_dir = tmp_path / "model"
    _train_dummy_model(artifact_dir)
    conn = _new_db()
    try:
        provider = LiveHistoryProvider(conn=conn, metadata={})
        detector = GateModelDetector(
            config=GateModelConfig(enabled=True, artifact_dir=artifact_dir),
            provider=provider,
            alerts_repo=AlertsRepo(conn),
        )
    finally:
        conn.close()

    # Build a kept_cols list mirroring what training_examples PRAGMA would
    # produce after _NEVER_LOAD_COLS is applied: every FeatureRow field +
    # the carrier columns the build_features pipeline writes alongside
    # them, MINUS the leakage cols stripped at SELECT time.
    feature_row_fields = tuple(f.name for f in dataclasses.fields(FeatureRow))
    extra_cols = ("condition_id", "trade_ts", "resolved_at", "label_won")
    full_pragma_cols = feature_row_fields + extra_cols
    kept_cols = tuple(c for c in full_pragma_cols if c not in LEAKAGE_COLS)
    # Use a richer encoder than the dummy one (to mimic production) so the
    # parity check covers the indicator-expansion path too.
    encoder = OneHotEncoder(
        levels={
            "side": ("NO", "YES"),
            "top_category": ("__none__", "esports", "sports", "thesis"),
            "market_category": ("esports", "sports", "thesis"),
        }
    )
    detector._encoder = encoder
    detector._feature_cols = detector._derive_feature_cols()

    training_cols = _derive_feature_names(kept_cols, encoder)
    assert detector._feature_cols == training_cols, (
        f"feature_cols drift between training and inference\n"
        f"  training:  {training_cols}\n"
        f"  inference: {detector._feature_cols}\n"
        f"  diff training-only: {set(training_cols) - set(detector._feature_cols)}\n"
        f"  diff inference-only: {set(detector._feature_cols) - set(training_cols)}"
    )
    # Sanity: time_to_resolution_seconds is in FeatureRow but must NOT
    # appear in either side (it's a LEAKAGE_COL).
    assert "time_to_resolution_seconds" not in training_cols
    assert "time_to_resolution_seconds" not in detector._feature_cols


@pytest.mark.asyncio
async def test_loader_rejects_v1_preprocessor_artifact(tmp_path: Path) -> None:
    """A preprocessor.json without ``version: 2`` must fail loading."""
    artifact_dir = tmp_path / "model"
    artifact_dir.mkdir()
    _train_dummy_model(artifact_dir)
    payload = json.loads((artifact_dir / "preprocessor.json").read_text())
    payload.pop("version", None)
    (artifact_dir / "preprocessor.json").write_text(json.dumps(payload))
    cfg = GateModelConfig(enabled=True, artifact_dir=artifact_dir)
    conn = _new_db()
    try:
        provider = LiveHistoryProvider(conn=conn, metadata={})
        alerts_repo = AlertsRepo(conn)
        with pytest.raises(ValueError, match=r"preprocessor\.json version"):
            GateModelDetector(config=cfg, provider=provider, alerts_repo=alerts_repo)
    finally:
        conn.close()
