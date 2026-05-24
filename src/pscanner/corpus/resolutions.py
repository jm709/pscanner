"""Market-resolution lookup for the corpus pipeline.

Translates a gamma ``Market`` into a ``MarketResolution`` (which side won)
and writes to ``market_resolutions``. Skips disputed/voided markets where
neither outcome price is at or above the resolved-threshold.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from typing import Any, Final

import structlog

from pscanner.corpus.repos import MarketResolution, MarketResolutionsRepo
from pscanner.kalshi.client import KalshiClient
from pscanner.manifold.client import ManifoldClient
from pscanner.poly.gamma import GammaClient
from pscanner.poly.models import Market

_log = structlog.get_logger(__name__)
# Older Polymarket markets store the winning outcome's price as a long
# decimal close to 1.0 (e.g. "0.9999996501077740101437594120537861")
# rather than the crisp "1.0" used on newer markets. Accept any price at
# or above this threshold as "this side won". Truly disputed/voided
# markets have both prices below the threshold.
_RESOLVED_THRESHOLD: Final[float] = 0.99


def determine_outcome_yes_won(market: Market) -> int | None:
    """Return 1 if YES (index 0) won, 0 if NO (index 1) won, else None.

    Returns None if ``outcome_prices`` is empty or no price is at or
    above ``_RESOLVED_THRESHOLD`` (disputed/voided markets).
    """
    if not market.outcome_prices:
        return None
    for idx, price in enumerate(market.outcome_prices):
        if price >= _RESOLVED_THRESHOLD:
            return 1 if idx == 0 else 0
    return None


async def record_resolutions(
    *,
    gamma: GammaClient,
    repo: MarketResolutionsRepo,
    targets: Iterable[tuple[str, str, int]],
    now_ts: int,
    platform: str = "polymarket",
) -> int:
    """Fetch resolutions for the given (condition_id, slug, resolved_at) tuples.

    Args:
        gamma: Gamma client with ``get_market_by_slug``.
        repo: ``MarketResolutionsRepo`` to upsert into.
        targets: Iterable of ``(condition_id, market_slug, resolved_at_hint)``.
        now_ts: Unix seconds, recorded as ``recorded_at`` on each row.
        platform: Platform tag written onto every ``MarketResolution`` row.
            Defaults to ``"polymarket"`` so the existing call site is
            unchanged.

    Returns:
        Count of resolutions actually written (excludes skipped/disputed).
    """
    written = 0
    for condition_id, slug, resolved_at in targets:
        market = await gamma.get_market_by_slug(slug)
        if market is None:
            _log.warning("corpus.resolution_market_not_found", condition_id=condition_id, slug=slug)
            continue
        yes_won = determine_outcome_yes_won(market)
        if yes_won is None:
            _log.warning("corpus.resolution_disputed", condition_id=condition_id, slug=slug)
            continue
        repo.upsert(
            MarketResolution(
                condition_id=condition_id,
                winning_outcome_index=0 if yes_won == 1 else 1,
                outcome_yes_won=yes_won,
                resolved_at=resolved_at,
                source="gamma",
                platform=platform,
            ),
            recorded_at=now_ts,
        )
        written += 1
    return written


async def _record_resolutions_loop(
    *,
    targets: Iterable[tuple[str, int]],
    fetch_market: Callable[[str], Awaitable[Any]],
    classify: Callable[[str, Any], tuple[int, int] | None],
    source: str,
    platform: str,
    repo: MarketResolutionsRepo,
    now_ts: int,
) -> int:
    """Drain ``targets`` via ``fetch_market`` + ``classify``; upsert and count.

    Shared loop for the Manifold and Kalshi resolution writers. The
    ``classify`` callback returns ``(outcome_yes_won, winning_outcome_index)``
    for a writeable row, or ``None`` to skip (in which case the callback is
    expected to log its own reason). Polymarket's gamma path stays separate
    (``record_resolutions``) because its target shape includes a slug.
    """
    written = 0
    for ident, resolved_at in targets:
        market = await fetch_market(ident)
        outcome = classify(ident, market)
        if outcome is None:
            continue
        outcome_yes_won, winning_outcome_index = outcome
        repo.upsert(
            MarketResolution(
                condition_id=ident,
                winning_outcome_index=winning_outcome_index,
                outcome_yes_won=outcome_yes_won,
                resolved_at=resolved_at,
                source=source,
                platform=platform,
            ),
            recorded_at=now_ts,
        )
        written += 1
    return written


async def record_manifold_resolutions(
    *,
    client: ManifoldClient,
    repo: MarketResolutionsRepo,
    targets: Iterable[tuple[str, int]],
    now_ts: int,
) -> int:
    """Fetch resolution outcomes for resolved Manifold markets.

    For each target, calls ``ManifoldClient.get_market(market_id)`` and reads
    the ``resolution`` field. YES/NO produce a ``market_resolutions`` row;
    MKT, CANCEL, and ``None`` are logged and skipped (no row written, so the
    inner JOIN in ``build_features`` excludes them from ``training_examples``).

    Args:
        client: Open ``ManifoldClient``.
        repo: ``MarketResolutionsRepo`` to upsert into.
        targets: Iterable of ``(market_id, resolved_at_hint)``.
        now_ts: Unix seconds, recorded as ``recorded_at`` on each row.

    Returns:
        Count of resolutions actually written (excludes skipped MKT/CANCEL/null).
    """

    def classify(market_id: str, market: Any) -> tuple[int, int] | None:
        if market.resolution == "YES":
            return (1, 0)
        if market.resolution == "NO":
            return (0, 1)
        _log.warning(
            "corpus.manifold_resolution_skipped",
            market_id=market_id,
            resolution=market.resolution,
        )
        return None

    return await _record_resolutions_loop(
        targets=targets,
        fetch_market=client.get_market,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        classify=classify,
        source="manifold-rest",
        platform="manifold",
        repo=repo,
        now_ts=now_ts,
    )


async def record_kalshi_resolutions(
    *,
    client: KalshiClient,
    repo: MarketResolutionsRepo,
    targets: Iterable[tuple[str, int]],
    now_ts: int,
) -> int:
    """Fetch resolution outcomes for settled Kalshi markets.

    For each target, calls ``KalshiClient.get_market(ticker)`` and reads
    ``market.result``. Writes a ``market_resolutions`` row for ``"yes"``
    or ``"no"``. Logs and skips for ``"scalar"`` (defensive — should be
    filtered at enumeration), ``""`` with terminal status (voided / odd
    state), and any market with ``status == "disputed"`` (contested
    resolution; the next refresh will pick it up once Kalshi moves it to
    a clean terminal state).

    Args:
        client: Open ``KalshiClient``.
        repo: ``MarketResolutionsRepo`` to upsert into.
        targets: Iterable of ``(market_ticker, resolved_at_hint)``.
        now_ts: Unix seconds, recorded as ``recorded_at`` on each row.

    Returns:
        Count of resolutions actually written (excludes skipped markets).
    """

    def classify(ticker: str, market: Any) -> tuple[int, int] | None:
        if market.status == "disputed":
            _log.warning(
                "corpus.kalshi_resolution_disputed",
                market_ticker=ticker,
                result=market.result,
            )
            return None
        if market.result == "yes":
            return (1, 0)
        if market.result == "no":
            return (0, 1)
        if market.result == "scalar":
            _log.warning(
                "corpus.kalshi_resolution_scalar",
                market_ticker=ticker,
            )
            return None
        _log.warning(
            "corpus.kalshi_resolution_undetermined",
            market_ticker=ticker,
            status=market.status,
            result=market.result,
        )
        return None

    return await _record_resolutions_loop(
        targets=targets,
        fetch_market=client.get_market,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
        classify=classify,
        source="kalshi-rest",
        platform="kalshi",
        repo=repo,
        now_ts=now_ts,
    )
