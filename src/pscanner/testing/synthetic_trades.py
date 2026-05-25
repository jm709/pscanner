"""Deterministic synthetic ``Trade`` + ``MarketMetadata`` streams for tests.

Used by ``tests/corpus/conftest.py``. The defaults match the shape that
conftest used originally (6 wallets x4 markets, ``MarketMetadata.categories``
populated from the trade's primary category). Callers that need a different
shape pass keyword overrides.
"""

from __future__ import annotations

import random

from pscanner.corpus.feature_projection import MarketMetadata, Trade


def build_synthetic_trades(
    *,
    seed: int,
    n: int,
    n_wallets: int = 6,
    n_markets: int = 4,
) -> list[Trade]:
    """Generate a deterministic synthetic trade stream covering common shapes.

    Each trade picks a wallet from ``[0xw00 .. 0xw{N-1}]``, a market from
    ``[0xm00 .. 0xm{M-1}]``, a YES/NO side, a 70/30 BUY/SELL bias, and a
    price/size from uniform ranges. The PRNG is seeded so the output is
    reproducible.
    """
    rng = random.Random(seed)  # noqa: S311  -- deterministic test seed, not crypto
    wallets = [f"0xw{i:02d}" for i in range(n_wallets)]
    markets = [f"0xm{i:02d}" for i in range(n_markets)]
    base_ts = 1_700_000_000
    trades: list[Trade] = []
    for i in range(n):
        wallet = rng.choice(wallets)
        market = rng.choice(markets)
        side = rng.choice(("YES", "NO"))
        bs = rng.choices(("BUY", "SELL"), weights=(0.7, 0.3))[0]
        price = round(rng.uniform(0.05, 0.95), 4)
        size = round(rng.uniform(50.0, 500.0), 2)
        trades.append(
            Trade(
                tx_hash=f"tx{i:04d}",
                asset_id=f"{market}-{side}",
                wallet_address=wallet,
                condition_id=market,
                outcome_side=side,
                bs=bs,
                price=price,
                size=size,
                notional_usd=round(price * size, 4),
                ts=base_ts + i * 60,
                category=rng.choice(("sports", "esports", "crypto")),
            )
        )
    return trades


def build_metadata(
    trades: list[Trade],
    *,
    with_categories: bool = True,
) -> dict[str, MarketMetadata]:
    """Build a ``MarketMetadata`` for every market in ``trades``.

    Each market's metadata inherits the primary category from the first
    trade seen on that market, and gets a ``closed_at`` 7 days after that
    trade's ts. When ``with_categories=True`` (the default), the
    ``categories`` tuple is populated with a single-element tuple of the
    primary category — matching the conftest fixture. When ``False``,
    ``categories`` is left as the empty-tuple default — matching the
    daemon parity-test shape.
    """
    by_market: dict[str, MarketMetadata] = {}
    for t in trades:
        if t.condition_id in by_market:
            continue
        kwargs: dict[str, object] = {
            "condition_id": t.condition_id,
            "category": t.category,
            "closed_at": t.ts + 86_400 * 7,
            "opened_at": t.ts - 60,
        }
        if with_categories:
            kwargs["categories"] = (t.category,)
        by_market[t.condition_id] = MarketMetadata(**kwargs)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    return by_market
