"""Pydantic models for Manifold Markets REST payloads.

Field names follow Manifold's camelCase JSON convention via pydantic aliases.
All models use ``extra="ignore"`` so unknown fields from future API versions
don't cause validation errors.

Note: Manifold amounts are in **mana** (play money) — never aggregate them
into real-money USD totals.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from pscanner.manifold.ids import ManifoldMarketId, ManifoldUserId

_BASE_CONFIG: ConfigDict = ConfigDict(
    populate_by_name=True,
    extra="ignore",
)


class ManifoldMarket(BaseModel):
    """A Manifold Markets contract (market).

    Supports binary YES/NO markets (``outcome_type == "BINARY"``). Multi-outcome
    CFMM markets are represented by the same model but the ``prob`` field may be
    ``None`` for non-binary types.
    """

    model_config = _BASE_CONFIG

    id: ManifoldMarketId
    creator_id: ManifoldUserId = Field(alias="creatorId")
    question: str
    outcome_type: str = Field(alias="outcomeType")
    mechanism: str
    prob: float | None = None
    volume: float
    total_liquidity: float = Field(alias="totalLiquidity", default=0.0)
    is_resolved: bool = Field(alias="isResolved")
    resolution_time: int | None = Field(alias="resolutionTime", default=None)
    close_time: int | None = Field(alias="closeTime", default=None)
    url: str | None = None
    slug: str | None = None

    @property
    def is_binary(self) -> bool:
        """True iff this is a binary YES/NO market."""
        return self.outcome_type == "BINARY"


class ManifoldBet(BaseModel):
    """A single bet (trade) on a Manifold market.

    Amounts are in mana (play money). ``limit_prob`` is non-``None`` only for
    limit orders; ``None`` indicates a market order.
    """

    model_config = _BASE_CONFIG

    id: str
    user_id: ManifoldUserId = Field(alias="userId")
    contract_id: ManifoldMarketId = Field(alias="contractId")
    outcome: str
    amount: float
    prob_before: float = Field(alias="probBefore")
    prob_after: float = Field(alias="probAfter")
    created_time: int = Field(alias="createdTime")
    is_filled: bool | None = Field(alias="isFilled", default=None)
    is_cancelled: bool | None = Field(alias="isCancelled", default=None)
    limit_prob: float | None = Field(alias="limitProb", default=None)
    shares: float | None = None
    fees: dict[str, Any] | None = None


class ManifoldUser(BaseModel):
    """A Manifold Markets user.

    The ``id`` field is an opaque hash (``ManifoldUserId``); ``username`` is the
    human-readable handle (e.g. ``"alice"``).
    """

    model_config = _BASE_CONFIG

    id: ManifoldUserId
    username: str
    name: str
    created_time: int = Field(alias="createdTime")
    balance: float | None = None
    avatar_url: str | None = Field(alias="avatarUrl", default=None)
