"""Insider-episode catalogue: rank pre-news buyers on documented case markets.

Reads ``docs/research/insider-cases.yaml``, extracts the pre-anchor window on
each case market from the corpus, enriches buyers with data-api provenance,
scores them with the frozen scorer, compares against matched control markets,
blind-tests published wallets on holdout cases, and writes markdown reports.

Spec: docs/superpowers/specs/2026-10-11-insider-episode-catalogue-design.md
"""

from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.insider_episode_lib import (  # noqa: E402  (after sys.path bootstrap)
    MarketRef,
    PublicCase,
    SealedCase,
)

_FULL_ADDRESS = re.compile(r"^0x[0-9a-f]{40}$")


def _parse_ts(value: str) -> int:
    """Parse an ISO-8601 UTC timestamp like ``2026-01-03T06:00:00Z`` to unix seconds."""
    return int(dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


def _published_wallets(raw: list[dict[str, Any]]) -> tuple[str, ...]:
    """Keep only full 42-character addresses, lowercased and stripped."""
    out: list[str] = []
    for w in raw:
        addr = w.get("address")
        if not isinstance(addr, str):
            continue
        addr = addr.strip().lower()
        if _FULL_ADDRESS.match(addr):
            out.append(addr)
    return tuple(out)


def _market_refs(raw: list[dict[str, Any]]) -> tuple[MarketRef, ...]:
    return tuple(
        MarketRef(
            slug=str(m["market_slug"]),
            winning_side=str(m["winning_side"]).upper(),
            role=str(m.get("role", "")),
        )
        for m in raw
    )


def load_registry(path: Path) -> tuple[list[PublicCase], dict[str, SealedCase]]:
    """Load the case registry and split it into public and sealed records.

    Args:
        path: Path to ``insider-cases.yaml``.

    Returns:
        ``(public_cases, sealed_by_case_id)``. Only the comparator may read the
        sealed mapping.
    """
    doc = yaml.safe_load(path.read_text())
    public: list[PublicCase] = []
    sealed: dict[str, SealedCase] = {}
    for raw in doc["cases"]:
        cid = str(raw["id"])
        public.append(
            PublicCase(
                id=cid,
                name=str(raw.get("name", cid)),
                split=str(raw["split"]),
                confidence=str(raw["confidence"]),
                analyzable=bool(raw.get("analyzable", False)),
                news_ts=_parse_ts(str(raw["news_ts_utc"])),
                news_precision=str(raw.get("news_ts_precision", "day")),
                markets=_market_refs(raw.get("corpus_markets") or []),
            )
        )
        sealed[cid] = SealedCase(id=cid, wallets=_published_wallets(raw.get("wallets") or []))
    return public, sealed
