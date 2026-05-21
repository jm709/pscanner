"""SubgraphCopyEvaluator — books paper copies of watchlisted wallets' trades.

The evaluator is paired with
:class:`pscanner.collectors.subgraph_trades.SubgraphTradeCollector`
(spec: ``docs/superpowers/specs/2026-05-21-issue-152-subgraph-trade-collector-design.md``).

Sizing is constant ``bankroll * position_fraction`` times a per-wallet
concentration multiplier that decays as one wallet's share of total
``subgraph_copy`` trades exceeds ``1.0 / active_watchlist_size``, floored at
``min_multiplier``. The floor guarantees the noisiest wallet still trades at
``>= min_multiplier * base`` rather than being silenced entirely.
"""

from __future__ import annotations

from pscanner.alerts.models import Alert
from pscanner.config import SubgraphCopyEvaluatorConfig
from pscanner.poly.ids import ConditionId
from pscanner.store.repo import PaperTradesRepo, WatchlistRepo
from pscanner.strategies.evaluators.protocol import ParsedSignal

_DETECTOR_NAME = "subgraph_copy"


class SubgraphCopyEvaluator:
    """Single-leg evaluator for the SubgraphTradeCollector alert stream."""

    def __init__(
        self,
        *,
        config: SubgraphCopyEvaluatorConfig,
        watchlist_repo: WatchlistRepo,
        paper_trades: PaperTradesRepo,
    ) -> None:
        """Bind config + read-only repos used at sizing time.

        Args:
            config: Tunables (position_fraction, min_multiplier).
            watchlist_repo: Source of ``active_watchlist_size`` for target_share.
            paper_trades: Source of ``count_by_source_wallet`` for share.
        """
        self._config = config
        self._watchlist = watchlist_repo
        self._paper_trades = paper_trades

    def accepts(self, alert: Alert) -> bool:
        """Return ``True`` only for ``subgraph_copy`` alerts."""
        return alert.detector == _DETECTOR_NAME

    def parse(self, alert: Alert) -> list[ParsedSignal]:
        """Extract a single :class:`ParsedSignal` from the alert body.

        Returns an empty list on body-shape mismatch so PaperTrader's
        soft-failure path applies (skip without crashing).
        """
        body = alert.body
        try:
            condition_id = ConditionId(str(body["condition_id"]))
            outcome = str(body["outcome"])
            wallet = str(body["source_wallet"])
            tx_hash = str(body["tx_hash"])
            ts = int(body["ts"])
        except (KeyError, TypeError, ValueError):
            return []
        return [
            ParsedSignal(
                condition_id=condition_id,
                side=outcome,
                rule_variant=None,
                metadata={"wallet": wallet, "tx_hash": tx_hash, "ts": ts},
            ),
        ]

    def quality_passes(self, parsed: ParsedSignal) -> bool:
        """No quality gate — watchlist admission is the gate."""
        del parsed
        return True

    def size(self, bankroll: float, parsed: ParsedSignal) -> float:
        """Return ``bankroll * position_fraction * concentration_multiplier``."""
        base = bankroll * self._config.position_fraction
        wallet = str(parsed.metadata.get("wallet", ""))
        return base * self._concentration_multiplier(wallet)

    def _concentration_multiplier(self, wallet: str) -> float:
        """Compute the per-wallet sizing multiplier in ``[min_multiplier, 1.0]``.

        ``share = trades_copied[wallet] / total_subgraph_copy_trades``.
        ``target_share = 1.0 / max(1, active_watchlist_size)``.
        ``raw = min(1.0, target_share / max(share, target_share))``.
        Final = ``max(raw, min_multiplier)``.
        """
        counts = self._paper_trades.count_by_source_wallet(detector=_DETECTOR_NAME)
        total = sum(counts.values())
        if total == 0:
            return 1.0
        wallet_lower = wallet.lower()
        wallet_count = sum(v for k, v in counts.items() if k.lower() == wallet_lower)
        share = wallet_count / total
        active_n = max(1, len(self._watchlist.list_active()))
        target_share = 1.0 / active_n
        raw = min(1.0, target_share / max(share, target_share))
        return max(raw, self._config.min_multiplier)
