"""Shared corpus-DB loaders for the scheduler and the bootstrap CLI.

Both ``Scanner.__init__`` (live daemon cold-start) and ``run_bootstrap``
(the ``pscanner daemon bootstrap-features`` CLI) need to read
``corpus_markets`` and ``market_resolutions`` to pre-warm the live
history tables. Previously the same SELECTs lived in both places
(``scheduler._load_corpus_metadata`` / ``_load_corpus_resolutions`` was
textually identical to ``bootstrap._load_metadata`` + the inline
resolutions walk in ``run_bootstrap``).
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Protocol

import structlog

from pscanner.corpus.features import MarketMetadata

_LOG = structlog.get_logger(__name__)

DEFAULT_CORPUS_DB = Path("data/corpus.sqlite3")
"""Filesystem path of the corpus DB. Hard-coded constant — historically
hard-coded at three call sites in ``scheduler.py`` and one in
``bootstrap.py``."""


class _ResolutionConsumer(Protocol):
    """Minimal contract for ``register_resolution``.

    Satisfied by both ``StreamingHistoryProvider`` and
    ``LiveHistoryProvider``.
    """

    def register_resolution(
        self,
        *,
        condition_id: str,
        resolved_at: int,
        outcome_yes_won: int,
    ) -> None: ...


def load_corpus_metadata(
    *,
    conn: sqlite3.Connection | None = None,
    corpus_path: Path = DEFAULT_CORPUS_DB,
    platform: str = "polymarket",
) -> dict[str, MarketMetadata]:
    """Read ``corpus_markets`` rows for ``platform`` into a metadata dict.

    When ``conn`` is None, opens a short-lived connection to
    ``corpus_path`` (used by the scheduler's cold-start). Otherwise reads
    on the caller's connection (used by ``run_bootstrap`` which already
    holds an open handle). Missing-file fallback returns an empty dict
    and emits a warning log; the consumer (``LiveHistoryProvider``)
    treats KeyError on missing metadata by skipping the trade.
    """
    if conn is not None:
        return _read_metadata(conn, platform=platform)
    if not corpus_path.exists():
        _LOG.warning("corpus_loader.corpus_db_missing", path=str(corpus_path))
        return {}
    with closing(sqlite3.connect(str(corpus_path))) as own_conn:
        return _read_metadata(own_conn, platform=platform)


def load_corpus_resolutions_into(
    provider: _ResolutionConsumer,
    *,
    conn: sqlite3.Connection | None = None,
    corpus_path: Path = DEFAULT_CORPUS_DB,
    platform: str = "polymarket",
) -> int:
    """Walk ``market_resolutions`` for ``platform`` and register each in ``provider``.

    Returns the count registered. Open-own-conn semantics mirror
    :func:`load_corpus_metadata`. On an unmigrated corpus DB (predates the
    ``market_resolutions.platform`` column), logs and returns 0 — the
    daemon will run with empty live-resolutions until a bootstrap or
    schema migration repairs the DB.
    """
    if conn is not None:
        return _drain_resolutions(conn, provider, platform=platform)
    if not corpus_path.exists():
        _LOG.warning("corpus_loader.corpus_db_missing", path=str(corpus_path))
        return 0
    with closing(sqlite3.connect(str(corpus_path))) as own_conn:
        return _drain_resolutions(own_conn, provider, platform=platform)


def _read_metadata(
    conn: sqlite3.Connection,
    *,
    platform: str,
) -> dict[str, MarketMetadata]:
    out: dict[str, MarketMetadata] = {}
    for cond_id, category, closed_at, opened_at in conn.execute(
        """
        SELECT condition_id,
               COALESCE(category, ''),
               COALESCE(closed_at, 0),
               COALESCE(enumerated_at, 0)
        FROM corpus_markets
        WHERE platform = ?
        """,
        (platform,),
    ):
        out[cond_id] = MarketMetadata(
            condition_id=cond_id,
            category=category,
            closed_at=int(closed_at),
            opened_at=int(opened_at),
        )
    return out


def _drain_resolutions(
    conn: sqlite3.Connection,
    provider: _ResolutionConsumer,
    *,
    platform: str,
) -> int:
    n = 0
    try:
        cursor = conn.execute(
            """
            SELECT condition_id, resolved_at, outcome_yes_won
            FROM market_resolutions
            WHERE platform = ?
            """,
            (platform,),
        )
    except sqlite3.OperationalError as exc:
        _LOG.warning(
            "corpus_loader.corpus_db_unmigrated_market_resolutions",
            err=str(exc),
        )
        return 0
    for cond_id, resolved_at, yes_won in cursor:
        provider.register_resolution(
            condition_id=cond_id,
            resolved_at=int(resolved_at),
            outcome_yes_won=int(yes_won),
        )
        n += 1
    _LOG.info("corpus_loader.resolutions_loaded", count=n, platform=platform)
    return n
