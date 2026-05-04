"""Manifold Markets data-ingest module.

Mana-denominated prediction markets (play money). Public REST and WebSocket
APIs require no authentication.

Sub-modules:
    ids      — ``ManifoldMarketId``, ``ManifoldUserId`` ``NewType`` wrappers.
    models   — Pydantic models for markets, bets, and users.
    client   — ``ManifoldClient``: async httpx wrapper with 500-rpm rate limit.
    ws       — ``ManifoldStream``: global bet firehose via WebSocket.
    db       — ``CREATE TABLE`` statements for ``manifold_*`` daemon tables.
    repos    — ``ManifoldMarketsRepo``, ``ManifoldBetsRepo``, ``ManifoldUsersRepo``.
"""
