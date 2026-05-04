"""Kalshi ingest module for pscanner.

Provides REST client, pydantic models, SQLite repos, and typed identifiers
for the Kalshi prediction-market platform. REST-only (Stage 1); WebSocket
streaming with RSA-signed auth is deferred to Stage 2.

See: https://api.elections.kalshi.com/trade-api/v2/
"""
