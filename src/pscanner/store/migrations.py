"""Shared migration utilities for pscanner SQLite modules.

Replaces four near-identical ``_apply_migrations`` loops (one per platform DB
module) with a single helper that swallows the standard set of idempotent
errors raised when a migration has already been applied.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable

_EXPECTED_IDEMPOTENT_FAILURES: tuple[str, ...] = (
    "duplicate column name",  # ADD COLUMN re-applied
    "no such column",  # RENAME COLUMN re-applied (source side gone)
    "no such table",  # ALTER against a not-yet-created table
)


def apply_additive_migrations(
    conn: sqlite3.Connection,
    migrations: Iterable[str],
) -> None:
    """Apply additive ALTER TABLE migrations in order. Idempotent.

    Each statement is executed via ``conn.execute``; any
    :class:`sqlite3.OperationalError` whose lowercased message matches an entry
    in :data:`_EXPECTED_IDEMPOTENT_FAILURES` is swallowed so re-runs are safe.
    All other ``OperationalError`` instances propagate.

    Args:
        conn: Open SQLite connection. Caller owns the lifecycle.
        migrations: Iterable of SQL statements (usually a module-private
            ``_MIGRATIONS`` tuple).
    """
    for stmt in migrations:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError as exc:
            if any(s in str(exc).lower() for s in _EXPECTED_IDEMPOTENT_FAILURES):
                continue
            raise
    conn.commit()
