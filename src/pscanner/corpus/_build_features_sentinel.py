"""Crash-recovery sentinel for ``pscanner corpus build-features``.

Writes ``corpus_state['build_features_in_progress']`` at run start, clears
on success. A prior crashed run leaves the key set; the CLI refuses to
proceed without ``--force``.
"""

from __future__ import annotations

from typing import Final

from pscanner.corpus.repos import CorpusStateRepo

SENTINEL_KEY: Final[str] = "build_features_in_progress"


class SentinelAlreadySetError(RuntimeError):
    """Raised when the sentinel is set and ``--force`` was not supplied."""


def check_and_set_sentinel(repo: CorpusStateRepo, *, now_ts: int, force: bool) -> None:
    """Write the sentinel; raise if already set unless ``force`` is True.

    The stored value is the run-start Unix timestamp as a string, so a
    ``--force`` recovery prints when the stuck run started.
    """
    existing = repo.get(SENTINEL_KEY)
    if existing is not None and not force:
        raise SentinelAlreadySetError(
            f"build_features in-progress sentinel is set "
            f"(started at ts={existing}). A prior run crashed mid-rebuild. "
            f"Inspect ./data/corpus.sqlite3 and re-run with --force to override."
        )
    repo.set(SENTINEL_KEY, str(now_ts), updated_at=now_ts)


def clear_sentinel(repo: CorpusStateRepo) -> None:
    """Remove the sentinel after a successful run."""
    repo.delete(SENTINEL_KEY)
