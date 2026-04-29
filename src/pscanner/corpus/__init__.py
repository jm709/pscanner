"""Historical trade corpus subsystem.

Builds a per-trade ML-trainable dataset from closed Polymarket markets.
Lives entirely separate from the live daemon: own SQLite file
(``data/corpus.sqlite3``), own CLI commands (``pscanner corpus ...``),
own runtime state.
"""
