# pscanner

Polymarket scanner — a persistent asyncio daemon that watches Polymarket for three
trading signals:

1. **Smart money** — high-conviction positions from wallets with proven winrate.
2. **Mispricing** — mutex outcomes within an event whose YES prices don't sum to 1.
3. **Whales** — newly-active accounts placing outsized bets on illiquid markets.

Output goes to a SQLite log at `./data/pscanner.sqlite3` and to a live `rich`
terminal table.

## Install

Requires Python 3.13.

```bash
uv sync
cp config.toml.example config.toml  # edit thresholds as needed
```

## Run

```bash
uv run pscanner            # long-running daemon (Wave 3 will wire this up)
uv run pscanner --once     # single pass and exit
```

## Develop

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest -q
```
