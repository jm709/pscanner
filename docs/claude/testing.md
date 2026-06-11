# Test gotchas

Referenced from `CLAUDE.md`. Read this before writing or changing any tests.

- `pyproject.toml` has `filterwarnings = ["error"]` — every warning fails tests. Clean up resources (httpx/respx fixtures especially).
- NEVER `monkeypatch.setattr(asyncio, "sleep", AsyncMock())` — deadlocks the suite (sibling supervisor loops become CPU spinners). Use `FakeClock` from `pscanner.util.clock` instead; inject via `clock=` ctor kwarg, drive with `await fake_clock.advance(seconds)`.
- Shared fixtures in `tests/conftest.py`: `tmp_db` (in-memory SQLite with schema applied), `fake_clock`.
- Evaluator/collector mocks: prefer real `AlertsRepo` against `tmp_db` over MagicMock when testing dedupe / persistence behavior.
- **Structlog log assertions need `capture_logs`, NOT `caplog`.** `cli.py` configures structlog via `PrintLoggerFactory`, so stdlib `caplog` never sees structured events. Use `from structlog.testing import capture_logs; with capture_logs() as logs: ...; assert any(l["event"] == "..." for l in logs)`.
- `tmp_db` has `row_factory = sqlite3.Row`; rows compare via `tuple(r)`, not against raw tuples.
- **SQLite `UNIQUE` indexes treat NULLs as distinct.** The `paper_trades` unique-on-entry index uses `COALESCE(rule_variant, '')` so non-paired sources (`rule_variant=NULL`) keep per-key uniqueness while any future paired-leg evaluators can coexist. Don't strip the COALESCE.
- Many test files use the `# type: ignore[arg-type]  # ty:ignore[invalid-argument-type]` doubled annotation when passing string literals where `Literal` types are expected — `ty` doesn't honor mypy ignores so both are needed.
- `CorpusTradesRepo.insert_batch` silently filters out trades with `notional_usd < _NOTIONAL_FLOOR_USD` (default $10). Test fixtures inserting trades for downstream behavior must use ≥$10 notional or the rows won't land and the test will see an empty `corpus_trades`.
- For functions that read from a dict but don't mutate it, prefer `Mapping[str, object]` over `dict[str, object]` in the parameter type — `dict` is invariant in its value type so `dict[str, str]` doesn't satisfy `dict[str, object]`, but `Mapping` is covariant and accepts narrower-valued caller dicts.
