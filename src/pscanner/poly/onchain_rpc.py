"""Async JSON-RPC client for Polygon mainnet (eth_* methods).

Targets any EVM-compatible RPC endpoint. Default is Polygon Foundation's
public RPC (``https://polygon-rpc.com/``), free and unauthenticated.
Override via constructor for Alchemy or other providers.

Wraps :class:`pscanner.util.http_client.RateLimitedHttpClient` for the
token-bucket + retry + Retry-After plumbing, then layers the JSON-RPC
request/response shape on top.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any, Self

from pscanner.util.http_client import RateLimitedHttpClient

_LOG_EVENT = "polygon_rpc_retry"
_BLOCK_TIMESTAMP_CACHE_SIZE = 4096


class OnchainRpcClient:
    """Async JSON-RPC client for ``eth_*`` calls against any Polygon RPC.

    Long-lived: open once, reuse across an ingest run, close on shutdown.
    Underlying httpx client + token bucket are constructed lazily on first
    use so construction is cheap.
    """

    def __init__(
        self,
        *,
        rpc_url: str,
        rpm: int,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Store config without opening any sockets.

        Args:
            rpc_url: Full RPC endpoint URL (e.g. ``https://polygon-rpc.com/``).
            rpm: Requests-per-minute budget.
            timeout_seconds: Default per-request timeout.
        """
        self.rpc_url = rpc_url
        self.rpm = rpm
        self.timeout_seconds = timeout_seconds
        self._inner = RateLimitedHttpClient(
            rpm=rpm,
            timeout_seconds=timeout_seconds,
            log_event=_LOG_EVENT,
        )
        self._next_id = 1
        self._ts_cache: dict[int, int] = {}

    async def _call(self, method: str, params: list[Any]) -> Any:
        """Issue a single JSON-RPC call with tenacity retry."""
        request_id = self._next_id
        self._next_id += 1
        body = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        response = await self._inner.post(self.rpc_url, json=body)
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(f"RPC error from {method}: {payload['error']}")
        return payload["result"]

    async def get_block_number(self) -> int:
        """Return the current Polygon head block number."""
        result = await self._call("eth_blockNumber", [])
        return int(result, 16)

    async def get_logs(
        self,
        *,
        address: str,
        topics: list[str],
        from_block: int,
        to_block: int,
    ) -> list[dict[str, Any]]:
        """Fetch logs matching ``address`` and ``topics`` between two block bounds.

        Args:
            address: Contract address (lowercase or checksummed; RPC accepts both).
            topics: Topic filter; ``topics[0]`` is the event signature hash.
            from_block: First block in the inclusive range.
            to_block: Last block in the inclusive range.

        Returns:
            List of raw log dicts as returned by the RPC.

        Raises:
            RuntimeError: If the RPC returns a JSON-RPC error.
            httpx.HTTPStatusError: On non-2xx HTTP status.
        """
        params: list[Any] = [
            {
                "address": address,
                "topics": topics,
                "fromBlock": hex(from_block),
                "toBlock": hex(to_block),
            }
        ]
        result = await self._call("eth_getLogs", params)
        if not isinstance(result, list):
            raise RuntimeError(f"eth_getLogs returned non-list result: {result!r}")
        return result

    async def get_block_timestamp(self, block_number: int) -> int:
        """Return the Unix-second timestamp of the given Polygon block.

        Caches the ``(block_number -> timestamp)`` mapping in-memory; capped to
        ``_BLOCK_TIMESTAMP_CACHE_SIZE`` entries. When the cap is hit the oldest
        insertion is evicted (FIFO — Polygon walk is forward-monotonic so older
        blocks rarely re-appear).

        Args:
            block_number: Polygon block height to look up.

        Returns:
            Unix timestamp in seconds.
        """
        cached = self._ts_cache.get(block_number)
        if cached is not None:
            return cached
        result = await self._call("eth_getBlockByNumber", [hex(block_number), False])
        if not isinstance(result, dict) or "timestamp" not in result:
            raise RuntimeError(
                f"eth_getBlockByNumber({block_number}) returned malformed payload: {result!r}"
            )
        ts = int(result["timestamp"], 16)
        if len(self._ts_cache) >= _BLOCK_TIMESTAMP_CACHE_SIZE:
            oldest = next(iter(self._ts_cache))
            del self._ts_cache[oldest]
        self._ts_cache[block_number] = ts
        return ts

    async def aclose(self) -> None:
        """Close the underlying :class:`httpx.AsyncClient` and release sockets."""
        await self._inner.aclose()

    async def __aenter__(self) -> Self:
        """Async context-manager entry — returns ``self`` for the with-block."""
        await self._inner.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Async context-manager exit — calls :meth:`aclose`."""
        await self._inner.__aexit__(exc_type, exc, tb)
