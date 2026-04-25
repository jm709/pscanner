"""Smoke test for the ``Collector`` protocol contract."""

from __future__ import annotations

import asyncio

from pscanner.collectors.base import Collector


class _Dummy:
    name: str = "dummy"

    async def run(self, stop_event: asyncio.Event) -> None:
        await stop_event.wait()


def test_collector_protocol_is_runtime_checkable() -> None:
    assert isinstance(_Dummy(), Collector)


def test_non_collector_fails_isinstance_check() -> None:
    class _Missing:
        name = "missing"

    assert not isinstance(_Missing(), Collector)
