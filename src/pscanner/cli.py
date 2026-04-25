"""CLI entrypoint stub — Wave 3 (``integration`` agent) implements ``main``.

The entrypoint is declared in ``pyproject.toml`` so ``uv sync`` produces a
``pscanner`` console script today. Calling it raises ``NotImplementedError``
until Wave 3 wires up the scheduler.
"""

from __future__ import annotations


def main() -> None:
    """Run the pscanner CLI. Wave 3 will implement subcommands.

    Raises:
        NotImplementedError: Always, until Wave 3 ships.
    """
    raise NotImplementedError("Wave 3: integration")
