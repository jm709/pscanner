"""Allow ``python -m pscanner`` to invoke the same CLI as the console script."""

from __future__ import annotations

import sys

from pscanner.cli import main

if __name__ == "__main__":
    sys.exit(main())
