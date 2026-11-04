"""``python -m nightshift ...`` is the same as the ``nightshift`` command."""
from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
