"""Nightshift: post-training research agents with an outer improvement loop."""

import os

__version__ = "0.1.0"
# Honest code identity: an operator/build may stamp the exact revision via
# the environment; otherwise only the package version is claimed.
CODE_REVISION = os.environ.get("NIGHTSHIFT_CODE_REVISION", "").strip() or __version__

__all__ = ["__version__", "CODE_REVISION"]
