"""Convenience wrapper for ``python -m src.evaluate.fact_audit``."""

from __future__ import annotations

import sys

from src.evaluate.fact_audit import main


if __name__ == "__main__":
    raise SystemExit(main())
