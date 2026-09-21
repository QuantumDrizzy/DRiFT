"""Rebuild the checked-in instance-bank manifest.

    python -m drift.benchmarks
"""

from .bank import main

raise SystemExit(main())
