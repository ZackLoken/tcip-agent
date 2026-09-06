"""``python -m tcip_web.cli``: a package's own ``__init__.py`` cannot be the ``-m`` target, so
this thin entry point is what a test (and an operator with no ``tcip`` console script installed
yet) actually spawns.
"""

from __future__ import annotations

import sys

from tcip_web.cli import main

if __name__ == "__main__":
    sys.exit(main())
