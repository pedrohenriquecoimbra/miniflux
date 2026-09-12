"""Entry point for ``python -m miniflux``. The command line itself lives in cli.py."""

import sys

from .cli import main

if __name__ == '__main__':
    sys.exit(main())
