"""miniflux -- a minimal eddy-covariance flux program.

Raw high-frequency data in, one flux row per averaging period out, on the standard library
alone. Start at ``docs/CONTRACT.md`` for the plumbing and ``docs/ALGORITHMS.md`` for the
mathematics; start at ``pipeline.STEPS`` for the program.

Importing the package imports nothing else: the modules are independent, and numpy (an
optional speed switch, never a dependency) is probed for only when ``kernels`` is imported.
"""

__version__ = '0.1.0'
