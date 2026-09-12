"""The three exceptions miniflux raises, and nothing else.

miniflux refuses only when continuing would produce a number that looks right and is
not: a fabricated pressure, an inferred timestamp format, an unsupported measure type,
a closed-path input, an uncorrected flux under a corrected name. Everything else is
reported as NaN and flagged.

Only ``config.py`` raises :class:`ConfigError` and only ``read.py`` raises
:class:`ReadError`. No step module raises at all; a step meets degeneracy by writing
``float('nan')`` and logging a warning.
"""


class MinifluxError(Exception):
    """Base class for every error miniflux raises on purpose."""


class ConfigError(MinifluxError):
    """The configuration cannot be honoured as written. CLI exit code 2."""


class ReadError(MinifluxError):
    """The input file cannot be read as the configuration describes it. CLI exit code 3."""
