"""The command line (CONTRACT section 16).

    python -m miniflux run   CONFIG.ini [--pure] [--limit N] [--log-level LEVEL]
    python -m miniflux check CONFIG.ini

``run`` computes the table; ``check`` parses and validates the configuration, echoes every
resolved value to stdout and reads no data at all -- so a config can be corrected before a
long run starts.

Exit codes: 0 success, 2 a ConfigError, 3 a ReadError, 1 anything else. They are distinct so
a wrapper script can tell "you wrote the config wrong" from "the data is not what the config
says it is" without parsing the log.
"""

import argparse
import logging
import sys

from . import __version__, config, kernels, pipeline, write
from .errors import ConfigError, ReadError

logger = logging.getLogger(__name__)

LOG_FORMAT = '%(levelname)s %(name)s: %(message)s'
LOG_LEVELS = ('DEBUG', 'INFO', 'WARNING', 'ERROR')

# The one stderr handler this process installs, kept so a second call to _configure_logging
# moves the level instead of doubling every line.
_handler = None


def main(argv=None):
    """Run the command line. argv: list of str, or None for sys.argv. Returns the exit code."""
    args = _parse(argv)
    # A default stream is installed before the config is read, so that a ConfigError is
    # reported through the same channel as everything else.
    _configure_logging(args.log_level or 'INFO')
    try:
        cfg = config.load(args.config)
        if args.log_level is None:
            _configure_logging(cfg.runtime.log_level)
        if args.pure:
            _force_pure(cfg)
        if args.command == 'check':
            # The point of `check` is to show the resolved configuration, so it goes to
            # stdout where it can be piped or diffed, whatever the log level is.
            sys.stdout.write(config.describe(cfg) + '\n')
            logger.info('%s is valid; no data was read', args.config)
            return 0
        logger.info('configuration:\n%s', config.describe(cfg))
        with write.Writer(cfg.files.output_csv, cfg) as writer:
            summary = pipeline.run(cfg, writer, limit=args.limit)
        logger.info('summary: %s', summary)
        logger.info('output: %s (units in %s)',
                    cfg.files.output_csv, write.units_path(cfg.files.output_csv))
        return 0
    except ConfigError as exc:
        logger.error('configuration: %s', exc)
        return 2
    except ReadError as exc:
        logger.error('input: %s', exc)
        return 3
    except Exception:
        logger.exception('miniflux stopped on an unexpected error')
        return 1


def _parse(argv):
    """Build the two subcommands and parse argv. Returns the argparse namespace."""
    parser = argparse.ArgumentParser(
        prog='miniflux', description='A minimal eddy-covariance flux program.')
    parser.add_argument('--version', action='version', version='miniflux ' + __version__)
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True      # 3.8 has no `required=` on add_subparsers

    run_parser = subparsers.add_parser('run', help='compute the flux table')
    run_parser.add_argument('config', metavar='CONFIG.ini', help='the configuration file')
    run_parser.add_argument('--pure', action='store_true',
                            help='force the pure-Python kernels for this run')
    run_parser.add_argument('--limit', type=_positive, metavar='N',
                            help='stop after N periods (a smoke run)')
    run_parser.add_argument('--log-level', choices=LOG_LEVELS,
                            help='override [runtime] log_level')

    check_parser = subparsers.add_parser(
        'check', help='validate the configuration and echo it; read no data')
    check_parser.add_argument('config', metavar='CONFIG.ini', help='the configuration file')
    check_parser.add_argument('--log-level', choices=LOG_LEVELS,
                              help='override [runtime] log_level')
    check_parser.set_defaults(pure=False, limit=None)

    return parser.parse_args(argv)


def _positive(text):
    """argparse type for --limit: a whole number of periods, at least one."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError('must be 1 or more, got %d' % value)
    return value


def _configure_logging(level):
    """Send the package's log to stderr at `level`. level: str.

    Safe to call twice, which it is: the level is settled once from ``--log-level`` and once
    from ``[runtime] log_level``. The second call only moves the level, because a second
    handler would print every line twice, and removing handlers we did not install would
    take away a host application's.
    """
    global _handler
    package = logging.getLogger(__package__)
    if _handler is None:
        _handler = logging.StreamHandler(sys.stderr)
        _handler.setFormatter(logging.Formatter(LOG_FORMAT))
    if _handler not in package.handlers:
        package.addHandler(_handler)
        package.propagate = False   # the root logger's handlers are not ours to write to
    package.setLevel(getattr(logging, level))


def _force_pure(cfg):
    """Apply --pure: run on the pure-Python kernels whatever the file says.

    The flag is applied after config.load, so a file that demands numpy on a machine without
    it is still refused -- that config is wrong however this run is invoked.
    """
    cfg.runtime.use_numpy = 'off'
    cfg.runtime.numpy_enabled = False
    kernels.set_numpy(False)
    logger.info('--pure: the numpy fast path is off for this run')
