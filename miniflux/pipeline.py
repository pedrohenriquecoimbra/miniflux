"""Apply the steps to every period, in the one order that is physically defensible.

``STEPS`` is the whole program: read one period, push it through nine functions, hand the
result to the writer. Adding a processing stage to miniflux means writing a
``step(period, cfg) -> period`` function and putting its name in this list; there is no
registry and nothing else to update (CONTRACT section 1.1).

The order below is normative -- see the comment on each line for what breaks if it moves.
"""

import logging

from . import despike, detrend, flux, lag, qc, read, rotate, wpl

logger = logging.getLogger(__name__)

#: ``(name, function)`` in execution order; the name is what a failure is reported under.
STEPS = [
    # First, because the median/MAD threshold has to see raw values: rotation and detrending
    # both reshape the distribution the spike test is calibrated against.
    ('despike', despike.despike),
    # Before lag, because the covariance the lag search maximises is taken on the rotated w.
    ('rotate', rotate.rotate),
    # Before thermodynamics, so every period mean is taken over the samples that actually
    # enter the covariances -- the lag shift truncates and NaN-fills the ends.
    ('lag', lag.apply_lags),
    # Before detrend, because detrending destroys exactly the means WPL and the molar flux
    # factor need (CONTRACT section 20.7).
    ('thermodynamics', flux.thermodynamics),
    # Before moments: a block average changes no covariance, a linear detrend does.
    ('detrend', detrend.detrend),
    ('moments', flux.moments),
    # Before wpl, because WPL consumes the Schotanus-corrected H, not H_L0.
    ('assemble', flux.assemble),
    ('wpl', wpl.correct),
    # Last, because the ITC test needs z/L, which assemble creates. Run earlier it would
    # silently grade the period against the neutral wind model.
    ('qc', qc.quality),
]


def run_period(period, cfg):
    """Apply every step of STEPS to one period, in order.

    period: the dict of CONTRACT section 1, as read.py built it. cfg: the Config of section
    5. Returns the period, with each step's meta keys added.

    A step that raises is caught here and only here: the period is logged at ERROR, the
    remaining steps are skipped, and the partial period is returned so that the caller still
    writes a row. A failed period is then visible in the output table as a line of na_value
    rather than as a hole between two timestamps.
    """
    for name, step in STEPS:
        try:
            period = step(period, cfg)
        except Exception:
            logger.error('step %s failed on period %s; the remaining steps are skipped and '
                         'the row is written from the meta computed so far',
                         name, _label(period), exc_info=True)
            break
    return period


def run(cfg, writer=None, limit=None):
    """Read every period the configuration points at, process it, and write it.

    cfg: the Config of CONTRACT section 5. writer: a write.Writer, or None to process
    without writing (a dry run). limit: stop after this many periods, or None for all of
    them -- the CLI's ``--limit`` for a smoke run.

    Returns the summary dict ``{'periods', 'skipped_short', 'rows', 'warnings'}``: periods
    read, periods read.py dropped for holding fewer than ``[period] min_samples`` samples,
    rows written, and log records at WARNING or above raised anywhere in the package while
    the run was in progress. Only the last of the four counts log records, and it says so;
    the other three are facts about the data and hold at every ``[runtime] log_level``.
    """
    summary = {'periods': 0, 'skipped_short': 0, 'rows': 0, 'warnings': 0}
    counter = _count_records(summary)
    loggers = _package_loggers()
    for one in loggers:
        one.addFilter(counter)
    try:
        # read.py counts the periods it drops straight into `summary`: they are never
        # yielded, and the warning it also logs does not exist at log_level = ERROR.
        for period in read.periods(cfg, summary):
            summary['periods'] += 1
            period = run_period(period, cfg)
            if writer is not None:
                writer.write(period)
                summary['rows'] += 1
            if limit is not None and summary['periods'] >= limit:
                logger.info('stopping after %d period(s): --limit', limit)
                break
    finally:
        for one in loggers:
            one.removeFilter(counter)
    return summary


def _label(period):
    """A period's name for a log line: its end timestamp, or a placeholder. Returns str."""
    try:
        return str(period['meta'].get('period_end', '<unlabelled>'))
    except (TypeError, KeyError):
        return '<unlabelled>'


def _count_records(summary):
    """A logging filter that counts records into `summary` and filters nothing out.

    A filter rather than a handler because miniflux installs no handlers of its own: how the
    records are rendered is the CLI's business, counting them is the run's.
    """
    def counter(record):
        if record.levelno >= logging.WARNING:
            summary['warnings'] += 1
        return True
    return counter


def _package_loggers():
    """Every logger already created inside the miniflux package. Returns list of Logger.

    A filter on the parent logger would not see these records: propagation calls an
    ancestor's handlers, never its filters, so the counter has to sit on each module's own
    logger. Every module logger exists by now -- importing this one imported them all.
    """
    root = logging.getLogger(__package__)
    names = [name for name in list(logging.Logger.manager.loggerDict)
             if name == root.name or name.startswith(root.name + '.')]
    return [logging.getLogger(name) for name in sorted(names)]
