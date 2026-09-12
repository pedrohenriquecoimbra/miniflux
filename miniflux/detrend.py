"""Stage 5 -- detrending (ALGORITHMS section 6).

The two methods are not the same kind of operation, and the difference is the whole
design of this module:

* ``block`` is **bookkeeping**. Removing the period mean cannot change a covariance by
  one bit, because :func:`kernels.cov` demeans internally. So miniflux computes the mean,
  stores it, and leaves the series alone (CONTRACT section 20.6). The mean is the part
  that matters: the flux factor, the Schotanus correction and WPL all read means that a
  materialised block average would have thrown away.
* ``linear`` is **arithmetic**. It removes a real part of the covariance -- it is a
  different high-pass transfer function, and it gives a different flux -- so the fitted
  line is subtracted from the samples for real.

The line is fitted and evaluated on the **original sample index**, gaps included
(CONTRACT section 20.2). The parent fits against the index of the NaN-deleted array and
then evaluates on the full-length index, which stretches the trend by ``N/m``; the two
agree exactly when nothing is missing. :func:`kernels.linfit` owns that arithmetic.

Citations: block average, Moncrieff, Clement, Finnigan and Meyers (2004), in *Handbook
of Micrometeorology*; linear detrend, Rannik and Vesala (1999), *Boundary-Layer
Meteorol.* 91, 259-280.
"""

import logging
import math

from . import kernels

logger = logging.getLogger(__name__)


def detrend(period, cfg):
    """Record every pre-detrend mean and, for ``linear``, remove the fitted trend.

    period: the period dict of CONTRACT section 1, mutated in place and returned.
    cfg: the Config of CONTRACT section 5.

    Config read: ``[detrend] method`` (``block`` | ``linear``) and
    ``[detrend] variables``.

    Writes ``meta['<var>_mean']`` for every variable listed: the mean over that
    variable's finite samples, in the unit of the variable, taken **before** any trend is
    removed. Which is why ``config.py`` refuses ``ta`` and ``p_air`` in this list:
    ``ta_mean`` is the *resolved* air temperature ``flux.thermodynamics`` builds per
    sample, and a plain column mean written over it is what H, FC, LE and E would then
    read. ``ts_mean``, ``co2_mean`` and ``h2o_mean`` overwrite what
    ``flux.thermodynamics`` wrote over the same samples (CONTRACT section 17). The two
    agree to the reduction tolerance of CONTRACT 3.8, not bit for bit: this one is
    ``kernels.nanmean``, which sums pairwise on the numpy path, while thermodynamics
    accumulates its own left-to-right sum inside its per-sample loop. The overwrite is
    still harmless -- every consumer of those three means runs after this step, so they
    all read this value.

    Series data is modified only under ``linear``; under ``block`` the samples are left
    bit-identical.
    """
    meta = period['meta']
    for name in cfg.detrend.variables:
        x = period[name]
        # The mean is taken first, whichever method runs: it is the pre-detrend mean that
        # downstream stages need, and under `linear` the samples are about to change.
        mean = kernels.nanmean(x)
        meta[name + '_mean'] = mean
        if not math.isfinite(mean):
            # An all-NaN (or empty) series has no trend to remove and no mean to report.
            logger.warning('%s has no finite sample: %s_mean is NaN and no trend is '
                           'removed', name, name)
            continue
        # config.py has already refused any method other than block or linear, so the one
        # test here is the one that changes the samples.
        if cfg.detrend.method == 'linear':
            _remove_line(name, x)
    return period


def _remove_line(name, x):
    """Subtract the least-squares line of ALGORITHMS 6.2 from ``x``, in place.

    x: array('d'), any unit; modified in place. Nothing is returned.
    """
    m = kernels.count_finite(x)
    if m < 3:
        # Two points define a line exactly and one defines none, so a fit there would
        # report a trend the data cannot support; linfit returns (0.0, mean) instead.
        logger.warning('%s has %d finite samples, fewer than the 3 a trend needs: '
                       'removing the mean instead', name, m)
    slope, offset = kernels.linfit(x)
    logger.debug('%s linear detrend: slope %.6g per sample, offset %.6g', name, slope, offset)
    kernels.sub_line_inplace(x, slope, offset)
