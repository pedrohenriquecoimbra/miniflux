"""Stage 2 -- despiking with the median/MAD test (CONTRACT section 8, ALGORITHMS section 3).

One pass, one window over the whole averaging period, each variable on its own: the period
median and median absolute deviation set a symmetric bound, whatever falls strictly outside
it becomes NaN, and nothing refills it. That last point is the one to carry downstream --
every statistic after this step has to be NaN-aware, and every covariance has to use a
pairwise-complete mask, or the holes this step opens poison a flux silently.

The statistics themselves live in ``kernels``; what belongs here is the trailing-run rule,
which is about where the record ends rather than about the distribution.
"""

import logging

from . import kernels

logger = logging.getLogger(__name__)


def despike(period, cfg):
    """Replace spikes with NaN in each configured series and count them, in place.

    In: ``period[name]`` in that variable's own unit (m s-1 for wind, K for the sonic
    temperature, mol mol-1 or mol m-3 for the gases), raw -- despiking runs before
    rotation, lag and detrending so the threshold sees what the instrument wrote.

    Out: the same period. Each despiked series keeps its unit and its length, with flagged
    samples replaced by NaN, and ``period['meta']['n_spike_<name>']`` holds how many were
    flagged [samples]. Variables outside ``cfg.despike.variables`` are not touched and get
    no counter; ``enabled = false`` writes zero counters and leaves every series alone.

    Mauder, M., Cuntz, M., Drue, C., Graf, A., Rebmann, C., Schmid, H. P., Schmidt, M.,
    Steinbrecher, R. (2013). A strategy for quality and uncertainty assessment of long-term
    eddy-covariance measurements. Agric. For. Meteorol. 169, 122-135.
    """
    meta = period['meta']

    if not cfg.despike.enabled:
        # The counters are still owed. A blank column means "this variable was never
        # despiked"; a configured variable that was switched off this run has a count,
        # and it is zero.
        for name in cfg.despike.variables:
            meta['n_spike_' + name] = 0
        return period

    # Per variable, independently: one series' spikes never mask another's samples. Two
    # sensors rarely spike on the same sample, and a joint mask would throw away the good
    # half of every such pair.
    for name in cfg.despike.variables:
        x = period[name]
        # median, MAD, lower = m - (q*MAD)/0.6745, upper = m + (q*MAD)/0.6745, both
        # comparisons strict. The parenthesisation is part of the number, so it is pinned
        # in one place (kernels.mad_spike_mask) rather than repeated here.
        mask = kernels.mad_spike_mask(x, cfg.despike.q)
        _clear_trailing_run(mask)
        kernels.apply_mask_nan(x, mask)

        n_spike = sum(mask)
        meta['n_spike_' + name] = n_spike

        if n_spike and not kernels.count_finite(x):
            # Degenerate rather than impossible: a series whose only finite samples sit
            # outside the bounds, with the last sample already NaN so the trailing-run
            # rule gives none of them back. Report it and carry on with NaN.
            logger.warning(
                "%s: despiking left no finite sample in this period (%d flagged); "
                "every statistic taken from it will be NaN", name, n_spike)
        elif n_spike:
            logger.debug("%s: %d spikes flagged", name, n_spike)

    return period


def _clear_trailing_run(mask):
    """Un-flag the run of outliers still open at the last sample, in place.

    ``mask`` is the bytearray from :func:`kernels.mad_spike_mask`, 1 outside the bounds.

    A spike is an excursion that returns, and deciding that a run *was* one takes the
    return. A run still outside the bounds when the period ends never returns within this
    record: it is equally the start of a step, a drift, or a change in the mixing, and
    dropping it would erase real data. So it is left in the series.

    Only the closing end. A run at the *start* does return and is flagged; this is not a
    general "touches a boundary" rule. ``kernels.mad_spike_mask`` deliberately stops short
    of applying it, because the mask is a property of the distribution and this is a
    property of the record's edge.
    """
    if not mask or not mask[-1]:
        return
    i = len(mask) - 1
    while i >= 0 and mask[i]:
        i -= 1
    # i is the last sample inside the bounds, or -1 when every sample is outside them --
    # in which case the whole mask clears, and the period keeps all its samples.
    for j in range(i + 1, len(mask)):
        mask[j] = 0
