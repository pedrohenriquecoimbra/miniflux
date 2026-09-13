"""Stage 1b -- a closed-path cell density becomes a dry mixing ratio (Ibrom et al. 2007).

A closed-path analyser does not measure the air at the tower. It measures the air *in
its own cell*, drawn down a tube, warmed by the instrument and dropped in pressure by
the pump. A number it reports as a molar density is a density in that cell, at the cell
temperature and the cell pressure -- not an ambient density, and not a quantity any
ambient correction describes.

The conversion back to something conserved is one line per sample::

    r = n_gas * V_cell / (1 - chi_h2o)        with  V_cell = R * T_cell / P_cell

``n_gas * V_cell`` is the sample's mole fraction; dividing by ``1 - chi_h2o``
re-expresses it per mole of *dry* air. A dry mixing ratio is conserved under exactly the
expansion and dilution that a density correction exists to undo, so once this has run
there is nothing left for WPL to do -- which is why an ambient WPL is not applied to a
closed-path gas. See ALGORITHMS.md section 2A for why that is the right answer and not a
missing feature.

**Per sample, never on the means.** It is precisely the fluctuations of cell temperature,
pressure and humidity that carry the spurious density signal; a conversion applied to a
block mean removes none of it.

The clean case does not come through here at all. An LI-7200 reports ``CO2_DRY`` and
``H2O_DRY`` beside its densities, and declaring those is exact, needs no cell state and
leaves this module with nothing to do. This route exists for the files that carry only
the density.

Citation: Ibrom, A., Dellwik, E., Larsen, S. E., Pilegaard, K. (2007). On the use of the
Webb-Pearman-Leuning theory for closed-path eddy correlation measurements. *Tellus B*
59, 937-946.

See ALGORITHMS.md section 2A and CONTRACT.md section 7A.
"""

import logging
import math

from . import kernels
from .constants import R

logger = logging.getLogger(__name__)

NAN = kernels.NAN

#: Per-sample cell quantities this step averages and publishes, in the order they are
#: accumulated. ``t_cell`` and ``p_cell`` are the means of the series as read; the other
#: two are means of quantities that exist only inside the conversion.
_MEAN_KEYS = ('t_cell', 'p_cell', 'v_cell', 'chi_h2o_cell')


def convert(period, cfg):
    """Rewrite each closed-path cell molar density as a dry mixing ratio, in place.

    Reads ``cfg.gases.convert_cell`` (which gases, decided once at config time),
    ``cfg.gases.reported['h2o']``, and the ``t_cell`` / ``p_cell`` / ``h2o`` series.
    Writes ``t_cell_mean`` [K], ``p_cell_mean`` [Pa], ``v_cell_mean`` [m3 mol-1],
    ``chi_h2o_cell_mean`` [mol mol-1] and ``n_cell_converted`` [samples].

    Nothing else in the program is told that a conversion happened. ``config.py`` has
    already resolved the converted gas's effective measure type to ``mixing_ratio``, so
    the flux factor, the mean density and whether WPL is owed all follow from the one
    vocabulary they already read.

    A sample whose cell state is unusable becomes NaN rather than a guess. Returns the
    same period dict.
    """
    meta = period['meta']
    gases = cfg.gases.convert_cell
    if not gases:
        return period

    t_cell = period['t_cell']                          # [K]
    p_cell = period['p_cell']                          # [Pa]
    h2o = period['h2o']
    h2o_is_density = cfg.gases.reported['h2o'] == 'molar_density'
    series = [period[gas] for gas in gases]

    sums = dict((name, 0.0) for name in _MEAN_KEYS)
    counts = dict((name, 0) for name in _MEAN_KEYS)
    converted = 0
    for i in range(len(t_cell)):
        t_k = t_cell[i]                                # [K] the CELL's temperature
        p_pa = p_cell[i]                               # [Pa] the CELL's pressure
        # A cell temperature or pressure of zero is a broken instrument, not a degenerate
        # measurement, and an infinite molar volume is not a more honest answer than no
        # answer. NaN fails both comparisons, so this one test covers both.
        v_cell = R * t_k / p_pa if t_k > 0.0 and p_pa > 0.0 else NAN   # [m3 mol-1]

        # The water in the cell, as a fraction of the MOIST air there. Read before the
        # write loop below, because h2o is itself one of the series being rewritten
        # whenever the analyser reports it as a density.
        water = h2o[i]
        chi = water * v_cell if h2o_is_density else _moist_fraction(water)
        dry = 1.0 - chi                                # [-] moles of dry air per mole
        factor = v_cell / dry if dry > 0.0 else NAN    # [m3 mol-1 of dry air]
        if math.isfinite(factor):
            converted += 1
        else:
            # An infinity would survive into a covariance and take a whole flux with it,
            # where a NaN drops only the samples it touches.
            factor = NAN

        for column in series:
            column[i] = column[i] * factor             # [mol m-3] -> [mol mol-1 dry]

        for name, value in (('t_cell', t_k), ('p_cell', p_pa),
                            ('v_cell', v_cell), ('chi_h2o_cell', chi)):
            if math.isfinite(value):
                sums[name] += value
                counts[name] += 1

    for name in _MEAN_KEYS:
        meta[name + '_mean'] = sums[name] / counts[name] if counts[name] else NAN
    meta['n_cell_converted'] = converted

    n = len(t_cell)
    if converted < n:
        logger.warning('cell: %d of %d sample(s) carried no usable cell state (T_cell, '
                       'P_cell, or a cell water fraction at or above 1); %s is NaN '
                       'there', n - converted, n, ' and '.join(gases))
    return period


def _moist_fraction(mixing_ratio):
    """A dry mixing ratio as a fraction of moist air: ``r / (1 + r)``. [-] -> [-].

    Used when the analyser reports its water as the dry mixing ratio (an LI-7200's
    ``H2O_DRY``) while reporting the other gas as a cell density: the dilution factor
    the conversion divides by is a *moist* fraction either way.
    """
    denominator = 1.0 + mixing_ratio
    return mixing_ratio / denominator if denominator > 0.0 else NAN
