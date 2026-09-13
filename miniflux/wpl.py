"""Stage 9 -- the WPL density correction (Webb, Pearman & Leuning 1980).

An open-path analyser that reports a gas *density* does not report a conserved quantity.
Warm the air and it expands; add water vapour and it displaces dry air. Either makes the
reported number fall with no gas having gone anywhere, so the eddy covariance of that
number carries a flux of heat and a flux of water alongside the flux of gas. Webb, Pearman
and Leuning separate them with the constraint that the mean flux of *dry air* through the
surface is zero.

Citation: Webb, E. K., Pearman, G. I., Leuning, R. (1980). Correction of flux measurements
for density effects due to heat and water vapour transfer. *Q. J. R. Meteorol. Soc.* 106,
85-100. Eq. 24 (gas) and Eq. 25 (vapour).

Every quantity this module needs is a scalar that an earlier step has already put in
``period['meta']``. It computes no mean and no covariance of its own, and that is
deliberate -- see the landmine note in :func:`_apply`.

**A closed-path gas is never given this correction, and that is the correct answer
rather than a missing feature.** Webb's derivation is about the expansion and the
dilution of *ambient* air at the sampling point; the air an LI-7200 measures has been
through a tube and a pump, so the fluctuations this equation would remove are the ones
the cell has already damped or imposed. The closed-path treatment is the per-sample
conversion to a dry mixing ratio in ``cell.py`` instead, after which the gas is
conserved and owes nothing. Nothing in this module tests for a cell: ``config.py``
resolves such a gas's effective measure type to ``mixing_ratio``, and the
``molar_density`` branches below simply never see it (ALGORITHMS 10.1).

See ALGORITHMS.md section 10 and CONTRACT.md section 12.
"""

import logging
import math

from .constants import MCO2, MU, MV

logger = logging.getLogger(__name__)

NAN = float('nan')

# Every branch of the correction scales with, or divides by, each of these. A missing one
# does not make the correction smaller, it makes it wrong (ALGORITHMS 10.7), so the whole
# correction is refused rather than applied in part.
REQUIRED = ('h', 'rho_m_mean', 'cp_mean', 'rho_v_mean', 'rho_d_mean', 'ta_mean',
            'lambda_v_mean', 'chi_v_mean')

# The four of them that appear in a denominator. A zero is as unusable as a NaN here -- a
# mean air density, heat capacity or absolute temperature of exactly zero is not a
# degenerate measurement but a broken one -- and it is declined with the others rather
# than divided by, because a step module may never raise (CONTRACT 19) and `qc` runs after
# this one: an exception here would cost the period its whole quality block as well.
DIVISORS = ('rho_m_mean', 'cp_mean', 'ta_mean', 'rho_d_mean')


def correct(period, cfg):
    """Report the WPL-corrected fluxes, or decline to report one at all.

    Reads `[wpl] enabled`, `[gases] *_measure_type` and the period means and `_L0` fluxes
    in ``period['meta']``. Writes ``fc`` [mol m-2 s-1], ``e`` [kg m-2 s-1], ``le`` [W m-2],
    ``wpl_applied`` [bool], and ``wt`` [K m s-1] whenever the correction runs.

    A gas reported as a molar density is owed the correction; a dry mixing ratio is per
    mole of dry air, hence already conserved, and owes none -- whether it was reported
    that way or converted from a cell density by ``cell.py``, which is why this reads
    ``cfg.gases.measure_type`` (the effective type) and not ``cfg.gases.reported``. When
    a gas is owed one and
    `enabled = off`, its corrected key is left absent: `write.py` emits `na_value` and no
    uncorrected number leaves the program wearing a corrected name.

    Returns the same period dict.
    """
    meta = period['meta']
    owed_co2 = cfg.gases.measure_type['co2'] == 'molar_density'
    owed_h2o = cfg.gases.measure_type['h2o'] == 'molar_density'
    enabled = cfg.wpl.enabled

    if enabled == 'on' or (enabled == 'auto' and (owed_co2 or owed_h2o)):
        _apply(meta, owed_co2, owed_h2o)
    elif enabled == 'off' and (owed_co2 or owed_h2o):
        if owed_co2:
            logger.warning(
                'wpl: co2 is a molar density and is owed a density correction, but '
                '[wpl] enabled = off; no FC is reported for this period. FC_L0 is kept.')
        if owed_h2o:
            logger.warning(
                'wpl: h2o is a molar density and is owed a density correction, but '
                '[wpl] enabled = off; no E or LE is reported for this period. '
                'E_L0 and LE_L0 are kept.')
        _report_l0(meta, owed_co2, owed_h2o)
    else:
        # auto or off with nothing owed: the _L0 flux *is* the corrected flux.
        _report_l0(meta, False, False)
    return period


def _report_l0(meta, withhold_co2, withhold_h2o):
    """Copy each `_L0` flux to its reported name, or withhold the ones that are owed WPL.

    Withholding means the key is absent, not NaN-valued: a period that was never offered a
    correction and a period whose correction failed are different things downstream.
    """
    if withhold_co2:
        meta.pop('fc', None)
    else:
        meta['fc'] = meta.get('fc_l0', NAN)
    if withhold_h2o:
        meta.pop('e', None)
        meta.pop('le', None)
    else:
        meta['e'] = meta.get('e_l0', NAN)
        meta['le'] = meta.get('le_l0', NAN)
    meta['wpl_applied'] = False


def _apply(meta, owed_co2, owed_h2o):
    """Evaluate Webb Eq. 25 for the water and Eq. 24 for the gas, in place on ``meta``."""
    missing = [key for key in REQUIRED
               if not math.isfinite(meta.get(key, NAN))
               or (key in DIVISORS and meta[key] == 0.0)]
    if missing:
        # No partial correction: an absent input here makes the answer wrong, not smaller.
        logger.warning('wpl: %s not finite and non-zero; fc, e and le are NaN for this '
                       'period.', ', '.join(missing))
        meta['fc'] = meta['e'] = meta['le'] = NAN
        meta['wpl_applied'] = False
        return

    # THE LANDMINE (ALGORITHMS 10.6). <rho_v> and <rho_c> are means of the RAW series,
    # taken by flux.thermodynamics before detrending, and are read from meta here. Never
    # rebuild a mean density from period['co2'] or period['h2o']: after block-average
    # detrending those series have a mean of order 1e-16, both Webb terms below vanish, and
    # the result is a silent no-op indistinguishable from a correction that ran.
    ta = meta['ta_mean']                 # KELVIN -- it appears below as rho/Ta, and a degC
    rho_v = meta['rho_v_mean']           # slip there is silent rather than loud
    rho_d = meta['rho_d_mean']

    # CONTRACT 20.9: wt is read back out of the published, Schotanus-corrected H. Using
    # cov_w_ts instead would be wrong by the latent contribution -- ~5 % on a midday parcel,
    # more as the Bowen ratio falls -- and since the thermal term below is 43-58 % of FC
    # itself, that excess would land nearly whole in the reported flux. cp cancels between
    # H_L0 = cov(w,ts)*rho_m*cp and this division, so a badly computed cp costs H its
    # accuracy and leaves FC exact.
    wt = meta['h'] / (meta['rho_m_mean'] * meta['cp_mean'])
    meta['wt'] = wt

    # (1 + mu sigma) is Webb's expansion factor. One number in two places: it multiplies
    # the whole vapour equation (Eq. 25) and the gas equation's heat term (Eq. 24).
    # sigma is <rho_v>/<rho_d>, the ratio of the two mean densities (ALGORITHMS 10.2), and
    # both of them are right here. meta['sigma_mean'] is a different number -- the
    # per-sample mean of section 8.2, which differs from this one by the Jensen gap.
    expansion = 1.0 + MU * (rho_v / rho_d)

    # The reporting form is absorbed into f_chi FIRST (ALGORITHMS 10.3). A mixing ratio's
    # FH2O_L0 fed straight into Webb's equation double-counts the thermal expansion that
    # form has already shed.
    fh2o_l0 = meta.get('fh2o_l0', NAN)
    if owed_h2o:
        f_chi = fh2o_l0 + (rho_v / MV) * wt / ta
    else:
        f_chi = fh2o_l0 * (1.0 - meta['chi_v_mean'])
    w_rho_v = MV * f_chi - rho_v * wt / ta        # <w' rho_v'> [kg m-2 s-1]

    # Webb Eq. 25. The bracket is Mv*f_chi again; it is written out rather than cancelled
    # so that the equation on the page is the equation in the code.
    e = expansion * (w_rho_v + (rho_v / ta) * wt)
    meta['e'] = e
    meta['le'] = meta['lambda_v_mean'] * e

    if not owed_co2:
        # A dry mixing ratio is per mole of dry air, so reporting it IS the correction;
        # applying Webb's equation again would subtract the same effect twice.
        meta['fc'] = meta.get('fc_l0', NAN)
    elif not math.isfinite(meta.get('rho_c_mean', NAN)):
        logger.warning('wpl: rho_c_mean is not finite; fc is NaN. e and le are corrected.')
        meta['fc'] = NAN
    else:
        rho_c = meta['rho_c_mean']
        # Webb Eq. 24, in the mass-density form it is written in. The multiply by MCO2 and
        # the divide by it cancel; they are kept so the three terms read as Webb's.
        raw_mass = meta.get('fc_l0', NAN) * MCO2
        corrected_mass = (raw_mass
                          + MU * (rho_c / rho_d) * w_rho_v      # dry air displaced by vapour
                          + expansion * (rho_c / ta) * wt)      # thermal expansion of the air
        meta['fc'] = corrected_mass / MCO2

    meta['wpl_applied'] = True
    logger.debug('wpl: wt=%.6g K m s-1, 1+mu*sigma=%.6g, w_rho_v=%.6g kg m-2 s-1',
                 wt, expansion, w_rho_v)
