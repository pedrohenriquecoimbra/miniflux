"""Stage 9b -- a first-order low-pass attenuation factor (Horst 1997, Eq. 11).

A tube, a finite optical path and a finite sensor response all act as low-pass filters
on the gas signal. Whatever fraction of the flux lives above the resulting cut-off is
simply missing from the covariance, so an uncorrected flux is an **underestimate** --
for a closed-path analyser typically a few per cent to tens of per cent.

Horst models the analyser as one first-order system with time constant ``tau`` and
integrates its transfer function against a similarity cospectrum. The integral has a
closed form, which is the whole reason this module is forty lines rather than four
hundred::

    A   = 1 / (1 + (2*pi*nm*tau*u/z)**alpha)      the attenuated fraction, 0 < A <= 1
    SCF = 1 / A = 1 + (2*pi*nm*tau*u/z)**alpha    the recovery factor, >= 1

with ``u`` the mean wind speed, ``z`` the measurement height above the displacement, and
the cospectral shape carried by two constants that depend only on stability::

    z/L <= 0 :  nm = 0.085                      alpha = 7/8
    z/L >  0 :  nm = 2.0 - 1.915/(1 + 0.5 z/L)  alpha = 1

The two ``nm`` branches meet exactly at ``z/L = 0`` (``2.0 - 1.915 = 0.085``); only the
exponent steps.

**This is a first-order approximation and not a substitute for an in-situ method.** It
knows nothing about tube sorption on the water channel, sensor separation, RH dependence
or the actual measured cospectra, all of which EddyPro and oneflux_preproc estimate from
the data. What it gives is one defensible number per period from geometry the user
declares, instead of a silent zero.

It is OFF by default, and when it is on it never overwrites a flux: the factors and the
scaled fluxes are published as their own columns (``SCF_CO2``, ``SCF_H2O``, ``FC_SPEC``,
``LE_SPEC``, ``E_SPEC``) beside the uncorrected ``FC``, ``LE`` and ``E``, so no number
in this table can be read as corrected when it is not.

Citation: Horst, T. W. (1997). A simple formula for attenuation of eddy fluxes measured
with first-order-response scalar sensors. *Boundary-Layer Meteorol.* 82, 219-233.

See ALGORITHMS.md section 10A and CONTRACT.md section 12A.
"""

import logging
import math

from .constants import Z_MINUS_D_FLOOR

logger = logging.getLogger(__name__)

NAN = float('nan')

#: Normalised cospectral peak frequency and transfer exponent, neutral and unstable
#: (Horst 1997 Eqs. 8 and 11). The stable branch computes its own ``nm``.
NM_NEUTRAL = 0.085
ALPHA_UNSTABLE = 7.0 / 8.0
ALPHA_STABLE = 1.0

#: Above this recovery factor the model is being asked to invent more than half the
#: flux. Horst's first-order form is derived for modest attenuation, so a factor this
#: large is a statement about the configuration -- a long tube, a low tower, a dead calm
#: -- and not a correction anyone should publish unexamined.
IMPLAUSIBLE_FACTOR = 2.0

#: ``(source flux, published name, which gas's factor scales it)``.
_SCALED = (('fc', 'fc_spec', 'scf_co2'),
           ('e', 'e_spec', 'scf_h2o'),
           ('le', 'le_spec', 'scf_h2o'))


def correct(period, cfg):
    """Publish one attenuation-recovery factor per gas, and the fluxes it scales.

    Reads ``[spectral] enabled/co2_tau_s/h2o_tau_s``, ``[site] measurement_height`` with
    the derived ``cfg.site.displacement``, and the ``wind_speed`` / ``z_l`` / ``fc`` /
    ``e`` / ``le`` already in ``period['meta']``. Writes ``scf_co2``, ``scf_h2o``,
    ``fc_spec`` [mol m-2 s-1], ``e_spec`` [kg m-2 s-1] and ``le_spec`` [W m-2].

    With ``enabled = false`` it writes nothing at all: the ``_SPEC`` columns then hold
    ``na_value``, which is the honest rendering of a correction that was not made.

    ``H`` is deliberately not scaled. This factor describes the analyser's response, and
    the sonic temperature does not pass through the analyser.

    Returns the same period dict.
    """
    if not cfg.spectral.enabled:
        return period

    meta = period['meta']
    # z - d, the same height flux.assemble puts under z/L, so the cospectrum is placed in
    # frequency by the geometry the stability it reads was computed at.
    height = cfg.site.measurement_height - max(Z_MINUS_D_FLOOR, cfg.site.displacement)
    wind = meta.get('wind_speed', NAN)
    z_l = meta.get('z_l', NAN)

    meta['scf_co2'] = _factor(wind, height, z_l, cfg.spectral.co2_tau_s, 'co2')
    meta['scf_h2o'] = _factor(wind, height, z_l, cfg.spectral.h2o_tau_s, 'h2o')

    for source, published, factor_key in _SCALED:
        if source in meta:
            meta[published] = meta[source] * meta[factor_key]
        else:
            # The uncorrected flux was withheld (wpl off on a gas that is owed it), so
            # there is nothing to scale and a scaled name would promise one.
            meta.pop(published, None)
    return period


def _factor(wind, height, z_l, tau, gas):
    """The recovery factor ``1/A`` of Horst Eq. 11. [m s-1], [m], [-], [s] -> [-], >= 1.

    NaN when the period does not carry a usable wind, height or stability: a correction
    is a measured quantity here, and a missing one is reported as missing.
    """
    if not (height > 0.0):
        logger.warning('spectral: the measurement height is at or below the '
                       'displacement (z - d = %r m); no %s factor', height, gas)
        return NAN
    if not (math.isfinite(wind) and math.isfinite(z_l)):
        logger.warning('spectral: wind_speed=%r or z_l=%r is not finite; the %s factor '
                       'is NaN for this period', wind, z_l, gas)
        return NAN

    if z_l > 0.0:
        nm = 2.0 - 1.915 / (1.0 + 0.5 * z_l)    # denominator > 1 for every z_l > 0
        alpha = ALPHA_STABLE
    else:
        nm = NM_NEUTRAL
        alpha = ALPHA_UNSTABLE
    # wind_speed is a hypot and cannot be negative, so the power is always real.
    factor = 1.0 + (2.0 * math.pi * nm * tau * wind / height) ** alpha
    if factor > IMPLAUSIBLE_FACTOR:
        logger.warning('spectral: the %s recovery factor is %.3g -- more than half the '
                       'reported flux would come from the model rather than from the '
                       'data. Check [spectral] %s_tau_s (%.4g s) against z - d = %.4g m '
                       'and a wind of %.4g m s-1.', gas, factor, gas, tau, height, wind)
    return factor
