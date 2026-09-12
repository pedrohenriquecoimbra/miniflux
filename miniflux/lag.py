"""Stage 4 -- time-lag maximisation: put each scalar back with the wind it belongs to.

An analyser sees an eddy a moment after the sonic does -- the air travels down a tube,
or the two instruments simply sit a few centimetres apart. That displacement destroys
the covariance, so the scalar is slid against w and the shift maximising ``|cov|`` is
kept. Citation: Aubinet, Vesala & Papale (eds., 2012), *Eddy Covariance: A Practical
Guide*; the implementation is EddyPro's ``CovMax`` with GEddySoft's ``MAX_WITH_DEFAULT``
boundary rule.

Runs after rotation, on the rotated w, and before detrending -- harmless, because the
scan's covariance removes the overlap means itself.

One sign convention, nowhere negated (ALGORITHMS 5.1): ``L`` is the physical lag in
samples, and a positive ``L`` means the scalar arrives *after* the wind, so the sample
recorded ``L`` steps later belongs with the wind now. Everything here is
``cov(w_i, c_{i+L})``; there is no internal "shift" variable.
"""

import logging
import math

from . import kernels

logger = logging.getLogger(__name__)

NAN = float('nan')

#: The two methods that actually search. ``fixed`` and ``none`` never look at the data.
SCAN_METHODS = ('covmax_default', 'covmax')

#: How far a later lag has to beat the running peak to take it, relative. Two covariances
#: closer together than this are tied as far as either kernel path can tell: they are
#: reductions, promised only to the 8 ULP of CONTRACT 3.8, so which of them is larger is
#: decided by the summation order and not by the data. A period symmetric about its
#: midpoint makes cov(+L) and cov(-L) mathematically equal, and without this margin the
#: pure and the numpy path select opposite lags on it -- the decision CONTRACT 3.8 says
#: must match exactly. With it, a tie falls through to ALGORITHMS 5.4's rule instead: the
#: lowest lag wins. The margin is four orders above the 8 ULP the two paths can differ by
#: and ten orders below a peak a real covariance curve resolves.
TIE_MARGIN = 1.0 + 1e-12


def apply_lags(period, cfg):
    """Shift every scalar of ``[lag] scalars`` onto w by its own lag. Returns `period`.

    Reads ``period['w']`` and ``period[s]`` (canonical units), ``meta['freq_hz']`` [Hz],
    ``cfg.lag.method``, ``cfg.lag.scalars`` and ``cfg.lag.windows[s] = (nominal_s,
    min_s, max_s)`` [s]. Writes six meta keys per scalar (CONTRACT 10): the applied lag
    in samples and in seconds, the scan's optimum in samples and in seconds, whether the
    nominal was used instead, and the covariance at the optimum.

    Only the scalar moves; w is the fixed reference, and the |L| samples the shift
    cannot fill are NaN, never wrapped (ALGORITHMS 5.6).
    """
    meta = period['meta']
    method = cfg.lag.method
    # The configured rate is authoritative and reaches this step through read.py.
    freq = meta.get('freq_hz', cfg.period.acquisition_frequency)
    if not (math.isfinite(freq) and freq > 0.0):
        # A sample window cannot be framed without a rate, and "22" read as 22 s when it
        # means 22 samples is not a diagnostic. Nothing is searched for and nothing is
        # shifted; the seconds go out as NaN below (ALGORITHMS 5.7).
        logger.warning('acquisition frequency %r is not a positive finite rate; '
                       'no time lag is searched for or applied', freq)
        method = 'none'
        freq = NAN

    # Pre-centring leaves every cov(L) unchanged in exact arithmetic and removes the
    # cancellation the scan's one-pass form would otherwise suffer on a 400 ppm CO2
    # series (ALGORITHMS 5.3). w's mean is the same for every scalar, so it is taken once.
    centred_w = kernels.sub_const(period['w'], kernels.nanmean(period['w']))

    for scalar in cfg.lag.scalars:
        nominal_s, min_s, max_s = cfg.lag.windows[scalar]
        # round() is half-to-even where EddyPro's nint is half-away-from-zero; they part
        # only on a lag of exactly half a sample (ALGORITHMS 5.2). Only a scanning method
        # frames a window, and an unusable rate became 'none' above, so no NaN reaches
        # round(), which would raise -- and a step never raises.
        lo = hi = nominal = 0
        scalar_method = method
        if method != 'none':
            nominal = int(round(nominal_s * freq))
            if method in SCAN_METHODS:
                if min_s == 0.0 and max_s == 0.0:
                    # ALGORITHMS 5.7, and it is the *seconds* that are the exemption: no
                    # window was configured, so there is nothing to search and a lag of
                    # zero is applied -- not the nominal. A window that is merely narrow
                    # (one that rounds to the single lag 0) is a one-lag search like any
                    # other, and its peak sits on both rims, which is 5.5's business.
                    scalar_method = 'none'
                else:
                    lo = int(round(min_s * freq))
                    hi = int(round(max_s * freq))
                    if hi < lo:        # config refuses an inverted window; defensive only
                        lo, hi = hi, lo

        centred_c = kernels.sub_const(period[scalar], kernels.nanmean(period[scalar]))
        applied, opt, default_used, cov_peak = find_lag(
            centred_w, centred_c, lo, hi, nominal, scalar_method)

        meta[scalar + '_lag_samples'] = applied
        meta[scalar + '_lag_s'] = _seconds(applied, freq)
        meta[scalar + '_lag_opt_samples'] = opt
        meta[scalar + '_lag_opt_s'] = _seconds(opt, freq)
        meta[scalar + '_lag_default_used'] = default_used
        # Diagnostic only: this is the scan's own overlap covariance, and the flux
        # covariance is computed later, on the aligned and detrended period, with the
        # two-pass estimator (ALGORITHMS 5.8). Never reuse this number as a flux.
        meta[scalar + '_lag_cov'] = cov_peak

        if applied:
            period[scalar] = kernels.shift_truncate(period[scalar], applied)
        logger.debug('%s lag %+d samples (optimum %+d, default_used=%s)',
                     scalar, applied, opt, default_used)

    return period


def find_lag(w, c, lo, hi, nominal, method):
    """Locate c against w by maximising ``|cov(w_i, c_{i+L})|`` over ``L`` in [lo, hi].

    w, c: array('d'), the two series *already pre-centred by the caller*.
    lo, hi, nominal: integer sample lags, positive = the scalar arrives after the wind.
    method: 'covmax_default' | 'covmax' | 'fixed' | 'none'.

    Returns ``(applied, opt, default_used, cov_peak)``: the lag to apply [samples], the
    lag the scan chose [samples], whether the nominal was substituted, and the signed
    covariance at the optimum [m s-1 * unit(c)]. Both lags are reported because they
    answer different questions and must not be confused.
    """
    if method == 'none':
        # No scan and no shift. The lags are a plain zero because that is what was
        # applied, but the covariance is NaN: nothing was measured, and a 0.0 here would
        # read as a measured covariance that happened to vanish.
        return 0, 0, False, NAN
    if method == 'fixed':
        # The nominal is the answer by construction, so there is no peak to report.
        return nominal, nominal, True, NAN

    curve = kernels.cov_curve(w, c, lo, hi)      # hi - lo + 1 values, both ends inclusive

    # |cov| needs no per-variable sign expectation: w'T' peaks positive by day, w'CO2'
    # negative by day and positive by night, w'u' negative -- one criterion covers all
    # of them. Scanning ascending and taking a later lag only when it beats the running
    # peak by TIE_MARGIN keeps the first (lowest) lag on a tie, mathematical or merely
    # unresolvable; the explicit L_star = nominal means a period with no peak at all
    # returns the nominal rather than whatever the loop last held. The floor is 0.0 and
    # the margin is multiplicative, so the first finite non-zero value still takes it.
    max_abs = 0.0
    opt = nominal
    cov_peak = NAN
    found = False
    for index, value in enumerate(curve):
        if math.isfinite(value) and abs(value) > max_abs * TIE_MARGIN:
            max_abs = abs(value)
            opt = lo + index
            cov_peak = value
            found = True

    if not found:
        # Every candidate was NaN, or the window is flat at zero: the scan has nothing
        # to say, so the nominal stands and the run is told so.
        logger.warning('no covariance peak in the lag window [%+d, %+d] samples; '
                       'applying the nominal lag %+d', lo, hi, nominal)
        return nominal, nominal, True, NAN

    if method == 'covmax_default' and (opt == lo or opt == hi):
        # A peak on the rim means the true peak is most likely outside the window: the
        # search failed, and applying a rim lag corrupts the flux. Report both lags.
        logger.warning('covariance peak sits on the window edge (%+d samples in '
                       '[%+d, %+d]); applying the nominal lag %+d instead',
                       opt, lo, hi, nominal)
        return nominal, opt, True, cov_peak

    return opt, opt, False, cov_peak


def _seconds(samples, freq):
    """A sample lag as seconds; NaN unless the rate is a positive finite number [Hz]."""
    if math.isfinite(freq) and freq > 0.0:
        return samples / freq
    return NAN
