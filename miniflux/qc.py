"""Stage 10 - quality flags: the steady-state test and the integral turbulence
characteristics of Foken & Wichura (1996).

Two independent diagnostics of whether the half hour honoured the assumptions eddy
covariance rests on:

* :func:`steady_state` asks whether the flux was the same at the start of the period as
  at the end, by splitting the period into five sub-intervals and comparing the mean of
  their covariances against the covariance of the whole. Each sub-interval covariance is
  taken about its **own** mean, which is what makes the comparison sensitive to a trend
  even after block-average detrending.
* :func:`itc` asks whether the turbulence was developed, by comparing measured
  normalised standard deviations against flux-variance similarity models.

Foken, T. and Wichura, B. (1996). Tools for quality assessment of surface-based flux
measurements. Agric. For. Meteorol. 78, 83-105. doi:10.1016/0168-1923(95)02248-1.
Thresholds: Foken, T. et al. (2004), Handbook of Micrometeorology, ch. 9. ITC models:
Thomas, C. and Foken, T. (2002), Re-evaluation of integral turbulence characteristics
and their parameterisations.

No flag is derived from the ITC deviations; they are reported as fractions and the
reader decides. The 0-9 grades of Foken (2003) and the combined classes of Mauder &
Foken (2004) are not implemented (ALGORITHMS.md section 11.2).
"""

import logging
import math

from . import kernels

logger = logging.getLogger(__name__)

NAN = float('nan')

#: Number of sub-intervals the steady-state test cuts the period into.
N_SUBINTERVALS = 5

#: Earth's angular velocity [rad s-1], over the 24 h solar day. Both reference
#: implementations use the solar day; the sidereal day would give 7.29212e-05.
OMEGA = 2.0 * math.pi / (24 * 60 * 60)

#: The dimensionless height of the neutral ITC models [-]. It is 1.0 and is **not** the
#: measurement height: the parameterisation is written at z+ = 1 and both reference
#: implementations hard-code it there.
ZPLUS = 1.0


def quality(period, cfg):
    """Write the period's quality diagnostics into ``period['meta']``. Returns period.

    Reads ``[qc] steady_state_pair``, ``[qc] itc`` and ``[site] latitude``, plus the
    moments and the stability that earlier steps left in meta.

    Writes ``sst_pct`` [%], ``sst_flag`` [0/1/2] and, when ``[qc] itc`` is on,
    ``tstar`` [K] and ``itc_w`` / ``itc_u`` / ``itc_t`` [fraction].

    This is the last step of the pipeline because ITC needs ``z_l``, which
    ``flux.assemble`` creates. Run earlier it would return the neutral wind models
    computed on an absent stability, which is two plausible numbers and no error.
    """
    meta = period['meta']
    label = meta.get('period_start', '?')

    x_name, y_name = cfg.qc.steady_state_pair
    x = period.get(x_name)
    y = period.get(y_name)
    if x is None or y is None:
        logger.warning('%s: [qc] steady_state_pair = %s,%s names a series the period '
                       'does not carry; steady state is NaN', label, x_name, y_name)
        pct, flag = NAN, 2
    else:
        pct, flag = steady_state(x, y)
        if not math.isfinite(pct):
            # Both possible causes are named, because neither is tested here: the period
            # covariance is zero or non-finite, or every sub-interval is (N < 5 leaves
            # all five empty). Blaming the period covariance alone points the reader at a
            # number that can be perfectly healthy.
            logger.warning('%s: the FW96 steady-state statistic on cov(%s,%s) could not '
                           'be formed over %d sample(s): either the period covariance is '
                           'zero or non-finite, or all five sub-intervals are; flag 2',
                           label, x_name, y_name, min(len(x), len(y)))
    meta['sst_pct'] = pct
    meta['sst_flag'] = flag

    if not cfg.qc.itc:
        return period

    ustar = meta.get('ustar', NAN)
    cov_w_ts = meta.get('cov_w_ts', NAN)
    z_l = meta.get('z_l', NAN)
    meta['tstar'] = _tstar(cov_w_ts, ustar)
    itc_w, itc_u, itc_t = itc(meta.get('var_w', NAN), meta.get('var_u', NAN),
                              meta.get('var_ts', NAN), ustar, cov_w_ts, z_l,
                              cfg.site.latitude)
    if not (math.isfinite(itc_w) and math.isfinite(itc_u) and math.isfinite(itc_t)):
        # Which deviation is undefined, and every input any of them divides by. ITC_T is
        # routinely the only one missing, and its cause (T* = 0 from a vanishing heat
        # flux) is not among ustar, z/L and latitude at all.
        logger.warning('%s: ITC is not fully defined: itc_w = %r, itc_u = %r, itc_t = %r '
                       '(ustar = %r m s-1, z/L = %r, latitude = %r deg, cov_w_ts = %r '
                       'K m s-1, var_w/var_u/var_ts = %r/%r/%r)',
                       label, itc_w, itc_u, itc_t, ustar, z_l, cfg.site.latitude,
                       cov_w_ts, meta.get('var_w', NAN), meta.get('var_u', NAN),
                       meta.get('var_ts', NAN))
    meta['itc_w'] = itc_w
    meta['itc_u'] = itc_u
    meta['itc_t'] = itc_t
    return period


def steady_state(w, c):
    """Foken & Wichura (1996) non-stationarity of cov(w, c).

    Returns ``(percentage [%], flag)`` where the flag is 0 (fine), 1 (usable) or 2
    (discard), and ``(nan, 2)`` when the statistic cannot be formed. The unit of the
    inputs is irrelevant: the statistic is a ratio of two covariances of the same pair.

        m     = floor(N / 5)
        C_j   = cov(w, c) over [(j-1)m, j*m)      for j = 1..5, each about its own mean
        C_tot = cov(w, c) over the whole period
        S     = | (nanmean(C_1..C_5) - C_tot) / C_tot |

    The remainder ``N - 5m`` is excluded from the sub-intervals and included in
    ``C_tot``; the bounds are handed to :func:`kernels.cov` rather than sliced, so both
    halves of the ratio come from the one covariance estimator. ``nanmean`` over the
    five means a sub-interval that came back NaN is skipped and the statistic can rest
    on fewer than five with no record that it did.

    The percentage is not capped: 30 and 100 are flag bounds, not clamps, and a period
    with a reversing flux can print thousands of per cent. ``N < 5`` gives ``m = 0``,
    every sub-interval is empty, and the statistic is NaN.
    """
    n = min(len(w), len(c))
    m = n // N_SUBINTERVALS
    subs = kernels.new(N_SUBINTERVALS)
    for j in range(N_SUBINTERVALS):
        subs[j] = kernels.cov(w, c, j * m, (j + 1) * m)
    total = kernels.cov(w, c, 0, n)
    # A zero covariance has no relative deviation, and neither has a NaN one.
    if total == 0.0 or not math.isfinite(total):
        return NAN, 2
    pct = 100.0 * abs((kernels.nanmean(subs) - total) / total)
    return pct, _sst_flag(pct)


def itc(var_w, var_u, var_ts, ustar, cov_w_ts, z_l, latitude):
    """Integral turbulence characteristics: ``(ITC_w, ITC_u, ITC_T)``, each a fraction.

    Each is the relative deviation ``|(model - measured) / model|`` of a normalised
    standard deviation from its flux-variance similarity model. Units in: ``var_w``,
    ``var_u`` [m2 s-2], ``var_ts`` [K2], ``ustar`` [m s-1], ``cov_w_ts`` [K m s-1],
    ``z_l`` [-], ``latitude`` [degrees north]. The variances are the ones already in
    meta, over the **rotated** wind components and the whole period, ddof = 1; there are
    no sub-intervals here.

    Returns ``(nan, nan, nan)`` when a guard fires. Both reference codebases leave these
    open and return ``inf`` or raise; miniflux reports NaN:

    * ``ustar <= 0`` or non-finite - the neutral model would take ``log(x/0)`` and T*
      would divide by zero;
    * ``z_l == 0`` exactly - the third temperature branch evaluates ``0.5 * 0**-0.5``;
    * ``z_l`` non-finite - every model is meaningless;
    * ``latitude == 0`` - the Coriolis parameter vanishes and the neutral model takes
      ``log(0)``.
    """
    if not math.isfinite(ustar) or ustar <= 0.0:
        return NAN, NAN, NAN
    if z_l == 0.0 or not math.isfinite(z_l):
        return NAN, NAN, NAN
    fcor = abs(2.0 * OMEGA * math.sin(latitude * math.pi / 180.0))
    if fcor == 0.0:                # latitude 0: the neutral models would take log(0)
        return NAN, NAN, NAN

    if z_l < -0.2:                 # unstable: free-convection scaling
        sw_model = 1.3 * (1.0 - 2.0 * z_l) ** (1.0 / 3.0)
        su_model = 4.15 * abs(z_l) ** (1.0 / 8.0)
    else:                          # neutral and stable: the Coriolis-scaled models
        neutral = math.log(fcor * ZPLUS / ustar)
        sw_model = 0.21 * neutral + 3.1
        su_model = 0.44 * neutral + 6.3

    # Four temperature branches over z/L: < -1, [-1, -0.0625), [-0.0625, 0.02), and the
    # rest. The last one is an else, not `elif z_l > 0.02`, so z/L of exactly 0.02 lands
    # there; upstream GEddySoft leaves the variable unassigned in that case.
    if z_l < -1.0:
        st_model = abs(z_l) ** (-1.0 / 3.0)
    elif z_l < -0.0625:
        st_model = abs(z_l) ** (-1.0 / 4.0)
    elif z_l < 0.02:
        st_model = 0.5 * abs(z_l) ** (-0.5)
    else:
        st_model = 1.4 * abs(z_l) ** (-1.0 / 4.0)

    tstar = _tstar(cov_w_ts, ustar)
    sw_meas = _sd(var_w) / ustar
    su_meas = _sd(var_u) / ustar
    # |T*|, not T*: CONTRACT.md section 20.5, a deliberate divergence from the parent
    # and from GEddySoft, which both divide by the signed T*. Under unstable conditions
    # cov(w,ts) > 0 makes T* < 0, their sT_meas negative and their ITC_T > 1 always.
    # Foken's definition is sigma_T / |T*|. miniflux's ITC_T is therefore not comparable
    # with the parent's on unstable periods; every other column is.
    if tstar == 0.0:               # a zero heat flux has no temperature scale
        st_meas = NAN
    else:
        st_meas = _sd(var_ts) / abs(tstar)

    return (_deviation(sw_model, sw_meas),
            _deviation(su_model, su_meas),
            _deviation(st_model, st_meas))


def _sst_flag(pct):
    """Foken-style 0/1/2 flag from a non-stationarity percentage [%].

    A non-finite statistic is flag 2, discard - not "missing". The period produced a
    number that says nothing about its own stationarity, which is a reason to drop it.
    """
    if not math.isfinite(pct):
        return 2
    if pct <= 30.0:
        return 0
    if pct <= 100.0:
        return 1
    return 2


def _tstar(cov_w_ts, ustar):
    """Temperature scale T* = -cov(w,ts)/ustar [K]. NaN unless ustar is finite and > 0."""
    if not math.isfinite(ustar) or ustar <= 0.0:
        return NAN
    return -cov_w_ts / ustar


def _sd(var):
    """sqrt of a variance already computed with ddof = 1; NaN for a non-finite one.

    Negative never happens from :func:`kernels.variance`, but this function is reached
    from a public signature, and math.sqrt raises on a negative argument.
    """
    if not math.isfinite(var) or var < 0.0:
        return NAN
    return math.sqrt(var)


def _deviation(model, measured):
    """Relative deviation |(model - measured) / model| [fraction]; NaN if model is 0."""
    if model == 0.0 or not math.isfinite(model):
        return NAN
    return abs((model - measured) / model)
