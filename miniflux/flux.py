"""Thermodynamic mean state, second moments, and flux assembly.

ALGORITHMS.md sections 7, 8 and 9, in the order the pipeline runs them:
``thermodynamics`` after the lag alignment and before detrending, so every period mean
is taken over the same samples that enter the covariances and while those means still
exist; ``moments`` and ``assemble`` after detrending.

Temperature appears here in three conventions in adjacent formulas and they are not
interchangeable (ALGORITHMS 8.3), so every name carries its unit: ``*_k`` is kelvin,
``*_c`` is celsius, ``theta_s`` is a potential temperature. Pressures are ``*_pa``.

No function here raises. Degeneracy is reported as ``nan`` and logged, which is why the
divisions go through :func:`_divide`: Python raises ``ZeroDivisionError`` where IEEE-754
returns an infinity, and ALGORITHMS 9.4 asks for the infinity.
"""

import logging
import math

from . import kernels
from .constants import (CPD, G, KAPPA, MCO2, MV, P0A, R, RD, RV, T0,
                        Z_MINUS_D_FLOOR)

logger = logging.getLogger(__name__)

NAN = kernels.NAN

# One per-sample quantity per name, averaged over its own finite samples and published as
# '<name>_mean'. ALGORITHMS 8.2 lists the same set.
_MEAN_KEYS = ('ta', 'ts', 'p_air', 'pd', 'e', 'es', 'rh', 'chi_v', 'va', 'vd', 'n_d',
              'rho_v', 'rho_d', 'rho_m', 'rho_c', 'q', 'sigma', 'cp', 'cp_d', 'cp_v',
              'lambda_v')


def _divide(numerator, denominator):
    """``numerator / denominator`` with the IEEE-754 answer where Python raises.

    A zero denominator gives a signed infinity, or NaN for 0/0 and NaN/0. ALGORITHMS 9.4
    leaves those degeneracies deliberately unguarded -- ``cov(w,ts) = 0`` is meant to give
    ``L = +-inf`` -- and a step module may never raise, so the standard's answer is
    spelled out here rather than left to the interpreter.
    """
    if denominator == 0.0:
        if numerator != numerator or numerator == 0.0:
            return NAN
        # copysign on the denominator so a negative zero keeps its sign.
        return math.copysign(math.inf, numerator) * math.copysign(1.0, denominator)
    return numerator / denominator


def _joint_finite(x, y):
    """How many samples are finite in *both* series. Returns int.

    This is the N of the covariance it accompanies: ``kernels`` counts one series at a
    time, and what CONTRACT 11.3 publishes is the count of pairs that actually entered.
    """
    n = 0
    for i in range(min(len(x), len(y))):
        if math.isfinite(x[i]) and math.isfinite(y[i]):
            n += 1
    return n


# --- 11.1 scalar helpers -----------------------------------------------------------

def saturation_vapour_pressure(ta_k):
    """Saturation vapour pressure over water. ta_k in KELVIN [K] -> [Pa].

    ``es = exp(77.345 + 0.0057 T - 7235/T) / T**8.2`` with T in kelvin. The kelvin
    magnitude is the argument even though a celsius formula is often quoted beside it;
    feeding celsius returns ~1e-135 Pa and drives RH to infinity (ALGORITHMS 8.3).
    Check value: ``saturation_vapour_pressure(293.15) = 2331.215``.
    """
    if not ta_k > 0.0:
        return NAN          # NaN, zero and negative temperatures have no es
    power = 77.345 + 0.0057 * ta_k - 7235.0 / ta_k
    if power > 709.0:
        return math.inf     # ln of the largest double is 709.78; exp would raise past it
    return math.exp(power) / ta_k ** 8.2


def cp_dry(ta_k):
    """Specific heat of dry air at constant pressure. ta_k in KELVIN [K] -> [J kg-1 K-1].

    ``cp_d = 1005 + (Tc + 23.12)**2 / 3364`` with Tc in CELSIUS. The 1005 is the intercept
    of that fit and is paired with its quadratic term; the textbook 1004.67 belongs to a
    different parameterisation and leaves cp_d low, and H with it.
    Check value: ``cp_dry(293.15) = 1005.5527``.
    """
    ta_c = ta_k - T0                                    # [degC], the unit the fit is in
    return CPD + (ta_c + 23.12) ** 2 / 3364.0


def cp_vapour(ta_k, rh_pct):
    """Specific heat of water vapour. ta_k in KELVIN [K], rh_pct in [%] -> [J kg-1 K-1].

    The fit is written in CELSIUS and in percent. RH is not clipped: a supersaturated
    period feeds RH > 100 straight through, which is the honest reading of the data.
    Check value: ``cp_vapour(293.15, 50.0) = 1876.36``.
    """
    ta_c = ta_k - T0                                    # [degC], the unit the fit is in
    return (1859.0 + 0.13 * rh_pct
            + (0.193 + 5.6e-3 * rh_pct) * ta_c
            + (1e-3 + 5e-5 * rh_pct) * ta_c ** 2)


def latent_heat(ta_k):
    """Latent heat of vaporisation of water. ta_k in KELVIN [K] -> [J kg-1].

    ``lambda_v = 1e3 * (3147.5 - 2.37 T)`` with T in kelvin; the 1e3 carries the fit's
    [J g-1] to [J kg-1]. Fleagle & Businger (1980), via EddyPro ``flux_params.f90``.
    Check value: ``latent_heat(293.15) = 2452734.5``.
    """
    return 1.0e3 * (3147.5 - 2.37 * ta_k)               # [J g-1] -> [J kg-1]


def air_temperature_from_sonic(ts_k, e_pa, p_pa):
    """Ambient air temperature from the sonic (virtual) one. [K], [Pa], [Pa] -> [K].

    ``Ta = Ts / (1 + 0.32 e/P)``, the sonic's humidity sensitivity inverted (Schotanus
    et al. 1983). The coefficient is 0.32 and it multiplies e/P; the 0.51 that appears
    two formulas away multiplies q and is a different correction. They are not
    interchangeable (ALGORITHMS 8.1).
    """
    return _divide(ts_k, 1.0 + 0.32 * _divide(e_pa, p_pa))


# --- 11.2 thermodynamics -----------------------------------------------------------

def thermodynamics(period, cfg):
    """Per-sample moist-air state, averaged over the period. ALGORITHMS section 8.

    Pass A seeds the humidity at the sonic temperature and resolves an air temperature
    per sample -- measured where finite, else sonic-derived, else the sonic temperature
    itself, so one period may mix all three sources. Pass B rebuilds the whole state at
    that air temperature.

    Every published value is the nan-mean of a per-sample quantity, never the quantity of
    the means: the flux factor downstream is ``1/<Vd>`` and cp is ``<cp(T_i, RH_i)>``.
    ``ta_mean`` is the mean of the *resolved* temperature, which is why
    ``n_ta_source_measured`` is published beside it.

    Reads ts, p_air, h2o, co2, ta and ``[gases] *_measure_type``. Writes the 24 meta keys
    of CONTRACT 11.2, all floats but that one count. Returns the period.
    """
    meta = period['meta']
    ts_series = period['ts']
    p_series = period['p_air']
    h2o_series = period['h2o']
    co2_series = period['co2']
    ta_series = period['ta']
    h2o_is_density = cfg.gases.measure_type['h2o'] == 'molar_density'
    co2_is_density = cfg.gases.measure_type['co2'] == 'molar_density'

    sums = dict((name, 0.0) for name in _MEAN_KEYS)
    counts = dict((name, 0) for name in _MEAN_KEYS)
    n_measured = 0
    for i in range(len(ts_series)):
        ts_k = ts_series[i]                                     # [K] sonic (virtual)
        p_pa = p_series[i]                                      # [Pa]
        h2o = h2o_series[i]                  # [mol mol-1 dry] or [mol m-3]

        # Pass A: humidity at the sonic temperature, only so Ta can be resolved.
        if h2o_is_density:
            rho_v = h2o * MV                                    # [mol m-3][kg mol-1] -> [kg m-3]
            chi_v = _divide(rho_v * R * ts_k, MV * p_pa)        # [mol mol-1] of moist air
        else:
            chi_v = _divide(h2o, 1.0 + h2o)                     # dry ratio -> moist fraction
            rho_v = _divide(chi_v * MV * p_pa, R * ts_k)        # [kg m-3]
        e_pa = rho_v * RV * ts_k                                # [Pa]

        ta_meas = ta_series[i]
        if math.isfinite(ta_meas):
            ta_k = ta_meas                                      # [K] measured, preferred
            n_measured += 1
        else:
            ta_k = air_temperature_from_sonic(ts_k, e_pa, p_pa)
            if not math.isfinite(ta_k):
                ta_k = ts_k                                     # [K] last resort

        # Pass B: the state, at Ta. Ts cancels out of e, so the two passes agree here.
        va = _divide(R * ta_k, p_pa)                            # [m3 mol-1] moist air
        if h2o_is_density:
            chi_v = _divide(rho_v * R * ta_k, MV * p_pa)        # [mol mol-1]
        else:
            rho_v = _divide(chi_v * MV, va)                     # [kg m-3]
        e_pa = rho_v * RV * ta_k                                # [Pa]
        es_pa = saturation_vapour_pressure(ta_k)                # [Pa]
        rh_pct = 100.0 * _divide(e_pa, es_pa)                   # [-] -> [%]
        pd_pa = p_pa - e_pa                                     # [Pa] dry-air partial
        vd = _divide(R * ta_k, pd_pa)                           # [m3 mol-1] dry air
        n_d = _divide(pd_pa, R * ta_k)                          # [mol m-3] dry air
        rho_d = _divide(pd_pa, RD * ta_k)                       # [kg m-3]
        rho_m = rho_d + rho_v                                   # [kg m-3] moist air
        q = _divide(rho_v, rho_m)                               # [kg kg-1]
        sigma = _divide(rho_v, rho_d)                           # [-]
        cp_d = cp_dry(ta_k)                                     # [J kg-1 K-1]
        cp_v = cp_vapour(ta_k, rh_pct)                          # [J kg-1 K-1]
        # The mixture, not cp_d*(1 + 0.84 q): cp_v is itself T- and RH-dependent.
        cp = (1.0 - q) * cp_d + q * cp_v                        # [J kg-1 K-1]
        lambda_v = latent_heat(ta_k)                            # [J kg-1]
        if co2_is_density:
            n_co2 = co2_series[i]                               # [mol m-3] already
        else:
            n_co2 = co2_series[i] * n_d      # [mol mol-1 dry][mol m-3 dry] -> [mol m-3]
        rho_c = n_co2 * MCO2                                    # [kg m-3]

        # Each quantity is folded into its own running nan-mean, left to right, so a
        # sample that is degenerate in one quantity still contributes to the others.
        for name, value in (('ta', ta_k), ('ts', ts_k), ('p_air', p_pa),
                            ('pd', pd_pa), ('e', e_pa), ('es', es_pa),
                            ('rh', rh_pct), ('chi_v', chi_v), ('va', va), ('vd', vd),
                            ('n_d', n_d), ('rho_v', rho_v), ('rho_d', rho_d),
                            ('rho_m', rho_m), ('rho_c', rho_c), ('q', q),
                            ('sigma', sigma), ('cp', cp), ('cp_d', cp_d),
                            ('cp_v', cp_v), ('lambda_v', lambda_v)):
            if math.isfinite(value):
                sums[name] += value
                counts[name] += 1

    for name in _MEAN_KEYS:
        n = counts[name]
        meta[name + '_mean'] = sums[name] / n if n else NAN

    meta['n_ta_source_measured'] = n_measured

    # The two published gas diagnostics carry a fixed unit whatever the measure type: a
    # molar density is divided by the dry-air molar density to reach a dry mole fraction.
    n_d_mean = meta['n_d_mean']
    co2_dry = kernels.nanmean(co2_series)          # [mol mol-1 dry] or [mol m-3]
    h2o_dry = kernels.nanmean(h2o_series)          # [mol mol-1 dry] or [mol m-3]
    if co2_is_density:
        co2_dry = _divide(co2_dry, n_d_mean)       # [mol m-3]/[mol m-3 dry]
    if h2o_is_density:
        h2o_dry = _divide(h2o_dry, n_d_mean)       # [mol m-3]/[mol m-3 dry]
    meta['co2_dry_ppm'] = 1.0e6 * co2_dry                       # [mol mol-1] -> [umol mol-1]
    meta['h2o_dry_ppt'] = 1.0e3 * h2o_dry                       # [mol mol-1] -> [mmol mol-1]

    if counts['rho_m'] == 0:
        logger.warning('thermodynamics: no sample of %d yielded a finite moist-air '
                       'state; every mean is NaN', len(ts_series))
    return period


# --- 11.3 second moments -----------------------------------------------------------

def moments(period, cfg):
    """Second moments and friction velocity. ALGORITHMS section 7.

    Every moment is :func:`kernels.cov`: the joint finite mask, both means over that same
    mask, two passes, ddof = 1. Runs after detrending, so the series carry whatever
    fluctuation the configured detrend left behind.

    Writes var_u/v/w/ts [m2 s-2, K2], cov_u_v/u_w/v_w [m2 s-2], cov_w_ts [K m s-1],
    cov_w_co2 and cov_w_h2o [in the gas's own unit times m s-1], their joint-finite
    counts, and ustar [m s-1]. The pair list is fixed, so cfg is unused. Returns period.
    """
    meta = period['meta']
    for name in ('u', 'v', 'w', 'ts'):
        meta['var_' + name] = kernels.variance(period[name])
    for a, b in (('u', 'v'), ('u', 'w'), ('v', 'w'),
                 ('w', 'ts'), ('w', 'co2'), ('w', 'h2o')):
        meta['cov_%s_%s' % (a, b)] = kernels.cov(period[a], period[b])
    for name in ('ts', 'co2', 'h2o'):
        meta['n_cov_w_' + name] = _joint_finite(period['w'], period[name])

    # The full stress vector, not sqrt(|cov_u_w|): the one-component form collapses
    # toward zero wherever the rotation left <v'w'> finite, and L goes as ustar**3, so
    # that error is cubed into every stability-dependent result.
    meta['ustar'] = (meta['cov_u_w'] ** 2 + meta['cov_v_w'] ** 2) ** 0.25    # [m s-1]

    if not math.isfinite(meta['ustar']):
        logger.warning('moments: ustar is not finite (cov_u_w=%r, cov_v_w=%r)',
                       meta['cov_u_w'], meta['cov_v_w'])
    return period


# --- 11.4 flux assembly ------------------------------------------------------------

def _flux_factor(measure_type, vd_mean):
    """Covariance -> molar flux factor. [mol m-3] for a dry mixing ratio, [-] for a density.

    ``<Vd>`` is the period mean of the per-sample dry-air molar volume; evaluating
    ``Pd_mean/(R*Ta_mean)`` instead is a slightly different number (ALGORITHMS 9.1).
    """
    if measure_type == 'molar_density':
        return 1.0
    return _divide(1.0, vd_mean)


def assemble(period, cfg):
    """Level-0 fluxes, the Schotanus heat flux, Obukhov length, stability. Section 9.

    Reads only the meta written by :func:`thermodynamics` and :func:`moments`, plus
    ``[site] measurement_height`` and the derived ``cfg.site.displacement``. Writes
    f_co2, f_h2o, fc_l0, fh2o_l0 [mol m-2 s-1], e_l0 [kg m-2 s-1], le_l0 and h_l0 [W m-2],
    h [W m-2], theta_s [K], mo_length [m] and z_l [-]. Returns the period.

    Degenerate inputs give inf or nan here by design; qc and write handle them.
    """
    meta = period['meta']
    ta_mean = meta['ta_mean']                                   # [K]
    cp_mean = meta['cp_mean']                                   # [J kg-1 K-1]
    q_mean = meta['q_mean']                                     # [kg kg-1]

    meta['f_co2'] = _flux_factor(cfg.gases.measure_type['co2'], meta['vd_mean'])
    meta['f_h2o'] = _flux_factor(cfg.gases.measure_type['h2o'], meta['vd_mean'])

    fc_l0 = meta['cov_w_co2'] * meta['f_co2']                   # [mol m-2 s-1]
    fh2o_l0 = meta['cov_w_h2o'] * meta['f_h2o']                 # [mol m-2 s-1]
    e_l0 = fh2o_l0 * MV                                # [mol m-2 s-1][kg mol-1] -> [kg m-2 s-1]
    le_l0 = e_l0 * meta['lambda_v_mean']               # [kg m-2 s-1][J kg-1] -> [W m-2]
    # H_L0 is the buoyancy (sonic) heat flux: the MOIST density and the MOIST heat
    # capacity, both of them.
    h_l0 = meta['cov_w_ts'] * meta['rho_m_mean'] * cp_mean      # [W m-2]
    meta['fc_l0'] = fc_l0
    meta['fh2o_l0'] = fh2o_l0
    meta['e_l0'] = e_l0
    meta['le_l0'] = le_l0
    meta['h_l0'] = h_l0

    # Schotanus et al. (1983); van Dijk et al. (2004) Eq. 3.53. The sonic reads a virtual
    # temperature, Ts = Ta(1 + 0.51 q), and inverting that covariance decomposition
    # exactly gives this DIVISION form. EddyPro's linearised subtraction drops the
    # denominator and differs by exactly 1/(1 + 0.51 q).
    # The latent term takes the UNCORRECTED E_L0: rebuilding it from a WPL-corrected or
    # spectrally scaled E inflates a 4-5 % term by the whole of that correction.
    missing = [name for name, value in (('h_l0', h_l0), ('e_l0', e_l0),
                                        ('ta_mean', ta_mean), ('cp_mean', cp_mean),
                                        ('q_mean', q_mean)) if not math.isfinite(value)]
    if missing:
        # The key stays absent so write.py emits na_value. H is never aliased to H_L0.
        logger.warning('assemble: H not produced, %s not finite', ', '.join(missing))
    else:
        meta['h'] = _divide(h_l0 - 0.51 * cp_mean * ta_mean * e_l0,
                            1.0 + 0.51 * q_mean)                # [W m-2]

    p_ratio = _divide(P0A, meta['p_air_mean'])                  # [-]
    # 0.286 is R/cp for dry air. A negative base has no real power in Python, and a
    # negative pressure has no potential temperature either.
    poisson = p_ratio ** 0.286 if p_ratio >= 0.0 else NAN       # [-]
    theta_s = meta['ts_mean'] * poisson                         # [K] sonic POTENTIAL
    meta['theta_s'] = theta_s

    # The sonic potential temperature pairs with the sonic (buoyancy) covariance, never
    # with the Schotanus-corrected H: a virtual temperature belongs with its own flux.
    # The leading minus makes an upward heat flux give L < 0, i.e. unstable.
    meta['mo_length'] = _divide(-(theta_s * meta['ustar'] ** 3),
                                KAPPA * G * meta['cov_w_ts'])   # [m]
    # The floor sits on d, so a site with no declared displacement still gets z-d > 0.
    height = cfg.site.measurement_height - max(Z_MINUS_D_FLOOR, cfg.site.displacement)
    meta['z_l'] = _divide(height, meta['mo_length'])            # [-]
    return period
