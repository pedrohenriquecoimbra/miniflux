"""Tests for miniflux.flux.

Pins ALGORITHMS.md 14.4 (covariance identity and translation invariance) and 14.7 (the
Schotanus contrast), the three temperature-unit conventions of 8.3, the friction velocity
of 7.2, the sign of the Obukhov length of 9.4, and the rule that a degenerate parcel
yields NaN rather than an exception.

Standard library only, including here: the expected values are written out as arithmetic
straight from ALGORITHMS.md so a wrong evaluation order in flux.py cannot agree with them.
"""

import array
import logging
import math
import unittest
from types import SimpleNamespace

from miniflux import flux
from miniflux import kernels
from miniflux.constants import G, KAPPA, MV, P0A, R, RD, RV, T0


NAN = float('nan')

# One parcel, used wherever a realistic number is needed: midday, 20 degC, 99 kPa,
# 12 mmol mol-1 of dry air of water vapour, 420 umol mol-1 of dry air of CO2.
TA_K = 293.15
P_PA = 99000.0
H2O_R = 0.012                   # [mol mol-1 dry]
CO2_R = 420.0e-6                # [mol mol-1 dry]


def make_cfg(co2='mixing_ratio', h2o='mixing_ratio', height=3.0, displacement=0.0):
    """The three fields flux.py reads off a Config, and nothing else."""
    return SimpleNamespace(
        gases=SimpleNamespace(measure_type={'co2': co2, 'h2o': h2o}),
        site=SimpleNamespace(measurement_height=height, displacement=displacement))


def make_period(**series):
    """A period dict whose nine series are all present and the same length."""
    lengths = set(len(v) for v in series.values())
    n = lengths.pop() if lengths else 0
    period = {'meta': {}}
    for name in ('u', 'v', 'w', 'ts', 'co2', 'h2o', 'ta', 'p_air'):
        values = series.get(name)
        period[name] = array.array('d', values) if values is not None \
            else array.array('d', [NAN] * n)
    return period


def constant_parcel(n=64, ta=TA_K, ts=None, p=P_PA, h2o=H2O_R, co2=CO2_R):
    """n identical samples of the reference parcel, so every period mean is the sample."""
    if ts is None:
        # Ts = Ta (1 + 0.51 q) would be the physical pairing, but the thermodynamics does
        # not read Ts once a measured Ta is present, so the simple choice is honest here.
        ts = TA_K
    return make_period(u=[1.0] * n, v=[0.0] * n, w=[0.0] * n,
                       ts=[ts] * n, co2=[co2] * n, h2o=[h2o] * n,
                       ta=[ta] * n, p_air=[p] * n)


def expected_state(ta_k, p_pa, r_h2o, r_co2):
    """The ALGORITHMS 8.2 chain for one sample, written out here independently.

    Dry mixing ratios in [mol mol-1 dry]. Returns a dict of the per-sample quantities.
    """
    chi_v = r_h2o / (1.0 + r_h2o)
    va = R * ta_k / p_pa
    rho_v = chi_v * MV / va
    e = rho_v * RV * ta_k
    es = math.exp(77.345 + 0.0057 * ta_k - 7235.0 / ta_k) / ta_k ** 8.2
    rh = 100.0 * e / es
    pd = p_pa - e
    vd = R * ta_k / pd
    n_d = pd / (R * ta_k)
    rho_d = pd / (RD * ta_k)
    rho_m = rho_d + rho_v
    q = rho_v / rho_m
    ta_c = ta_k - T0
    cp_d = 1005.0 + (ta_c + 23.12) ** 2 / 3364.0
    cp_v = (1859.0 + 0.13 * rh + (0.193 + 5.6e-3 * rh) * ta_c
            + (1e-3 + 5e-5 * rh) * ta_c ** 2)
    return {'chi_v': chi_v, 'va': va, 'rho_v': rho_v, 'e': e, 'es': es, 'rh': rh,
            'pd': pd, 'vd': vd, 'n_d': n_d, 'rho_d': rho_d, 'rho_m': rho_m, 'q': q,
            'sigma': rho_v / rho_d, 'cp_d': cp_d, 'cp_v': cp_v,
            'cp': (1.0 - q) * cp_d + q * cp_v,
            'lambda_v': 1e3 * (3147.5 - 2.37 * ta_k),
            'rho_c': r_co2 * n_d * 0.04401}


class TestTemperatureConventions(unittest.TestCase):
    """ALGORITHMS 8.3: es and lambda_v take kelvin, cp_d and cp_v take celsius."""

    def test_saturation_vapour_pressure_takes_kelvin(self):
        # The formula's own value at 20 degC. ALGORITHMS 8.3 quotes 2343 Pa as the sanity
        # check; the formula it prescribes gives 2331.215 (the true es is 2339 Pa).
        self.assertAlmostEqual(flux.saturation_vapour_pressure(293.15),
                               2331.2150168593826, places=9)

    def test_saturation_vapour_pressure_refuses_celsius(self):
        # Feeding celsius is the classic failure: es collapses to ~1e-135 Pa and RH runs
        # to infinity, so the two arguments must never be confused.
        self.assertLess(flux.saturation_vapour_pressure(20.0), 1e-100)

    def test_saturation_vapour_pressure_degenerate(self):
        for bad in (0.0, -5.0, NAN):
            self.assertTrue(math.isnan(flux.saturation_vapour_pressure(bad)))

    def test_cp_dry_takes_kelvin_and_converts_internally(self):
        self.assertAlmostEqual(flux.cp_dry(293.15), 1005.5527153388823, places=9)
        # At 0 degC the quadratic term is 23.12^2/3364 on top of the 1005 intercept.
        self.assertAlmostEqual(flux.cp_dry(T0), 1005.0 + 23.12 ** 2 / 3364.0, places=12)

    def test_cp_vapour_takes_kelvin_and_percent(self):
        self.assertAlmostEqual(flux.cp_vapour(293.15, 50.0), 1876.36, places=6)
        # RH is not clipped: supersaturation feeds straight through.
        self.assertGreater(flux.cp_vapour(293.15, 120.0), flux.cp_vapour(293.15, 100.0))

    def test_latent_heat_takes_kelvin(self):
        self.assertAlmostEqual(flux.latent_heat(293.15), 2452734.5, places=6)

    def test_air_temperature_from_sonic_uses_032_on_e_over_p(self):
        got = flux.air_temperature_from_sonic(293.15, 1170.0, 99000.0)
        self.assertAlmostEqual(got, 293.15 / (1.0 + 0.32 * 1170.0 / 99000.0), places=12)
        # The sonic temperature is the warmer one: the correction removes humidity.
        self.assertLess(got, 293.15)


class TestThermodynamics(unittest.TestCase):

    def test_constant_parcel_matches_the_chain_sample_by_sample(self):
        period = constant_parcel()
        flux.thermodynamics(period, make_cfg())
        meta = period['meta']
        want = expected_state(TA_K, P_PA, H2O_R, CO2_R)
        for name, value in want.items():
            self.assertAlmostEqual(meta[name + '_mean'], value, delta=abs(value) * 1e-12,
                                   msg=name)
        self.assertAlmostEqual(meta['ta_mean'], TA_K, places=12)
        self.assertAlmostEqual(meta['p_air_mean'], P_PA, places=9)
        self.assertEqual(meta['n_ta_source_measured'], 64)

    def test_published_gas_diagnostics_have_a_fixed_unit(self):
        period = constant_parcel()
        flux.thermodynamics(period, make_cfg())
        # 1e6 mol mol-1 -> umol mol-1 and 1e3 mol mol-1 -> mmol mol-1, both of dry air.
        self.assertAlmostEqual(period['meta']['co2_dry_ppm'], 420.0, places=9)
        self.assertAlmostEqual(period['meta']['h2o_dry_ppt'], 12.0, places=12)

    def test_molar_density_branch_reaches_the_same_state(self):
        mixing = constant_parcel()
        flux.thermodynamics(mixing, make_cfg())
        # The same parcel written as densities. n_v comes from the vapour density itself
        # and n_co2 from the dry-air molar density, which is what the two branches mean
        # by "the same air"; chi_v = rho_v R Ta/(Mv P) is the exact inverse of the
        # mixing-ratio branch's rho_v = chi_v Mv/Va, so the states must coincide.
        density = constant_parcel(h2o=mixing['meta']['rho_v_mean'] / MV,
                                  co2=CO2_R * mixing['meta']['n_d_mean'])
        flux.thermodynamics(density, make_cfg(co2='molar_density', h2o='molar_density'))
        for name in ('rho_v', 'chi_v', 'e', 'rh', 'q', 'cp', 'rho_c', 'vd', 'n_d'):
            want = mixing['meta'][name + '_mean']
            self.assertAlmostEqual(density['meta'][name + '_mean'], want,
                                   delta=abs(want) * 1e-12, msg=name)
        # The diagnostics carry the same fixed unit by the other route. h2o_dry_ppt is the
        # one number that cannot agree exactly: the density branch reaches it through
        # n_v/n_d, and Pd carries the Mv Rv/R = 1.00021 rounding of ALGORITHMS 8.4 --
        # 3.1e-5 mmol mol-1 on 12, which is 0.021 % of the vapour itself.
        self.assertAlmostEqual(density['meta']['co2_dry_ppm'], 420.0, delta=1e-9)
        self.assertAlmostEqual(density['meta']['h2o_dry_ppt'], 12.0, delta=1e-4)

    def test_every_mean_is_taken_over_its_own_finite_samples(self):
        # Real periods have holes -- despiking opens them and the lag shift NaN-fills an
        # end -- so the denominator of every one of these means is the finite count, not
        # the sample count. A nan-mean of identical samples is exactly the sample, so
        # this needs no tolerance to say it; dividing by len() instead would put every
        # density, heat capacity and molar volume 10 % low here, and H 19 % low.
        # A different hole pattern per series, so no two quantities share a mask.
        n = 600
        period = constant_parcel(n=n)
        for i in range(0, n, 10):
            period['ts'][i] = NAN
        for i in range(0, n, 3):
            period['p_air'][i] = NAN
        for i in range(1, n, 7):
            period['h2o'][i] = NAN
            period['co2'][i] = NAN
        flux.thermodynamics(period, make_cfg())
        meta = period['meta']
        want = expected_state(TA_K, P_PA, H2O_R, CO2_R)
        for name, value in want.items():
            self.assertAlmostEqual(meta[name + '_mean'], value, delta=abs(value) * 1e-12,
                                   msg=name)
        # A running left-to-right sum of a few hundred identical samples is the sample to
        # about n*eps relative, not bit for bit, so these carry the same 1e-12 as above.
        self.assertAlmostEqual(meta['ta_mean'], TA_K, delta=TA_K * 1e-12)
        self.assertAlmostEqual(meta['ts_mean'], TA_K, delta=TA_K * 1e-12)
        self.assertAlmostEqual(meta['p_air_mean'], P_PA, delta=P_PA * 1e-12)
        self.assertEqual(meta['n_ta_source_measured'], n)     # ta itself is never missing

    def test_air_temperature_falls_back_per_sample(self):
        # Sample 0 has a measured Ta; sample 1 has none and is derived from the sonic;
        # sample 2 has no pressure either and falls back to the sonic temperature itself.
        period = make_period(u=[1.0] * 3, v=[0.0] * 3, w=[0.0] * 3,
                             ts=[295.0] * 3, co2=[CO2_R] * 3, h2o=[H2O_R] * 3,
                             ta=[290.0, NAN, NAN], p_air=[P_PA, P_PA, NAN])
        flux.thermodynamics(period, make_cfg())
        self.assertEqual(period['meta']['n_ta_source_measured'], 1)
        chi_v = H2O_R / (1.0 + H2O_R)
        rho_v = chi_v * MV * P_PA / (R * 295.0)
        derived = 295.0 / (1.0 + 0.32 * (rho_v * RV * 295.0) / P_PA)
        self.assertAlmostEqual(period['meta']['ta_mean'],
                               (290.0 + derived + 295.0) / 3.0, places=10)

    def test_degenerate_parcel_is_nan_not_an_exception(self):
        period = make_period(u=[NAN] * 8, v=[NAN] * 8, w=[NAN] * 8, ts=[NAN] * 8,
                             co2=[NAN] * 8, h2o=[NAN] * 8, ta=[NAN] * 8,
                             p_air=[NAN] * 8)
        logging.disable(logging.CRITICAL)
        try:
            flux.thermodynamics(period, make_cfg())
            flux.moments(period, make_cfg())
            flux.assemble(period, make_cfg())
        finally:
            logging.disable(logging.NOTSET)
        for key in ('ta_mean', 'cp_mean', 'q_mean', 'vd_mean', 'ustar', 'cov_w_ts',
                    'fc_l0', 'e_l0', 'le_l0', 'h_l0', 'theta_s', 'mo_length', 'z_l'):
            self.assertTrue(math.isnan(period['meta'][key]), key)
        self.assertEqual(period['meta']['n_ta_source_measured'], 0)
        self.assertEqual(period['meta']['n_cov_w_ts'], 0)
        # H is never aliased to H_L0; the key is simply absent.
        self.assertNotIn('h', period['meta'])

    def test_a_zero_pressure_sample_does_not_divide_by_zero(self):
        period = constant_parcel(n=4, p=0.0)
        logging.disable(logging.CRITICAL)
        try:
            flux.thermodynamics(period, make_cfg())
        finally:
            logging.disable(logging.NOTSET)
        self.assertTrue(math.isnan(period['meta']['va_mean']))


class TestMoments(unittest.TestCase):
    """ALGORITHMS 14.4 and 7.2."""

    def series(self):
        # One gust drives every series, so the covariances are a real fraction of the
        # standard deviations; translation invariance on a near-zero covariance would
        # test nothing but the cancellation it is meant to rule out.
        gust = [math.sin(0.11 * i) + 0.4 * math.sin(0.37 * i + 1.0) for i in range(200)]
        u = [3.0 + 0.4 * g for g in gust]
        v = [0.2 * math.cos(0.31 * i) for i in range(200)]
        w = [-0.15 * g for g in gust]                 # downward momentum transport
        ts = [293.0 + 0.3 * g for g in gust]
        co2 = [4.2e-4 - 2e-6 * g for g in gust]
        h2o = [1.2e-2 + 3e-5 * g for g in gust]
        return u, v, w, ts, co2, h2o

    def test_cov_of_a_series_with_itself_is_its_variance(self):
        u, v, w, ts, co2, h2o = self.series()
        period = make_period(u=u, v=v, w=w, ts=ts, co2=co2, h2o=h2o)
        flux.moments(period, make_cfg())
        self.assertEqual(period['meta']['var_u'],
                         kernels.cov(period['u'], period['u']))
        self.assertEqual(period['meta']['var_ts'],
                         kernels.cov(period['ts'], period['ts']))

    def test_covariance_is_translation_invariant(self):
        # ALGORITHMS 14.4: cov(x + 1e6, y) == cov(x, y) to within 1e-9 relative. That is
        # the whole point of the two-pass form; a one-pass raw-sums kernel loses this.
        u, v, w, ts, co2, h2o = self.series()
        base = make_period(u=u, v=v, w=w, ts=ts, co2=co2, h2o=h2o)
        shifted = make_period(u=[x + 1e6 for x in u], v=v, w=w, ts=ts, co2=co2, h2o=h2o)
        flux.moments(base, make_cfg())
        flux.moments(shifted, make_cfg())
        for key in ('cov_u_w', 'cov_u_v', 'var_u'):
            want = base['meta'][key]
            self.assertAlmostEqual(shifted['meta'][key], want,
                                   delta=abs(want) * 1e-9, msg=key)

    def test_gas_covariance_survives_its_own_mean(self):
        # The case ALGORITHMS 0.4 names: a 420 ppm mean over a 1 ppm fluctuation, here in
        # canonical mol mol-1. Offsetting the whole series by its mean must not move the
        # covariance. (A 1e6 offset could not be survived by any kernel at this scale:
        # it leaves the 1e-6 fluctuation about five significant bits.)
        _u, _v, w, _ts, _co2, _h2o = self.series()
        gust = [x / -0.15 for x in w]
        centred = make_period(u=[0.0] * 200, v=[0.0] * 200, w=w, ts=[293.0] * 200,
                              co2=[1e-6 * g for g in gust], h2o=[H2O_R] * 200)
        offset = make_period(u=[0.0] * 200, v=[0.0] * 200, w=w, ts=[293.0] * 200,
                             co2=[4.2e-4 + 1e-6 * g for g in gust], h2o=[H2O_R] * 200)
        logging.disable(logging.CRITICAL)
        try:
            flux.moments(centred, make_cfg())
            flux.moments(offset, make_cfg())
        finally:
            logging.disable(logging.NOTSET)
        want = centred['meta']['cov_w_co2']
        self.assertAlmostEqual(offset['meta']['cov_w_co2'], want,
                               delta=abs(want) * 1e-9)

    def test_joint_finite_counts_pairs_not_samples(self):
        period = make_period(u=[1.0, 2.0, 3.0, 4.0], v=[0.0] * 4,
                             w=[1.0, NAN, 3.0, 4.0], ts=[1.0, 2.0, NAN, 4.0],
                             co2=[NAN] * 4, h2o=[1.0] * 4)
        flux.moments(period, make_cfg())
        self.assertEqual(period['meta']['n_cov_w_ts'], 2)     # samples 0 and 3
        self.assertEqual(period['meta']['n_cov_w_co2'], 0)
        self.assertEqual(period['meta']['n_cov_w_h2o'], 3)

    def test_ustar_from_planted_covariances(self):
        # A square wave of period 2 has mean zero exactly, so cov(x, w) = -4a/3 over the
        # four samples: a = 0.225 plants cov_u_w = -0.3 and b = 0.03 plants
        # cov_v_w = -0.04, both exact in binary.
        period = make_period(u=[0.225, -0.225, 0.225, -0.225],
                             v=[0.03, -0.03, 0.03, -0.03],
                             w=[-1.0, 1.0, -1.0, 1.0], ts=[293.0] * 4,
                             co2=[CO2_R] * 4, h2o=[H2O_R] * 4)
        flux.moments(period, make_cfg())
        meta = period['meta']
        self.assertAlmostEqual(meta['cov_u_w'], -0.3, places=15)
        self.assertAlmostEqual(meta['cov_v_w'], -0.04, places=15)
        # (0.09 + 0.0016)**0.25, the full stress vector.
        self.assertAlmostEqual(meta['ustar'], 0.5501408174353464, places=15)
        # sqrt(|cov_u_w|) = 0.5477 is the one-component answer; it is not this one.
        self.assertNotAlmostEqual(meta['ustar'], math.sqrt(0.3), places=3)

    def test_ustar_from_real_series(self):
        u, v, w, ts, co2, h2o = self.series()
        period = make_period(u=u, v=v, w=w, ts=ts, co2=co2, h2o=h2o)
        flux.moments(period, make_cfg())
        meta = period['meta']
        want = (kernels.cov(period['u'], period['w']) ** 2
                + kernels.cov(period['v'], period['w']) ** 2) ** 0.25
        self.assertEqual(meta['ustar'], want)
        self.assertGreater(meta['ustar'], 0.0)

    def test_zero_variance_gives_zero_covariance_not_nan(self):
        period = make_period(u=[2.0] * 16, v=[1.0] * 16, w=[0.0] * 16, ts=[293.0] * 16,
                             co2=[CO2_R] * 16, h2o=[H2O_R] * 16)
        logging.disable(logging.CRITICAL)
        try:
            flux.moments(period, make_cfg())
        finally:
            logging.disable(logging.NOTSET)
        self.assertEqual(period['meta']['var_u'], 0.0)
        self.assertEqual(period['meta']['cov_u_w'], 0.0)
        self.assertEqual(period['meta']['ustar'], 0.0)


def planted_period(cov_w_ts, cov_w_h2o, cov_w_co2=-1.0e-7, ustar=0.4,
                   p_air=P_PA, ts_mean=None):
    """A period whose thermodynamic and moment meta are planted, ready for assemble.

    ``ts_mean`` defaults to the virtual temperature ``Ta (1 + 0.51 q)`` -- what a sonic
    actually reads, +1.10 K on this parcel. A fixture where it equals ``ta_mean`` cannot
    tell ALGORITHMS 9.3 and 9.4's two temperature pairings apart: theta_s built from the
    ambient mean instead of the sonic one moves every stability-dependent number by
    0.37 % and no assertion would move with it.
    """
    state = expected_state(TA_K, P_PA, H2O_R, CO2_R)
    if ts_mean is None:
        ts_mean = TA_K * (1.0 + 0.51 * state['q'])
    meta = {'ta_mean': TA_K, 'ts_mean': ts_mean, 'p_air_mean': p_air,
            'cp_mean': state['cp'], 'q_mean': state['q'],
            'rho_m_mean': state['rho_m'], 'lambda_v_mean': state['lambda_v'],
            'vd_mean': state['vd'], 'cov_w_ts': cov_w_ts, 'cov_w_h2o': cov_w_h2o,
            'cov_w_co2': cov_w_co2, 'ustar': ustar}
    period = make_period()
    period['meta'] = meta
    return period, state


def cov_for_le(le_w_m2, state):
    """The w'h2o' covariance [mol mol-1 m s-1] that produces le_l0 = le_w_m2 [W m-2]."""
    return le_w_m2 / state['lambda_v'] / MV * state['vd']


class TestAssemble(unittest.TestCase):

    def test_level_zero_fluxes_and_the_flux_factor(self):
        period, state = planted_period(0.16, 1.0e-4)
        flux.assemble(period, make_cfg())
        meta = period['meta']
        # A dry mixing ratio needs the dry-air molar density; a molar density does not.
        self.assertAlmostEqual(meta['f_co2'], 1.0 / state['vd'], places=12)
        self.assertAlmostEqual(meta['fc_l0'], -1.0e-7 / state['vd'], places=18)
        self.assertAlmostEqual(meta['e_l0'], 1.0e-4 / state['vd'] * MV, places=15)
        self.assertAlmostEqual(meta['le_l0'], meta['e_l0'] * state['lambda_v'],
                               places=9)
        self.assertAlmostEqual(meta['h_l0'], 0.16 * state['rho_m'] * state['cp'],
                               places=9)

    def test_flux_factor_is_one_for_a_molar_density(self):
        period, _state = planted_period(0.16, 1.0e-4)
        flux.assemble(period, make_cfg(co2='molar_density', h2o='molar_density'))
        self.assertEqual(period['meta']['f_co2'], 1.0)
        self.assertEqual(period['meta']['f_h2o'], 1.0)
        self.assertAlmostEqual(period['meta']['fc_l0'], -1.0e-7, places=18)

    def test_each_gas_gets_the_factor_of_its_own_measure_type(self):
        # config.py accepts a mixed pair -- an open-path H2O beside a closed-path CO2 is
        # an ordinary instrument set -- and the two factors differ by 1/<Vd> = 40. Read
        # one gas's declaration for the other and LE comes out 40x wrong.
        for co2, h2o in (('molar_density', 'mixing_ratio'),
                         ('mixing_ratio', 'molar_density')):
            period, state = planted_period(0.16, 1.0e-4)
            flux.assemble(period, make_cfg(co2=co2, h2o=h2o))
            meta = period['meta']
            self.assertAlmostEqual(
                meta['f_co2'], 1.0 if co2 == 'molar_density' else 1.0 / state['vd'],
                places=12, msg=co2)
            self.assertAlmostEqual(
                meta['f_h2o'], 1.0 if h2o == 'molar_density' else 1.0 / state['vd'],
                places=12, msg=h2o)
            self.assertNotAlmostEqual(meta['f_co2'], meta['f_h2o'], places=6)
            self.assertAlmostEqual(meta['e_l0'], 1.0e-4 * meta['f_h2o'] * MV, places=15)

    def test_schotanus_against_a_hand_computed_parcel(self):
        # 20 degC, 99 kPa, q = 0.00735 kg kg-1, cp = 1011.96 J kg-1 K-1.
        # cov(w,ts) = 0.16 K m s-1 and an E_L0 chosen to give LE_L0 = 180 W m-2.
        period, state = planted_period(0.16, 0.0)
        period['meta']['cov_w_h2o'] = cov_for_le(180.0, state)
        flux.assemble(period, make_cfg())
        meta = period['meta']
        self.assertAlmostEqual(meta['le_l0'], 180.0, places=9)
        # The division form of van Dijk et al. (2004) Eq. 3.53, written out here. The
        # latent term takes the AMBIENT mean temperature (ALGORITHMS 9.3): the sonic mean
        # is 1.10 K warmer on this parcel, so the two are separable here.
        numerator = meta['h_l0'] - 0.51 * state['cp'] * TA_K * meta['e_l0']
        denominator = 1.0 + 0.51 * state['q']
        self.assertAlmostEqual(meta['h'], numerator / denominator,
                               delta=abs(meta['h']) * 1e-13)
        sonic_variant = meta['h_l0'] - 0.51 * state['cp'] * meta['ts_mean'] * meta['e_l0']
        self.assertNotAlmostEqual(meta['h'], sonic_variant / denominator, places=4)
        # The linearised subtraction EddyPro uses differs by exactly the denominator.
        self.assertAlmostEqual(numerator / meta['h'], denominator, places=12)
        # The latent term is ~6 % of H_L0 here and the dilution ~0.4 %.
        self.assertLess(meta['h'], meta['h_l0'])

    def test_schotanus_contrast_algorithms_14_7(self):
        # ALGORITHMS 14.7: H_L0 / H must land near 1.04 on a midday period. With
        # cp = 1011.96, Ta = 293.15 K and lambda_v = 2.4527e6 the ratio is
        # (1 + 0.51 q) + 0.51 cp Ta / lambda_v / Bowen = 1.00375 + 0.06169/Bowen,
        # so 1.04 is the Bowen = 1.70 period, not the Bowen = 1 one.
        _p, state = planted_period(0.0, 0.0)
        k = 0.51 * state['cp'] * TA_K / state['lambda_v']
        h_target = 180.0

        period, _state = planted_period(0.16, 0.0)
        period['meta']['cov_w_h2o'] = cov_for_le(h_target / 1.7018, state)
        # cov(w,ts) that makes H come out at h_target, so the ratio is H_L0/h_target.
        h_l0 = (h_target * (1.0 + 0.51 * state['q'])
                + k * h_target / 1.7018)
        period['meta']['cov_w_ts'] = h_l0 / (state['rho_m'] * state['cp'])
        flux.assemble(period, make_cfg())
        meta = period['meta']
        self.assertAlmostEqual(meta['h'], h_target, delta=1e-9)
        self.assertAlmostEqual(meta['h_l0'] / meta['h'], 1.04, delta=0.001)

        # And the Bowen = 1 period ALGORITHMS 14.7 names, which gives 1.0654, not 1.04.
        period, _state = planted_period(0.16, 0.0)
        period['meta']['cov_w_h2o'] = cov_for_le(h_target, state)
        h_l0 = h_target * (1.0 + 0.51 * state['q']) + k * h_target
        period['meta']['cov_w_ts'] = h_l0 / (state['rho_m'] * state['cp'])
        flux.assemble(period, make_cfg())
        self.assertAlmostEqual(period['meta']['h_l0'] / period['meta']['h'],
                               1.0 + 0.51 * state['q'] + k, delta=1e-9)
        self.assertAlmostEqual(period['meta']['h_l0'] / period['meta']['h'],
                               1.0654, delta=0.0005)

    def test_h_is_absent_when_an_input_is_missing(self):
        period, _state = planted_period(0.16, 1.0e-4)
        period['meta']['cp_mean'] = NAN
        logging.disable(logging.CRITICAL)
        try:
            flux.assemble(period, make_cfg())
        finally:
            logging.disable(logging.NOTSET)
        self.assertNotIn('h', period['meta'])
        self.assertTrue(math.isnan(period['meta']['h_l0']))

    def test_theta_s_is_the_sonic_potential_temperature(self):
        # ALGORITHMS 9.4: the SONIC mean, because theta_s pairs with the sonic
        # (buoyancy) covariance. The ambient mean is 1.10 K colder on this parcel and
        # would move L and z/L by 0.37 % with nothing else changing.
        period, _state = planted_period(0.16, 1.0e-4)
        flux.assemble(period, make_cfg())
        poisson = (P0A / P_PA) ** 0.286
        self.assertAlmostEqual(period['meta']['theta_s'],
                               period['meta']['ts_mean'] * poisson, places=12)
        self.assertNotAlmostEqual(period['meta']['theta_s'], TA_K * poisson, places=3)
        # Below the reference pressure the potential temperature is the warmer one.
        self.assertGreater(period['meta']['theta_s'], period['meta']['ts_mean'])

    def test_obukhov_length_is_negative_when_unstable(self):
        # An upward buoyancy flux (cov(w,ts) > 0) is unstable: L < 0 and z/L < 0.
        period, _state = planted_period(0.16, 1.0e-4, ustar=0.4)
        flux.assemble(period, make_cfg(height=3.0, displacement=0.0))
        meta = period['meta']
        want = -(meta['theta_s'] * 0.4 ** 3) / (KAPPA * G * 0.16)
        self.assertAlmostEqual(meta['mo_length'], want, delta=abs(want) * 1e-13)
        self.assertLess(meta['mo_length'], 0.0)
        self.assertLess(meta['z_l'], 0.0)
        # z - d, with the floor sitting on d so an undeclared displacement still leaves
        # a positive height.
        self.assertAlmostEqual(meta['z_l'], (3.0 - 1.0e-4) / meta['mo_length'],
                               places=12)

    def test_obukhov_length_is_positive_when_stable(self):
        # A downward buoyancy flux (cov(w,ts) < 0) is stable: L > 0 and z/L > 0.
        period, _state = planted_period(-0.05, -1.0e-5, ustar=0.25)
        flux.assemble(period, make_cfg(height=3.0, displacement=0.0))
        meta = period['meta']
        self.assertGreater(meta['mo_length'], 0.0)
        self.assertGreater(meta['z_l'], 0.0)

    def test_displacement_is_subtracted_from_the_measurement_height(self):
        period, _state = planted_period(0.16, 1.0e-4)
        flux.assemble(period, make_cfg(height=3.0, displacement=1.2))
        meta = period['meta']
        self.assertAlmostEqual(meta['z_l'], (3.0 - 1.2) / meta['mo_length'], places=12)

    def test_zero_buoyancy_flux_gives_an_infinite_length_not_an_exception(self):
        # ALGORITHMS 9.4 leaves this degeneracy unguarded on purpose: L = +-inf, z/L = 0.
        period, _state = planted_period(0.0, 1.0e-4, ustar=0.4)
        flux.assemble(period, make_cfg())
        self.assertTrue(math.isinf(period['meta']['mo_length']))
        self.assertLess(period['meta']['mo_length'], 0.0)
        self.assertEqual(period['meta']['z_l'], 0.0)

    def test_zero_ustar_gives_a_zero_length_and_an_infinite_stability(self):
        period, _state = planted_period(0.16, 1.0e-4, ustar=0.0)
        flux.assemble(period, make_cfg())
        self.assertEqual(period['meta']['mo_length'], 0.0)
        self.assertTrue(math.isinf(period['meta']['z_l']))

    def test_zero_pressure_does_not_divide_by_zero(self):
        period, _state = planted_period(0.16, 1.0e-4, p_air=0.0)
        flux.assemble(period, make_cfg())
        self.assertTrue(math.isinf(period['meta']['theta_s']))


class TestEndToEnd(unittest.TestCase):
    """thermodynamics -> moments -> assemble on one synthetic period."""

    def test_a_synthetic_period_produces_a_plausible_flux_set(self):
        n = 600
        u, v, w, ts, co2, h2o, ta, p_air = [], [], [], [], [], [], [], []
        for i in range(n):
            gust = math.sin(0.11 * i) + 0.4 * math.sin(0.37 * i + 1.0)
            u.append(2.5 + 0.3 * gust)
            v.append(0.1 * math.cos(0.23 * i))
            w.append(0.25 * gust)
            ts.append(TA_K + 0.35 * gust)               # warm updraughts: H > 0
            h2o.append(H2O_R + 8.0e-5 * gust)           # moist updraughts: LE > 0
            co2.append(CO2_R - 6.0e-6 * gust)           # uptake: FC < 0
            ta.append(TA_K + 0.3 * gust)
            p_air.append(P_PA)
        period = make_period(u=u, v=v, w=w, ts=ts, co2=co2, h2o=h2o, ta=ta, p_air=p_air)
        cfg = make_cfg(height=3.0, displacement=0.0)
        flux.thermodynamics(period, cfg)
        flux.moments(period, cfg)
        flux.assemble(period, cfg)
        meta = period['meta']
        self.assertEqual(meta['n_ta_source_measured'], n)
        self.assertGreater(meta['h_l0'], 0.0)
        self.assertGreater(meta['le_l0'], 0.0)
        self.assertLess(meta['fc_l0'], 0.0)
        self.assertLess(meta['h'], meta['h_l0'])        # Schotanus always removes here
        self.assertLess(meta['mo_length'], 0.0)         # upward heat flux -> unstable
        self.assertLess(meta['z_l'], 0.0)
        self.assertTrue(1.0 < meta['rho_m_mean'] < 1.3)
        self.assertTrue(1000.0 < meta['cp_mean'] < 1030.0)
        self.assertTrue(0.0 < meta['rh_mean'] < 100.0)
        self.assertAlmostEqual(meta['co2_dry_ppm'], 420.0, delta=0.5)


if __name__ == '__main__':
    unittest.main()
