"""Tests for `miniflux.wpl`.

Pins ALGORITHMS.md 14.5 (the water identity), 14.6 and the sanity magnitudes of 10.8 (the
midday parcel), CONTRACT.md 20.9 (`wt` comes from the published H), and the `off`-but-owed
refusal of CONTRACT.md 12.

The parcel's thermodynamic state is transcribed here from ALGORITHMS.md 8.2 rather than
imported from `flux.py`, so that this file exercises `wpl.py` and nothing else.
"""

import math
import unittest
from array import array
from types import SimpleNamespace

from miniflux import wpl
from miniflux.constants import MCO2, MD, MU, MV, R, RD, RV

# The midday parcel of ALGORITHMS 10.8.
TA = 293.0            # K
P_AIR = 99000.0       # Pa
CHI_V = 0.012         # mol mol-1, water vapour mole fraction of moist air
CO2 = 420e-6          # mol mol-1 of dry air
H = 180.0             # W m-2, the Schotanus-corrected sensible heat flux
LE_L0 = 186.6         # W m-2, the uncorrected latent heat flux
FC_L0 = -20.0e-6      # mol m-2 s-1, the uncorrected CO2 flux


def _state():
    """The parcel's mean state, by the per-sample chain of ALGORITHMS 8.2."""
    va = R * TA / P_AIR                                        # m3 mol-1
    rho_v = CHI_V * MV / va                                    # kg m-3
    e = rho_v * RV * TA                                        # Pa
    es = math.exp(77.345 + 0.0057 * TA - 7235.0 / TA) / TA ** 8.2   # Pa, TA in KELVIN
    rh = 100.0 * e / es                                        # %
    pd = P_AIR - e                                             # Pa, dry partial pressure
    rho_d = pd / (RD * TA)                                     # kg m-3
    rho_m = rho_d + rho_v                                      # kg m-3
    q = rho_v / rho_m                                          # kg kg-1
    tc = TA - 273.15                                           # degC for the cp fits only
    cp_d = 1005.0 + (tc + 23.12) ** 2 / 3364.0
    cp_v = 1859.0 + 0.13 * rh + (0.193 + 5.6e-3 * rh) * tc + (1e-3 + 5e-5 * rh) * tc ** 2
    n_co2 = CO2 * pd / (R * TA)                                # mol m-3
    return {
        'ta_mean': TA,
        'rho_v_mean': rho_v,
        'rho_d_mean': rho_d,
        'rho_m_mean': rho_m,
        'chi_v_mean': CHI_V,
        'sigma_mean': rho_v / rho_d,
        'cp_mean': (1.0 - q) * cp_d + q * cp_v,
        'lambda_v_mean': 1e3 * (3147.5 - 2.37 * TA),           # J kg-1, TA in KELVIN
        'rho_c_mean': n_co2 * MCO2,                            # kg m-3
        'n_co2': n_co2,                                        # mol m-3, for cross-checks
        'n_d': rho_d / MD,                                     # mol m-3, for cross-checks
    }


def _period(**overrides):
    """A period whose meta is the parcel, with the series left as detrending leaves them.

    The series are mean-zero on purpose: a `wpl.py` that rebuilt a mean density from one
    instead of reading the mean thermodynamics took before detrending would find ~0 and
    both Webb terms would vanish (ALGORITHMS 10.6).
    """
    state = _state()
    lambda_v = state['lambda_v_mean']
    meta = dict(state)
    meta.update({
        'h': H,
        'h_l0': 1.05 * H,
        'fc_l0': FC_L0,
        'e_l0': LE_L0 / lambda_v,
        'le_l0': LE_L0,
        'fh2o_l0': LE_L0 / lambda_v / MV,
        'cov_w_ts': 9.9e9,          # absurd on purpose: wpl must never read it (CONTRACT 20.9)
    })
    meta.update(overrides)
    zero_mean = array('d', [-1.0, 1.0, -1.0, 1.0])
    period = {'meta': meta, 't': []}
    for name in ('u', 'v', 'w', 'ts', 'co2', 'h2o', 'ta', 'p_air'):
        period[name] = array('d', zero_mean)
    return period, state


def _cfg(enabled='auto', co2='molar_density', h2o='molar_density'):
    return SimpleNamespace(
        wpl=SimpleNamespace(enabled=enabled),
        gases=SimpleNamespace(co2_measure_type=co2, h2o_measure_type=h2o,
                              measure_type={'co2': co2, 'h2o': h2o}))


class WaterIdentity(unittest.TestCase):
    """ALGORITHMS 14.5."""

    def test_a_dry_mixing_ratio_gives_back_mv_times_the_raw_vapour_flux(self):
        # With rho_v and rho_d from one ideal-gas state, (1 + mu sigma) = 1/(1 - chi_v),
        # so Webb's E collapses to Mv * FH2O_L0: a dry mixing ratio owes nothing.
        period, _ = _period()
        wpl.correct(period, _cfg(enabled='on', co2='mixing_ratio', h2o='mixing_ratio'))
        expected = MV * period['meta']['fh2o_l0']
        self.assertAlmostEqual(period['meta']['e'], expected,
                               delta=5e-4 * abs(expected))     # the Rd/Rv rounding

    def test_the_identity_also_holds_through_le(self):
        period, state = _period()
        wpl.correct(period, _cfg(enabled='on', co2='mixing_ratio', h2o='mixing_ratio'))
        self.assertAlmostEqual(period['meta']['le'], LE_L0, delta=5e-4 * LE_L0)


class MiddayMagnitudes(unittest.TestCase):
    """ALGORITHMS 10.8 and 14.6, on a parcel where both gases are molar densities."""

    def setUp(self):
        self.period, self.state = _period()
        wpl.correct(self.period, _cfg())
        self.meta = self.period['meta']

    def test_the_correction_is_tens_of_percent_of_the_flux_not_a_few_percent(self):
        correction = self.meta['fc'] - FC_L0
        # The rule of thumb of ALGORITHMS 10.8: by day the CO2 correction is tens of
        # percent to about one times the flux. A few percent means wt came from cov(w,ts)
        # or a mean density survived detrending.
        self.assertGreater(correction, 0.3 * abs(FC_L0))
        self.assertLess(correction, 1.2 * abs(FC_L0))

    def test_the_corrected_flux_reproduces_the_published_parcel(self):
        fc_umol = self.meta['fc'] * 1e6
        # ALGORITHMS 14.6 prints -20.00 -> -9.49. Implementing the 8.2 state exactly gives
        # -9.386: 10.8's parcel used rho_m = P/(Rd Ta) + rho_v, i.e. the dry-air density at
        # the TOTAL pressure rather than at Pd = P - e. That is 1.2 % high on rho_m, hence
        # 1.2 % low on wt and on the thermal term. The difference is entirely in the input
        # state; PublishedParcel below feeds 10.8's own rho_m and lands on -9.49.
        self.assertAlmostEqual(fc_umol, -9.386, delta=0.005 * 9.386)
        self.assertAlmostEqual(fc_umol, -9.49, delta=0.016 * 9.49)

    def test_the_two_webb_terms_have_their_published_magnitudes(self):
        wt = self.meta['wt']
        f_chi = self.meta['fh2o_l0'] + (self.state['rho_v_mean'] / MV) * wt / TA
        thermal = 1e6 * self.state['n_co2'] * wt / TA
        dilution = 1e6 * self.state['n_co2'] * f_chi / self.state['n_d']
        self.assertAlmostEqual(thermal, 8.735, delta=0.005 * 8.735)   # 10.8 prints 8.63
        self.assertAlmostEqual(dilution, 1.879, delta=0.005 * 1.879)  # 10.8 prints 1.88
        self.assertAlmostEqual(1e6 * self.meta['fc'], -20.0 + thermal + dilution, places=9)

    def test_the_molar_cross_check_form_is_exact(self):
        # ALGORITHMS 10.5: FC = FC_L0 + c_c f_chi/n_d + c_c wT/<Ta>, algebraically identical
        # to the mass form the module writes.
        wt = self.meta['wt']
        f_chi = self.meta['fh2o_l0'] + (self.state['rho_v_mean'] / MV) * wt / TA
        molar = (FC_L0 + self.state['n_co2'] * f_chi / self.state['n_d']
                 + self.state['n_co2'] * wt / TA)
        self.assertAlmostEqual(self.meta['fc'], molar, delta=1e-12 * abs(molar))

    def test_the_water_expansion_lifts_le_by_about_seven_percent(self):
        self.assertAlmostEqual(self.meta['le'] / LE_L0, 1.073, delta=0.005)
        self.assertAlmostEqual(self.meta['le'], self.meta['lambda_v_mean'] * self.meta['e'],
                               places=12)

    def test_the_daytime_correction_is_positive_and_shrinks_the_sink(self):
        self.assertGreater(self.meta['wt'], 0.0)
        self.assertGreater(self.meta['fc'], FC_L0)
        self.assertLess(self.meta['fc'], 0.0)
        self.assertTrue(self.meta['wpl_applied'])


class PublishedParcel(unittest.TestCase):
    """ALGORITHMS 14.6 with 10.8's own mean air density, which is where the 1 % lives."""

    def test_fc_lands_on_the_published_value(self):
        state = _state()
        rho_m_of_10_8 = P_AIR / (RD * TA) + state['rho_v_mean']   # vapour not subtracted
        period, _ = _period(rho_m_mean=rho_m_of_10_8)
        wpl.correct(period, _cfg())
        self.assertAlmostEqual(period['meta']['fc'] * 1e6, -9.49, delta=0.005 * 9.49)
        self.assertAlmostEqual(period['meta']['le'], 200.0, delta=0.005 * 200.0)


class TheExpansionFactor(unittest.TestCase):
    """ALGORITHMS 10.2: sigma is <rho_v>/<rho_d>, a ratio of the two mean densities."""

    def test_sigma_is_the_ratio_of_the_means_not_the_mean_of_the_ratio(self):
        # flux.thermodynamics also publishes `sigma_mean`, which is the per-sample mean of
        # ALGORITHMS 8.2 -- a different number by the Jensen gap (2.6e-5 relative on the
        # integration fixture). 10.2's sigma, and the parent's, is the ratio of the means,
        # and this module holds both of them. Planting a wrong sigma_mean shows which one
        # the expansion factor is built from.
        period, state = _period(sigma_mean=2.0 * _state()['sigma_mean'])
        wpl.correct(period, _cfg(enabled='on'))
        wt = period['meta']['wt']
        rho_v, rho_d = state['rho_v_mean'], state['rho_d_mean']
        expansion = 1.0 + MU * (rho_v / rho_d)
        f_chi = period['meta']['fh2o_l0'] + (rho_v / MV) * wt / TA
        w_rho_v = MV * f_chi - rho_v * wt / TA
        expected = expansion * (w_rho_v + (rho_v / TA) * wt)
        self.assertAlmostEqual(period['meta']['e'], expected, delta=1e-15 * abs(expected))


class TheHeatInput(unittest.TestCase):
    """CONTRACT 20.9: wt is the published H divided back down, never cov(w, ts)."""

    def test_wt_is_h_over_rho_m_cp(self):
        period, state = _period()
        wpl.correct(period, _cfg())
        self.assertAlmostEqual(period['meta']['wt'],
                               H / (state['rho_m_mean'] * state['cp_mean']), places=15)

    def test_the_sonic_covariance_in_meta_is_ignored(self):
        # _period plants cov_w_ts = 9.9e9; reading it would blow the flux up.
        period, _ = _period()
        wpl.correct(period, _cfg())
        self.assertLess(abs(period['meta']['fc']), 1e-4)

    def test_a_ten_percent_error_in_h_moves_fc_by_several_percent(self):
        base, _ = _period()
        wpl.correct(base, _cfg())
        high, _ = _period(h=1.1 * H)
        wpl.correct(high, _cfg())
        shift = abs(high['meta']['fc'] - base['meta']['fc']) / abs(base['meta']['fc'])
        self.assertGreater(shift, 0.05)          # the thermal term really is that large


class DecisionTable(unittest.TestCase):
    """CONTRACT 12, the five rows."""

    def test_auto_with_nothing_owed_copies_the_l0_fluxes(self):
        period, _ = _period()
        wpl.correct(period, _cfg(enabled='auto', co2='mixing_ratio', h2o='mixing_ratio'))
        meta = period['meta']
        self.assertEqual(meta['fc'], FC_L0)
        self.assertEqual(meta['le'], LE_L0)
        self.assertEqual(meta['e'], meta['e_l0'])
        self.assertFalse(meta['wpl_applied'])
        self.assertNotIn('wt', meta)

    def test_off_with_nothing_owed_copies_the_l0_fluxes(self):
        period, _ = _period()
        wpl.correct(period, _cfg(enabled='off', co2='mixing_ratio', h2o='mixing_ratio'))
        self.assertEqual(period['meta']['fc'], FC_L0)
        self.assertFalse(period['meta']['wpl_applied'])

    def test_on_runs_even_when_nothing_is_owed(self):
        period, _ = _period()
        wpl.correct(period, _cfg(enabled='on', co2='mixing_ratio', h2o='mixing_ratio'))
        meta = period['meta']
        self.assertTrue(meta['wpl_applied'])
        self.assertEqual(meta['fc'], FC_L0)      # a dry mixing ratio is owed nothing
        self.assertIn('wt', meta)

    def test_auto_with_one_gas_owed_runs_and_leaves_the_other_alone(self):
        period, _ = _period()
        wpl.correct(period, _cfg(co2='mixing_ratio', h2o='molar_density'))
        meta = period['meta']
        self.assertTrue(meta['wpl_applied'])
        self.assertEqual(meta['fc'], FC_L0)
        self.assertGreater(meta['le'], LE_L0)    # the water still gets Eq. 25


class OffButOwed(unittest.TestCase):
    """CONTRACT 12 and 19: an uncorrected number never wears a corrected name."""

    def test_no_fc_is_reported_for_a_molar_density_co2(self):
        period, _ = _period()
        with self.assertLogs('miniflux.wpl', level='WARNING') as caught:
            wpl.correct(period, _cfg(enabled='off', co2='molar_density',
                                     h2o='mixing_ratio'))
        meta = period['meta']
        self.assertNotIn('fc', meta)
        self.assertEqual(meta['fc_l0'], FC_L0)   # the raw flux is kept and still reported
        self.assertEqual(meta['e'], meta['e_l0'])
        self.assertEqual(meta['le'], LE_L0)
        self.assertFalse(meta['wpl_applied'])
        self.assertIn('co2', caught.output[0])

    def test_no_e_or_le_is_reported_for_a_molar_density_h2o(self):
        period, _ = _period()
        with self.assertLogs('miniflux.wpl', level='WARNING'):
            wpl.correct(period, _cfg(enabled='off', co2='mixing_ratio',
                                     h2o='molar_density'))
        meta = period['meta']
        self.assertNotIn('e', meta)
        self.assertNotIn('le', meta)
        self.assertEqual(meta['fc'], FC_L0)
        self.assertEqual(meta['le_l0'], LE_L0)

    def test_both_owed_leaves_all_three_absent(self):
        period, _ = _period()
        with self.assertLogs('miniflux.wpl', level='WARNING') as caught:
            wpl.correct(period, _cfg(enabled='off'))
        for key in ('fc', 'e', 'le'):
            self.assertNotIn(key, period['meta'])
        self.assertEqual(len(caught.output), 2)

    def test_a_previously_written_fc_is_removed_not_left_stale(self):
        period, _ = _period()
        wpl.correct(period, _cfg(enabled='auto'))
        self.assertIn('fc', period['meta'])
        with self.assertLogs('miniflux.wpl', level='WARNING'):
            wpl.correct(period, _cfg(enabled='off'))
        self.assertNotIn('fc', period['meta'])


class Refusals(unittest.TestCase):
    """CONTRACT 12: NaN, never a partial correction -- and never an exception."""

    def test_a_missing_shared_input_nans_all_three_and_names_it(self):
        for key in ('h', 'rho_m_mean', 'cp_mean', 'rho_v_mean', 'rho_d_mean', 'ta_mean',
                    'lambda_v_mean', 'chi_v_mean'):
            period, _ = _period(**{key: float('nan')})
            with self.assertLogs('miniflux.wpl', level='WARNING') as caught:
                wpl.correct(period, _cfg())
            meta = period['meta']
            for out in ('fc', 'e', 'le'):
                self.assertTrue(math.isnan(meta[out]), '%s survived a missing %s' % (out, key))
            self.assertFalse(meta['wpl_applied'])
            self.assertIn(key, caught.output[0])

    def test_an_absent_key_counts_as_missing(self):
        period, _ = _period()
        del period['meta']['h']              # flux.assemble omits h rather than aliasing it
        with self.assertLogs('miniflux.wpl', level='WARNING'):
            wpl.correct(period, _cfg())
        self.assertTrue(math.isnan(period['meta']['fc']))
        self.assertNotIn('wt', period['meta'])

    def test_a_missing_rho_c_disables_fc_only(self):
        period, _ = _period(rho_c_mean=float('nan'))
        with self.assertLogs('miniflux.wpl', level='WARNING') as caught:
            wpl.correct(period, _cfg())
        meta = period['meta']
        self.assertTrue(math.isnan(meta['fc']))
        self.assertTrue(math.isfinite(meta['e']))
        self.assertGreater(meta['le'], LE_L0)
        self.assertTrue(meta['wpl_applied'])
        self.assertIn('rho_c_mean', caught.output[0])

    def test_a_zero_divisor_is_declined_rather_than_raised(self):
        # CONTRACT 19: no step module raises. Each of these four is finite and divided by,
        # so a bare `/` would take the period's whole QC block with it -- qc runs after
        # wpl, and a step that raises skips every step after it.
        for key in ('rho_m_mean', 'cp_mean', 'ta_mean', 'rho_d_mean'):
            period, _ = _period(**{key: 0.0})
            with self.assertLogs('miniflux.wpl', level='WARNING') as caught:
                wpl.correct(period, _cfg())
            meta = period['meta']
            for out in ('fc', 'e', 'le'):
                self.assertTrue(math.isnan(meta[out]), '%s survived a zero %s' % (out, key))
            self.assertFalse(meta['wpl_applied'])
            self.assertIn(key, caught.output[0])

    def test_a_nan_l0_flux_propagates_without_raising(self):
        period, _ = _period(fc_l0=float('nan'), fh2o_l0=float('nan'))
        wpl.correct(period, _cfg())
        meta = period['meta']
        self.assertTrue(math.isnan(meta['fc']))
        self.assertTrue(math.isnan(meta['e']))
        self.assertTrue(math.isfinite(meta['wt']))

    def test_correct_returns_the_same_period(self):
        period, _ = _period()
        self.assertIs(wpl.correct(period, _cfg()), period)


if __name__ == '__main__':
    unittest.main()
