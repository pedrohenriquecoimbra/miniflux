"""Tests for `miniflux.spectral`.

Pins ALGORITHMS.md 10A and CONTRACT.md 12A: Horst (1997) Eq. 11, the two stability
branches and where they meet, the direction every argument moves the factor in, and the
publishing rule that keeps a corrected number and an uncorrected one apart.

The expected factors are transcribed from the paper's formula, not called back out of
the module, and the three check values below are literals.
"""

import logging
import math
import unittest
from types import SimpleNamespace

from miniflux import spectral

# A short tower over a crop, which is where a closed-path tube costs the most: the
# cospectrum sits at high natural frequency, so more of the flux is inside the roll-off.
WIND = 2.0            # m s-1
HEIGHT = 1.9          # m, z - d
TAU = 0.13            # s, first-order time constant of the gas channel


def _cfg(enabled=True, co2_tau=TAU, h2o_tau=TAU, height=2.0, displacement=0.1):
    return SimpleNamespace(
        spectral=SimpleNamespace(enabled=enabled, co2_tau_s=co2_tau, h2o_tau_s=h2o_tau),
        site=SimpleNamespace(measurement_height=height, displacement=displacement))


def _period(**meta):
    base = {'wind_speed': WIND, 'z_l': -0.5, 'fc': -2.0e-5, 'e': 8.0e-5, 'le': 200.0}
    base.update(meta)
    return {'meta': base, 't': []}


def _horst(wind, height, z_l, tau):
    """Horst (1997) Eq. 11, written out here so the module is checked against the paper."""
    if z_l > 0:
        nm, alpha = 2.0 - 1.915 / (1.0 + 0.5 * z_l), 1.0
    else:
        nm, alpha = 0.085, 7.0 / 8.0
    return 1.0 + (2.0 * math.pi * nm * tau * wind / height) ** alpha


class TheFormula(unittest.TestCase):
    """Horst Eq. 11, both branches."""

    def setUp(self):
        # The sweeps below deliberately reach implausible factors; their warning is the
        # subject of a test in Degeneracy, not something to print here.
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    def test_the_unstable_branch_matches_the_paper(self):
        meta = spectral.correct(_period(), _cfg())['meta']
        self.assertAlmostEqual(meta['scf_co2'], _horst(WIND, HEIGHT, -0.5, TAU), places=15)
        # The check value: 2 m s-1 over a 1.9 m (z-d) with a 0.13 s channel recovers a
        # flux measured 9.2 % low, which is the order the bundled FR-Gri closed-path run
        # sits at.
        self.assertAlmostEqual(meta['scf_co2'], 1.1013541, places=7)
        self.assertAlmostEqual(1.0 - 1.0 / meta['scf_co2'], 0.092, places=3)

    def test_the_stable_branch_uses_its_own_peak_and_a_unit_exponent(self):
        meta = spectral.correct(_period(z_l=0.5), _cfg())['meta']
        self.assertAlmostEqual(meta['scf_co2'], _horst(WIND, HEIGHT, 0.5, TAU), places=15)
        # nm = 2.0 - 1.915/1.25 = 0.468, five and a half times the neutral peak, and the
        # exponent is 1 rather than 7/8.
        self.assertAlmostEqual(meta['scf_co2'], 1.4023884, places=7)

    def test_the_peak_frequency_is_continuous_at_neutral_but_the_exponent_is_not(self):
        # 2.0 - 1.915 = 0.085 exactly: Horst's stable peak frequency runs into his neutral
        # one at z/L = 0. Only the EXPONENT steps there, so the factor itself does jump,
        # and the two facts have to be checked separately or the jump reads as a bug.
        x = 2.0 * math.pi * 0.085 * TAU * WIND / HEIGHT
        just_stable = spectral.correct(_period(z_l=1e-12), _cfg())['meta']['scf_co2']
        neutral = spectral.correct(_period(z_l=0.0), _cfg())['meta']['scf_co2']
        # Relative, because z/L = 1e-12 leaves 1e-11 of itself in nm.
        self.assertAlmostEqual(just_stable - 1.0, x, delta=1e-10 * x)    # nm -> 0.085
        self.assertAlmostEqual(neutral - 1.0, x ** (7.0 / 8.0), places=12)
        self.assertGreater(neutral, just_stable)

    def test_a_stable_period_loses_more_flux_than_an_unstable_one(self):
        # Its cospectrum peaks at a higher natural frequency, so more of it is attenuated.
        unstable = spectral.correct(_period(z_l=-1.0), _cfg())['meta']['scf_co2']
        stable = spectral.correct(_period(z_l=1.0), _cfg())['meta']['scf_co2']
        self.assertGreater(stable, unstable)

    def test_the_factor_is_never_below_one(self):
        for z_l in (-100.0, -1.0, -0.01, 0.0, 0.01, 1.0, 100.0):
            for wind in (0.0, 0.5, 5.0):
                factor = spectral.correct(
                    _period(wind_speed=wind, z_l=z_l), _cfg())['meta']['scf_co2']
                self.assertGreaterEqual(factor, 1.0, (z_l, wind))

    def test_it_moves_the_way_the_physics_does(self):
        base = spectral.correct(_period(), _cfg())['meta']['scf_co2']
        slower = spectral.correct(_period(), _cfg(co2_tau=2 * TAU))['meta']['scf_co2']
        windier = spectral.correct(_period(wind_speed=2 * WIND), _cfg())['meta']['scf_co2']
        taller = spectral.correct(_period(), _cfg(height=20.0))['meta']['scf_co2']
        self.assertGreater(slower, base)     # a slower sensor loses more
        self.assertGreater(windier, base)    # faster wind puts the flux at higher f
        self.assertLess(taller, base)        # a taller tower carries bigger eddies

    def test_each_gas_gets_its_own_time_constant(self):
        # Water is the slower channel in a tube: it sorbs on the walls.
        meta = spectral.correct(_period(), _cfg(co2_tau=0.1, h2o_tau=0.4))['meta']
        self.assertAlmostEqual(meta['scf_co2'], _horst(WIND, HEIGHT, -0.5, 0.1), places=15)
        self.assertAlmostEqual(meta['scf_h2o'], _horst(WIND, HEIGHT, -0.5, 0.4), places=15)
        self.assertGreater(meta['scf_h2o'], meta['scf_co2'])


class WhatItPublishes(unittest.TestCase):
    """CONTRACT 12A: new columns, never a rewrite."""

    def test_off_writes_nothing_at_all(self):
        meta = spectral.correct(_period(), _cfg(enabled=False))['meta']
        for key in ('scf_co2', 'scf_h2o', 'fc_spec', 'e_spec', 'le_spec'):
            self.assertNotIn(key, meta)

    def test_the_measured_fluxes_are_left_exactly_as_they_were(self):
        period = _period()
        before = dict((key, period['meta'][key]) for key in ('fc', 'e', 'le'))
        meta = spectral.correct(period, _cfg())['meta']
        for key, value in before.items():
            self.assertEqual(meta[key], value)

    def test_each_scaled_flux_is_its_own_factor_times_the_measured_one(self):
        meta = spectral.correct(_period(), _cfg(co2_tau=0.1, h2o_tau=0.4))['meta']
        self.assertAlmostEqual(meta['fc_spec'], meta['fc'] * meta['scf_co2'], places=18)
        self.assertAlmostEqual(meta['e_spec'], meta['e'] * meta['scf_h2o'], places=18)
        self.assertAlmostEqual(meta['le_spec'], meta['le'] * meta['scf_h2o'], places=12)
        # An uptake gets more negative, not less: the flux was under-measured.
        self.assertLess(meta['fc_spec'], meta['fc'])
        self.assertGreater(meta['le_spec'], meta['le'])

    def test_h_is_not_scaled(self):
        # The sonic does not sample through the analyser's tube.
        meta = spectral.correct(_period(h=180.0, h_l0=190.0), _cfg())['meta']
        self.assertEqual(meta['h'], 180.0)
        self.assertNotIn('h_spec', meta)

    def test_a_withheld_flux_gets_no_scaled_name(self):
        # wpl off on a gas that is owed it leaves `fc` absent; a FC_SPEC would then be a
        # correction of a number that was deliberately not reported.
        period = _period()
        del period['meta']['fc']
        meta = spectral.correct(period, _cfg())['meta']
        self.assertNotIn('fc_spec', meta)
        self.assertIn('le_spec', meta)
        self.assertIn('scf_co2', meta)          # the factor itself is still a fact


class Degeneracy(unittest.TestCase):
    """A step writes NaN and logs; it never raises (CONTRACT 19)."""

    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    def test_a_non_finite_wind_or_stability_gives_nan(self):
        for meta in ({'wind_speed': float('nan')}, {'z_l': float('nan')},
                     {'z_l': float('inf')}):
            result = spectral.correct(_period(**meta), _cfg())['meta']
            self.assertTrue(math.isnan(result['scf_co2']), meta)
            self.assertTrue(math.isnan(result['fc_spec']), meta)

    def test_a_measurement_height_below_the_displacement_gives_nan(self):
        meta = spectral.correct(_period(), _cfg(height=1.0, displacement=3.0))['meta']
        self.assertTrue(math.isnan(meta['scf_co2']))

    def test_an_implausible_factor_is_reported_at_warning(self):
        logging.disable(logging.NOTSET)
        with self.assertLogs('miniflux.spectral', level='WARNING') as captured:
            meta = spectral.correct(_period(wind_speed=20.0), _cfg(co2_tau=5.0))['meta']
        self.assertGreater(meta['scf_co2'], spectral.IMPLAUSIBLE_FACTOR)
        self.assertIn('more than half', '\n'.join(captured.output))

    def test_a_zero_wind_costs_nothing_and_warns_about_nothing(self):
        meta = spectral.correct(_period(wind_speed=0.0), _cfg())['meta']
        self.assertEqual(meta['scf_co2'], 1.0)
