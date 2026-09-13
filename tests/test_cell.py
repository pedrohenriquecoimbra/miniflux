"""Tests for `miniflux.cell`.

Pins ALGORITHMS.md 2A and CONTRACT.md 7A: the Ibrom et al. (2007) conversion of a
closed-path cell molar density into a dry mixing ratio, the refusal to invent a cell
state, and the two properties that make the whole thing worth having --

* it is **exact**: a dry mixing ratio expressed as a cell density and converted back
  returns the same number, so the two shapes an LI-7200 writes are one measurement;
* it is done **per sample**: run on the block means instead, it removes nothing, which
  is the whole reason a closed-path density is not simply corrected afterwards.

The synthesis here goes the other way round from the code under test (dry ratio ->
cell density), so a shared algebra error would have to be made twice, in opposite
directions, to pass.
"""

import logging
import math
import os
import random
import shutil
import tempfile
import unittest
from array import array
from datetime import datetime, timedelta
from types import SimpleNamespace

from miniflux import cell, config, kernels, pipeline
from miniflux.constants import R

# One cell, warmed and pumped down as an LI-7200's is: 25 degC and 99.3 kPa while the air
# outside is near 20 degC and 100.7 kPa.
T_CELL = 298.1948     # K
P_CELL = 99320.8      # Pa
V_CELL = R * T_CELL / P_CELL           # m3 mol-1, the cell's molar volume

CO2_DRY = 408.238e-6  # mol mol-1 of dry air
H2O_DRY = 9.9831e-3   # mol mol-1 of dry air


def _densities(r_co2=CO2_DRY, r_h2o=H2O_DRY, t_cell=T_CELL, p_cell=P_CELL):
    """The cell molar densities [mol m-3] a dry pair would be reported as.

    The inverse of what `cell.convert` does, written out independently: the water is a
    fraction ``chi = r/(1+r)`` of the moist air in the cell, the gas is the same dry
    ratio diluted by that water, and both divide by the cell's molar volume.
    """
    v_cell = R * t_cell / p_cell
    chi_h2o = r_h2o / (1.0 + r_h2o)                 # mol mol-1 of MOIST cell air
    chi_co2 = r_co2 * (1.0 - chi_h2o)               # ditto
    return chi_co2 / v_cell, chi_h2o / v_cell


def _period(n_co2, n_h2o, t_cell=T_CELL, p_cell=P_CELL, extra=None):
    """A period holding the ten series of CONTRACT 1, gases as cell densities.

    Scalars are broadcast to four samples; a list is taken as the series itself, which is
    how the per-sample tests plant a fluctuating cell state.
    """
    def series(value):
        return array('d', value if isinstance(value, (list, tuple)) else [value] * 4)

    period = {'meta': {}, 't': [],
              'co2': series(n_co2), 'h2o': series(n_h2o),
              't_cell': series(t_cell), 'p_cell': series(p_cell)}
    for name in ('u', 'v', 'w', 'ts', 'ta', 'p_air'):
        period[name] = series(0.0)
    if extra:
        period.update(extra)
    return period


def _cfg(co2='molar_density', h2o='molar_density', path='closed'):
    """The two attributes `cell.convert` reads, resolved as `config._derive_gases` does."""
    reported = {'co2': co2, 'h2o': h2o}
    convert_cell = tuple(gas for gas in ('co2', 'h2o')
                         if path == 'closed' and reported[gas] == 'molar_density')
    return SimpleNamespace(gases=SimpleNamespace(
        reported=reported, convert_cell=convert_cell,
        measure_type=dict((gas, 'mixing_ratio' if gas in convert_cell else reported[gas])
                          for gas in ('co2', 'h2o'))))


class RoundTrip(unittest.TestCase):
    """The two shapes an LI-7200 writes are one measurement (ALGORITHMS 2A)."""

    def test_a_dry_ratio_survives_the_round_trip_through_a_cell_density(self):
        n_co2, n_h2o = _densities()
        period = cell.convert(_period(n_co2, n_h2o), _cfg())
        for i in range(4):
            self.assertAlmostEqual(period['co2'][i], CO2_DRY, delta=1e-16)
            self.assertAlmostEqual(period['h2o'][i], H2O_DRY, delta=1e-16)

    def test_the_dilution_divisor_is_not_optional(self):
        # Dropping the /(1 - chi) leaves a MOIST mole fraction wearing a dry name. On this
        # parcel that is 1 %, which is the size of a whole day's CO2 signal.
        n_co2, _n_h2o = _densities()
        moist = n_co2 * V_CELL
        self.assertAlmostEqual(moist / CO2_DRY, 1.0 - H2O_DRY / (1.0 + H2O_DRY), places=12)
        self.assertGreater(abs(moist / CO2_DRY - 1.0), 0.0098)

    def test_water_given_as_a_dry_ratio_converts_the_other_gas_correctly(self):
        # An LI-7200 may be read as CO2_CONC + H2O_DRY: the dilution factor is then built
        # from a dry mixing ratio, and must land on the same answer.
        n_co2, _n_h2o = _densities()
        period = _period(n_co2, H2O_DRY)
        period = cell.convert(period, _cfg(h2o='mixing_ratio'))
        self.assertAlmostEqual(period['co2'][0], CO2_DRY, delta=1e-16)
        self.assertEqual(period['h2o'][0], H2O_DRY)          # untouched, not converted twice

    def test_the_published_cell_state_is_the_cell_and_not_the_ambient_air(self):
        n_co2, n_h2o = _densities()
        meta = cell.convert(_period(n_co2, n_h2o), _cfg())['meta']
        self.assertAlmostEqual(meta['t_cell_mean'], T_CELL, places=9)
        self.assertAlmostEqual(meta['p_cell_mean'], P_CELL, places=6)
        self.assertAlmostEqual(meta['v_cell_mean'], V_CELL, places=12)
        self.assertAlmostEqual(meta['chi_h2o_cell_mean'], H2O_DRY / (1.0 + H2O_DRY),
                               places=12)
        self.assertEqual(meta['n_cell_converted'], 4)


class PerSampleNotOnTheMeans(unittest.TestCase):
    """ALGORITHMS 2A: the fluctuations of the cell state are the whole point."""

    def test_a_fluctuating_cell_temperature_changes_the_converted_series(self):
        # Two periods with the same MEAN cell temperature: one steady, one swinging by
        # +-1 K. A conversion applied to the block mean cannot tell them apart; this one
        # must, because a swing in T_cell is a swing in the reported density that is not
        # a swing in the gas.
        n_co2, n_h2o = _densities()
        steady = cell.convert(_period(n_co2, n_h2o), _cfg())
        swing = cell.convert(
            _period(n_co2, n_h2o, t_cell=[T_CELL - 1.0, T_CELL + 1.0,
                                          T_CELL - 1.0, T_CELL + 1.0]), _cfg())
        self.assertAlmostEqual(sum(swing['t_cell']) / 4.0, T_CELL, places=9)
        self.assertNotEqual(swing['co2'][0], steady['co2'][0])
        self.assertGreater(swing['co2'][1], swing['co2'][0])
        # +-1 K in 298 K is +-0.34 % on the molar volume. The sample-to-sample ratio is
        # NOT simply T+/T-: the same molar volume also moves the cell water fraction, so
        # the dilution divisor moves with it. Writing that coupling out is the point --
        # it is exactly what a conversion applied to the block means would lose.
        warm = R * (T_CELL + 1.0) / P_CELL
        cool = R * (T_CELL - 1.0) / P_CELL
        expected = (warm * (1.0 - n_h2o * cool)) / (cool * (1.0 - n_h2o * warm))
        self.assertAlmostEqual(swing['co2'][1] / swing['co2'][0], expected, places=12)
        self.assertGreater(expected, (T_CELL + 1.0) / (T_CELL - 1.0))

    def test_a_cell_pressure_swing_moves_the_ratio_the_other_way(self):
        n_co2, n_h2o = _densities()
        period = cell.convert(
            _period(n_co2, n_h2o, p_cell=[P_CELL, P_CELL * 1.01, P_CELL, P_CELL]), _cfg())
        self.assertLess(period['co2'][1], period['co2'][0])


class Degeneracy(unittest.TestCase):
    """A step writes NaN and logs; it never raises and it never guesses (CONTRACT 19)."""

    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    def test_a_missing_cell_temperature_gives_nan_for_that_sample_only(self):
        n_co2, n_h2o = _densities()
        period = cell.convert(
            _period(n_co2, n_h2o, t_cell=[T_CELL, float('nan'), T_CELL, T_CELL]), _cfg())
        self.assertTrue(math.isnan(period['co2'][1]))
        self.assertTrue(math.isnan(period['h2o'][1]))
        self.assertAlmostEqual(period['co2'][0], CO2_DRY, delta=1e-16)
        self.assertEqual(period['meta']['n_cell_converted'], 3)

    def test_a_zero_cell_pressure_is_nan_and_not_an_exception(self):
        # Python raises where IEEE-754 returns an infinity, and an infinity here would
        # reach a covariance and take a whole flux with it.
        n_co2, n_h2o = _densities()
        period = cell.convert(_period(n_co2, n_h2o, p_cell=0.0), _cfg())
        self.assertTrue(all(math.isnan(x) for x in period['co2']))
        self.assertEqual(period['meta']['n_cell_converted'], 0)

    def test_a_cell_water_fraction_at_or_above_one_is_nan(self):
        # 1 - chi <= 0 has no dry air in it to be a ratio of.
        n_h2o = 1.0 / V_CELL                        # chi = 1 exactly
        period = cell.convert(_period(1.0e-5, n_h2o), _cfg())
        self.assertTrue(all(math.isnan(x) for x in period['co2']))

    def test_an_all_nan_period_is_survived(self):
        nan = float('nan')
        period = cell.convert(_period(nan, nan, t_cell=nan, p_cell=nan), _cfg())
        self.assertEqual(period['meta']['n_cell_converted'], 0)
        self.assertTrue(math.isnan(period['meta']['v_cell_mean']))

    def test_a_degenerate_sample_is_reported_at_warning(self):
        logging.disable(logging.NOTSET)
        n_co2, n_h2o = _densities()
        with self.assertLogs('miniflux.cell', level='WARNING') as captured:
            cell.convert(_period(n_co2, n_h2o, t_cell=[T_CELL, 0.0, T_CELL, T_CELL]),
                         _cfg())
        self.assertIn('1 of 4', '\n'.join(captured.output))


class NothingOwed(unittest.TestCase):
    """Every open-path run, and every closed-path run already in dry mixing ratios."""

    def test_an_open_path_density_is_left_alone(self):
        # An open-path molar density IS an ambient density: it is owed WPL, not this.
        n_co2, n_h2o = _densities()
        period = cell.convert(_period(n_co2, n_h2o), _cfg(path='open'))
        self.assertEqual(period['co2'][0], n_co2)
        self.assertNotIn('n_cell_converted', period['meta'])
        self.assertNotIn('v_cell_mean', period['meta'])

    def test_a_closed_path_dry_mixing_ratio_needs_no_cell_state(self):
        cfg = _cfg(co2='mixing_ratio', h2o='mixing_ratio')
        self.assertEqual(cfg.gases.convert_cell, ())
        period = cell.convert(_period(CO2_DRY, H2O_DRY, t_cell=float('nan'),
                                      p_cell=float('nan')), cfg)
        self.assertEqual(period['co2'][0], CO2_DRY)          # exact, and no cell was read
        self.assertNotIn('n_cell_converted', period['meta'])

    def test_the_effective_measure_type_retires_the_density_correction(self):
        # The one line that makes flux.py and wpl.py need no change at all.
        self.assertEqual(_cfg().gases.measure_type,
                         {'co2': 'mixing_ratio', 'h2o': 'mixing_ratio'})
        self.assertEqual(_cfg(path='open').gases.measure_type,
                         {'co2': 'molar_density', 'h2o': 'molar_density'})


# --------------------------------------------------------------------------------------
# The whole pipeline, the same half hour declared both ways.

INI = """
[files]
input_glob = {tmp}/*.dat
output_csv = {tmp}/out.csv

[runtime]
log_level = ERROR

[site]
site_id = XX-Cel
latitude = 50.0
measurement_height = 3.0
canopy_height = 1.5
pressure_pa = 100000.0

[period]
averaging_minutes = 30
acquisition_frequency = 10.0

[variables]
u = Ux
v = Uy
w = Uz
ts = Ts
co2 = CO2
h2o = H2O
ta =
p_air =
t_cell = {t_cell}
p_cell = {p_cell}

[units]
ts = K
co2 = {co2_unit}
h2o = {h2o_unit}

[gases]
co2_measure_type = {measure}
h2o_measure_type = {measure}
analyser_path = closed
"""


def _synthetic(n=1800, freq=10.0):
    """One midday half hour, in dry mixing ratios, with a live analyser cell.

    Returns ``(series dict, t list)``. The cell state fluctuates -- that is the point:
    with a steady cell the two declarations would agree for the trivial reason that the
    conversion is a constant.
    """
    rng = random.Random(20260913)
    values = dict((name, []) for name in
                  ('u', 'v', 'w', 'ts', 'co2', 'h2o', 'p_air', 't_cell', 'p_cell'))
    for i in range(n):
        gust = rng.gauss(0.0, 0.5)                             # w' [m s-1]
        values['u'].append(2.5 + rng.gauss(0.0, 0.6))
        values['v'].append(0.3 + rng.gauss(0.0, 0.4))
        values['w'].append(0.02 + gust)
        values['ts'].append(293.15 + 0.5 * gust + rng.gauss(0.0, 0.1))
        values['co2'].append(CO2_DRY - 5.0e-6 * gust + rng.gauss(0.0, 2.0e-6))
        values['h2o'].append(H2O_DRY + 4.4e-4 * gust + rng.gauss(0.0, 1.0e-4))
        values['p_air'].append(100000.0)
        # The pump and the instrument body: warm, below ambient, and both drifting.
        values['t_cell'].append(T_CELL + 0.4 * math.sin(i / 97.0) + rng.gauss(0.0, 0.05))
        values['p_cell'].append(P_CELL + 30.0 * math.cos(i / 61.0) + rng.gauss(0.0, 5.0))
    start = datetime(2026, 6, 21, 12, 0, 0)
    return values, [start + timedelta(seconds=i / freq) for i in range(n)]


class TwoWayAgreement(unittest.TestCase):
    """One measurement, two declarations, one flux (ALGORITHMS 2A).

    This is the test that the conversion is right, and it is the synthetic twin of the
    bundled FR-Gri pair: `examples/fr_gri_closedpath_dry.ini` and
    `..._cell.ini` read the same LI-7200 half hours from CO2_DRY/H2O_DRY and from
    CO2_CONC/H2O_CONC. Here the two shapes are built from one another exactly, so the
    agreement has no instrument rounding in it and the tolerance can be tight.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.values, self.t = _synthetic()

    def _cfg(self, measure):
        cell_columns = ('T_CELL', 'PRESS_CELL') if measure == 'molar_density' else ('', '')
        path = os.path.join(self.tmp, 'miniflux_%s.ini' % measure)
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(INI.format(
                tmp=self.tmp.replace('\\', '/'), measure=measure,
                t_cell=cell_columns[0], p_cell=cell_columns[1],
                co2_unit='mol/m3' if measure == 'molar_density' else 'ppm',
                h2o_unit='mol/m3' if measure == 'molar_density' else 'ppt'))
        return config.load(path)

    def _period(self, measure):
        """The period as read.py would hand it over under one of the two declarations."""
        values = dict((name, list(series)) for name, series in self.values.items())
        if measure == 'molar_density':
            co2, h2o = [], []
            for i in range(len(self.t)):
                n_co2, n_h2o = _densities(values['co2'][i], values['h2o'][i],
                                          values['t_cell'][i], values['p_cell'][i])
                co2.append(n_co2)
                h2o.append(n_h2o)
            values['co2'], values['h2o'] = co2, h2o
        period = {'meta': {'period_start': self.t[0],
                           'period_end': self.t[0] + timedelta(minutes=30),
                           'n_in': len(self.t), 'n_dup': 0, 'freq_hz': 10.0},
                  't': list(self.t), 'ta': kernels.new(len(self.t))}
        for name, series in values.items():
            period[name] = kernels.from_values(series)
        return period

    def test_the_two_declarations_give_the_same_fluxes(self):
        dry = pipeline.run_period(self._period('mixing_ratio'), self._cfg('mixing_ratio'))
        cel = pipeline.run_period(self._period('molar_density'),
                                  self._cfg('molar_density'))
        for key in ('fc', 'le', 'e', 'h', 'ustar', 'cov_w_co2', 'cov_w_h2o',
                    'co2_dry_ppm', 'h2o_dry_ppt', 'rho_v_mean'):
            a, b = dry['meta'][key], cel['meta'][key]
            self.assertTrue(math.isfinite(a) and math.isfinite(b), key)
            # Not bit-equal: the density route divides by the same molar volume the
            # synthesis multiplied by, and a double does not round twice to the same
            # place. 1e-9 relative is eleven orders below the answer.
            self.assertAlmostEqual(b / a, 1.0, delta=1e-9,
                                   msg='%s: %r against %r' % (key, a, b))

    def test_neither_run_is_owed_a_density_correction(self):
        for measure in ('mixing_ratio', 'molar_density'):
            meta = pipeline.run_period(self._period(measure), self._cfg(measure))['meta']
            self.assertIs(meta['wpl_applied'], False, measure)
            self.assertEqual(meta['fc'], meta['fc_l0'], measure)

    def test_only_the_density_route_reports_a_conversion(self):
        dry = pipeline.run_period(self._period('mixing_ratio'), self._cfg('mixing_ratio'))
        cel = pipeline.run_period(self._period('molar_density'),
                                  self._cfg('molar_density'))
        self.assertNotIn('n_cell_converted', dry['meta'])
        self.assertEqual(cel['meta']['n_cell_converted'], len(self.t))
        self.assertAlmostEqual(cel['meta']['t_cell_mean'], T_CELL, delta=0.05)
        self.assertAlmostEqual(cel['meta']['p_cell_mean'], P_CELL, delta=10.0)

    def test_reading_the_cell_densities_as_ambient_ones_is_a_different_number(self):
        # The failure mode the analyser_path key exists to prevent: the same two columns
        # declared open-path get an ambient WPL instead of a conversion, and produce a
        # plausible FC for air that was never at the tower.
        period = self._period('molar_density')
        cfg = self._cfg('molar_density')
        cfg.gases.analyser_path = 'open'
        cfg.gases.convert_cell = ()
        cfg.gases.measure_type = {'co2': 'molar_density', 'h2o': 'molar_density'}
        wrong = pipeline.run_period(period, cfg)['meta']
        right = pipeline.run_period(self._period('mixing_ratio'),
                                    self._cfg('mixing_ratio'))['meta']
        self.assertIs(wrong['wpl_applied'], True)
        self.assertGreater(abs(wrong['fc'] / right['fc'] - 1.0), 0.05)
