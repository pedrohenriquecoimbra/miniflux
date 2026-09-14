"""CONTRACT section 22: the optional xarray adapter.

What this file pins: that a period survives the round trip bit for bit, that every step in
``pipeline.STEPS`` computes the same numbers through ``apply`` as it does natively, that a
step reaches the caller's Dataset only when the caller asks for it, that a contradicted
unit is refused rather than reinterpreted, and that the time coordinate is the arithmetic
CONTRACT section 1.2 describes.

Every test here skips when xarray is absent -- except the last class, which is the reason
the rest may exist at all: it asserts that no module in the core imports xarray or the
adapter, and it runs on an interpreter that has neither.
"""

import os
import random
import subprocess
import sys
import tempfile
import unittest
from array import array
from datetime import datetime, timedelta

from miniflux import config, kernels, pipeline, read

try:
    import numpy as np
    import xarray as xr

    from miniflux import xarray_adapter as xa
except ImportError:                      # the core does not need either of them
    np = xr = xa = None

HAVE_XARRAY = xr is not None
WITHOUT = 'xarray is not installed; the adapter is optional and so is its test'

CONFIG_TEXT = """
[files]
input_glob = {glob}
output_csv = {output}

[runtime]
log_level = ERROR

[site]
site_id = XX-Syn
latitude = 50.0
measurement_height = 3.0
canopy_height = 1.5
pressure_pa = 99000.0

[period]
averaging_minutes = 30
acquisition_frequency = 20.0

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
co2_measure_type = {co2_type}
h2o_measure_type = {h2o_type}
analyser_path = {path}

[lag]
method = covmax
"""

DEFAULTS = {'t_cell': '', 'p_cell': '', 'co2_unit': 'ppm', 'h2o_unit': 'ppt',
            'co2_type': 'mixing_ratio', 'h2o_type': 'mixing_ratio', 'path': 'open'}

DENSITY = {'co2_unit': 'mmol/m3', 'h2o_unit': 'mmol/m3',
           'co2_type': 'molar_density', 'h2o_type': 'molar_density'}

CLOSED = dict(DENSITY, path='closed', t_cell='Tc', p_cell='Pc')


def make_config(tmpdir, **overrides):
    """Write a valid .ini in tmpdir and load it. Returns the Config of CONTRACT section 5.

    Nothing reads a data file: ``config.load`` only parses and validates, so a
    configuration can name columns that no file has.
    """
    settings = dict(DEFAULTS, glob=os.path.join(tmpdir, '*.dat'),
                    output=os.path.join(tmpdir, 'out.csv'))
    settings.update(overrides)
    path = os.path.join(tmpdir, 'miniflux.ini')
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(CONFIG_TEXT.format(**settings))
    return config.load(path)


def synthetic_period(n=1200, freq=20.0, density=False, cell=False):
    """One unstable midday period, built in memory. Returns the period of section 1.

    Warm air rises, CO2 goes down as the air goes up and water goes up with it, so no
    covariance is degenerate. ``density`` puts the gases in mol m-3 instead of mol mol-1 so
    that WPL actually runs; ``cell`` adds a cell state so that ``cell.convert`` does.
    """
    rng = random.Random(20260914)
    start = datetime(2026, 6, 21, 12, 0, 0)
    scale = (41.0, 0.5) if density else (1.0, 1.0)      # mol m-3 vs mol mol-1
    names = ('u', 'v', 'w', 'ts', 'co2', 'h2o', 'p_air', 't_cell', 'p_cell')
    values = dict((name, []) for name in names)
    for _i in range(n):
        gust = rng.gauss(0.0, 0.5)                      # w' [m s-1]
        values['u'].append(2.5 + rng.gauss(0.0, 0.6))
        values['v'].append(0.3 + rng.gauss(0.0, 0.4))
        values['w'].append(0.02 + gust)
        values['ts'].append(293.15 + 0.5 * gust + rng.gauss(0.0, 0.1))
        values['co2'].append(scale[0] * (4.20e-4 - 2.0e-6 * gust + rng.gauss(0.0, 2.0e-6)))
        values['h2o'].append(scale[1] * (1.20e-2 + 4.4e-4 * gust + rng.gauss(0.0, 1.0e-4)))
        values['p_air'].append(99000.0)
        values['t_cell'].append(303.15 + rng.gauss(0.0, 0.05) if cell else kernels.NAN)
        values['p_cell'].append(92000.0 + rng.gauss(0.0, 20.0) if cell else kernels.NAN)
    period = {'meta': {'period_start': start,
                       'period_end': start + timedelta(minutes=30),
                       'n_in': n, 'n_dup': 0, 'freq_hz': freq},
              'ta': kernels.new(n)}       # no measured air temperature in this file
    for name in names:
        period[name] = kernels.from_values(values[name])
    return period


def same_number(a, b):
    """Exact equality, with NaN equal to NaN. Returns bool."""
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (a != a and b != b)
    return type(a) is type(b) and a == b


class _PeriodAssertions(unittest.TestCase):
    """assertPeriodEqual, shared by the round-trip and the acceptance tests."""

    def assertPeriodEqual(self, left, right, where=''):
        for name in read.SERIES:
            self.assertEqual(len(left[name]), len(right[name]), '%s %s' % (where, name))
            for i, (a, b) in enumerate(zip(left[name], right[name])):
                if not same_number(a, b):
                    self.fail('%s %s[%d]: %r != %r' % (where, name, i, a, b))
        self.assertEqual(sorted(left['meta']), sorted(right['meta']), where)
        for key, value in left['meta'].items():
            if not same_number(value, right['meta'][key]):
                self.fail('%s meta[%r]: %r != %r' % (where, key, value, right['meta'][key]))


@unittest.skipUnless(HAVE_XARRAY, WITHOUT)
class RoundTripTest(_PeriodAssertions):
    """period -> Dataset -> period returns the same numbers, bit for bit."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(_remove_tree, self.tmpdir)
        self.cfg = make_config(self.tmpdir)

    def test_round_trip_is_bit_identical(self):
        period = synthetic_period()
        period['ts'][7] = kernels.NAN                    # a gap must survive as a gap
        back = xa.from_dataset(xa.to_dataset(period, self.cfg), self.cfg)
        self.assertPeriodEqual(period, back)
        for name in read.SERIES:
            # Not merely equal: the same 8 bytes per sample, which is what a boundary that
            # only relabels memory owes its caller.
            self.assertEqual(array('d', period[name]).tobytes(),
                             array('d', back[name]).tobytes(), name)

    def test_the_series_come_back_as_something_the_kernels_accept(self):
        back = xa.from_dataset(xa.to_dataset(synthetic_period(), self.cfg), self.cfg)
        for name in read.SERIES:
            kernels.nanmean(back[name])                  # raises on a list or a DataArray
            self.assertIsInstance(back[name], array)

    def test_an_absent_series_comes_back_all_nan_not_missing(self):
        dataset = xa.to_dataset(synthetic_period(), self.cfg).drop_vars(['ta', 't_cell'])
        back = xa.from_dataset(dataset, self.cfg)
        for name in ('ta', 't_cell'):
            self.assertEqual(len(back[name]), 1200)
            self.assertEqual(kernels.count_finite(back[name]), 0)

    def test_meta_is_copied_not_shared(self):
        period = synthetic_period()
        dataset = xa.to_dataset(period, self.cfg)
        back = xa.from_dataset(dataset, self.cfg)
        back['meta']['n_in'] = -1
        self.assertEqual(dataset.attrs['n_in'], 1200)
        self.assertEqual(period['meta']['n_in'], 1200)

    def test_a_dataset_with_no_canonical_series_is_refused(self):
        with self.assertRaises(ValueError):
            xa.from_dataset(xr.Dataset({'wind': ('time', [1.0, 2.0])}), self.cfg)

    def test_a_period_missing_a_series_is_refused_as_a_value_error(self):
        # A hand-built period, or one from before t_cell and p_cell existed. CONTRACT
        # section 22 fixes this file's exceptions at ValueError and TypeError, so the
        # KeyError a bare lookup would raise is not one of the answers available.
        period = synthetic_period(n=10)
        del period['t_cell']
        with self.assertRaises(ValueError) as caught:
            xa.to_dataset(period, self.cfg)
        self.assertIn('t_cell', str(caught.exception))


@unittest.skipUnless(HAVE_XARRAY, WITHOUT)
class StepEquivalenceTest(_PeriodAssertions):
    """The acceptance test: every step of pipeline.STEPS, natively and through apply().

    Run twice over the same data, once with the gases as dry mixing ratios and once as
    molar densities, so that both branches of the flux factor and both branches of WPL are
    exercised; a third pass adds a cell state so ``cell.convert`` has work to do. The two
    chains are compared after *each* step, not only at the end, so a divergence is reported
    where it happens rather than three steps later.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(_remove_tree, self.tmpdir)

    def _chain(self, cfg, period):
        native = period
        dataset = xa.to_dataset(_copy_period(period), cfg)
        for name, function in pipeline.STEPS:
            with self.subTest(step=name):
                native = function(native, cfg)
                dataset = xa.apply(name, dataset, cfg)
                self.assertPeriodEqual(native, xa.from_dataset(dataset, cfg), where=name)
        return native

    def test_mixing_ratio_run(self):
        cfg = make_config(self.tmpdir)
        final = self._chain(cfg, synthetic_period())
        self.assertTrue(abs(final['meta']['h']) > 1.0)          # not a degenerate period
        self.assertFalse(final['meta']['wpl_applied'])

    def test_molar_density_run(self):
        cfg = make_config(self.tmpdir, **DENSITY)
        final = self._chain(cfg, synthetic_period(density=True))
        self.assertTrue(final['meta']['wpl_applied'])           # the correction really ran

    def test_closed_path_cell_run(self):
        cfg = make_config(self.tmpdir, **CLOSED)
        final = self._chain(cfg, synthetic_period(density=True, cell=True))
        self.assertEqual(final['meta']['n_cell_converted'], 1200)

    def test_the_whole_pipeline_as_one_callable(self):
        cfg = make_config(self.tmpdir)
        period = synthetic_period()
        native = pipeline.run_period(_copy_period(period), cfg)
        dataset = xa.apply(pipeline.run_period, xa.to_dataset(period, cfg), cfg)
        self.assertPeriodEqual(native, xa.from_dataset(dataset, cfg))

    def test_an_unknown_step_name_is_refused(self):
        cfg = make_config(self.tmpdir)
        with self.assertRaises(ValueError):
            xa.apply('rotat', xa.to_dataset(synthetic_period(), cfg), cfg)


@unittest.skipUnless(HAVE_XARRAY, WITHOUT)
class CopySemanticsTest(unittest.TestCase):
    """A step mutates in place; the caller's Dataset only sees it on request."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(_remove_tree, self.tmpdir)
        self.cfg = make_config(self.tmpdir)
        period = synthetic_period()
        period['w'][100] = 50.0                    # a spike despike is certain to remove
        self.dataset = xa.to_dataset(period, self.cfg)

    def test_apply_does_not_touch_the_caller_by_default(self):
        out = xa.apply('despike', self.dataset, self.cfg)
        self.assertEqual(float(self.dataset['w'][100]), 50.0)
        self.assertTrue(np.isnan(float(out['w'][100])))
        self.assertNotIn('n_spike_w', self.dataset.attrs)
        self.assertEqual(out.attrs['n_spike_w'], 1)

    def test_copy_false_writes_through_to_the_caller(self):
        out = xa.apply('despike', self.dataset, self.cfg, copy=False)
        self.assertTrue(np.isnan(float(self.dataset['w'][100])))
        self.assertTrue(np.isnan(float(out['w'][100])))
        self.assertEqual(self.dataset.attrs['n_spike_w'], 1)
        # np.shares_memory, not `.base`: both bases are None here, so comparing them is a
        # test that passes whatever the adapter does.
        for name in read.SERIES:
            self.assertTrue(np.shares_memory(out[name].values,
                                             self.dataset[name].values), name)

    def test_copy_false_lands_a_step_that_rebinds_rather_than_writes(self):
        # rotate hands back new u, v, w arrays instead of writing into the ones it was
        # given (lag does the same for each scalar it shifts). Without the write-back, a
        # caller who rotated with copy=False and then read their own Dataset would be
        # holding UNROTATED wind -- wrong rather than absent.
        before = dict((name, self.dataset[name].values.copy()) for name in ('u', 'v', 'w'))
        out = xa.apply('rotate', self.dataset, self.cfg, copy=False)
        self.assertGreater(abs(self.dataset.attrs['theta']), 0.0)   # it really rotated
        for name in ('u', 'v', 'w'):
            self.assertFalse(np.array_equal(self.dataset[name].values, before[name]), name)
            self.assertTrue(np.array_equal(self.dataset[name].values, out[name].values,
                                           equal_nan=True), name)
            self.assertTrue(np.shares_memory(out[name].values,
                                             self.dataset[name].values), name)

    def test_copy_false_lands_the_whole_pipeline_in_the_caller(self):
        out = xa.apply(pipeline.run_period, self.dataset, self.cfg, copy=False)
        for name in read.SERIES:
            self.assertTrue(np.array_equal(self.dataset[name].values, out[name].values,
                                           equal_nan=True), name)
        self.assertIn('fc', self.dataset.attrs)

    def test_a_step_that_changed_a_length_is_refused_under_copy_false(self):
        # No step does this (shift_truncate NaN-fills rather than shortening), but a
        # shorter result cannot be landed in the caller's buffer, and dropping it silently
        # is the one thing copy=False must not do.
        def shorten(period, _cfg):
            for name in read.SERIES:
                period[name] = kernels.new(len(period[name]) - 1)
            return period
        with self.assertRaises(ValueError) as caught:
            xa.apply(shorten, self.dataset, self.cfg, copy=False)
        self.assertIn('copy=True', str(caught.exception))

    def test_a_step_that_dropped_a_series_is_refused_under_copy_false(self):
        # The write-back runs before to_dataset, so it is the one that has to answer for a
        # result that is no longer a period -- with a ValueError, not a KeyError.
        def drop(period, _cfg):
            del period['t_cell']
            return period
        with self.assertRaises(ValueError) as caught:
            xa.apply(drop, self.dataset, self.cfg, copy=False)
        self.assertIn('t_cell', str(caught.exception))

    def test_a_non_numeric_variable_is_refused_rather_than_cast(self):
        # Every one of these is cast to float64 by numpy without complaining, and none of
        # the numbers that come out is the measurement: a datetime64 becomes nanoseconds
        # since the epoch, a bool becomes 1.0, a complex loses its imaginary part, a
        # string of digits is parsed. Refusing is the only honest answer.
        n = 1200
        for values in (np.arange(n).astype('datetime64[ns]'),
                       np.ones(n, dtype=bool),
                       np.ones(n, dtype=np.complex128),
                       np.array(['293.15'] * n),
                       np.array([293.15] * n, dtype=object)):
            dataset = self.dataset.assign(ts=('time', values))
            with self.assertRaises(ValueError, msg=str(values.dtype)) as caught:
                xa.from_dataset(dataset, self.cfg)
            self.assertIn('ts', str(caught.exception))

    def test_from_dataset_copies_by_default(self):
        period = xa.from_dataset(self.dataset, self.cfg)
        period['w'][0] = 123.0
        self.assertNotEqual(float(self.dataset['w'][0]), 123.0)

    def test_from_dataset_aliases_on_request(self):
        period = xa.from_dataset(self.dataset, self.cfg, copy=False)
        period['w'][0] = 123.0
        self.assertEqual(float(self.dataset['w'][0]), 123.0)

    def test_float32_cannot_be_aliased_and_says_why(self):
        # np.frombuffer would NOT raise on this: it would reinterpret four float32 as two
        # float64 and return numbers that are wrong rather than absent. The refusal is the
        # adapter's, not numpy's.
        dataset = self.dataset.assign(w=self.dataset['w'].astype('float32'))
        with self.assertRaises(ValueError) as caught:
            xa.from_dataset(dataset, self.cfg, copy=False)
        self.assertIn('float64', str(caught.exception))
        period = xa.from_dataset(dataset, self.cfg)            # copying casts instead
        self.assertAlmostEqual(period['w'][3], float(dataset['w'][3]), places=6)

    def test_a_strided_or_read_only_series_cannot_be_aliased(self):
        strided = self.dataset.isel(time=slice(None, None, 2))
        with self.assertRaises(ValueError):
            xa.from_dataset(strided, self.cfg, copy=False)
        frozen = self.dataset.copy()
        for name in read.SERIES:
            frozen[name].values.flags['WRITEABLE'] = False
        with self.assertRaises(ValueError):
            xa.from_dataset(frozen, self.cfg, copy=False)

    def test_extra_variables_and_the_callers_dimension_survive_apply(self):
        dataset = self.dataset.rename({'time': 'sample'})
        dataset['diag'] = ('sample', np.zeros(1200))
        out = xa.apply('despike', dataset, self.cfg)
        self.assertEqual(out['w'].dims, ('sample',))
        self.assertIn('diag', out.data_vars)


@unittest.skipUnless(HAVE_XARRAY, WITHOUT)
class UnitTest(unittest.TestCase):
    """CONTRACT section 2 travels with the numbers, and a contradiction is refused."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(_remove_tree, self.tmpdir)
        self.cfg = make_config(self.tmpdir)
        self.dataset = xa.to_dataset(synthetic_period(), self.cfg)

    def test_every_variable_carries_its_canonical_unit(self):
        expected = {'u': 'm s-1', 'v': 'm s-1', 'w': 'm s-1', 'ts': 'K', 'ta': 'K',
                    't_cell': 'K', 'p_air': 'Pa', 'p_cell': 'Pa',
                    'co2': 'mol mol-1', 'h2o': 'mol mol-1'}
        for name, unit in expected.items():
            self.assertEqual(self.dataset[name].attrs['units'], unit, name)

    def test_a_gas_unit_follows_the_measure_type(self):
        cfg = make_config(self.tmpdir, **DENSITY)
        dataset = xa.to_dataset(synthetic_period(density=True), cfg)
        self.assertEqual(dataset['co2'].attrs['units'], 'mol m-3')

    def test_a_cell_gas_is_labelled_by_where_in_the_pipeline_it_is(self):
        # The effective measure type of a converted gas is mixing_ratio (section 5.1), but
        # that is only true once cell.convert has run. Before it, the numbers are still the
        # densities the file reported, and the label has to say so.
        cfg = make_config(self.tmpdir, **CLOSED)
        period = synthetic_period(density=True, cell=True)
        self.assertEqual(xa.to_dataset(period, cfg)['co2'].attrs['units'], 'mol m-3')
        converted = pipeline.STEPS[0][1](period, cfg)
        self.assertEqual(xa.to_dataset(converted, cfg)['co2'].attrs['units'], 'mol mol-1')

    def test_a_contradicted_unit_is_refused(self):
        for name, wrong in (('ts', 'degC'), ('co2', 'ppm'), ('p_air', 'hPa'),
                            ('u', 'km/h')):
            dataset = self.dataset.copy()
            dataset[name].attrs['units'] = wrong
            with self.assertRaises(ValueError, msg=name) as caught:
                xa.from_dataset(dataset, self.cfg)
            self.assertIn(name, str(caught.exception))

    def test_a_synonym_is_accepted_and_an_absent_unit_is_not_a_contradiction(self):
        dataset = self.dataset.copy()
        dataset['u'].attrs['units'] = 'm/s'
        del dataset['ts'].attrs['units']
        xa.from_dataset(dataset, self.cfg)


@unittest.skipUnless(HAVE_XARRAY, WITHOUT)
class TimeCoordinateTest(unittest.TestCase):
    """CONTRACT section 1.2: the coordinate is arithmetic, and it is honest about it."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(_remove_tree, self.tmpdir)
        self.cfg = make_config(self.tmpdir)

    def test_start_and_spacing_against_a_known_period(self):
        dataset = xa.to_dataset(synthetic_period(n=10, freq=20.0), self.cfg)
        stamps = dataset['time'].values
        self.assertEqual(stamps[0], np.datetime64('2026-06-21T12:00:00.000000000'))
        self.assertEqual(stamps[9], np.datetime64('2026-06-21T12:00:00.450000000'))
        self.assertTrue((np.diff(stamps) == np.timedelta64(50, 'ms')).all())

    def test_a_period_with_no_start_gets_a_sample_index_not_an_invented_clock(self):
        period = synthetic_period(n=10)
        del period['meta']['period_start']
        stamps = xa.to_dataset(period, self.cfg)['time'].values
        self.assertEqual(stamps.dtype, np.dtype('int64'))
        self.assertEqual(list(stamps), list(range(10)))

    def test_apply_gives_back_a_dataset_that_had_no_coordinate_without_one(self):
        # to_dataset builds a coordinate because a bare period has nowhere else to put its
        # clock. apply must not: handing back an index the caller never had is something a
        # later merge or concat would silently align on.
        period = synthetic_period(n=100)
        bare = xr.Dataset(dict((name, ('time', np.array(period[name], dtype='float64')))
                               for name in read.SERIES))
        self.assertEqual(list(bare.coords), [])
        self.assertEqual(list(xa.apply('despike', bare, self.cfg).coords), [])

    def test_apply_gives_back_the_callers_own_coordinate_untouched(self):
        period = synthetic_period(n=100)
        own = xr.Dataset(dict((name, ('time', np.array(period[name], dtype='float64')))
                              for name in read.SERIES),
                         coords={'time': np.arange(1000, 1100)})
        out = xa.apply('despike', own, self.cfg)
        self.assertTrue((out['time'].values == np.arange(1000, 1100)).all())

    def test_an_awkward_frequency_never_accumulates_error(self):
        period = synthetic_period(n=1200, freq=3.0)
        period['meta']['freq_hz'] = 3.0
        stamps = xa.to_dataset(period, self.cfg)['time'].values
        elapsed = (stamps[-1] - stamps[0]) / np.timedelta64(1, 'ns')
        self.assertLess(abs(elapsed - 1199 * 1e9 / 3.0), 1.0)   # under one nanosecond


class ImportIsolationTest(unittest.TestCase):
    """The adapter may import the core. The core may not know the adapter exists.

    This class is deliberately NOT skipped when xarray is absent: that is the case it is
    about. It reads the sources rather than the import graph so that it also catches an
    import written inside a function, which a probe of ``sys.modules`` would not.
    """

    def _core_sources(self):
        folder = os.path.dirname(os.path.abspath(read.__file__))
        for name in sorted(os.listdir(folder)):
            if name.endswith('.py') and name != 'xarray_adapter.py':
                with open(os.path.join(folder, name), encoding='utf-8') as handle:
                    yield name, handle.read()

    def test_no_core_module_mentions_xarray_or_the_adapter(self):
        for name, source in self._core_sources():
            self.assertNotIn('xarray', source, name)
            self.assertNotIn('xarray_adapter', source, name)

    def test_the_package_imports_and_runs_with_xarray_unimportable(self):
        # A subprocess, because this one cannot be proved in a process that has already
        # imported the adapter: sys.modules[name] = None is what makes `import name` fail.
        code = ('import sys\n'
                "sys.modules['xarray'] = None\n"
                "sys.modules['pandas'] = None\n"
                'import miniflux, miniflux.pipeline, miniflux.cli, miniflux.write\n'
                "assert 'miniflux.xarray_adapter' not in sys.modules\n"
                "assert [name for name, _f in miniflux.pipeline.STEPS]\n"
                "print('ok')\n")
        root = os.path.dirname(os.path.dirname(os.path.abspath(read.__file__)))
        out = subprocess.check_output([sys.executable, '-c', code], cwd=root)
        self.assertEqual(out.strip(), b'ok')

    def test_the_adapter_is_not_named_xr(self):
        # `from miniflux import xr` next to `import xarray as xr` is one line away from a
        # silent shadow; the file is named after what it adapts instead.
        folder = os.path.dirname(os.path.abspath(read.__file__))
        self.assertFalse(os.path.exists(os.path.join(folder, 'xr.py')))


def _copy_period(period):
    """An independent copy of a period, so two chains can run on the same numbers."""
    out = {'meta': dict(period['meta'])}
    for name in read.SERIES:
        out[name] = kernels.copy(period[name])
    return out


def _remove_tree(path):
    """Delete a temporary directory and everything in it."""
    for name in os.listdir(path):
        os.remove(os.path.join(path, name))
    os.rmdir(path)


if __name__ == '__main__':
    unittest.main()
