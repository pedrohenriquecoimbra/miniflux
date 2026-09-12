"""The seams between modules: a real file in, a real CSV row out.

Every other test module pins one module against its own spec. This one pins the joints,
which is where a package written in parallel breaks: a meta key written under one name and
read under another, a config key one module spells differently, a step whose branch nothing
ever ran. So it plants three known quantities in a file on disk -- a covariance, a lag and a
spike count -- and asserts they come back out of ``miniflux.csv``; then it runs the whole
pipeline once for every branch of every method key, because a branch that no test enters is
a branch nobody knows compiles.
"""

import contextlib
import csv
import logging
import math
import os
import random
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta

from miniflux.errors import ConfigError
from miniflux import cli, config, kernels, pipeline, read, write

HEADER = 'TIMESTAMP,Ux,Uy,Uz,Ts,CO2,H2O,Press'

# A one-minute period at 20 Hz. Small enough to run the whole matrix in a second, long
# enough that the lag window (+-1 s = +-20 samples) is a small fraction of it.
FREQ_HZ = 20.0
N = 1200
START = datetime(2025, 9, 8, 19, 30, 0, 50000)
LAG_SAMPLES = 5                     # planted CO2 lag: 0.25 s at 20 Hz
SPIKES = {'u': (100, 200), 'co2': (300,)}
TS_GAIN = 1.20                      # [K / (m s-1)]: the planted w'T'
CO2_GAIN = -12.0                    # [ppm / (m s-1)], delayed by LAG_SAMPLES

QUIET = logging.NullHandler()


def setUpModule():
    # The pipeline warns on purpose (a period with no ITC, a short period); a handler of any
    # kind keeps logging's last-resort writer off stderr while leaving assertLogs working.
    logging.getLogger('miniflux').addHandler(QUIET)


def tearDownModule():
    logging.getLogger('miniflux').removeHandler(QUIET)


def planted(seed=7):
    """The synthetic period as seven columns of float. Returns dict name -> list.

    Built the way ``examples/make_sample.py`` builds its 30 minutes: an AR(1) vertical
    wind, a temperature proportional to it, a CO2 delayed by ``LAG_SAMPLES`` so that
    ``cov(w_i, co2_{i+L})`` peaks at exactly that lag, and single-sample spikes.
    """
    rng = random.Random(seed)
    source = [0.0]
    for _i in range(N + LAG_SAMPLES - 1):
        source.append(0.7 * source[-1] + 0.25 * rng.gauss(0.0, 1.0))
    w = [0.05 + value for value in source[LAG_SAMPLES:]]
    columns = {
        'u': [2.0 - 0.8 * source[LAG_SAMPLES + i] + 0.4 * rng.gauss(0.0, 1.0)
              for i in range(N)],
        'v': [-0.2 + 0.3 * rng.gauss(0.0, 1.0) for _i in range(N)],
        'w': w,
        'ts': [293.15 + TS_GAIN * source[LAG_SAMPLES + i] + 0.1 * rng.gauss(0.0, 1.0)
               for i in range(N)],
        'co2': [420.0 + CO2_GAIN * source[i] + 1.0 * rng.gauss(0.0, 1.0) for i in range(N)],
        'h2o': [12.0 + 0.5 * source[LAG_SAMPLES + i] + 0.05 * rng.gauss(0.0, 1.0)
                for i in range(N)],
        'p_air': [99000.0] * N,
    }
    for name, indices in SPIKES.items():
        for index in indices:
            columns[name][index] += 40.0
    return columns


class IntegrationTestCase(unittest.TestCase):
    """A temp directory holding one raw file, one .ini and whatever the run writes."""

    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix='miniflux_integration_')
        self.addCleanup(shutil.rmtree, self.directory, True)
        self.addCleanup(kernels.set_numpy, kernels.NUMPY_AVAILABLE)
        self.columns = planted()

    def path(self, name):
        return os.path.join(self.directory, name)

    def write_raw(self, name='data.dat', numeric_stamps=False, columns=None,
                  count=N, header=HEADER):
        """Write the planted columns as a raw file. Returns its path (str)."""
        columns = columns or self.columns
        keys = ('u', 'v', 'w', 'ts', 'co2', 'h2o', 'p_air')
        pattern = '%Y%m%d%H%M%S.%f' if numeric_stamps else '%Y-%m-%d %H:%M:%S.%f'
        path = self.path(name)
        with open(path, 'w') as handle:
            handle.write(header + '\n')
            for i in range(count):
                stamp = START + timedelta(seconds=i / FREQ_HZ)
                # Wrap: a caller asking for more samples than were planted wants a
                # longer stream, not different statistics.
                cells = ['%.6f' % columns[key][i % N] for key in keys]
                handle.write(stamp.strftime(pattern) + ',' + ','.join(cells) + '\n')
        return path

    def load(self, sections=None):
        """Write an .ini over the shipped defaults and load it. Returns the Config."""
        merged = {'files': {'input_glob': self.path('*.dat'),
                            'output_csv': self.path('out.csv')},
                  'period': {'averaging_minutes': '1'},
                  'site': {'latitude': '48.85', 'measurement_height': '3.0'},
                  'lag': {'co2_min_s': '-1.0', 'co2_max_s': '1.0',
                          'h2o_min_s': '-1.0', 'h2o_max_s': '1.0'}}
        for section, keys in (sections or {}).items():
            merged.setdefault(section, {})
            merged[section].update(keys)
        path = self.path('miniflux.ini')
        with open(path, 'w') as handle:
            for section in sorted(merged):
                handle.write('[%s]\n' % section)
                for key in sorted(merged[section]):
                    handle.write('%s = %s\n' % (key, merged[section][key]))
        return config.load(path)

    def run_rows(self, sections=None, **kwargs):
        """Write the file, run the pipeline, and read the output table back.

        Returns (rows, summary): each row is a dict of column name -> str, exactly as
        written. Usually one row -- but ``closed = left`` puts the sample that lands on the
        closing boundary in a period of its own, and that is the correct answer, so the
        count is the caller's to assert.
        """
        self.write_raw(**kwargs)
        cfg = self.load(sections)
        with write.Writer(cfg.files.output_csv, cfg) as writer:
            summary = pipeline.run(cfg, writer)
        with open(cfg.files.output_csv) as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(rows, 'the run wrote no row at all')
        return rows, summary

    def run_to_csv(self, sections=None, **kwargs):
        """One period in, one row out. Returns (row, summary)."""
        rows, summary = self.run_rows(sections, **kwargs)
        self.assertEqual(len(rows), 1, 'one period in, one row out')
        return rows[0], summary


class EndToEndTest(IntegrationTestCase):
    """One file on disk, one row out, checked against what was planted in the file."""

    def test_the_row_reproduces_the_planted_lag_spikes_and_covariance(self):
        row, summary = self.run_to_csv()
        self.assertEqual(summary, {'periods': 1, 'skipped_short': 0, 'rows': 1,
                                   'warnings': summary['warnings']})
        self.assertEqual(row['TIMESTAMP_START'], '202509081930')
        self.assertEqual(row['TIMESTAMP_END'], '202509081931')
        self.assertEqual(row['N_IN'], str(N))
        self.assertEqual(row['N_DUP'], '0')
        # The MAD test finds the planted spikes and nothing else.
        self.assertEqual(row['N_SPIKE_U'], '2')
        self.assertEqual(row['N_SPIKE_CO2'], '1')
        for column in ('N_SPIKE_V', 'N_SPIKE_W', 'N_SPIKE_TS', 'N_SPIKE_H2O'):
            self.assertEqual(row[column], '0')
        # The lag search recovers the delay that was built into the CO2 series.
        self.assertEqual(float(row['CO2_LAG']), LAG_SAMPLES / FREQ_HZ)
        self.assertEqual(float(row['CO2_LAG_OPT']), LAG_SAMPLES / FREQ_HZ)
        self.assertEqual(row['CO2_LAG_DEFAULT'], '0')
        self.assertEqual(float(row['H2O_LAG']), 0.0)
        # cov(w', T') is TS_GAIN * var(w') to within the sampling noise of 1200 samples,
        # and H = cov * rho * cp to the precision the table is printed at.
        covariance = float(row['COV_W_TS'])
        self.assertAlmostEqual(covariance, TS_GAIN * float(row['VAR_W']), delta=0.02)
        self.assertAlmostEqual(float(row['H_L0']),
                               covariance * float(row['RHO_A']) * float(row['CP']),
                               delta=0.01)
        # Signs: a warm updraught carries CO2 down and water up.
        self.assertGreater(float(row['H']), 0.0)
        self.assertLess(float(row['FC']), 0.0)
        self.assertGreater(float(row['LE']), 0.0)

    def test_the_published_state_is_physically_plausible(self):
        row, _summary = self.run_to_csv()
        self.assertAlmostEqual(float(row['PA']), 99000.0, delta=1.0)
        self.assertAlmostEqual(float(row['T_SONIC']), 293.15, delta=0.5)
        self.assertLess(float(row['TA']), float(row['T_SONIC']))   # sonic reads virtual
        self.assertTrue(0.0 < float(row['RH']) < 100.0)
        self.assertAlmostEqual(float(row['CO2_MEAN']), 420.0, delta=2.0)
        self.assertAlmostEqual(float(row['RHO_A']), 1.18, delta=0.05)
        self.assertAlmostEqual(float(row['CP']), 1012.0, delta=5.0)
        self.assertLess(float(row['ZL']), 0.0)                     # unstable: H > 0
        for column in ('ITC_W', 'ITC_U', 'ITC_T'):
            self.assertGreater(float(row[column]), 0.0)            # latitude is non-zero

    def test_the_units_sidecar_is_written_beside_the_table(self):
        self.run_to_csv()
        with open(write.units_path(self.path('out.csv'))) as handle:
            sidecar = list(csv.reader(handle))
        self.assertEqual(sidecar[0], ['name', 'unit'])
        self.assertEqual([line[0] for line in sidecar[1:]],
                         [name for name, _key, _unit in write.COLUMNS])

    def test_a_period_below_min_samples_is_dropped_and_counted_in_the_summary(self):
        # read.py counts a period it drops into the summary dict pipeline.run hands it.
        # Nothing else can see a period that is never yielded.
        self.write_raw(count=40)
        cfg = self.load({'period': {'min_samples': '100'}})
        summary = pipeline.run(cfg)
        self.assertEqual(summary['periods'], 0)
        self.assertEqual(summary['rows'], 0)
        self.assertEqual(summary['skipped_short'], 1)

    def test_a_dropped_period_is_counted_at_every_log_level(self):
        # skipped_short is a fact about the data, not about the log. Harvested from a log
        # record it reads 0 at [runtime] log_level = ERROR -- a legal value, and the one a
        # long run is kept quiet with -- telling the user nothing was dropped.
        self.write_raw(count=40)
        cfg = self.load({'period': {'min_samples': '100'}})
        miniflux = logging.getLogger('miniflux')
        before = miniflux.level
        miniflux.setLevel(logging.ERROR)
        self.addCleanup(miniflux.setLevel, before)
        self.assertEqual(pipeline.run(cfg)['skipped_short'], 1)

    def test_the_stream_cuts_one_file_into_consecutive_periods(self):
        # Two minutes of samples, one-minute periods: the same file yields two rows,
        # labelled by their own boundaries, and --limit stops the stream after the first.
        sections = {'period': {'averaging_minutes': '1'}}
        rows, summary = self.run_rows(sections=sections, count=2 * N)
        self.assertEqual(summary['periods'], 2)
        self.assertEqual([row['N_IN'] for row in rows], ['1200', '1200'])
        # Distinct labels, which is the whole reason config refuses a fractional period:
        # the timestamp columns are %Y%m%d%H%M (CONTRACT 15), so two rows inside one
        # minute would print the same TIMESTAMP_START.
        starts = [row['TIMESTAMP_START'] for row in rows]
        self.assertEqual(starts, ['202509081930', '202509081931'])
        self.assertEqual(len(set(starts)), len(starts))

        self.write_raw(count=2 * N)
        cfg = self.load(sections)
        with write.Writer(cfg.files.output_csv, cfg) as writer:
            limited = pipeline.run(cfg, writer, limit=1)
        self.assertEqual((limited['periods'], limited['rows']), (1, 1))

    def test_a_fractional_averaging_period_is_refused(self):
        # Half a minute is arithmetically fine and unlabelable: refuse at config time
        # rather than write two rows that print as one.
        with self.assertRaises(ConfigError) as caught:
            self.load({'period': {'averaging_minutes': '0.5'}})
        message = str(caught.exception)
        self.assertIn('averaging_minutes', message)
        self.assertIn('whole number of minutes', message)

    def test_an_all_na_period_writes_a_row_of_na_instead_of_raising(self):
        # The degeneracy contract (CONTRACT 19) end to end: no step may raise on data it
        # cannot use. Every gas and temperature token is an NA token, so thermodynamics
        # has no state, assemble omits H, wpl finds its inputs missing and qc every guard
        # fired -- and the run still produces one labelled row.
        rows = ['%s,1.0,2.0,3.0,-9999,-9999,-9999,99000.0'
                % (START + timedelta(seconds=i / FREQ_HZ)).strftime('%Y-%m-%d %H:%M:%S.%f')
                for i in range(200)]
        with open(self.path('data.dat'), 'w') as handle:
            handle.write(HEADER + '\n' + '\n'.join(rows) + '\n')
        cfg = self.load()
        with write.Writer(cfg.files.output_csv, cfg) as writer:
            summary = pipeline.run(cfg, writer)
        self.assertEqual(summary['rows'], 1)
        with open(cfg.files.output_csv) as handle:
            row = list(csv.DictReader(handle))[0]
        self.assertEqual(row['TIMESTAMP_END'], '202509081931')
        self.assertEqual(row['N_IN'], '200')
        self.assertNotEqual(row['WS'], '-9999')          # the wind is still real
        for column in ('T_SONIC', 'H_L0', 'H', 'FC', 'LE', 'RH', 'ITC_W'):
            self.assertEqual(row[column], '-9999', column)

    def test_a_numeric_timestamp_reads_the_same_period_as_an_iso_one(self):
        row, _summary = self.run_to_csv(sections={'timestamp': {'format': 'numeric'}},
                                        numeric_stamps=True)
        self.assertEqual(row['TIMESTAMP_END'], '202509081931')
        self.assertEqual(row['N_IN'], str(N))


class BothKernelPathsTest(IntegrationTestCase):
    """CONTRACT 3.8 at the scale of a whole period: same decisions, same numbers."""

    def test_the_pure_and_numpy_paths_agree_on_every_published_value(self):
        if not kernels.NUMPY_AVAILABLE:
            self.skipTest('numpy is not installed')
        self.write_raw()
        cfg = self.load()
        rows = []
        for enabled in (True, False):
            kernels.set_numpy(enabled)
            rows.append(pipeline.run_period(next(read.periods(cfg)), cfg)['meta'])
        fast, pure = rows
        self.assertEqual(sorted(fast), sorted(pure))
        for key in sorted(fast):
            a, b = fast[key], pure[key]
            if isinstance(a, float) and math.isfinite(a):
                # 1e-12 relative (CONTRACT 3.8) OR 1e-12 absolute, whichever is looser: a
                # few published quantities are differences that the processing has driven
                # to zero -- mean(v) and mean(w) after the rotation, and the steady-state
                # percentage of a stationary period -- and a relative tolerance on a number
                # that is all cancellation measures nothing. The flags and the lags they
                # feed are compared exactly, below, which is the decision the contract binds.
                self.assertLessEqual(abs(a - b), max(1e-12 * abs(a), 1e-12), key)
            else:
                self.assertEqual(a, b, key)                        # counts, flags, lags


class EveryBranchRunsTest(IntegrationTestCase):
    """Every branch of every method key, end to end. A branch nothing enters is a rumour."""

    VARIANTS = (
        ('despike off', {'despike': {'enabled': 'false'}}),
        ('rotate none', {'rotate': {'method': 'none'}}),
        ('lag covmax', {'lag': {'method': 'covmax'}}),
        ('lag fixed', {'lag': {'method': 'fixed', 'co2_nominal_s': '0.25'}}),
        ('lag none', {'lag': {'method': 'none'}}),
        ('lag no window', {'lag': {'co2_min_s': '0.0', 'co2_max_s': '0.0'}}),
        ('detrend linear', {'detrend': {'method': 'linear'}}),
        ('detrend subset', {'detrend': {'variables': 'w,ts'}}),
        ('despike subset', {'despike': {'variables': 'w'}}),
        ('closed left', {'period': {'closed': 'left'}}),
        ('no itc', {'qc': {'itc': 'false'}}),
        ('sst on h2o', {'qc': {'steady_state_pair': 'w,h2o'}}),
        ('wpl on', {'wpl': {'enabled': 'on'}}),
        ('constant pressure', {'variables': {'p_air': ''}, 'site': {'pressure_pa': '99000'}}),
        ('measured ta', {'variables': {'ta': 'Ts'}}),
        ('pure kernels', {'runtime': {'use_numpy': 'off'}}),
    )

    def test_every_configuration_branch_produces_a_full_row(self):
        for label, sections in self.VARIANTS:
            with self.subTest(variant=label):
                rows, summary = self.run_rows(sections=sections)
                self.assertEqual(summary['rows'], len(rows))
                row = rows[0]
                self.assertEqual(sorted(row), sorted(name for name, _k, _u in write.COLUMNS))
                # The four columns that no branch above may turn into na_value: the period
                # was read, so it has an extent, a wind and a temperature.
                for column in ('N_IN', 'WS', 'T_SONIC', 'COV_W_TS'):
                    self.assertNotEqual(row[column], '-9999', column)

    def test_a_declared_air_temperature_column_is_used_instead_of_the_sonic_one(self):
        # Pointing [variables] ta at the sonic column is artificial, but it is the one way
        # to prove which branch ran: every sample has a finite measured Ta, so TA must come
        # out as the mean of that column rather than as the humidity-corrected Ts.
        sonic, _summary = self.run_to_csv()
        measured, _summary = self.run_to_csv(sections={'variables': {'ta': 'Ts'}})
        self.assertEqual(measured['TA'], measured['T_SONIC'])
        self.assertLess(float(sonic['TA']), float(sonic['T_SONIC']))

    def test_a_molar_density_run_applies_the_wpl_correction(self):
        # The same period declared as densities: the gas columns are read as mol m-3, so
        # WPL is owed and must change FC and LE rather than copying them.
        columns = dict(self.columns)
        n_d = 99000.0 / (8.314462618 * 293.15)          # [mol m-3] dry air, near enough
        columns['co2'] = [value * 1e-6 * n_d * 1e3 for value in self.columns['co2']]
        columns['h2o'] = [value * 1e-3 * n_d * 1e3 for value in self.columns['h2o']]
        row, _summary = self.run_to_csv(
            sections={'units': {'co2': 'mmol/m3', 'h2o': 'mmol/m3'},
                      'gases': {'co2_measure_type': 'molar_density',
                                'h2o_measure_type': 'molar_density'}},
            columns=columns)
        self.assertEqual(row['WPL_APPLIED'], '1')
        self.assertNotEqual(row['FC'], row['FC_L0'])
        self.assertNotEqual(row['LE'], row['LE_L0'])
        # The published diagnostics carry their fixed unit whatever the measure type.
        self.assertAlmostEqual(float(row['CO2_MEAN']), 420.0, delta=5.0)
        self.assertAlmostEqual(float(row['H2O_MEAN']), 12.0, delta=1.0)

    def test_wpl_off_with_a_gas_that_is_owed_a_correction_reports_no_flux(self):
        columns = dict(self.columns)
        n_d = 99000.0 / (8.314462618 * 293.15)
        columns['co2'] = [value * 1e-6 * n_d * 1e3 for value in self.columns['co2']]
        row, _summary = self.run_to_csv(
            sections={'units': {'co2': 'mmol/m3'},
                      'gases': {'co2_measure_type': 'molar_density'},
                      'wpl': {'enabled': 'off'}},
            columns=columns)
        # CONTRACT 12: the owed gas gets no number at all rather than an uncorrected one
        # under a corrected name. FC_L0 is still published beside it.
        self.assertEqual(row['FC'], '-9999')
        self.assertNotEqual(row['FC_L0'], '-9999')


class CommandLineTest(IntegrationTestCase):
    """CONTRACT 16: the exit codes and the flags, driven as a user drives them."""

    def test_run_writes_the_table_and_exits_zero(self):
        self.write_raw()
        self.load()                                    # writes the .ini this run reads
        self.assertEqual(cli.main(['run', self.path('miniflux.ini')]), 0)
        self.assertTrue(os.path.exists(self.path('out.csv')))

    def test_pure_produces_the_same_table_as_the_default_path(self):
        self.write_raw()
        self.load()
        ini = self.path('miniflux.ini')
        cli.main(['run', ini])
        with open(self.path('out.csv')) as handle:
            default = handle.read()
        cli.main(['run', ini, '--pure'])
        with open(self.path('out.csv')) as handle:
            pure = handle.read()
        self.assertEqual(default, pure)
        self.assertFalse(kernels.HAVE_NUMPY)           # --pure is a process-wide switch

    def test_check_reads_no_data_and_a_broken_config_exits_two(self):
        self.load()
        with open(os.devnull, 'w') as sink:
            with contextlib.redirect_stdout(sink):     # `check` echoes the config to stdout
                self.assertEqual(cli.main(['check', self.path('miniflux.ini')]), 0)
        with open(self.path('broken.ini'), 'w') as handle:
            handle.write('[period]\nacquisition_frequency = 0\n')
        self.assertEqual(cli.main(['check', self.path('broken.ini')]), 2)

    def test_a_missing_input_file_exits_three(self):
        self.load()                                    # writes the .ini, but no .dat
        self.assertEqual(cli.main(['run', self.path('miniflux.ini')]), 3)


if __name__ == '__main__':
    unittest.main()
