"""CONTRACT sections 14, 15, 16 and 18.

What this file pins: the normative step order, one synthetic period carried end to end, the
exact 62-column table with its header and its units sidecar, the na policy, the published
unit scaling, and the rule that a step which raises still produces a written row.

The physics of each step is pinned by that step's own test file; here the assertions are
about plumbing -- shape, order, and what reaches the CSV.
"""

import contextlib
import csv
import io
import logging
import os
import random
import tempfile
import unittest
from datetime import datetime, timedelta

from miniflux import (cell, cli, config, despike, detrend, flux, kernels, lag, pipeline,
                      qc, rotate, spectral, wpl, write)

# CONTRACT section 14, verbatim.
EXPECTED_STEPS = ['cell', 'despike', 'rotate', 'lag', 'thermodynamics', 'detrend',
                  'moments', 'assemble', 'wpl', 'spectral', 'qc']

# CONTRACT section 18, verbatim, in order.
EXPECTED_COLUMNS = [
    'TIMESTAMP_START', 'TIMESTAMP_END', 'N_IN', 'N_DUP', 'N_SPIKE_U', 'N_SPIKE_V',
    'N_SPIKE_W', 'N_SPIKE_TS', 'N_SPIKE_CO2', 'N_SPIKE_H2O', 'N_CELL_CONV', 'WS', 'WD',
    'THETA', 'PHI', 'USTAR', 'MO_LENGTH', 'ZL', 'TA', 'T_SONIC', 'PA', 'T_CELL', 'P_CELL',
    'RH', 'CO2_MEAN', 'H2O_MEAN',
    'RHO_A', 'Q', 'CP', 'LAMBDA_V', 'VAR_U', 'VAR_V', 'VAR_W', 'VAR_TS', 'COV_U_W',
    'COV_V_W', 'COV_W_TS', 'CO2_LAG', 'CO2_LAG_OPT', 'CO2_LAG_DEFAULT', 'H2O_LAG',
    'H2O_LAG_OPT', 'H2O_LAG_DEFAULT', 'H_L0', 'H', 'FC_L0', 'FC', 'LE_L0', 'LE', 'E_L0',
    'E', 'WPL_APPLIED', 'SCF_CO2', 'SCF_H2O', 'FC_SPEC', 'LE_SPEC', 'E_SPEC',
    'SST_PCT', 'SST_FLAG', 'ITC_W', 'ITC_U', 'ITC_T',
]

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

[units]
ts = K
co2 = ppm
h2o = ppt

[gases]
co2_measure_type = mixing_ratio
h2o_measure_type = mixing_ratio
"""


def make_config(tmpdir, **overrides):
    """Write a minimal valid .ini in tmpdir and load it. Returns the Config of section 5."""
    path = os.path.join(tmpdir, 'miniflux.ini')
    settings = {'glob': os.path.join(tmpdir, '*.dat'),
                'output': os.path.join(tmpdir, 'out.csv')}
    settings.update(overrides)
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(CONFIG_TEXT.format(**settings))
    return config.load(path)


def synthetic_period(n=1200, freq=10.0):
    """One unstable midday period of 10 Hz data, built in memory (no file, no read.py).

    Correlations are put in by hand so the period is not degenerate: warm air rises (H > 0),
    CO2 goes down while air goes up (uptake, FC < 0) and water goes up with it (LE > 0).
    Returns the period dict of CONTRACT section 1.
    """
    rng = random.Random(20260912)
    start = datetime(2026, 6, 21, 12, 0, 0)
    values = {name: [] for name in ('u', 'v', 'w', 'ts', 'co2', 'h2o', 'p_air')}
    for _i in range(n):
        gust = rng.gauss(0.0, 0.5)                       # w' [m s-1]
        values['u'].append(2.5 + rng.gauss(0.0, 0.6))
        values['v'].append(0.3 + rng.gauss(0.0, 0.4))
        values['w'].append(0.02 + gust)
        values['ts'].append(293.15 + 0.5 * gust + rng.gauss(0.0, 0.1))
        values['co2'].append(4.20e-4 - 2.0e-6 * gust + rng.gauss(0.0, 2.0e-6))
        values['h2o'].append(1.20e-2 + 4.4e-4 * gust + rng.gauss(0.0, 1.0e-4))
        values['p_air'].append(99000.0)
    period = {
        'meta': {'period_start': start,
                 'period_end': start + timedelta(minutes=30),
                 'n_in': n, 'n_dup': 0, 'freq_hz': freq},
        # No measured air temperature in this file: an all-NaN series, never a missing key.
        'ta': kernels.new(n),
    }
    for name, series in values.items():
        period[name] = kernels.from_values(series)
    return period


def read_table(path):
    """Read a written table back. Returns (header, list of rows), both lists of str."""
    with open(path, newline='', encoding='utf-8') as handle:
        rows = list(csv.reader(handle))
    return rows[0], rows[1:]


class StepOrderTest(unittest.TestCase):
    """CONTRACT section 14: the order is normative and every position is load-bearing."""

    def test_names_and_order(self):
        self.assertEqual([name for name, _step in pipeline.STEPS], EXPECTED_STEPS)

    def test_each_name_is_bound_to_the_contract_function(self):
        expected = [cell.convert, despike.despike, rotate.rotate, lag.apply_lags,
                    flux.thermodynamics, detrend.detrend, flux.moments, flux.assemble,
                    wpl.correct, spectral.correct, qc.quality]
        self.assertEqual([step for _name, step in pipeline.STEPS], expected)

    def test_cell_is_first_and_qc_is_last(self):
        # Stated separately because these two are the positions a refactor moves first: the
        # cell conversion has to precede the MAD threshold so the threshold sees a
        # conserved quantity, and the ITC test needs the z/L assemble writes.
        self.assertEqual(pipeline.STEPS[0][0], 'cell')
        self.assertEqual(pipeline.STEPS[-1][0], 'qc')

    def test_despike_precedes_every_step_that_reshapes_a_distribution(self):
        names = [name for name, _step in pipeline.STEPS]
        self.assertLess(names.index('despike'), names.index('rotate'))
        self.assertLess(names.index('despike'), names.index('detrend'))
        # cell.convert is the one thing allowed before it, and only because the cell state
        # it divides out is not screened anywhere else.
        self.assertEqual(names.index('despike'), names.index('cell') + 1)


class ColumnTableTest(unittest.TestCase):
    """CONTRACT section 18: the exact table, in order."""

    def test_sixty_two_columns_in_order(self):
        self.assertEqual(len(write.COLUMNS), 62)
        self.assertEqual([name for name, _key, _unit in write.COLUMNS], EXPECTED_COLUMNS)

    def test_every_entry_is_name_key_unit(self):
        for entry in write.COLUMNS:
            self.assertEqual(len(entry), 3)
            for field in entry:
                self.assertTrue(isinstance(field, str) and field)

    def test_no_meta_key_is_published_twice(self):
        keys = [key for _name, key, _unit in write.COLUMNS]
        self.assertEqual(len(keys), len(set(keys)))


class WriterTest(unittest.TestCase):
    """CONTRACT section 15: the header, the sidecar, the na policy, the unit scaling."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(_remove_tree, self.tmpdir)
        self.cfg = make_config(self.tmpdir)
        self.path = self.cfg.files.output_csv

    def test_header_is_written_on_construction(self):
        writer = write.Writer(self.path, self.cfg)
        writer.close()
        header, rows = read_table(self.path)
        self.assertEqual(header, EXPECTED_COLUMNS)
        self.assertEqual(rows, [])

    def test_units_sidecar(self):
        with write.Writer(self.path, self.cfg):
            pass
        header, rows = read_table(write.units_path(self.path))
        self.assertEqual(header, ['name', 'unit'])
        self.assertEqual(rows, [[name, unit] for name, _key, unit in write.COLUMNS])

    def test_missing_nan_and_inf_all_become_na(self):
        period = {'meta': {'ustar': float('nan'), 'h': float('inf'),
                           'n_in': None, 'wind_dir': -float('inf')}}
        with write.Writer(self.path, self.cfg) as writer:
            writer.write(period)
        _header, rows = read_table(self.path)
        self.assertEqual(rows[0], ['-9999'] * 62)

    def test_published_unit_scaling_happens_here(self):
        # meta is SI; the table is in the units of section 18.
        period = {'meta': {'theta': 3.141592653589793, 'phi': -1.5707963267948966,
                           'fc_l0': -2.0e-5, 'fc': -1.8e-5, 'e_l0': 8.0e-5, 'e': 8.2e-5,
                           'le': 200.0, 'n_in': 36000, 'wpl_applied': True,
                           'co2_lag_default_used': False, 'sst_flag': 2}}
        with write.Writer(self.path, self.cfg) as writer:
            writer.write(period)
        _header, rows = read_table(self.path)
        row = dict(zip(EXPECTED_COLUMNS, rows[0]))
        self.assertEqual(float(row['THETA']), 180.0)
        self.assertEqual(float(row['PHI']), -90.0)
        self.assertAlmostEqual(float(row['FC_L0']), -20.0, places=9)
        self.assertAlmostEqual(float(row['FC']), -18.0, places=9)
        self.assertAlmostEqual(float(row['E_L0']), 0.08, places=9)
        self.assertAlmostEqual(float(row['E']), 0.082, places=9)
        self.assertEqual(row['LE'], '200')                  # W m-2, unscaled
        self.assertEqual(row['N_IN'], '36000')              # a count, not %.6g
        self.assertEqual(row['WPL_APPLIED'], '1')
        self.assertEqual(row['CO2_LAG_DEFAULT'], '0')
        self.assertEqual(row['SST_FLAG'], '2')

    def test_timestamps_are_yyyymmddhhmm(self):
        period = {'meta': {'period_start': datetime(2026, 6, 21, 12, 0),
                           'period_end': datetime(2026, 6, 21, 12, 30)}}
        with write.Writer(self.path, self.cfg) as writer:
            writer.write(period)
        _header, rows = read_table(self.path)
        self.assertEqual(rows[0][:2], ['202606211200', '202606211230'])

    def test_na_value_is_configurable(self):
        self.cfg.output.na_value = 'NA'
        with write.Writer(self.path, self.cfg) as writer:
            writer.write({'meta': {}})
        _header, rows = read_table(self.path)
        self.assertEqual(set(rows[0]), set(['NA']))


class EndToEndTest(unittest.TestCase):
    """One synthetic period through STEPS and out to the table."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(_remove_tree, self.tmpdir)
        self.cfg = make_config(self.tmpdir)
        self.path = self.cfg.files.output_csv

    def _row(self, period):
        with write.Writer(self.path, self.cfg) as writer:
            writer.write(period)
        header, rows = read_table(self.path)
        self.assertEqual(header, EXPECTED_COLUMNS)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]), 62)
        return dict(zip(header, rows[0]))

    def test_synthetic_period_produces_one_complete_row(self):
        period = pipeline.run_period(synthetic_period(), self.cfg)
        row = self._row(period)
        self.assertEqual(row['TIMESTAMP_START'], '202606211200')
        self.assertEqual(row['TIMESTAMP_END'], '202606211230')
        self.assertEqual(row['N_IN'], '1200')
        self.assertEqual(row['N_DUP'], '0')
        # The columns that only exist if every step ran and the period was not degenerate.
        for name in ('WS', 'WD', 'THETA', 'PHI', 'USTAR', 'T_SONIC', 'PA', 'VAR_W',
                     'COV_W_TS', 'H_L0', 'H', 'FC_L0', 'LE_L0', 'CO2_MEAN', 'H2O_MEAN'):
            self.assertNotEqual(row[name], '-9999', '%s was not computed' % name)
            float(row[name])            # and it parses as a number

    def test_the_synthetic_period_is_physically_the_right_way_up(self):
        # Cheap guard that the plumbing did not transpose a sign on the way through: this
        # period was built as warm air rising over a taking-up canopy.
        period = pipeline.run_period(synthetic_period(), self.cfg)
        row = self._row(period)
        self.assertGreater(float(row['COV_W_TS']), 0.0)
        self.assertGreater(float(row['H_L0']), 0.0)
        self.assertLess(float(row['FC_L0']), 0.0)

    def test_run_period_carries_every_step_forward(self):
        # A step may return a new dict (CONTRACT section 1.1), so what is pinned is that the
        # caller rebinds: one meta key from each of the nine steps has to survive.
        period = pipeline.run_period(synthetic_period(n=300), self.cfg)
        self.assertIsInstance(period, dict)
        for key in ('n_spike_u', 'wind_speed', 'co2_lag_samples', 'ts_mean', 'co2_mean',
                    'ustar', 'h_l0', 'wpl_applied', 'sst_flag'):
            self.assertIn(key, period['meta'], 'meta key %s did not survive' % key)

    def test_a_raising_step_still_produces_a_row(self):
        original = list(pipeline.STEPS)
        self.addCleanup(_restore_steps, original)

        def explode(period, cfg):
            raise ZeroDivisionError('synthetic failure inside a step')

        # rotate is position 2: cell and despike have run, everything after them has not.
        pipeline.STEPS[2] = ('rotate', explode)
        with self.assertLogs('miniflux', level='ERROR') as captured:
            period = pipeline.run_period(synthetic_period(n=300), self.cfg)
        self.assertIn('step rotate failed', '\n'.join(captured.output))
        row = self._row(period)
        # The row exists and is a full line: the failure is visible, not a hole.
        self.assertEqual(row['TIMESTAMP_START'], '202606211200')
        self.assertNotEqual(row['N_SPIKE_U'], '-9999')      # despike ran
        self.assertEqual(row['USTAR'], '-9999')             # moments never did
        self.assertEqual(row['H'], '-9999')


class RunTest(unittest.TestCase):
    """pipeline.run: the loop, the summary counters and --limit."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(_remove_tree, self.tmpdir)
        self.cfg = make_config(self.tmpdir)
        self.path = self.cfg.files.output_csv

    def _fake_periods(self, count, warn=0, short=0):
        """Stand in for read.periods so run() can be exercised without a data file."""
        def periods(cfg, counters=None):
            reader = logging.getLogger('miniflux.read')
            for _i in range(short):
                # How read.py reports a period it dropped for [period] min_samples: the
                # count goes into the caller's dict, the message only into the log.
                if counters is not None:
                    counters['skipped_short'] = counters.get('skipped_short', 0) + 1
                reader.warning('period too short, skipped')
            for _i in range(warn):
                reader.warning('something a run should count')
            for index in range(count):
                period = synthetic_period(n=300)
                period['meta']['period_start'] += timedelta(minutes=30 * index)
                period['meta']['period_end'] += timedelta(minutes=30 * index)
                yield period
        original = pipeline.read.periods
        pipeline.read.periods = periods
        self.addCleanup(_restore_reader, original)

    def test_summary_counts_periods_rows_warnings_and_short_periods(self):
        self._fake_periods(3, warn=2, short=1)
        with self.assertLogs('miniflux', level='WARNING'):
            with write.Writer(self.path, self.cfg) as writer:
                summary = pipeline.run(self.cfg, writer)
        self.assertEqual(summary['periods'], 3)
        self.assertEqual(summary['rows'], 3)
        self.assertEqual(summary['skipped_short'], 1)
        self.assertEqual(summary['warnings'], 3)            # the two plus the short one
        self.assertEqual(sorted(summary), ['periods', 'rows', 'skipped_short', 'warnings'])
        _header, rows = read_table(self.path)
        self.assertEqual(len(rows), 3)
        self.assertEqual([row[0] for row in rows],
                         ['202606211200', '202606211230', '202606211300'])

    def test_limit_stops_early(self):
        self._fake_periods(5)
        with write.Writer(self.path, self.cfg) as writer:
            summary = pipeline.run(self.cfg, writer, limit=2)
        self.assertEqual(summary['periods'], 2)
        self.assertEqual(summary['rows'], 2)
        _header, rows = read_table(self.path)
        self.assertEqual(len(rows), 2)

    def test_no_writer_writes_nothing(self):
        self._fake_periods(2)
        summary = pipeline.run(self.cfg, None)
        self.assertEqual(summary['periods'], 2)
        self.assertEqual(summary['rows'], 0)
        self.assertFalse(os.path.exists(self.path))

    def test_the_counter_is_removed_after_the_run(self):
        self._fake_periods(1)
        before = [len(logging.getLogger(name).filters)
                  for name in ('miniflux', 'miniflux.read', 'miniflux.pipeline')]
        pipeline.run(self.cfg, None)
        after = [len(logging.getLogger(name).filters)
                 for name in ('miniflux', 'miniflux.read', 'miniflux.pipeline')]
        self.assertEqual(before, after)


class CliTest(unittest.TestCase):
    """CONTRACT section 16: the two subcommands and the exit codes."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(_remove_tree, self.tmpdir)
        self.addCleanup(_restore_logging, logging.getLogger('miniflux'))
        self.cfg = make_config(self.tmpdir)            # also writes miniflux.ini
        self.ini = os.path.join(self.tmpdir, 'miniflux.ini')

    def _main(self, argv):
        """Run the CLI with both streams captured. Returns (exit code, stdout, stderr)."""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_check_validates_echoes_and_reads_nothing(self):
        code, echoed, _err = self._main(['check', self.ini, '--log-level', 'ERROR'])
        self.assertEqual(code, 0)
        self.assertIn('period.averaging_minutes = 30.0', echoed)
        self.assertIn('site.pressure_pa = 99000.0', echoed)
        self.assertFalse(os.path.exists(self.cfg.files.output_csv))

    def test_a_bad_config_is_exit_2(self):
        with self.assertLogs('miniflux', level='ERROR') as captured:
            code, _out, _err = self._main(['check',
                                           os.path.join(self.tmpdir, 'absent.ini')])
        self.assertEqual(code, 2)
        self.assertIn('config file not found', '\n'.join(captured.output))

    def test_run_with_no_input_files_is_exit_3(self):
        # The glob matches nothing, which is read.py's refusal, which is exit 3.
        with self.assertLogs('miniflux', level='ERROR'):
            code, _out, _err = self._main(['run', self.ini, '--log-level', 'ERROR'])
        self.assertEqual(code, 3)

    def test_pure_turns_the_numpy_path_off(self):
        self.addCleanup(kernels.set_numpy, True)
        with self.assertLogs('miniflux', level='ERROR'):
            self._main(['run', self.ini, '--pure', '--limit', '1', '--log-level', 'ERROR'])
        self.assertFalse(kernels.HAVE_NUMPY)


def _remove_tree(path):
    """Delete a temporary directory tree. Only ever called on tempfile.mkdtemp output."""
    for name in os.listdir(path):
        os.remove(os.path.join(path, name))
    os.rmdir(path)


def _restore_steps(original):
    pipeline.STEPS[:] = original


def _restore_reader(original):
    pipeline.read.periods = original


def _restore_logging(logger):
    """Undo what cli._configure_logging did to the package logger."""
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.propagate = True
    logger.setLevel(logging.NOTSET)


if __name__ == '__main__':
    unittest.main()
