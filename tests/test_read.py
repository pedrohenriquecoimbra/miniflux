"""What read.py promises: CONTRACT 7, ALGORITHMS 2.

Both timestamp shapes, both sides of `closed`, duplicate and backwards rows, the NA
tokens, every ReadError, and one end-to-end read of a real 20 Hz TOA5 file whose CO2 is
in mg/m3, whose H2O is in g/m3, whose sonic temperature is in degC and whose pressure is
in kPa -- the four unit rows that only this module ever applies.
"""

import csv
import logging
import math
import os
import shutil
import tempfile
import unittest
from array import array
from datetime import datetime

from miniflux import config, read
from miniflux.constants import MCO2, MV, T0
from miniflux.errors import ReadError

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'fixtures', 'toa5_10s.csv')

# The generated files use the shipped default column names, so a test's .ini only has to
# say what it changes.
HEADER = 'TIMESTAMP,Ux,Uy,Uz,Ts,CO2,H2O,Press'
ROW = '%s,1.0,2.0,3.0,300.0,400.0,10.0,101325.0'

# CONTRACT 6.1 as the fixture declares it: degC -> K, kPa -> Pa, mg/m3 and g/m3 -> mol m-3
# through the gas's own molar mass.
FIXTURE_INI = {
    'files': {'header_line': '2', 'first_data_line': '5'},
    'variables': {'u': 'RAW_IRGASON_Ux', 'v': 'RAW_IRGASON_Uy', 'w': 'RAW_IRGASON_Uz',
                  'ts': 'RAW_IRGASON_Ts', 'co2': 'RAW_IRGASON_CO2',
                  'h2o': 'RAW_IRGASON_H2O', 'p_air': 'RAW_IRGASON_CellPrs'},
    'units': {'ts': 'degC', 'p_air': 'kPa', 'co2': 'mg/m3', 'h2o': 'g/m3'},
    'gases': {'co2_measure_type': 'molar_density', 'h2o_measure_type': 'molar_density'},
}


QUIET = logging.NullHandler()


def setUpModule():
    # read.py warns on purpose (duplicates, short periods, the observed rate) and the test
    # output is not the place. A handler of any kind stops logging's last-resort writer to
    # stderr, and unlike logging.disable() it leaves the records for assertLogs to see.
    logging.getLogger('miniflux').addHandler(QUIET)


def tearDownModule():
    logging.getLogger('miniflux').removeHandler(QUIET)


class ReadTestCase(unittest.TestCase):
    """A temp directory, a .ini writer, and a .csv writer."""

    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix='miniflux_read_')
        self.addCleanup(shutil.rmtree, self.directory, True)

    def write_csv(self, rows, name='data.csv', header=HEADER, preamble=()):
        path = os.path.join(self.directory, name)
        with open(path, 'w') as handle:
            for line in preamble:
                handle.write(line + '\n')
            handle.write(header + '\n')
            for row in rows:
                handle.write(row + '\n')
        return path

    def load(self, sections=None, pattern='*.csv', source=None):
        """Write an .ini over the shipped defaults and load it through config.load."""
        merged = {'files': {'input_glob': source or os.path.join(self.directory, pattern)}}
        for section, keys in (sections or {}).items():
            merged.setdefault(section, {})
            merged[section].update(keys)
        path = os.path.join(self.directory, 'miniflux.ini')
        with open(path, 'w') as handle:
            for section in sorted(merged):
                handle.write('[%s]\n' % section)
                for key in sorted(merged[section]):
                    handle.write('%s = %s\n' % (key, merged[section][key]))
        return config.load(path)

    def read_one(self, rows, sections=None, **kwargs):
        """Read one generated file and return its single period."""
        self.write_csv(rows, **kwargs)
        found = list(read.periods(self.load(sections)))
        self.assertEqual(len(found), 1)
        return found[0]


class PeriodBoundsTest(unittest.TestCase):
    """ALGORITHMS 0.1: the cut, and which side a boundary sample falls on."""

    def test_interior_sample_is_unaffected_by_closed(self):
        t = datetime(2025, 9, 8, 19, 32, 0, 50000)
        expected = (datetime(2025, 9, 8, 19, 30), datetime(2025, 9, 8, 20, 0))
        self.assertEqual(read.period_bounds(t, 1800.0, 'right'), expected)
        self.assertEqual(read.period_bounds(t, 1800.0, 'left'), expected)

    def test_boundary_sample_closes_the_earlier_period_when_closed_is_right(self):
        t = datetime(2025, 9, 8, 19, 30)
        start, end = read.period_bounds(t, 1800.0, 'right')
        self.assertEqual((start, end), (datetime(2025, 9, 8, 19, 0), t))

    def test_boundary_sample_opens_the_later_period_when_closed_is_left(self):
        t = datetime(2025, 9, 8, 19, 30)
        start, end = read.period_bounds(t, 1800.0, 'left')
        self.assertEqual((start, end), (t, datetime(2025, 9, 8, 20, 0)))

    def test_the_two_conventions_differ_by_exactly_one_period(self):
        t = datetime(2025, 9, 8, 19, 30)
        right = read.period_bounds(t, 1800.0, 'right')
        left = read.period_bounds(t, 1800.0, 'left')
        self.assertEqual(right[1], left[0])

    def test_a_ten_minute_period(self):
        t = datetime(2025, 9, 8, 19, 32, 0, 50000)
        self.assertEqual(read.period_bounds(t, 600.0, 'right'),
                         (datetime(2025, 9, 8, 19, 30), datetime(2025, 9, 8, 19, 40)))

    def test_before_the_epoch_the_cut_still_floors(self):
        t = datetime(1969, 12, 31, 23, 45)
        self.assertEqual(read.period_bounds(t, 1800.0, 'left'),
                         (datetime(1969, 12, 31, 23, 30), datetime(1970, 1, 1, 0, 0)))


class ParseTimestampTest(unittest.TestCase):
    """ALGORITHMS 2.2: parse, never infer."""

    def test_iso_without_a_fraction(self):
        self.assertEqual(read.parse_timestamp('2025-09-08 19:32:00', 'iso'),
                         datetime(2025, 9, 8, 19, 32, 0))

    def test_iso_with_a_two_digit_fraction(self):
        # The whole second and the nineteen samples after it must both parse: this is the
        # mixed-shape column a single inferred format silently coerces to NaT.
        self.assertEqual(read.parse_timestamp('2025-09-08 19:32:00.05', 'iso'),
                         datetime(2025, 9, 8, 19, 32, 0, 50000))

    def test_iso_with_six_digits_and_a_t_separator(self):
        self.assertEqual(read.parse_timestamp('2025-09-08T19:32:00.123456', 'iso'),
                         datetime(2025, 9, 8, 19, 32, 0, 123456))

    def test_surrounding_whitespace_is_tolerated(self):
        self.assertEqual(read.parse_timestamp('  2025-09-08 19:32:00 ', 'iso'),
                         datetime(2025, 9, 8, 19, 32, 0))

    def test_numeric_without_a_fraction(self):
        self.assertEqual(read.parse_timestamp('20220512233000', 'numeric'),
                         datetime(2022, 5, 12, 23, 30, 0))

    def test_numeric_keeps_a_fraction_a_double_would_lose(self):
        # 20220512233000.05 is sixteen significant digits; a float64 carries about 15.95,
        # so the same token through float() lands 781 us away. Reading it as text is what
        # makes the sample clock exact.
        self.assertEqual(read.parse_timestamp('20220512233000.05', 'numeric'),
                         datetime(2022, 5, 12, 23, 30, 0, 50000))
        self.assertEqual(int(round((float('20220512233000.05') % 1) * 1e6)), 50781)

    def test_a_shape_that_is_neither_is_refused(self):
        for token in ('', '2025/09/08 19:32:00', '19:32:00', '2025-09-08', 'nan',
                      '2025-09-08 19:32:0', '2025-09-08 19:32:00,5'):
            with self.assertRaises(ReadError):
                read.parse_timestamp(token, 'iso')

    def test_numeric_shapes_that_are_refused(self):
        for token in ('2022051223300', '2022-05-12 23:30:00', '20220512233000.'):
            with self.assertRaises(ReadError):
                read.parse_timestamp(token, 'numeric')

    def test_sub_microsecond_is_refused_rather_than_rounded(self):
        with self.assertRaises(ReadError) as caught:
            read.parse_timestamp('2025-09-08 19:32:00.1234567', 'iso')
        self.assertIn('fractional digits', str(caught.exception))

    def test_an_impossible_instant_is_refused(self):
        with self.assertRaises(ReadError) as caught:
            read.parse_timestamp('2025-13-08 19:32:00', 'iso')
        self.assertIn('not a real instant', str(caught.exception))

    def test_an_unknown_kind_is_refused(self):
        with self.assertRaises(ReadError):
            read.parse_timestamp('2025-09-08 19:32:00', 'guess')


class Toa5FixtureTest(ReadTestCase):
    """End to end over tests/fixtures/toa5_10s.csv, a real 20 Hz IRGASON stream."""

    def setUp(self):
        ReadTestCase.setUp(self)
        cfg = self.load(FIXTURE_INI, source=FIXTURE)
        found = list(read.periods(cfg))
        self.assertEqual(len(found), 1)
        self.period = found[0]

    def test_the_file_cut_keeps_the_header_and_drops_the_other_preamble_lines(self):
        # header_line 2, first_data_line 5: the TOA5 line, the units line and the
        # aggregation line are all discarded, and every remaining row is a sample.
        self.assertEqual(self.period['meta']['n_in'], 200)
        self.assertEqual(self.period['meta']['n_dup'], 0)
        self.assertEqual(self.period['meta']['freq_hz'], 20.0)

    def test_the_period_is_labelled_by_its_end(self):
        self.assertEqual(self.period['meta']['period_start'], datetime(2025, 9, 8, 19, 30))
        self.assertEqual(self.period['meta']['period_end'], datetime(2025, 9, 8, 20, 0))

    def test_all_nine_series_exist_with_one_length_and_one_type(self):
        for name in read.SERIES:
            self.assertIsInstance(self.period[name], array, name)
            self.assertEqual(self.period[name].typecode, 'd', name)
            self.assertEqual(len(self.period[name]), 200, name)
        self.assertEqual(len(self.period['t']), 200)
        self.assertIsInstance(self.period['t'], list)

    def test_the_quoted_fractional_timestamps_all_survive(self):
        self.assertEqual(self.period['t'][0], datetime(2025, 9, 8, 19, 32, 0))
        self.assertEqual(self.period['t'][1], datetime(2025, 9, 8, 19, 32, 0, 50000))
        self.assertEqual(self.period['t'][-1], datetime(2025, 9, 8, 19, 32, 10, 250000))
        for earlier, later in zip(self.period['t'], self.period['t'][1:]):
            self.assertLessEqual(earlier, later)

    def test_wind_passes_through_unscaled(self):
        self.assertEqual(self.period['u'][0], 1.801356)
        self.assertEqual(self.period['v'][0], -0.08548526)
        self.assertEqual(self.period['w'][0], -0.06308883)

    def test_degc_becomes_kelvin(self):
        self.assertEqual(self.period['ts'][0], 22.74528 + T0)

    def test_kpa_becomes_pascal(self):
        self.assertEqual(self.period['p_air'][0], 100.929 * 1000.0)

    def test_mass_density_becomes_molar_density_through_the_molar_mass(self):
        # mg m-3 -> mol m-3 is 1e-6/MCO2; g m-3 -> mol m-3 is 1e-3/MV (CONTRACT 6.1).
        self.assertEqual(self.period['co2'][0], 748.347 * (1e-6 / MCO2))
        self.assertEqual(self.period['h2o'][0], 7.582086 * (1e-3 / MV))
        self.assertAlmostEqual(self.period['co2'][0], 0.0170, places=4)
        self.assertAlmostEqual(self.period['h2o'][0], 0.4208, places=4)

    def test_an_absent_column_is_an_all_nan_series_not_a_missing_key(self):
        self.assertIn('ta', self.period)
        self.assertTrue(all(math.isnan(v) for v in self.period['ta']))

    def test_a_rate_below_the_configured_one_is_logged_not_enforced(self):
        # The fixture skips samples, so it runs at about 19.4 Hz; 20 Hz is still what the
        # run uses.
        cfg = self.load(FIXTURE_INI, source=FIXTURE)
        with self.assertLogs('miniflux.read', level='WARNING') as logged:
            list(read.periods(cfg))
        self.assertTrue(any('acquisition_frequency' in line for line in logged.output))


class NumericTimestampTest(ReadTestCase):
    def test_a_numeric_column_is_read_as_text(self):
        # Both shapes in one column -- the whole second and the fractions after it.
        rows = [ROW % '20220512233001', ROW % '20220512233001.05']
        period = self.read_one(rows, {'timestamp': {'format': 'numeric'}})
        self.assertEqual(period['t'],
                         [datetime(2022, 5, 12, 23, 30, 1),
                          datetime(2022, 5, 12, 23, 30, 1, 50000)])
        self.assertEqual(period['meta']['period_end'], datetime(2022, 5, 13, 0, 0))


class SecondsCacheTest(ReadTestCase):
    """The samples of one second share the instant of that second and nothing else."""

    def test_every_fraction_of_one_second_keeps_its_own_microseconds(self):
        rows = [ROW % '2025-09-08 19:32:00',
                ROW % '2025-09-08 19:32:00.000001',
                ROW % '2025-09-08 19:32:00.05',
                ROW % '2025-09-08 19:32:00.5',
                ROW % '2025-09-08 19:32:01']
        period = self.read_one(rows)
        self.assertEqual(period['t'], [datetime(2025, 9, 8, 19, 32, 0),
                                       datetime(2025, 9, 8, 19, 32, 0, 1),
                                       datetime(2025, 9, 8, 19, 32, 0, 50000),
                                       datetime(2025, 9, 8, 19, 32, 0, 500000),
                                       datetime(2025, 9, 8, 19, 32, 1)])

    def test_a_bad_fraction_is_still_refused_after_a_good_one_in_the_same_second(self):
        self.write_csv([ROW % '2025-09-08 19:32:00.05',
                        ROW % '2025-09-08 19:32:00.1234567'])
        with self.assertRaises(ReadError) as caught:
            list(read.periods(self.load()))
        self.assertIn('fractional digits', str(caught.exception))

    def test_whitespace_around_a_timestamp_is_tolerated_row_after_row(self):
        rows = [ROW % ' 2025-09-08 19:32:00.05 ', ROW % ' 2025-09-08 19:32:01.05 ']
        period = self.read_one(rows)
        self.assertEqual(period['t'], [datetime(2025, 9, 8, 19, 32, 0, 50000),
                                       datetime(2025, 9, 8, 19, 32, 1, 50000)])


class ClosedTest(ReadTestCase):
    """The one sample that sits exactly on the boundary, in both directions."""

    ROWS = ['2025-09-08 19:30:00,1.0,2.0,3.0,300.0,400.0,10.0,101325.0',
            '2025-09-08 19:30:00.05,4.0,2.0,3.0,300.0,400.0,10.0,101325.0']

    def test_right_closed_puts_it_in_the_earlier_period(self):
        self.write_csv(self.ROWS)
        found = list(read.periods(self.load({'period': {'closed': 'right'}})))
        self.assertEqual([p['meta']['period_end'] for p in found],
                         [datetime(2025, 9, 8, 19, 30), datetime(2025, 9, 8, 20, 0)])
        self.assertEqual([p['meta']['n_in'] for p in found], [1, 1])
        self.assertEqual(found[0]['u'][0], 1.0)

    def test_left_closed_puts_it_in_the_later_period(self):
        self.write_csv(self.ROWS)
        found = list(read.periods(self.load({'period': {'closed': 'left'}})))
        self.assertEqual([p['meta']['period_end'] for p in found],
                         [datetime(2025, 9, 8, 20, 0)])
        self.assertEqual(found[0]['meta']['n_in'], 2)
        self.assertEqual(list(found[0]['u']), [1.0, 4.0])


class DuplicateAndBackwardsTest(ReadTestCase):
    def test_a_repeated_timestamp_keeps_the_first_and_counts_the_rest(self):
        rows = ['2025-09-08 19:32:00,1.0,2.0,3.0,300.0,400.0,10.0,101325.0',
                '2025-09-08 19:32:00,9.0,2.0,3.0,300.0,400.0,10.0,101325.0',
                '2025-09-08 19:32:00.05,4.0,2.0,3.0,300.0,400.0,10.0,101325.0']
        period = self.read_one(rows)
        self.assertEqual(period['meta']['n_in'], 2)
        self.assertEqual(period['meta']['n_dup'], 1)
        self.assertEqual(list(period['u']), [1.0, 4.0])

    def test_the_duplicate_count_is_logged_once_for_the_period(self):
        rows = ['2025-09-08 19:32:00,1.0,2.0,3.0,300.0,400.0,10.0,101325.0',
                '2025-09-08 19:32:00,9.0,2.0,3.0,300.0,400.0,10.0,101325.0']
        self.write_csv(rows)
        with self.assertLogs('miniflux.read', level='WARNING') as logged:
            list(read.periods(self.load()))
        self.assertEqual(len([line for line in logged.output if 'dropped' in line]), 1)

    def test_a_backwards_sample_inside_the_period_is_dropped(self):
        rows = ['2025-09-08 19:32:00.05,1.0,2.0,3.0,300.0,400.0,10.0,101325.0',
                '2025-09-08 19:32:00.00,9.0,2.0,3.0,300.0,400.0,10.0,101325.0',
                '2025-09-08 19:32:00.10,4.0,2.0,3.0,300.0,400.0,10.0,101325.0']
        period = self.read_one(rows)
        self.assertEqual(list(period['u']), [1.0, 4.0])
        self.assertEqual(period['meta']['n_dup'], 1)
        self.assertEqual(period['t'], sorted(period['t']))

    def test_a_sample_falling_back_into_a_closed_period_is_refused(self):
        # ALGORITHMS 2.3 scopes the keep-first rule to duplicates *within a period*. A
        # sample belonging to a period already yielded cannot be put back, and dropping it
        # computes a flux from a fraction of the data that exists on disk.
        rows = ['2025-09-08 19:32:00,1.0,2.0,3.0,300.0,400.0,10.0,101325.0',
                '2025-09-08 20:02:00,2.0,2.0,3.0,300.0,400.0,10.0,101325.0',
                '2025-09-08 19:45:00,9.0,2.0,3.0,300.0,400.0,10.0,101325.0']
        self.write_csv(rows)
        with self.assertRaises(ReadError) as caught:
            list(read.periods(self.load()))
        message = str(caught.exception)
        self.assertIn('line 4', message)
        self.assertIn('2025-09-08 19:45:00', message)
        self.assertIn('20:00:00', message)      # the period it belongs to, already closed


class FileOrderTest(ReadTestCase):
    """ALGORITHMS 2.3: the files are concatenated in timestamp order, not in name order."""

    EARLY = '2025-09-08 19:32:00,1.0,2.0,3.0,300.0,400.0,10.0,101325.0'
    LATE = '2025-09-08 19:33:00,2.0,2.0,3.0,300.0,400.0,10.0,101325.0'

    def test_files_are_ordered_by_their_first_sample_not_by_their_name(self):
        # sorted() puts 'data_10.csv' before 'data_2.csv': the rollover from a one-digit
        # to a two-digit counter is the ordinary way a logger names its files.
        self.write_csv([self.LATE], name='data_10.csv')
        self.write_csv([self.EARLY], name='data_2.csv')
        found = list(read.periods(self.load()))
        self.assertEqual(len(found), 1)
        self.assertEqual(list(found[0]['u']), [1.0, 2.0])
        self.assertEqual(found[0]['meta']['n_in'], 2)
        self.assertEqual(found[0]['meta']['n_dup'], 0)

    def test_a_whole_period_is_not_lost_to_the_file_order(self):
        # Name order would close 19:30..20:00 on data_10's sample and then throw the
        # whole 18:30..19:00 period away as "duplicates" of a period it never opened.
        self.write_csv(['2025-09-08 18:50:00,1.0,2.0,3.0,300.0,400.0,10.0,101325.0'],
                       name='data_9.csv')
        self.write_csv(['2025-09-08 19:50:00,2.0,2.0,3.0,300.0,400.0,10.0,101325.0'],
                       name='data_10.csv')
        found = list(read.periods(self.load()))
        self.assertEqual([p['meta']['period_end'] for p in found],
                         [datetime(2025, 9, 8, 19, 0), datetime(2025, 9, 8, 20, 0)])


class QuotingTest(ReadTestCase):
    """CONTRACT 7.1: a line is split exactly as ``csv.reader`` splits it.

    Most lines of a 20 Hz file carry no quote at all and are cut with ``str.split``; the
    shapes below are the ones where that would be wrong, and each of them is handed back
    to the reader for that record alone. One quoted line in the middle of a file is legal,
    so the choice is made per line and never per file.
    """

    # A text column in front of the numbers, so that a line split the wrong way shifts
    # every value read after it rather than falling off the end unnoticed.
    NOTE = 'TIMESTAMP,Note,' + HEADER.split(',', 1)[1]

    def test_a_quoted_timestamp_in_front_of_unquoted_numbers(self):
        # What a TOA5 logger writes, and the only quoting on a data row of the real files.
        rows = ['"2025-09-08 19:32:00",1.0,2.0,3.0,300.0,400.0,10.0,101325.0',
                '"2025-09-08 19:32:00.05",4.0,2.0,3.0,300.0,400.0,10.0,101325.0']
        period = self.read_one(rows)
        self.assertEqual(list(period['u']), [1.0, 4.0])
        self.assertEqual(period['t'], [datetime(2025, 9, 8, 19, 32),
                                       datetime(2025, 9, 8, 19, 32, 0, 50000)])

    def test_one_quoted_line_in_the_middle_of_an_unquoted_file(self):
        rows = ['2025-09-08 19:32:00,1.0,2.0,3.0,300.0,400.0,10.0,101325.0',
                '"2025-09-08 19:32:00.05","4.0",2.0,3.0,300.0,400.0,10.0,101325.0',
                '2025-09-08 19:32:00.10,7.0,2.0,3.0,300.0,400.0,10.0,101325.0']
        period = self.read_one(rows)
        self.assertEqual(list(period['u']), [1.0, 4.0, 7.0])

    def test_a_quoted_field_holding_the_delimiter_is_still_one_field(self):
        # The naive split would shift every column after it by one; the reader takes it.
        rows = ['2025-09-08 19:32:00,"a,b",1.0,2.0,3.0,300.0,400.0,10.0,101325.0',
                '"2025-09-08 19:32:00.05","c,d",4.0,2.0,3.0,300.0,400.0,10.0,101325.0']
        period = self.read_one(rows, header=self.NOTE)
        self.assertEqual(list(period['u']), [1.0, 4.0])

    def test_a_doubled_quote_inside_a_quoted_field(self):
        rows = ['2025-09-08 19:32:00,"say, ""hi""",1.0,2.0,3.0,300.0,400.0,10.0,101325.0']
        period = self.read_one(rows, header=self.NOTE)
        self.assertEqual(list(period['u']), [1.0])

    def test_a_quoted_field_holding_a_newline_is_one_row(self):
        # The record runs past the end of its first line, and the line count runs with it:
        # the row after it is refused by the line it is actually on.
        rows = ['2025-09-08 19:32:00,"two\nlines",1.0,2.0,3.0,300.0,400.0,10.0,101325.0',
                '2025-09-08 19:32:00.05,plain,4.0,2.0,3.0,300.0,400.0,10.0,101325.0']
        period = self.read_one(rows, header=self.NOTE)
        self.assertEqual(list(period['u']), [1.0, 4.0])

    def test_a_line_after_a_record_holding_a_newline_is_refused_by_its_own_line(self):
        rows = ['2025-09-08 19:32:00,"two\nlines",1.0,2.0,3.0,300.0,400.0,10.0,101325.0',
                '2025-09-08 19:32:00.05,plain,4.0,2.0,3.0,300.0,400.0,10.0,101325.0',
                '2025-09-08 19:32:00.10,plain,ten,2.0,3.0,300.0,400.0,10.0,101325.0']
        self.write_csv(rows, header=self.NOTE)
        with self.assertRaises(ReadError) as caught:
            list(read.periods(self.load()))
        self.assertIn('line 5', str(caught.exception))

    def test_every_shape_is_split_the_way_csv_reader_splits_it(self):
        # The equivalence the fast path rests on, asserted against csv.reader itself.
        lines = ['a,b,c',
                 '"a","b","c"',
                 'plain,"quoted",plain',
                 '"holds,the,delimiter",b',
                 '"holds ""a"" quote",b',
                 'an "inner" quote,b',
                 '"never closes,b',
                 '"",a',
                 '"a",',
                 ' "a",b',
                 'a,"",b',
                 '',
                 'trailing,fields,']
        path = os.path.join(self.directory, 'shapes.csv')
        with open(path, 'w', newline='') as handle:
            handle.write('\n'.join(lines) + '\n')
        cfg = self.load()
        with open(path, newline='') as handle:
            expected = list(csv.reader(handle))
        self.assertEqual([row for _lineno, row in read._rows(path, cfg)], expected)


class NaValueTest(ReadTestCase):
    def test_every_default_na_token_including_the_empty_field(self):
        rows = ['2025-09-08 19:32:00,-9999,NAN,nan,NaN,,1.0,101325.0']
        period = self.read_one(rows)
        for name in ('u', 'v', 'w', 'ts', 'co2'):
            self.assertTrue(math.isnan(period[name][0]), name)
        self.assertEqual(period['h2o'][0], 1.0 * 1e-3)     # ppt -> mol mol-1, still read
        self.assertEqual(period['p_air'][0], 101325.0)

    def test_a_declared_na_token_replaces_the_defaults(self):
        rows = ['2025-09-08 19:32:00,-999.9,2.0,3.0,300.0,400.0,10.0,101325.0']
        period = self.read_one(rows, {'files': {'na_values': '-999.9'}})
        self.assertTrue(math.isnan(period['u'][0]))

    def test_an_undeclared_sentinel_is_read_as_the_number_it_is(self):
        # -999.9 is a number, so nothing here can tell it is a sentinel: a site that uses
        # one has to write it into na_values or it enters the statistics.
        rows = ['2025-09-08 19:32:00,-999.9,2.0,3.0,300.0,400.0,10.0,101325.0']
        period = self.read_one(rows)
        self.assertEqual(period['u'][0], -999.9)


class ConstantPressureTest(ReadTestCase):
    def test_no_pressure_column_gives_the_declared_constant(self):
        rows = ['2025-09-08 19:32:00,1.0,2.0,3.0,300.0,400.0,10.0,101325.0',
                '2025-09-08 19:32:00.05,1.0,2.0,3.0,300.0,400.0,10.0,101325.0']
        period = self.read_one(rows, {'variables': {'p_air': ''},
                                      'site': {'pressure_pa': '99500'}})
        self.assertEqual(list(period['p_air']), [99500.0, 99500.0])


class MinSamplesTest(ReadTestCase):
    def test_a_short_period_is_skipped_and_logged(self):
        rows = ['2025-09-08 19:32:00,1.0,2.0,3.0,300.0,400.0,10.0,101325.0',
                '2025-09-08 20:02:00,1.0,2.0,3.0,300.0,400.0,10.0,101325.0',
                '2025-09-08 20:02:00.05,1.0,2.0,3.0,300.0,400.0,10.0,101325.0']
        self.write_csv(rows)
        with self.assertLogs('miniflux.read', level='WARNING'):
            found = list(read.periods(self.load({'period': {'min_samples': '2'}})))
        self.assertEqual([p['meta']['period_end'] for p in found],
                         [datetime(2025, 9, 8, 20, 30)])


class ManyFilesTest(ReadTestCase):
    def test_files_are_concatenated_in_sorted_order_and_periods_straddle_them(self):
        self.write_csv(['2025-09-08 19:32:00,1.0,2.0,3.0,300.0,400.0,10.0,101325.0'],
                       name='a.csv')
        self.write_csv(['2025-09-08 19:33:00,2.0,2.0,3.0,300.0,400.0,10.0,101325.0'],
                       name='b.csv')
        found = list(read.periods(self.load()))
        self.assertEqual(len(found), 1)
        self.assertEqual(list(found[0]['u']), [1.0, 2.0])


class RefusalTest(ReadTestCase):
    """Every ReadError of CONTRACT 7.2, each naming the file and the line."""

    GOOD = '2025-09-08 19:32:00,1.0,2.0,3.0,300.0,400.0,10.0,101325.0'

    def refuse(self, cfg):
        with self.assertRaises(ReadError) as caught:
            list(read.periods(cfg))
        return str(caught.exception)

    def test_an_empty_glob(self):
        message = self.refuse(self.load(pattern='nothing_here_*.csv'))
        self.assertIn('input_glob', message)

    def test_a_timestamp_of_neither_shape(self):
        self.write_csv([self.GOOD, '08/09/2025 19:32:00,1.0,2.0,3.0,300.0,400.0,10.0,1.0'])
        message = self.refuse(self.load())
        self.assertIn('line 3', message)
        self.assertIn('never infers', message)

    def test_a_header_missing_a_declared_column(self):
        self.write_csv([self.GOOD], header='TIMESTAMP,Ux,Uy,Uz,Ts,CO2,H2O,Pressure')
        message = self.refuse(self.load())
        self.assertIn("'Press'", message)
        self.assertIn('[variables] p_air', message)

    def test_a_header_missing_the_timestamp_column(self):
        self.write_csv([self.GOOD], header='TIME,Ux,Uy,Uz,Ts,CO2,H2O,Press')
        self.assertIn('[timestamp] column', self.refuse(self.load()))

    def test_a_row_with_too_few_fields(self):
        self.write_csv([self.GOOD, '2025-09-08 19:32:00.05,1.0,2.0,3.0'])
        message = self.refuse(self.load())
        self.assertIn('line 3', message)
        self.assertIn('4 field(s)', message)

    def test_a_non_numeric_token_in_a_data_column(self):
        self.write_csv([self.GOOD, '2025-09-08 19:32:00.05,1.0,2.0,3.0,300.0,400.0,ten,1.0'])
        message = self.refuse(self.load())
        self.assertIn('line 3', message)
        self.assertIn("'H2O'", message)

    def test_a_file_with_no_header_line(self):
        path = os.path.join(self.directory, 'empty.csv')
        open(path, 'w').close()
        self.assertIn('header_line', self.refuse(self.load()))

    def test_a_file_that_cannot_be_decoded(self):
        # A latin-1 degree sign in a preamble line read.py never looks at. An input
        # problem is a ReadError naming the file (cli exit 3), not a UnicodeDecodeError.
        path = os.path.join(self.directory, 'latin.csv')
        with open(path, 'wb') as handle:
            handle.write(b'"units","m/s","\xb0C"\n')
            handle.write((HEADER + '\n').encode('ascii'))
            handle.write((self.GOOD + '\n').encode('ascii'))
        message = self.refuse(self.load(sections={'files': {'header_line': '2',
                                                           'first_data_line': '3'}}))
        self.assertIn('latin.csv', message)
        self.assertIn('encoding', message)

    def test_a_glob_that_matches_a_directory(self):
        os.mkdir(os.path.join(self.directory, 'sub.csv'))
        self.write_csv([self.GOOD])
        message = self.refuse(self.load())
        self.assertIn('sub.csv', message)

    def test_an_unknown_encoding_name(self):
        self.write_csv([self.GOOD])
        message = self.refuse(self.load(sections={'files': {'encoding': 'utf-47'}}))
        self.assertIn('utf-47', message)

    def test_a_number_written_with_a_digit_separator(self):
        # float('1_0') is 10.0 under PEP 515. A token no logger writes is corruption,
        # and reading it as a number is exactly the "looks right and is not" case.
        self.write_csv([self.GOOD,
                        '2025-09-08 19:32:00.05,1.0,2.0,1_0,300.0,400.0,10.0,1.0'])
        message = self.refuse(self.load())
        self.assertIn('line 3', message)
        self.assertIn("'1_0'", message)


class ByteOrderMarkTest(ReadTestCase):
    def test_a_utf8_bom_does_not_hide_the_first_column(self):
        # Every Windows tool that exports a CSV writes the BOM; it lands on the first
        # header cell and the run dies accusing the file of having no TIMESTAMP column.
        path = os.path.join(self.directory, 'bom.csv')
        with open(path, 'wb') as handle:
            handle.write(b'\xef\xbb\xbf')
            handle.write((HEADER + '\n').encode('ascii'))
            handle.write((RefusalTest.GOOD + '\n').encode('ascii'))
        found = list(read.periods(self.load()))
        self.assertEqual(len(found), 1)
        self.assertEqual(list(found[0]['u']), [1.0])


if __name__ == '__main__':
    unittest.main()
