"""Every refusal of CONTRACT 5.2, the two mass-density rules, and the shipped defaults.

The shipped `examples/miniflux.ini` is the contract's section 6 verbatim, so asserting that
it loads to the documented values is the same as asserting the documented values are real.
"""

import contextlib
import logging
import os
import shutil
import tempfile
import unittest

from miniflux import config, kernels
from miniflux.config import describe, load
from miniflux.constants import CANOPY_DISPLACEMENT_RATIO, MCO2, MV, T0
from miniflux.errors import ConfigError

EXAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       os.pardir, 'examples', 'miniflux.ini')

# A file that loads: the sonic and gas columns are named and exactly one pressure source is.
VALID = {
    'variables': {'u': 'Ux', 'v': 'Uy', 'w': 'Uz', 'ts': 'Ts',
                  'co2': 'CO2', 'h2o': 'H2O', 'p_air': 'Press'},
}


def setUpModule():
    # _derive_site warns on the default (canopy-less) site; the test output is not the place.
    logging.disable(logging.CRITICAL)


def tearDownModule():
    logging.disable(logging.NOTSET)


@contextlib.contextmanager
def _records(logger):
    """Collect the records `logger` emits, with this module's blanket silence lifted."""
    seen = []
    handler = logging.Handler()
    handler.emit = seen.append
    logging.disable(logging.NOTSET)
    logger.addHandler(handler)
    try:
        yield seen
    finally:
        logger.removeHandler(handler)
        logging.disable(logging.CRITICAL)


class ConfigTestCase(unittest.TestCase):
    """Writes .ini files into a temporary directory and loads them."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def write(self, sections):
        """sections: {section: {key: raw string}}. '' writes the key with an empty value."""
        lines = []
        for name in sections:
            lines.append('[%s]' % name)
            for key, value in sections[name].items():
                lines.append('%s = %s' % (key, value))
            lines.append('')
        path = os.path.join(self.dir, 'miniflux.ini')
        handle = open(path, 'w')
        handle.write('\n'.join(lines))
        handle.close()
        return path

    def load(self, extra=None, base=VALID):
        """Load the valid base config, with `extra` merged in section by section."""
        sections = dict((name, dict(keys)) for name, keys in base.items())
        for name in (extra or {}):
            sections.setdefault(name, {}).update(extra[name])
        return load(self.write(sections))

    def refuse(self, extra, *fragments):
        """Assert the merged config raises ConfigError, naming each fragment."""
        try:
            self.load(extra)
        except ConfigError as exc:
            message = str(exc)
            for fragment in fragments:
                self.assertIn(fragment, message,
                              'message %r does not name %r' % (message, fragment))
            return message
        self.fail('expected a ConfigError for %r' % (extra,))


class TestShippedDefaults(ConfigTestCase):
    """examples/miniflux.ini is the contract's file; every default must round-trip."""

    def setUp(self):
        ConfigTestCase.setUp(self)
        self.cfg = load(EXAMPLE)

    def test_files(self):
        self.assertEqual(self.cfg.files.input_glob, './*.dat')
        self.assertEqual(self.cfg.files.output_csv, './miniflux.csv')
        self.assertEqual(self.cfg.files.header_line, 1)
        self.assertEqual(self.cfg.files.first_data_line, 2)
        self.assertEqual(self.cfg.files.delimiter, ',')
        self.assertEqual(self.cfg.files.quotechar, '"')
        self.assertEqual(self.cfg.files.encoding, 'utf-8-sig')

    def test_na_values_keeps_the_trailing_empty_token(self):
        # The trailing comma of the default is what makes an empty field read as NaN.
        self.assertEqual(self.cfg.files.na_values, ('-9999', 'NAN', 'NaN', 'nan', ''))

    def test_runtime(self):
        self.assertEqual(self.cfg.runtime.use_numpy, 'auto')
        self.assertEqual(self.cfg.runtime.log_level, 'INFO')

    def test_site(self):
        self.assertEqual(self.cfg.site.site_id, 'XX-Xxx')
        self.assertEqual(self.cfg.site.latitude, 0.0)
        self.assertEqual(self.cfg.site.measurement_height, 3.0)
        self.assertEqual(self.cfg.site.canopy_height, 0.0)
        self.assertIsNone(self.cfg.site.displacement_height)
        self.assertEqual(self.cfg.site.north_offset, 0.0)
        self.assertIsNone(self.cfg.site.pressure_pa)
        self.assertEqual(self.cfg.site.displacement, 0.0)
        self.assertEqual(self.cfg.site.pressure_source, 'column')

    def test_period(self):
        self.assertEqual(self.cfg.period.averaging_minutes, 30.0)
        self.assertEqual(self.cfg.period.closed, 'right')
        self.assertEqual(self.cfg.period.acquisition_frequency, 20.0)
        self.assertEqual(self.cfg.period.min_samples, 0)
        self.assertEqual(self.cfg.period.seconds, 1800.0)
        self.assertEqual(self.cfg.period.dt, 0.05)

    def test_timestamp_and_variables(self):
        self.assertEqual(self.cfg.timestamp.column, 'TIMESTAMP')
        self.assertEqual(self.cfg.timestamp.format, 'iso')
        self.assertEqual(self.cfg.variables.u, 'Ux')
        self.assertEqual(self.cfg.variables.v, 'Uy')
        self.assertEqual(self.cfg.variables.w, 'Uz')
        self.assertEqual(self.cfg.variables.ts, 'Ts')
        self.assertEqual(self.cfg.variables.co2, 'CO2')
        self.assertEqual(self.cfg.variables.h2o, 'H2O')
        self.assertIsNone(self.cfg.variables.ta)      # empty means absent, not ''
        self.assertEqual(self.cfg.variables.p_air, 'Press')
        self.assertIsNone(self.cfg.variables.t_cell)
        self.assertIsNone(self.cfg.variables.p_cell)

    def test_units_and_gases(self):
        self.assertEqual(self.cfg.units.ts, 'K')
        self.assertEqual(self.cfg.units.ta, 'K')
        self.assertEqual(self.cfg.units.p_air, 'Pa')
        self.assertEqual(self.cfg.units.co2, 'ppm')
        self.assertEqual(self.cfg.units.h2o, 'ppt')
        self.assertEqual(self.cfg.gases.co2_measure_type, 'mixing_ratio')
        self.assertEqual(self.cfg.gases.h2o_measure_type, 'mixing_ratio')
        self.assertEqual(self.cfg.gases.analyser_path, 'open')
        self.assertEqual(self.cfg.gases.measure_type,
                         {'co2': 'mixing_ratio', 'h2o': 'mixing_ratio'})
        self.assertEqual(self.cfg.gases.reported,
                         {'co2': 'mixing_ratio', 'h2o': 'mixing_ratio'})
        self.assertEqual(self.cfg.gases.convert_cell, ())
        # The cell keys default to the LI-7200's own units, not to the canonical K/Pa the
        # ambient ones default to: there is no safe default for a cell temperature, so the
        # shipped one matches the instrument the key exists for.
        self.assertEqual(self.cfg.units.t_cell, 'degC')
        self.assertEqual(self.cfg.units.p_cell, 'kPa')
        self.assertEqual(self.cfg.units.convert, {
            'u': (1.0, 0.0), 'v': (1.0, 0.0), 'w': (1.0, 0.0),
            'ts': (1.0, 0.0), 'ta': (1.0, 0.0), 'p_air': (1.0, 0.0),
            't_cell': (1.0, T0), 'p_cell': (1000.0, 0.0),
            'co2': (1e-6, 0.0), 'h2o': (1e-3, 0.0)})

    def test_spectral_ships_off_and_declares_no_time_constant(self):
        # The shipped default has to be inert: an enabled correction with a guessed tau
        # would put a plausible few per cent on every flux of every user who switched the
        # section on without reading it.
        self.assertIs(self.cfg.spectral.enabled, False)
        self.assertIsNone(self.cfg.spectral.co2_tau_s)
        self.assertIsNone(self.cfg.spectral.h2o_tau_s)

    def test_steps(self):
        self.assertIs(self.cfg.despike.enabled, True)
        self.assertEqual(self.cfg.despike.variables, ('u', 'v', 'w', 'ts', 'co2', 'h2o'))
        self.assertEqual(self.cfg.despike.q, 7.0)
        self.assertEqual(self.cfg.rotate.method, 'double')
        self.assertEqual(self.cfg.lag.method, 'covmax_default')
        self.assertEqual(self.cfg.lag.scalars, ('co2', 'h2o'))
        self.assertEqual(self.cfg.lag.windows, {'co2': (0.0, -2.0, 2.0),
                                                'h2o': (0.0, -2.0, 2.0)})
        self.assertEqual(self.cfg.detrend.method, 'block')
        self.assertEqual(self.cfg.detrend.variables, ('u', 'v', 'w', 'ts', 'co2', 'h2o'))
        self.assertEqual(self.cfg.wpl.enabled, 'auto')
        self.assertEqual(self.cfg.qc.steady_state_pair, ('w', 'co2'))
        self.assertIs(self.cfg.qc.itc, True)
        self.assertEqual(self.cfg.output.na_value, '-9999')
        self.assertEqual(self.cfg.output.float_format, '%.6g')

    def test_an_omitted_key_resolves_to_the_shipped_default(self):
        # A file with nothing in it must be the shipped file: the defaults in the code and
        # the defaults in the .ini are the same strings.
        bare = load(self.write({}))
        self.assertEqual(describe(bare), describe(self.cfg))

    def test_describe_prints_every_resolved_value_including_derived(self):
        text = describe(self.cfg)
        for section, _keys in config.SPEC:
            for key in vars(getattr(self.cfg, section)):
                self.assertIn('%s.%s = ' % (section, key), text)
        self.assertIn('period.seconds = 1800.0', text)
        self.assertIn('period.dt = 0.05', text)
        self.assertIn('site.displacement = 0.0', text)
        self.assertIn('site.pressure_source = column', text)   # where the pressure came from
        self.assertIn('variables.ta = <not set>', text)        # unset is spelled out


class TestRefusal1Enums(ConfigTestCase):
    """5.2.1 -- an enumerated key outside its enumeration."""

    def test_every_enumerated_key(self):
        cases = [
            ('period', 'closed', 'middle'),
            ('timestamp', 'format', 'auto'),
            ('detrend', 'method', 'spline'),
            ('rotate', 'method', 'triple'),
            ('lag', 'method', 'xcorr'),
            ('wpl', 'enabled', 'maybe'),
            ('runtime', 'use_numpy', 'yes'),        # beyond 5.2, same principle
            ('runtime', 'log_level', 'LOUD'),
        ]
        for section, key, value in cases:
            message = self.refuse({section: {key: value}}, '[%s]' % section, key, value)
            self.assertIn('write one of', message)

    def test_an_enum_is_case_insensitive_and_normalised(self):
        cfg = self.load({'period': {'closed': 'LEFT'}, 'rotate': {'method': 'None'}})
        self.assertEqual(cfg.period.closed, 'left')
        self.assertEqual(cfg.rotate.method, 'none')


class TestRefusal2MeasureType(ConfigTestCase):
    """5.2.2 -- a wet mole fraction is not supported, and is not silently treated as dry."""

    def test_mole_fraction_is_refused_with_the_remedy(self):
        for gas in ('co2', 'h2o'):
            message = self.refuse({'gases': {gas + '_measure_type': 'mole_fraction'}},
                                  '[gases]', gas + '_measure_type', 'mole_fraction')
            self.assertIn('mixing_ratio', message)
            self.assertIn('molar_density', message)
            self.assertIn('upstream', message)

    def test_any_other_spelling_is_refused_too(self):
        self.refuse({'gases': {'co2_measure_type': 'density'}}, '[gases]', 'co2_measure_type')


class TestRefusal3UnitsAgainstMeasureType(ConfigTestCase):
    """5.2.3 -- the declared unit and the declared measure type must describe one quantity."""

    def test_mixing_ratio_unit_under_molar_density(self):
        self.refuse({'units': {'co2': 'ppm'},
                     'gases': {'co2_measure_type': 'molar_density'}},
                    '[units]', 'co2', 'ppm', 'molar_density')

    def test_density_unit_under_mixing_ratio(self):
        self.refuse({'units': {'h2o': 'mol/m3'},
                     'gases': {'h2o_measure_type': 'mixing_ratio'}},
                    '[units]', 'h2o', 'mol/m3', 'mixing_ratio')

    def test_an_unknown_unit_lists_the_legal_ones(self):
        message = self.refuse({'units': {'co2': 'ppb'}}, '[units]', 'co2', 'ppb')
        self.assertIn('mg/m3', message)
        message = self.refuse({'units': {'ts': 'degF'}}, '[units]', 'ts', 'degF')
        self.assertIn('degC', message)


class TestRefusals4And5Pressure(ConfigTestCase):
    """5.2.4 / 5.2.5 -- exactly one pressure source. miniflux never invents a pressure."""

    def test_both_empty_is_refused(self):
        message = self.refuse({'variables': {'p_air': ''}, 'site': {'pressure_pa': ''}},
                              '[variables] p_air', '[site] pressure_pa')
        self.assertIn('never', message)

    def test_both_set_says_which_one_to_delete(self):
        message = self.refuse({'variables': {'p_air': 'Press'},
                               'site': {'pressure_pa': '101325'}},
                              '[variables] p_air', '[site] pressure_pa')
        self.assertIn('Delete', message)

    def test_the_constant_alone_is_accepted(self):
        cfg = self.load({'variables': {'p_air': ''}, 'site': {'pressure_pa': '99000'}})
        self.assertEqual(cfg.site.pressure_pa, 99000.0)
        self.assertEqual(cfg.site.pressure_source, 'constant')
        self.assertIn('site.pressure_source = constant', describe(cfg))

    def test_empty_is_not_zero(self):
        # An empty pressure_pa means "no constant" (refusal 4); a written 0 means a
        # pressure of zero, which is a different mistake with a different message.
        empty = self.refuse({'variables': {'p_air': ''}, 'site': {'pressure_pa': ''}})
        zero = self.refuse({'variables': {'p_air': ''}, 'site': {'pressure_pa': '0'}},
                           '[site]', 'pressure_pa', '> 0 Pa')
        self.assertNotEqual(empty, zero)


class TestRefusal6RequiredColumns(ConfigTestCase):
    """5.2.6 -- the six series every flux needs must name their column."""

    def test_each_required_variable(self):
        for name in ('u', 'v', 'w', 'ts', 'co2', 'h2o'):
            self.refuse({'variables': {name: ''}}, '[variables]', name, 'required')

    def test_ta_may_be_absent(self):
        self.assertIsNone(self.load({'variables': {'ta': ''}}).variables.ta)


class TestEmptyIsNotOmitted(ConfigTestCase):
    """An omitted key takes its default; a key written empty says "not set"."""

    def test_omitted_takes_the_default(self):
        cfg = self.load({'period': {'closed': 'left'}})     # averaging_minutes not written
        self.assertEqual(cfg.period.averaging_minutes, 30.0)

    def test_written_empty_is_refused_where_unset_means_nothing(self):
        message = self.refuse({'period': {'averaging_minutes': ''}},
                              '[period]', 'averaging_minutes', 'no value')
        self.assertIn('delete the line', message)
        self.refuse({'files': {'delimiter': ''}}, '[files]', 'delimiter', 'no value')
        self.refuse({'lag': {'co2_max_s': ''}}, '[lag]', 'co2_max_s', 'no value')

    def test_written_empty_is_the_way_to_unset_an_optional_key(self):
        cfg = self.load({'site': {'displacement_height': '', 'canopy_height': '2.0'},
                         'variables': {'ta': ''}})
        self.assertIsNone(cfg.site.displacement_height)
        self.assertIsNone(cfg.variables.ta)

    def test_an_empty_column_name_does_not_fall_back_to_the_default_column(self):
        # `p_air =` means "there is no pressure column", not "use the default header".
        cfg = self.load({'variables': {'p_air': ''}, 'site': {'pressure_pa': '99000'}})
        self.assertIsNone(cfg.variables.p_air)


class TestRefusal7Positive(ConfigTestCase):
    """5.2.7 -- the three quantities that are meaningless at or below zero."""

    def test_acquisition_frequency(self):
        self.refuse({'period': {'acquisition_frequency': '0'}},
                    '[period]', 'acquisition_frequency', '> 0')

    def test_averaging_minutes(self):
        self.refuse({'period': {'averaging_minutes': '-1'}},
                    '[period]', 'averaging_minutes', '> 0')

    def test_despike_q(self):
        self.refuse({'despike': {'q': '0'}}, '[despike]', 'q', '> 0')


class TestRefusal8LagWindow(ConfigTestCase):
    """5.2.8 -- an inverted search window."""

    def test_min_above_max(self):
        self.refuse({'lag': {'co2_min_s': '1.0', 'co2_max_s': '-1.0'}},
                    '[lag]', 'co2_min_s', 'swap')

    def test_equal_bounds_are_a_one_lag_window_and_are_legal(self):
        cfg = self.load({'lag': {'h2o_min_s': '0.3', 'h2o_max_s': '0.3'}})
        self.assertEqual(cfg.lag.windows['h2o'], (0.0, 0.3, 0.3))

    def test_a_scalar_without_window_keys_takes_the_documented_default(self):
        cfg = self.load({'lag': {'scalars': 'co2,h2o,ts'}})
        self.assertEqual(cfg.lag.windows['ts'], (0.0, -2.0, 2.0))
        self.assertEqual(cfg.lag.ts_min_s, -2.0)


class TestRefusal9CanonicalNames(ConfigTestCase):
    """5.2.9 -- a name that is not in the vocabulary of CONTRACT section 2."""

    def test_every_list_of_names(self):
        cases = [('despike', 'variables', 'u,foo'),
                 ('detrend', 'variables', 'bar'),
                 ('lag', 'scalars', 'baz'),
                 ('qc', 'steady_state_pair', 'w,quux')]
        for section, key, value in cases:
            message = self.refuse({section: {key: value}}, '[%s]' % section, key)
            self.assertIn('canonical', message)

    def test_the_steady_state_pair_is_a_pair(self):
        self.refuse({'qc': {'steady_state_pair': 'w'}},
                    '[qc]', 'steady_state_pair', 'exactly two')
        self.refuse({'qc': {'steady_state_pair': 'w,co2,h2o'}},
                    '[qc]', 'steady_state_pair', 'exactly two')

    def test_a_name_written_twice_is_refused(self):
        # A copy-paste typo, and every one of these lists is applied once per entry: a
        # second despiking pass runs the MAD test on already-despiked data and overwrites
        # the spike count with the second pass's zero, and a second linear detrend stores
        # the post-detrend mean (~0) for a 293 K series.
        for section, key, value in (('despike', 'variables', 'u,ts,ts'),
                                    ('detrend', 'variables', 'co2,co2'),
                                    ('lag', 'scalars', 'co2,h2o,co2')):
            message = self.refuse({section: {key: value}}, '[%s]' % section, key)
            self.assertIn('twice', message)

    def test_a_thermodynamic_mean_cannot_be_detrended(self):
        # detrend writes meta['<var>_mean'] for whatever it is given, and ta_mean is
        # flux.thermodynamics' resolved air temperature (measured where finite, else
        # sonic-derived), which flux.assemble reads for H and wpl for every term.
        for name in ('ta', 'p_air'):
            message = self.refuse({'detrend': {'variables': 'u,v,w,ts,co2,h2o,' + name}},
                                  '[detrend]', 'variables')
            self.assertIn(name, message)
            self.assertIn('thermodynamics', message)

    def test_a_thermodynamic_mean_may_still_be_despiked(self):
        # The refusal is about the meta key, not about the variable: despiking ta is a
        # perfectly ordinary thing to ask for, and it writes n_spike_ta, not ta_mean.
        cfg = self.load({'despike': {'variables': 'u,v,w,ts,co2,h2o,ta,p_air'}})
        self.assertIn('ta', cfg.despike.variables)


class TestRefusal10Numpy(ConfigTestCase):
    """5.2.10 -- `use_numpy = on` is a demand, and an unmet demand is a refusal."""

    def setUp(self):
        ConfigTestCase.setUp(self)
        # The refusal is about the numpy the *kernels* got, which is the only one that
        # decides anything: a package that is importable in name and raises on import is
        # exactly the case the try/except in kernels.py exists for.
        self.available = kernels.NUMPY_AVAILABLE
        self.have = kernels.HAVE_NUMPY

    def tearDown(self):
        kernels.NUMPY_AVAILABLE = self.available
        kernels.set_numpy(self.have)
        ConfigTestCase.tearDown(self)

    def test_on_without_numpy_is_refused(self):
        kernels.NUMPY_AVAILABLE = False
        self.refuse({'runtime': {'use_numpy': 'on'}},
                    '[runtime]', 'use_numpy', 'install numpy')

    def test_auto_without_numpy_is_fine(self):
        kernels.NUMPY_AVAILABLE = False
        cfg = self.load({'runtime': {'use_numpy': 'auto'}})
        self.assertIs(cfg.runtime.numpy_enabled, False)

    def test_off_is_always_fine(self):
        cfg = self.load({'runtime': {'use_numpy': 'off'}})
        self.assertIs(cfg.runtime.numpy_enabled, False)

    def test_the_recorded_value_is_the_path_the_run_actually_takes(self):
        # describe() puts runtime.numpy_enabled in the run's log. It is a provenance
        # record, so it has to be the resolved kernels.HAVE_NUMPY and not a second guess
        # at it -- one that can disagree in both directions.
        for spelling in ('auto', 'off'):
            cfg = self.load({'runtime': {'use_numpy': spelling}})
            self.assertIs(cfg.runtime.numpy_enabled, kernels.HAVE_NUMPY)
        kernels.NUMPY_AVAILABLE = False
        cfg = self.load({'runtime': {'use_numpy': 'auto'}})
        self.assertIs(cfg.runtime.numpy_enabled, kernels.HAVE_NUMPY)

    def test_auto_on_a_machine_without_numpy_does_not_warn(self):
        # `auto` is not a request, so there is nothing to report as unmet.
        kernels.NUMPY_AVAILABLE = False
        logger = logging.getLogger('miniflux')
        with _records(logger) as seen:
            self.load({'runtime': {'use_numpy': 'auto'}})
        self.assertEqual([r.getMessage() for r in seen
                          if r.levelno >= logging.WARNING and 'numpy' in r.getMessage()],
                         [])


CLOSED_DENSITY = {'gases': {'analyser_path': 'closed',
                            'co2_measure_type': 'molar_density',
                            'h2o_measure_type': 'molar_density'},
                  'units': {'co2': 'mmol/m3', 'h2o': 'mmol/m3'}}


class TestRefusal12CellState(ConfigTestCase):
    """5.2.12 -- a cell density is unusable without the cell's own state."""

    def cells(self, **names):
        extra = dict((name, dict(keys)) for name, keys in CLOSED_DENSITY.items())
        extra.setdefault('variables', {}).update(names)
        return extra

    def test_a_closed_path_density_without_a_cell_state_is_refused(self):
        message = self.refuse(self.cells(), '[variables]', 't_cell')
        self.assertIn('ANALYSER CELL', message)
        # The message has to say what to do instead, and the exact way out is the column
        # the analyser already writes.
        self.assertIn('CO2_DRY', message)

    def test_half_a_cell_state_is_still_a_refusal(self):
        self.refuse(self.cells(t_cell='T_CELL'), '[variables]', 'p_cell')
        self.refuse(self.cells(p_cell='PRESS_CELL'), '[variables]', 't_cell')

    def test_the_ambient_pressure_is_not_a_substitute_and_the_message_says_so(self):
        message = self.refuse(self.cells(), '[variables]')
        self.assertIn('not substituted', message)

    def test_a_declared_cell_state_loads(self):
        cfg = self.load(self.cells(t_cell='T_CELL', p_cell='PRESS_CELL'))
        self.assertEqual(cfg.gases.convert_cell, ('co2', 'h2o'))
        self.assertEqual(cfg.gases.measure_type,
                         {'co2': 'mixing_ratio', 'h2o': 'mixing_ratio'})
        self.assertEqual(cfg.gases.reported,
                         {'co2': 'molar_density', 'h2o': 'molar_density'})

    def test_an_open_path_density_needs_no_cell_and_stays_owed_wpl(self):
        extra = dict((name, dict(keys)) for name, keys in CLOSED_DENSITY.items())
        extra['gases']['analyser_path'] = 'open'
        cfg = self.load(extra)
        self.assertEqual(cfg.gases.convert_cell, ())
        self.assertEqual(cfg.gases.measure_type,
                         {'co2': 'molar_density', 'h2o': 'molar_density'})

    def test_a_closed_path_dry_mixing_ratio_needs_no_cell_state(self):
        # The recommended declaration: exact, and nothing to convert.
        cfg = self.load({'gases': {'analyser_path': 'closed'}})
        self.assertEqual(cfg.gases.convert_cell, ())
        self.assertEqual(cfg.gases.measure_type,
                         {'co2': 'mixing_ratio', 'h2o': 'mixing_ratio'})

    def test_one_gas_may_be_a_cell_density_while_the_other_is_a_dry_ratio(self):
        # An LI-7200 read as CO2_CONC + H2O_DRY. The water still supplies the dilution
        # factor, as a moist fraction built from the dry ratio.
        cfg = self.load({'gases': {'analyser_path': 'closed',
                                   'co2_measure_type': 'molar_density'},
                         'units': {'co2': 'mmol/m3'},
                         'variables': {'t_cell': 'T_CELL', 'p_cell': 'PRESS_CELL'}})
        self.assertEqual(cfg.gases.convert_cell, ('co2',))
        self.assertEqual(cfg.gases.measure_type,
                         {'co2': 'mixing_ratio', 'h2o': 'mixing_ratio'})

    def test_a_cell_state_nothing_needs_is_warned_about_and_not_refused(self):
        # Asymmetric on purpose: a missing cell state produces a wrong number, an unused
        # one produces two columns of reading (CONTRACT 19).
        logger = logging.getLogger('miniflux')
        with _records(logger) as seen:
            cfg = self.load({'variables': {'t_cell': 'T_CELL', 'p_cell': 'PRESS_CELL'}})
        self.assertEqual(cfg.gases.convert_cell, ())
        messages = [r.getMessage() for r in seen if r.levelno >= logging.WARNING]
        self.assertTrue(any('t_cell' in m and 'not used' in m for m in messages), messages)

    def test_an_unknown_analyser_path_is_refused(self):
        self.refuse({'gases': {'analyser_path': 'closed_path'}},
                    '[gases]', 'analyser_path', 'open | closed')

    def test_the_cell_units_are_checked_like_any_other_scalar(self):
        self.refuse({'units': {'t_cell': 'degF'}}, '[units]', 't_cell', 'unknown unit')
        self.refuse({'units': {'p_cell': 'bar'}}, '[units]', 'p_cell', 'unknown unit')

    def test_the_cell_state_is_not_a_canonical_variable(self):
        # It enters no covariance and is consumed before despiking, so naming it in one of
        # the per-variable lists would be a no-op that reads like a safeguard.
        for section, key in (('despike', 'variables'), ('detrend', 'variables'),
                             ('lag', 'scalars')):
            self.refuse({section: {key: 'u,t_cell'}}, '[%s]' % section, 't_cell',
                        'not a canonical variable')


class TestRefusal13SpectralTimeConstant(ConfigTestCase):
    """5.2.13 -- no time constant, no correction. miniflux does not guess a tube."""

    def test_enabled_without_a_time_constant_is_refused(self):
        message = self.refuse({'spectral': {'enabled': 'true'}},
                              '[spectral]', 'co2_tau_s', '<not set>')
        self.assertIn('ships no default', message)

    def test_a_non_positive_time_constant_is_refused(self):
        self.refuse({'spectral': {'enabled': 'true', 'co2_tau_s': '0.0',
                                  'h2o_tau_s': '0.3'}}, '[spectral]', 'co2_tau_s')
        self.refuse({'spectral': {'enabled': 'true', 'co2_tau_s': '0.3',
                                  'h2o_tau_s': '-1'}}, '[spectral]', 'h2o_tau_s')

    def test_both_gases_are_required_not_just_one(self):
        self.refuse({'spectral': {'enabled': 'true', 'co2_tau_s': '0.3'}},
                    '[spectral]', 'h2o_tau_s')

    def test_a_disabled_section_needs_nothing(self):
        cfg = self.load({'spectral': {'enabled': 'false'}})
        self.assertIs(cfg.spectral.enabled, False)
        self.assertIsNone(cfg.spectral.co2_tau_s)

    def test_two_declared_time_constants_load(self):
        cfg = self.load({'spectral': {'enabled': 'true', 'co2_tau_s': '0.13',
                                      'h2o_tau_s': '0.40'}})
        self.assertIs(cfg.spectral.enabled, True)
        self.assertEqual((cfg.spectral.co2_tau_s, cfg.spectral.h2o_tau_s), (0.13, 0.40))


class TestUnitConversions(ConfigTestCase):
    """CONTRACT 6.1, including the two mass-density rows and the rule that gates them."""

    def convert(self, extra):
        return self.load(extra).units.convert

    def test_temperature_and_pressure(self):
        convert = self.convert({'units': {'ts': 'degC', 'ta': 'degC', 'p_air': 'hPa'}})
        self.assertEqual(convert['ts'], (1.0, T0))
        self.assertEqual(convert['ta'], (1.0, T0))
        self.assertEqual(convert['p_air'], (100.0, 0.0))
        self.assertEqual(self.convert({'units': {'p_air': 'kPa'}})['p_air'], (1000.0, 0.0))

    def test_wind_is_already_canonical(self):
        convert = self.convert({})
        for name in ('u', 'v', 'w'):
            self.assertEqual(convert[name], (1.0, 0.0))

    def test_mixing_ratio_units(self):
        convert = self.convert({'units': {'co2': 'umol/mol', 'h2o': 'mmol/mol'}})
        self.assertEqual(convert['co2'], (1e-6, 0.0))
        self.assertEqual(convert['h2o'], (1e-3, 0.0))

    def test_molar_density_units(self):
        convert = self.convert({'units': {'co2': 'mmol/m3', 'h2o': 'mol/m3'},
                                'gases': {'co2_measure_type': 'molar_density',
                                          'h2o_measure_type': 'molar_density'}})
        self.assertEqual(convert['co2'], (1e-3, 0.0))
        self.assertEqual(convert['h2o'], (1.0, 0.0))

    def test_mass_density_uses_the_molar_mass_of_the_gas_the_key_names(self):
        convert = self.convert({'units': {'co2': 'mg/m3', 'h2o': 'g/m3'},
                                'gases': {'co2_measure_type': 'molar_density',
                                          'h2o_measure_type': 'molar_density'}})
        self.assertEqual(convert['co2'], (1e-6 / MCO2, 0.0))
        self.assertEqual(convert['h2o'], (1e-3 / MV, 0.0))

    def test_the_other_mass_density_row(self):
        convert = self.convert({'units': {'co2': 'g/m3', 'h2o': 'mg/m3'},
                                'gases': {'co2_measure_type': 'molar_density',
                                          'h2o_measure_type': 'molar_density'}})
        self.assertEqual(convert['co2'], (1e-3 / MCO2, 0.0))
        self.assertEqual(convert['h2o'], (1e-6 / MV, 0.0))

    def test_a_mass_density_is_illegal_under_a_mixing_ratio(self):
        for gas, unit in (('co2', 'mg/m3'), ('h2o', 'g/m3')):
            message = self.refuse({'units': {gas: unit},
                                   'gases': {gas + '_measure_type': 'mixing_ratio'}},
                                  '[units]', gas, unit, 'molar_density')
            self.assertIn('mass density', message)

    def test_a_gas_may_declare_any_unit_of_its_own_measure_type(self):
        # 6.1 gates on measure_type, not on which gas: ppm for a dry H2O mixing ratio is
        # unusual but exact, so it is accepted.
        self.assertEqual(self.convert({'units': {'h2o': 'ppm'}})['h2o'], (1e-6, 0.0))


class TestSiteDisplacement(ConfigTestCase):
    """5.1 -- displacement is declared, else two thirds of the canopy, else zero and warned."""

    def test_declared_wins(self):
        self.assertEqual(self.load({'site': {'displacement_height': '1.5',
                                             'canopy_height': '10.0'}}).site.displacement, 1.5)

    def test_empty_falls_back_to_the_canopy(self):
        cfg = self.load({'site': {'displacement_height': '', 'canopy_height': '3.0'}})
        self.assertEqual(cfg.site.displacement, CANOPY_DISPLACEMENT_RATIO * 3.0)
        self.assertIsNone(cfg.site.displacement_height)

    def test_a_declared_zero_is_not_a_declaration(self):
        # "declared if non-empty and > 0" -- 0 is a value, but not one that can be used.
        cfg = self.load({'site': {'displacement_height': '0', 'canopy_height': '3.0'}})
        self.assertEqual(cfg.site.displacement_height, 0.0)
        self.assertEqual(cfg.site.displacement, CANOPY_DISPLACEMENT_RATIO * 3.0)

    def test_no_canopy_either_gives_zero(self):
        self.assertEqual(self.load({'site': {'canopy_height': '0.0'}}).site.displacement, 0.0)


class TestTypingAndSyntax(ConfigTestCase):
    """The coercions the rest of the program depends on, and the mistakes they catch."""

    def test_delimiter_escape_and_width(self):
        self.assertEqual(self.load({'files': {'delimiter': '\\t'}}).files.delimiter, '\t')
        self.refuse({'files': {'delimiter': 'ab'}}, '[files]', 'delimiter', 'one character')

    def test_a_semicolon_delimiter_needs_its_escape(self):
        # A bare ';' is an inline comment, so the key would arrive empty.
        self.refuse({'files': {'delimiter': ';'}}, '[files]', 'delimiter')
        self.assertEqual(self.load({'files': {'delimiter': '\\;'}}).files.delimiter, ';')

    def test_na_values_are_kept_as_written(self):
        cfg = self.load({'files': {'na_values': '-9999, NA ,'}})
        self.assertEqual(cfg.files.na_values, ('-9999', 'NA', ''))

    def test_a_non_numeric_number_is_refused(self):
        self.refuse({'site': {'latitude': 'north'}}, '[site]', 'latitude', 'number')
        self.refuse({'period': {'min_samples': '2.5'}}, '[period]', 'min_samples', 'whole')
        self.refuse({'site': {'latitude': 'nan'}}, '[site]', 'latitude', 'finite')

    def test_a_non_boolean_boolean_is_refused(self):
        self.refuse({'despike': {'enabled': 'sometimes'}},
                    '[despike]', 'enabled', 'true or false')

    def test_booleans_accept_the_usual_spellings(self):
        self.assertIs(self.load({'despike': {'enabled': 'FALSE'}}).despike.enabled, False)
        self.assertIs(self.load({'qc': {'itc': 'no'}}).qc.itc, False)

    def test_a_percent_in_float_format_survives_parsing(self):
        # ConfigParser's default interpolation would read '%.6g' as a broken reference.
        self.assertEqual(self.load({'output': {'float_format': '%.9e'}}).output.float_format,
                         '%.9e')

    def test_an_unusable_float_format_is_refused(self):
        self.refuse({'output': {'float_format': '%.6g %.6g'}}, '[output]', 'float_format')

    def test_line_numbers_are_one_based_and_ordered(self):
        self.refuse({'files': {'header_line': '0'}}, '[files]', 'header_line', '1-based')
        self.refuse({'files': {'header_line': '5', 'first_data_line': '5'}},
                    '[files]', 'first_data_line')

    def test_toa5_line_numbers_are_accepted(self):
        cfg = self.load({'files': {'header_line': '2', 'first_data_line': '5'}})
        self.assertEqual((cfg.files.header_line, cfg.files.first_data_line), (2, 5))

    def test_latitude_bounds(self):
        self.refuse({'site': {'latitude': '91'}}, '[site]', 'latitude', '-90')

    def test_an_unknown_key_or_section_is_refused(self):
        self.refuse({'period': {'avaraging_minutes': '10'}},
                    '[period]', 'avaraging_minutes', 'unknown key')
        message = self.refuse({'footprint': {'method': 'kl15'}}, '[footprint]')
        self.assertIn('unknown section', message)
        self.refuse({'spectral': {'model': 'horst'}},
                    '[spectral]', 'model', 'unknown key')

    def test_a_default_section_is_refused(self):
        try:
            self.load(base={'DEFAULT': {'method': 'none'}, 'variables': VALID['variables']})
        except ConfigError as exc:
            self.assertIn('[DEFAULT]', str(exc))
        else:
            self.fail('expected a ConfigError for a [DEFAULT] section')

    def test_a_missing_file_is_refused(self):
        try:
            load(os.path.join(self.dir, 'absent.ini'))
        except ConfigError as exc:
            self.assertIn('absent.ini', str(exc))
        else:
            self.fail('expected a ConfigError for a missing file')

    def test_a_malformed_file_is_refused(self):
        path = os.path.join(self.dir, 'broken.ini')
        handle = open(path, 'w')
        handle.write('u = Ux\n')          # no section header
        handle.close()
        self.assertRaises(ConfigError, load, path)


if __name__ == '__main__':
    unittest.main()
