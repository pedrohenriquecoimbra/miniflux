"""Read, type, validate and report the configuration (CONTRACT section 5).

One ``.ini`` file in, one ``Config`` out: a plain object whose attributes are one
``SimpleNamespace`` per section. Every value downstream is already typed; no other module
parses a string, and none of them guesses. Two rules run through the whole module:

* an empty value means **not set**, and is never the same as ``0`` or ``""``;
* miniflux refuses rather than inventing a number that would look right and be wrong.
"""

import configparser
import logging
import math
import os
from types import SimpleNamespace

from . import kernels
from .constants import CANOPY_DISPLACEMENT_RATIO, MCO2, MV, T0
from .errors import ConfigError

logger = logging.getLogger(__name__)

CANONICAL = ('u', 'v', 'w', 'ts', 'co2', 'h2o', 'ta', 'p_air')
REQUIRED_COLUMNS = ('u', 'v', 'w', 'ts', 'co2', 'h2o')
GASES = ('co2', 'h2o')
MEASURE_TYPES = ('mixing_ratio', 'molar_density')
MOLAR_MASS = {'co2': MCO2, 'h2o': MV}      # kg mol-1, for the mass-density units of 6.1

# CONTRACT 6.1, x_canonical = a * x_file + b.
SCALAR_UNITS = {
    'ts': {'K': (1.0, 0.0), 'degC': (1.0, T0)},
    'ta': {'K': (1.0, 0.0), 'degC': (1.0, T0)},
    'p_air': {'Pa': (1.0, 0.0), 'hPa': (100.0, 0.0), 'kPa': (1000.0, 0.0)},
}
MIXING_UNITS = {'ppm': 1e-6, 'umol/mol': 1e-6, 'ppt': 1e-3, 'mmol/mol': 1e-3}
MOLAR_DENSITY_UNITS = {'mmol/m3': 1e-3, 'mol/m3': 1.0}
MASS_DENSITY_UNITS = {'mg/m3': 1e-6, 'g/m3': 1e-3}   # a = factor / M(gas)

# One row per key: (key, kind, default). The default is the raw string of CONTRACT section 6
# and goes through the same coercion the file does, so a shipped default and an omitted key
# cannot drift apart. A default of '' means the key has no value until the user writes one.
# A tuple kind is the set of legal spellings.
SPEC = (
    ('files', (
        ('input_glob', 'str', './*.dat'),
        ('output_csv', 'str', './miniflux.csv'),
        ('header_line', 'int', '1'),
        ('first_data_line', 'int', '2'),
        ('delimiter', 'char', ','),
        ('quotechar', 'char', '"'),
        ('na_values', 'tokens', '-9999,NAN,NaN,nan,'),
        # utf-8-sig decodes plain utf-8 unchanged and strips a leading byte-order mark
        # when one is there. Without it the BOM every Windows exporter writes sticks to
        # the first header cell, and the run is refused for a column the file does have.
        ('encoding', 'str', 'utf-8-sig'),
    )),
    ('runtime', (
        ('use_numpy', ('auto', 'on', 'off'), 'auto'),
        ('log_level', ('DEBUG', 'INFO', 'WARNING', 'ERROR'), 'INFO'),
    )),
    ('site', (
        ('site_id', 'str', 'XX-Xxx'),
        ('latitude', 'float', '0.0'),
        ('measurement_height', 'float', '3.0'),
        ('canopy_height', 'float', '0.0'),
        ('displacement_height', 'float', ''),
        ('north_offset', 'float', '0.0'),
        ('pressure_pa', 'float', ''),
    )),
    ('period', (
        ('averaging_minutes', 'float', '30'),
        ('closed', ('right', 'left'), 'right'),
        ('acquisition_frequency', 'float', '20.0'),
        ('min_samples', 'int', '0'),
    )),
    ('timestamp', (
        ('column', 'str', 'TIMESTAMP'),
        ('format', ('iso', 'numeric'), 'iso'),
    )),
    ('variables', (
        ('u', 'str', 'Ux'), ('v', 'str', 'Uy'), ('w', 'str', 'Uz'),
        ('ts', 'str', 'Ts'), ('co2', 'str', 'CO2'), ('h2o', 'str', 'H2O'),
        ('ta', 'str', ''), ('p_air', 'str', 'Press'),
    )),
    ('units', (
        ('ts', 'str', 'K'), ('ta', 'str', 'K'), ('p_air', 'str', 'Pa'),
        ('co2', 'str', 'ppm'), ('h2o', 'str', 'ppt'),
    )),
    ('gases', (
        ('co2_measure_type', 'str', 'mixing_ratio'),
        ('h2o_measure_type', 'str', 'mixing_ratio'),
    )),
    ('despike', (
        ('enabled', 'bool', 'true'),
        ('variables', 'names', 'u,v,w,ts,co2,h2o'),
        ('q', 'float', '7.0'),
    )),
    ('rotate', (
        ('method', ('double', 'none'), 'double'),
    )),
    ('lag', (   # the per-scalar window keys are dynamic; see _lag_windows
        ('method', ('covmax_default', 'covmax', 'fixed', 'none'), 'covmax_default'),
        ('scalars', 'names', 'co2,h2o'),
    )),
    ('detrend', (
        ('method', ('block', 'linear'), 'block'),
        ('variables', 'names', 'u,v,w,ts,co2,h2o'),
    )),
    ('wpl', (
        ('enabled', ('auto', 'on', 'off'), 'auto'),
    )),
    ('qc', (
        ('steady_state_pair', 'names', 'w,co2'),
        ('itc', 'bool', 'true'),
    )),
    ('output', (
        ('na_value', 'str', '-9999'),
        ('float_format', 'str', '%.6g'),
    )),
)

# The only four keys that may be left empty, because for them "not set" says something the
# program acts on: no measured air temperature, no pressure column, no declared displacement,
# no constant pressure. Everywhere else an empty value is a slip of the keyboard.
OPTIONAL = (('site', 'displacement_height'), ('site', 'pressure_pa'),
            ('variables', 'ta'), ('variables', 'p_air'))

LAG_WINDOW_DEFAULTS = (('nominal_s', '0.0'), ('min_s', '-2.0'), ('max_s', '2.0'))
BOOLEANS = {'true': True, 'yes': True, 'on': True, '1': True,
            'false': False, 'no': False, 'off': False, '0': False}
# A bare ';' or '#' would start an inline comment and leave the key empty, so both have an
# escape: the backslash in front is what stops the comment scanner from seeing them.
ESCAPES = {'\\t': '\t', '\\n': '\n', '\\r': '\r',
           '\\;': ';', '\\#': '#', '\\\\': '\\'}


def load(path):
    """Parse, type and validate a miniflux .ini file.

    path: str. Returns the Config of CONTRACT section 5. Raises ConfigError, whose message
    always names the section and the key and says what to do about it.
    """
    if not os.path.isfile(path):
        raise ConfigError('config file not found: %s; pass the path to an .ini file' % path)
    # interpolation=None because the shipped [output] float_format is '%.6g', which the
    # default BasicInterpolation reads as a broken reference.
    parser = configparser.ConfigParser(interpolation=None,
                                       inline_comment_prefixes=('#', ';'))
    try:
        with open(path, encoding='utf-8') as handle:
            parser.read_file(handle, source=path)
    except configparser.Error as exc:
        raise ConfigError('%s is not a readable .ini file: %s' % (path, exc))

    _reject_unknown(parser)
    cfg = SimpleNamespace()
    for section, keys in SPEC:
        namespace = SimpleNamespace()
        for key, kind, default in keys:
            setattr(namespace, key, _value(parser, section, key, kind, default))
        setattr(cfg, section, namespace)

    _check_blanks(cfg)
    _derive_gases(cfg)
    _derive_units(cfg)
    _derive_period(cfg)
    _derive_site(cfg)
    cfg.lag.windows = _lag_windows(parser, cfg.lag)
    _validate(cfg)
    _resolve_numpy(cfg)
    return cfg


def describe(cfg):
    """Render every resolved value, derived ones included, one 'section.key = value' per line.

    The run's log therefore records what the run actually used -- including which of the two
    legal pressure sources it read, so no reader has to reconstruct it from the file.
    """
    lines = []
    for section, _keys in SPEC:
        for key, value in vars(getattr(cfg, section)).items():
            lines.append('%s.%s = %s' % (section, key, _show(value)))
    return '\n'.join(lines)


def _show(value):
    """Format one resolved value for describe(). Unset is spelled out, never blank."""
    if value is None:
        return '<not set>'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, dict):
        return '{%s}' % ', '.join('%s: %s' % (k, _show(v)) for k, v in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return '[%s]' % ', '.join(_show(v) for v in value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return value if value else "''"     # the empty na_values token has to be visible
    return str(value)


# --------------------------------------------------------------------------- parsing

def _fail(section, key, problem):
    raise ConfigError('[%s] %s: %s' % (section, key, problem))


def _raw(parser, section, key, default):
    """The raw string for a key, or None when it is not set.

    An omitted key takes the documented default. A key *written* with an empty value is
    something else entirely: the user has said "not set", which is why `p_air =` means
    there is no pressure column rather than the default column name. `_check_blanks`
    decides afterwards whether "not set" is a legal thing to say about that key.
    """
    text = parser.get(section, key) if parser.has_option(section, key) else default
    text = text.strip()
    return text if text else None


def _value(parser, section, key, kind, default):
    raw = _raw(parser, section, key, default)
    if raw is None:
        return None
    return _coerce(raw, kind, section, key)


def _coerce(raw, kind, section, key):
    """Turn one raw string into its typed value, or refuse with a message the user can act on."""
    if isinstance(kind, tuple):
        for allowed in kind:
            if raw.lower() == allowed.lower():
                return allowed
        _fail(section, key, 'unknown value %r; write one of %s' % (raw, ' | '.join(kind)))
    if kind == 'str':
        return raw
    if kind == 'int':
        try:
            return int(raw, 10)
        except ValueError:
            _fail(section, key, 'expected a whole number, got %r' % raw)
    if kind == 'float':
        try:
            number = float(raw)
        except ValueError:
            _fail(section, key, 'expected a number, got %r' % raw)
        if not math.isfinite(number):
            _fail(section, key, 'expected a finite number, got %r' % raw)
        return number
    if kind == 'bool':
        if raw.lower() not in BOOLEANS:
            _fail(section, key, 'expected true or false, got %r' % raw)
        return BOOLEANS[raw.lower()]
    if kind == 'char':
        text = ESCAPES.get(raw, raw)
        if len(text) != 1:
            _fail(section, key, 'expected exactly one character, got %r; write \\t for a '
                                'tab and \\; for a semicolon' % raw)
        return text
    if kind == 'names':
        return tuple(item.strip() for item in raw.split(',') if item.strip())
    if kind == 'tokens':
        # Empty items are kept: the trailing comma of the default is what makes an empty
        # field read as NaN.
        return tuple(item.strip() for item in raw.split(','))
    raise AssertionError('unknown kind %r' % kind)   # a typo in SPEC, not in a user file


def _reject_unknown(parser):
    """Refuse a section or key miniflux does not read, so a typo cannot silently do nothing."""
    known = dict((section, set(key for key, _k, _d in keys)) for section, keys in SPEC)
    for suffix, _default in LAG_WINDOW_DEFAULTS:
        known['lag'].update('%s_%s' % (name, suffix) for name in CANONICAL)
    if parser.defaults():
        raise ConfigError('[DEFAULT]: miniflux reads no [DEFAULT] section, because a key '
                          'there would silently answer for every section; move %s into the '
                          'sections that need it' % ', '.join(sorted(parser.defaults())))
    for section in parser.sections():
        if section not in known:
            raise ConfigError('[%s]: unknown section; delete it or fix the spelling '
                              '(known sections: %s)' % (section, ', '.join(known)))
        for key in parser.options(section):
            if key not in known[section]:
                _fail(section, key, 'unknown key; delete it or fix the spelling')


def _check_blanks(cfg):
    """Refuse a key written with no value where "not set" means nothing (5.2.6 among them).

    Only the four keys of OPTIONAL may be left blank.
    """
    for section, keys in SPEC:
        namespace = getattr(cfg, section)
        for key, _kind, default in keys:
            if getattr(namespace, key) is not None or (section, key) in OPTIONAL:
                continue
            if section == 'variables' and key in REQUIRED_COLUMNS:          # 5.2.6
                problem = 'is required; write the exact column header that holds %s' % key
            else:
                problem = ('is written with no value; write one, or delete the line to '
                           'use the default (%s)' % default)
            _fail(section, key, problem)


# --------------------------------------------------------------------------- derived

def _derive_gases(cfg):
    """cfg.gases.measure_type[gas] -- refusal 5.2.2 for anything but the two supported kinds."""
    cfg.gases.measure_type = {}
    for gas in GASES:
        key = gas + '_measure_type'
        value = getattr(cfg.gases, key)
        if value not in MEASURE_TYPES:
            _fail('gases', key,
                  'miniflux supports mixing_ratio (dry) and molar_density only; '
                  'convert a wet mole fraction upstream (got %r)' % value)
        cfg.gases.measure_type[gas] = value


def _derive_units(cfg):
    """cfg.units.convert[name] = (a, b) of CONTRACT 6.1; refusals 5.2.3 and the mass-density rule."""
    convert = {'u': (1.0, 0.0), 'v': (1.0, 0.0), 'w': (1.0, 0.0)}   # sonic wind is m s-1 already
    for name in ('ts', 'ta', 'p_air'):
        table = SCALAR_UNITS[name]
        unit = getattr(cfg.units, name)
        if unit not in table:
            _fail('units', name, 'unknown unit %r; write one of %s'
                  % (unit, ' | '.join(sorted(table))))
        convert[name] = table[unit]
    for gas in GASES:
        unit = getattr(cfg.units, gas)
        measure = cfg.gases.measure_type[gas]
        if unit in MIXING_UNITS:
            _require_measure(gas, unit, measure, 'mixing_ratio', 'a dry mixing ratio')
            convert[gas] = (MIXING_UNITS[unit], 0.0)
        elif unit in MOLAR_DENSITY_UNITS:
            _require_measure(gas, unit, measure, 'molar_density', 'a molar density')
            convert[gas] = (MOLAR_DENSITY_UNITS[unit], 0.0)
        elif unit in MASS_DENSITY_UNITS:
            _require_measure(gas, unit, measure, 'molar_density', 'a mass density')
            # The molar mass is a constant of the gas, not of the site, so mass -> mole is
            # exact: a = (kg per declared unit) / M.
            convert[gas] = (MASS_DENSITY_UNITS[unit] / MOLAR_MASS[gas], 0.0)
        else:
            _fail('units', gas, 'unknown unit %r; write one of %s' % (unit, ' | '.join(
                sorted(MIXING_UNITS) + sorted(MOLAR_DENSITY_UNITS) + sorted(MASS_DENSITY_UNITS))))
    cfg.units.convert = convert


def _require_measure(gas, unit, measure, needed, what):
    if measure != needed:
        _fail('units', gas,
              '%r is %s and is only legal when [gases] %s_measure_type = %s, but it is %s; '
              'fix whichever of the two is wrong' % (unit, what, gas, needed, measure))


def _derive_period(cfg):
    """cfg.period.seconds [s] and cfg.period.dt [s], so no other module repeats the arithmetic."""
    if cfg.period.averaging_minutes <= 0:
        _fail('period', 'averaging_minutes',
              'must be > 0, got %r' % cfg.period.averaging_minutes)
    # The output timestamps are YYYYMMDDHHMM (CONTRACT section 15), so a sub-minute period
    # would print two consecutive rows under one label -- two different numbers wearing the
    # same name. Refuse it rather than emit that.
    if abs(cfg.period.averaging_minutes - round(cfg.period.averaging_minutes)) > 1e-9:
        _fail('period', 'averaging_minutes',
              'must be a whole number of minutes, got %r: TIMESTAMP_START and '
              'TIMESTAMP_END are written to minute resolution, so a fractional period '
              'would label two rows identically'
              % cfg.period.averaging_minutes)
    if cfg.period.acquisition_frequency <= 0:
        _fail('period', 'acquisition_frequency',
              'must be > 0 [Hz], got %r' % cfg.period.acquisition_frequency)
    cfg.period.seconds = cfg.period.averaging_minutes * 60.0
    cfg.period.dt = 1.0 / cfg.period.acquisition_frequency


def _derive_site(cfg):
    """cfg.site.displacement [m]: declared, else 2/3 of the canopy, else 0.0 and say so."""
    declared = cfg.site.displacement_height
    if declared is not None and declared > 0:
        cfg.site.displacement = declared
    elif cfg.site.canopy_height > 0:
        if declared is not None:
            logger.warning('[site] displacement_height = %r is not > 0; using 2/3 of '
                           'canopy_height instead', declared)
        cfg.site.displacement = CANOPY_DISPLACEMENT_RATIO * cfg.site.canopy_height
    else:
        logger.warning('[site] displacement_height and canopy_height are both unset or '
                       'zero; using displacement = 0.0 m')
        cfg.site.displacement = 0.0


def _lag_windows(parser, lag):
    """cfg.lag.windows[scalar] = (nominal_s, min_s, max_s) [s]; refusal 5.2.8."""
    windows = {}
    for scalar in lag.scalars:
        values = []
        written = False
        for suffix, default in LAG_WINDOW_DEFAULTS:
            key = '%s_%s' % (scalar, suffix)
            written = written or parser.has_option('lag', key)
            value = _value(parser, 'lag', key, 'float', default)
            if value is None:
                _fail('lag', key, 'is written with no value; write a time in seconds, or '
                                  'delete the line to use the default (%s)' % default)
            values.append(value)
            setattr(lag, key, value)
        nominal, low, high = values
        if low > high:
            _fail('lag', '%s_min_s' % scalar,
                  'min_s (%r) is above max_s (%r); swap them' % (low, high))
        if not written:
            logger.info('[lag] %s has no window keys; using the default %r .. %r s '
                        'about a nominal %r s', scalar, low, high, nominal)
        windows[scalar] = (nominal, low, high)
    return windows


# --------------------------------------------------------------------------- validation

def _validate(cfg):
    """Every remaining refusal of CONTRACT 5.2, each naming its section and key."""
    column, constant = cfg.variables.p_air, cfg.site.pressure_pa           # 5.2.4 / 5.2.5
    if column is None and constant is None:
        raise ConfigError(
            '[variables] p_air and [site] pressure_pa are both unset: miniflux never '
            'invents a pressure. Name the pressure column in [variables] p_air, or write a '
            'constant in [site] pressure_pa.')
    if column is not None and constant is not None:
        raise ConfigError(
            '[variables] p_air = %r and [site] pressure_pa = %r are both set, which is '
            'ambiguous. Delete [site] pressure_pa to use the column, or clear '
            '[variables] p_air to use the constant.' % (column, constant))
    if constant is not None and constant <= 0:
        _fail('site', 'pressure_pa', 'must be > 0 Pa, got %r; leaving it empty is how you '
                                     'say "read the pressure from a column"' % constant)
    cfg.site.pressure_source = 'column' if column is not None else 'constant'

    if cfg.despike.q <= 0:                                               # 5.2.7
        _fail('despike', 'q', 'must be > 0 robust standard deviations, got %r' % cfg.despike.q)

    for section, key in (('despike', 'variables'), ('detrend', 'variables'),   # 5.2.9
                         ('lag', 'scalars'), ('qc', 'steady_state_pair')):
        for name in getattr(getattr(cfg, section), key):
            if name not in CANONICAL:
                _fail(section, key, '%r is not a canonical variable; write one of %s'
                      % (name, ', '.join(CANONICAL)))

    # Each of these three lists is *applied once per entry*, so a repeat is a second pass:
    # despiking already-despiked data (and overwriting the spike count with the second
    # pass's zero), detrending an already-detrended series (and storing its ~0 post-detrend
    # mean for a 293 K sonic temperature), lagging an already-lagged scalar. None of that
    # is ever what a copy-paste typo meant.
    for section, key in (('despike', 'variables'), ('detrend', 'variables'),
                         ('lag', 'scalars')):
        names = getattr(getattr(cfg, section), key)
        for name in names:
            if names.count(name) > 1:
                _fail(section, key, '%r is written twice; each variable is processed once '
                                    'per name in this list, so a repeat is a second pass '
                                    'over the result of the first' % name)

    # `ta_mean` and `p_air_mean` belong to flux.thermodynamics, which resolves the air
    # temperature per sample (measured where finite, else sonic-derived) before detrend
    # runs. detrend writes meta['<var>_mean'] for whatever it is given, so listing either
    # here replaces the resolved mean with a plain column mean -- and H, FC, LE and E are
    # computed from it. Neither is a turbulence series, and neither enters a covariance.
    for name in cfg.detrend.variables:
        if name in ('ta', 'p_air'):
            _fail('detrend', 'variables',
                  '%r cannot be detrended: %s_mean is a thermodynamic mean owned by '
                  'flux.thermodynamics, which flux.assemble and wpl read, and detrending '
                  'it would overwrite that value. %r enters no covariance, so removing a '
                  'trend from it changes no flux.' % (name, name, name))

    if len(cfg.qc.steady_state_pair) != 2:
        _fail('qc', 'steady_state_pair', 'expected exactly two canonical names, got %s'
              % (', '.join(cfg.qc.steady_state_pair) or 'nothing'))

    # Guards beyond 5.2: a step must never raise, so anything that would only fail later,
    # deep in read.py or write.py, is refused here instead.
    if cfg.files.header_line < 1:
        _fail('files', 'header_line', 'is a 1-based line number, got %r' % cfg.files.header_line)
    if cfg.files.first_data_line <= cfg.files.header_line:
        _fail('files', 'first_data_line', 'must be below the header at line %d, got %r'
              % (cfg.files.header_line, cfg.files.first_data_line))
    if cfg.period.min_samples < 0:
        _fail('period', 'min_samples', 'must be >= 0 samples, got %r' % cfg.period.min_samples)
    if not -90.0 <= cfg.site.latitude <= 90.0:
        _fail('site', 'latitude', 'must be between -90 and 90 degrees north, got %r'
              % cfg.site.latitude)
    try:
        cfg.output.float_format % 1.0
    except (TypeError, ValueError):
        _fail('output', 'float_format',
              '%r is not a format that accepts one float; write something like %%.6g'
              % cfg.output.float_format)


def _resolve_numpy(cfg):
    """Refusal 5.2.10, then fix HAVE_NUMPY once for the run (CONTRACT 3.9).

    The question is whether ``import numpy`` *succeeded*, which only kernels.py can
    answer: a package that is findable and raises on import -- a half-installed one, or
    one built against another ABI -- is exactly what its try/except is there for, and
    asking the import system to find the spec answers a different question.

    ``numpy_enabled`` is then the resolved ``HAVE_NUMPY`` itself rather than a second
    guess at it, because ``describe()`` puts that value in the run's log as the record of
    which path the numbers came from.
    """
    enabled = cfg.runtime.use_numpy != 'off'
    if cfg.runtime.use_numpy == 'on' and not kernels.NUMPY_AVAILABLE:
        _fail('runtime', 'use_numpy',
              'is "on" but numpy cannot be imported; install numpy or write auto')
    # `auto` is a preference, not a demand, so it is resolved here rather than handed to
    # set_numpy as a request it would report as unmet on every numpy-less machine.
    cfg.runtime.numpy_enabled = kernels.set_numpy(enabled and kernels.NUMPY_AVAILABLE)
