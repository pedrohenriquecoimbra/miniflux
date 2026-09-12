"""Read the input files and cut them into averaging periods (CONTRACT section 7).

This is the only module that knows about files, quoting, NA tokens and units as written.
Everything below it sees the canonical vocabulary of CONTRACT section 2 in canonical
units: nine series keys that always exist, always ``array('d')``, always the same length.

Two of the rules here change the numbers rather than the plumbing (ALGORITHMS section 2):
a timestamp is parsed by explicit field extraction and **never** inferred, because a format
guessed from the first row of a 20 Hz stream discards the nineteen samples per second that
carry a fractional part; and ``closed`` decides which period a sample landing exactly on a
boundary belongs to, which displaces the whole period by one sample if it is wrong.

Only this module raises :class:`ReadError`, and it refuses rather than guessing: a
timestamp of an unsupported shape, a missing column, a short row, a token that is neither
a number nor an NA token, an empty glob, a file that cannot be opened or decoded, and a
sample that arrives after the period it belongs to has already been closed.
"""

import csv
import glob
import logging
import os
from datetime import datetime, timedelta

from . import kernels
from .errors import ReadError

logger = logging.getLogger(__name__)

SERIES = ('u', 'v', 'w', 'ts', 'co2', 'h2o', 'ta', 'p_air')
EPOCH = datetime(1970, 1, 1)   # every period boundary is anchored here, as pandas' floor/ceil is
US = 1000000                   # microseconds per second; period arithmetic is done in whole ones
RATE_TOLERANCE = 0.01          # relative gap between observed and configured rate worth a warning
ISO_SHAPE = 'YYYY-MM-DD HH:MM:SS[.ffffff], with " " or "T" between date and time'
NUMERIC_SHAPE = 'YYYYMMDDHHMMSS[.ffffff]'


# --------------------------------------------------------------------------- period cut

def period_bounds(t, seconds, closed):
    """The averaging period one sample instant falls in.

    t: datetime. seconds: the averaging interval A [s]. closed: 'right' | 'left'.
    Returns ``(start, end)``, both datetime; the period is labelled by ``end`` under both
    conventions. A sample exactly on a boundary belongs to the earlier period when
    ``closed = 'right'`` and to the later one when ``closed = 'left'`` (ALGORITHMS 0.1).
    """
    width = int(round(seconds * US))          # whole microseconds, so the cut cannot drift
    delta = t - EPOCH
    elapsed = (delta.days * 86400 + delta.seconds) * US + delta.microseconds
    # Python's % is non-negative for a positive modulus, so this floors before 1970 too.
    floor = elapsed - elapsed % width
    if closed == 'right':
        end = floor if floor == elapsed else floor + width
        start = end - width
    else:
        start = floor
        end = floor + width
    return EPOCH + timedelta(microseconds=start), EPOCH + timedelta(microseconds=end)


# --------------------------------------------------------------------------- timestamps

def parse_timestamp(token, kind):
    """Parse one timestamp token by explicit field extraction, never by inference.

    token: str exactly as written in the file. kind: 'iso'
    (``YYYY-MM-DD[ T]HH:MM:SS[.ffffff]``) or 'numeric' (``YYYYMMDDHHMMSS[.ffffff]``).
    Returns a naive datetime. Raises ReadError on any other shape.

    A numeric stamp is read as text: ``20220512233000.05`` carries sixteen significant
    digits and a double carries about fifteen and a half (ALGORITHMS 2.2).
    """
    text = token.strip()
    if kind == 'iso':
        if (len(text) < 19 or text[4] != '-' or text[7] != '-' or text[10] not in ' T'
                or text[13] != ':' or text[16] != ':'):
            raise ReadError(_shape_error(text, ISO_SHAPE))
        fields = (text[0:4], text[5:7], text[8:10], text[11:13], text[14:16], text[17:19])
        return _instant(fields, text[19:], text, ISO_SHAPE)
    if kind == 'numeric':
        if len(text) < 14:
            raise ReadError(_shape_error(text, NUMERIC_SHAPE))
        fields = (text[0:4], text[4:6], text[6:8], text[8:10], text[10:12], text[12:14])
        return _instant(fields, text[14:], text, NUMERIC_SHAPE)
    raise ReadError('[timestamp] format = %r is not one of iso | numeric' % kind)


def _instant(fields, rest, text, shape):
    """Build the datetime from six digit fields and whatever trails them."""
    for part in fields:
        if not part.isdigit():
            raise ReadError(_shape_error(text, shape))
    micro = 0
    if rest:
        digits = rest[1:]
        if rest[0] != '.' or not digits or not digits.isdigit():
            raise ReadError(_shape_error(text, shape))
        if len(digits) > 6:
            raise ReadError('timestamp %r carries %d fractional digits; a datetime holds '
                            'six, and miniflux does not round a sample clock to fit'
                            % (text, len(digits)))
        micro = int(digits.ljust(6, '0'))     # ".05" is 50000 us, not 5 us
    try:
        year, month, day, hour, minute, second = (int(part) for part in fields)
        return datetime(year, month, day, hour, minute, second, micro)
    except ValueError as exc:
        raise ReadError('timestamp %r is not a real instant: %s' % (text, exc))


def _shape_error(text, shape):
    return ('timestamp %r does not have the shape declared by [timestamp] format, which is '
            '%s; miniflux parses timestamps and never infers their format' % (text, shape))


# --------------------------------------------------------------------------- the stream

def periods(cfg, counters=None):
    """Yield one period dict per averaging period, in chronological order.

    Reads every file matched by ``[files] input_glob`` **in timestamp order**
    (ALGORITHMS 2.3), converts each declared column with its affine map from CONTRACT
    6.1, and closes a period as soon as a sample belonging to a later one arrives.
    Writes all nine series keys, ``t``, and the meta keys ``period_start``,
    ``period_end``, ``n_in``, ``n_dup``, ``freq_hz``.

    counters: an optional dict this function adds facts about the stream to, because a
    period it drops is never yielded and a log record is not a number. ``skipped_short``
    is incremented once per period dropped for holding fewer than
    ``[period] min_samples`` samples.

    Guarantees to everything downstream: ``t`` is non-decreasing, the nine series have
    equal length, an absent column is all-NaN (``ta``) or the declared constant
    (``p_air``), and the units are canonical.
    """
    paths = glob.glob(cfg.files.input_glob)
    if not paths:
        raise ReadError('[files] input_glob = %r matches no file (resolved from %s)'
                        % (cfg.files.input_glob, os.getcwd()))
    logger.info('reading %d file(s) matching %r', len(paths), cfg.files.input_glob)
    declared = _declared(cfg)
    acc = None
    for path in _ordered(paths, cfg):
        for stamp, values, lineno in _samples(path, cfg, declared):
            start, end = period_bounds(stamp, cfg.period.seconds, cfg.period.closed)
            if acc is None:
                acc = _open(start, end, declared)
            elif end != acc['end']:
                if end < acc['end']:
                    # Not a duplicate: ALGORITHMS 2.3 scopes the keep-first rule to a
                    # repeat *within* a period, and this sample belongs to one already
                    # yielded. Dropping it would compute that period's flux from part of
                    # the data and report neither the loss nor which period lost it.
                    raise ReadError(
                        '%s line %d: the sample at %s belongs to the period ending %s, '
                        'and the stream has already moved past it into the period ending '
                        '%s. The files are read in the order of their first sample, so '
                        'this is a clock that steps backwards inside a file or two files '
                        'that overlap -- sort the rows, or run the overlapping files '
                        'separately. miniflux will not silently drop them.'
                        % (path, lineno, stamp, end, acc['end']))
                period = _close(acc, cfg, declared, counters)
                if period is not None:
                    yield period
                acc = _open(start, end, declared)
            # Keeping only samples strictly later than the last one kept is what makes `t`
            # non-decreasing downstream, and since it is non-decreasing, a repeat of any
            # earlier instant can only be a repeat of this one: first wins, rest counted.
            if acc['t'] and stamp <= acc['t'][-1]:
                acc['n_dup'] += 1
                continue
            acc['t'].append(stamp)
            for column, value in zip(acc['columns'], values):
                column.append(value)
    if acc is not None:
        period = _close(acc, cfg, declared, counters)
        if period is not None:
            yield period


def _ordered(paths, cfg):
    """The matched files in the order their samples arrive in. Returns a list of paths.

    ALGORITHMS 2.3 cuts periods on the sample clock of the files *concatenated in
    timestamp order*, and a file name is not a clock: ``sorted()`` puts ``data_10.csv``
    before ``data_2.csv``, which walks the stream backwards over the rollover every
    logger's counter eventually makes. So each file is peeked at -- its header and its
    first data row, nothing more -- and sorted on that instant, with the name breaking
    ties and carrying any file that holds no data row at all.
    """
    keyed = []
    for path in sorted(paths):
        stamp = _first_stamp(path, cfg)
        keyed.append(((1, EPOCH, path) if stamp is None else (0, stamp, path), path))
    keyed.sort(key=lambda item: item[0])
    return [path for _key, path in keyed]


def _first_stamp(path, cfg):
    """The instant of the first data row of one file, or None when it has none.

    The header and one row, then the file is closed again -- the peek this costs is one
    open per file, against reading the whole stream in an order the clock disagrees with.
    """
    rows = _rows(path, cfg)
    try:
        time_index = None
        for lineno, row in rows:
            if lineno == cfg.files.header_line:
                time_index = _locate(_names(row), cfg.timestamp.column,
                                     'timestamp', 'column', cfg, path)
                continue
            if lineno < cfg.files.first_data_line or not row:
                continue
            if time_index is None or len(row) <= time_index:
                return None               # refused for real by _samples, in one place
            try:
                return parse_timestamp(row[time_index], cfg.timestamp.format)
            except ReadError as exc:
                raise ReadError('%s line %d: %s' % (path, lineno, exc))
        return None
    finally:
        rows.close()                      # the peek always stops early; close it here


def _declared(cfg):
    """The canonical names that have a column in the file, in the order of SERIES."""
    return tuple(name for name in SERIES if getattr(cfg.variables, name))


def _open(start, end, declared):
    return {'start': start, 'end': end, 't': [], 'n_dup': 0,
            'columns': [[] for _name in declared]}


def _close(acc, cfg, declared, counters=None):
    """Finish one period: the nine arrays and the five meta keys, or None when it is short."""
    label = '%s .. %s' % (acc['start'], acc['end'])
    n = len(acc['t'])
    if acc['n_dup']:
        logger.warning('period %s: dropped %d row(s) whose timestamp repeats an earlier '
                       'one; the first sample at each instant is kept', label, acc['n_dup'])
    if n < cfg.period.min_samples:
        # A short period is never yielded, so the caller's counter is the only number that
        # records it. It is counted here rather than harvested from the log record below,
        # which does not exist at all at [runtime] log_level = ERROR.
        if counters is not None:
            counters['skipped_short'] = counters.get('skipped_short', 0) + 1
        logger.warning('period %s: %d row(s) is below [period] min_samples = %d; the period '
                       'is skipped', label, n, cfg.period.min_samples)
        return None
    period = {'meta': {'period_start': acc['start'], 'period_end': acc['end'],
                       'n_in': n, 'n_dup': acc['n_dup'],
                       'freq_hz': cfg.period.acquisition_frequency},
              't': acc['t']}
    for name, column in zip(declared, acc['columns']):
        period[name] = kernels.from_values(column)
    for name in SERIES:
        if name in period:
            continue
        # Only `ta` and `p_air` can be absent; config refuses a run without the other six.
        # A missing pressure column is the one constant miniflux fills, and only because
        # the user wrote the number themselves in [site] pressure_pa.
        fill = cfg.site.pressure_pa if name == 'p_air' else kernels.NAN
        period[name] = kernels.new(n, fill)
    return period


# --------------------------------------------------------------------------- one file

def _rows(path, cfg):
    """Yield ``(lineno, row)`` for one file, as ``csv.reader`` splits it.

    Every way a file can refuse to be read -- it is a directory, the permissions deny it,
    the bytes are not the declared encoding, ``[files] encoding`` names no codec -- is an
    input problem the user can fix, so it leaves this module as a ReadError naming the
    file (cli exit 3) and not as an internal crash (CONTRACT 19).
    """
    try:
        handle = open(path, 'r', encoding=cfg.files.encoding, newline='')
    except (OSError, LookupError) as exc:
        raise ReadError(_unreadable(path, cfg, exc))
    try:
        reader = csv.reader(handle, delimiter=cfg.files.delimiter,
                            quotechar=cfg.files.quotechar)
        for lineno, row in enumerate(reader, 1):
            yield lineno, row
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ReadError(_unreadable(path, cfg, exc))
    finally:
        handle.close()


def _unreadable(path, cfg, exc):
    return ('%s cannot be read as [files] encoding = %s: %s'
            % (path, cfg.files.encoding, exc))


def _samples(path, cfg, declared):
    """Yield ``(datetime, [float, ...], lineno)`` for every data row of one file.

    The floats are in the order of ``declared`` and already in canonical units. Every
    refusal names the file and the 1-based line number.
    """
    plan = None
    first = last = None
    count = 0
    for lineno, row in _rows(path, cfg):
        if lineno == cfg.files.header_line:
            plan = _plan(row, cfg, declared, path)
            continue
        if lineno < cfg.files.first_data_line or not row:
            continue                      # a preamble line, or a blank line at the end
        time_index, entries, width = plan
        if len(row) < width:
            raise ReadError('%s line %d: the row holds %d field(s) but the '
                            'configuration reads column %d'
                            % (path, lineno, len(row), width))
        try:
            stamp = parse_timestamp(row[time_index], cfg.timestamp.format)
        except ReadError as exc:
            raise ReadError('%s line %d: %s' % (path, lineno, exc))
        values = [_number(row[index], cfg.files.na_values, path, lineno, column) * a + b
                  for index, a, b, column in entries]
        count += 1
        if first is None:
            first = stamp
        last = stamp
        yield stamp, values, lineno
    if plan is None:
        raise ReadError('%s has no line %d for [files] header_line to read the column '
                        'names from' % (path, cfg.files.header_line))
    _check_rate(path, cfg, count, first, last)


def _names(header):
    """The header cells, stripped. Returns a list of str."""
    return [cell.strip() for cell in header]


def _locate(names, column, section, key, cfg, path):
    """The position of one declared column in the header. Returns int; refuses if absent."""
    for position, name in enumerate(names):
        if name == column:                    # a repeated header name: the first one wins
            return position
    raise ReadError('%s: the header at line %d has no column %r, named by [%s] %s; '
                    'it holds %s' % (path, cfg.files.header_line, column, section,
                                     key, ', '.join(repr(n) for n in names)))


def _plan(header, cfg, declared, path):
    """Resolve the header row: ``(time_index, entries, width)``.

    Each entry is ``(index, a, b, column)``, so a data row costs one lookup and one affine
    map per declared variable. ``width`` is one past the highest column index used.
    """
    names = _names(header)
    time_index = _locate(names, cfg.timestamp.column, 'timestamp', 'column', cfg, path)
    entries = []
    width = time_index + 1
    for name in declared:
        column = getattr(cfg.variables, name)
        position = _locate(names, column, 'variables', name, cfg, path)
        a, b = cfg.units.convert[name]
        entries.append((position, a, b, column))
        width = max(width, position + 1)
    return time_index, tuple(entries), width


def _number(text, na_values, path, lineno, column):
    """One data token as a float; an NA token becomes NaN, anything else is refused."""
    token = text.strip()
    if token in na_values:
        return kernels.NAN
    try:
        # float() also accepts PEP 515 digit grouping, so '1_0' would arrive as 10.0. No
        # logger writes that; a token carrying one is corruption, and reading it as a
        # number is the "looks right and is not" case CONTRACT 19 refuses.
        if '_' in token:
            raise ValueError(token)
        return float(token)
    except ValueError:
        raise ReadError('%s line %d: column %r holds %r, which is neither a number nor one '
                        'of [files] na_values = %s'
                        % (path, lineno, column, text, ','.join(na_values)))


def _check_rate(path, cfg, count, first, last):
    """Compare the observed sample rate with the configured one; the configured one wins."""
    if count < 2 or last <= first:
        return
    observed = (count - 1) / (last - first).total_seconds()
    if abs(observed - cfg.period.acquisition_frequency) > \
            RATE_TOLERANCE * cfg.period.acquisition_frequency:
        logger.warning('%s: its %d rows arrive at %.4g Hz, but [period] '
                       'acquisition_frequency is %.4g Hz, which is the rate every '
                       'calculation uses', path, count, observed,
                       cfg.period.acquisition_frequency)
