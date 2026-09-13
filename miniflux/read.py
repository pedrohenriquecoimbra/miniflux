"""Read the input files and cut them into averaging periods (CONTRACT section 7).

This is the only module that knows about files, quoting, NA tokens and units as written.
Everything below it sees the canonical vocabulary of CONTRACT section 2 in canonical
units: eleven series keys that always exist, always ``array('d')``, always the same
length.

Every conversion done here is **affine** (``x = a*x_file + b``). The one non-affine
conversion miniflux performs -- a closed-path cell molar density into a dry mixing ratio
-- is a pipeline step of its own (``cell.py``), not a unit, which is why the cell
temperature and cell pressure arrive downstream as series rather than being consumed here.

Two of the rules here change the numbers rather than the plumbing (ALGORITHMS section 2):
a timestamp is parsed by explicit field extraction and **never** inferred, because a format
guessed from the first row of a 20 Hz stream discards the nineteen samples per second that
carry a fractional part; and ``closed`` decides which period a sample landing exactly on a
boundary belongs to, which displaces the whole period by one sample if it is wrong.

A sample clock is carried through this module as whole microseconds since ``EPOCH``, an
exact int. That is everything the period cut, the duplicate test and the sample count ask
of it, and the period dict keeps no per-sample time at all (CONTRACT 1.2): a datetime is
built twice per period, for its two boundaries, instead of once per sample.

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

SERIES = ('u', 'v', 'w', 'ts', 'co2', 'h2o', 'ta', 'p_air', 't_cell', 'p_cell')
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
    start, end = _bounds(_elapsed(t), width, closed)
    return _moment(start), _moment(end)


def _bounds(key, width, closed):
    """``period_bounds`` on the integer clock the stream is actually cut on.

    key: one sample instant in whole microseconds since EPOCH. width: the averaging
    interval in the same unit. closed: as above. Returns ``(start, end)``, both int.
    """
    # Python's % is non-negative for a positive modulus, so this floors before 1970 too.
    floor = key - key % width
    if closed == 'right':
        end = floor if floor == key else floor + width
        return end - width, end
    return floor, floor + width


def _elapsed(t):
    """One datetime as whole microseconds since EPOCH. Returns int."""
    delta = t - EPOCH
    return (delta.days * 86400 + delta.seconds) * US + delta.microseconds


def _moment(key):
    """Whole microseconds since EPOCH back as a datetime. The inverse of ``_elapsed``."""
    return EPOCH + timedelta(microseconds=key)


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


def _key(token, kind, cut, seconds, fractions):
    """One timestamp token as whole microseconds since EPOCH, built from two lookups.

    token, kind: as ``parse_timestamp`` takes them. cut: where the fraction starts, 19 for
    'iso' and 14 for 'numeric'. seconds: one dict per file, the token up to the fraction ->
    that whole second. fractions: one dict per file, the fraction as written -> its
    microseconds. At 20 Hz ``fractions`` holds twenty entries however long the file is and
    ``seconds`` one per second of stream, so almost every row is two hits and an addition
    and the date, the time of day and the fraction are each parsed once.

    Every token either dict does not recognise goes to ``parse_timestamp``, which owns
    every refusal. A token carrying leading whitespace is one of them, always: ``cut``
    then falls inside its own seconds field and the leftover starts with a digit rather
    than a '.', which is the shape ``fractions`` refuses to learn, so a shifted slice can
    never collide with a real fraction.
    """
    base = seconds.get(token[:cut])
    if base is not None:
        micro = fractions.get(token[cut:])
        if micro is not None:
            return base + micro
    stamp = parse_timestamp(token, kind)
    key = _elapsed(stamp)
    seconds[token[:cut]] = key - stamp.microsecond
    rest = token[cut:]
    if not rest or rest[0] == '.':
        fractions[rest] = stamp.microsecond
    return key


def _shape_error(text, shape):
    return ('timestamp %r does not have the shape declared by [timestamp] format, which is '
            '%s; miniflux parses timestamps and never infers their format' % (text, shape))


# --------------------------------------------------------------------------- the stream

def periods(cfg, counters=None):
    """Yield one period dict per averaging period, in chronological order.

    Reads every file matched by ``[files] input_glob`` **in timestamp order**
    (ALGORITHMS 2.3), converts each declared column with its affine map from CONTRACT
    6.1, and closes a period as soon as a sample belonging to a later one arrives.
    Writes all ten series keys and the meta keys ``period_start``, ``period_end``,
    ``n_in``, ``n_dup``, ``freq_hz``. No per-sample time is written (CONTRACT 1.2): the
    sample clock lives in this function as an integer key and leaves it as the two
    boundaries of each period.

    counters: an optional dict this function adds facts about the stream to, because a
    period it drops is never yielded and a log record is not a number. ``skipped_short``
    is incremented once per period dropped for holding fewer than
    ``[period] min_samples`` samples.

    Guarantees to everything downstream: the samples of a period are in strictly
    increasing clock order, the ten series have equal length, an absent column is all-NaN
    (``ta``, ``t_cell``, ``p_cell``) or the declared constant (``p_air``), and the units
    are canonical.
    """
    paths = glob.glob(cfg.files.input_glob)
    if not paths:
        raise ReadError('[files] input_glob = %r matches no file (resolved from %s)'
                        % (cfg.files.input_glob, os.getcwd()))
    logger.info('reading %d file(s) matching %r', len(paths), cfg.files.input_glob)
    declared = _declared(cfg)
    closed = cfg.period.closed
    width = int(round(cfg.period.seconds * US))
    acc = None
    for path in _ordered(paths, cfg):
        for key, values, lineno in _samples(path, cfg, declared):
            # The cut is the most expensive thing in this loop and at 20 Hz it lands on
            # the period already open 35999 times out of 36000, so it is computed only for
            # the sample that leaves it -- `_holds` is `_bounds`' own rule, read as a
            # test instead of a division.
            if acc is not None and _holds(acc, key, closed):
                start, end = acc['start'], acc['end']
            else:
                start, end = _bounds(key, width, closed)
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
                        % (path, lineno, _moment(key), _moment(end),
                           _moment(acc['end'])))
                period = _close(acc, cfg, declared, counters)
                if period is not None:
                    yield period
                acc = _open(start, end, declared)
            # Keeping only samples strictly later than the last one kept is what puts the
            # ten series in clock order, and since they are in clock order, a repeat of any
            # earlier instant can only be a repeat of this one: first wins, rest counted.
            if key <= acc['last']:
                acc['n_dup'] += 1
                continue
            acc['last'] = key
            acc['n'] += 1
            for column, value in zip(acc['columns'], values):
                column.append(value)
    if acc is not None:
        period = _close(acc, cfg, declared, counters)
        if period is not None:
            yield period


def _holds(acc, key, closed):
    """Whether one sample key falls in the period already open.

    Exactly ``_bounds(key, ...) == (acc['start'], acc['end'])``, written the other way
    round: the period is ``(start, end]`` when ``closed = 'right'`` and ``[start, end)``
    when it is 'left' (ALGORITHMS 0.1), which is one comparison instead of the floor.
    """
    if closed == 'right':
        return acc['start'] < key <= acc['end']
    return acc['start'] <= key < acc['end']


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
    # 'last' starts one microsecond before the earliest instant this period can hold, so
    # the first sample is later than it under either `closed` and needs no separate test.
    return {'start': start, 'end': end, 'last': start - 1, 'n': 0, 'n_dup': 0,
            'columns': [[] for _name in declared]}


def _close(acc, cfg, declared, counters=None):
    """Finish one period: the ten arrays and the five meta keys, or None when it is short.

    The two datetimes this builds are the only ones the stream costs: the samples
    themselves are counted and ordered on the integer clock of ``_key``.
    """
    start, end = _moment(acc['start']), _moment(acc['end'])
    label = '%s .. %s' % (start, end)
    n = acc['n']
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
    period = {'meta': {'period_start': start, 'period_end': end,
                       'n_in': n, 'n_dup': acc['n_dup'],
                       'freq_hz': cfg.period.acquisition_frequency}}
    for name, column in zip(declared, acc['columns']):
        period[name] = kernels.from_values(column)
    for name in SERIES:
        if name in period:
            continue
        # Only `ta`, `p_air` and the two cell columns can be absent; config refuses a run
        # without the other six, and refuses a closed-path cell density without the cell
        # state. A missing pressure column is the one constant miniflux fills, and only
        # because the user wrote the number themselves in [site] pressure_pa.
        fill = cfg.site.pressure_pa if name == 'p_air' else kernels.NAN
        period[name] = kernels.new(n, fill)
    return period


# --------------------------------------------------------------------------- one file

def _rows(path, cfg):
    """Yield ``(lineno, row)`` for one file, split exactly as ``csv.reader`` splits it.

    Most lines of a 20 Hz file are numbers and carry no quote character at all, and those
    ``str.split`` cuts in half the time the reader takes; ``_split`` takes them, hands the
    rest back to ``csv.reader``, and never guesses (CONTRACT 7.1).

    Every way a file can refuse to be read -- it is a directory, the permissions deny it,
    the bytes are not the declared encoding, ``[files] encoding`` names no codec -- is an
    input problem the user can fix, so it leaves this module as a ReadError naming the
    file (cli exit 3) and not as an internal crash (CONTRACT 19).
    """
    try:
        handle = open(path, 'r', encoding=cfg.files.encoding, newline='')
    except (OSError, LookupError) as exc:
        raise ReadError(_unreadable(path, cfg, exc))
    delimiter = cfg.files.delimiter
    quotechar = cfg.files.quotechar
    try:
        lineno = 0
        for line in handle:
            lineno += 1
            first = lineno
            row = _split(line.rstrip('\r\n'), delimiter, quotechar)
            if row is None:
                taken = [line]
                row = next(csv.reader(_record(taken, handle), delimiter=delimiter,
                                      quotechar=quotechar), [])
                lineno += len(taken) - 1   # a quoted field may hold the newline it read
            yield first, row
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ReadError(_unreadable(path, cfg, exc))
    finally:
        handle.close()


def _split(text, delimiter, quotechar):
    """One line as ``csv.reader`` would split it, or None when the reader itself is needed.

    text: one line with its terminator already off. Returns a list of str, or None for
    every quote shape not proved equivalent to a plain split -- a field holding the
    delimiter somewhere other than the first column, a doubled quote, a quote inside an
    unquoted field, a field whose quote never closes on this line. Those are rare, legal
    and easy to get subtly wrong, so this refuses to be clever about them and the reader
    parses them instead.
    """
    if quotechar not in text:
        return text.split(delimiter) if text else []
    if text[0] == quotechar and text.count(quotechar) == 2:
        # One quoted field in front of unquoted ones: a TOA5 timestamp, and the only
        # quoting an eddy-covariance logger writes on a data row.
        head, _quote, rest = text[1:].partition(quotechar)
        if not rest:
            return [head]
        if rest[0] == delimiter:
            row = rest[1:].split(delimiter)
            row.insert(0, head)
            return row
    return None


def _record(taken, handle):
    """The lines of one csv record: the one that opened it, then any it still needs.

    A quoted field may hold a newline, which makes a record longer than a line. Every line
    the reader pulls lands in ``taken``, which is how the caller keeps its line count.
    """
    yield taken[0]
    for line in handle:
        taken.append(line)
        yield line


def _unreadable(path, cfg, exc):
    return ('%s cannot be read as [files] encoding = %s: %s'
            % (path, cfg.files.encoding, exc))


def _samples(path, cfg, declared):
    """Yield ``(key, [float, ...], lineno)`` for every data row of one file.

    ``key`` is the sample instant as whole microseconds since EPOCH (``_key``). The floats
    are in the order of ``declared`` and already in canonical units. Every refusal names
    the file and the 1-based line number.

    Each test around ``float()`` below earns its place: ``strip`` because an NA token may
    be written with spaces around it, the NA test *before* the conversion because a
    sentinel like -9999 is a number ``float()`` would happily return, and the underscore
    scan because PEP 515 makes '1_0' one too. Calling ``float()`` first and sorting the
    rest out in the exception path was measured and is not faster -- ``float()`` itself is
    most of what this loop costs, and the three tests together are a fraction of it.
    """
    plan = None
    first = last = None
    count = 0
    kind = cfg.timestamp.format
    cut = 19 if kind == 'iso' else 14
    seconds = {}
    fractions = {}
    # A set for the per-token test, which runs once per declared column of every row; the
    # configured tuple keeps its order for the message that names it.
    na_values = frozenset(cfg.files.na_values)
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
            key = _key(row[time_index], kind, cut, seconds, fractions)
        except ReadError as exc:
            raise ReadError('%s line %d: %s' % (path, lineno, exc))
        values = []
        for index, a, b, column in entries:
            token = row[index].strip()
            if token in na_values:
                values.append(kernels.NAN)
                continue
            try:
                # A token carrying a digit-grouping underscore is corruption, and reading
                # it as the number float() makes of it is the "looks right and is not"
                # case CONTRACT 19 refuses.
                if '_' in token:
                    raise ValueError(token)
                values.append(float(token) * a + b)
            except ValueError:
                _refuse_number(row[index], cfg, path, lineno, column)
        count += 1
        if first is None:
            first = key
        last = key
        yield key, values, lineno
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


def _refuse_number(text, cfg, path, lineno, column):
    """Refuse one token that is neither a number nor an NA token. Never returns."""
    raise ReadError('%s line %d: column %r holds %r, which is neither a number nor one '
                    'of [files] na_values = %s'
                    % (path, lineno, column, text, ','.join(cfg.files.na_values)))


def _check_rate(path, cfg, count, first, last):
    """Compare the observed sample rate with the configured one; the configured one wins.

    first and last are the keys of the first and the last row of the file, so their
    difference is the span of the file in whole microseconds.
    """
    if count < 2 or last <= first:
        return
    observed = (count - 1) * US / (last - first)
    if abs(observed - cfg.period.acquisition_frequency) > \
            RATE_TOLERANCE * cfg.period.acquisition_frequency:
        logger.warning('%s: its %d rows arrive at %.4g Hz, but [period] '
                       'acquisition_frequency is %.4g Hz, which is the rate every '
                       'calculation uses', path, count, observed,
                       cfg.period.acquisition_frequency)
