"""Optional boundary: a period as an xarray Dataset, and any miniflux step run on one.

The core has no dependencies and this module does not change that. **Nothing in
``miniflux/`` imports this file.** It imports the core; the core does not know it exists;
an interpreter with no xarray and no pandas runs the whole program and the whole test
suite exactly as before. It is the pattern ``kernels.py`` already uses for numpy, moved
one level out -- numpy is optional *inside* a module, xarray is optional *outside* every
module.

Why it is not in the core, in one line: xarray pulls numpy **and** pandas, of the order of
a hundred megabytes, and none of its strengths are live on a single-axis 2.3 MB period that
is already in memory, already aligned and already in one unit vocabulary. CONTRACT section
22 is the law for this file and gives the argument in full; what follows is how to use it.

Three functions::

    to_dataset(period, cfg, copy=True)    -> xarray.Dataset
    from_dataset(dataset, cfg, copy=True) -> period dict (CONTRACT section 1)
    apply(step, dataset, cfg, copy=True)  -> xarray.Dataset

``apply`` takes a step function, a name from ``pipeline.STEPS``, or anything else with the
``(period, cfg) -> period`` signature -- ``pipeline.run_period`` included, which runs the
whole pipeline on a Dataset.

**Copies.** ``copy=True``, the default, gives the step its own memory, so nothing a step
does reaches the caller's Dataset. ``copy=False`` aliases instead: input and output share
one buffer per series and one ``meta`` dict, and :func:`apply` guarantees the step's result
is in that buffer when it returns -- rebinding steps included (:func:`_write_back`).
Aliasing needs a writable, C-contiguous ``float64`` array that is not dask-backed, and
anything else is refused rather than quietly copied, because a silent copy under
``copy=False`` means the mutation the caller asked for never arrives. :func:`from_dataset`
alone cannot promise that much: it hands out the aliases and never sees the step, so a
caller driving the step themselves reads the returned period, not their Dataset.

**Units and dtypes.** Every variable carries its CONTRACT section 2 unit in
``attrs['units']``, and coming the other way a declared unit that contradicts the canonical
one is refused, as is a variable whose dtype is not a number (see :data:`_NUMERIC_KINDS`).
Those two refusals are the one thing this boundary exists to provide: numbers that are
silently reinterpreted -- ppm read as mol mol-1, degC read as K, a datetime64 cast to
nanoseconds -- are exactly the failure a hand-written round trip produces, and they look
right all the way to the flux.

**Exceptions.** Plain ``ValueError`` and ``TypeError``. CONTRACT section 19 fixes the
package's exception list at three, all raised by the core; an adapter outside the core
does not get to add a fourth.

Serialising the result needs care that is the caller's, not this module's: ``period_start``
and ``period_end`` are ``datetime`` objects in ``Dataset.attrs``, which is fine in memory
and not writable by ``to_netcdf`` without an encoding of the caller's choosing.
"""

from array import array

import numpy as np
import xarray as xr

from . import kernels, read

#: Dimension name of a freshly built Dataset. ``from_dataset`` accepts any name; ``apply``
#: gives the caller's name back.
DIM = 'time'

#: CONTRACT section 2, the fixed rows. The two gases depend on their measure type and are
#: resolved per period by :func:`_unit`.
CANONICAL_UNITS = {
    'u': 'm s-1', 'v': 'm s-1', 'w': 'm s-1',
    'ts': 'K', 'ta': 'K', 't_cell': 'K',
    'p_air': 'Pa', 'p_cell': 'Pa',
}

#: CONTRACT section 2 again: what a gas means, per measure type.
GAS_UNITS = {'mixing_ratio': 'mol mol-1', 'molar_density': 'mol m-3'}

#: dtype kinds a measurement may arrive in: IEEE floats and integers, both of which widen
#: to float64 exactly or with a documented rounding. Every other kind is refused rather
#: than cast, because numpy casts them all without complaining and the numbers that come
#: out are not the measurement: a ``datetime64`` becomes nanoseconds since the epoch, a
#: bool becomes 1.0, a complex loses its imaginary part, a string of digits is parsed.
_NUMERIC_KINDS = 'fiu'

#: Spellings of the same unit that are accepted on the way in. Deliberately short: this is
#: a synonym list, not a unit parser, and anything not on it (``ppm``, ``degC``, ``g/m3``)
#: is a different number and is refused.
_SYNONYMS = {
    'm s-1': ('m/s', 'm s^-1', 'm.s-1', 'm s**-1'),
    'K': ('kelvin',),
    'Pa': ('pascal',),
    'mol mol-1': ('mol/mol', 'mol mol^-1', 'mol mol**-1'),
    'mol m-3': ('mol/m3', 'mol/m^3', 'mol m^-3', 'mol m**-3'),
}


def to_dataset(period, cfg, copy=True):
    """Build an xarray Dataset from a period dict.

    period: the dict of CONTRACT section 1. cfg: the Config of section 5, read for the
    gases' measure type and for the acquisition frequency. copy: see the module docstring.

    Returns a Dataset with the ten canonical series as variables over one dimension, each
    carrying its canonical unit in ``attrs['units']``, the period's ``meta`` as the Dataset
    attrs, and a time coordinate (:func:`_coordinate`).
    """
    meta = period['meta']
    n = _period_length(period)
    data = {}
    for name in read.SERIES:
        values = _ndarray(period[name], copy)
        data[name] = xr.DataArray(values, dims=DIM,
                                  attrs={'units': _unit(name, cfg, meta)})
    return xr.Dataset(data, coords={DIM: _coordinate(meta, cfg, n)}, attrs=dict(meta))


def from_dataset(dataset, cfg, copy=True):
    """Build a period dict from an xarray Dataset.

    dataset: a Dataset over one dimension carrying at least one canonical series; a
    canonical name it does not carry becomes an all-NaN series, never a missing key
    (CONTRACT section 1). Non-canonical variables are dropped -- the period has no room for
    them. cfg: the Config of section 5. copy: see the module docstring.

    Returns the period dict. Its series are ``array('d')`` when copying and the Dataset's
    own ``float64`` arrays when aliasing; both are what ``kernels`` accepts. Its ``meta`` is
    a copy of ``dataset.attrs`` when copying and that same dict when aliasing.

    Raises ValueError on a Dataset with no canonical series, on a series of the wrong shape
    or length, on a declared unit that contradicts CONTRACT section 2, and -- under
    ``copy=False`` only -- on a series that cannot be aliased.
    """
    dim = _dim(dataset)
    n = dataset.sizes[dim]
    meta = dict(dataset.attrs) if copy else dataset.attrs
    period = {'meta': meta}
    for name in read.SERIES:
        if name in dataset.variables:
            period[name] = _series(dataset[name], name, dim, n, cfg, meta, copy)
        else:
            period[name] = kernels.new(n)
    return period


def apply(step, dataset, cfg, copy=True):
    """Run one miniflux step on a Dataset and get a Dataset back.

    step: a ``(period, cfg) -> period`` callable, or the name of one in
    ``pipeline.STEPS``. ``pipeline.run_period`` is such a callable, so the whole pipeline
    runs this way too. dataset, cfg, copy: as :func:`from_dataset`.

    The returned Dataset carries the step's series and its meta, the caller's dimension
    name, the caller's coordinate when there was one (and none when there was not), and any
    non-canonical variables the caller had. Under ``copy=False`` input and output share one
    buffer per series and both hold the step's result, rebinding steps included.
    """
    function = _resolve(step)
    dim = _dim(dataset)
    period = from_dataset(dataset, cfg, copy=copy)
    aliases = None if copy else dict((name, period[name]) for name in read.SERIES)
    result = function(period, cfg)
    if aliases is not None:
        _write_back(aliases, result)
    # copy=False here in both branches: the period's arrays are either the caller's (asked
    # for) or ours alone (nobody else holds them), so a second copy would buy nothing.
    out = to_dataset(result, cfg, copy=False)
    if dim != DIM:
        out = out.rename({DIM: dim})
    if dim in dataset.coords:
        if dataset.sizes[dim] == out.sizes[dim]:
            out = out.assign_coords({dim: dataset[dim]})
    else:
        # The caller had no coordinate on this dimension, and handing one back would give
        # their Dataset an index it never had -- which a later merge or concat would then
        # align on. to_dataset builds a coordinate because a bare period has nowhere else
        # to put its clock; apply gives back the shape it was handed.
        out = out.drop_vars(dim)
    extra = dict((name, dataset[name]) for name in dataset.data_vars
                 if name not in read.SERIES)
    return out.assign(extra) if extra else out


# --------------------------------------------------------------------------- internals

def _resolve(step):
    """The step function for a callable or a ``pipeline.STEPS`` name. Returns callable.

    ``pipeline`` is imported here rather than at module scope so that the adapter's own
    import graph stays the core's smallest useful corner (``kernels`` and ``read``) when a
    caller passes the function itself.
    """
    if callable(step):
        return step
    from . import pipeline
    for name, function in pipeline.STEPS:
        if name == step:
            return function
    raise ValueError('%r is not a step: pass a (period, cfg) -> period callable or one of '
                     '%s' % (step, ', '.join(name for name, _f in pipeline.STEPS)))


def _write_back(aliases, result):
    """Put a rebound series back in the caller's buffer, under ``copy=False``. Returns None.

    Two steps hand back a *new* array rather than writing into the one they were given:
    ``rotate`` replaces u, v and w, and ``lag`` replaces each scalar it shifts. Without
    this, ``copy=False`` would deliver the result of every other step and silently not
    those -- a caller who rotated in place and then read their own Dataset would be holding
    unrotated wind, which is the wrong-rather-than-absent failure this boundary exists to
    prevent. Rebinding the period's entry to the caller's buffer afterwards keeps the
    promise that there is one buffer per series: the returned Dataset aliases it too.

    No step changes a series' length (``kernels.shift_truncate`` NaN-fills rather than
    shortening), so one that did is a step this boundary does not know how to alias.
    """
    _period_length(result)          # a ragged or nine-series result, as a ValueError
    for name, original in aliases.items():
        produced = result[name]
        if produced is original:
            continue
        if len(produced) != len(original):
            raise ValueError('the step returned %d sample(s) of %r where it was given %d; '
                             'copy=False cannot land a result of a different length in the '
                             'caller\'s Dataset. Use copy=True.'
                             % (len(produced), name, len(original)))
        original[:] = _ndarray(produced, copy=False)
        result[name] = original


def _period_length(period):
    """Common length of the ten series. Returns int; raises ValueError if they differ."""
    missing = [name for name in read.SERIES if name not in period]
    if missing:
        raise ValueError('the period has no %s; CONTRACT section 1 says a period carries '
                         'all ten series, a missing one as all-NaN rather than as a missing '
                         'key' % ', '.join(repr(name) for name in missing))
    lengths = set(len(period[name]) for name in read.SERIES)
    if len(lengths) != 1:
        raise ValueError('the period\'s series have different lengths (%s); CONTRACT '
                         'section 1 says all ten are always the same length'
                         % ', '.join('%s=%d' % (name, len(period[name]))
                                     for name in read.SERIES))
    return lengths.pop()


def _dim(dataset):
    """The dimension the canonical series lie on. Returns str."""
    for name in read.SERIES:
        if name in dataset.variables:
            variable = dataset[name]
            if variable.ndim != 1:
                raise ValueError('%r has dimensions %r; a miniflux series is one axis of '
                                 'samples' % (name, variable.dims))
            return variable.dims[0]
    raise ValueError('the Dataset carries none of the canonical series (%s); there is no '
                     'period in it' % ', '.join(read.SERIES))


def _unit(name, cfg, meta):
    """The canonical unit of one series in this period. Returns str.

    A gas is ``mol mol-1`` or ``mol m-3`` depending on its measure type, and the type that
    applies depends on where in the pipeline the period is: on a closed-path run,
    ``cell.convert`` rewrites a cell molar density into a dry mixing ratio, and
    ``cfg.gases.measure_type`` is already the *effective*, post-conversion type (CONTRACT
    section 5.1). Before that step has run, the numbers are still what the file reported,
    so the presence of the key ``cell.convert`` always writes when it has work to do is
    what decides which of the two labels is true.
    """
    if name in CANONICAL_UNITS:
        return CANONICAL_UNITS[name]
    if name in cfg.gases.convert_cell and 'n_cell_converted' not in meta:
        return GAS_UNITS[cfg.gases.reported[name]]
    return GAS_UNITS[cfg.gases.measure_type[name]]


def _coordinate(meta, cfg, n):
    """The time coordinate of a period. Returns an ndarray of datetime64[ns] or int64.

    Rebuilt as ``period_start + i / freq``, which for a regular stream is exact arithmetic
    and not data (CONTRACT section 1.2): the period carries no per-sample time, because
    nothing in the program read one. Two things it therefore is not. A **gap** in the
    stream is a missing row, not a missing instant, so every sample after a gap is labelled
    earlier than it was recorded -- ``meta['n_in']`` against the period width is what says
    whether that happened. And with ``[period] closed = right`` the first sample sits one
    interval after ``period_start``, so the origin is the period's boundary, not the first
    sample.

    A period with no ``period_start`` gets a plain integer sample index instead. Inventing
    a clock for it would be worse than admitting there is none.

    The offsets are rounded to the nanosecond one at a time, so a frequency whose interval
    is not a whole number of nanoseconds (3 Hz, say; 10, 20 and 100 Hz are exact) is off by
    at most half a nanosecond per sample and never accumulates.
    """
    start = meta.get('period_start')
    freq = meta.get('freq_hz', cfg.period.acquisition_frequency)
    if start is None or not np.isfinite(freq) or freq <= 0.0:
        return np.arange(n, dtype='int64')
    origin = np.datetime64(start, 'us').astype('datetime64[ns]')
    offsets = np.round(np.arange(n, dtype='float64') * (1.0e9 / freq)).astype('int64')
    return origin + offsets.astype('timedelta64[ns]')


def _series(variable, name, dim, n, cfg, meta, copy):
    """One Dataset variable as a period series. Returns array('d') or ndarray."""
    if variable.dims != (dim,):
        raise ValueError('%r has dimensions %r, not (%r,)' % (name, variable.dims, dim))
    if variable.sizes[dim] != n:
        raise ValueError('%r has %d sample(s), the period has %d'
                         % (name, variable.sizes[dim], n))
    if variable.dtype.kind not in _NUMERIC_KINDS:
        raise ValueError('%r is %s, which is not a measurement this boundary will turn '
                         'into float64: numpy would cast it without complaining and the '
                         'numbers would not be the ones the instrument reported (a '
                         'datetime64 becomes nanoseconds, a bool becomes 1.0, a complex '
                         'loses its imaginary part). Convert it yourself if the numbers '
                         'really are %s.' % (name, variable.dtype, _unit(name, cfg, meta)))
    declared = variable.attrs.get('units')
    if declared is not None:
        _check_unit(name, declared, _unit(name, cfg, meta))
    if copy:
        # tobytes/frombytes rather than array('d', values): bit-exact for a float64 input,
        # and an explicit widening cast for anything else, which is the one thing
        # np.frombuffer would not do -- see _alias.
        out = array('d')
        out.frombytes(np.ascontiguousarray(variable.values, dtype=np.float64).tobytes())
        return out
    return _alias(variable, name)


def _alias(variable, name):
    """The variable's own buffer, if a step may safely write into it. Returns ndarray.

    ``kernels`` reaches numpy through ``np.frombuffer(x, dtype=float64)``, which raises on a
    list, on a DataArray and on a strided slice -- but **not** on a float32 array: it
    reinterprets the bytes and returns numbers that are wrong rather than absent. So the
    dtype is checked here, explicitly, instead of being left to fail downstream.
    """
    if variable.chunks:
        raise ValueError('%r is dask-backed; copy=False cannot alias it, because .values '
                         'would compute a fresh array and the step\'s writes would land in '
                         'a temporary. Use copy=True (the default).' % name)
    values = variable.values
    if not isinstance(values, np.ndarray):
        raise TypeError('%r is backed by %s, not a numpy array; use copy=True'
                        % (name, type(values).__name__))
    if values.dtype != np.float64:
        raise ValueError('%r is %s; copy=False needs float64, and a narrower dtype cannot '
                         'be aliased -- miniflux\'s kernels would reinterpret its bytes as '
                         'float64 and compute wrong numbers from them. Use copy=True, which '
                         'casts.' % (name, values.dtype))
    if not values.flags['C_CONTIGUOUS']:
        raise ValueError('%r is not C-contiguous (a strided slice or a transpose); '
                         'copy=False cannot alias it. Use copy=True.' % name)
    if not values.flags['WRITEABLE']:
        raise ValueError('%r is read-only; a step writes in place. Use copy=True.' % name)
    return values


def _ndarray(x, copy):
    """A period series as a float64 ndarray. Returns ndarray.

    Copying is a real copy; aliasing an ``array('d')`` is the same zero-copy
    ``np.frombuffer`` view ``kernels`` itself takes, and the buffer stays alive because the
    view holds a reference to it.
    """
    if copy:
        return np.array(x, dtype=np.float64)
    if isinstance(x, np.ndarray):
        return x
    return np.frombuffer(x, dtype=np.float64)


def _check_unit(name, declared, canonical):
    """Refuse a declared unit that is not the canonical one. Returns None."""
    spelling = str(declared).strip()
    if spelling == canonical or spelling in _SYNONYMS.get(canonical, ()):
        return
    raise ValueError(
        '%r is declared as %r but miniflux\'s canonical unit for it is %r (CONTRACT '
        'section 2). Convert the numbers, or fix the attribute if they are already in %r; '
        'this boundary will not reinterpret them.' % (name, declared, canonical, canonical))
