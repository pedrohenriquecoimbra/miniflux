"""The array layer: the only module that knows about ``array('d')`` layout, NaN
bookkeeping and numpy. Everything that touches more than one sample goes through here.

Two estimator conventions live side by side, on purpose (ALGORITHMS.md sections 0.4,
5.3, 7.1):

* :func:`cov` (and :func:`variance`, :func:`nanstd`) is the *reported moment*
  estimator: joint finite mask, both means over that mask, two-pass, ``/(Np - 1)``.
  It is translation-invariant, which is the whole point on a 400 ppm CO2 series.
* :func:`cov_at_lag` and :func:`cov_curve` are the *scan* estimator: one-pass raw
  sums, which is what both reference engines compute. They are only ever called on
  series the caller has already centred, which removes the cancellation the one-pass
  form would otherwise suffer.

Missing is NaN, always. A comparison against NaN is False, so NaN is never flagged and
never selected as an extremum. Infinities count as non-finite everywhere.

numpy, when present and enabled, is a speed switch and nothing else: same decisions
bit for bit, reductions within 8 ULP (ALGORITHMS.md section 12).
"""

import logging
import math
from array import array

from .constants import MAD_SCALE

logger = logging.getLogger(__name__)

try:
    import numpy as _np
except ImportError:  # the standard-library path is the reference path
    _np = None

NAN = float('nan')

#: True when ``import numpy`` succeeded. Fixed at import; ``config.py`` reads it to
#: decide whether ``[runtime] use_numpy = on`` can be honoured.
NUMPY_AVAILABLE = _np is not None

#: True when the fast path is actually in use. Resolved once, at ``config.load()``
#: time, through :func:`set_numpy`.
HAVE_NUMPY = NUMPY_AVAILABLE


def set_numpy(enabled):
    """Turn the numpy fast path on or off. Returns the resolved ``HAVE_NUMPY``.

    Asking for numpy when it is not installed leaves the pure path in place and logs;
    refusing the run is ``config.py``'s decision, not this module's.
    """
    global HAVE_NUMPY
    if enabled and not NUMPY_AVAILABLE:
        logger.warning("numpy fast path requested but numpy is not importable; "
                       "running on the pure-Python path")
    HAVE_NUMPY = bool(enabled) and NUMPY_AVAILABLE
    return HAVE_NUMPY


def _span(x, start, stop):
    """(start, stop) clamped to len(x), with Python slice semantics."""
    return slice(start, stop).indices(len(x))[:2]


def _pair_span(x, y, start, stop):
    """(start, stop) clamped to the shorter of the two series."""
    return slice(start, stop).indices(min(len(x), len(y)))[:2]


def _view(x, start=0, stop=None):
    """Zero-copy float64 view of ``x[start:stop]``.

    ``array('d')`` exposes a writable buffer, so the view aliases the caller's memory
    and the in-place kernels stay in place.
    """
    lo, hi = _span(x, start, stop)
    if not len(x):
        return _np.empty(0, dtype=_np.float64)
    return _np.frombuffer(x, dtype=_np.float64)[lo:hi]


def _np_finite(a):
    """The finite values of an ndarray, in order."""
    return a[_np.isfinite(a)]


# --- 3.1 construction (allocation, not arithmetic: no numpy path) -------------------

def new(n, fill=NAN):
    """Fresh array('d') of length n, every element `fill` (NaN by default)."""
    return array('d', [fill]) * n


def from_values(values):
    """array('d') from an iterable of float|None; None becomes NaN."""
    return array('d', [NAN if v is None else float(v) for v in values])


def copy(x):
    """Independent copy of an array('d')."""
    return array('d', x)


# --- 3.2 reductions ----------------------------------------------------------------

def count_finite(x, start=0, stop=None):
    """How many samples of ``x[start:stop]`` are finite. Returns int."""
    if HAVE_NUMPY:
        return int(_np.count_nonzero(_np.isfinite(_view(x, start, stop))))
    lo, hi = _span(x, start, stop)
    k = 0
    for i in range(lo, hi):
        if math.isfinite(x[i]):
            k += 1
    return k


def nansum(x, start=0, stop=None):
    """(sum of the finite samples, how many there were) over ``x[start:stop]``.

    Left to right, uncompensated: a Kahan sum could not be matched by numpy's pairwise
    sum to the 8 ULP the two paths owe each other.
    """
    if HAVE_NUMPY:
        v = _np_finite(_view(x, start, stop))
        return float(v.sum()), len(v)
    lo, hi = _span(x, start, stop)
    total = 0.0
    k = 0
    for i in range(lo, hi):
        v = x[i]
        if math.isfinite(v):
            total += v
            k += 1
    return total, k


def nanmean(x, start=0, stop=None):
    """Mean of the finite samples of ``x[start:stop]``; NAN when there are none."""
    total, k = nansum(x, start, stop)
    return total / k if k else NAN


def _median_of_sorted(v):
    """Median of an ascending sequence: the parity is that of len(v)."""
    k = len(v)
    if k == 0:
        return NAN
    if k % 2:
        return float(v[k // 2])
    return float((v[k // 2 - 1] + v[k // 2]) / 2)


def nanmedian(x):
    """Median of the finite samples of x; NAN when there are none.

    The non-finite values are dropped *first*, so the even/odd rule runs on the
    surviving count k, not on len(x).
    """
    if HAVE_NUMPY:
        return _median_of_sorted(_np.sort(_np_finite(_view(x))))
    return _median_of_sorted(sorted(v for v in x if math.isfinite(v)))


def mad(x, med):
    """median(|x_i - med|) over the finite x_i. No 0.6745 scaling here.

    Units: the unit of x. NAN when nothing is finite, and NAN when med is NAN.
    """
    if HAVE_NUMPY:
        return _median_of_sorted(_np.sort(_np.abs(_np_finite(_view(x)) - med)))
    return _median_of_sorted(sorted(abs(v - med) for v in x if math.isfinite(v)))


# --- 3.3 second moments ------------------------------------------------------------

def cov(x, y, start=0, stop=None):
    """Sample covariance of x and y over ``[start, stop)``; unit(x)*unit(y).

    The one reported-moment recipe (ALGORITHMS section 7.1): the joint finite mask,
    both means taken over that same mask, two passes, divided by ``Np - 1``. NAN when
    fewer than two joint-finite pairs remain. Two-pass because the one-pass raw-sums
    form cancels catastrophically on a large mean with a small fluctuation.
    """
    lo, hi = _pair_span(x, y, start, stop)
    if HAVE_NUMPY:
        a = _view(x, lo, hi)
        b = _view(y, lo, hi)
        m = _np.isfinite(a) & _np.isfinite(b)
        a = a[m]
        b = b[m]
        n = len(a)
        if n < 2:
            return NAN
        da = a - a.sum() / n
        db = b - b.sum() / n
        return float((da * db).sum() / (n - 1))
    sx = 0.0
    sy = 0.0
    n = 0
    for i in range(lo, hi):
        xi = x[i]
        yi = y[i]
        if math.isfinite(xi) and math.isfinite(yi):
            sx += xi
            sy += yi
            n += 1
    if n < 2:
        return NAN
    xbar = sx / n
    ybar = sy / n
    acc = 0.0
    for i in range(lo, hi):
        xi = x[i]
        yi = y[i]
        if math.isfinite(xi) and math.isfinite(yi):
            acc += (xi - xbar) * (yi - ybar)
    return acc / (n - 1)


def variance(x, start=0, stop=None):
    """Sample variance, ddof = 1. Exactly ``cov(x, x, start, stop)``."""
    return cov(x, x, start, stop)


def nanstd(x, start=0, stop=None):
    """Sample standard deviation, ddof = 1. sqrt of :func:`variance`."""
    v = variance(x, start, stop)
    return math.sqrt(v) if v == v else NAN


# --- 3.4 despiking primitives ------------------------------------------------------

def mad_spike_mask(x, q):
    """1 where x lies strictly outside the MAD bounds, 0 elsewhere. Returns bytearray.

    Mauder et al. (2013). ``lower = m - (q*d)/0.6745`` and ``upper = m + (q*d)/0.6745``
    with m the period median and d the MAD. The parenthesisation is part of the number:
    ``(q*d)/0.6745``, ``q*(d/0.6745)`` and ``q*d*1.4826`` are three different doubles.
    Both comparisons are strict, so a sample exactly on a bound survives. Only a *finite*
    sample can be flagged: NaN fails every comparison anyway, and an infinity is missing
    data like every other non-finite value here, so flagging one would make despike.py's
    counter claim a removal that removed nothing -- and would let a single trailing inf
    carry the last flag, which the trailing-run rule then walks back, handing a real spike
    run to the flux. A constant series has d = 0 and flags nothing; a nearly constant one
    has d = 0 and flags everything that differs at all -- deliberately unguarded.

    The trailing-run rule belongs to despike.py, not here.
    """
    m = nanmedian(x)
    d = mad(x, m)
    lower = m - (q * d) / MAD_SCALE
    upper = m + (q * d) / MAD_SCALE
    if HAVE_NUMPY:
        a = _view(x)
        return bytearray((_np.isfinite(a) & ((a < lower) | (a > upper))).tobytes())
    mask = bytearray(len(x))
    for i, v in enumerate(x):
        if math.isfinite(v) and (v < lower or v > upper):
            mask[i] = 1
    return mask


def apply_mask_nan(x, mask):
    """In place: ``x[i] = NaN`` wherever ``mask[i]`` is non-zero. mask is len(x) long."""
    if HAVE_NUMPY:
        a = _view(x)
        a[_np.frombuffer(mask, dtype=_np.uint8) != 0] = NAN
        return
    for i in range(len(mask)):
        if mask[i]:
            x[i] = NAN


# --- 3.5 series transforms ---------------------------------------------------------

def shift_truncate(x, lag):
    """``out[i] = x[i+lag]`` where that index exists, NaN elsewhere. Never wraps.

    lag is the physical lag in samples: positive means the scalar arrives after the
    wind, so the sample recorded `lag` steps later belongs with the wind now. The
    period is |lag| samples shorter for that variable, and the NaN edge is excluded by
    the joint mask of every statistic downstream.
    """
    n = len(x)
    out = new(n)
    lo = max(0, -lag)
    hi = min(n, n - lag)
    if hi <= lo:
        return out
    if HAVE_NUMPY:
        _view(out, lo, hi)[:] = _view(x, lo + lag, hi + lag)
        return out
    for i in range(lo, hi):
        out[i] = x[i + lag]
    return out


def sub_const(x, c):
    """``out[i] = x[i] - c``, as a new array('d')."""
    out = new(len(x))
    if not len(x):
        return out
    if HAVE_NUMPY:
        _view(out)[:] = _view(x) - c
        return out
    for i, v in enumerate(x):
        out[i] = v - c
    return out


def sub_const_inplace(x, c):
    """In place: ``x[i] -= c``."""
    if not len(x):
        return
    if HAVE_NUMPY:
        a = _view(x)
        a -= c
        return
    for i in range(len(x)):
        x[i] -= c


def sub_line_inplace(x, slope, offset):
    """In place: ``x[i] -= slope*i + offset``, over the original sample index."""
    n = len(x)
    if not n:
        return
    if HAVE_NUMPY:
        a = _view(x)
        a -= slope * _np.arange(n, dtype=_np.float64) + offset
        return
    for i in range(n):
        x[i] -= slope * i + offset


def linfit(x):
    """Least-squares line through the finite samples of x: returns (slope, offset).

    slope is in unit(x) per sample, offset is unit(x) at i = 0. The fit runs on the
    **original** sample index, not on the compacted one, so a gap does not stretch the
    trend (ALGORITHMS section 6.2). Fewer than three finite samples gives
    ``(0.0, nanmean(x))`` -- mean removal, no trend; no finite sample gives (NAN, NAN).
    """
    m = count_finite(x)
    if m == 0:
        return NAN, NAN
    if m < 3:
        return 0.0, nanmean(x)
    if HAVE_NUMPY:
        a = _view(x)
        keep = _np.isfinite(a)
        idx = _np.arange(len(x), dtype=_np.float64)[keep]
        val = a[keep]
        ibar = float(idx.sum()) / m
        xbar = float(val.sum()) / m
        di = idx - ibar
        slope = float((di * (val - xbar)).sum()) / float((di * di).sum())
        return slope, xbar - slope * ibar
    si = 0.0
    sx = 0.0
    for i, v in enumerate(x):
        if math.isfinite(v):
            si += i
            sx += v
    ibar = si / m
    xbar = sx / m
    num = 0.0
    den = 0.0
    for i, v in enumerate(x):
        if math.isfinite(v):
            di = i - ibar
            num += di * (v - xbar)
            den += di * di
    slope = num / den
    return slope, xbar - slope * ibar


def rotate_double(u, v, w):
    """Double rotation, Wilczak et al. (2001). Returns (u2, v2, w2, theta, phi).

    u, v, w are sonic-frame wind [m s-1]; u2, v2, w2 are streamline-frame wind
    [m s-1]; theta (yaw) and phi (pitch) are in radians.

        theta = atan2(mean v, mean u)
        u1 = u cos(theta) + v sin(theta);  v1 = -u sin(theta) + v cos(theta);  w1 = w
        phi   = atan2(mean w1, mean u1)
        u2 = u1 cos(phi) + w1 sin(phi);  v2 = v1;  w2 = -u1 sin(phi) + w1 cos(phi)

    The two rotations are applied in sequence, not as one pre-multiplied matrix, so
    the arithmetic matches the reference implementation operation for operation. phi
    needs the mean of the rotated u1, so u1 is materialised before the second pass --
    the closed form phi = atan2(mean w, hypot(mean u, mean v)) is a different angle
    whenever the three series carry different NaN masks, which despiking makes routine.
    atan2, never atan: with atan a period whose mean u is negative rotates to
    u = -|wind| and flips the sign of <u'w'> on exactly those periods.

    Every output mixes at least two inputs, so the output NaN mask is the union of the
    input masks. An all-NaN period gives theta = phi = NaN and all-NaN output; a period
    with mean u = mean v = 0 gives atan2(0, 0) = 0.0, not an exception.

    Three series of unequal length is the one input this refuses. It cannot arrive from
    inside miniflux -- read.py guarantees equal lengths and no step changes one -- but a
    caller that got it wrong must hear the same answer on both paths, and numpy would
    broadcast a length-1 component silently across the period.
    """
    if not (len(u) == len(v) == len(w)):
        raise ValueError('rotate_double needs three series of the same length, got '
                         '%d / %d / %d' % (len(u), len(v), len(w)))
    n = len(u)
    theta = math.atan2(nanmean(v), nanmean(u))
    ct = math.cos(theta)
    st = math.sin(theta)
    u1 = new(n)
    v1 = new(n)
    if HAVE_NUMPY and n:
        au = _view(u)
        av = _view(v)
        # an infinity in the data gives inf - inf = NaN, which is what the pure path
        # does silently; errstate keeps numpy from warning about a non-event.
        with _np.errstate(invalid='ignore'):
            _view(u1)[:] = au * ct + av * st
            _view(v1)[:] = -au * st + av * ct
    else:
        for i in range(n):
            ui = u[i]
            vi = v[i]
            u1[i] = ui * ct + vi * st
            v1[i] = -ui * st + vi * ct
    phi = math.atan2(nanmean(w), nanmean(u1))   # w1 is w, untouched by the first rotation
    cp = math.cos(phi)
    sp = math.sin(phi)
    u2 = new(n)
    w2 = new(n)
    if HAVE_NUMPY and n:
        a1 = _view(u1)
        aw = _view(w)
        with _np.errstate(invalid='ignore'):
            _view(u2)[:] = a1 * cp + aw * sp
            _view(w2)[:] = -a1 * sp + aw * cp
    else:
        for i in range(n):
            a = u1[i]
            b = w[i]
            u2[i] = a * cp + b * sp
            w2[i] = -a * sp + b * cp
    return u2, v1, w2, theta, phi


# --- 3.6 lag search ----------------------------------------------------------------

def cov_at_lag(w, c, lag):
    """Covariance of w with c displaced by `lag` samples: ``cov(w_i, c_{i+lag})``.

    The scan estimator (ALGORITHMS section 5.3): overlap only, never a wraparound;
    the joint finite mask, the count k and both sums are recomputed on that overlap at
    every lag; one-pass form ``(Sab - Sa*Sb/k)/(k-1)``; NAN when k < 2.

    The caller pre-centres both series (lag.py subtracts each whole-period nan-mean),
    which changes no covariance in exact arithmetic and removes the cancellation the
    one-pass form would otherwise suffer on a 400 ppm series.
    """
    n = min(len(w), len(c))
    lo = max(0, -lag)             # first index of w whose partner c[i+lag] exists
    hi = min(n, n - lag)          # one past the last such index
    if hi - lo < 2:
        return NAN
    if HAVE_NUMPY:
        a = _view(w, lo, hi)
        b = _view(c, lo + lag, hi + lag)
        m = _np.isfinite(a) & _np.isfinite(b)
        a = a[m]
        b = b[m]
        k = len(a)
        if k < 2:
            return NAN
        return float(((a * b).sum() - a.sum() * b.sum() / k) / (k - 1))
    sa = 0.0
    sb = 0.0
    sab = 0.0
    k = 0
    for i in range(lo, hi):
        ai = w[i]
        bi = c[i + lag]
        if math.isfinite(ai) and math.isfinite(bi):
            sa += ai
            sb += bi
            sab += ai * bi
            k += 1
    if k < 2:
        return NAN
    return (sab - sa * sb / k) / (k - 1)


def cov_curve(w, c, lo, hi):
    """:func:`cov_at_lag` for every lag in [lo, hi]: a list of ``hi - lo + 1`` floats.

    Both ends inclusive -- a symmetric window ``L_nom +- I`` holds 2I+1 lags, which is
    EddyPro's inclusive Fortran loop. The caller swaps a reversed window itself; an
    empty curve would silently become "no peak found", so a reversed one is refused.
    """
    if hi < lo:
        raise ValueError("cov_curve needs lo <= hi, got lo=%r hi=%r" % (lo, hi))
    return [cov_at_lag(w, c, lag) for lag in range(lo, hi + 1)]
