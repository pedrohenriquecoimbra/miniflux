"""What CONTRACT section 21 pins for ``detrend``: the block mean is stored and the series
is left bit-identical, and a linear detrend on a *gapped* series matches the closed-form
line on the original sample index.

The gapped case is the point. :func:`compacted_line` below reproduces the parent's fit --
least squares against the index of the NaN-deleted array, evaluated on the full-length
index -- and the tests show it does not satisfy the assertions miniflux does
(CONTRACT section 20.2, ALGORITHMS section 6.2).
"""

import math
import unittest
from datetime import datetime
from types import SimpleNamespace

from miniflux import detrend, kernels

NAN = float('nan')
SERIES = ('u', 'v', 'w', 'ts', 'co2', 'h2o', 'ta', 'p_air')
DETRENDABLE = ('u', 'v', 'w', 'ts', 'co2', 'h2o')


def make_period(**series):
    """A period dict carrying all nine series keys; unnamed ones are all-NaN."""
    n = max(len(v) for v in series.values())
    period = {'meta': {}, 't': [datetime(2020, 6, 1, 12, 0)] * n}
    for name in SERIES:
        period[name] = kernels.from_values(series[name]) if name in series else kernels.new(n)
    return period


def make_cfg(method, variables):
    """The slice of the Config that detrend is allowed to read."""
    return SimpleNamespace(detrend=SimpleNamespace(method=method,
                                                   variables=list(variables)))


def compacted_line(x):
    """The parent's fit: least squares against the NaN-deleted index (slope per *kept*
    sample), which the parent then evaluates on the full-length index. Returns
    (slope, offset). Identical to the miniflux fit only when nothing is missing."""
    vals = [v for v in x if math.isfinite(v)]
    m = len(vals)
    jbar = (m - 1) / 2.0                      # the compacted index is 0, 1, ... m-1
    xbar = sum(vals) / m
    num = sum((j - jbar) * (v - xbar) for j, v in enumerate(vals))
    den = sum((j - jbar) ** 2 for j in range(m))
    slope = num / den
    return slope, xbar - slope * jbar


def residuals(x, slope, offset):
    """x_i - (slope*i + offset) on the original index, NaN where x is NaN."""
    return [v - (slope * i + offset) for i, v in enumerate(x)]


def max_abs_finite(values):
    """Largest |value| over the finite entries; 0.0 when there are none."""
    finite = [abs(v) for v in values if math.isfinite(v)]
    return max(finite) if finite else 0.0


class BlockAverageIsBookkeeping(unittest.TestCase):
    """ALGORITHMS 6.1 / CONTRACT 20.6: compute the mean, store it, touch nothing."""

    def setUp(self):
        # mean 2.5 exactly: no rounding hides in the comparison below.
        self.period = make_period(u=[1.0, 2.0, NAN, 3.0, 4.0],
                                  v=[-1.0, 0.0, 1.0, 2.0, NAN],
                                  w=[0.5, 0.25, 0.125, NAN, NAN])
        self.cfg = make_cfg('block', ('u', 'v', 'w'))

    def test_series_stay_bit_identical(self):
        before = dict((name, bytes(self.period[name])) for name in ('u', 'v', 'w'))
        arrays = dict((name, self.period[name]) for name in ('u', 'v', 'w'))
        detrend.detrend(self.period, self.cfg)
        for name in ('u', 'v', 'w'):
            self.assertEqual(bytes(self.period[name]), before[name],
                             '%s was modified by a block average' % name)
            self.assertIs(self.period[name], arrays[name],
                          '%s was reallocated rather than left in place' % name)

    def test_means_are_stored(self):
        detrend.detrend(self.period, self.cfg)
        self.assertEqual(self.period['meta']['u_mean'], 2.5)
        self.assertEqual(self.period['meta']['v_mean'], 0.5)
        self.assertEqual(self.period['meta']['w_mean'], (0.5 + 0.25 + 0.125) / 3.0)

    def test_mean_matches_the_kernel_bit_for_bit(self):
        # CONTRACT 17: thermodynamics has already written ts_mean with this same kernel,
        # so the overwrite has to be the identical float.
        period = make_period(ts=[290.15, 291.4, NAN, 292.05, 289.9])
        expected = kernels.nanmean(period['ts'])
        detrend.detrend(period, make_cfg('block', ('ts',)))
        self.assertEqual(period['meta']['ts_mean'], expected)

    def test_only_the_configured_variables_are_summarised(self):
        detrend.detrend(self.period, make_cfg('block', ('u',)))
        self.assertIn('u_mean', self.period['meta'])
        self.assertNotIn('v_mean', self.period['meta'])

    def test_all_nan_series_gives_nan_and_a_warning(self):
        period = make_period(u=[NAN, NAN, NAN])
        with self.assertLogs('miniflux.detrend', level='WARNING') as caught:
            detrend.detrend(period, make_cfg('block', ('u',)))
        self.assertTrue(math.isnan(period['meta']['u_mean']))
        self.assertIn('no finite sample', caught.output[0])


class LinearDetrendOnTheOriginalIndex(unittest.TestCase):
    """ALGORITHMS 6.2 / CONTRACT 20.2: fit and evaluate on the original sample index."""

    # x_i = A + B*i exactly, with a three-sample gap near the end. The gap is what
    # separates the two fits: over the compacted index the last kept sample sits at
    # j = 8 while it really belongs at i = 11, so a compacted fit reads a steeper trend.
    A = 10.0
    B = 0.25
    N = 12
    GAP = (8, 9, 10)

    def line_with_gap(self):
        return [NAN if i in self.GAP else self.A + self.B * i for i in range(self.N)]

    def test_gapped_series_matches_the_closed_form_line(self):
        values = self.line_with_gap()
        period = make_period(co2=values)
        detrend.detrend(period, make_cfg('linear', ('co2',)))
        # The data IS the line A + B*i on the original index, so the exact fit removes it
        # completely: every finite residual is zero.
        self.assertEqual(len(period['co2']), self.N)
        self.assertLess(max_abs_finite(period['co2']), 1e-12)
        for i in self.GAP:
            self.assertTrue(math.isnan(period['co2'][i]), 'gap at %d was filled' % i)

    def test_the_compacted_fit_would_fail_that_assertion(self):
        # The same series through the parent's abscissa: the removed trend is too steep by
        # N/m, so the residuals are a leftover ramp, not zero. This is the bug 20.2 fixes.
        values = self.line_with_gap()
        slope, offset = compacted_line(kernels.from_values(values))
        left = residuals(values, slope, offset)
        self.assertGreater(max_abs_finite(left), 0.1)
        self.assertNotAlmostEqual(slope, self.B, places=6)

    def test_the_two_fits_agree_when_nothing_is_missing(self):
        values = [self.A + self.B * i for i in range(self.N)]
        period = make_period(co2=list(values))
        detrend.detrend(period, make_cfg('linear', ('co2',)))
        slope, offset = compacted_line(kernels.from_values(values))
        self.assertLess(max_abs_finite(period['co2']), 1e-12)
        self.assertLess(max_abs_finite(residuals(values, slope, offset)), 1e-12)

    def test_a_trend_plus_a_fluctuation_leaves_the_fluctuation(self):
        # A gapped ramp with a known zero-mean wiggle on top: detrending must return the
        # wiggle, evaluated at the original index of each sample.
        wiggle = [0.4, -0.4, 0.1, -0.1, 0.3, -0.3, 0.2, -0.2, 0.0, 0.0, 0.5, -0.5]
        values = [NAN if i in self.GAP else self.A + self.B * i + wiggle[i]
                  for i in range(self.N)]
        period = make_period(h2o=list(values))
        detrend.detrend(period, make_cfg('linear', ('h2o',)))
        slope, offset = kernels.linfit(kernels.from_values(values))
        expected = residuals(values, slope, offset)
        for i in range(self.N):
            if math.isnan(expected[i]):
                self.assertTrue(math.isnan(period['h2o'][i]))
            else:
                self.assertAlmostEqual(period['h2o'][i], expected[i], places=12)
        # ... and the fluctuation is what is left: the mean of the residuals is ~0 but the
        # series is not, so the detrend removed a trend rather than the whole signal.
        self.assertGreater(max_abs_finite(period['h2o']), 0.1)

    def test_the_stored_mean_is_the_pre_detrend_mean(self):
        values = self.line_with_gap()
        expected = kernels.nanmean(kernels.from_values(values))
        period = make_period(co2=values)
        detrend.detrend(period, make_cfg('linear', ('co2',)))
        self.assertEqual(period['meta']['co2_mean'], expected)
        # the detrended series itself is centred on zero, which is why the mean had to be
        # taken first
        self.assertLess(abs(kernels.nanmean(period['co2'])), 1e-12)

    def test_fewer_than_three_finite_samples_removes_the_mean(self):
        values = [4.0, NAN, NAN, 6.0]
        period = make_period(ts=list(values))
        with self.assertLogs('miniflux.detrend', level='WARNING') as caught:
            detrend.detrend(period, make_cfg('linear', ('ts',)))
        self.assertEqual(period['meta']['ts_mean'], 5.0)
        self.assertEqual(period['ts'][0], -1.0)
        self.assertEqual(period['ts'][3], 1.0)
        self.assertIn('fewer than the 3', caught.output[0])

    def test_all_nan_series_is_left_alone(self):
        period = make_period(u=[NAN, NAN, NAN, NAN])
        before = bytes(period['u'])
        with self.assertLogs('miniflux.detrend', level='WARNING'):
            detrend.detrend(period, make_cfg('linear', ('u',)))
        self.assertEqual(bytes(period['u']), before)
        self.assertTrue(math.isnan(period['meta']['u_mean']))

    def test_untouched_variables_keep_their_samples(self):
        period = make_period(u=[1.0, 2.0, 3.0, 4.0], v=[5.0, 6.0, 7.0, 8.0])
        before = bytes(period['v'])
        detrend.detrend(period, make_cfg('linear', ('u',)))
        self.assertEqual(bytes(period['v']), before)


class TheStepContract(unittest.TestCase):
    """CONTRACT 1.1: a step returns the period and never raises on degenerate data."""

    def test_returns_the_same_dict(self):
        period = make_period(u=[1.0, 2.0, 3.0])
        self.assertIs(detrend.detrend(period, make_cfg('block', ('u',))), period)

    def test_an_empty_variable_list_is_a_no_op(self):
        period = make_period(u=[1.0, 2.0, 3.0])
        detrend.detrend(period, make_cfg('linear', ()))
        self.assertEqual(period['meta'], {})

    def test_every_detrendable_variable_can_be_detrended(self):
        # The six turbulence series and no more: `ta` and `p_air` are refused by
        # config._validate, because `ta_mean` and `p_air_mean` are thermodynamic means
        # this module does not own (tests/test_config.py pins that refusal).
        period = make_period(u=[1.0, 2.0, 3.0, 4.0])
        with self.assertLogs('miniflux.detrend', level='WARNING'):
            detrend.detrend(period, make_cfg('linear', DETRENDABLE))   # five all-NaN but u
        for name in DETRENDABLE:
            self.assertIn(name + '_mean', period['meta'])


if __name__ == '__main__':
    unittest.main()
