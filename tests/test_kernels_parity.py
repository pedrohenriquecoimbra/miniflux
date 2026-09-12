"""The numpy fast path and the pure path must agree (CONTRACT section 3.8).

Three obligations, and they are not the same obligation:

* decisions -- the spike mask and the selected lag -- are exact on both paths, because
  they are comparisons, not sums;
* elementwise transforms are the same IEEE operations in the same order, so they come
  out identical value by value;
* reductions may differ in the last bits (numpy sums pairwise, the pure path sums left
  to right); the contract is 8 ULP and this asserts 1e-12 relative.

Skipped in full when numpy is not installed.
"""

import math
import random
import unittest

from miniflux import kernels, lag

NAN = float('nan')
INF = float('inf')


def series(seed, n=1200, sigma=1.0, mean=0.0, holes=True):
    """A deterministic test series, with NaN gaps and one infinity when holes is set."""
    rng = random.Random(seed)
    values = [mean + rng.gauss(0.0, sigma) for _ in range(n)]
    if holes:
        for i in range(7, n, 53):
            values[i] = None
        values[n // 3] = INF          # non-finite, so every reduction must skip it
    return kernels.from_values(values)


def exact_series(seed, n=256, offset=0.0):
    """Small integers scaled by a power of two: every partial sum is exact, so the two
    summation orders give the same mean and the derived angles are identical."""
    rng = random.Random(seed)
    return kernels.from_values([offset + rng.randint(-64, 64) / 8.0 for _ in range(n)])


@unittest.skipUnless(kernels.NUMPY_AVAILABLE, "numpy is not installed")
class KernelParityTest(unittest.TestCase):

    def setUp(self):
        self.x = series(101)
        self.y = series(102, mean=0.5)
        self.big = series(103, mean=4.2e-4, sigma=1e-6)      # the 400 ppm CO2 shape

    def tearDown(self):
        kernels.set_numpy(kernels.NUMPY_AVAILABLE)

    # -- helpers --------------------------------------------------------------------

    def pure(self, fn, *args):
        kernels.set_numpy(False)
        return fn(*args)

    def fast(self, fn, *args):
        kernels.set_numpy(True)
        return fn(*args)

    def assertIdentical(self, a, b, msg=''):
        """Value-for-value equality, NaN matching NaN."""
        self.assertEqual(len(a), len(b))
        for i, (p, q) in enumerate(zip(a, b)):
            if p != p and q != q:
                continue
            self.assertEqual(p, q, "%sindex %d: %r vs %r" % (msg, i, p, q))

    def assertClose(self, a, b, tol=1e-12, msg='', scale=None):
        """Scalar agreement within `tol` of `scale`; NaN must match NaN.

        `scale` defaults to `max(|a|, |b|)`, the plain relative test. Pass the scale of
        the terms that were *summed* whenever the result is a difference of nearly equal
        numbers: two summation orders agree to a few ULP of what they accumulated, which
        is not a few ULP of a result that cancelled down to a millionth of it. A
        covariance on a nearly uncorrelated pair is exactly that, and at a real period
        length it is not a rare case.
        """
        if a != a or b != b:
            self.assertTrue(a != a and b != b, "%s%r vs %r (one is NaN)" % (msg, a, b))
            return
        if scale is None:
            scale = max(abs(a), abs(b))
        if scale == 0.0:
            self.assertEqual(a, b, msg)
            return
        self.assertLessEqual(abs(a - b) / scale, tol,
                             "%s%r vs %r differ by more than %g of %g" % (msg, a, b, tol,
                                                                          scale))

    # -- decisions: exact ------------------------------------------------------------

    def test_spike_mask_decisions_are_identical(self):
        spiked = kernels.copy(self.x)
        spiked[500] = 50.0
        spiked[501] = -50.0
        for q in (3.5, 7.0, 20.0):
            a = self.pure(kernels.mad_spike_mask, spiked, q)
            b = self.fast(kernels.mad_spike_mask, spiked, q)
            self.assertEqual(a, b, "q = %r" % q)
            self.assertEqual(a[500], 1)

    def test_the_selected_lag_is_identical(self):
        w = series(201, n=800, holes=False)
        c = kernels.shift_truncate(w, -12)
        c[300] = NAN
        curve_pure = self.pure(kernels.cov_curve, w, c, -25, 25)
        curve_fast = self.fast(kernels.cov_curve, w, c, -25, 25)
        pick = lambda curve: max(range(len(curve)), key=lambda j: abs(curve[j]))
        self.assertEqual(pick(curve_pure), pick(curve_fast))
        self.assertEqual(pick(curve_pure) - 25, 12)

    def test_the_lag_lag_py_selects_is_identical(self):
        # CONTRACT 3.8 names "the lag selected by lag.find_lag" as one of the three
        # decisions that must match exactly, and the covariance values it is taken from
        # are only promised to 1e-12 -- which is the width inside which two near-tied
        # lags can be ordered differently. So the decision is asserted, not just the
        # curve it comes from.
        w = series(201, n=2000, holes=False)
        c = kernels.shift_truncate(w, -12)
        c[300] = NAN
        wc = kernels.sub_const(w, kernels.nanmean(w))
        cc = kernels.sub_const(c, kernels.nanmean(c))
        pure = self.pure(lag.find_lag, wc, cc, -25, 25, 0, 'covmax_default')
        fast = self.fast(lag.find_lag, wc, cc, -25, 25, 0, 'covmax_default')
        self.assertEqual(pure[:3], fast[:3])          # applied, opt, default_used
        self.assertEqual(pure[1], 12)
        # And a flat curve, where no lag stands out and the tie-break decides alone.
        flat = series(202, n=2000, holes=False)
        other = series(203, n=2000, sigma=1e-6, holes=False)
        fc = kernels.sub_const(flat, kernels.nanmean(flat))
        oc = kernels.sub_const(other, kernels.nanmean(other))
        self.assertEqual(self.pure(lag.find_lag, fc, oc, -30, 30, 0, 'covmax')[:3],
                         self.fast(lag.find_lag, fc, oc, -30, 30, 0, 'covmax')[:3])

    def test_the_selected_lag_survives_a_mathematical_tie(self):
        # A palindromic period (w[i] = w[n-1-i], and the same for c) makes the pair
        # multiset at +L and at -L identical, so cov(+L) and cov(-L) are equal in exact
        # arithmetic and differ only in the order the two paths summed them. The lowest
        # lag has to win on both, or CONTRACT 3.8's "the lag selected by lag.find_lag"
        # is decided by rounding rather than by ALGORITHMS 5.4's tie rule.
        for seed in (4, 5, 7, 19, 23):
            rng = random.Random(seed)
            half = [rng.gauss(0.0, 0.3) for _ in range(300)]
            other = [rng.gauss(0.0, 1e-6) for _ in range(300)]
            w = kernels.from_values(half + half[::-1])
            c = kernels.from_values(other + other[::-1])
            wc = kernels.sub_const(w, kernels.nanmean(w))
            cc = kernels.sub_const(c, kernels.nanmean(c))
            pure = self.pure(lag.find_lag, wc, cc, -20, 20, 0, 'covmax')
            fast = self.fast(lag.find_lag, wc, cc, -20, 20, 0, 'covmax')
            self.assertEqual(pure[:3], fast[:3], "seed %d: " % seed)

    def test_count_finite_and_median_order_statistics_are_exact(self):
        self.assertEqual(self.pure(kernels.count_finite, self.x),
                         self.fast(kernels.count_finite, self.x))
        self.assertEqual(self.pure(kernels.nanmedian, self.x),
                         self.fast(kernels.nanmedian, self.x))
        med = kernels.nanmedian(self.x)
        self.assertEqual(self.pure(kernels.mad, self.x, med),
                         self.fast(kernels.mad, self.x, med))

    # -- elementwise transforms: identical -------------------------------------------

    def test_apply_mask_nan_is_identical_and_in_place(self):
        mask = kernels.mad_spike_mask(self.x, 2.0)
        self.assertGreater(sum(mask), 0)
        a = kernels.copy(self.x)
        b = kernels.copy(self.x)
        self.pure(kernels.apply_mask_nan, a, mask)
        self.fast(kernels.apply_mask_nan, b, mask)
        self.assertIdentical(a, b)
        for i, flagged in enumerate(mask):
            if flagged:
                self.assertNotEqual(b[i], b[i])

    def test_shift_truncate_is_identical(self):
        for shift in (-17, -1, 0, 1, 17, 5000):
            self.assertIdentical(self.pure(kernels.shift_truncate, self.x, shift),
                                 self.fast(kernels.shift_truncate, self.x, shift),
                                 "lag %d: " % shift)

    def test_sub_const_is_identical(self):
        self.assertIdentical(self.pure(kernels.sub_const, self.x, 0.1234567),
                             self.fast(kernels.sub_const, self.x, 0.1234567))
        a = kernels.copy(self.x)
        b = kernels.copy(self.x)
        self.pure(kernels.sub_const_inplace, a, 0.1234567)
        self.fast(kernels.sub_const_inplace, b, 0.1234567)
        self.assertIdentical(a, b)

    def test_sub_line_inplace_is_identical(self):
        a = kernels.copy(self.x)
        b = kernels.copy(self.x)
        self.pure(kernels.sub_line_inplace, a, 1.7e-5, -0.3)
        self.fast(kernels.sub_line_inplace, b, 1.7e-5, -0.3)
        self.assertIdentical(a, b)

    def test_rotation_is_identical_when_the_means_are_exact(self):
        # the angles come from reductions, so they are only bit-identical when the two
        # summation orders are; small dyadic samples make both sums exact.
        u = exact_series(301, offset=3.0)
        v = exact_series(302, offset=0.5)
        w = exact_series(303)
        pure = self.pure(kernels.rotate_double, u, v, w)
        fast = self.fast(kernels.rotate_double, u, v, w)
        self.assertEqual(pure[3], fast[3])          # theta
        self.assertEqual(pure[4], fast[4])          # phi
        for k in range(3):
            self.assertIdentical(pure[k], fast[k], "component %d: " % k)

    def test_rotation_agrees_within_tolerance_on_a_gappy_period(self):
        # A full 30-minute period at 20 Hz, because this is where the classification
        # matters: rotate_double is NOT bit-identical between the paths and cannot be.
        # Its angles come from nanmean, a reduction, so a 1-ULP difference in theta moves
        # every output sample. The rotated components agree to 7.8e-15 m s-1 here -- 8
        # ULP of the 3.04 m s-1 wind they were built from -- but v2 = -u sin(theta) +
        # v cos(theta) is a cancelling combination (-0.51 + 0.49 -> 0.003), so on its own
        # sample's scale that is 1.4e-09 relative on the worst of 36000 samples. The wind
        # speed is the scale the arithmetic actually works on.
        u = series(311, n=36000, mean=3.0)
        v = series(312, n=36000, mean=0.5)
        w = series(313, n=36000, sigma=0.3)
        pure = self.pure(kernels.rotate_double, u, v, w)
        fast = self.fast(kernels.rotate_double, u, v, w)
        speed = math.hypot(kernels.nanmean(u), kernels.nanmean(v))
        self.assertClose(pure[3], fast[3], msg='theta: ')
        self.assertClose(pure[4], fast[4], msg='phi: ')
        for k in range(3):
            for i in range(0, len(u), 41):
                self.assertClose(pure[k][i], fast[k][i], scale=speed,
                                 msg="component %d[%d]: " % (k, i))

    # -- reductions: 1e-12 relative --------------------------------------------------

    def test_nansum_and_nanmean_agree(self):
        sp, kp = self.pure(kernels.nansum, self.x)
        sf, kf = self.fast(kernels.nansum, self.x)
        self.assertEqual(kp, kf)
        self.assertClose(sp, sf)
        self.assertClose(self.pure(kernels.nanmean, self.x),
                         self.fast(kernels.nanmean, self.x))
        self.assertClose(self.pure(kernels.nanmean, self.x, 100, 700),
                         self.fast(kernels.nanmean, self.x, 100, 700))

    def test_second_moments_agree(self):
        for a, b, label in ((self.x, self.y, 'xy'), (self.big, self.x, 'co2'),
                            (self.big, self.big, 'co2 auto')):
            self.assertClose(self.pure(kernels.cov, a, b),
                             self.fast(kernels.cov, a, b), msg=label + ': ')
        self.assertClose(self.pure(kernels.variance, self.big),
                         self.fast(kernels.variance, self.big))
        self.assertClose(self.pure(kernels.nanstd, self.x),
                         self.fast(kernels.nanstd, self.x))
        self.assertClose(self.pure(kernels.cov, self.x, self.y, 50, 900),
                         self.fast(kernels.cov, self.x, self.y, 50, 900))

    def test_second_moments_agree_at_a_production_period_length(self):
        # 36000 samples (30 min at 20 Hz), a vertical wind and a sonic temperature with
        # no correlation planted between them -- a dawn or dusk period, where H passes
        # through zero. cov = 1.32e-06 K m s-1 here while the terms summed to reach it
        # average 0.14, so the result has cancelled by five orders of magnitude: the two
        # summation orders land 7.0e-12 apart *relative to the covariance* and 6.7e-17
        # apart relative to sd(w)*sd(ts). The second number is the one the kernel can
        # promise, and a period like this is ordinary rather than contrived.
        w = series(252, n=36000, mean=0.05, sigma=0.35)
        ts = series(9252, n=36000, mean=293.15, sigma=0.4)
        scale = kernels.nanstd(w) * kernels.nanstd(ts)
        self.assertClose(self.pure(kernels.cov, w, ts), self.fast(kernels.cov, w, ts),
                         scale=scale, msg='near-zero flux: ')
        self.assertClose(self.pure(kernels.variance, w), self.fast(kernels.variance, w),
                         scale=kernels.variance(w))

    def test_linfit_agrees(self):
        trended = kernels.copy(self.x)
        kernels.sub_line_inplace(trended, -3e-4, -12.0)
        sp, op = self.pure(kernels.linfit, trended)
        sf, of = self.fast(kernels.linfit, trended)
        self.assertClose(sp, sf, msg='slope: ')
        self.assertClose(op, of, msg='offset: ')

    def test_the_lag_covariances_agree(self):
        w = kernels.sub_const(self.x, kernels.nanmean(self.x))
        c = kernels.sub_const(self.big, kernels.nanmean(self.big))
        for lag in (-30, -1, 0, 7, 30):
            self.assertClose(self.pure(kernels.cov_at_lag, w, c, lag),
                             self.fast(kernels.cov_at_lag, w, c, lag),
                             msg="lag %d: " % lag)
        curve_pure = self.pure(kernels.cov_curve, w, c, -30, 30)
        curve_fast = self.fast(kernels.cov_curve, w, c, -30, 30)
        self.assertEqual(len(curve_pure), len(curve_fast))
        for j, (p, f) in enumerate(zip(curve_pure, curve_fast)):
            self.assertClose(p, f, msg="curve[%d]: " % j)

    def test_degenerate_inputs_agree(self):
        empty = kernels.new(0)
        allnan = kernels.new(5)
        for fn, args in ((kernels.nanmean, (empty,)), (kernels.nanmean, (allnan,)),
                         (kernels.nanmedian, (empty,)), (kernels.nanmedian, (allnan,)),
                         (kernels.variance, (allnan,)), (kernels.nanstd, (allnan,)),
                         (kernels.cov_at_lag, (allnan, allnan, 0))):
            p = self.pure(fn, *args)
            f = self.fast(fn, *args)
            self.assertTrue(p != p and f != f, "%s: %r vs %r" % (fn.__name__, p, f))
        self.assertEqual(self.pure(kernels.count_finite, empty),
                         self.fast(kernels.count_finite, empty))
        self.assertIdentical(self.pure(kernels.sub_const, empty, 1.0),
                             self.fast(kernels.sub_const, empty, 1.0))
        self.assertEqual(self.pure(kernels.linfit, allnan)[0] != self.pure(kernels.linfit, allnan)[0],
                         self.fast(kernels.linfit, allnan)[0] != self.fast(kernels.linfit, allnan)[0])


if __name__ == '__main__':
    unittest.main()
