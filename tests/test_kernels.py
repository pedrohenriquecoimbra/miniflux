"""Semantics of every kernel, including the NaN and short-series edges.

The numeric vectors are ALGORITHMS.md section 14 where they apply to this layer.
Every test runs twice: once with the numpy fast path off (the reference path) and
once with it on, so the two paths are pinned to the same *stated* semantics and not
only to each other (that is test_kernels_parity.py's job).
"""

import math
import random
import struct
import unittest
from array import array

from miniflux import kernels

NAN = float('nan')
INF = float('inf')


def next_up(v):
    """The next representable double above a positive v. math.nextafter is 3.9+."""
    bits = struct.unpack('<Q', struct.pack('<d', v))[0]
    return struct.unpack('<d', struct.pack('<Q', bits + 1))[0]


def gaussian_series(n, seed, sigma=1.0):
    """n deterministic N(0, sigma) samples as an array('d')."""
    rng = random.Random(seed)
    return kernels.from_values([rng.gauss(0.0, sigma) for _ in range(n)])


class KernelSemantics(object):
    """The tests. Mixed into one TestCase per code path (see the bottom of the file)."""

    USE_NUMPY = False

    def setUp(self):
        self._restore = kernels.HAVE_NUMPY
        kernels.set_numpy(self.USE_NUMPY)
        self.assertEqual(kernels.HAVE_NUMPY, self.USE_NUMPY)

    def tearDown(self):
        kernels.set_numpy(self._restore)

    # -- helpers --------------------------------------------------------------------

    def assertNan(self, v):
        self.assertNotEqual(v, v, "expected NaN, got %r" % (v,))

    def assertRel(self, got, want, tol=1e-12):
        scale = max(abs(got), abs(want), 1.0)
        self.assertLessEqual(abs(got - want), tol * scale,
                             "%r != %r within %g relative" % (got, want, tol))

    def assertSeries(self, got, want):
        """Element-by-element equality, NaN matching NaN."""
        self.assertEqual(len(got), len(want))
        for i, (a, b) in enumerate(zip(got, want)):
            if b != b:
                self.assertNotEqual(a, a, "index %d: expected NaN, got %r" % (i, a))
            else:
                self.assertEqual(a, b, "index %d" % i)

    # -- 3.1 construction -----------------------------------------------------------

    def test_new_is_all_nan_by_default(self):
        x = kernels.new(4)
        self.assertIsInstance(x, array)
        self.assertEqual(len(x), 4)
        self.assertSeries(x, [NAN] * 4)
        self.assertSeries(kernels.new(3, 7.5), [7.5, 7.5, 7.5])
        self.assertEqual(len(kernels.new(0)), 0)

    def test_from_values_maps_none_to_nan(self):
        self.assertSeries(kernels.from_values([1.0, None, 3]), [1.0, NAN, 3.0])
        self.assertEqual(len(kernels.from_values([])), 0)

    def test_copy_is_independent(self):
        x = kernels.from_values([1.0, 2.0])
        y = kernels.copy(x)
        y[0] = 99.0
        self.assertEqual(x[0], 1.0)

    # -- 3.2 reductions -------------------------------------------------------------

    def test_count_finite_excludes_nan_and_infinities(self):
        x = kernels.from_values([1.0, NAN, INF, -INF, 2.0])
        self.assertEqual(kernels.count_finite(x), 2)
        self.assertEqual(kernels.count_finite(x, 1), 1)
        self.assertEqual(kernels.count_finite(x, 0, 2), 1)
        self.assertEqual(kernels.count_finite(kernels.new(3)), 0)

    def test_nansum_returns_sum_and_count(self):
        x = kernels.from_values([1.0, NAN, 2.0, 4.0])
        self.assertEqual(kernels.nansum(x), (7.0, 3))
        self.assertEqual(kernels.nansum(x, 2), (6.0, 2))
        self.assertEqual(kernels.nansum(x, 0, 2), (1.0, 1))
        self.assertEqual(kernels.nansum(kernels.new(3)), (0.0, 0))

    def test_nanmean_is_nan_without_a_finite_sample(self):
        x = kernels.from_values([1.0, NAN, 2.0, 4.0])
        self.assertRel(kernels.nanmean(x), 7.0 / 3.0)
        self.assertRel(kernels.nanmean(x, 2, 4), 3.0)
        self.assertNan(kernels.nanmean(kernels.new(3)))
        self.assertNan(kernels.nanmean(kernels.new(0)))

    def test_nanmedian_parity_is_that_of_the_finite_count(self):
        # four finite samples -> even rule, the two middles averaged
        self.assertEqual(kernels.nanmedian(kernels.from_values([1., 2., 3., 4.])), 2.5)
        # the same four with a NaN appended: still four survivors, still the even rule
        self.assertEqual(kernels.nanmedian(kernels.from_values([1., 2., 3., 4., NAN])), 2.5)
        # dropping one of the four leaves three: the odd rule, not (2+3)/2
        self.assertEqual(kernels.nanmedian(kernels.from_values([1., 2., NAN, 4.])), 2.0)
        # order does not matter
        self.assertEqual(kernels.nanmedian(kernels.from_values([4., 1., 3., 2.])), 2.5)

    def test_nanmedian_without_finite_samples_is_nan(self):
        self.assertNan(kernels.nanmedian(kernels.new(5)))
        self.assertNan(kernels.nanmedian(kernels.new(0)))
        self.assertEqual(kernels.nanmedian(kernels.from_values([3.0])), 3.0)

    def test_mad_is_the_unscaled_median_absolute_deviation(self):
        x = kernels.from_values([1., 2., 3., 4., 100.])
        # |x - 3| = 2, 1, 0, 1, 97 -> sorted 0, 1, 1, 2, 97 -> 1.0, no 0.6745 anywhere
        self.assertEqual(kernels.mad(x, 3.0), 1.0)
        self.assertEqual(kernels.mad(kernels.from_values([5., 5., 5.]), 5.0), 0.0)
        self.assertNan(kernels.mad(kernels.new(4), NAN))

    def test_mad_of_a_nan_median_is_nan(self):
        self.assertNan(kernels.mad(kernels.from_values([1.0, 2.0]), NAN))

    # -- 3.3 second moments ---------------------------------------------------------

    def test_cov_known_value(self):
        x = kernels.from_values([1., 2., 3., 4.])
        y = kernels.from_values([2., 4., 6., 8.])
        self.assertRel(kernels.cov(x, y), 10.0 / 3.0)
        self.assertRel(kernels.variance(x), 5.0 / 3.0)
        self.assertRel(kernels.nanstd(x), math.sqrt(5.0 / 3.0))

    def test_cov_of_x_with_itself_is_the_variance(self):
        # ALGORITHMS section 14.4
        x = gaussian_series(500, seed=11)
        self.assertEqual(kernels.cov(x, x), kernels.variance(x))

    def test_cov_is_translation_invariant(self):
        # ALGORITHMS section 14.4: the whole point of the two-pass form. A one-pass
        # raw-sums cov would lose ~10 digits here.
        x = gaussian_series(500, seed=12)
        y = gaussian_series(500, seed=13)
        plain = kernels.cov(x, y)
        shifted = kernels.cov(kernels.sub_const(x, -1e6), y)
        self.assertRel(shifted, plain, 1e-9)

    def test_cov_uses_the_joint_finite_mask(self):
        x = kernels.from_values([1., 2., 3., NAN])
        y = kernels.from_values([2., NAN, 6., 8.])
        # survivors are indices 0 and 2: means 2 and 4, products (-1)(-2) + (1)(2) = 4
        self.assertRel(kernels.cov(x, y), 4.0)

    def test_cov_needs_two_joint_finite_pairs(self):
        x = kernels.from_values([1., NAN])
        y = kernels.from_values([NAN, 2.])
        self.assertNan(kernels.cov(x, y))
        self.assertNan(kernels.cov(kernels.from_values([1.0]), kernels.from_values([2.0])))
        self.assertNan(kernels.variance(kernels.new(10)))

    def test_a_stuck_sensor_gives_zero_variance_not_nan(self):
        # ALGORITHMS section 7.1: the QC layer guards this, the kernel does not.
        stuck = kernels.from_values([4.0] * 8)
        self.assertEqual(kernels.variance(stuck), 0.0)
        self.assertEqual(kernels.nanstd(stuck), 0.0)
        self.assertEqual(kernels.cov(stuck, gaussian_series(8, seed=1)), 0.0)

    def test_cov_honours_slice_bounds(self):
        x = kernels.from_values([1., 2., 3., 4., 99.])
        y = kernels.from_values([2., 4., 6., 8., -99.])
        self.assertRel(kernels.cov(x, y, 0, 4), 10.0 / 3.0)
        self.assertNan(kernels.cov(x, y, 3, 4))

    def test_nanstd_of_a_nan_variance_is_nan(self):
        self.assertNan(kernels.nanstd(kernels.new(3)))

    # -- 3.4 despiking primitives ---------------------------------------------------

    def test_one_spike_in_a_thousand_gaussian_samples_is_flagged(self):
        # ALGORITHMS section 14.2. q = 7 puts the bounds at 7/0.6745 = 10.378 MADs,
        # about 7 sigma for a normal series, so only the planted +50 is outside.
        x = gaussian_series(1000, seed=2013)
        x[400] = 50.0
        mask = kernels.mad_spike_mask(x, 7.0)
        self.assertEqual(sum(mask), 1)
        self.assertEqual(mask[400], 1)

    def test_a_spike_at_the_last_index_is_still_flagged_here(self):
        # the trailing-run rule belongs to despike.py; the kernel reports the bound test
        x = gaussian_series(1000, seed=2013)
        x[999] = 50.0
        mask = kernels.mad_spike_mask(x, 7.0)
        self.assertEqual(sum(mask), 1)
        self.assertEqual(mask[999], 1)

    def test_a_constant_series_flags_nothing(self):
        # ALGORITHMS section 14.2: MAD = 0 collapses the bounds onto m, and m is not
        # strictly outside itself.
        mask = kernels.mad_spike_mask(kernels.from_values([3.0] * 50), 7.0)
        self.assertEqual(sum(mask), 0)

    def test_zero_mad_flags_every_sample_that_differs_at_all(self):
        # deliberately unguarded: a stuck sensor with one different reading
        x = kernels.from_values([5., 5., 5., 5., 7.])
        self.assertEqual(list(kernels.mad_spike_mask(x, 7.0)), [0, 0, 0, 0, 1])

    def test_nan_is_never_flagged_and_an_all_nan_series_flags_nothing(self):
        x = kernels.from_values([1.0, NAN, 500.0, 1.0, 1.0, 1.0, 1.0])
        mask = kernels.mad_spike_mask(x, 7.0)
        self.assertEqual(mask[1], 0)
        self.assertEqual(mask[2], 1)
        self.assertEqual(sum(kernels.mad_spike_mask(kernels.new(6), 7.0)), 0)
        self.assertEqual(len(kernels.mad_spike_mask(kernels.new(0), 7.0)), 0)

    def test_the_bound_is_m_plus_q_times_d_divided_by_the_scale(self):
        # (q*d)/0.6745, q*(d/0.6745) and q*d*1.4826 are three different doubles; this
        # pins the one the contract names, by putting a sample exactly on it (strict
        # comparison, so it survives) and one ULP above it (flagged).
        q = 7.0
        d = None
        for trial in range(1, 5000):
            candidate = trial * 1e-3
            if (q * candidate) / 0.6745 != q * (candidate / 0.6745):
                d = candidate
                break
        self.assertIsNotNone(d, "no d found where the two parenthesisations differ")
        upper = (q * d) / 0.6745
        # median 0 and MAD exactly d, whatever the probe is, as long as probe > d
        body = [-d, -d, 0.0, 0.0, 0.0, d, d]
        on_the_bound = kernels.mad_spike_mask(kernels.from_values(body + [upper]), q)
        self.assertEqual(sum(on_the_bound), 0)
        one_ulp_over = kernels.mad_spike_mask(kernels.from_values(body + [next_up(upper)]), q)
        self.assertEqual(list(one_ulp_over), [0] * 7 + [1])

    def test_apply_mask_nan_edits_in_place(self):
        x = kernels.from_values([1., 2., 3.])
        kernels.apply_mask_nan(x, bytearray([0, 1, 0]))
        self.assertSeries(x, [1.0, NAN, 3.0])

    # -- 3.5 series transforms ------------------------------------------------------

    def test_shift_truncate_never_wraps(self):
        x = kernels.from_values([1., 2., 3., 4.])
        self.assertSeries(kernels.shift_truncate(x, 1), [2., 3., 4., NAN])
        self.assertSeries(kernels.shift_truncate(x, -1), [NAN, 1., 2., 3.])
        self.assertSeries(kernels.shift_truncate(x, 0), [1., 2., 3., 4.])
        self.assertSeries(kernels.shift_truncate(x, 4), [NAN] * 4)
        self.assertSeries(kernels.shift_truncate(x, -9), [NAN] * 4)

    def test_sub_const(self):
        x = kernels.from_values([1., NAN, 3.])
        self.assertSeries(kernels.sub_const(x, 1.0), [0.0, NAN, 2.0])
        self.assertSeries(x, [1., NAN, 3.])          # the input is untouched
        kernels.sub_const_inplace(x, 1.0)
        self.assertSeries(x, [0.0, NAN, 2.0])

    def test_sub_line_inplace_uses_the_original_index(self):
        x = kernels.from_values([5., 8., NAN, 14.])
        kernels.sub_line_inplace(x, 3.0, 5.0)
        self.assertSeries(x, [0.0, 0.0, NAN, 0.0])

    def test_linfit_recovers_a_clean_line(self):
        x = kernels.from_values([5.0 + 3.0 * i for i in range(10)])
        slope, offset = kernels.linfit(x)
        self.assertRel(slope, 3.0)
        self.assertRel(offset, 5.0)

    def test_linfit_fits_a_gapped_series_on_the_original_index(self):
        # CONTRACT section 20.2: the gap must not stretch the trend by N/m
        values = [5.0 + 3.0 * i for i in range(20)]
        values[2] = None
        values[7] = None
        values[18] = None
        slope, offset = kernels.linfit(kernels.from_values(values))
        self.assertRel(slope, 3.0)
        self.assertRel(offset, 5.0)

    def test_linfit_falls_back_to_mean_removal_when_too_short(self):
        self.assertEqual(kernels.linfit(kernels.from_values([4.0, 6.0])), (0.0, 5.0))
        self.assertEqual(kernels.linfit(kernels.from_values([4.0, NAN])), (0.0, 4.0))
        slope, offset = kernels.linfit(kernels.new(5))
        self.assertNan(slope)
        self.assertNan(offset)

    # -- rotation, ALGORITHMS sections 4 and 14.1 -----------------------------------

    def test_rotate_double_invariants(self):
        u = gaussian_series(400, seed=21)
        v = gaussian_series(400, seed=22)
        w = gaussian_series(400, seed=23, sigma=0.3)
        kernels.sub_const_inplace(u, -3.0)          # a 3 m s-1 mean wind
        kernels.sub_const_inplace(v, -0.7)
        kernels.sub_const_inplace(w, 0.05)
        mu, mv, mw = kernels.nanmean(u), kernels.nanmean(v), kernels.nanmean(w)
        u2, v2, w2, theta, phi = kernels.rotate_double(u, v, w)
        self.assertLess(abs(kernels.nanmean(v2)), 1e-9)
        self.assertLess(abs(kernels.nanmean(w2)), 1e-9)
        self.assertRel(kernels.nanmean(u2), math.sqrt(mu * mu + mv * mv + mw * mw), 1e-9)
        self.assertRel(theta, math.atan2(mv, mu), 1e-9)
        for i in range(0, 400, 37):                 # speed is preserved sample by sample
            before = math.sqrt(u[i] ** 2 + v[i] ** 2 + w[i] ** 2)
            after = math.sqrt(u2[i] ** 2 + v2[i] ** 2 + w2[i] ** 2)
            self.assertRel(after, before, 1e-9)

    def test_rotate_double_keeps_u_positive_when_the_mean_wind_is_negative(self):
        # atan2, not atan: with atan this period would rotate to u = -|wind|
        u = kernels.from_values([-3.0, -3.2, -2.8, -3.1])
        v = kernels.from_values([0.4, 0.2, 0.5, 0.3])
        w = kernels.from_values([0.05, -0.02, 0.01, 0.03])
        u2, v2, w2, theta, phi = kernels.rotate_double(u, v, w)
        self.assertGreater(kernels.nanmean(u2), 0.0)
        self.assertLess(abs(kernels.nanmean(w2)), 1e-12)

    def test_rotate_double_on_an_all_nan_period(self):
        n = 4
        u2, v2, w2, theta, phi = kernels.rotate_double(kernels.new(n), kernels.new(n),
                                                       kernels.new(n))
        self.assertNan(theta)
        self.assertNan(phi)
        self.assertSeries(u2, [NAN] * n)
        self.assertSeries(v2, [NAN] * n)
        self.assertSeries(w2, [NAN] * n)

    def test_rotate_double_refuses_three_series_of_different_lengths(self):
        # The numpy path would broadcast a length-1 component across the whole period and
        # hand back plausible fluxes computed against a fabricated series, where the pure
        # path raises IndexError. Both refuse, identically -- the only answer that can be
        # the same on both paths.
        u = kernels.from_values([1.0, 2.0, 3.0, 4.0])
        v = kernels.from_values([1.0])
        w = kernels.from_values([0.1, 0.1, 0.1, 0.1])
        with self.assertRaises(ValueError) as caught:
            kernels.rotate_double(u, v, w)
        self.assertIn('4', str(caught.exception))
        self.assertIn('1', str(caught.exception))

    def test_rotate_double_with_no_mean_wind_gives_zero_angles(self):
        u = kernels.from_values([1.0, -1.0])
        v = kernels.from_values([2.0, -2.0])
        w = kernels.from_values([3.0, -3.0])
        u2, v2, w2, theta, phi = kernels.rotate_double(u, v, w)
        self.assertEqual(theta, 0.0)                 # atan2(0, 0) is 0.0, not an error
        self.assertEqual(phi, 0.0)
        self.assertSeries(u2, [1.0, -1.0])

    # -- 3.6 lag search -------------------------------------------------------------

    def test_cov_at_lag_finds_a_scalar_that_arrives_late(self):
        # ALGORITHMS section 14.3: c[i] = w[i-15], so c[i+15] = w[i] and the peak is
        # at L = +15 (1.5 s at 10 Hz). One convention, nothing negated.
        w = gaussian_series(1000, seed=31)
        c = kernels.shift_truncate(w, -15)
        curve = kernels.cov_curve(w, c, -30, 30)
        best = max(range(len(curve)), key=lambda j: abs(curve[j]))
        self.assertEqual(best - 30, 15)
        # at that lag the overlap is w[0:985] against itself, so the peak is its variance
        self.assertRel(kernels.cov_at_lag(w, c, 15), kernels.variance(w, 0, 985), 1e-9)

    def test_cov_curve_is_inclusive_at_both_ends(self):
        w = gaussian_series(50, seed=32)
        c = gaussian_series(50, seed=33)
        curve = kernels.cov_curve(w, c, -3, 4)
        self.assertEqual(len(curve), 8)              # 4 - (-3) + 1
        for j, lag in enumerate(range(-3, 5)):
            self.assertEqual(curve[j], kernels.cov_at_lag(w, c, lag))
        self.assertEqual(len(kernels.cov_curve(w, c, 2, 2)), 1)

    def test_cov_curve_refuses_a_reversed_window(self):
        w = gaussian_series(10, seed=34)
        self.assertRaises(ValueError, kernels.cov_curve, w, w, 3, 2)

    def test_cov_at_lag_uses_the_overlap_only(self):
        # no wraparound: at lag 2 only two pairs remain, and at lag 3 only one
        x = kernels.from_values([1., 2., 3., 4.])
        y = kernels.from_values([10., 20., 30., 40.])
        pairs = [(1., 30.), (2., 40.)]
        sa = sum(p[0] for p in pairs)
        sb = sum(p[1] for p in pairs)
        sab = sum(p[0] * p[1] for p in pairs)
        self.assertRel(kernels.cov_at_lag(x, y, 2), (sab - sa * sb / 2) / 1)
        self.assertNan(kernels.cov_at_lag(x, y, 3))
        self.assertNan(kernels.cov_at_lag(x, y, -3))
        self.assertNan(kernels.cov_at_lag(x, y, 99))

    def test_cov_at_lag_needs_two_joint_finite_pairs(self):
        x = kernels.from_values([1., NAN, 3., NAN])
        y = kernels.from_values([NAN, 2., NAN, 4.])
        self.assertNan(kernels.cov_at_lag(x, y, 0))

    def test_the_two_estimators_agree_on_a_centred_series(self):
        # the scan form is one-pass and the reported form two-pass; on centred series
        # they differ only in the last bits, which is why lag.py centres.
        x = gaussian_series(300, seed=41)
        y = gaussian_series(300, seed=42)
        kernels.sub_const_inplace(x, kernels.nanmean(x))
        kernels.sub_const_inplace(y, kernels.nanmean(y))
        self.assertRel(kernels.cov_at_lag(x, y, 0), kernels.cov(x, y), 1e-9)


class PurePathTest(KernelSemantics, unittest.TestCase):
    USE_NUMPY = False


@unittest.skipUnless(kernels.NUMPY_AVAILABLE, "numpy is not installed")
class NumpyPathTest(KernelSemantics, unittest.TestCase):
    USE_NUMPY = True


class NumpySwitchTest(unittest.TestCase):

    def tearDown(self):
        kernels.set_numpy(kernels.NUMPY_AVAILABLE)

    def test_set_numpy_off_returns_and_sets_false(self):
        self.assertFalse(kernels.set_numpy(False))
        self.assertFalse(kernels.HAVE_NUMPY)

    @unittest.skipUnless(kernels.NUMPY_AVAILABLE, "numpy is not installed")
    def test_set_numpy_on_enables_the_fast_path(self):
        self.assertTrue(kernels.set_numpy(True))
        self.assertTrue(kernels.HAVE_NUMPY)

    def test_asking_for_an_absent_numpy_warns_rather_than_raising(self):
        # refusing the run is config.py's decision; this layer logs and carries on. The
        # machine without numpy is simulated by hiding the import probe's result, which is
        # the single global set_numpy consults -- so the branch is exercised wherever the
        # suite runs, rather than only where it cannot be.
        available = kernels.NUMPY_AVAILABLE
        kernels.NUMPY_AVAILABLE = False
        try:
            with self.assertLogs('miniflux.kernels', level='WARNING'):
                self.assertFalse(kernels.set_numpy(True))
            self.assertFalse(kernels.HAVE_NUMPY)
        finally:
            kernels.NUMPY_AVAILABLE = available      # tearDown restores HAVE_NUMPY from it


if __name__ == '__main__':
    unittest.main()
