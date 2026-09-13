"""What the double rotation must do (ALGORITHMS.md section 4, CONTRACT.md section 9).

The invariants of section 4.2 are asserted to machine precision, the two angles against
a hand-computed pair, and the traps of section 4.3 one by one: atan2 rather than atan on
a period whose mean u is negative, the (y, x) argument order, and phi taken from the
rotated u1 rather than from u.

Everything runs twice, once on each kernel path, because the rotation arithmetic has a
numpy fast path and the invariants must hold on both.
"""

import logging
import math
import random
import unittest
from types import SimpleNamespace

from miniflux import kernels, rotate

NAN = float('nan')

# The series keys a period always carries; rotate touches only the first three.
SERIES = ('u', 'v', 'w', 'ts', 'co2', 'h2o', 'ta', 'p_air')


def make_period(u, v, w):
    """A period dict holding the three wind series; every other series is all-NaN."""
    period = {'meta': {}}
    for name in SERIES:
        period[name] = kernels.new(len(u))
    period['u'] = kernels.from_values(u)
    period['v'] = kernels.from_values(v)
    period['w'] = kernels.from_values(w)
    return period


def make_cfg(method='double', north_offset=0.0):
    """The only two config keys rotate may read: [rotate] method and [site] north_offset."""
    return SimpleNamespace(rotate=SimpleNamespace(method=method),
                           site=SimpleNamespace(north_offset=north_offset))


def wind(n=2000, seed=7, mean_u=2.7, mean_v=-1.3, mean_w=0.11, gaps=True):
    """Deterministic turbulent wind about a tilted mean, as three lists of float|None.

    The gaps sit at the same index in all three components. That is the case in which
    the rotation invariants are exact: the angles come from three independent nan-means,
    so a component despiked to its own mask satisfies them only approximately.
    """
    rng = random.Random(seed)
    u, v, w = [], [], []
    for i in range(n):
        if gaps and i % 97 == 5:
            u.append(None)
            v.append(None)
            w.append(None)
            continue
        u.append(mean_u + rng.gauss(0.0, 0.9))
        v.append(mean_v + rng.gauss(0.0, 0.7))
        w.append(mean_w + rng.gauss(0.0, 0.4))
    return u, v, w


class Collect(logging.Handler):
    """A handler that keeps the records rather than printing them."""

    def __init__(self, level):
        logging.Handler.__init__(self, level)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def records_at(level, run):
    """Call ``run()`` with the rotate logger at ``level`` and return what it emitted.

    ``assertLogs`` forces the logger to DEBUG and fails when nothing is logged, so it
    cannot answer the question "does this stay quiet at INFO?".
    """
    logger = logging.getLogger('miniflux.rotate')
    handler = Collect(level)
    before = (logger.level, logger.propagate)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    try:
        run()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(before[0])
        logger.propagate = before[1]
    return handler.records


class RotateBehaviour(object):
    """The tests. Mixed into one TestCase per kernel path (see the bottom of the file)."""

    USE_NUMPY = False

    def setUp(self):
        self._restore = kernels.HAVE_NUMPY
        kernels.set_numpy(self.USE_NUMPY)

    def tearDown(self):
        kernels.set_numpy(self._restore)

    # -- helpers ----------------------------------------------------------------------

    def assertNan(self, value):
        self.assertNotEqual(value, value, "expected NaN, got %r" % (value,))

    def assertRel(self, got, want, tol=1e-12):
        scale = max(abs(got), abs(want), 1.0)
        self.assertLessEqual(abs(got - want), tol * scale,
                             "%r != %r within %g relative" % (got, want, tol))

    # -- section 4.2, the invariants --------------------------------------------------

    def test_invariants_hold_to_machine_precision(self):
        u, v, w = wind()
        period = make_period(u, v, w)
        mu = kernels.nanmean(period['u'])
        mv = kernels.nanmean(period['v'])
        mw = kernels.nanmean(period['w'])

        rotate.rotate(period, make_cfg())

        # mean(v2) = mean(w2) = 0 exactly by construction; 1e-12 m s-1 on a 3 m s-1 wind
        # is rounding, three orders below the 1e-9 the contract asks the self-check for.
        self.assertLess(abs(kernels.nanmean(period['v'])), 1e-12)
        self.assertLess(abs(kernels.nanmean(period['w'])), 1e-12)
        # the whole horizontal-plus-vertical mean speed ends up in u2, and positive
        self.assertRel(kernels.nanmean(period['u']),
                       math.sqrt(mu * mu + mv * mv + mw * mw))
        self.assertGreater(kernels.nanmean(period['u']), 0.0)

    def test_rotation_preserves_the_instantaneous_speed(self):
        u, v, w = wind()
        period = make_period(u, v, w)
        before = [math.sqrt(a * a + b * b + c * c)
                  for a, b, c in zip(period['u'], period['v'], period['w'])]

        rotate.rotate(period, make_cfg())

        # R = Ry(phi) . Rz(theta) is orthogonal with det = +1, so the speed is invariant
        after = [math.sqrt(a * a + b * b + c * c)
                 for a, b, c in zip(period['u'], period['v'], period['w'])]
        for i, (a, b) in enumerate(zip(before, after)):
            if a != a:
                self.assertNan(b)
                continue
            self.assertLessEqual(abs(a - b), 1e-9 * max(a, 1.0), 'sample %d' % i)

    # -- section 4.1, the two angles --------------------------------------------------

    def test_hand_computed_angle_pair(self):
        # mean u = mean v = 1 -> theta = atan2(1, 1) = pi/4; the first rotation puts
        # sqrt(2) into u1, and mean w1 = sqrt(2) as well -> phi = atan2(1, 1) = pi/4.
        # Then u2 = sqrt(2)cos(pi/4) + sqrt(2)sin(pi/4) = 2 = sqrt(1 + 1 + 2).
        root2 = math.sqrt(2.0)
        period = make_period([1.0, 1.0], [1.0, 1.0], [root2, root2])

        rotate.rotate(period, make_cfg())

        self.assertRel(period['meta']['theta'], math.pi / 4.0)
        self.assertRel(period['meta']['phi'], math.pi / 4.0)
        for i in range(2):
            self.assertRel(period['u'][i], 2.0)
            self.assertLess(abs(period['v'][i]), 1e-15)
            self.assertLess(abs(period['w'][i]), 1e-15)

    def test_theta_is_atan2_of_mean_v_over_mean_u_in_that_order(self):
        # asymmetric means, so the swapped order atan2(mean_u, mean_v) is a different
        # angle (the complement) rather than the same one
        period = make_period([2.0, 2.0], [1.0, 1.0], [0.0, 0.0])

        rotate.rotate(period, make_cfg())

        self.assertRel(period['meta']['theta'], math.atan2(1.0, 2.0))
        self.assertNotAlmostEqual(period['meta']['theta'], math.atan2(2.0, 1.0))
        # radians, not degrees: 0.4636 rad is 26.57 deg
        self.assertLess(period['meta']['theta'], math.pi / 2.0)

    def test_phi_is_computed_from_the_rotated_u1_not_from_u(self):
        # mean v != 0, so mean u1 = hypot(2, 2) = 2.828 differs from mean u = 2 and the
        # two candidate pitch angles are visibly different
        period = make_period([2.0, 2.0], [2.0, 2.0], [1.0, 1.0])

        rotate.rotate(period, make_cfg())

        self.assertRel(period['meta']['phi'], math.atan2(1.0, math.hypot(2.0, 2.0)))
        self.assertNotAlmostEqual(period['meta']['phi'], math.atan2(1.0, 2.0))

    def test_zero_horizontal_mean_gives_a_zero_angle_not_an_exception(self):
        period = make_period([1.0, -1.0], [2.0, -2.0], [0.0, 0.0])

        rotate.rotate(period, make_cfg())

        self.assertEqual(period['meta']['theta'], 0.0)   # atan2(0, 0) = 0.0
        self.assertEqual(period['meta']['phi'], 0.0)

    # -- section 4.3, the negative-mean-u quadrant guard -------------------------------

    def test_negative_mean_u_keeps_the_streamwise_axis_positive(self):
        # A wind blowing along -x. With atan instead of atan2 the quadrant collapses,
        # theta comes out 0 and u2 stays at -|wind|, which flips the sign of <u'w'> on
        # exactly these periods and on no others.
        rng = random.Random(3)
        fluct_u = [rng.gauss(0.0, 0.5) for _ in range(400)]
        # w positively correlated with the sonic-frame u', hence NEGATIVELY correlated
        # with the streamwise u2' once the frame is turned round
        fluct_w = [0.6 * a + rng.gauss(0.0, 0.1) for a in fluct_u]
        mean_u = sum(fluct_u) / len(fluct_u)
        mean_w = sum(fluct_w) / len(fluct_w)
        u = [-3.0 + a - mean_u for a in fluct_u]
        w = [b - mean_w for b in fluct_w]
        period = make_period(u, [0.0] * len(u), w)
        cov_sonic = kernels.cov(period['u'], period['w'])
        self.assertGreater(cov_sonic, 0.0)

        rotate.rotate(period, make_cfg())

        self.assertRel(period['meta']['theta'], math.pi)
        self.assertRel(kernels.nanmean(period['u']), 3.0, 1e-9)
        self.assertGreater(kernels.nanmean(period['u']), 0.0)
        # the momentum flux in the streamline frame is the downward one, sign flipped
        self.assertRel(kernels.cov(period['u'], period['w']), -cov_sonic, 1e-9)

    # -- section 4.4, NaN behaviour ---------------------------------------------------

    def test_each_output_carries_the_union_of_the_masks_it_mixes(self):
        u = [1.0, 1.0, 1.0, NAN, 1.0, 1.0, 1.0, 1.0]
        v = [0.5, 0.5, 0.5, 0.5, 0.5, NAN, 0.5, 0.5]
        w = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, NAN]
        period = make_period(u, v, w)

        rotate.rotate(period, make_cfg())

        # u2 and w2 mix all three components; v2 mixes u and v only, so w's gap does not
        # reach it
        for i in (3, 5, 7):
            self.assertNan(period['u'][i])
            self.assertNan(period['w'][i])
        self.assertNan(period['v'][3])
        self.assertNan(period['v'][5])
        self.assertTrue(math.isfinite(period['v'][7]))

    def test_all_nan_period_gives_nan_angles_and_warns(self):
        period = make_period([NAN] * 5, [NAN] * 5, [NAN] * 5)

        with self.assertLogs('miniflux.rotate', level='WARNING'):
            rotate.rotate(period, make_cfg())

        self.assertNan(period['meta']['theta'])
        self.assertNan(period['meta']['phi'])
        self.assertNan(period['meta']['wind_speed'])
        self.assertNan(period['meta']['wind_dir'])
        for name in ('u', 'v', 'w'):
            for value in period[name]:
                self.assertNan(value)

    def test_empty_period_does_not_raise(self):
        period = make_period([], [], [])

        with self.assertLogs('miniflux.rotate', level='WARNING'):
            rotate.rotate(period, make_cfg())

        self.assertEqual(len(period['u']), 0)
        self.assertNan(period['meta']['theta'])

    # -- section 4.5, wind speed and direction ----------------------------------------

    def test_wind_direction_is_the_direction_the_wind_comes_from(self):
        # the checks written into ALGORITHMS section 4.5: a wind blowing forward comes
        # from behind
        for u, v, expected in ((1.0, 0.0, 180.0),    # blowing +u -> from behind
                               (0.0, 1.0, 90.0),     # blowing +v (left) -> from the right
                               (-1.0, 0.0, 0.0),     # blowing -u -> from the front
                               (0.0, -1.0, 270.0)):
            period = make_period([u, u], [v, v], [0.0, 0.0])
            rotate.rotate(period, make_cfg())
            self.assertRel(period['meta']['wind_dir'], expected, 1e-12)

    def test_wind_direction_applies_the_north_offset_and_wraps(self):
        for offset, expected in ((30.0, 210.0), (200.0, 20.0), (-200.0, 340.0)):
            period = make_period([1.0, 1.0], [0.0, 0.0], [0.0, 0.0])
            rotate.rotate(period, make_cfg(north_offset=offset))
            self.assertRel(period['meta']['wind_dir'], expected, 1e-12)

    def test_north_offset_never_reaches_the_rotation_angles(self):
        u, v, w = wind(n=200)
        plain = make_period(u, v, w)
        offset = make_period(u, v, w)

        rotate.rotate(plain, make_cfg())
        rotate.rotate(offset, make_cfg(north_offset=137.0))

        self.assertEqual(plain['meta']['theta'], offset['meta']['theta'])
        self.assertEqual(plain['meta']['phi'], offset['meta']['phi'])
        self.assertEqual(plain['meta']['wind_speed'], offset['meta']['wind_speed'])
        self.assertNotEqual(plain['meta']['wind_dir'], offset['meta']['wind_dir'])

    def test_wind_speed_and_means_are_the_unrotated_ones(self):
        u, v, w = wind(n=500)
        period = make_period(u, v, w)
        mu = kernels.nanmean(period['u'])
        mv = kernels.nanmean(period['v'])
        mw = kernels.nanmean(period['w'])

        rotate.rotate(period, make_cfg())

        meta = period['meta']
        self.assertEqual(meta['u_unrot_mean'], mu)
        self.assertEqual(meta['v_unrot_mean'], mv)
        self.assertEqual(meta['w_unrot_mean'], mw)
        # horizontal only: the vertical mean is recorded but stays out of WS
        self.assertEqual(meta['wind_speed'], math.hypot(mu, mv))
        self.assertNotAlmostEqual(meta['wind_speed'], kernels.nanmean(period['u']))

    # -- CONTRACT section 9, the plumbing ---------------------------------------------

    def test_method_none_leaves_the_series_and_zeroes_the_angles(self):
        u, v, w = wind(n=300)
        period = make_period(u, v, w)
        before = (list(period['u']), list(period['v']), list(period['w']))

        returned = rotate.rotate(period, make_cfg(method='none'))

        self.assertIs(returned, period)
        self.assertEqual(period['meta']['theta'], 0.0)
        self.assertEqual(period['meta']['phi'], 0.0)
        for name, original in zip(('u', 'v', 'w'), before):
            for i, (got, want) in enumerate(zip(period[name], original)):
                if want != want:
                    self.assertNan(got)
                else:
                    self.assertEqual(got, want, '%s[%d]' % (name, i))
        # WS and WD are written whatever the method, because they are sonic-frame
        self.assertRel(period['meta']['wind_speed'],
                       math.hypot(period['meta']['u_unrot_mean'],
                                  period['meta']['v_unrot_mean']))

    def test_writes_exactly_the_meta_keys_of_the_contract(self):
        u, v, w = wind(n=100)
        period = make_period(u, v, w)

        rotate.rotate(period, make_cfg())

        self.assertEqual(sorted(period['meta']), sorted([
            'u_unrot_mean', 'v_unrot_mean', 'w_unrot_mean',
            'wind_speed', 'wind_dir', 'theta', 'phi']))

    def test_the_other_series_are_untouched(self):
        u, v, w = wind(n=64)
        period = make_period(u, v, w)
        untouched = dict((name, period[name]) for name in SERIES[3:])

        rotate.rotate(period, make_cfg())

        for name, series in untouched.items():
            self.assertIs(period[name], series)

    def test_self_check_runs_at_debug_only(self):
        u, v, w = wind(n=300)

        with self.assertLogs('miniflux.rotate', level='DEBUG') as captured:
            rotate.rotate(make_period(u, v, w), make_cfg())
        self.assertTrue(any('yaw identity' in line for line in captured.output))
        self.assertFalse(any(record.levelno >= logging.ERROR
                             for record in captured.records))

        # The check bounds the identity on the means the angles came from, not the mean
        # of the rotated series. Despiking gives the three components different NaN
        # masks, which moves the sample mean off zero by ~1e-5 m s-1 on healthy data;
        # that must not be reported as an error.
        u2, v2, w2 = wind(n=300)
        v2[7] = float('nan')
        w2[11] = float('nan')
        w2[12] = float('nan')
        with self.assertLogs('miniflux.rotate', level='DEBUG') as ragged:
            rotate.rotate(make_period(u2, v2, w2), make_cfg())
        self.assertFalse(any(record.levelno >= logging.ERROR
                             for record in ragged.records))

        quiet = records_at(logging.INFO,
                           lambda: rotate.rotate(make_period(u, v, w), make_cfg()))
        self.assertEqual(quiet, [])

    def test_the_self_check_catches_a_rotation_that_did_not_zero_the_means(self):
        """CONTRACT 9: the check is on `nanmean(v)` and `nanmean(w)` AFTER rotation.

        Driven with the trap ALGORITHMS 4.3 names by name -- the pitch angle taken from
        `mean(u)` instead of from `mean(u1)`. The angles the kernel reports are
        self-consistent with the series it returns, so no identity built from the angles
        alone can see this; only the rotated series can.
        """
        real = kernels.rotate_double

        def trapped(u, v, w):
            theta = math.atan2(kernels.nanmean(v), kernels.nanmean(u))
            ct, st = math.cos(theta), math.sin(theta)
            u1 = [a * ct + b * st for a, b in zip(u, v)]
            v1 = [-a * st + b * ct for a, b in zip(u, v)]
            phi = math.atan2(kernels.nanmean(w), kernels.nanmean(u))      # the trap
            cp, sp = math.cos(phi), math.sin(phi)
            u2 = [a * cp + b * sp for a, b in zip(u1, w)]
            w2 = [-a * sp + b * cp for a, b in zip(u1, w)]
            return (kernels.from_values(u2), kernels.from_values(v1),
                    kernels.from_values(w2), theta, phi)

        kernels.rotate_double = trapped
        self.addCleanup(setattr, kernels, 'rotate_double', real)

        u, v, w = wind(n=2000)
        with self.assertLogs('miniflux.rotate', level='DEBUG') as captured:
            period = rotate.rotate(make_period(u, v, w), make_cfg())
        errors = [record for record in captured.records
                  if record.levelno >= logging.ERROR]
        self.assertEqual(len(errors), 1, captured.output)
        self.assertIn('mean(w)', errors[0].getMessage())
        self.assertGreater(abs(kernels.nanmean(period['w'])), 1e-2)


class PurePathTest(RotateBehaviour, unittest.TestCase):
    USE_NUMPY = False


@unittest.skipUnless(kernels.NUMPY_AVAILABLE, "numpy is not installed")
class NumpyPathTest(RotateBehaviour, unittest.TestCase):
    USE_NUMPY = True


if __name__ == '__main__':
    unittest.main()
