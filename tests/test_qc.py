"""What the quality flags are pinned to: the three steady-state flag bounds, the
sub-interval arithmetic, and every ITC stability branch and guard.

The steady-state series are built from one closed form so the expected statistic is a
hand-computable number rather than whatever the code happened to return. Each period of
five sub-intervals of m samples carries, in sub-interval j,

    w_i = p_i + A_j,   c_i = s_j * p_i + B_j,   p = [1, -1, 1, -1]

Because ``p`` sums to zero inside a sub-interval, the constants ``A_j`` and ``B_j`` are
invisible to the sub-interval covariances (a covariance is translation-invariant) and
visible to the covariance of the whole period. That is exactly the trend the test is
built to catch, and it lets one knob move the statistic across a flag bound:

    M = mean_j( C_j ) = (m / (m-1)) * mean_j(s_j)
    T = C_tot        = (m * sum_j(s_j) + m * Q) / (N - 1),  Q = sum_j (A_j-Abar)(B_j-Bbar)
    SST_pct = 100 * |M - T| / T

With m = 4, N = 20 and every s_j = 1 that is M = 4/3 and T = (20 + 4Q)/19.
"""

import logging
import math
import unittest
from types import SimpleNamespace

from miniflux import kernels, qc

NAN = float('nan')
INF = float('inf')

QUIET = logging.NullHandler()


def setUpModule():
    # qc warns on purpose whenever a guard fires, and several of these tests fire one. A
    # handler of any kind keeps logging's last-resort writer off stderr, and unlike
    # logging.disable() it leaves the records where assertLogs can still see them.
    logging.getLogger('miniflux').addHandler(QUIET)


def tearDownModule():
    logging.getLogger('miniflux').removeHandler(QUIET)

#: The zero-mean pattern one sub-interval of the synthetic series is built from.
PATTERN = (1.0, -1.0, 1.0, -1.0)


def trend_series(offsets):
    """(w, c) of five 4-sample sub-intervals, each with covariance 4/3.

    ``offsets`` is one (A_j, B_j) pair per sub-interval: a constant added to w and to c
    over that sub-interval only, which no sub-interval covariance can see.
    """
    w = []
    c = []
    for a, b in offsets:
        for p in PATTERN:
            w.append(p + a)
            c.append(p + b)
    return kernels.from_values(w), kernels.from_values(c)


def tilted(b):
    """The synthetic series with a step of +b, -b in c across the first two sub-intervals.

    Q = 2b, so the whole-period covariance is (20 + 8b)/19 while the mean of the five
    sub-interval covariances stays 4/3.
    """
    return trend_series([(1.0, b), (-1.0, -b), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)])


def expected_pct(q):
    """SST_pct [%] of the synthetic series, from the closed form in the module docstring."""
    m_sub = 4.0 / 3.0
    total = (20.0 + 4.0 * q) / 19.0
    return 100.0 * abs((m_sub - total) / total)


def deviation(model, measured):
    """|(model - measured) / model|, written out here so the test pins the definition."""
    return abs((model - measured) / model)


def make_cfg(pair=('w', 'co2'), itc=True, latitude=50.0):
    """The three config keys qc.quality is allowed to read."""
    return SimpleNamespace(qc=SimpleNamespace(steady_state_pair=list(pair), itc=itc),
                           site=SimpleNamespace(latitude=latitude))


def make_period(w, c, **meta):
    """A period carrying only what qc needs: the pair of series and the meta scalars."""
    period = {'meta': dict(meta), 'w': w, 'co2': c}
    return period


class SteadyStateArithmetic(unittest.TestCase):
    """The five sub-intervals, the remainder, and how the five combine."""

    # w = [0,1] five times then one leftover sample; c pairs with differences 2,4,6,8,10.
    # Each two-sample sub-interval covariance is (dw * dc)/(2-1-1+1) = dc/2, so the five
    # are 1, 2, 3, 4, 5 and their mean is 3. The 11th sample is outside every
    # sub-interval and inside the total.
    W = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    C = [0.0, 2.0, 0.0, 4.0, 0.0, 6.0, 0.0, 8.0, 0.0, 10.0, 0.0]

    def test_hand_computed_series(self):
        """N = 11, m = 2: mean of the subs is 3, C_tot is 18/11, so SST is 83.33 %."""
        w = kernels.from_values(self.W)
        c = kernels.from_values(self.C)
        # C_tot = (sum(w*c) - N*wbar*cbar)/(N-1) = (30 - 11*(5/11)*(30/11))/10 = 18/11.
        total = 18.0 / 11.0
        self.assertAlmostEqual(kernels.cov(w, c, 0, 11), total, places=12)
        pct, flag = qc.steady_state(w, c)
        self.assertAlmostEqual(pct, 100.0 * abs(3.0 - total) / total, places=9)
        self.assertAlmostEqual(pct, 83.33333333333333, places=9)
        self.assertEqual(flag, 1)

    def test_remainder_is_outside_the_subintervals_and_inside_the_total(self):
        """Dropping the 11th sample changes the statistic: it counted in C_tot only."""
        full = qc.steady_state(kernels.from_values(self.W), kernels.from_values(self.C))[0]
        short = qc.steady_state(kernels.from_values(self.W[:10]),
                                kernels.from_values(self.C[:10]))[0]
        self.assertAlmostEqual(short, 80.0, places=9)
        self.assertNotAlmostEqual(full, short, places=6)

    def test_nan_subinterval_is_skipped_and_the_mean_rests_on_four(self):
        """The fifth sub-interval goes NaN; the statistic is the mean of the other four."""
        c = list(self.C)
        c[8] = NAN
        c[9] = NAN
        w = kernels.from_values(self.W)
        pct, flag = qc.steady_state(w, kernels.from_values(c))
        # Nine finite pairs remain: C_tot = (20 - 9*(4/9)*(20/9))/8 = 25/18, and the mean
        # of the four surviving sub-intervals is 2.5, not 2.0.
        total = 25.0 / 18.0
        self.assertAlmostEqual(pct, 100.0 * abs(2.5 - total) / total, places=9)
        self.assertAlmostEqual(pct, 80.0, places=9)
        self.assertEqual(flag, 1)

    def test_subinterval_covariance_is_about_its_own_mean(self):
        """A step between sub-intervals moves C_tot and leaves every C_j alone.

        That is the whole mechanism of the test: were the sub-intervals demeaned with the
        period mean, the statistic would not respond to a trend at all.
        """
        flat = qc.steady_state(*tilted(0.0))[0]
        stepped = qc.steady_state(*tilted(-1.0))[0]
        self.assertAlmostEqual(flat, expected_pct(0.0), places=9)
        self.assertAlmostEqual(stepped, expected_pct(-2.0), places=9)
        self.assertGreater(stepped, flat)


class SteadyStateFlags(unittest.TestCase):
    """The 0/1/2 bounds, on both sides of both crossings."""

    def check(self, b, pct, flag):
        got_pct, got_flag = qc.steady_state(*tilted(b))
        self.assertAlmostEqual(got_pct, expected_pct(2.0 * b), places=9)
        self.assertAlmostEqual(got_pct, pct, places=6)
        self.assertEqual(got_flag, flag)

    def test_below_and_above_the_30_percent_bound(self):
        self.check(0.0, 26.666666666666668, 0)
        self.check(-0.25, 40.74074074074074, 1)

    def test_below_and_above_the_100_percent_bound(self):
        self.check(-0.875, 94.87179487179486, 1)
        self.check(-1.0, 111.11111111111111, 2)

    def test_the_bounds_themselves_are_inclusive(self):
        # ALGORITHMS 11.1 writes `<= 30` and `<= 100`. A period landing exactly on a
        # bound is the only thing that tells `<=` from `<`, and the tilt family above
        # never lands on one (30.000000000000004 is the nearest it gets to 30), so the
        # 30 is pinned on the flag function directly and the 100 through a tilt that
        # reaches it exactly.
        self.assertEqual(qc._sst_flag(30.0), 0)
        self.assertEqual(qc._sst_flag(100.0), 1)
        self.assertEqual(qc.steady_state(*tilted(-0.9166666666666667)), (100.0, 1))

    def test_the_statistic_is_not_capped(self):
        """30 and 100 are flag bounds, not clamps: a reversing flux prints hundreds."""
        pct, flag = qc.steady_state(*tilted(-5.0))
        self.assertAlmostEqual(pct, expected_pct(-10.0), places=9)
        self.assertGreater(pct, 200.0)
        self.assertEqual(flag, 2)

    def test_zero_period_covariance_is_nan_and_flag_two(self):
        w = kernels.from_values([1.0] * 20)
        c = kernels.from_values(range(20))
        pct, flag = qc.steady_state(w, c)
        self.assertTrue(math.isnan(pct))
        self.assertEqual(flag, 2)

    def test_non_finite_statistic_is_flag_two(self):
        """An all-NaN pair gives a NaN covariance, which is discard, not missing."""
        w = kernels.from_values([NAN] * 20)
        c = kernels.from_values(range(20))
        pct, flag = qc.steady_state(w, c)
        self.assertTrue(math.isnan(pct))
        self.assertEqual(flag, 2)

    def test_all_subintervals_nan_is_flag_two(self):
        """N < 5 gives m = 0: every sub-interval is empty even though C_tot is fine."""
        w = kernels.from_values([0.0, 1.0, 2.0])
        c = kernels.from_values([0.0, 2.0, 4.0])
        self.assertTrue(math.isfinite(kernels.cov(w, c)))
        pct, flag = qc.steady_state(w, c)
        self.assertTrue(math.isnan(pct))
        self.assertEqual(flag, 2)


class ItcModels(unittest.TestCase):
    """Every stability branch, every coefficient, and the |T*| divergence."""

    # sd(w) = 0.5, sd(u) = 1.0, sd(ts) = 0.5, ustar = 0.5, T* = -0.2 so |T*| = 0.2.
    USTAR = 0.5
    VAR_W = 0.25
    VAR_U = 1.0
    VAR_TS = 0.25
    COV_W_TS = 0.1
    LAT = 50.0
    SW_MEAS = 1.0
    SU_MEAS = 2.0
    ST_MEAS = 2.5

    def call(self, z_l, **kw):
        kwargs = {'var_w': self.VAR_W, 'var_u': self.VAR_U, 'var_ts': self.VAR_TS,
                  'ustar': self.USTAR, 'cov_w_ts': self.COV_W_TS, 'latitude': self.LAT}
        kwargs.update(kw)
        return qc.itc(kwargs['var_w'], kwargs['var_u'], kwargs['var_ts'],
                      kwargs['ustar'], kwargs['cov_w_ts'], z_l, kwargs['latitude'])

    def neutral_log(self, ustar=None, lat=None):
        """ln(Fcor * z+ / ustar), with the Coriolis parameter written out longhand."""
        omega = 2.0 * math.pi / (24 * 60 * 60)
        fcor = abs(2.0 * omega * math.sin((lat if lat is not None else self.LAT)
                                          * math.pi / 180.0))
        return math.log(fcor * 1.0 / (ustar if ustar is not None else self.USTAR))

    def test_omega_is_the_solar_day(self):
        self.assertAlmostEqual(qc.OMEGA, 7.2722052166430395e-05, places=18)
        self.assertEqual(qc.ZPLUS, 1.0)

    def test_unstable_wind_models(self):
        """z/L < -0.2: 1.3*(1-2z)^(1/3) and 4.15*|z|^(1/8), no Coriolis term."""
        itc_w, itc_u, _ = self.call(-0.5)
        self.assertAlmostEqual(itc_w, deviation(1.3 * 2.0 ** (1.0 / 3.0), self.SW_MEAS),
                               places=12)
        self.assertAlmostEqual(itc_u, deviation(4.15 * 0.5 ** (1.0 / 8.0), self.SU_MEAS),
                               places=12)

    def test_neutral_wind_models(self):
        """z/L >= -0.2: 0.21*ln(Fcor*z+/ustar)+3.1 and 0.44*ln(...)+6.3."""
        for z_l in (-0.2, 0.0625, 1.0):
            itc_w, itc_u, _ = self.call(z_l)
            ln = self.neutral_log()
            self.assertAlmostEqual(itc_w, deviation(0.21 * ln + 3.1, self.SW_MEAS),
                                   places=12)
            self.assertAlmostEqual(itc_u, deviation(0.44 * ln + 6.3, self.SU_MEAS),
                                   places=12)

    def test_wind_model_boundary_is_minus_zero_point_two(self):
        """-0.2 exactly is the neutral branch; a hair below it is the unstable one."""
        neutral = self.call(-0.2)[0]
        unstable = self.call(-0.20000001)[0]
        self.assertAlmostEqual(neutral, deviation(0.21 * self.neutral_log() + 3.1,
                                                 self.SW_MEAS), places=12)
        self.assertNotAlmostEqual(neutral, unstable, places=6)

    def test_temperature_branches(self):
        """The four sT models, including both interior boundaries."""
        cases = [
            (-2.0, 2.0 ** (-1.0 / 3.0)),          # z/L < -1
            (-1.0, 1.0 ** (-1.0 / 4.0)),          # -1 is the second branch, not the first
            (-0.5, 0.5 ** (-1.0 / 4.0)),
            (-0.0625, 0.5 * 0.0625 ** (-0.5)),    # -0.0625 is the third branch
            (-0.01, 0.5 * 0.01 ** (-0.5)),
            (0.01, 0.5 * 0.01 ** (-0.5)),
            (0.5, 1.4 * 0.5 ** (-1.0 / 4.0)),
        ]
        for z_l, model in cases:
            itc_t = self.call(z_l)[2]
            self.assertAlmostEqual(itc_t, deviation(model, self.ST_MEAS), places=12,
                                   msg='z/L = %r' % (z_l,))

    def test_z_l_of_exactly_002_falls_through_to_the_else_branch(self):
        """The last branch is an else: 0.02 takes 1.4*|z|^(-1/4), not 0.5*|z|^(-1/2)."""
        itc_t = self.call(0.02)[2]
        self.assertAlmostEqual(itc_t, deviation(1.4 * 0.02 ** (-1.0 / 4.0), self.ST_MEAS),
                               places=12)
        self.assertNotAlmostEqual(itc_t, deviation(0.5 * 0.02 ** (-0.5), self.ST_MEAS),
                                  places=6)

    def test_measured_characteristics_use_ustar_and_the_variances(self):
        """sw = sd(w)/ustar, su = sd(u)/ustar, sT = sd(ts)/|T*|, all ddof = 1."""
        model_w = 1.3 * 2.0 ** (1.0 / 3.0)
        itc_w = self.call(-0.5, var_w=4.0)[0]
        self.assertAlmostEqual(itc_w, deviation(model_w, 2.0 / self.USTAR), places=12)

    def test_itc_t_divides_by_the_absolute_temperature_scale(self):
        """CONTRACT 20.5: |T*|, not the parent's signed T*.

        Unstable air has cov(w,ts) > 0, hence T* < 0. With the signed scale sT_meas is
        negative, the deviation exceeds 1 by construction and carries no information.
        """
        model = 0.5 ** (-1.0 / 4.0)
        tstar = -self.COV_W_TS / self.USTAR
        self.assertLess(tstar, 0.0)
        # A period that sits exactly on the model: the honest deviation is zero, and the
        # parent's signed scale reports 2, which is what "always > 1" costs.
        var_ts = (model * abs(tstar)) ** 2
        itc_t = self.call(-0.5, var_ts=var_ts)[2]
        self.assertAlmostEqual(itc_t, deviation(model, math.sqrt(var_ts) / abs(tstar)),
                               places=12)
        self.assertAlmostEqual(itc_t, 0.0, places=12)
        self.assertAlmostEqual(deviation(model, math.sqrt(var_ts) / tstar), 2.0, places=12)

    def test_zero_heat_flux_gives_no_temperature_scale(self):
        """T* = 0 would be a division by zero; only ITC_T is lost."""
        itc_w, itc_u, itc_t = self.call(-0.5, cov_w_ts=0.0)
        self.assertTrue(math.isfinite(itc_w))
        self.assertTrue(math.isfinite(itc_u))
        self.assertTrue(math.isnan(itc_t))

    def test_non_finite_variance_gives_nan_for_that_component_only(self):
        itc_w, itc_u, itc_t = self.call(-0.5, var_w=NAN)
        self.assertTrue(math.isnan(itc_w))
        self.assertTrue(math.isfinite(itc_u))
        self.assertTrue(math.isfinite(itc_t))


class ItcGuards(unittest.TestCase):
    """Every guard returns three NaNs rather than an inf or a domain error."""

    # var_w, var_u, var_ts, ustar, cov_w_ts, z_l, latitude
    ARGS = (0.25, 1.0, 0.25, 0.5, 0.1, -0.5, 50.0)

    def call(self, **kw):
        names = ('var_w', 'var_u', 'var_ts', 'ustar', 'cov_w_ts', 'z_l', 'latitude')
        args = dict(zip(names, self.ARGS))
        args.update(kw)
        return qc.itc(*[args[n] for n in names])

    def assert_all_nan(self, result):
        self.assertEqual(len(result), 3)
        for value in result:
            self.assertTrue(math.isnan(value), msg='expected NaN, got %r' % (value,))

    def test_the_reference_case_is_finite(self):
        for value in self.call():
            self.assertTrue(math.isfinite(value))

    def test_ustar_not_positive(self):
        for ustar in (0.0, -0.5, NAN, INF):
            self.assert_all_nan(self.call(ustar=ustar))

    def test_latitude_zero_is_the_coriolis_guard(self):
        """Fcor = 0 at the equator, and the neutral model would take ln(0).

        The guard is unconditional: it fires on the unstable branch too, which does not
        use Fcor, because a latitude of 0 is the config default and means "not set".
        """
        self.assert_all_nan(self.call(latitude=0.0, z_l=-0.5))
        self.assert_all_nan(self.call(latitude=0.0, z_l=0.5))
        self.assert_all_nan(self.call(latitude=-0.0))
        for value in self.call(latitude=-50.0):
            self.assertTrue(math.isfinite(value))

    def test_z_l_exactly_zero(self):
        """0.5 * 0**(-0.5) is a division by zero in the third temperature branch."""
        self.assert_all_nan(self.call(z_l=0.0))
        self.assert_all_nan(self.call(z_l=-0.0))

    def test_z_l_non_finite(self):
        for z_l in (NAN, INF, -INF):
            self.assert_all_nan(self.call(z_l=z_l))


class Quality(unittest.TestCase):
    """The step: what it reads from meta, what it writes, and that it never raises."""

    META = {'var_w': 0.25, 'var_u': 1.0, 'var_ts': 0.25, 'ustar': 0.5,
            'cov_w_ts': 0.1, 'z_l': -0.5}

    def period(self, b=0.0, **meta):
        w, c = tilted(b)
        full = dict(self.META)
        full.update(meta)
        return make_period(w, c, **full)

    def test_writes_every_key(self):
        period = qc.quality(self.period(), make_cfg())
        meta = period['meta']
        self.assertAlmostEqual(meta['sst_pct'], expected_pct(0.0), places=9)
        self.assertEqual(meta['sst_flag'], 0)
        self.assertAlmostEqual(meta['tstar'], -0.1 / 0.5, places=12)
        for key in ('itc_w', 'itc_u', 'itc_t'):
            self.assertTrue(math.isfinite(meta[key]), msg=key)

    def test_returns_the_same_dict(self):
        period = self.period()
        self.assertIs(qc.quality(period, make_cfg()), period)

    def test_steady_state_pair_is_configurable(self):
        period = self.period()
        period['ts'] = period['co2']
        same = qc.quality(period, make_cfg(pair=('w', 'ts')))['meta']['sst_pct']
        self.assertAlmostEqual(same, expected_pct(0.0), places=9)

    def test_itc_off_writes_only_the_steady_state(self):
        meta = qc.quality(self.period(), make_cfg(itc=False))['meta']
        self.assertIn('sst_pct', meta)
        for key in ('itc_w', 'itc_u', 'itc_t', 'tstar'):
            self.assertNotIn(key, meta)

    def test_latitude_zero_leaves_itc_nan_but_keeps_the_steady_state(self):
        meta = qc.quality(self.period(), make_cfg(latitude=0.0))['meta']
        self.assertEqual(meta['sst_flag'], 0)
        for key in ('itc_w', 'itc_u', 'itc_t'):
            self.assertTrue(math.isnan(meta[key]), msg=key)
        self.assertAlmostEqual(meta['tstar'], -0.2, places=12)

    def test_missing_stability_gives_nan_itc_and_does_not_raise(self):
        period = self.period()
        del period['meta']['z_l']
        meta = qc.quality(period, make_cfg())['meta']
        self.assertTrue(math.isnan(meta['itc_w']))
        self.assertEqual(meta['sst_flag'], 0)

    def test_missing_pair_series_is_nan_and_flag_two(self):
        period = self.period()
        del period['co2']
        meta = qc.quality(period, make_cfg())['meta']
        self.assertTrue(math.isnan(meta['sst_pct']))
        self.assertEqual(meta['sst_flag'], 2)

    def test_the_itc_warning_names_the_input_that_is_actually_missing(self):
        # ustar, z/L and latitude are all healthy here; T* = 0 because the heat flux is,
        # and that is what leaves itc_t undefined. A warning that prints the three
        # healthy numbers and not the zero sends the reader to the wrong config key.
        period = self.period(cov_w_ts=0.0)
        with self.assertLogs('miniflux.qc', level='WARNING') as caught:
            meta = qc.quality(period, make_cfg())['meta']
        self.assertTrue(math.isnan(meta['itc_t']))
        self.assertTrue(math.isfinite(meta['itc_w']))
        self.assertIn('cov_w_ts', caught.output[0])
        self.assertIn('itc_t', caught.output[0])

    def test_the_steady_state_warning_does_not_blame_the_period_covariance(self):
        # N = 3 gives m = 0: every sub-interval is empty, so the statistic cannot be
        # formed even though cov over the period is a perfectly good 2.0.
        period = make_period(kernels.from_values([0.0, 1.0, 2.0]),
                             kernels.from_values([0.0, 2.0, 4.0]), **self.META)
        with self.assertLogs('miniflux.qc', level='WARNING') as caught:
            meta = qc.quality(period, make_cfg())['meta']
        self.assertEqual(meta['sst_flag'], 2)
        self.assertEqual(kernels.cov(period['w'], period['co2']), 2.0)
        self.assertNotIn('non-finite period covariance', caught.output[0])

    def test_empty_period_does_not_raise(self):
        period = {'meta': {}, 'w': kernels.new(0), 'co2': kernels.new(0)}
        meta = qc.quality(period, make_cfg())['meta']
        self.assertTrue(math.isnan(meta['sst_pct']))
        self.assertEqual(meta['sst_flag'], 2)
        self.assertTrue(math.isnan(meta['tstar']))
        self.assertTrue(math.isnan(meta['itc_t']))


if __name__ == '__main__':
    unittest.main()
