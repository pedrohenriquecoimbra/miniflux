"""What despike.py does to a period: ALGORITHMS section 14.2 and the edges of section 3.3,
plus the trailing-run rule that kernels.mad_spike_mask deliberately leaves to this module.

Every test runs twice, once per kernels path. The spike mask is a *decision*, and CONTRACT
section 3.8 requires decisions to be identical with and without numpy -- not merely close.
"""

import datetime
import math
import random
import struct
import unittest

from types import SimpleNamespace

from miniflux import despike, kernels

NAN = float('nan')
SERIES = ('u', 'v', 'w', 'ts', 'co2', 'h2o', 'ta', 'p_air')


def next_up(v):
    """The next representable double above a positive v. math.nextafter is 3.9+."""
    bits = struct.unpack('<Q', struct.pack('<d', v))[0]
    return struct.unpack('<d', struct.pack('<Q', bits + 1))[0]


def make_cfg(variables=('u', 'v', 'w', 'ts', 'co2', 'h2o'), q=7.0, enabled=True):
    """The [despike] section alone, in the SimpleNamespace shape config.py builds."""
    return SimpleNamespace(
        despike=SimpleNamespace(enabled=enabled, variables=list(variables), q=q))


def make_period(**series):
    """A period of CONTRACT section 1 shape: every canonical key present, the named ones
    from the given lists of float, the rest all-NaN arrays of the same length."""
    n = len(next(iter(series.values()))) if series else 0
    start = datetime.datetime(2023, 7, 8, 12, 0, 0)
    period = {'meta': {}}
    for name in SERIES:
        values = series.get(name)
        period[name] = kernels.from_values(values) if values is not None else kernels.new(n)
    return period


def gaussian(n, seed):
    """n deterministic N(0, 1) samples as a list of float."""
    rng = random.Random(seed)
    return [rng.gauss(0.0, 1.0) for _ in range(n)]


class DespikeBehaviour(object):
    """The tests. Mixed into one TestCase per kernels path (see the bottom of the file)."""

    USE_NUMPY = False

    def setUp(self):
        self._restore = kernels.HAVE_NUMPY
        kernels.set_numpy(self.USE_NUMPY)

    def tearDown(self):
        kernels.set_numpy(self._restore)

    def assertNaN(self, value, msg=None):
        self.assertTrue(math.isnan(value), msg or ('expected NaN, got %r' % (value,)))

    # --- ALGORITHMS section 14.2, the vector a port must reproduce -------------------

    def test_one_interior_spike_in_a_thousand_normal_samples(self):
        # 1000 samples of N(0,1) with one set to +50: the half-width is q/0.6745 = 10.38
        # MADs, i.e. about 7 sigma for a normal series, so the spike is the only flag.
        values = gaussian(1000, seed=14)
        values[500] = 50.0
        period = make_period(u=values)
        despike.despike(period, make_cfg())

        self.assertEqual(period['meta']['n_spike_u'], 1)
        self.assertNaN(period['u'][500])
        self.assertEqual(period['u'][499], values[499])
        self.assertEqual(period['u'][501], values[501])
        self.assertEqual(kernels.count_finite(period['u']), 999)

    def test_the_same_spike_at_the_last_sample_is_not_flagged(self):
        # The trailing-run rule: the excursion never returns inside this record, so it is
        # left alone -- it is as likely a step or a drift as a spike.
        values = gaussian(1000, seed=14)
        values[999] = 50.0
        period = make_period(u=values)
        despike.despike(period, make_cfg())

        self.assertEqual(period['meta']['n_spike_u'], 0)
        self.assertEqual(period['u'][999], 50.0)
        self.assertEqual(kernels.count_finite(period['u']), 1000)

    def test_a_constant_series_flags_nothing(self):
        # MAD = 0 collapses the bounds onto the median, but both comparisons are strict,
        # so a sample sitting exactly on the median survives -- and here all of them do.
        period = make_period(ts=[293.15] * 200)
        despike.despike(period, make_cfg())

        self.assertEqual(period['meta']['n_spike_ts'], 0)
        self.assertEqual(list(period['ts']), [293.15] * 200)

    # --- section 3.3 edges -----------------------------------------------------------

    def test_an_all_nan_series_flags_nothing_and_does_not_raise(self):
        # median and MAD are NaN, every comparison is False. No guard, no crash.
        period = make_period(co2=[NAN] * 120)
        despike.despike(period, make_cfg())

        self.assertEqual(period['meta']['n_spike_co2'], 0)
        self.assertEqual(kernels.count_finite(period['co2']), 0)

    def test_a_stuck_sensor_flags_everything_that_differs_at_all(self):
        # MAD = 0 with a minority of differing samples is the stuck/quantised case, and it
        # is deliberately unguarded: >50 % identical values set both bounds to the median.
        values = [4.0] * 20 + [4.000001, 3.999999] + [4.0] * 20
        period = make_period(h2o=values)
        despike.despike(period, make_cfg())

        self.assertEqual(period['meta']['n_spike_h2o'], 2)
        self.assertNaN(period['h2o'][20])
        self.assertNaN(period['h2o'][21])

    def test_a_single_sample_and_an_empty_series(self):
        # Below any length the statistic could use: one sample has MAD = 0 and equals its
        # own median; an empty period has no median at all. Both flag nothing.
        one = make_period(w=[1.25])
        despike.despike(one, make_cfg())
        self.assertEqual(one['meta']['n_spike_w'], 0)
        self.assertEqual(list(one['w']), [1.25])

        empty = make_period(w=[])
        despike.despike(empty, make_cfg())
        self.assertEqual(empty['meta']['n_spike_w'], 0)
        self.assertEqual(len(empty['w']), 0)

    def test_the_half_width_is_q_over_0_6745_mads_and_the_bound_is_exclusive(self):
        # median 0, MAD 1 by construction, so the upper bound is exactly (q*1)/0.6745 =
        # 10.37805782... A probe sitting on it survives; one ULP above it is a spike.
        base = [-1.0, -1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0]
        bound = (7.0 * 1.0) / 0.6745
        # the probe is never last, so the trailing-run rule cannot take the flag back
        on = make_period(v=base[:4] + [bound] + base[4:])
        above = make_period(v=base[:4] + [next_up(bound)] + base[4:])
        despike.despike(on, make_cfg())
        despike.despike(above, make_cfg())

        self.assertEqual(on['meta']['n_spike_v'], 0)
        self.assertEqual(above['meta']['n_spike_v'], 1)
        self.assertNaN(above['v'][4])

    # --- the trailing-run rule, at its boundary ---------------------------------------

    def test_a_run_that_closes_one_sample_before_the_end_is_kept(self):
        values = gaussian(300, seed=7)
        for i in (295, 296, 297, 298):
            values[i] = 50.0
        period = make_period(co2=values)
        despike.despike(period, make_cfg())

        self.assertEqual(period['meta']['n_spike_co2'], 4)
        for i in (295, 296, 297, 298):
            self.assertNaN(period['co2'][i])
        self.assertEqual(period['co2'][299], values[299])

    def test_the_same_run_touching_the_last_sample_is_cleared_whole(self):
        # Not just the last flag: every sample of the open run comes back.
        values = gaussian(300, seed=7)
        for i in (296, 297, 298, 299):
            values[i] = 50.0
        period = make_period(co2=values)
        despike.despike(period, make_cfg())

        self.assertEqual(period['meta']['n_spike_co2'], 0)
        self.assertEqual(list(period['co2'])[296:], [50.0] * 4)

    def test_a_run_ending_on_a_nan_is_kept_because_nan_is_never_flagged(self):
        # The rule asks about the last *flag*, and NaN cannot carry one, so a trailing NaN
        # closes the run. Here that leaves the period with no finite sample at all.
        period = make_period(u=[0.0, 1.0, 2.0, 3.0, NAN])
        with self.assertLogs('miniflux.despike', level='WARNING'):
            despike.despike(period, make_cfg(q=1e-6))

        self.assertEqual(period['meta']['n_spike_u'], 4)
        self.assertEqual(kernels.count_finite(period['u']), 0)

    def test_an_infinity_is_missing_data_and_never_a_flag(self):
        # kernels: "Infinities count as non-finite everywhere". An inf carries no data, so
        # flagging it has the counter claim a removal that removed nothing.
        values = gaussian(1000, seed=14)
        values[400] = float('inf')
        period = make_period(u=values)
        despike.despike(period, make_cfg())

        self.assertEqual(period['meta']['n_spike_u'], 0)
        self.assertEqual(kernels.count_finite(period['u']), 999)

    def test_a_trailing_infinity_does_not_hand_a_spike_run_back(self):
        # The trailing-run rule asks about the last *flag*. If an inf can carry one, one
        # non-finite sample at the end walks the whole run back and returns three genuine
        # +50 spikes to the covariance -- the same three a finite last sample removes.
        values = gaussian(1000, seed=14)
        values[996] = values[997] = values[998] = 50.0
        values[999] = float('inf')
        period = make_period(u=values)
        despike.despike(period, make_cfg())

        self.assertEqual(period['meta']['n_spike_u'], 3)
        for i in (996, 997, 998):
            self.assertNaN(period['u'][i])

    def test_when_every_sample_is_flagged_the_whole_mask_clears(self):
        # The same four samples without the trailing NaN: the run is open at the end and
        # there is no sample inside the bounds to walk back to, so nothing is flagged.
        period = make_period(u=[0.0, 1.0, 2.0, 3.0])
        despike.despike(period, make_cfg(q=1e-6))

        self.assertEqual(period['meta']['n_spike_u'], 0)
        self.assertEqual(list(period['u']), [0.0, 1.0, 2.0, 3.0])

    # --- per-series independence and the counters that reach the output row ------------

    def test_each_series_is_despiked_on_its_own_statistic(self):
        # u spikes, v does not: the flag must not travel across variables. Sharing a mask
        # would throw away v's good sample at the same index for no reason.
        u = gaussian(400, seed=3)
        v = gaussian(400, seed=4)
        u[200] = 50.0
        period = make_period(u=u, v=v)
        despike.despike(period, make_cfg())

        self.assertEqual(period['meta']['n_spike_u'], 1)
        self.assertEqual(period['meta']['n_spike_v'], 0)
        self.assertNaN(period['u'][200])
        self.assertEqual(period['v'][200], v[200])

    def test_one_counter_per_configured_variable_and_none_for_the_others(self):
        period = make_period(u=gaussian(100, seed=5), v=gaussian(100, seed=6))
        returned = despike.despike(period, make_cfg(variables=('u', 'v')))

        self.assertIs(returned, period)
        for name in ('u', 'v'):
            self.assertIn('n_spike_' + name, period['meta'])
            self.assertIsInstance(period['meta']['n_spike_' + name], int)
        for name in ('w', 'ts', 'co2', 'h2o'):
            self.assertNotIn('n_spike_' + name, period['meta'])

    def test_a_variable_left_out_of_the_list_is_not_touched(self):
        values = gaussian(300, seed=8)
        values[100] = 50.0
        period = make_period(u=values)
        despike.despike(period, make_cfg(variables=('w',)))

        self.assertEqual(period['u'][100], 50.0)
        self.assertEqual(kernels.count_finite(period['u']), 300)

    def test_disabled_writes_zero_counters_and_changes_nothing(self):
        values = gaussian(300, seed=9)
        values[100] = 50.0
        period = make_period(u=values)
        despike.despike(period, make_cfg(variables=('u', 'ts'), enabled=False))

        self.assertEqual(period['meta']['n_spike_u'], 0)
        self.assertEqual(period['meta']['n_spike_ts'], 0)
        self.assertEqual(period['u'][100], 50.0)
        self.assertEqual(kernels.count_finite(period['u']), 300)


class PurePathTest(DespikeBehaviour, unittest.TestCase):
    USE_NUMPY = False


@unittest.skipUnless(kernels.NUMPY_AVAILABLE, "numpy is not installed")
class NumpyPathTest(DespikeBehaviour, unittest.TestCase):
    USE_NUMPY = True


if __name__ == '__main__':
    unittest.main()
