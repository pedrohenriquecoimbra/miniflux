"""Time-lag maximisation: the sign convention, the window, the fallback, the shift.

ALGORITHMS.md section 14.3 is the headline case -- a scalar built as w delayed by 15
samples at 10 Hz must come back as +1.5 s -- and the rest of this file pins the four
things that are easy to get wrong around it: the window is inclusive at both ends
(2I+1 lags, not 2I), a peak on the rim is a failed search rather than an answer, the
applied shift truncates with NaN instead of wrapping, and the scan's covariance stays
a diagnostic that nothing downstream can mistake for a flux.
"""

import math
import random
import unittest
from types import SimpleNamespace

from miniflux import kernels, lag

NAN = float('nan')

#: Spare record on each side of a cut series, so any lag up to this is exact.
PAD = 40

#: The eight series keys of a period dict (CONTRACT 1); only some carry signal here.
SERIES_KEYS = ('u', 'v', 'w', 'ts', 'co2', 'h2o', 'ta', 'p_air')

#: The six meta keys lag.py writes per scalar (CONTRACT 10).
LAG_KEYS = ('_lag_samples', '_lag_s', '_lag_opt_samples', '_lag_opt_s',
            '_lag_default_used', '_lag_cov')


def record(n, seed=7):
    """One white-noise record long enough to cut a lagged series of length n out of.

    White noise gives the covariance curve a single sharp peak at the planted lag, so
    the recovered lag is a fact about the search and not about the waveform.
    """
    rng = random.Random(seed)
    return [rng.gauss(0.0, 1.0) for _ in range(n + 2 * PAD)]


def sensor(base, n, samples):
    """The n samples an instrument with physical lag `samples` records from `base`.

    Positive `samples` means the scalar arrives after the wind (ALGORITHMS 5.1), so
    its cut starts earlier in the record and ``c[i + samples] == w[i]`` holds exactly.
    Cutting every series out of one longer record wraps nothing around the ends.
    """
    start = PAD - samples
    return base[start:start + n]


def planted(n, samples, seed=7):
    """(w, c) of length n where c is w delayed by `samples`: ``c[i + samples] == w[i]``."""
    base = record(n, seed)
    return sensor(base, n, 0), sensor(base, n, samples)


def make_period(w, scalars, freq=10.0):
    """A period dict of the contract's shape carrying w and the named scalars."""
    period = {'meta': {'freq_hz': freq}, 't': []}
    for key in SERIES_KEYS:
        period[key] = kernels.new(len(w))
    period['w'] = kernels.from_values(w)
    for name in scalars:
        period[name] = kernels.from_values(scalars[name])
    return period


def make_cfg(method='covmax_default', windows=None, freq=10.0):
    """A stand-in Config: the [lag] and [period] namespaces lag.py is allowed to read."""
    if windows is None:
        windows = {'co2': (0.0, -2.0, 2.0)}        # (nominal_s, min_s, max_s)
    return SimpleNamespace(
        lag=SimpleNamespace(method=method, scalars=sorted(windows), windows=dict(windows)),
        period=SimpleNamespace(acquisition_frequency=freq),
    )


def centred(values):
    """The array a pre-centring caller would hand the scan."""
    arr = kernels.from_values(values)
    return kernels.sub_const(arr, kernels.nanmean(arr))


def forbidden(*args, **kwargs):
    """A cov_curve that fails the test if a method that must not scan scans anyway."""
    raise AssertionError('cov_curve was called by a method that does not scan')


class LagTest(unittest.TestCase):

    def spy_on_cov_curve(self):
        """Record every (lo, hi, len(curve)) the step asks for; returns the list."""
        calls = []
        real = kernels.cov_curve

        def spy(w, c, lo, hi):
            curve = real(w, c, lo, hi)
            calls.append((lo, hi, len(curve)))
            return curve

        kernels.cov_curve = spy
        self.addCleanup(setattr, kernels, 'cov_curve', real)
        return calls

    def forbid_cov_curve(self):
        real = kernels.cov_curve
        kernels.cov_curve = forbidden
        self.addCleanup(setattr, kernels, 'cov_curve', real)

    # ------------------------------------------------------- the planted lag comes back

    def test_planted_lag_is_recovered_exactly(self):
        """ALGORITHMS 14.3: co2 delayed 15 samples at 10 Hz must read +1.5 s."""
        w, c = planted(2000, 15)
        period = lag.apply_lags(make_period(w, {'co2': c}), make_cfg())
        meta = period['meta']
        self.assertEqual(meta['co2_lag_samples'], 15)
        self.assertEqual(meta['co2_lag_s'], 1.5)
        self.assertEqual(meta['co2_lag_opt_samples'], 15)
        self.assertEqual(meta['co2_lag_opt_s'], 1.5)
        self.assertFalse(meta['co2_lag_default_used'])
        # The peak is the variance of the shared record, so it is near 1 and positive.
        self.assertAlmostEqual(meta['co2_lag_cov'], 1.0, delta=0.15)

    def test_negative_lag_is_recovered_with_no_second_negation(self):
        """A scalar arriving BEFORE the wind is a negative lag, not a flipped positive."""
        w, c = planted(2000, -12)
        meta = lag.apply_lags(make_period(w, {'co2': c}), make_cfg())['meta']
        self.assertEqual(meta['co2_lag_samples'], -12)
        self.assertAlmostEqual(meta['co2_lag_s'], -1.2)

    def test_each_scalar_gets_its_own_lag(self):
        n = 1500
        base = record(n)
        period = make_period(sensor(base, n, 0),
                             {'co2': sensor(base, n, 8), 'h2o': sensor(base, n, -5)})
        cfg = make_cfg(windows={'co2': (0.0, -2.0, 2.0), 'h2o': (0.0, -2.0, 2.0)})
        meta = lag.apply_lags(period, cfg)['meta']
        self.assertEqual(meta['co2_lag_samples'], 8)
        self.assertEqual(meta['h2o_lag_samples'], -5)

    # ------------------------------------------------------------ the window is 2I + 1

    def test_window_is_evaluated_at_exactly_two_i_plus_one_lags(self):
        """+-2.0 s at 10 Hz is I = 20, so 41 candidate lags -- inclusive at both ends."""
        calls = self.spy_on_cov_curve()
        w, c = planted(800, 3)
        lag.apply_lags(make_period(w, {'co2': c}), make_cfg())
        self.assertEqual(len(calls), 1)
        lo, hi, evaluated = calls[0]
        self.assertEqual((lo, hi), (-20, 20))
        self.assertEqual(evaluated, 41)
        self.assertEqual(evaluated, 2 * 20 + 1)

    def test_the_top_lag_of_the_window_is_reachable(self):
        """GEddySoft slices 2I lags and is one short at the top; this one is not."""
        w, c = planted(1200, 5)                            # exactly the upper rim
        cfg = make_cfg(method='covmax', windows={'co2': (0.0, -0.5, 0.5)})
        meta = lag.apply_lags(make_period(w, {'co2': c}), cfg)['meta']
        self.assertEqual(meta['co2_lag_samples'], 5)
        self.assertEqual(meta['co2_lag_opt_samples'], 5)
        self.assertFalse(meta['co2_lag_default_used'])

    # ---------------------------------------------------------- the boundary fallback

    def test_peak_on_the_edge_falls_back_and_reports_both_lags(self):
        """maxcov&default: a rim peak is a failed search, and both numbers are kept."""
        w, c = planted(1200, 5)                            # the peak sits on hi = +5
        cfg = make_cfg(windows={'co2': (0.3, -0.5, 0.5)})  # nominal = +3 samples
        with self.assertLogs('miniflux.lag', level='WARNING'):
            meta = lag.apply_lags(make_period(w, {'co2': c}), cfg)['meta']
        self.assertTrue(meta['co2_lag_default_used'])
        self.assertEqual(meta['co2_lag_samples'], 3)       # the nominal was applied
        self.assertEqual(meta['co2_lag_s'], 0.3)
        self.assertEqual(meta['co2_lag_opt_samples'], 5)   # the optimum is still reported
        self.assertEqual(meta['co2_lag_opt_s'], 0.5)
        self.assertTrue(math.isfinite(meta['co2_lag_cov']))

    def test_lower_edge_also_falls_back(self):
        w, c = planted(1200, -5)
        cfg = make_cfg(windows={'co2': (0.0, -0.5, 0.5)})
        with self.assertLogs('miniflux.lag', level='WARNING'):
            meta = lag.apply_lags(make_period(w, {'co2': c}), cfg)['meta']
        self.assertTrue(meta['co2_lag_default_used'])
        self.assertEqual(meta['co2_lag_samples'], 0)
        self.assertEqual(meta['co2_lag_opt_samples'], -5)

    def test_covmax_keeps_the_edge_peak(self):
        """Without the &default rule the rim lag is applied as found."""
        w, c = planted(1200, 5)
        cfg = make_cfg(method='covmax', windows={'co2': (0.3, -0.5, 0.5)})
        meta = lag.apply_lags(make_period(w, {'co2': c}), cfg)['meta']
        self.assertEqual(meta['co2_lag_samples'], 5)
        self.assertFalse(meta['co2_lag_default_used'])

    # -------------------------------------------------------- applying it: truncation

    def test_positive_shift_truncates_with_nan_and_never_wraps(self):
        w, c = planted(400, 15)
        period = lag.apply_lags(make_period(w, {'co2': c}), make_cfg())
        aligned = period['co2']
        self.assertEqual(len(aligned), 400)
        for i in range(400 - 15):
            self.assertEqual(aligned[i], c[i + 15])
            self.assertEqual(aligned[i], w[i])             # which is the point of it all
        for i in range(400 - 15, 400):
            self.assertTrue(math.isnan(aligned[i]))
            # a wraparound would have put the head of the record here instead
            self.assertNotEqual(aligned[i], c[i - (400 - 15)])

    def test_negative_shift_puts_the_nan_edge_at_the_front(self):
        w, c = planted(400, -12)
        period = lag.apply_lags(make_period(w, {'co2': c}), make_cfg())
        aligned = period['co2']
        for i in range(12):
            self.assertTrue(math.isnan(aligned[i]))
        for i in range(12, 400):
            self.assertEqual(aligned[i], c[i - 12])

    def test_w_is_never_shifted(self):
        w, c = planted(400, 15)
        period = lag.apply_lags(make_period(w, {'co2': c}), make_cfg())
        self.assertEqual(list(period['w']), list(w))

    def test_scalars_outside_the_list_are_left_alone(self):
        """[lag] scalars names what moves; h2o is not in it here."""
        w, c = planted(400, 15)
        period = make_period(w, {'co2': c, 'h2o': c})
        lag.apply_lags(period, make_cfg())
        self.assertEqual(list(period['h2o']), list(c))
        self.assertNotIn('h2o_lag_samples', period['meta'])

    # ------------------------------------------------- the selection rule, in isolation

    def test_absolute_peak_wins_and_a_tie_keeps_the_lowest_lag(self):
        """|cov| covers both signs; strict '>' scanning ascending keeps the first lag."""
        curve = [1.0, -2.0, 2.0, 0.5]                      # lags -1, 0, +1, +2
        real = kernels.cov_curve
        kernels.cov_curve = lambda w, c, lo, hi: curve
        self.addCleanup(setattr, kernels, 'cov_curve', real)
        applied, opt, default_used, cov_peak = lag.find_lag(
            kernels.new(4), kernels.new(4), -1, 2, 0, 'covmax_default')
        self.assertEqual((applied, opt), (0, 0))           # the -2.0 at lag 0, not the +2.0
        self.assertFalse(default_used)
        self.assertEqual(cov_peak, -2.0)                   # reported signed

    def test_a_peak_that_only_rounding_separates_keeps_the_lowest_lag(self):
        """ALGORITHMS 5.4's tie rule has to survive the last bits of the covariance.

        Two lags whose covariances differ by less than the reduction tolerance of
        CONTRACT 3.8 are tied as far as either kernel path can tell: which of them is
        larger is decided by the summation order, not by the data. The rule says the
        lowest lag wins, so the margin is part of the comparison -- otherwise the pure
        and numpy paths select different lags on a period that is exactly symmetric.
        """
        peak = 0.08103054794871895
        for later in (peak * (1.0 + 1e-15), peak * (1.0 - 1e-15), peak):
            curve = [0.001, peak, 0.002, later, 0.001]     # lags -2 .. +2
            real = kernels.cov_curve
            kernels.cov_curve = lambda w, c, lo, hi: curve
            self.addCleanup(setattr, kernels, 'cov_curve', real)
            applied, opt, _used, _cov = lag.find_lag(
                kernels.new(5), kernels.new(5), -2, 2, 0, 'covmax')
            self.assertEqual((applied, opt), (-1, -1), "later = %r: " % later)

    # -------------------------------------------------------------- degenerate windows

    def test_all_nan_curve_falls_back_to_the_nominal_and_warns(self):
        w, _ = planted(500, 0)
        period = make_period(w, {'co2': [None] * 500})
        cfg = make_cfg(windows={'co2': (0.4, -1.0, 1.0)})
        with self.assertLogs('miniflux.lag', level='WARNING'):
            meta = lag.apply_lags(period, cfg)['meta']
        self.assertEqual(meta['co2_lag_samples'], 4)
        self.assertEqual(meta['co2_lag_opt_samples'], 4)
        self.assertTrue(meta['co2_lag_default_used'])
        self.assertTrue(math.isnan(meta['co2_lag_cov']))

    def test_unconfigured_window_applies_zero_not_the_nominal(self):
        """ALGORITHMS 5.7: tl_min = tl_max = 0 is no search, and a lag of zero."""
        self.forbid_cov_curve()
        w, c = planted(400, 15)
        cfg = make_cfg(windows={'co2': (0.4, 0.0, 0.0)})
        period = lag.apply_lags(make_period(w, {'co2': c}), cfg)
        meta = period['meta']
        self.assertEqual(meta['co2_lag_samples'], 0)
        self.assertEqual(meta['co2_lag_opt_samples'], 0)
        self.assertFalse(meta['co2_lag_default_used'])
        self.assertTrue(math.isnan(meta['co2_lag_cov']))
        self.assertEqual(list(period['co2']), list(c))

    def test_a_window_narrower_than_a_sample_is_still_a_search(self):
        """ALGORITHMS 5.7 exempts tl_min = tl_max = 0 **seconds**, and nothing else.

        At 10 Hz a -0.04 .. +0.04 s window rounds to the single lag 0, which 5.2 frames
        as a one-lag search: 5.4 selects it, and 5.5 finds it on both rims, so
        `covmax_default` reports the scan's optimum and applies the nominal instead.
        """
        w, c = planted(400, 15)
        cfg = make_cfg(windows={'co2': (0.5, -0.04, 0.04)})
        with self.assertLogs('miniflux.lag', level='WARNING'):
            period = lag.apply_lags(make_period(w, {'co2': c}), cfg)
        meta = period['meta']
        self.assertEqual(meta['co2_lag_opt_samples'], 0)
        self.assertEqual(meta['co2_lag_samples'], 5)          # the nominal, 0.5 s at 10 Hz
        self.assertTrue(meta['co2_lag_default_used'])
        self.assertTrue(math.isfinite(meta['co2_lag_cov']))   # a lag was evaluated

    def test_fixed_applies_the_nominal_without_scanning(self):
        self.forbid_cov_curve()
        w, c = planted(400, 15)
        cfg = make_cfg(method='fixed', windows={'co2': (0.7, -2.0, 2.0)})
        period = lag.apply_lags(make_period(w, {'co2': c}), cfg)
        meta = period['meta']
        self.assertEqual(meta['co2_lag_samples'], 7)
        self.assertEqual(meta['co2_lag_opt_samples'], 7)
        self.assertTrue(meta['co2_lag_default_used'])
        self.assertTrue(math.isnan(meta['co2_lag_cov']))
        self.assertTrue(math.isnan(period['co2'][-1]))     # it did shift, and truncated

    def test_none_shifts_nothing_and_reports_no_covariance(self):
        self.forbid_cov_curve()
        w, c = planted(400, 15)
        cfg = make_cfg(method='none', windows={'co2': (0.7, -2.0, 2.0)})
        period = lag.apply_lags(make_period(w, {'co2': c}), cfg)
        meta = period['meta']
        self.assertEqual(meta['co2_lag_samples'], 0)
        self.assertEqual(meta['co2_lag_s'], 0.0)
        self.assertEqual(meta['co2_lag_opt_samples'], 0)
        self.assertEqual(meta['co2_lag_opt_s'], 0.0)
        self.assertFalse(meta['co2_lag_default_used'])
        # NaN, not 0.0: no scan ran, so there is no covariance to report, and a zero
        # here would read as a measured covariance that happened to vanish.
        self.assertTrue(math.isnan(meta['co2_lag_cov']))
        self.assertEqual(list(period['co2']), list(c))

    def test_a_useless_frequency_reports_nan_seconds_and_never_raises(self):
        """A step never raises; an unusable rate means no shift and NaN seconds."""
        self.forbid_cov_curve()
        w, c = planted(400, 15)
        period = make_period(w, {'co2': c})
        period['meta']['freq_hz'] = NAN
        with self.assertLogs('miniflux.lag', level='WARNING'):
            meta = lag.apply_lags(period, make_cfg())['meta']
        self.assertEqual(meta['co2_lag_samples'], 0)
        self.assertTrue(math.isnan(meta['co2_lag_s']))
        self.assertTrue(math.isnan(meta['co2_lag_opt_s']))
        self.assertEqual(list(period['co2']), list(c))

    # --------------------------------------------- the scan's covariance stays a scan's

    def test_step_writes_only_its_six_keys_per_scalar(self):
        """No flux covariance leaves this step: <s>_lag_cov is the only number it reports,
        and ALGORITHMS 5.8 forbids reusing it -- the flux covariance is computed later,
        on the aligned and detrended period, with the two-pass estimator."""
        w, c = planted(800, 15)
        period = make_period(w, {'co2': c})
        before = set(period['meta'])
        meta = lag.apply_lags(period, make_cfg())['meta']
        self.assertEqual(set(meta) - before,
                         set('co2' + suffix for suffix in LAG_KEYS))

    def test_the_scan_runs_on_pre_centred_series(self):
        """ALGORITHMS 5.3: the one-pass scan is only ever handed centred series, which
        is what keeps it honest on a 400 ppm CO2 record."""
        w, c = planted(2000, 15)
        raw = [v + 400.0 for v in c]
        meta = lag.apply_lags(make_period(w, {'co2': raw}), make_cfg())['meta']
        self.assertEqual(meta['co2_lag_samples'], 15)
        self.assertEqual(meta['co2_lag_cov'],
                         kernels.cov_at_lag(centred(w), centred(raw), 15))


if __name__ == '__main__':
    unittest.main()
