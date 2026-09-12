"""Generate a deterministic synthetic 20 Hz raw file, and print what was planted in it.

The file matches what ``examples/miniflux.ini`` declares: a one-line header, ISO
timestamps, comma separated, wind in m s-1, a sonic temperature in K, CO2 in ppm and H2O
in ppt as dry mixing ratios, and a pressure in Pa. One 30-minute period, 36000 samples.

Three quantities are planted so that a run can be checked against them:

* a **w'T' covariance**: the sonic temperature is a linear function of the vertical wind
  plus independent noise, so the covariance is known -- and it is reported here in the
  *rotated* frame, because that is the frame ``moments`` computes in;
* a **CO2 lag**: the CO2 series is the vertical wind delayed by 15 samples (0.75 s at
  20 Hz), so the lag search must return exactly that;
* a **spike count**: single samples set far outside the distribution, which the MAD test
  must flag and nothing else.

The generator repeats, in about forty lines of arithmetic, the parts of the pipeline that
move those three numbers (the double rotation, the lag shift, the joint-finite covariance).
It does that deliberately: a prediction computed by the code under test would prove
nothing.

Usage::

    python examples/make_sample.py [OUT.dat]          # default ./sample_20hz.dat
    python -m miniflux run examples/miniflux.ini      # input_glob = ./*.dat

The random seed is a literal, so two runs produce byte-identical files.
"""

import argparse
import math
import sys
from datetime import datetime, timedelta

SEED = 20250908                     # a literal, so the file is reproducible
START = datetime(2025, 9, 8, 19, 30, 0, 50000)   # first sample; 19:30:00.000 belongs to the
                                                 # previous period under closed = right
FREQ_HZ = 20.0
N = 36000                           # 30 minutes at 20 Hz
LAG_SAMPLES = 15                    # planted CO2 lag: 0.75 s at 20 Hz

# Mean state. The tilt is real: mean w and mean v are non-zero in the sonic frame, which is
# what gives the double rotation something to remove.
U_MEAN = 2.00                       # [m s-1] along the sonic axis
V_MEAN = -0.25                      # [m s-1] cross-wind offset -> a yaw angle
W_MEAN = 0.06                       # [m s-1] tilt              -> a pitch angle
TS_MEAN = 293.15                    # [K]
CO2_MEAN = 420.0                    # [ppm], dry mixing ratio
H2O_MEAN = 12.0                     # [ppt], dry mixing ratio
P_AIR = 99000.0                     # [Pa], constant

# Turbulence: an AR(1) process per component, so the autocovariance has a single peak at
# zero shift and the planted lag is the only maximum of the lag curve.
U_SD, U_PHI = 0.60, 0.90            # [m s-1]
V_SD, V_PHI = 0.40, 0.85
W_SD, W_PHI = 0.35, 0.70

U_W_GAIN = -0.80                    # [-] momentum: u' carries -0.8 w', so <u'w'> < 0 and
                                    # ustar ~ 0.31 m s-1 instead of the ~0 of two
                                    # independent processes
TS_GAIN = 1.24                      # [K / (m s-1)]  -> w'T' ~ 0.152 K m s-1 ~ H ~ 180 W m-2
TS_NOISE = 0.15                     # [K]
CO2_GAIN = -14.0                    # [ppm / (m s-1)] negative: daytime uptake
CO2_NOISE = 2.0                     # [ppm]
H2O_GAIN = 0.60                     # [ppt / (m s-1)] positive: evaporation
H2O_NOISE = 0.05                    # [ppt]

# Planted spikes: sample index -> replaced by the mean plus this many standard deviations.
# None sits at the last index, where the trailing-run rule would clear the flag instead.
SPIKE_SD = 50.0
SPIKES = {'u': (1000, 12345, 23456), 'v': (5000, 30001), 'co2': (700, 7007, 17017, 27027)}

# Written with a fixed number of decimals, as a logger writes them. Each value is read back
# out of its own text before anything is predicted from it, so the planted numbers below are
# the numbers the file actually holds, not the ones that were in memory before formatting.
FORMATS = (('Ux', '%.6f'), ('Uy', '%.6f'), ('Uz', '%.6f'), ('Ts', '%.4f'),
           ('CO2', '%.3f'), ('H2O', '%.5f'), ('Press', '%.1f'))
NAN = float('nan')


def main(argv=None):
    """Write the sample file and print the planted values. Returns the exit code (int)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('output', nargs='?', default='./sample_20hz.dat',
                        help='where to write the raw file (default ./sample_20hz.dat)')
    args = parser.parse_args(argv)

    columns = build()
    write_file(args.output, columns)
    report(args.output, columns)
    return 0


# ------------------------------------------------------------------ the synthetic signal

def build():
    """Build the seven data columns, quantised to the precision they are written at.

    Returns a dict of name -> list of float, each of length N: 'u', 'v', 'w', 'ts', 'co2',
    'h2o', 'p_air'. Units are the file's: m s-1, K, ppm, ppt, Pa.
    """
    import random                    # local, so the module-level namespace stays the data
    rng = random.Random(SEED)

    # N + LAG_SAMPLES of w: the first LAG_SAMPLES drive the CO2 that arrives inside the
    # period, so the planted lag holds from the very first sample of the file.
    w_source = ar1(rng, N + LAG_SAMPLES, W_SD, W_PHI)
    w = [W_MEAN + value for value in w_source[LAG_SAMPLES:]]
    u = [U_MEAN + value + U_W_GAIN * w_source[LAG_SAMPLES + i]
         for i, value in enumerate(ar1(rng, N, U_SD, U_PHI))]
    v = [V_MEAN + value for value in ar1(rng, N, V_SD, V_PHI)]

    ts = [TS_MEAN + TS_GAIN * w_source[LAG_SAMPLES + i] + TS_NOISE * rng.gauss(0.0, 1.0)
          for i in range(N)]
    # co2[j] is driven by the wind LAG_SAMPLES earlier, so cov(w_i, co2_{i+L}) peaks at
    # L = +LAG_SAMPLES -- the sign convention of ALGORITHMS 5.1, scalar after the wind.
    co2 = [CO2_MEAN + CO2_GAIN * w_source[i] + CO2_NOISE * rng.gauss(0.0, 1.0)
           for i in range(N)]
    h2o = [H2O_MEAN + H2O_GAIN * w_source[LAG_SAMPLES + i] + H2O_NOISE * rng.gauss(0.0, 1.0)
           for i in range(N)]
    columns = {'u': u, 'v': v, 'w': w, 'ts': ts, 'co2': co2, 'h2o': h2o,
               'p_air': [P_AIR] * N}

    for name, indices in SPIKES.items():
        spread = {'u': U_SD, 'v': V_SD, 'co2': abs(CO2_GAIN) * W_SD}[name]
        for index in indices:
            columns[name][index] = columns[name][index] + SPIKE_SD * spread

    # Quantise now: every prediction below is made on the values the file will parse back to.
    for (name, fmt), key in zip(FORMATS, ('u', 'v', 'w', 'ts', 'co2', 'h2o', 'p_air')):
        columns[key] = [float(fmt % value) for value in columns[key]]
    return columns


def ar1(rng, n, sd, phi):
    """A zero-mean AR(1) series: x_i = phi x_{i-1} + sqrt(1-phi^2) sd g_i. Returns list."""
    step = sd * math.sqrt(1.0 - phi * phi)
    out = [sd * rng.gauss(0.0, 1.0)]
    for _i in range(n - 1):
        out.append(phi * out[-1] + step * rng.gauss(0.0, 1.0))
    return out


def write_file(path, columns):
    """Write the raw file: one header line, then one line per sample. path: str."""
    names = [name for name, _fmt in FORMATS]
    keys = ('u', 'v', 'w', 'ts', 'co2', 'h2o', 'p_air')
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        handle.write('TIMESTAMP,' + ','.join(names) + '\n')
        step = timedelta(seconds=1.0 / FREQ_HZ)
        stamp = START
        for i in range(N):
            cells = [fmt % columns[key][i] for (_name, fmt), key in zip(FORMATS, keys)]
            handle.write(stamp.strftime('%Y-%m-%d %H:%M:%S.%f') + ',' + ','.join(cells) + '\n')
            stamp = stamp + step


# ------------------------------------------------------------------ what was planted

def report(path, columns):
    """Print the planted values a run must reproduce. Nothing here calls miniflux."""
    despiked = dict((name, mask_spikes(values, SPIKES.get(name, ())))
                    for name, values in columns.items())
    u2, v2, w2, theta, phi = double_rotation(despiked['u'], despiked['v'], despiked['w'])
    co2_aligned = shift(despiked['co2'], LAG_SAMPLES)

    print('wrote %d samples to %s' % (N, path))
    print('  period            %s .. %s (closed right, labelled by the end)'
          % (START - timedelta(seconds=1.0 / FREQ_HZ), START + timedelta(seconds=1800.0
                                                                        - 1.0 / FREQ_HZ)))
    print('PLANTED -> the run must reproduce these')
    print('  N_IN                 %d' % N)
    for name in ('u', 'v', 'w', 'ts', 'co2', 'h2o'):
        print('  N_SPIKE_%-12s %d' % (name.upper(), len(SPIKES.get(name, ()))))
    print('  CO2_LAG              %+.2f s   (%+d samples)'
          % (LAG_SAMPLES / FREQ_HZ, LAG_SAMPLES))
    print('  H2O_LAG              +0.00 s   (no delay planted in H2O)')
    print('  THETA                %+.6f deg  (yaw, from mean v / mean u)'
          % math.degrees(theta))
    print('  PHI                  %+.6f deg  (pitch, from mean w1 / mean u1)'
          % math.degrees(phi))
    # WS is a mean over the despiked series: rotate runs after despike, so the spiked
    # samples are already NaN when the unrotated means are taken.
    print('  WS                   %.6f m s-1' % math.hypot(nanmean(despiked['u']),
                                                           nanmean(despiked['v'])))
    print('  COV_W_TS             %.9f K m s-1   (rotated w, joint finite mask)'
          % cov(w2, despiked['ts']))
    print('  cov(w, co2) at lag   %.9f ppm m s-1  -> FC_L0 = that x 1e6 x <1/Vd>'
          % cov(w2, co2_aligned))
    print('  cov(w, h2o) at lag 0 %.9f ppt m s-1' % cov(w2, despiked['h2o']))
    print('  VAR_U VAR_V VAR_W    %.6f %.6f %.6f m2 s-2'
          % (cov(u2, u2), cov(v2, v2), cov(w2, w2)))
    print('  USTAR                %.6f m s-1'
          % (cov(u2, w2) ** 2 + cov(v2, w2) ** 2) ** 0.25)
    print('  T_SONIC PA           %.4f K  %.1f Pa' % (nanmean(columns['ts']), P_AIR))


def mask_spikes(values, indices):
    """The series as despiking leaves it: the planted spikes replaced by NaN. Returns list."""
    out = list(values)
    for index in indices:
        out[index] = NAN
    return out


def shift(values, lag):
    """values[i + lag], NaN past the end -- the truncation of ALGORITHMS 5.6. Returns list."""
    n = len(values)
    return [values[i + lag] if 0 <= i + lag < n else NAN for i in range(n)]


def double_rotation(u, v, w):
    """The double rotation of ALGORITHMS 4, written out here as an independent check.

    Returns (u2, v2, w2, theta, phi); the angles are in radians. Every output is NaN
    wherever any input it mixes is NaN, which is why the covariances above are taken on the
    joint mask.
    """
    theta = math.atan2(nanmean(v), nanmean(u))
    ct, st = math.cos(theta), math.sin(theta)
    u1 = [a * ct + b * st for a, b in zip(u, v)]
    v1 = [-a * st + b * ct for a, b in zip(u, v)]
    phi = math.atan2(nanmean(w), nanmean(u1))
    cp, sp = math.cos(phi), math.sin(phi)
    u2 = [a * cp + b * sp for a, b in zip(u1, w)]
    w2 = [-a * sp + b * cp for a, b in zip(u1, w)]
    return u2, v1, w2, theta, phi


def nanmean(values):
    """The mean of the finite samples. Returns float, NaN when there are none."""
    total, count = 0.0, 0
    for value in values:
        if math.isfinite(value):
            total += value
            count += 1
    return total / count if count else NAN


def cov(x, y):
    """Two-pass covariance over the samples finite in both series, ddof = 1. Returns float."""
    pairs = [(a, b) for a, b in zip(x, y) if math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 2:
        return NAN
    mx = sum(a for a, _b in pairs) / len(pairs)
    my = sum(b for _a, b in pairs) / len(pairs)
    return sum((a - mx) * (b - my) for a, b in pairs) / (len(pairs) - 1)


if __name__ == '__main__':
    sys.exit(main())
