"""Measure what miniflux costs, with and without the numpy fast path.

    python scripts/benchmark.py

It generates its own 20 Hz data, sweeps the period length on both paths, breaks one period
down by stage, times the real TOA5 file end to end through the command line, then prints
docs/BENCHMARK.md and writes it. About ten minutes, much of it spent deliberately idle.

Three rules about the measurement, each learned by getting it wrong first: every
configuration runs in a fresh subprocess, best of REPEATS with the median beside it, since a
loop measures its own warmed caches as much as the code; the per-stage breakdown is the
single repetition with the best total, since a per-stage minimum is a run that never
happened; and the two paths are interleaved and every run idles first, since the laptop this
was written on ran the same float loop three times slower once warm, which is more than
numpy buys.

Data comes from examples/make_sample.py and the configurations from the shipped
examples/*.ini with a few keys substituted, so neither drifts from what the repository
ships. Standard library only, like the program it measures. Most of this file is the report
it writes: every sentence there quoting a number is formatted from the measurement, so
re-running replaces the conclusions and not only the tables.

Internal: ``python scripts/benchmark.py --worker SPEC.json`` is one measurement, one process.
"""

import configparser
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from time import perf_counter

#: This file is ``<root>/scripts/benchmark.py``; the root goes on ``sys.path`` so the
#: benchmark runs from any working directory without an install.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

#: One file of the FR-Jus sample, in the sibling ONEFlux_preproc checkout. Absent is not an
#: error: that section of the report is skipped and says so.
REAL_FILE = ('D:/_gitRepo/ONEFlux_preproc/data/sample/FR-Jus_20250908/FLUX/'
             '20213_RAW_20Hz_2025-09-08-2000.csv')
#: ``(averaging_minutes, samples)`` at 20 Hz. The report quotes the middle row throughout:
#: one 30-minute period is the unit a flux tower is run in.
SWEEP = ((10, 12000), (30, 36000), (60, 72000))
DEFAULT_MINUTES = 30
REPEATS = 5
#: Longest idle before a timed run, so each starts at full clock instead of measuring the
#: power limit the run before it left behind.
COOLDOWN_S = 15.0
PERIODS_PER_DAY = 48
DAYS_PER_MONTH = 30
DAYS_PER_YEAR = 365
#: How much slower one Raspberry-Pi-class core is taken to be than the core this runs on. An
#: ASSUMPTION, not a measurement, and labelled as one where the report uses it.
SBC_FACTOR = 4.0
REPORT = os.path.join(REPO, 'docs', 'BENCHMARK.md')
#: A worker inherits stderr, where the package log goes, so the parent finds its answer in
#: stdout by this prefix.
RESULT = '__RESULT__'


# --------------------------------------------------------------- the worker (one process)

def time_stages(config_path, pure):
    """Time one run, split into read+parse, the pipeline steps and write. Returns a dict.

    ``pipeline.run`` is unrolled by hand because that is the only way to see the stages
    apart: the generator is advanced with an explicit ``next``, so the time spent reading a
    period is the time spent inside that call. What is left out -- its logging-filter
    bookkeeping and its per-step ``try`` -- is far below the resolution reported here.
    """
    import logging
    logging.getLogger('miniflux').addHandler(logging.NullHandler())
    logging.getLogger('miniflux').setLevel(logging.ERROR)   # identical on both paths
    from miniflux import config, kernels, pipeline, read, write

    cfg = config.load(config_path)
    if pure:
        # The three assignments cli._force_pure makes, without argparse in the way.
        cfg.runtime.use_numpy = 'off'
        cfg.runtime.numpy_enabled = False
        kernels.set_numpy(False)

    steps = dict((name, 0.0) for name, _fn in pipeline.STEPS)
    read_s = write_s = 0.0
    periods = samples = 0
    t_start = perf_counter()
    with write.Writer(cfg.files.output_csv, cfg) as writer:
        stream = read.periods(cfg, {'periods': 0, 'skipped_short': 0, 'rows': 0,
                                    'warnings': 0})
        while True:
            t0 = perf_counter()
            try:
                period = next(stream)
            except StopIteration:
                read_s += perf_counter() - t0    # the last period is cut inside this call
                break
            read_s += perf_counter() - t0
            for name, step in pipeline.STEPS:
                t1 = perf_counter()
                period = step(period, cfg)
                steps[name] += perf_counter() - t1
            t2 = perf_counter()
            writer.write(period)
            write_s += perf_counter() - t2
            periods += 1
            samples += period['meta'].get('n_in', 0)
    total_s = perf_counter() - t_start

    # The two configured numbers the prose quotes back travel with the timings, so the
    # sentences read the same configuration the run did.
    span = max(hi - lo for _nominal, lo, hi in cfg.lag.windows.values())
    return {'total_s': total_s, 'read_s': read_s, 'write_s': write_s,
            'compute_s': total_s - read_s - write_s, 'periods': periods, 'samples': samples,
            'steps': [[name, steps[name]] for name, _fn in pipeline.STEPS],
            'shifts': int(round(span * cfg.period.acquisition_frequency)) + 1,
            'columns': len([n for n in read.SERIES if getattr(cfg.variables, n, None)])}


# ----------------------------------------------------------- the parent (repetitions)

def run_jobs(jobs):
    """Run each ``(key, command)`` in fresh subprocesses, interleaved.

    Returns ``{key: [(wall_clock, result_or_None)]}``: the wall clock always, which is the
    only way to see interpreter startup, plus the worker's own dict when it printed one.
    """
    out = dict((key, []) for key, _command in jobs)
    last_s = 0.0
    for i in range(REPEATS):
        for key, command in jobs:
            rest = min(COOLDOWN_S, max(0.5, 2.0 * last_s))
            sys.stderr.write('  %s  rep %d/%d, cooling %4.1f s \r'
                             % (key.ljust(24), i + 1, REPEATS, rest))
            sys.stderr.flush()
            time.sleep(rest)
            t0 = perf_counter()
            done = subprocess.run(command, cwd=REPO,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            last_s = perf_counter() - t0
            if done.returncode != 0:
                raise SystemExit('%s exited %d:\n%s' % (key, done.returncode,
                                                        done.stderr.decode('utf-8',
                                                                           'replace')))
            result = None
            for line in done.stdout.decode('utf-8', 'replace').splitlines():
                if line.startswith(RESULT):
                    result = json.loads(line[len(RESULT):])
            out[key].append((last_s, result))
    sys.stderr.write('  %s  %d reps done              \n'
                     % (', '.join(key for key, _c in jobs)[:24].ljust(24), REPEATS))
    return out


def measure(workdir, name, config_path):
    """Both paths on one config, interleaved. Returns {path: best run, path+'_median': s}."""
    jobs = []
    for key, pure in (('numpy', False), ('pure', True)):
        spec = os.path.join(workdir, 'spec_%s_%s.json' % (name.replace(' ', '_'), key))
        with open(spec, 'w', encoding='utf-8') as handle:
            json.dump({'config': config_path, 'pure': pure}, handle)
        jobs.append(('%s %s' % (name, key),
                     [sys.executable, os.path.abspath(__file__), '--worker', spec]))
    runs = run_jobs(jobs)
    row = {}
    for key in ('numpy', 'pure'):
        reps = runs['%s %s' % (name, key)]
        row[key] = min((run for _wall, run in reps), key=lambda run: run['total_s'])
        row[key + '_median'] = statistics.median(run['total_s'] for _wall, run in reps)
    row['samples'] = row['numpy']['samples']
    return row


# ------------------------------------------------------------- the data and the configs

def generate(workdir):
    """Write one synthetic raw file per swept sample count. Returns {samples: directory}.

    The example is driven, never copied, so its turbulence, planted lag and planted spikes
    stay as it generates them. The start instant is one sample past midnight, so N samples
    fill exactly one period of any swept length.
    """
    sys.path.insert(0, os.path.join(REPO, 'examples'))
    import make_sample

    planted = dict(make_sample.SPIKES)
    out = {}
    for _minutes, samples in SWEEP:
        folder = os.path.join(workdir, 'n%d' % samples)
        os.makedirs(folder, exist_ok=True)
        make_sample.N = samples
        make_sample.START = datetime(2025, 9, 8, 0, 0, 0, 50000)
        # A planted spike past the end of a shorter series would raise; the ones that fit
        # keep the physics the example gives them.
        make_sample.SPIKES = dict((name, tuple(i for i in indices if i < samples))
                                  for name, indices in planted.items())
        make_sample.write_file(os.path.join(folder, 'sample_20hz.dat'), make_sample.build())
        out[samples] = folder
        sys.stderr.write('  generated %6d samples in %s\n' % (samples, folder))
    return out


def write_config(example, path, changes):
    """``examples/<example>`` with ``{(section, key): value}`` substituted. Returns path.

    Read exactly as ``config.load`` reads it, so a key that has moved or been renamed in the
    example fails here instead of leaving a substitution silently unapplied. ``log_level``
    is forced: formatting a log is not part of what is measured.
    """
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#', ';'))
    with open(os.path.join(REPO, 'examples', example), 'r', encoding='utf-8') as handle:
        parser.read_file(handle)
    changes = dict(changes)
    changes[('runtime', 'log_level')] = 'ERROR'
    for (section, key), value in changes.items():
        if not parser.has_option(section, key):
            raise SystemExit('%s has no [%s] %s' % (example, section, key))
        parser.set(section, key, value)
    with open(path, 'w', encoding='utf-8') as handle:
        parser.write(handle)
    return path


def synthetic_config(workdir, folder, minutes, samples):
    """Write the config for one synthetic size. Returns its path."""
    return write_config('miniflux.ini', os.path.join(workdir, 'n%d.ini' % samples), {
        ('files', 'input_glob'): folder.replace('\\', '/') + '/*.dat',
        ('files', 'output_csv'): _slash(workdir, 'out_n%d.csv' % samples),
        ('period', 'averaging_minutes'): str(minutes),
        # The example ships latitude 0.0, which zeroes the Coriolis parameter and makes the
        # three ITC deviations NaN -- and a step returning NaN early is not being timed.
        ('site', 'latitude'): '45.0'})


def real_config(workdir):
    """The shipped FR-Jus config pointed at the one real file. Returns its path, or None.

    Its IRGASON column mapping, TOA5 header lines and open-path units are not repeated here,
    which would be a second place to get them wrong. Being an open path, both gases arrive
    as molar densities and WPL runs, which the synthetic file does not exercise.
    """
    if not os.path.exists(REAL_FILE):
        return None
    return write_config('fr_jus_openpath.ini', os.path.join(workdir, 'real.ini'), {
        ('files', 'input_glob'): REAL_FILE,
        ('files', 'output_csv'): _slash(workdir, 'out_real.csv')})


def _slash(folder, name):
    """A path inside `folder` written the way an ini file wants it. Returns str."""
    return os.path.join(folder, name).replace('\\', '/')


def environment():
    """What this ran on, read at runtime. Returns a dict of str."""
    try:
        import numpy
        version = numpy.__version__
    except ImportError:
        version = 'not installed'
    return {'when': datetime.now().strftime('%Y-%m-%d %H:%M'), 'numpy': version,
            'platform': platform.platform(), 'cores': os.cpu_count(), 'cpu': _cpu_name(),
            'python': '%s %s' % (platform.python_implementation(), platform.python_version())}


def _cpu_name():
    """The processor's own name for itself. Returns str.

    Each OS is asked where it keeps the string, because every number in the report is read
    against the chip that made it and ``platform.processor()`` answers with a model number
    on Windows and with nothing at all on some Linuxes.
    """
    try:
        if platform.system() == 'Windows':
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                return winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        if platform.system() == 'Linux':
            with open('/proc/cpuinfo', 'r', encoding='utf-8') as handle:
                for line in handle:
                    if line.startswith('model name'):
                        return line.split(':', 1)[1].strip()
    except Exception:               # a benchmark never fails over a cosmetic string
        pass
    return platform.processor() or platform.machine() or 'unknown'


# ------------------------------------------------------------------------- the report

#: Why each expensive stage costs what it does and what can be done about it, keyed by its
#: name in the table. ``%(...)s`` is filled from the measurement, so a claim about a number
#: cannot drift from the number. A stage with no entry is summarised in one line.
NOTES = {
    'read + parse': """`csv.reader` splits every row, `float()` converts the %(columns)d
declared columns and applies each one's affine unit map, and the timestamp now costs two dict
lookups and an integer addition instead of a `datetime` per sample. All C or close to it, and
the same code on both paths -- the %(ratio)s between them is this machine's noise and nothing
else, which is what the spread quoted above is for. The floor is what it costs to turn
%(tokens)d thousand ASCII numbers into doubles. Under it, in
increasing order of what they cost to keep honest: read fewer columns (every field
`csv.reader` splits and miniflux never names is paid for at full price); read a binary logger
format; or give the numpy path a bulk parser of its own -- the fastest, and the one to
refuse, since `read.py` is deliberately the single place a row is interpreted.""",

    'thermodynamics': """A per-sample scalar loop in `flux.py` that reaches a kernel exactly
twice, in the two gas means at the end, so both paths run the same code through all %(n)d
samples of it and %(ratio)s is all numpy has to give. Per sample: about fifty floating-point
operations, a dozen `_divide` calls and a 21-entry tuple folded into two dicts --
%(divides)d thousand Python-level calls and %(folds).1f million dict lookups a
period, on top of arithmetic that is genuinely owed. Two cuts that change no published
number: unroll the nan-mean fold into named accumulators, losing the tuple and the lookups;
and inline `_divide`'s zero test where the denominator is a pressure or a temperature.
Vectorising the pass would be worth more and would mean a second implementation of
ALGORITHMS section 8 -- the duplication this program exists not to have.""",

    'lag': """The one stage whose work is samples x lags rather than samples: the configured
window is %(shifts)d shifts, each a fresh covariance over the whole overlap, for each of two
scalars -- %(pairs).1f million sample pairs a period. Three cuts, cheapest first. (1) Narrow
the window in the configuration: +-0.5 s is 21 shifts instead of %(shifts)d, a %(narrow).1fx
cut, and for a collocated open-path analyser it is the more defensible window anyway. (2)
Hoist the finite masks out of the scan -- `kernels.cov_at_lag` rebuilds `isfinite(w) &
isfinite(c)` at every lag although only the overlap window moves, so each mask could be built
once per scalar and sliced: two of the nine array passes each lag makes on the numpy path,
two `math.isfinite` calls per iteration on the pure one, result bit-identical. (3) A
coarse-to-fine scan would cut most of the rest and is not an optimisation -- it changes which
lag wins on a flat covariance curve, which is ALGORITHMS 5.4's business. An FFT
cross-correlation needs the numeric stack the pure path exists to avoid.""",

    'moments': """Ten reductions -- four variances, six covariances -- each two passes over
%(n)d samples, plus three joint-finite counts. The counts are the interesting part:
`_joint_finite` in `flux.py` is a Python loop on **both** paths, %(joint)d thousand
`math.isfinite` calls a period, and it is why this stage speeds up %(ratio)s where the stages
that live wholly inside kernels get ten to sixty. `kernels.cov` computes that count on its
own first pass and throws it
away; returning it, or giving `kernels` a joint-finite count with a numpy branch, is the
cheapest real win on this page.""",

    'despike': """Every configured series sorted once for the median and again for the MAD of
the absolute deviations, then one pass for the mask: the %(ratio)s is `np.sort` against
`sorted()` on a list. One cut is left and it is numpy-only -- `kernels.nanmedian` and
`kernels.mad` sort a whole series to read one order statistic, where `np.partition` gives the
same value in O(n) and still hands `_median_of_sorted` the two middle elements it needs. In
pure Python the sort is the fastest thing the standard library has.""",
}

TEMPLATE = """# What miniflux costs, stage by stage

Measured by `scripts/benchmark.py`, which generates its own data and writes this file; re-run
it and every number here is replaced. Each configuration runs in a fresh subprocess
%(repeats)d times, interleaved with the other path and idling first, so a session that slows
as it proceeds slows both and not whichever ran last. The tables quote the best repetition
with the median beside it; the per-stage breakdown is the single repetition with the best
total, so its parts add up to it.

| the machine | |
|---|---|
| measured | %(when)s |
| cpu | %(cpu)s, %(cores)s logical cores, one used |
| platform | %(platform)s |
| python | %(python)s |
| numpy | %(numpy)s |

## Where the time goes

One 30-minute period -- %(n)d samples at %(freq).0f Hz, the unit a flux tower is run in --
read and parsed, pushed through the %(n_steps)d steps of `pipeline.STEPS`, and written.
Sorted by cost on the numpy path; the pure path's own order is in its share column. Shares
are of that path's whole run. A stage under a millisecond is at the resolution of this
measurement: read it as free, not as three significant figures.

%(stage_table)s

**Two stages are the whole numpy bill: %(top_two)s, %(top_share).0f%% of the run between
them**, and neither is a kernel -- reading never was one, and `thermodynamics` is a scalar
loop numpy cannot reach. **On the pure path `lag` alone is %(lag_share).0f%%**, being the one
stage whose work grows with the number of lags as well as with the number of samples. The two
profiles have little in common, which is the useful thing here: optimising for one path is
not optimising for the other.

Read + parse is the same work on both paths at every length, so it doubles as this page's
noise floor: over the six runs of the next table it measures between %(noise_lo).1f and
%(noise_hi).1f us a sample, a spread of %(noise_pct).0f%% on an idle machine. Read no ratio
above as a result unless it is further from 1 than that.

%(notes)s

**The rest** -- %(rest)s -- costs %(rest_numpy)s together on the numpy path and %(rest_pure)s
on the pure one. Nothing there is worth an hour of anyone's time.

## One period, end to end

The same run at three period lengths; the lag window is fixed in seconds, so it is
%(shifts)d shifts at every length and the work per sample does not change with it.

%(sweep_table)s

Both paths are linear enough in the period length to quote one throughput and ragged enough
that two significant figures would be generous: **%(numpy_hi)s to %(numpy_lo)s with numpy,
%(pure_hi)s to %(pure_lo)s without**, across the three lengths. The 30-minute row is the one
the rest of this page reads: **%(numpy_period)s** a period and **%(pure_period)s**.

%(real_section)s## What numpy buys, and what a year costs

**numpy buys %(speedup).1fx overall on the 30-minute period**, %(step_speedup).0fx across the
steps alone. It is not more because %(unreachable_share).0f%% of the numpy run is already
outside every kernel: read + parse is %(read_share).0f%% of it and `thermodynamics` another
%(thermo_share).0f%%, and neither calls a kernel that could be made faster. Amdahl, measured
rather than quoted: an infinitely fast kernel layer would still leave %(unreachable)s of this
period standing, so **%(ceiling).1fx is the whole remaining ceiling** against the
%(speedup).1fx already collected -- and it is not in the kernels at all. It is the parser and
one scalar loop, both named above.

Continuous 20 Hz data, %(per_day)d periods a day, one core, in process:

%(cost_table)s

A tower-year is %(numpy_year)s of cpu with numpy and %(pure_year)s without%(startup_note)s,
against the %(days)d days of data it covers. Neither is a reason to choose a path; the reason
to choose is that one of them needs numpy.

**On a single-board computer.** Not measured here. Taking one Raspberry-Pi-class core as
**%(sbc).0fx slower than the core above, which is an ESTIMATE and not a measurement**, a
30-minute period would cost about %(sbc_numpy)s with numpy and %(sbc_pure)s without against
the %(period_s)d s of data it represents -- %(sbc_numpy_pct).1f%% and %(sbc_pure_pct).1f%% of
real time on one core. Such a board keeps up with a live 20 Hz tower on either path and
reprocesses a year of archive in about %(sbc_numpy_year)s or %(sbc_pure_year)s. That factor
is the only guess in this document: measure it on the board before relying on it.

miniflux targets Python 3.8 and this is %(python)s. A 3.8 interpreter has neither the
specialising interpreter nor the faster call sequence, so the pure-Python numbers above are
the optimistic end of what 3.8 would show; the numpy numbers, which spend their time inside
numpy, would barely move.
"""

REAL_SECTION = """## A real file, through the command line

`%(real_name)s` -- TOA5, %(real_samples)d rows cut into %(real_periods)d periods, an
open-path IRGASON whose gases arrive as molar densities, so WPL runs, which the synthetic
file does not exercise. Timed as `python -m miniflux run CONFIG`, the whole process, because
interpreter startup and imports are part of what a user waits for, and again in process, so
the stages separate.

%(real_table)s

Startup -- `python -m miniflux --version`, which does every import a run does and then stops
-- is %(startup)s, %(startup_share).0f%% of the numpy command line, spent before the first
byte is read. In process the real file runs at %(real_numpy_rate)s and %(real_pure_rate)s
against %(numpy_rate)s and %(pure_rate)s on the synthetic one, so the synthetic file is a
fair stand-in: the only claim made here, since the two differ in several ways at once (unread
columns `csv.reader` still splits, gaps, two unequal periods).

"""


def render(env, sweep, real):
    """Build the whole of docs/BENCHMARK.md from the measurements. Returns str."""
    default = sweep[DEFAULT_MINUTES]
    fast, slow = default['numpy'], default['pure']
    stages = _stages(fast, slow)
    cost = dict((name, (one, other)) for name, one, other in stages)
    rest = [name for name, _o, _t in stages if name not in NOTES]
    n, shifts = default['samples'], fast['shifts']
    left = fast['read_s'] + cost['thermodynamics'][0]       # what no kernel can be blamed for
    # The paragraphs below name these three stages by hand -- which two lead the numpy path
    # (in either order; the sentence reads them from the measurement) and which leads the
    # pure one. A run that promotes a fourth has changed the conclusion and not only the
    # numbers, so it stops rather than print the sentence that used to be true.
    if (set(row[0] for row in stages[:2]) != set(['read + parse', 'thermodynamics'])
            or max(stages, key=lambda row: row[2])[0] != 'lag'):
        raise SystemExit('the ranking has changed -- numpy now leads with %s, pure with %s. '
                         'Rewrite the paragraphs that name them.'
                         % (' and '.join(row[0] for row in stages[:2]),
                            max(stages, key=lambda row: row[2])[0]))
    # Read + parse is the same work on both paths at every length, so the spread of its
    # per-sample cost over the whole sweep is this machine's noise, measured on the code
    # rather than on a synthetic loop.
    read_us = [row[key]['read_s'] / row['samples'] * 1e6
               for _m, row in ((m, sweep[m]) for m, _n in SWEEP) for key in ('numpy', 'pure')]
    rates = dict((key, [sweep[m][key]['samples'] / sweep[m][key]['total_s'] for m, _n in SWEEP])
                 for key in ('numpy', 'pure'))

    v = dict(env, repeats=REPEATS, n=n, shifts=shifts, sbc=SBC_FACTOR, days=DAYS_PER_YEAR,
             per_day=PERIODS_PER_DAY, period_s=DEFAULT_MINUTES * 60,
             freq=n / float(DEFAULT_MINUTES * 60), n_steps=len(fast['steps']),
             columns=fast['columns'], tokens=n * fast['columns'] / 1000.0,
             divides=12.0 * n / 1000.0, folds=84.0 * n / 1e6, joint=6.0 * n / 1000.0,
             pairs=2.0 * shifts * n / 1e6, narrow=shifts / 21.0,
             top_two='%s and %s' % (stages[0][0], stages[1][0]),
             top_share=100.0 * (stages[0][1] + stages[1][1]) / fast['total_s'],
             lag_share=100.0 * cost['lag'][1] / slow['total_s'],
             rest=', '.join('`%s`' % name for name in rest),
             rest_numpy=_s(sum(cost[name][0] for name in rest)),
             rest_pure=_s(sum(cost[name][1] for name in rest)),
             numpy_rate=_rate(n, fast['total_s']), pure_rate=_rate(n, slow['total_s']),
             numpy_period=_s(fast['total_s']), pure_period=_s(slow['total_s']),
             noise_lo=min(read_us), noise_hi=max(read_us),
             noise_pct=100.0 * (max(read_us) / min(read_us) - 1.0),
             numpy_lo=_rate(n, n / min(rates['numpy'])),
             numpy_hi=_rate(n, n / max(rates['numpy'])),
             pure_lo=_rate(n, n / min(rates['pure'])),
             pure_hi=_rate(n, n / max(rates['pure'])),
             speedup=slow['total_s'] / fast['total_s'], unreachable=_s(left),
             step_speedup=slow['compute_s'] / fast['compute_s'],
             ceiling=slow['total_s'] / left,
             unreachable_share=100.0 * left / fast['total_s'],
             read_share=100.0 * fast['read_s'] / fast['total_s'],
             thermo_share=100.0 * cost['thermodynamics'][0] / fast['total_s'],
             numpy_year=_long(fast['total_s'] * PERIODS_PER_DAY * DAYS_PER_YEAR),
             pure_year=_long(slow['total_s'] * PERIODS_PER_DAY * DAYS_PER_YEAR),
             sbc_numpy=_s(fast['total_s'] * SBC_FACTOR),
             sbc_pure=_s(slow['total_s'] * SBC_FACTOR),
             sbc_numpy_pct=100.0 * fast['total_s'] * SBC_FACTOR / (DEFAULT_MINUTES * 60),
             sbc_pure_pct=100.0 * slow['total_s'] * SBC_FACTOR / (DEFAULT_MINUTES * 60),
             sbc_numpy_year=_long(fast['total_s'] * SBC_FACTOR * PERIODS_PER_DAY
                                  * DAYS_PER_YEAR),
             sbc_pure_year=_long(slow['total_s'] * SBC_FACTOR * PERIODS_PER_DAY
                                 * DAYS_PER_YEAR),
             startup_note='', real_section='')
    v['stage_table'] = _table(
        'stage | numpy | share | pure | share | pure/numpy', 6,
        [(name, _s(one), _pct(one, fast['total_s']), _s(other), _pct(other, slow['total_s']),
          '%.1fx' % (other / one)) for name, one, other in stages])
    v['sweep_table'] = _table(
        'period | samples | path | best | median | read + parse | steps | write | samples/s', 9,
        [('%d min' % minutes, row['samples'], key, _s(row[key]['total_s']),
          _s(row[key + '_median']), _s(row[key]['read_s']), _s(row[key]['compute_s']),
          _s(row[key]['write_s']), _rate(row['samples'], row[key]['total_s']))
         for minutes, row in ((m, sweep[m]) for m, _n in SWEEP) for key in ('numpy', 'pure')])
    v['cost_table'] = _table(
        'path | one period | one day | one month | one year', 5,
        [(key, _s(run['total_s']), _long(run['total_s'] * PERIODS_PER_DAY),
          _long(run['total_s'] * PERIODS_PER_DAY * DAYS_PER_MONTH),
          _long(run['total_s'] * PERIODS_PER_DAY * DAYS_PER_YEAR))
         for key, run in (('numpy', fast), ('pure', slow))])
    v['notes'] = '\n\n'.join(
        '**%s.** %s' % (name, ' '.join(NOTES[name].split())
                        % dict(v, ratio='%.1fx' % (other / one)))
        for name, one, other in stages if name in NOTES)
    if real:
        v.update(real_name=os.path.basename(REAL_FILE), real_samples=real['samples'],
                 real_periods=real['numpy']['periods'], startup=_s(real['startup_s']),
                 startup_share=100.0 * real['startup_s'] / real['numpy_cli_best'],
                 real_numpy_rate=_rate(real['samples'], real['numpy']['total_s']),
                 real_pure_rate=_rate(real['samples'], real['pure']['total_s']),
                 startup_note=', plus %s of startup per command line' % _s(real['startup_s']))
        v['real_table'] = _table(
            'path | command line, best | median | in process | read + parse | steps', 6,
            [(key, _s(real[key + '_cli_best']), _s(real[key + '_cli_median']),
              _s(real[key]['total_s']), _s(real[key]['read_s']), _s(real[key]['compute_s']))
             for key in ('numpy', 'pure')])
        v['real_section'] = REAL_SECTION % v
    return TEMPLATE % v


def _stages(fast, slow):
    """Every timed stage as ``(name, numpy_s, pure_s)``, most expensive on numpy first."""
    pure = dict(slow['steps'])
    rows = ([('read + parse', fast['read_s'], slow['read_s'])]
            + [(name, seconds, pure[name]) for name, seconds in fast['steps']]
            + [('write', fast['write_s'], slow['write_s'])])
    return sorted(rows, key=lambda row: -row[1])


def _table(header, columns, rows):
    """A markdown table: '|'-separated header, first column left, the rest right."""
    return '\n'.join(['| %s |' % header, '|---' + '|---:' * (columns - 1) + '|']
                     + ['| %s |' % ' | '.join(str(cell) for cell in row) for row in rows])


def _pct(part, whole):
    """One share of a run, as a percentage. Returns str."""
    return '%.0f%%' % (100.0 * part / whole)


def _s(seconds):
    """A duration, in the unit that shows three significant figures. Returns str."""
    if seconds < 1e-3:
        return '%.0f us' % (seconds * 1e6)
    if seconds < 1.0:
        return '%.0f ms' % (seconds * 1e3)
    return '%.2f s' % seconds


def _long(seconds):
    """A duration as a person would state it: seconds, minutes, hours or days. Returns str."""
    if seconds < 60.0:
        return '%.0f s' % seconds
    if seconds < 3600.0:
        return '%.0f min' % (seconds / 60.0)
    if seconds < 86400.0:
        return '%.1f h' % (seconds / 3600.0)
    return '%.1f days' % (seconds / 86400.0)


def _rate(samples, seconds):
    """Throughput as samples per second. Returns str."""
    rate = samples / seconds
    return '%.1fM/s' % (rate / 1e6) if rate >= 1e6 else '%.0fk/s' % (rate / 1e3)


# ---------------------------------------------------------------------------- the run

def main(argv):
    """Run the whole benchmark, or one worker task. Returns the exit code."""
    if len(argv) == 2 and argv[0] == '--worker':
        with open(argv[1], 'r', encoding='utf-8') as handle:
            spec = json.load(handle)
        sys.stdout.write(RESULT + json.dumps(time_stages(spec['config'], spec['pure'])) + '\n')
        return 0
    if argv:
        raise SystemExit(__doc__.strip())

    env = environment()
    if env['numpy'] == 'not installed':
        raise SystemExit('numpy is not installed, so there is nothing to compare against')
    workdir = tempfile.mkdtemp(prefix='miniflux-bench-')
    sys.stderr.write('working in %s\ngenerating synthetic data\n' % workdir)
    folders = generate(workdir)
    sys.stderr.write('measuring: %d fresh subprocesses per configuration, interleaved\n'
                     % REPEATS)

    sweep = {}
    for minutes, samples in SWEEP:
        sweep[minutes] = measure(workdir, '%d min' % minutes,
                                 synthetic_config(workdir, folders[samples], minutes, samples))

    real = None
    real_path = real_config(workdir)
    if real_path is None:
        sys.stderr.write('  %s not found; skipping the real-file section\n' % REAL_FILE)
    else:
        # `--version` does every import a `run` pays before it reads a byte, and startup can
        # only be seen from outside: a timer inside the process has already missed it.
        command = [sys.executable, '-m', 'miniflux', 'run', real_path, '--log-level', 'ERROR']
        times = run_jobs([('version', [sys.executable, '-m', 'miniflux', '--version']),
                          ('real numpy cli', command),
                          ('real pure cli', command + ['--pure'])])
        real = measure(workdir, 'real file', real_path)
        real['startup_s'] = min(wall for wall, _r in times['version'])
        for key in ('numpy', 'pure'):
            walls = [wall for wall, _r in times['real %s cli' % key]]
            real[key + '_cli_best'] = min(walls)
            real[key + '_cli_median'] = statistics.median(walls)

    # The measurements, before anything is said about them: ten minutes of wall clock is too
    # much to lose to a sentence that will not format.
    raw = os.path.join(workdir, 'results.json')
    with open(raw, 'w', encoding='utf-8') as handle:
        json.dump({'env': env, 'sweep': sweep, 'real': real}, handle, indent=1)
    sys.stderr.write('raw measurements in %s\n' % raw)

    report = render(env, sweep, real)
    sys.stdout.write(report)
    with open(REPORT, 'w', encoding='utf-8') as handle:
        handle.write(report)
    sys.stderr.write('wrote %s\n' % REPORT)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
