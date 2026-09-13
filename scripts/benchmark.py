"""Measure what miniflux costs, with and without the numpy fast path.

    python scripts/benchmark.py

It generates its own 20 Hz data, sweeps the period length on both paths, breaks one period down
by pipeline step, times the real TOA5 file end to end through the command line, then prints a
markdown report and writes it to ``docs/BENCHMARK.md``.

Four rules about the measurement, each learned by getting it wrong first: every configuration
runs in a **fresh subprocess**, best of ``--repeats`` with the median beside it, since a loop
measures its own warmed caches as much as the code; the per-step breakdown is the **single
repetition with the best total**, since a per-step minimum is a run that never happened; the
two paths are **interleaved** and every run idles first, since the laptop this was written on
ran the same float loop three times slower once warm, which is more than numpy buys; and every
run **probes the cpu clock before and after itself**, so a run throttled while it ran is marked
``(!)`` as an upper bound instead of passing as a measurement.

Data comes from ``examples/make_sample.py`` driven at a different sample count, configurations
from the shipped ``examples/*.ini`` with a few keys substituted, so neither drifts away from
what the repository ships. Standard library only, like the program it measures. About ten
minutes, most of it spent deliberately idle.

Internal: ``python scripts/benchmark.py --worker SPEC.json`` is one measurement, one process.
"""

import argparse
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

#: This file is ``<root>/scripts/benchmark.py``; the root goes on ``sys.path`` so the benchmark
#: runs from any working directory without an install.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

#: One file of the FR-Jus sample, in the sibling ONEFlux_preproc checkout. Absent is not an
#: error: that section of the report is skipped and says so.
REAL_FILE = ('D:/_gitRepo/ONEFlux_preproc/data/sample/FR-Jus_20250908/FLUX/'
             '20213_RAW_20Hz_2025-09-08-2000.csv')

#: ``(averaging_minutes, samples)`` at 20 Hz. The prose quotes the middle row: one 30-minute
#: period is the unit a flux tower is run in.
SWEEP = ((10, 12000), (30, 36000), (60, 72000))
DEFAULT_MINUTES = 30
PERIODS_PER_DAY = 48
DAYS_PER_YEAR = 365
REPORT = os.path.join(REPO, 'docs', 'BENCHMARK.md')

#: A worker inherits stderr, where the package log goes, so the parent picks its answer out of
#: stdout by this prefix.
RESULT = '__RESULT__'

#: How long one clock probe should take, and the ratio above which a run is reported as having
#: been throttled while it ran.
PROBE_S = 0.02
DRIFT_LIMIT = 1.25
_PROBE_COUNT = None


# ------------------------------------------------------------------ the worker (one process)

def worker(spec_path):
    """Time one full run in this process and print the result as JSON. Returns 0."""
    with open(spec_path, 'r', encoding='utf-8') as handle:
        spec = json.load(handle)
    sys.stdout.write(RESULT + json.dumps(time_stages(spec['config'], spec['pure'])) + '\n')
    return 0


def time_stages(config_path, pure):
    """Time one run, split into read+parse, the nine steps and write. Returns a dict.

    ``pipeline.run`` is unrolled by hand because that is the only way to see the stages apart:
    the generator is advanced with an explicit ``next``, so the time spent reading a period is
    the time spent inside that call. What is left out -- its logging-filter bookkeeping and its
    per-step ``try`` -- is a per-period constant far below the resolution reported here.
    """
    probe_before = _probe()
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
        stream = read.periods(cfg, {'periods': 0, 'skipped_short': 0, 'rows': 0, 'warnings': 0})
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

    return {'total_s': total_s, 'read_s': read_s, 'write_s': write_s,
            'compute_s': total_s - read_s - write_s, 'periods': periods, 'samples': samples,
            'steps': [[name, steps[name]] for name, _fn in pipeline.STEPS],
            'drift': _probe() / probe_before}


def _burn(count):
    """A fixed float loop, the shape of the pure path's inner loops. Returns its sum."""
    import math
    total = 0.0
    for i in range(count):
        total += math.sqrt(i * 1.000001) + math.log(i + 1.0)
    return total


def _probe():
    """The cpu's current speed on that loop. Returns the best of three, in seconds.

    Best of three, because one 20 ms probe is easily half as fast again when the scheduler takes
    the core away, and a drift ratio built from two such probes would flag runs that were never
    throttled. The loop is calibrated to PROBE_S once, so this is a duration on any machine.
    """
    global _PROBE_COUNT
    if _PROBE_COUNT is None:
        _burn(20000)                            # warm the bytecode, not the silicon
        t0 = perf_counter()
        _burn(50000)
        _PROBE_COUNT = max(1000, int(PROBE_S * 50000.0 / (perf_counter() - t0)))
    times = []
    for _try in range(3):
        t0 = perf_counter()
        _burn(_PROBE_COUNT)
        times.append(perf_counter() - t0)
    return min(times)


# ------------------------------------------------------------------ the parent (repetitions)

def run_jobs(jobs, repeats, cooldown):
    """Run each ``(key, command)`` in fresh subprocesses, interleaved.

    Returns ``{key: [(wall_clock, result_or_None)]}``: the wall clock always, which is what a
    user waits for and the only way to see interpreter startup, plus the worker's own dict when
    the command printed one. Interleaved means repetition *i* of every job runs before
    repetition *i+1* of any of them, so thermal drift is common to all of them, which is what a
    ratio needs. Each run first idles twice the last run's wall clock, capped.
    """
    out = dict((key, []) for key, _command in jobs)
    last_s = 0.0
    for i in range(repeats):
        for key, command in jobs:
            rest = min(cooldown, max(0.5, 2.0 * last_s))
            sys.stderr.write('  %s  rep %d/%d, cooling %4.1f s \r'
                             % (key.ljust(28), i + 1, repeats, rest))
            sys.stderr.flush()
            time.sleep(rest)
            t0 = perf_counter()
            done = subprocess.run(command, cwd=REPO,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            last_s = perf_counter() - t0
            if done.returncode != 0:
                raise SystemExit('%s exited %d:\n%s'
                                 % (key, done.returncode,
                                    done.stderr.decode('utf-8', 'replace')))
            result = None
            for line in done.stdout.decode('utf-8', 'replace').splitlines():
                if line.startswith(RESULT):
                    result = json.loads(line[len(RESULT):])
            out[key].append((last_s, result))
    sys.stderr.write('  %s  %d reps done              \n'
                     % (', '.join(key for key, _c in jobs)[:28].ljust(28), repeats))
    return out


def worker_job(workdir, key, config_path, pure):
    """Write one worker's spec and return its ``(key, command)``."""
    spec_path = os.path.join(workdir, 'spec_%s.json' % key.replace(' ', '_'))
    with open(spec_path, 'w', encoding='utf-8') as handle:
        json.dump({'config': config_path, 'pure': pure}, handle)
    return (key, [sys.executable, os.path.abspath(__file__), '--worker', spec_path])


def best_run(reps):
    """The repetition with the smallest total, whole, so its stages add up to it. Returns it."""
    return min((result for _wall, result in reps), key=lambda run: run['total_s'])


# ------------------------------------------------------------------ the data and the configs

def generate(workdir, sizes):
    """Write one synthetic raw file per sample count. Returns {samples: directory}.

    ``examples/make_sample.py`` is driven, never copied, so the AR(1) turbulence, the planted
    lag and the planted spikes stay as the example generates them. The start instant is one
    sample past midnight, so N samples at 20 Hz fill exactly one period of any swept length.
    """
    sys.path.insert(0, os.path.join(REPO, 'examples'))
    import make_sample

    planted = dict(make_sample.SPIKES)
    out = {}
    for samples in sizes:
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


def synthetic_config(workdir, folder, minutes, samples):
    """Write the config for one synthetic size. Returns its path."""
    return _write_config('miniflux.ini', os.path.join(workdir, 'n%d.ini' % samples), {
        ('files', 'input_glob'): folder.replace('\\', '/') + '/*.dat',
        ('files', 'output_csv'): os.path.join(workdir, 'out_n%d.csv' % samples).replace('\\',
                                                                                        '/'),
        ('period', 'averaging_minutes'): str(minutes),
        # The example ships latitude 0.0, which zeroes the Coriolis parameter and makes the
        # three ITC deviations NaN -- and a step that returns NaN early is not being timed.
        ('site', 'latitude'): '45.0'})


def real_config(workdir):
    """Point the shipped FR-Jus config at the one real file. Returns its path, or None.

    The IRGASON column mapping, the TOA5 header lines and the open-path units live in
    ``examples/fr_jus_openpath.ini``; repeating them here would be a second place to get them
    wrong. Being an open path, both gases arrive as molar densities and WPL runs -- which the
    synthetic file, in mixing ratios, does not exercise.
    """
    if not os.path.exists(REAL_FILE):
        return None
    return _write_config('fr_jus_openpath.ini', os.path.join(workdir, 'real.ini'), {
        ('files', 'input_glob'): REAL_FILE,
        ('files', 'output_csv'): os.path.join(workdir, 'out_real.csv').replace('\\', '/')})


def _write_config(example, path, changes):
    """``examples/<example>`` with ``{(section, key): value}`` substituted. Returns path.

    Line based rather than configparser round-tripping, so the comments that document every key
    in the example survive into the generated file. ``log_level`` is forced on every config,
    because formatting a log is not part of what is measured.
    """
    with open(os.path.join(REPO, 'examples', example), 'r', encoding='utf-8') as handle:
        text = handle.read()
    changes = dict(changes)
    changes[('runtime', 'log_level')] = 'ERROR'
    out = []
    section = None
    seen = set()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('[') and ']' in stripped:
            section = stripped[1:stripped.index(']')]    # headers carry a trailing comment
        elif '=' in line:
            key = (section, line.split('=', 1)[0].strip())
            if key in changes:
                line = '%-15s = %s' % (key[1], changes[key])
                seen.add(key)
        out.append(line)
    if set(changes) - seen:
        raise SystemExit('%s has no %s' % (example, sorted(set(changes) - seen)))
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(out) + '\n')
    return path


# ------------------------------------------------------------------ the machine

def environment():
    """What this ran on, read at runtime. Returns a dict of str."""
    try:
        import numpy
        numpy_version = numpy.__version__
    except ImportError:
        numpy_version = None
    return {'when': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'platform': platform.platform(), 'numpy': numpy_version,
            'python': '%s %s' % (platform.python_implementation(), platform.python_version()),
            'cpu': _cpu_name(), 'cores': os.cpu_count()}


def _cpu_name():
    """The processor's own name for itself, per OS. Returns str.

    ``platform.processor()`` answers with a family and model number on Windows and with nothing
    at all on some Linuxes, so each OS is asked where it keeps the string.
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
        if platform.system() == 'Darwin':
            done = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                  stdout=subprocess.PIPE)
            return done.stdout.decode('utf-8', 'replace').strip()
    except Exception:               # a benchmark never fails over a cosmetic string
        pass
    return platform.processor() or platform.machine() or 'unknown'


# ------------------------------------------------------------------ the report

def render(env, sweep, real, repeats):
    """Build the whole of docs/BENCHMARK.md. Returns str.

    Every number in the prose is computed from the measurements passed in, so re-running on
    another machine rewrites the conclusions with it instead of leaving last month's numbers
    inside the sentences.
    """
    lines = []
    add = lines.append
    default = sweep[DEFAULT_MINUTES]
    fast, slow = default['numpy'], default['pure']
    steps = {'numpy': dict(fast['steps']), 'pure': dict(slow['steps'])}
    order = [name for name, _seconds in fast['steps']]
    timed = [('%d min %s' % (minutes, key), sweep[minutes][key])
             for minutes, _n in SWEEP for key in ('numpy', 'pure')]
    if real:
        timed += [('real file %s' % key, real[key + '_shape']) for key in ('numpy', 'pure')]
    drifted = ['%s %.1fx' % (name, run['drift']) for name, run in timed if _mark(run)]

    add('# What miniflux costs')
    add('')
    add('Measured by `scripts/benchmark.py`, which generates its own data and writes this file.'
        ' Re-run it and every number here is replaced.')
    add('')
    add('    python scripts/benchmark.py')
    add('')
    add('Each configuration runs in a **fresh subprocess**, %d times; the tables quote the best '
        'repetition with the median beside it, and the per-step breakdown is the single '
        'repetition with the best total, so its parts add up to it. The two paths are '
        '**interleaved**, so a session that slows as it proceeds slows both and not whichever '
        'ran last. Every run idles first and probes the cpu clock around itself: %s'
        % (repeats,
           'nothing here slowed by more than %d%% while it ran.' % (100 * DRIFT_LIMIT - 100)
           if not drifted else
           '%d run(s) slowed while running (%s) and are marked **(!)** below -- upper bounds on '
           'the time, not measurements of the code.' % (len(drifted), ', '.join(drifted))))
    add('')
    add('| the machine | |')
    add('|---|---|')
    add('| measured | %s |' % env['when'])
    add('| cpu | %s (%s logical cores, one used) |' % (env['cpu'], env['cores']))
    add('| platform | %s |' % env['platform'])
    add('| python | %s |' % env['python'])
    add('| numpy | %s |' % (env['numpy'] or 'not installed'))
    add('')
    add('miniflux targets Python 3.8 and this is %s. A 3.8 interpreter has neither the '
        'specialising interpreter nor the faster call sequence, so the pure-Python numbers below'
        ' are the optimistic end of what 3.8 would show; the numpy numbers, which spend their '
        'time inside numpy, would barely move.' % env['python'])
    add('')

    add('## One period, end to end')
    add('')
    add('20 Hz synthetic data, one period per run, read from disk, pushed through all nine steps'
        ' and written out. The lag window is fixed at +-2 s (81 shifts) at every length, so the'
        ' work per sample does not change with the period.')
    add('')
    add('| period | samples | path | best | median | read + parse | nine steps | write | '
        'samples/s |')
    add('|---|---:|---|---:|---:|---:|---:|---:|---:|')
    for minutes, _samples in SWEEP:
        row = sweep[minutes]
        for key in ('numpy', 'pure'):
            run = row[key]
            add('| %d min | %d | %s | %s | %s | %s | %s | %s | %s |'
                % (minutes, row['samples'], key, _s(run['total_s']) + _mark(run),
                   _s(row[key + '_median']), _s(run['read_s']), _s(run['compute_s']),
                   _s(run['write_s']), _rate(row['samples'], run['total_s'])))
    add('')
    add('On the 30-minute period -- %d samples, the unit a flux tower is run in -- **numpy buys '
        '%.1fx overall and all of it in the compute stages (%.0fx there)**. Reading is the same '
        'work either way, none of it through a kernel, so the %.0f%% between the two read '
        'columns is run-to-run noise; on the numpy path reading is %.0f%% of the run. Quote the '
        'throughput as %s (numpy) and %s (pure), no better than two significant figures.'
        % (default['samples'], slow['total_s'] / fast['total_s'],
           slow['compute_s'] / fast['compute_s'],
           abs(100.0 * slow['read_s'] / fast['read_s'] - 100.0),
           100.0 * fast['read_s'] / fast['total_s'],
           _rate(default['samples'], fast['total_s']),
           _rate(default['samples'], slow['total_s'])))
    add('')

    add('## Per step')
    add('')
    add('The nine steps of `pipeline.STEPS` on that 30-minute period, in the order they run, '
        'with the share of each path\'s own compute time. A step under a millisecond is at the '
        'resolution of this measurement: read it as free, not as three significant figures.')
    add('')
    add('| step | numpy | share | pure | share | speedup |')
    add('|---|---:|---:|---:|---:|---:|')
    for name in order:
        one, other = steps['numpy'][name], steps['pure'][name]
        add('| %s | %s | %.0f%% | %s | %.0f%% | %.1fx |'
            % (name, _s(one), 100.0 * one / fast['compute_s'],
               _s(other), 100.0 * other / slow['compute_s'], other / one))
    add('')
    top_pure = max(order, key=lambda name: steps['pure'][name])
    top_numpy = max(order, key=lambda name: steps['numpy'][name])
    add('**Pure Python is dominated by `%s` (%.0f%% of its compute).** It is the one step whose '
        'work is samples x lags rather than samples: +-2 s at 20 Hz is 81 shifts, each a '
        'covariance over the whole period -- ~%.1f million multiply-adds per scalar, and there '
        'are two scalars. With numpy it costs %s and the profile inverts: **`%s` is then the '
        'largest (%.0f%% of compute)**, and it is the step that does not speed up -- %.1fx, '
        'which is 1 plus noise. `thermodynamics` is an explicit per-sample Python loop in '
        '`flux.py` that calls no kernel, so `--pure` and the default run the same code through '
        'it.'
        % (top_pure, 100.0 * steps['pure'][top_pure] / slow['compute_s'],
           81 * default['samples'] / 1e6, _s(steps['numpy'][top_pure]), top_numpy,
           100.0 * steps['numpy'][top_numpy] / fast['compute_s'],
           steps['pure']['thermodynamics'] / steps['numpy']['thermodynamics']))
    add('')

    if real:
        add('## A real file, through the command line')
        add('')
        add('`%s` -- TOA5, %d data rows cut into %d periods, an open-path IRGASON whose gases '
            'arrive as molar densities, so WPL runs, which the synthetic file does not exercise.'
            ' Timed as `python -m miniflux run CONFIG`, the whole process, because interpreter '
            'startup and imports are part of what a user waits for; and again in process, so '
            'the stages separate.'
            % (os.path.basename(REAL_FILE), real['samples'], real['periods']))
        add('')
        add('| path | command line, best | median | in process | read + parse | nine steps | '
            'samples/s |')
        add('|---|---:|---:|---:|---:|---:|---:|')
        for key in ('numpy', 'pure'):
            run = real[key + '_shape']
            add('| %s | %s | %s | %s | %s | %s | %s |'
                % (key, _s(real[key + '_best']), _s(real[key + '_median']),
                   _s(run['total_s']) + _mark(run), _s(run['read_s']), _s(run['compute_s']),
                   _rate(real['samples'], run['total_s'])))
        add('')
        add('Startup -- `python -m miniflux --version`, which does every import a run does and '
            'then stops -- is %s, or %.0f%% of the numpy command line above, spent before the '
            'first byte is read. `--pure` still imports numpy: it is installed, and the flag '
            'only stops it being *used*. Throughput on the real file is %s (numpy) and %s '
            '(pure) against %s and %s on the synthetic one, so the synthetic file is a fair '
            'stand-in -- the only claim here, since the two differ in several ways at once '
            '(unread columns `csv.reader` still splits, gaps, two very unequal periods).'
            % (_s(real['startup_s']), 100.0 * real['startup_s'] / real['numpy_best'],
               _rate(real['samples'], real['numpy_shape']['total_s']),
               _rate(real['samples'], real['pure_shape']['total_s']),
               _rate(default['samples'], fast['total_s']),
               _rate(default['samples'], slow['total_s'])))
        add('')

    add('## Where the time goes')
    add('')
    add('The numpy path is bound by the parser -- %.0f%% of its run, already `csv.reader` plus '
        '`float()` plus a `datetime` per row, all C -- which leaves the file format as the '
        'ceiling. The pure path is bound by the lag window (%.0f%% of its compute), and '
        'narrowing that is configuration, not code: `co2_min_s`/`co2_max_s` at +-0.5 s is 21 '
        'shifts instead of 81. An incremental or FFT lag curve would need the numeric stack '
        'that path exists to avoid.'
        % (100.0 * fast['read_s'] / fast['total_s'],
           100.0 * steps['pure']['lag'] / slow['compute_s']))
    add('')
    add('Continuous 20 Hz data at %d periods a day costs %.1f h of cpu a year on the numpy path '
        'and %.1f h on the pure one, in process%s. Neither is a reason to choose a path; the '
        'reason to choose is that one of them needs numpy.'
        % (PERIODS_PER_DAY, fast['total_s'] * PERIODS_PER_DAY * DAYS_PER_YEAR / 3600.0,
           slow['total_s'] * PERIODS_PER_DAY * DAYS_PER_YEAR / 3600.0,
           (', plus %s of startup for every command line' % _s(real['startup_s'])) if real
           else ''))
    add('')
    return '\n'.join(lines)


def _s(seconds):
    """A duration, in the unit that shows three significant figures. Returns str."""
    if seconds < 1e-3:
        return '%.0f us' % (seconds * 1e6)
    if seconds < 1.0:
        return '%.0f ms' % (seconds * 1e3)
    return '%.2f s' % seconds


def _rate(samples, seconds):
    """Throughput as samples per second. Returns str."""
    rate = samples / seconds
    return '%.1fM/s' % (rate / 1e6) if rate >= 1e6 else '%.0fk/s' % (rate / 1e3)


def _mark(run):
    """The marker a run whose cpu clock fell while it ran carries in a table. Returns str."""
    return ' (!)' if run['drift'] > DRIFT_LIMIT else ''


# ------------------------------------------------------------------ the run

def main(argv=None):
    """Run the whole benchmark, or one worker task. Returns the exit code."""
    args = _parse(argv)
    if args.worker:
        return worker(args.worker)

    env = environment()
    if env['numpy'] is None:
        raise SystemExit('numpy is not installed, so there is nothing to compare against')
    workdir = tempfile.mkdtemp(prefix='miniflux-bench-')
    sys.stderr.write('working in %s\ngenerating synthetic data\n' % workdir)
    folders = generate(workdir, [samples for _minutes, samples in SWEEP])
    sys.stderr.write('measuring (%d fresh subprocesses per configuration, both paths '
                     'interleaved)\n' % args.repeats)

    sweep = {}
    for minutes, samples in SWEEP:
        config_path = synthetic_config(workdir, folders[samples], minutes, samples)
        runs = run_jobs([worker_job(workdir, '%d min numpy' % minutes, config_path, False),
                         worker_job(workdir, '%d min pure' % minutes, config_path, True)],
                        args.repeats, args.cooldown)
        row = {}
        for key in ('numpy', 'pure'):
            reps = runs['%d min %s' % (minutes, key)]
            row[key] = best_run(reps)
            row[key + '_median'] = statistics.median(run['total_s'] for _wall, run in reps)
            row['samples'] = row[key]['samples']
        sweep[minutes] = row

    real = None
    real_path = real_config(workdir)
    if real_path is None:
        sys.stderr.write('  %s not found; skipping the real-file section\n' % REAL_FILE)
    else:
        # `--version` does every import a `run` pays before it reads a byte, and startup can
        # only be seen from outside: a timer started inside the process has already missed it.
        command = [sys.executable, '-m', 'miniflux', 'run', real_path, '--log-level', 'ERROR']
        times = run_jobs([('miniflux --version',
                           [sys.executable, '-m', 'miniflux', '--version']),
                          ('real file numpy cli', command),
                          ('real file pure cli', command + ['--pure'])],
                         args.repeats, args.cooldown)
        runs = run_jobs([worker_job(workdir, 'real file numpy', real_path, False),
                         worker_job(workdir, 'real file pure', real_path, True)],
                        args.repeats, args.cooldown)
        real = {'startup_s': min(wall for wall, _r in times['miniflux --version'])}
        for key in ('numpy', 'pure'):
            walls = [wall for wall, _r in times['real file %s cli' % key]]
            real[key + '_best'] = min(walls)
            real[key + '_median'] = statistics.median(walls)
            real[key + '_shape'] = best_run(runs['real file %s' % key])
        real['periods'] = real['numpy_shape']['periods']
        real['samples'] = real['numpy_shape']['samples']

    report = render(env, sweep, real, args.repeats)
    sys.stdout.write(report)
    with open(REPORT, 'w', encoding='utf-8') as handle:
        handle.write(report)
    sys.stderr.write('wrote %s\n' % REPORT)
    return 0


def _parse(argv):
    """Parse the command line. Returns the argparse namespace."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--repeats', type=int, default=5, metavar='N',
                        help='fresh subprocesses per configuration (default 5)')
    parser.add_argument('--cooldown', type=float, default=15.0, metavar='SECONDS',
                        help='longest idle wait before a timed run, so each starts at full '
                             'clock (default 15; 0 disables it, and the long runs then measure '
                             'the power limit as much as the code)')
    parser.add_argument('--worker', metavar='SPEC.json', help=argparse.SUPPRESS)
    return parser.parse_args(argv)


if __name__ == '__main__':
    sys.exit(main())
