"""Measure what miniflux costs, with and without the numpy fast path.

Run it with no arguments::

    python scripts/benchmark.py

It generates its own data, measures four things, prints a markdown report and writes it to
``docs/BENCHMARK.md``:

* a **period-length sweep** -- 10, 30 and 60 minutes of 20 Hz data (12000, 36000 and 72000
  samples) -- on both paths, in seconds per period and samples per second;
* the **read+parse stage separately from the compute stages**, because reading is the same
  work on both paths and dominates one of them;
* a **per-step breakdown** of the nine steps of ``pipeline.STEPS``;
* the **real file** ``20213_RAW_20Hz_2025-09-08-2000.csv`` from the ONEFlux_preproc sample
  data, end to end through the command line on both paths.

Four rules about the measurement itself, each of them learned by getting it wrong first:

* every configuration runs in a **fresh subprocess**, and the number reported is the best of
  ``--repeats`` (5 by default). In-process repetition drifts -- the allocator, the bytecode
  caches and the OS scheduler all move under a long loop -- so a loop measures the loop as
  much as the code.
* the per-step breakdown comes from the **single repetition with the best total**, not from
  a per-step minimum across repetitions. A per-step minimum is a run that never happened,
  and its parts would not add up to any total.
* the two paths are **interleaved and preceded by an idle wait**. This was written on a
  laptop that ran the same loop three times slower once warm, which is a bigger factor than
  numpy buys; measuring all of one path and then all of the other would have handed that
  drift to whichever went second.
* every run **probes the cpu clock before and after itself**, and a run whose clock fell
  while it ran is marked in the report as an upper bound rather than a measurement.

The synthetic data is built by importing ``examples/make_sample.py`` and calling it with a
different sample count: the physics lives there and is not repeated here. The two
configurations are generated from ``examples/miniflux.ini`` by substituting the handful of
keys that differ, so the benchmark cannot drift away from the shipped example.

Standard library only, like the program it measures. It takes about ten minutes, most of it
spent deliberately idle.

Every measurement is also saved as ``results.json`` in the working directory, so the wording
of the report can be edited and rebuilt without measuring again::

    python scripts/benchmark.py --from-json <workdir>/results.json

Internal option::

    python scripts/benchmark.py --worker SPEC.json    # one measurement, one process
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

#: The repository root: this file is ``<root>/scripts/benchmark.py``. Inserted on
#: ``sys.path`` so the benchmark runs from any working directory without an install.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

#: The real file, as shipped in the sibling ONEFlux_preproc checkout. Absent is not an
#: error: that section of the report is skipped and says so.
REAL_FILE = ('D:/_gitRepo/ONEFlux_preproc/data/sample/FR-Jus_20250908/FLUX/'
             '20213_RAW_20Hz_2025-09-08-2000.csv')

#: ``(averaging_minutes, samples)`` at 20 Hz. The middle row is the one every number in
#: the prose is quoted from: one 30-minute period is the unit a flux tower is run in.
SWEEP = ((10, 12000), (30, 36000), (60, 72000))
DEFAULT_MINUTES = 30

HZ = 20.0
PERIODS_PER_DAY = 48
DAYS_PER_MONTH = 30
DAYS_PER_YEAR = 365

#: The line a worker's result arrives on. A worker also inherits stderr, where the package
#: log goes, so the parent picks its answer out by prefix rather than reading all of stdout.
RESULT = '__RESULT__'

#: Assumed, never measured: how much slower one core of a small single-board computer is
#: than one core of the machine this ran on, for scalar CPython. Stated in the report as an
#: assumption, with its consequences computed from it.
PI_FACTORS = ((4.0, 'a Raspberry Pi 5 class core (Cortex-A76, 2.4 GHz)'),
              (10.0, 'a Raspberry Pi 4 class core (Cortex-A72, 1.8 GHz)'))

#: How long one clock probe should take, and the ratio above which a run is reported as
#: having been throttled while it ran. The probe is a fixed float loop measured immediately
#: before and immediately after the timed region; its size is calibrated at runtime, so it
#: is a duration on any machine rather than an iteration count tuned to one.
PROBE_S = 0.02
DRIFT_LIMIT = 1.25

#: The wall clock of the last timed subprocess, which is how much heat the next one has to
#: wait out. A module global because it is a property of the machine, not of any one
#: measurement: the rest a run needs depends on the run before it, whatever it measured.
_LAST_RUN_S = 0.0


# ------------------------------------------------------------------ the worker (one process)

def worker(spec_path):
    """Run one measurement in this process and print its result as JSON. Returns 0.

    This is the other side of :func:`measure`: the parent writes a spec file, spawns a fresh
    interpreter on it, and reads the line beginning with ``RESULT``. Everything measured here
    is measured once, in a process that has done nothing else.
    """
    with open(spec_path, 'r', encoding='utf-8') as handle:
        spec = json.load(handle)
    task = spec['task']
    if task == 'stages':
        result = _time_stages(spec['config'], spec['pure'])
    elif task == 'readsplit':
        result = _time_read_split(spec['config'])
    else:
        raise SystemExit('unknown task %r' % task)
    sys.stdout.write(RESULT + json.dumps(result) + '\n')
    return 0


def _quiet():
    """Silence the package log and return the modules under test.

    A run at INFO writes a few dozen lines to stderr; a run at ERROR writes none. The level
    is set before ``config.load`` so that even the configuration echo is never formatted,
    and it is set identically on both paths, so it biases neither.
    """
    import logging
    package = logging.getLogger('miniflux')
    package.addHandler(logging.NullHandler())   # or logging's lastResort prints WARNINGs
    package.setLevel(logging.ERROR)
    from miniflux import config, kernels, pipeline, read, write
    return config, kernels, pipeline, read, write


def _time_stages(config_path, pure):
    """Time one full run, split into read+parse, the nine steps, and write. Returns a dict.

    The loop is ``pipeline.run`` unrolled by hand, because that is the only way to see the
    stages separately: the generator is advanced with an explicit ``next`` so that the time
    spent reading a period is the time spent inside that call, and each step is called by
    name. The differences from ``pipeline.run`` itself are the logging-filter bookkeeping it
    installs (a counter on nine loggers, once per run) and its per-step ``try``, which costs
    nothing until something raises. Both are per-period constants far below the resolution of
    anything reported here.

    ``drift`` is the second clock probe divided by the first: a laptop under sustained load
    drops its clock, and on the machine this was written on the drop was larger than the
    whole numpy speedup. Probing around this run and no other is the only way to say whether
    a given row measures the code or the power limit. The probes cost about 120 ms in total
    and sit outside the timed region.
    """
    probe_count = _probe_size()
    probe_before = _probe(probe_count)
    config, kernels, pipeline, read, write = _quiet()
    t_config = perf_counter()
    cfg = config.load(config_path)
    config_s = perf_counter() - t_config
    if pure:
        # The same three assignments cli._force_pure makes, without argparse in the way.
        cfg.runtime.use_numpy = 'off'
        cfg.runtime.numpy_enabled = False
        kernels.set_numpy(False)

    steps = dict((name, 0.0) for name, _fn in pipeline.STEPS)
    read_s = write_s = 0.0
    periods = samples = 0
    counters = {'periods': 0, 'skipped_short': 0, 'rows': 0, 'warnings': 0}

    t_start = perf_counter()
    with write.Writer(cfg.files.output_csv, cfg) as writer:
        stream = read.periods(cfg, counters)
        while True:
            t0 = perf_counter()
            try:
                period = next(stream)
            except StopIteration:
                # The last period is cut inside this call, so its cost is read time too.
                read_s += perf_counter() - t0
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
    probe_after = _probe(probe_count)

    return {'pure': bool(pure), 'numpy_enabled': bool(cfg.runtime.numpy_enabled),
            'config_s': config_s, 'total_s': total_s, 'read_s': read_s, 'write_s': write_s,
            'compute_s': total_s - read_s - write_s, 'steps': steps,
            'periods': periods, 'samples': samples,
            'drift': probe_after / probe_before if probe_before > 0 else 1.0}


def _burn(count):
    """A fixed float loop, the shape of the pure path's inner loops. Returns its sum."""
    import math
    total = 0.0
    for i in range(count):
        total += math.sqrt(i * 1.000001) + math.log(i + 1.0)
    return total


def _probe_size():
    """Iterations of :func:`_burn` that take about PROBE_S here. Returns int."""
    _burn(20000)                                # warm the bytecode, not the silicon
    t0 = perf_counter()
    _burn(50000)
    unit = (perf_counter() - t0) / 50000.0
    return max(1000, int(PROBE_S / unit))


def _probe(count, tries=3):
    """The cpu's current speed on a fixed loop. Returns the best of `tries`, in seconds.

    Best, not one shot: a single 20 ms probe is easily half as fast again because the
    scheduler took the core away, and a drift figure built from two such probes would flag
    runs that were never throttled at all. The minimum is the one the clock actually
    delivered.
    """
    best = None
    for _try in range(tries):
        t0 = perf_counter()
        _burn(count)
        one = perf_counter() - t0
        best = one if best is None else min(best, one)
    return best


def _time_read_split(config_path):
    """Split the read stage into tokenising and converting. Returns a dict.

    Reaches into ``read._rows`` on purpose: it is the csv layer on its own, and the only way
    to say how much of the read stage is ``csv.reader`` splitting bytes into strings and how
    much is turning those strings into a datetime and nine floats. Nothing in the report
    depends on this beyond the "where does the time go" paragraph.
    """
    config, _kernels, _pipeline, read, _write = _quiet()
    cfg = config.load(config_path)
    import glob
    paths = sorted(glob.glob(cfg.files.input_glob))

    t0 = perf_counter()
    rows = 0
    for path in paths:
        for _lineno, _row in read._rows(path, cfg):
            rows += 1
    csv_s = perf_counter() - t0

    t1 = perf_counter()
    samples = 0
    for period in read.periods(cfg, None):
        samples += period['meta'].get('n_in', 0)
    full_s = perf_counter() - t1
    return {'csv_s': csv_s, 'full_s': full_s, 'rows': rows, 'samples': samples}


# ------------------------------------------------------------------ the parent (repetitions)

def cool(limit, label):
    """Idle until the cpu has forgotten the last measurement. Returns the seconds slept.

    Twice the last run's wall clock, capped: a machine that was seen to need about ten
    seconds of idle to recover from eight seconds of load needs proportionally less after a
    short one, and sleeping a flat maximum before a 200 ms measurement would spend the whole
    benchmark asleep. The floor is there because even a short run leaves some heat.
    """
    rest = min(limit, max(0.5, 2.0 * _LAST_RUN_S))
    if rest <= 0:
        return 0.0
    sys.stderr.write('  %s  cooling %4.1f s\r' % (label.ljust(34), rest))
    sys.stderr.flush()
    time.sleep(rest)
    return rest


def measure(specs, repeats, workdir, cooldown):
    """Run each ``(key, spec)`` in fresh subprocesses, interleaved. Returns {key: [results]}.

    Two rules, both about not measuring the machine instead of the code:

    * **fresh**: a repetition that shares an interpreter with the one before it shares its
      warmed caches and its fragmented heap, and measures those too;
    * **interleaved**: repetition *i* of every spec runs before repetition *i+1* of any of
      them. A laptop gets slower as a benchmark proceeds, so measuring all of numpy and then
      all of pure would charge the whole session's thermal drift to whichever ran last.
      Interleaving makes the drift common to both, which is what a ratio needs.
    """
    global _LAST_RUN_S
    out = dict((key, []) for key, _spec in specs)
    paths = {}
    for key, spec in specs:
        paths[key] = os.path.join(workdir, 'spec_%s.json' % _slug(key))
        with open(paths[key], 'w', encoding='utf-8') as handle:
            json.dump(spec, handle)
    for i in range(repeats):
        for key, _spec in specs:
            cool(cooldown, key)
            sys.stderr.write('  %s  rep %d/%d      \r' % (key.ljust(34), i + 1, repeats))
            sys.stderr.flush()
            t0 = perf_counter()
            done = subprocess.run([sys.executable, os.path.abspath(__file__),
                                   '--worker', paths[key]],
                                  cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _LAST_RUN_S = perf_counter() - t0
            if done.returncode != 0:
                raise SystemExit('worker %s failed (%d):\n%s'
                                 % (key, done.returncode,
                                    done.stderr.decode('utf-8', 'replace')))
            out[key].append(_result_of(done.stdout))
    sys.stderr.write('  %s  %d reps done      \n'
                     % ((', '.join(key for key, _s in specs))[:34].ljust(34), repeats))
    return out


def _slug(key):
    """A filename-safe form of a measurement's label. Returns str."""
    return ''.join(character if character.isalnum() else '_' for character in key)


def time_processes(commands, repeats, cooldown):
    """Time whole processes, interleaved. Returns {key: [wall clocks in seconds]}.

    Wall clock of each command as a shell would see it: interpreter start, imports, work,
    exit. That is what a user waits for, and it is the only way to measure the startup a
    per-period invocation pays -- a timer started inside the process has already missed it.
    Interleaved for the reason :func:`measure` is.
    """
    global _LAST_RUN_S
    out = dict((key, []) for key, _command in commands)
    for i in range(repeats):
        for key, command in commands:
            cool(cooldown, key)
            sys.stderr.write('  %s  rep %d/%d      \r' % (key.ljust(34), i + 1, repeats))
            sys.stderr.flush()
            t0 = perf_counter()
            done = subprocess.run(command, cwd=REPO,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            elapsed = perf_counter() - t0
            _LAST_RUN_S = elapsed
            if done.returncode != 0:
                raise SystemExit('%s exited %d:\n%s'
                                 % (' '.join(command), done.returncode,
                                    done.stderr.decode('utf-8', 'replace')))
            out[key].append(elapsed)
    sys.stderr.write('  %s  %d reps done      \n'
                     % ((', '.join(key for key, _c in commands))[:34].ljust(34), repeats))
    return out


def cli_command(config_path, pure):
    """The command line a user would type for one run. Returns a list of str."""
    command = [sys.executable, '-m', 'miniflux', 'run', config_path, '--log-level', 'ERROR']
    if pure:
        command.append('--pure')
    return command


def _result_of(stdout):
    """The worker's JSON line out of its stdout. Returns a dict."""
    for line in stdout.decode('utf-8', 'replace').splitlines():
        if line.startswith(RESULT):
            return json.loads(line[len(RESULT):])
    raise SystemExit('worker printed no result line')


def best_run(runs, key='total_s'):
    """The single repetition with the smallest `key`. Returns that dict.

    One whole run, so its stages add up to its total. A per-stage minimum taken across
    repetitions would be lower than any run that happened, and would not add up to anything.
    """
    return min(runs, key=lambda run: run[key])


def spread(values):
    """``(best, median)`` of a list of floats."""
    return min(values), statistics.median(values)


# ------------------------------------------------------------------ the data and the configs

def generate(workdir, sizes):
    """Write one synthetic raw file per sample count. Returns {samples: directory}.

    ``examples/make_sample.py`` is imported and driven, never copied: its module constants are
    what a different period length has to change, and the AR(1) turbulence, the planted lag
    and the planted spikes stay exactly as the example generates them. Each file gets its own
    directory so that one ``input_glob`` matches one file.

    The start instant is moved to one sample past midnight so that N samples at 20 Hz fill
    exactly one period of any of the lengths swept: the last sample lands on the closing
    boundary, which ``closed = right`` puts in the same period as the first.
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
        # A spike index past the end of a shorter series would raise; the physics of the
        # ones that fit is untouched.
        make_sample.SPIKES = dict((name, tuple(i for i in indices if i < samples))
                                  for name, indices in planted.items())
        columns = make_sample.build()
        make_sample.write_file(os.path.join(folder, 'sample_20hz.dat'), columns)
        out[samples] = folder
        sys.stderr.write('  generated %6d samples in %s\n' % (samples, folder))
    return out


def synthetic_config(workdir, folder, minutes, samples):
    """Write the config for one synthetic size. Returns its path.

    Everything but the four keys below is ``examples/miniflux.ini`` verbatim -- same despike
    threshold, same +-2 s lag window, same block detrend, same ITC test.
    """
    text = _template()
    text = _set(text, 'files', 'input_glob', folder.replace('\\', '/') + '/*.dat')
    text = _set(text, 'files', 'output_csv',
                os.path.join(workdir, 'out_n%d.csv' % samples).replace('\\', '/'))
    text = _set(text, 'period', 'averaging_minutes', str(minutes))
    text = _set(text, 'runtime', 'log_level', 'ERROR')
    text = _set(text, 'site', 'latitude', '45.0')     # see _template
    path = os.path.join(workdir, 'synthetic_n%d.ini' % samples)
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(text)
    return path


def real_config(workdir):
    """Write the config for the real TOA5 file. Returns its path, or None if it is absent.

    TOA5: the column names are on line 2 and the data starts on line 5. The IRGASON is an
    open path, so both gases arrive as molar densities (mg m-3 and g m-3) and WPL runs; the
    sonic temperature is in degC and the pressure in kPa. The site block is the example's
    placeholder -- the benchmark times this file, it does not publish its fluxes.
    """
    if not os.path.exists(REAL_FILE):
        return None
    text = _template()
    text = _set(text, 'files', 'input_glob', REAL_FILE)
    text = _set(text, 'files', 'output_csv',
                os.path.join(workdir, 'out_real.csv').replace('\\', '/'))
    text = _set(text, 'files', 'header_line', '2')
    text = _set(text, 'files', 'first_data_line', '5')
    text = _set(text, 'runtime', 'log_level', 'ERROR')
    text = _set(text, 'site', 'latitude', '45.0')
    for key, column in (('u', 'RAW_IRGASON_Ux'), ('v', 'RAW_IRGASON_Uy'),
                        ('w', 'RAW_IRGASON_Uz'), ('ts', 'RAW_IRGASON_Ts'),
                        ('co2', 'RAW_IRGASON_CO2'), ('h2o', 'RAW_IRGASON_H2O'),
                        ('p_air', 'RAW_IRGASON_CellPrs')):
        text = _set(text, 'variables', key, column)
    for key, unit in (('ts', 'degC'), ('ta', 'degC'), ('p_air', 'kPa'),
                      ('co2', 'mg/m3'), ('h2o', 'g/m3')):
        text = _set(text, 'units', key, unit)
    for key in ('co2_measure_type', 'h2o_measure_type'):
        text = _set(text, 'gases', key, 'molar_density')
    path = os.path.join(workdir, 'real.ini')
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(text)
    return path


def _template():
    """``examples/miniflux.ini`` as text.

    The one substitution every generated config makes beyond its own data is
    ``latitude = 45.0``: the example ships 0.0, which makes the Coriolis parameter zero and
    the three ITC deviations NaN, and a step that returns NaN early is a step that is not
    being timed.
    """
    with open(os.path.join(REPO, 'examples', 'miniflux.ini'), 'r', encoding='utf-8') as h:
        return h.read()


def _set(text, section, key, value):
    """Replace ``key`` inside ``[section]`` of an ini file. Returns the new text.

    Line based rather than configparser round-tripping, so the comments that document every
    key in the example survive into the generated file and it stays readable.
    """
    out = []
    current = None
    found = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('[') and ']' in stripped:
            # Section headers in the example carry a trailing comment, so the name is what
            # sits inside the brackets and not the whole line.
            current = stripped[1:stripped.index(']')]
        elif current == section and '=' in line:
            name = line.split('=', 1)[0].strip()
            if name == key:
                line = '%-19s = %s' % (key, value)
                found = True
        out.append(line)
    if not found:
        raise SystemExit('no key %r in section [%s] of examples/miniflux.ini' % (key, section))
    return '\n'.join(out) + '\n'


# ------------------------------------------------------------------ the machine

def environment():
    """What this ran on, read at runtime. Returns a dict of str."""
    try:
        import numpy
        numpy_version = numpy.__version__
    except ImportError:
        numpy_version = None
    return {'when': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'platform': platform.platform(),
            'machine': platform.machine(),
            'python': '%s %s' % (platform.python_implementation(), platform.python_version()),
            'numpy': numpy_version,
            'cpu': _cpu_name(),
            'cores': os.cpu_count()}


def _cpu_name():
    """The processor's own name for itself, per OS. Returns str.

    ``platform.processor()`` answers with a family and model number on Windows and with
    nothing at all on some Linuxes, so each OS is asked where it actually keeps the string.
    """
    system = platform.system()
    try:
        if system == 'Windows':
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
            with key:
                return winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        if system == 'Linux':
            with open('/proc/cpuinfo', 'r', encoding='utf-8') as handle:
                for line in handle:
                    if line.startswith('model name'):
                        return line.split(':', 1)[1].strip()
        if system == 'Darwin':
            done = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                  stdout=subprocess.PIPE)
            if done.returncode == 0:
                return done.stdout.decode('utf-8', 'replace').strip()
    except Exception:               # a benchmark never fails over a cosmetic string
        pass
    return platform.processor() or platform.machine() or 'unknown'


# ------------------------------------------------------------------ the report

def render(env, sweep, steps_best, readsplit, real, startup, repeats):
    """Build the whole of docs/BENCHMARK.md. Returns str.

    Every number in the prose is computed from the measurements passed in, so re-running the
    benchmark on another machine rewrites the conclusions with it instead of leaving last
    month's numbers in the sentences.
    """
    lines = []
    add = lines.append
    default = sweep[DEFAULT_MINUTES]
    numpy_run, pure_run = default['numpy'], default['pure']
    speedup = pure_run['total_s'] / numpy_run['total_s']

    add('# What miniflux costs')
    add('')
    add('Measured by `scripts/benchmark.py`, which generates its own data and writes this '
        'file. Re-run it and every number here is replaced.')
    add('')
    add('    python scripts/benchmark.py')
    add('')
    add('Each configuration runs in a **fresh subprocess** and is repeated %d times; the '
        'tables quote the **best** repetition, with the median beside it where it fits. '
        'Best, not mean: the slow repetitions are the machine doing something else, and '
        'averaging them in measures the machine. The per-step breakdown is taken from the '
        'single repetition with the best total, so its parts add up to its total.' % repeats)
    add('')
    add('The two paths are **interleaved**: repetition *i* of numpy runs before repetition '
        '*i+1* of pure. A laptop gets slower as a benchmark proceeds, and running one path '
        'to completion and then the other would charge that drift to whichever went last. '
        'Each timed run also idles first, and probes the cpu clock immediately before and '
        'after itself, so a row that was throttled while it ran says so.')
    add('')

    add('## The machine')
    add('')
    add('| | |')
    add('|---|---|')
    add('| measured | %s |' % env['when'])
    add('| cpu | %s (%s logical cores, one used) |' % (env['cpu'], env['cores']))
    add('| platform | %s |' % env['platform'])
    add('| python | %s |' % env['python'])
    add('| numpy | %s |' % (env['numpy'] or 'not installed'))
    add('')
    add('miniflux targets Python 3.8 and this is %s. A 3.8 interpreter has neither the '
        'specialising interpreter nor the faster call sequence, so the pure-Python numbers '
        'below are the optimistic end of what 3.8 would show; the numpy numbers, which '
        'spend their time inside numpy, would barely move.' % env['python'])
    add('')
    drifted = [(name, run) for name, run in _all_runs(sweep, real)
               if run.get('drift', 1.0) > DRIFT_LIMIT]
    add('**Thermal note, and it is not a footnote.** This benchmark was written on a laptop '
        'whose cpu, under sustained single-core load, was seen running the same fixed float '
        'loop more than three times slower than when cold -- a bigger factor than numpy '
        'buys, and enough to swamp the comparison it is measuring. So every timed run probes '
        'that loop immediately before and immediately after itself, and reports the ratio.')
    add('')
    if not drifted:
        add('On this run no measurement slowed by more than %.0f%% while it ran: the idle '
            'waits were long enough and the rows below are measurements of the code.'
            % (100.0 * (DRIFT_LIMIT - 1.0)))
    else:
        add('On this run %d measurement(s) slowed while running -- %s -- and they are marked '
            '**(!)** below. Those rows are upper bounds on the time, not measurements of the '
            'code; on a machine with a steady clock, expect less.'
            % (len(drifted), ', '.join('%s %.1fx' % (name, run['drift'])
                                       for name, run in drifted)))
    add('')

    add('## The headline')
    add('')
    add('One 30-minute period of 20 Hz data -- %d samples, the unit a flux tower is run in '
        '-- read from disk, pushed through all nine steps and written out:'
        % default['samples'])
    add('')
    add('| | numpy | pure Python | ratio |')
    add('|---|---:|---:|---:|')
    add('| total per period | %s | %s | %.1fx |'
        % (_s(numpy_run['total_s']) + _flag(numpy_run),
           _s(pure_run['total_s']) + _flag(pure_run), speedup))
    add('| read + parse | %s | %s | %.2fx |'
        % (_s(numpy_run['read_s']), _s(pure_run['read_s']),
           pure_run['read_s'] / numpy_run['read_s']))
    add('| the nine steps | %s | %s | %.1fx |'
        % (_s(numpy_run['compute_s']), _s(pure_run['compute_s']),
           pure_run['compute_s'] / numpy_run['compute_s']))
    add('| write the row | %s | %s | |'
        % (_s(numpy_run['write_s']), _s(pure_run['write_s'])))
    add('| throughput | %s | %s | |'
        % (_rate(default['samples'], numpy_run['total_s']),
           _rate(default['samples'], pure_run['total_s'])))
    add('')
    add('**numpy buys %.1fx overall, and it buys all of it in the compute stages (%.0fx '
        'there).** Reading is the same work either way -- `csv.reader`, a timestamp per '
        'row, nine `float()` calls per row, none of it through a kernel -- so the %.0f%% '
        'between the two read columns is run-to-run noise and not a difference. On the '
        'numpy path reading is now %.0f%% of the whole run.'
        % (speedup, pure_run['compute_s'] / numpy_run['compute_s'],
           abs(100.0 * pure_run['read_s'] / numpy_run['read_s'] - 100.0),
           100.0 * numpy_run['read_s'] / numpy_run['total_s']))
    add('')

    add('## Per step')
    add('')
    add('The nine steps of `pipeline.STEPS`, on the same %d-sample period, in the order '
        'they run. Seconds, and the share of that path\'s own compute time.'
        % default['samples'])
    add('')
    add('| step | numpy | share | pure | share | speedup |')
    add('|---|---:|---:|---:|---:|---:|')
    order = steps_best['order']
    for name in order:
        fast, slow = steps_best['numpy'][name], steps_best['pure'][name]
        add('| %s | %s | %.0f%% | %s | %.0f%% | %s |'
            % (name, _s(fast), 100.0 * fast / numpy_run['compute_s'],
               _s(slow), 100.0 * slow / pure_run['compute_s'], _times(slow, fast)))
    add('')
    add('A step costing under a millisecond is at the resolution of this measurement; read '
        '`assemble`, `wpl` and `detrend` as "free", not as three significant figures.')
    add('')
    top_pure = max(order, key=lambda name: steps_best['pure'][name])
    top_numpy = max(order, key=lambda name: steps_best['numpy'][name])
    add('**Pure Python is dominated by `%s` (%.0f%% of its compute time).** '
        % (top_pure, 100.0 * steps_best['pure'][top_pure] / pure_run['compute_s'])
        + 'It is the one step whose work is samples x lags rather than samples: the '
        'example config searches +-2 s, which at 20 Hz is 81 shifts, and every shift is a '
        'covariance over the whole period. That is ~%.1f million multiply-adds per scalar '
        'per period, and there are two scalars.' % (81 * default['samples'] / 1e6))
    add('')
    add('**With numpy the same step costs %s and the profile inverts: `%s` is now the '
        'largest (%.0f%% of compute).** `thermodynamics` is the step that does not speed up, '
        'and the table shows why it cannot: it measures %s, which is 1 plus noise. It is an '
        'explicit per-sample Python loop in `flux.py` -- two passes of moist-air state, a '
        'saturation vapour pressure and a specific heat per sample -- and it calls no kernel '
        'at all, so `--pure` and the default run the *same code* through it. Every other '
        'step hands whole series to `kernels.py`, which is where numpy lives.'
        % (_s(steps_best['numpy']['lag']), top_numpy,
           100.0 * steps_best['numpy'][top_numpy] / numpy_run['compute_s'],
           _times(steps_best['pure']['thermodynamics'],
                  steps_best['numpy']['thermodynamics'])))
    add('')
    add('`assemble` and `wpl` are scalar arithmetic on numbers the earlier steps already '
        'reduced, so they are free on both paths and always will be.')
    add('')

    add('## Period length')
    add('')
    add('20 Hz throughout, one period per run, the lag window fixed at +-2 s (81 shifts) so '
        'the work per sample is the same at every length.')
    add('')
    add('| period | samples | numpy best | median | samples/s | pure best | median | '
        'samples/s | ratio |')
    add('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    throttled = []
    for minutes, _samples in SWEEP:
        row = sweep[minutes]
        fast, slow = row['numpy'], row['pure']
        for key in ('numpy', 'pure'):
            if _flag(row[key]):
                throttled.append('%d min %s' % (minutes, key))
        add('| %d min | %d | %s | %s | %s | %s | %s | %s | %.1fx |'
            % (minutes, row['samples'],
               _s(fast['total_s']) + _flag(fast),
               _s(row['numpy_median']), _rate(row['samples'], fast['total_s']),
               _s(slow['total_s']) + _flag(slow),
               _s(row['pure_median']), _rate(row['samples'], slow['total_s']),
               slow['total_s'] / fast['total_s']))
    add('')
    clean = {}
    for key in ('numpy', 'pure'):
        values = [sweep[minutes]['samples'] / sweep[minutes][key]['total_s']
                  for minutes, _n in SWEEP if not _flag(sweep[minutes][key])]
        clean[key] = (min(values), max(values)) if values else None
    varies = ' and '.join('%.0f%% (%s)' % (100.0 * (clean[key][1] / clean[key][0] - 1.0), key)
                          for key in ('numpy', 'pure') if clean[key])
    add('Nothing in the pipeline is quadratic in the period: the lag window is set in '
        'seconds, so a longer period does not widen the search, and the per-sample cost '
        'should be flat. %s Quote the throughput as %s (numpy) and %s (pure), and do not '
        'expect better than two significant figures from it.'
        % (('Across the rows that were not throttled it varies by %s.' % varies) if varies
           else 'Every row of this sweep was throttled, so this run cannot show that.',
           _rate(default['samples'], numpy_run['total_s']),
           _rate(default['samples'], pure_run['total_s'])))
    add('')
    if throttled:
        longest = SWEEP[-1][0]
        expected = (sweep[DEFAULT_MINUTES]['pure']['total_s']
                    * SWEEP[-1][1] / default['samples'])
        add('The rows marked (!) (%s) ran longer than this machine holds its clock, so they '
            'are slower than the code is. Scaling the clean %d-minute pure row by sample '
            'count predicts %s for %d minutes; the measurement says %s. The difference is '
            'the cpu, not the pipeline -- on a machine with a steady clock, expect the '
            'linear number.'
            % (', '.join(throttled), DEFAULT_MINUTES, _s(expected), longest,
               _s(sweep[longest]['pure']['total_s'])))
        add('')

    if real:
        fast_shape, slow_shape = real['numpy_shape'], real['pure_shape']
        add('## A real file, through the command line')
        add('')
        add('`%s`' % os.path.basename(REAL_FILE))
        add('')
        add('TOA5, %d data rows cut into %d periods, an open-path IRGASON: both gases arrive '
            'as molar densities (mg m-3, g m-3), so WPL runs -- which the synthetic file, '
            'in mixing ratios, does not exercise. Timed as `python -m miniflux run CONFIG`: '
            'the whole process, interpreter startup and imports included, because that is '
            'what a user waits for.' % (real['samples'], real['periods']))
        add('')
        add('| | best | median | of it, startup | the rest |')
        add('|---|---:|---:|---:|---:|')
        add('| numpy | %s | %s | %s | %s |'
            % (_s(real['numpy_best']), _s(real['numpy_median']), _s(startup['cli_s']),
               _s(real['numpy_best'] - startup['cli_s'])))
        add('| pure (`--pure`) | %s | %s | %s | %s |'
            % (_s(real['pure_best']), _s(real['pure_median']), _s(startup['cli_s']),
               _s(real['pure_best'] - startup['cli_s'])))
        add('')
        add('(The command line cannot probe its own clock, so these two rows carry no (!) '
            'either way. Their in-process twins below can, and did not.)'
            if not (_flag(fast_shape) or _flag(slow_shape)) else
            '(The command line cannot probe its own clock; the in-process twins below show '
            'this file did trip the power limit, so read these as upper bounds too.)')
        add('')
        add('Startup -- `python -m miniflux --version`, which does every import a run does '
            'and then stops -- is %s, against %s for `python -c pass` and %s for '
            '`python -c "import numpy"`. So %.0f%% of the numpy run and %.0f%% of the pure '
            'one is spent before the first byte is read, and most of that is numpy. '
            '`--pure` still imports numpy, because it is installed and the flag only stops '
            'it being *used*; on a machine with no numpy at all the pure run would be about '
            '%s shorter.'
            % (_s(startup['cli_s']), _s(startup['bare_s']), _s(startup['numpy_s']),
               100.0 * startup['cli_s'] / real['numpy_best'],
               100.0 * startup['cli_s'] / real['pure_best'],
               _s(startup['numpy_s'] - startup['bare_s'])))
        add('')
        add('The same file measured in process, so the stages separate:')
        add('')
        add('| | total | read + parse | nine steps | samples/s |')
        add('|---|---:|---:|---:|---:|')
        add('| numpy | %s | %s | %s | %s |'
            % (_s(fast_shape['total_s']) + _flag(fast_shape), _s(fast_shape['read_s']),
               _s(fast_shape['compute_s']), _rate(real['samples'], fast_shape['total_s'])))
        add('| pure | %s | %s | %s | %s |'
            % (_s(slow_shape['total_s']) + _flag(slow_shape), _s(slow_shape['read_s']),
               _s(slow_shape['compute_s']), _rate(real['samples'], slow_shape['total_s'])))
        add('')
        add('Throughput on real data is %s (numpy) and %s (pure), against %s and %s on the '
            'synthetic file: %.0f%% and %.0f%% apart. The synthetic file is therefore a '
            'fair stand-in, which is the only claim this comparison is making -- the two '
            'files differ in more than one way at once (the real one carries %d columns '
            'nobody reads but `csv.reader` still splits, holds gaps, and is cut into two '
            'periods of very different length), so the small difference is not attributable '
            'to any one of them.'
            % (_rate(real['samples'], fast_shape['total_s']),
               _rate(real['samples'], slow_shape['total_s']),
               _rate(default['samples'], numpy_run['total_s']),
               _rate(default['samples'], pure_run['total_s']),
               abs(100.0 * (default['samples'] / numpy_run['total_s'])
                   / (real['samples'] / fast_shape['total_s']) - 100.0),
               abs(100.0 * (default['samples'] / pure_run['total_s'])
                   / (real['samples'] / slow_shape['total_s']) - 100.0),
               7))
        add('')

    add('## What a year costs')
    add('')
    add('Continuous 20 Hz data, %d periods a day, from the per-period totals above. These '
        'are in-process figures: one command line per period would add %s of startup each '
        'time, which is %s a year.'
        % (PERIODS_PER_DAY, _s(startup['cli_s']),
           _long(startup['cli_s'] * PERIODS_PER_DAY * DAYS_PER_YEAR)))
    add('')
    add('| | numpy | pure Python |')
    add('|---|---:|---:|')
    for label, factor in (('one 30-min period', 1),
                          ('one day (%d periods)' % PERIODS_PER_DAY, PERIODS_PER_DAY),
                          ('one month (%d days)' % DAYS_PER_MONTH,
                           PERIODS_PER_DAY * DAYS_PER_MONTH),
                          ('one year (%d days)' % DAYS_PER_YEAR,
                           PERIODS_PER_DAY * DAYS_PER_YEAR)):
        add('| %s | %s | %s |' % (label, _long(numpy_run['total_s'] * factor),
                                  _long(pure_run['total_s'] * factor)))
    add('')
    add('| | numpy | pure Python |')
    add('|---|---:|---:|')
    add('| real time processed per second of cpu | %s | %s |'
        % (_ratio_real(numpy_run['total_s']), _ratio_real(pure_run['total_s'])))
    add('| duty cycle to keep up live | %.4f%% | %.3f%% |'
        % (100.0 * numpy_run['total_s'] / 1800.0, 100.0 * pure_run['total_s'] / 1800.0))
    add('')
    add('A year of 20 Hz data is %s on the numpy path and %s on the pure one. Neither is a '
        'reason to choose a path. The reason to choose is that one of them needs numpy.'
        % (_long(numpy_run['total_s'] * PERIODS_PER_DAY * DAYS_PER_YEAR),
           _long(pure_run['total_s'] * PERIODS_PER_DAY * DAYS_PER_YEAR)))
    add('')

    add('## On a single-board computer')
    add('')
    add('Not measured. The extrapolation below assumes a factor, and the factor is a guess '
        'from published CPython benchmarks, not something this script ran:')
    add('')
    add('| assumed core | factor | one period, numpy | one period, pure | live duty cycle, '
        'pure |')
    add('|---|---:|---:|---:|---:|')
    for factor, description in PI_FACTORS:
        add('| %s | %.0fx slower | %s | %s | %.1f%% |'
            % (description, factor, _s(numpy_run['total_s'] * factor),
               _s(pure_run['total_s'] * factor),
               100.0 * pure_run['total_s'] * factor / 1800.0))
    add('')
    worst = max(factor for factor, _d in PI_FACTORS)
    add('**Yes, comfortably** -- with the caveat that these are estimates. Even at the '
        'pessimistic %.0fx, one 30-minute period costs %s on the pure path, which is %.1f%% '
        'of the 1800 s of real time it covers; the board spends %.0f%% of its life idle and '
        'keeps up live with no numpy, no compiler and no wheels. Memory is the easier '
        'question: one period is nine float series of %d samples, %.1f MB of `array(\'d\')` '
        'plus the timestamps, and only one period is ever in memory.'
        % (worst, _s(pure_run['total_s'] * worst),
           100.0 * pure_run['total_s'] * worst / 1800.0,
           100.0 - 100.0 * pure_run['total_s'] * worst / 1800.0,
           default['samples'], 9 * 8 * default['samples'] / 1e6))
    add('')
    add('What would make that estimate wrong: a board with much less memory bandwidth, a '
        'longer lag window (the search is samples x lags, and the window is in seconds, so '
        'doubling it doubles the largest pure-path step), or a 32-bit build where every '
        'float is boxed differently.')
    add('')

    add('## Where the remaining time goes')
    add('')
    if readsplit:
        add('The read stage of the %d-sample period, split:' % default['samples'])
        add('')
        add('| | seconds | share of read |')
        add('|---|---:|---:|')
        add('| `csv.reader` tokenising | %s | %.0f%% |'
            % (_s(readsplit['csv_s']), 100.0 * readsplit['csv_s'] / readsplit['full_s']))
        add('| timestamp + %d floats per row, into the period | %s | %.0f%% |'
            % (9, _s(readsplit['full_s'] - readsplit['csv_s']),
               100.0 * (readsplit['full_s'] - readsplit['csv_s']) / readsplit['full_s']))
        add('| all of read | %s | 100%% |' % _s(readsplit['full_s']))
        add('')
        add('(Measured in its own process, which is why the total is %.0f%% off the %s the '
            'same stage takes inside a full run. The split is what matters here, not the '
            'absolute.)'
            % (abs(100.0 * readsplit['full_s'] / numpy_run['read_s'] - 100.0),
               _s(numpy_run['read_s'])))
        add('')
    add('On the **numpy path**, %.0f%% of the run is read+parse and %.0f%% of what is left '
        'is `thermodynamics`. Neither touches a kernel. Making this path faster means '
        'making the parser faster -- and the parser is `csv.reader` plus `float()` plus a '
        '`datetime` per row, all of which are already C. The honest ceiling is the file '
        'format: a text CSV of %d rows has to be tokenised %d times whatever reads it.'
        % (100.0 * numpy_run['read_s'] / numpy_run['total_s'],
           100.0 * steps_best['numpy']['thermodynamics'] / numpy_run['compute_s'],
           default['samples'], default['samples']))
    add('')
    add('On the **pure path**, %.0f%% of compute is the lag search. Three things would move '
        'it, in order of how much they cost in complexity:'
        % (100.0 * steps_best['pure']['lag'] / pure_run['compute_s']))
    add('')
    add('1. **Narrow the window.** `co2_min_s`/`co2_max_s` are configuration, not physics. '
        '+-0.5 s instead of +-2 s is 21 shifts instead of 81 and cuts the step by about '
        '4x. This costs nothing and needs no code.')
    add('2. **Compute the lag curve incrementally.** Each shift currently sums its own '
        'products over the whole period; consecutive shifts share all but a few terms. '
        'A running update, or an FFT cross-correlation, turns samples x lags into samples '
        'x log(samples) -- but an FFT needs a complex numeric stack, which is exactly what '
        'the pure path exists to avoid.')
    add('3. **Give `thermodynamics` a fast path.** It is %.0f%% of pure compute and does '
        'not speed up with numpy either, so it is the second target on both paths. Most of '
        'its cost is `saturation_vapour_pressure` and the moist-air state being rebuilt '
        'twice per sample; the second pass exists because Ta has to be resolved before the '
        'state can be built at it.'
        % (100.0 * steps_best['pure']['thermodynamics'] / pure_run['compute_s']))
    add('')
    add('None of these are worth doing for speed alone at %s a period. They are worth '
        'knowing about if the window ever widens or the sample rate ever goes to 100 Hz.'
        % _s(pure_run['total_s']))
    add('')
    return '\n'.join(lines)


def _s(seconds):
    """A duration, in the unit that shows three significant figures. Returns str."""
    if seconds is None:
        return '-'
    if seconds < 1e-3:
        return '%.0f us' % (seconds * 1e6)
    if seconds < 1.0:
        return '%.0f ms' % (seconds * 1e3)
    return '%.2f s' % seconds


def _all_runs(sweep, real):
    """Every timed in-process run, as ``(name, run)``. Returns a list."""
    out = []
    for minutes, _samples in SWEEP:
        for key in ('numpy', 'pure'):
            out.append(('%d min %s' % (minutes, key), sweep[minutes][key]))
    if real:
        for key in ('numpy', 'pure'):
            out.append(('real file %s' % key, real[key + '_shape']))
    return out


def _flag(run):
    """A marker when the cpu clock fell while this run was running. Returns str."""
    return ' (!)' if run.get('drift', 1.0) > DRIFT_LIMIT else ''


def _times(slow, fast):
    """A speed ratio at the precision it deserves. Returns str.

    Two significant figures below 10x: the difference between 1.1x and 1x is the whole
    point of the `thermodynamics` row, and rounding it away would tell the opposite story.
    """
    if fast <= 0 or slow <= 0:
        return '-'
    ratio = slow / fast
    return ('%.1fx' % ratio) if ratio < 10 else ('%.0fx' % ratio)


def _long(seconds):
    """A long duration in human units. Returns str."""
    if seconds < 90:
        return '%.1f s' % seconds
    if seconds < 5400:
        return '%.1f min' % (seconds / 60.0)
    if seconds < 172800:
        return '%.1f h' % (seconds / 3600.0)
    return '%.1f days' % (seconds / 86400.0)


def _rate(samples, seconds):
    """Throughput as samples per second. Returns str."""
    rate = samples / seconds
    if rate >= 1e6:
        return '%.1fM/s' % (rate / 1e6)
    return '%.0fk/s' % (rate / 1e3)


def _ratio_real(seconds_per_period):
    """How much real time one second of cpu covers. Returns str."""
    return _long(1800.0 / seconds_per_period)


# ------------------------------------------------------------------ the run

def main(argv=None):
    """Run the whole benchmark, or one worker task. Returns the exit code."""
    args = _parse(argv)
    if args.worker:
        return worker(args.worker)

    if args.from_json:
        # Re-render an earlier measurement. The wording of this report is edited far more
        # often than the machine changes, and re-measuring to fix a sentence would both
        # waste ten minutes and quietly swap every number under the edit.
        with open(args.from_json, 'r', encoding='utf-8') as handle:
            saved = json.load(handle)
        report = render(saved['env'], _keys_to_int(saved['sweep']), saved['steps_best'],
                        saved['readsplit'], saved['real'], saved['startup'],
                        saved['repeats'])
        return _emit(report, args, None)

    env = environment()
    if env['numpy'] is None:
        raise SystemExit('numpy is not installed, so there is nothing to compare against')

    workdir = args.workdir or tempfile.mkdtemp(prefix='miniflux-bench-')
    os.makedirs(workdir, exist_ok=True)
    sys.stderr.write('working in %s\n' % workdir)
    sys.stderr.write('generating synthetic data\n')
    folders = generate(workdir, [samples for _minutes, samples in SWEEP])

    sweep = {}
    steps_best = None
    readsplit = None
    sys.stderr.write('measuring (%d fresh subprocesses per configuration, both paths '
                     'interleaved)\n' % args.repeats)
    for minutes, samples in SWEEP:
        config_path = synthetic_config(workdir, folders[samples], minutes, samples)
        runs = measure([('%d min, numpy' % minutes,
                         {'task': 'stages', 'config': config_path, 'pure': False}),
                        ('%d min, pure' % minutes,
                         {'task': 'stages', 'config': config_path, 'pure': True})],
                       args.repeats, workdir, args.cooldown)
        row = {'samples': samples}
        for key in ('numpy', 'pure'):
            these = runs['%d min, %s' % (minutes, key)]
            row[key] = best_run(these)
            row[key + '_median'] = statistics.median(run['total_s'] for run in these)
            row['samples'] = row[key]['samples']
        sweep[minutes] = row
        if minutes == DEFAULT_MINUTES:
            steps_best = {'numpy': row['numpy']['steps'], 'pure': row['pure']['steps'],
                          'order': [name for name in _step_order()]}
            splits = measure([('%d min, read split' % minutes,
                               {'task': 'readsplit', 'config': config_path})],
                             args.repeats, workdir, args.cooldown)
            readsplit = best_run(splits['%d min, read split' % minutes], key='full_s')

    # Startup is measured the only way it can be: from outside the process. `--version`
    # imports the package, which imports kernels, which imports numpy -- the same imports a
    # `run` pays before it reads a byte.
    started = time_processes([('bare interpreter', [sys.executable, '-c', 'pass']),
                              ('miniflux --version',
                               [sys.executable, '-m', 'miniflux', '--version']),
                              ('import numpy', [sys.executable, '-c', 'import numpy'])],
                             args.repeats, args.cooldown)
    startup = {'bare_s': min(started['bare interpreter']),
               'cli_s': min(started['miniflux --version']),
               'numpy_s': min(started['import numpy'])}

    real = None
    real_path = real_config(workdir)
    if real_path is None:
        sys.stderr.write('  %s not found; skipping the real-file section\n' % REAL_FILE)
    else:
        times = time_processes([('real file, numpy, cli', cli_command(real_path, False)),
                                ('real file, pure, cli', cli_command(real_path, True))],
                               args.repeats, args.cooldown)
        runs = measure([('real file, numpy',
                         {'task': 'stages', 'config': real_path, 'pure': False}),
                        ('real file, pure',
                         {'task': 'stages', 'config': real_path, 'pure': True})],
                       args.repeats, workdir, args.cooldown)
        numpy_times = times['real file, numpy, cli']
        pure_times = times['real file, pure, cli']
        shapes = dict((key, best_run(runs['real file, ' + key])) for key in ('numpy', 'pure'))
        real = {'numpy_best': min(numpy_times), 'numpy_median': statistics.median(numpy_times),
                'pure_best': min(pure_times), 'pure_median': statistics.median(pure_times),
                'numpy_shape': shapes['numpy'], 'pure_shape': shapes['pure'],
                'periods': shapes['numpy']['periods'], 'samples': shapes['numpy']['samples']}

    saved_path = os.path.join(workdir, 'results.json')
    with open(saved_path, 'w', encoding='utf-8') as handle:
        json.dump({'env': env, 'sweep': sweep, 'steps_best': steps_best,
                   'readsplit': readsplit, 'real': real, 'startup': startup,
                   'repeats': args.repeats}, handle, indent=1, sort_keys=True)
    report = render(env, sweep, steps_best, readsplit, real, startup, args.repeats)
    return _emit(report, args, saved_path)


def _emit(report, args, saved_path):
    """Print the report, write it where asked, and say where the raw numbers are. Returns 0."""
    sys.stdout.write(report)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as handle:
            handle.write(report)
        sys.stderr.write('wrote %s\n' % args.output)
    if saved_path:
        sys.stderr.write('raw measurements in %s (re-render with --from-json)\n' % saved_path)
    return 0


def _keys_to_int(sweep):
    """JSON turns the sweep's integer minute keys into strings; turn them back. Returns dict."""
    return dict((int(minutes), row) for minutes, row in sweep.items())


def _step_order():
    """The nine step names in execution order. Returns a list of str."""
    from miniflux import pipeline
    return [name for name, _fn in pipeline.STEPS]


def _parse(argv):
    """Parse the command line. Returns the argparse namespace."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--repeats', type=int, default=5, metavar='N',
                        help='fresh subprocesses per configuration (default 5, minimum 3)')
    parser.add_argument('--cooldown', type=float, default=15.0, metavar='SECONDS',
                        help='longest idle wait before a timed run, so the cpu starts each '
                             'one at full clock (default 15; 0 disables the wait and the '
                             'long runs then measure the power limit as much as the code)')
    parser.add_argument('--workdir', metavar='DIR',
                        help='where to put the generated data (default: a temp directory)')
    parser.add_argument('--output', metavar='FILE',
                        default=os.path.join(REPO, 'docs', 'BENCHMARK.md'),
                        help='where to write the report (default docs/BENCHMARK.md)')
    parser.add_argument('--no-write', dest='output', action='store_const', const=None,
                        help='print the report and write no file')
    parser.add_argument('--from-json', metavar='RESULTS.json', dest='from_json',
                        help='re-render the report from an earlier run\'s results.json '
                             'instead of measuring anything again')
    parser.add_argument('--worker', metavar='SPEC.json', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.worker and not args.from_json and args.repeats < 3:
        parser.error('--repeats must be at least 3; one repetition measures the machine')
    return args


if __name__ == '__main__':
    sys.exit(main())
