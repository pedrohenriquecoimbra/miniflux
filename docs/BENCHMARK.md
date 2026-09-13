# What miniflux costs

Measured by `scripts/benchmark.py`, which generates its own data and writes this file. Re-run it and every number here is replaced.

    python scripts/benchmark.py

Each configuration runs in a **fresh subprocess**, 5 times; the tables quote the best repetition with the median beside it, and the per-step breakdown is the single repetition with the best total, so its parts add up to it. The two paths are **interleaved**, so a session that slows as it proceeds slows both and not whichever ran last. Every run idles first and probes the cpu clock around itself: nothing here slowed by more than 25% while it ran.

| the machine | |
|---|---|
| measured | 2026-09-13 11:49 |
| cpu | 11th Gen Intel(R) Core(TM) i9-11950H @ 2.60GHz (16 logical cores, one used) |
| platform | Windows-11-10.0.26200-SP0 |
| python | CPython 3.14.3 |
| numpy | 2.5.1 |

miniflux targets Python 3.8 and this is CPython 3.14.3. A 3.8 interpreter has neither the specialising interpreter nor the faster call sequence, so the pure-Python numbers below are the optimistic end of what 3.8 would show; the numpy numbers, which spend their time inside numpy, would barely move.

## One period, end to end

20 Hz synthetic data, one period per run, read from disk, pushed through all nine steps and written out. The lag window is fixed at +-2 s (81 shifts) at every length, so the work per sample does not change with the period.

| period | samples | path | best | median | read + parse | nine steps | write | samples/s |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 10 min | 12000 | numpy | 177 ms | 186 ms | 108 ms | 69 ms | 131 us | 68k/s |
| 10 min | 12000 | pure | 568 ms | 602 ms | 109 ms | 459 ms | 117 us | 21k/s |
| 30 min | 36000 | numpy | 540 ms | 614 ms | 327 ms | 214 ms | 128 us | 67k/s |
| 30 min | 36000 | pure | 1.82 s | 1.89 s | 329 ms | 1.49 s | 126 us | 20k/s |
| 60 min | 72000 | numpy | 1.23 s | 1.28 s | 725 ms | 503 ms | 120 us | 59k/s |
| 60 min | 72000 | pure | 3.64 s | 3.82 s | 700 ms | 2.94 s | 157 us | 20k/s |

On the 30-minute period -- 36000 samples, the unit a flux tower is run in -- **numpy buys 3.4x overall and all of it in the compute stages (7x there)**. Reading is the same work either way, none of it through a kernel, so the 1% between the two read columns is run-to-run noise; on the numpy path reading is 60% of the run. Quote the throughput as 67k/s (numpy) and 20k/s (pure), no better than two significant figures.

## Per step

The nine steps of `pipeline.STEPS` on that 30-minute period, in the order they run, with the share of each path's own compute time. A step under a millisecond is at the resolution of this measurement: read it as free, not as three significant figures.

| step | numpy | share | pure | share | speedup |
|---|---:|---:|---:|---:|---:|
| cell | 3 us | 0% | 3 us | 0% | 1.3x |
| despike | 5 ms | 2% | 115 ms | 8% | 25.0x |
| rotate | 2 ms | 1% | 36 ms | 2% | 21.5x |
| lag | 18 ms | 8% | 989 ms | 66% | 54.7x |
| thermodynamics | 171 ms | 80% | 213 ms | 14% | 1.2x |
| detrend | 332 us | 0% | 17 ms | 1% | 51.9x |
| moments | 15 ms | 7% | 102 ms | 7% | 6.9x |
| assemble | 14 us | 0% | 22 us | 0% | 1.5x |
| wpl | 9 us | 0% | 10 us | 0% | 1.1x |
| spectral | 2 us | 0% | 2 us | 0% | 0.9x |
| qc | 784 us | 0% | 19 ms | 1% | 23.9x |

**Pure Python is dominated by `lag` (66% of its compute).** It is the one step whose work is samples x lags rather than samples: +-2 s at 20 Hz is 81 shifts, each a covariance over the whole period -- ~2.9 million multiply-adds per scalar, and there are two scalars. With numpy it costs 18 ms and the profile inverts: **`thermodynamics` is then the largest (80% of compute)**, and it is the step that does not speed up -- 1.2x, which is 1 plus noise. `thermodynamics` is an explicit per-sample Python loop in `flux.py` that calls no kernel, so `--pure` and the default run the same code through it.

## A real file, through the command line

`20213_RAW_20Hz_2025-09-08-2000.csv` -- TOA5, 35633 data rows cut into 2 periods, an open-path IRGASON whose gases arrive as molar densities, so WPL runs, which the synthetic file does not exercise. Timed as `python -m miniflux run CONFIG`, the whole process, because interpreter startup and imports are part of what a user waits for; and again in process, so the stages separate.

| path | command line, best | median | in process | read + parse | nine steps | samples/s |
|---|---:|---:|---:|---:|---:|---:|
| numpy | 860 ms | 897 ms | 592 ms | 377 ms | 214 ms | 60k/s |
| pure | 2.29 s | 2.39 s | 1.93 s | 394 ms | 1.53 s | 18k/s |

Startup -- `python -m miniflux --version`, which does every import a run does and then stops -- is 212 ms, or 25% of the numpy command line above, spent before the first byte is read. `--pure` still imports numpy: it is installed, and the flag only stops it being *used*. Throughput on the real file is 60k/s (numpy) and 18k/s (pure) against 67k/s and 20k/s on the synthetic one, so the synthetic file is a fair stand-in -- the only claim here, since the two differ in several ways at once (unread columns `csv.reader` still splits, gaps, two very unequal periods).

## Where the time goes

The numpy path is bound by the parser -- 60% of its run, already `csv.reader` plus `float()` plus a `datetime` per row, all C -- which leaves the file format as the ceiling. The pure path is bound by the lag window (66% of its compute), and narrowing that is configuration, not code: `co2_min_s`/`co2_max_s` at +-0.5 s is 21 shifts instead of 81. An incremental or FFT lag curve would need the numeric stack that path exists to avoid.

Continuous 20 Hz data at 48 periods a day costs 2.6 h of cpu a year on the numpy path and 8.9 h on the pure one, in process, plus 212 ms of startup for every command line. Neither is a reason to choose a path; the reason to choose is that one of them needs numpy.
