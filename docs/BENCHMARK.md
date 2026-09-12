# What miniflux costs

Measured by `scripts/benchmark.py`, which generates its own data and writes this file. Re-run it and every number here is replaced.

    python scripts/benchmark.py

Each configuration runs in a **fresh subprocess** and is repeated 5 times; the tables quote the **best** repetition, with the median beside it where it fits. Best, not mean: the slow repetitions are the machine doing something else, and averaging them in measures the machine. The per-step breakdown is taken from the single repetition with the best total, so its parts add up to its total.

The two paths are **interleaved**: repetition *i* of numpy runs before repetition *i+1* of pure. A laptop gets slower as a benchmark proceeds, and running one path to completion and then the other would charge that drift to whichever went last. Each timed run also idles first, and probes the cpu clock immediately before and after itself, so a row that was throttled while it ran says so.

## The machine

| | |
|---|---|
| measured | 2026-09-12 23:58 |
| cpu | 11th Gen Intel(R) Core(TM) i9-11950H @ 2.60GHz (16 logical cores, one used) |
| platform | Windows-11-10.0.26200-SP0 |
| python | CPython 3.14.3 |
| numpy | 2.5.1 |

miniflux targets Python 3.8 and this is CPython 3.14.3. A 3.8 interpreter has neither the specialising interpreter nor the faster call sequence, so the pure-Python numbers below are the optimistic end of what 3.8 would show; the numpy numbers, which spend their time inside numpy, would barely move.

**Thermal note, and it is not a footnote.** This benchmark was written on a laptop whose cpu, under sustained single-core load, was seen running the same fixed float loop more than three times slower than when cold -- a bigger factor than numpy buys, and enough to swamp the comparison it is measuring. So every timed run probes that loop immediately before and immediately after itself, and reports the ratio.

On this run 1 measurement(s) slowed while running -- 60 min pure 3.3x -- and they are marked **(!)** below. Those rows are upper bounds on the time, not measurements of the code; on a machine with a steady clock, expect less.

## The headline

One 30-minute period of 20 Hz data -- 36000 samples, the unit a flux tower is run in -- read from disk, pushed through all nine steps and written out:

| | numpy | pure Python | ratio |
|---|---:|---:|---:|
| total per period | 600 ms | 1.85 s | 3.1x |
| read + parse | 348 ms | 376 ms | 1.08x |
| the nine steps | 252 ms | 1.48 s | 5.9x |
| write the row | 120 us | 125 us | |
| throughput | 60k/s | 19k/s | |

**numpy buys 3.1x overall, and it buys all of it in the compute stages (6x there).** Reading is the same work either way -- `csv.reader`, a timestamp per row, nine `float()` calls per row -- and on the numpy path it is now 58% of the run.

## Per step

The nine steps of `pipeline.STEPS`, on the same 36000-sample period, in the order they run. Seconds, and the share of that path's own compute time.

| step | numpy | share | pure | share | speedup |
|---|---:|---:|---:|---:|---:|
| despike | 5 ms | 2% | 107 ms | 7% | 21x |
| rotate | 1 ms | 0% | 38 ms | 3% | 31x |
| lag | 20 ms | 8% | 1.01 s | 68% | 51x |
| thermodynamics | 206 ms | 82% | 181 ms | 12% | 0.9x |
| detrend | 353 us | 0% | 20 ms | 1% | 56x |
| moments | 15 ms | 6% | 97 ms | 7% | 6.5x |
| assemble | 15 us | 0% | 17 us | 0% | 1.1x |
| wpl | 9 us | 0% | 12 us | 0% | 1.3x |
| qc | 848 us | 0% | 21 ms | 1% | 24x |

A step costing under a millisecond is at the resolution of this measurement; read `assemble`, `wpl` and `detrend` as "free", not as three significant figures.

**Pure Python is dominated by `lag` (68% of its compute time).** It is the one step whose work is samples x lags rather than samples: the example config searches +-2 s, which at 20 Hz is 81 shifts, and every shift is a covariance over the whole period. That is ~2.9 million multiply-adds per scalar per period, and there are two scalars.

**With numpy the same step costs 20 ms and the profile inverts: `thermodynamics` is now the largest (82% of compute).** `thermodynamics` is the step that does not speed up (0.9x): it is an explicit per-sample Python loop in `flux.py` -- two passes of moist-air state, a saturation vapour pressure and a specific heat per sample -- and it calls no kernel at all, so the numpy switch never reaches it. Every other step hands whole series to `kernels.py`, which is where numpy lives.

`assemble` and `wpl` are scalar arithmetic on numbers the earlier steps already reduced, so they are free on both paths and always will be.

## Period length

20 Hz throughout, one period per run, the lag window fixed at +-2 s (81 shifts) so the work per sample is the same at every length.

| period | samples | numpy best | median | samples/s | pure best | median | samples/s | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 min | 12000 | 196 ms | 216 ms | 61k/s | 627 ms | 666 ms | 19k/s | 3.2x |
| 30 min | 36000 | 600 ms | 648 ms | 60k/s | 1.85 s | 2.05 s | 19k/s | 3.1x |
| 60 min | 72000 | 1.23 s | 1.30 s | 59k/s | 5.97 s (!) | 6.16 s | 12k/s | 4.9x |

Nothing in the pipeline is quadratic in the period: the lag window is set in seconds, so a longer period does not widen the search, and the per-sample cost should be flat. Across the rows that were not throttled it varies by 5% (numpy) and 1% (pure). Quote the throughput as 60k/s (numpy) and 19k/s (pure), and do not expect better than two significant figures from it.

The rows marked (!) (60 min pure) ran longer than this machine holds its clock, so they are slower than the code is. Scaling the clean 30-minute pure row by sample count predicts 3.71 s for 60 minutes; the measurement says 5.97 s. The difference is the cpu, not the pipeline -- on a machine with a steady clock, expect the linear number.

## A real file, through the command line

`20213_RAW_20Hz_2025-09-08-2000.csv`

TOA5, 35633 data rows cut into 2 periods, an open-path IRGASON: both gases arrive as molar densities (mg m-3, g m-3), so WPL runs -- which the synthetic file, in mixing ratios, does not exercise. Timed as `python -m miniflux run CONFIG`: the whole process, interpreter startup and imports included, because that is what a user waits for.

| | best | median | of it, startup | the rest |
|---|---:|---:|---:|---:|
| numpy | 889 ms | 927 ms | 217 ms | 672 ms |
| pure (`--pure`) | 2.17 s | 2.20 s | 217 ms | 1.95 s |

(The command line cannot probe its own clock, so these two rows carry no (!) either way. Their in-process twins below can, and did not.)

Startup -- `python -m miniflux --version`, which does every import a run does and then stops -- is 217 ms, against 55 ms for `python -c pass` and 184 ms for `python -c "import numpy"`. So 24% of the numpy run and 10% of the pure one is spent before the first byte is read, and most of that is numpy. `--pure` still imports numpy, because it is installed and the flag only stops it being *used*; on a machine with no numpy at all the pure run would be about 129 ms shorter.

The same file measured in process, so the stages separate:

| | total | read + parse | nine steps | samples/s |
|---|---:|---:|---:|---:|
| numpy | 599 ms | 362 ms | 237 ms | 60k/s |
| pure | 1.88 s | 371 ms | 1.51 s | 19k/s |

Throughput on real data is 60k/s (numpy) and 19k/s (pure), against 60k/s and 19k/s on the synthetic file: 1% and 2% apart. The synthetic file is therefore a fair stand-in, which is the only claim this comparison is making -- the two files differ in more than one way at once (the real one carries 7 columns nobody reads but `csv.reader` still splits, holds gaps, and is cut into two periods of very different length), so the small difference is not attributable to any one of them.

## What a year costs

Continuous 20 Hz data, 48 periods a day, from the per-period totals above. These are in-process figures: one command line per period would add 217 ms of startup each time, which is 63.2 min a year.

| | numpy | pure Python |
|---|---:|---:|
| one 30-min period | 0.6 s | 1.9 s |
| one day (48 periods) | 28.8 s | 89.0 s |
| one month (30 days) | 14.4 min | 44.5 min |
| one year (365 days) | 2.9 h | 9.0 h |

| | numpy | pure Python |
|---|---:|---:|
| real time processed per second of cpu | 50.0 min | 16.2 min |
| duty cycle to keep up live | 0.0333% | 0.103% |

A year of 20 Hz data is 2.9 h on the numpy path and 9.0 h on the pure one. Neither is a reason to choose a path. The reason to choose is that one of them needs numpy.

## On a single-board computer

Not measured. The extrapolation below assumes a factor, and the factor is a guess from published CPython benchmarks, not something this script ran:

| assumed core | factor | one period, numpy | one period, pure | live duty cycle, pure |
|---|---:|---:|---:|---:|
| a Raspberry Pi 5 class core (Cortex-A76, 2.4 GHz) | 4x slower | 2.40 s | 7.42 s | 0.4% |
| a Raspberry Pi 4 class core (Cortex-A72, 1.8 GHz) | 10x slower | 6.00 s | 18.55 s | 1.0% |

**Yes, comfortably** -- with the caveat that these are estimates. Even at the pessimistic 10x, one 30-minute period costs 18.55 s on the pure path, which is 1.0% of the 1800 s of real time it covers; the board spends 99% of its life idle and keeps up live with no numpy, no compiler and no wheels. Memory is the easier question: one period is nine float series of 36000 samples, 2.6 MB of `array('d')` plus the timestamps, and only one period is ever in memory.

What would make that estimate wrong: a board with much less memory bandwidth, a longer lag window (the search is samples x lags, and the window is in seconds, so doubling it doubles the largest pure-path step), or a 32-bit build where every float is boxed differently.

## Where the remaining time goes

The read stage of the 36000-sample period, split:

| | seconds | share of read |
|---|---:|---:|
| `csv.reader` tokenising | 33 ms | 9% |
| timestamp + 9 floats per row, into the period | 344 ms | 91% |
| all of read | 376 ms | 100% |

(Measured in its own process, with nothing else running between the rows, so the total is a few per cent under the 348 ms the same stage takes inside a full run. The split is what matters here, not the absolute.)

On the **numpy path**, 58% of the run is read+parse and 82% of what is left is `thermodynamics`. Neither touches a kernel. Making this path faster means making the parser faster -- and the parser is `csv.reader` plus `float()` plus a `datetime` per row, all of which are already C. The honest ceiling is the file format: a text CSV of 36000 rows has to be tokenised 36000 times whatever reads it.

On the **pure path**, 68% of compute is the lag search. Three things would move it, in order of how much they cost in complexity:

1. **Narrow the window.** `co2_min_s`/`co2_max_s` are configuration, not physics. +-0.5 s instead of +-2 s is 21 shifts instead of 81 and cuts the step by about 4x. This costs nothing and needs no code.
2. **Compute the lag curve incrementally.** Each shift currently sums its own products over the whole period; consecutive shifts share all but a few terms. A running update, or an FFT cross-correlation, turns samples x lags into samples x log(samples) -- but an FFT needs a complex numeric stack, which is exactly what the pure path exists to avoid.
3. **Give `thermodynamics` a fast path.** It is 12% of pure compute and does not speed up with numpy either, so it is the second target on both paths. Most of its cost is `saturation_vapour_pressure` and the moist-air state being rebuilt twice per sample; the second pass exists because Ta has to be resolved before the state can be built at it.

None of these are worth doing for speed alone at 1.85 s a period. They are worth knowing about if the window ever widens or the sample rate ever goes to 100 Hz.
