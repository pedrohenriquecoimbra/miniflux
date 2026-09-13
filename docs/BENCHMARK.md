# What miniflux costs, stage by stage

Measured by `scripts/benchmark.py`, which generates its own data and writes this file; re-run
it and every number here is replaced. Each configuration runs in a fresh subprocess
5 times, interleaved with the other path and idling first, so a session that slows
as it proceeds slows both and not whichever ran last. The tables quote the best repetition
with the median beside it; the per-stage breakdown is the single repetition with the best
total, so its parts add up to it.

| the machine | |
|---|---|
| measured | 2026-09-14 00:11 |
| cpu | 11th Gen Intel(R) Core(TM) i9-11950H @ 2.60GHz, 16 logical cores, one used |
| platform | Windows-11-10.0.26200-SP0 |
| python | CPython 3.14.3 |
| numpy | 2.5.1 |

## Where the time goes

One 30-minute period -- 36000 samples at 20 Hz, the unit a flux tower is run in --
read and parsed, pushed through the 11 steps of `pipeline.STEPS`, and written.
Sorted by cost on the numpy path; the pure path's own order is in its share column. Shares
are of that path's whole run. A stage under a millisecond is at the resolution of this
measurement: read it as free, not as three significant figures.

| stage | numpy | share | pure | share | pure/numpy |
|---|---:|---:|---:|---:|---:|
| thermodynamics | 170 ms | 46% | 193 ms | 12% | 1.1x |
| read + parse | 158 ms | 42% | 163 ms | 10% | 1.0x |
| lag | 18 ms | 5% | 986 ms | 60% | 54.9x |
| moments | 16 ms | 4% | 106 ms | 7% | 6.7x |
| despike | 5 ms | 1% | 112 ms | 7% | 21.2x |
| rotate | 1 ms | 0% | 35 ms | 2% | 30.1x |
| qc | 1 ms | 0% | 19 ms | 1% | 18.5x |
| detrend | 321 us | 0% | 17 ms | 1% | 52.8x |
| write | 111 us | 0% | 128 us | 0% | 1.2x |
| wpl | 24 us | 0% | 10 us | 0% | 0.4x |
| assemble | 18 us | 0% | 21 us | 0% | 1.2x |
| cell | 3 us | 0% | 2 us | 0% | 0.9x |
| spectral | 1 us | 0% | 1 us | 0% | 1.0x |

**Two stages are the whole numpy bill: thermodynamics and read + parse, 88% of the run between
them**, and neither is a kernel -- reading never was one, and `thermodynamics` is a scalar
loop numpy cannot reach. **On the pure path `lag` alone is 60%**, being the one
stage whose work grows with the number of lags as well as with the number of samples. The two
profiles have little in common, which is the useful thing here: optimising for one path is
not optimising for the other.

Read + parse is the same work on both paths at every length, so it doubles as this page's
noise floor: over the six runs of the next table it measures between 4.2 and
4.7 us a sample, a spread of 11% on an idle machine. Read no ratio
above as a result unless it is further from 1 than that.

**thermodynamics.** A per-sample scalar loop in `flux.py` that reaches a kernel exactly twice, in the two gas means at the end, so both paths run the same code through all 36000 samples of it and 1.1x is all numpy has to give. Per sample: about fifty floating-point operations, a dozen `_divide` calls and a 21-entry tuple folded into two dicts -- 432 thousand Python-level calls and 3.0 million dict lookups a period, on top of arithmetic that is genuinely owed. Two cuts that change no published number: unroll the nan-mean fold into named accumulators, losing the tuple and the lookups; and inline `_divide`'s zero test where the denominator is a pressure or a temperature. Vectorising the pass would be worth more and would mean a second implementation of ALGORITHMS section 8 -- the duplication this program exists not to have.

**read + parse.** `csv.reader` splits every row, `float()` converts the 7 declared columns and applies each one's affine unit map, and the timestamp now costs two dict lookups and an integer addition instead of a `datetime` per sample. All C or close to it, and the same code on both paths -- the 1.0x between them is this machine's noise and nothing else, which is what the spread quoted above is for. The floor is what it costs to turn 252 thousand ASCII numbers into doubles. Under it, in increasing order of what they cost to keep honest: read fewer columns (every field `csv.reader` splits and miniflux never names is paid for at full price); read a binary logger format; or give the numpy path a bulk parser of its own -- the fastest, and the one to refuse, since `read.py` is deliberately the single place a row is interpreted.

**lag.** The one stage whose work is samples x lags rather than samples: the configured window is 81 shifts, each a fresh covariance over the whole overlap, for each of two scalars -- 5.8 million sample pairs a period. Three cuts, cheapest first. (1) Narrow the window in the configuration: +-0.5 s is 21 shifts instead of 81, a 3.9x cut, and for a collocated open-path analyser it is the more defensible window anyway. (2) Hoist the finite masks out of the scan -- `kernels.cov_at_lag` rebuilds `isfinite(w) & isfinite(c)` at every lag although only the overlap window moves, so each mask could be built once per scalar and sliced: two of the nine array passes each lag makes on the numpy path, two `math.isfinite` calls per iteration on the pure one, result bit-identical. (3) A coarse-to-fine scan would cut most of the rest and is not an optimisation -- it changes which lag wins on a flat covariance curve, which is ALGORITHMS 5.4's business. An FFT cross-correlation needs the numeric stack the pure path exists to avoid.

**moments.** Ten reductions -- four variances, six covariances -- each two passes over 36000 samples, plus three joint-finite counts. The counts are the interesting part: `_joint_finite` in `flux.py` is a Python loop on **both** paths, 216 thousand `math.isfinite` calls a period, and it is why this stage speeds up 6.7x where the stages that live wholly inside kernels get ten to sixty. `kernels.cov` computes that count on its own first pass and throws it away; returning it, or giving `kernels` a joint-finite count with a numpy branch, is the cheapest real win on this page.

**despike.** Every configured series sorted once for the median and again for the MAD of the absolute deviations, then one pass for the mask: the 21.2x is `np.sort` against `sorted()` on a list. One cut is left and it is numpy-only -- `kernels.nanmedian` and `kernels.mad` sort a whole series to read one order statistic, where `np.partition` gives the same value in O(n) and still hands `_median_of_sorted` the two middle elements it needs. In pure Python the sort is the fastest thing the standard library has.

**The rest** -- `rotate`, `qc`, `detrend`, `write`, `wpl`, `assemble`, `cell`, `spectral` -- costs 3 ms together on the numpy path and 72 ms
on the pure one. Nothing there is worth an hour of anyone's time.

## One period, end to end

The same run at three period lengths; the lag window is fixed in seconds, so it is
81 shifts at every length and the work per sample does not change with it.

| period | samples | path | best | median | read + parse | steps | write | samples/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 min | 12000 | numpy | 119 ms | 122 ms | 50 ms | 69 ms | 103 us | 101k/s |
| 10 min | 12000 | pure | 486 ms | 503 ms | 52 ms | 434 ms | 113 us | 25k/s |
| 30 min | 36000 | numpy | 373 ms | 394 ms | 158 ms | 215 ms | 111 us | 97k/s |
| 30 min | 36000 | pure | 1.63 s | 1.67 s | 163 ms | 1.47 s | 128 us | 22k/s |
| 60 min | 72000 | numpy | 766 ms | 862 ms | 331 ms | 435 ms | 113 us | 94k/s |
| 60 min | 72000 | pure | 3.38 s | 3.47 s | 337 ms | 3.04 s | 125 us | 21k/s |

Both paths are linear enough in the period length to quote one throughput and ragged enough
that two significant figures would be generous: **101k/s to 94k/s with numpy,
25k/s to 21k/s without**, across the three lengths. The 30-minute row is the one
the rest of this page reads: **373 ms** a period and **1.63 s**.

## A real file, through the command line

`20213_RAW_20Hz_2025-09-08-2000.csv` -- TOA5, 35633 rows cut into 2 periods, an
open-path IRGASON whose gases arrive as molar densities, so WPL runs, which the synthetic
file does not exercise. Timed as `python -m miniflux run CONFIG`, the whole process, because
interpreter startup and imports are part of what a user waits for, and again in process, so
the stages separate.

| path | command line, best | median | in process | read + parse | steps |
|---|---:|---:|---:|---:|---:|
| numpy | 572 ms | 589 ms | 391 ms | 183 ms | 208 ms |
| pure | 1.67 s | 1.73 s | 1.51 s | 176 ms | 1.33 s |

Startup -- `python -m miniflux --version`, which does every import a run does and then stops
-- is 181 ms, 32% of the numpy command line, spent before the first
byte is read. In process the real file runs at 91k/s and 24k/s
against 97k/s and 22k/s on the synthetic one, so the synthetic file is a
fair stand-in: the only claim made here, since the two differ in several ways at once (unread
columns `csv.reader` still splits, gaps, two unequal periods).

## What numpy buys, and what a year costs

**numpy buys 4.4x overall on the 30-minute period**, 7x across the
steps alone. It is not more because 88% of the numpy run is already
outside every kernel: read + parse is 42% of it and `thermodynamics` another
46%, and neither calls a kernel that could be made faster. Amdahl, measured
rather than quoted: an infinitely fast kernel layer would still leave 329 ms of this
period standing, so **5.0x is the whole remaining ceiling** against the
4.4x already collected -- and it is not in the kernels at all. It is the parser and
one scalar loop, both named above.

Continuous 20 Hz data, 48 periods a day, one core, in process:

| path | one period | one day | one month | one year |
|---|---:|---:|---:|---:|
| numpy | 373 ms | 18 s | 9 min | 1.8 h |
| pure | 1.63 s | 1 min | 39 min | 8.0 h |

A tower-year is 1.8 h of cpu with numpy and 8.0 h without, plus 181 ms of startup per command line,
against the 365 days of data it covers. Neither is a reason to choose a path; the reason
to choose is that one of them needs numpy.

**On a single-board computer.** Not measured here. Taking one Raspberry-Pi-class core as
**4x slower than the core above, which is an ESTIMATE and not a measurement**, a
30-minute period would cost about 1.49 s with numpy and 6.54 s without against
the 1800 s of data it represents -- 0.1% and 0.4% of
real time on one core. Such a board keeps up with a live 20 Hz tower on either path and
reprocesses a year of archive in about 7.3 h or 1.3 days. That factor
is the only guess in this document: measure it on the board before relying on it.

miniflux targets Python 3.8 and this is CPython 3.14.3. A 3.8 interpreter has neither the
specialising interpreter nor the faster call sequence, so the pure-Python numbers above are
the optimistic end of what 3.8 would show; the numpy numbers, which spend their time inside
numpy, would barely move.
