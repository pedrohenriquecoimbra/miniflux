# miniflux

miniflux is a minimal eddy-covariance flux program: raw high-frequency data in, one flux row
per averaging period out. It has no dependencies — the standard library computes every
number, and numpy, when it happens to be installed, is a speed switch inside a single module.
The whole program is 3,272 lines of Python across seventeen files, about 1,800 of them
statements and the rest comment and docstring.

```
20 Hz raw file(s)  ->  despike, rotate, lag, thermodynamics, detrend, moments, assemble,
                       WPL, QC  ->  one CSV row per 30 minutes (54 columns + units sidecar)
```

Two documents are the law, and the code is checked against them line by line:

* [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md) — the mathematics: every formula, constant,
  default, edge case and citation (1,194 lines).
* [`docs/CONTRACT.md`](docs/CONTRACT.md) — the plumbing: every module, signature, meta key,
  config key and refusal (1,063 lines).

Python >= 3.8. EUPL-1.2. 478 tests, `unittest` only.

## Why it exists

**To be forked.** Every step is a plain function `(period, cfg) -> period`, the pipeline is a
list of nine of them, and there is no registry, no plugin system, no provenance layer and no
units library to learn first. If you want to write your own flux software, or teach how one
works, this is a base whose whole behaviour fits in one person's head — and whose physics is
written down separately from its code, so you can disagree with a decision by reading it.

**To run where nothing can be installed.** A logger PC, an air-gapped field laptop, an old
cluster node with no wheels for its architecture: a copy of the `miniflux/` directory and a
Python interpreter is the entire install. That is enough for a preliminary flux while the
real processing happens elsewhere — a first look at yesterday's data, a sanity check on an
instrument that was serviced this morning.

**Because it can exist.** This is roughly the smallest thing that is still a real flux
program: it despikes, rotates, finds its own lags, does the thermodynamics per sample, and
applies WPL and Foken's quality tests. The Alpine of flux software. Nothing is stubbed to
make that sentence true — what is missing is listed below, with what it costs.

## What it computes

`pipeline.STEPS`, in order. The order is normative; each line in that file says what breaks
if it moves.

| # | step | what it does | source |
|---|---|---|---|
| 1 | `despike` | median/MAD test per variable per period, spikes become NaN and are never refilled | Mauder et al. (2013) |
| 2 | `rotate` | double rotation into the mean streamline; wind speed and direction from the unrotated means | Wilczak et al. (2001) |
| 3 | `lag` | slides each scalar against `w` and keeps the shift that maximises `abs(cov)`, with a nominal fallback when the peak lands on a window edge | EddyPro `CovMax`; Aubinet et al. (2012) |
| 4 | `thermodynamics` | per-sample moist-air state (e, es, RH, densities, cp, lambda), averaged over the period; air temperature measured where finite, else sonic-derived | ALGORITHMS §8; Schotanus et al. (1983) |
| 5 | `detrend` | block average (stores the mean, leaves the samples alone) or a least-squares line removed for real | Moncrieff et al. (2004); Rannik & Vesala (1999) |
| 6 | `moments` | variances and covariances on the joint finite mask, two passes, ddof = 1; `ustar` from the full stress vector | ALGORITHMS §7 |
| 7 | `assemble` | level-0 fluxes, the Schotanus heat flux in its exact division form, Obukhov length, z/L | Schotanus et al. (1983); van Dijk et al. (2004) Eq. 3.53 |
| 8 | `wpl` | density correction for a gas reported as a molar density; a dry mixing ratio is already conserved and gets none | Webb, Pearman & Leuning (1980), Eq. 24 and 25 |
| 9 | `qc` | steady-state test on one covariance pair, plus three integral turbulence deviations | Foken & Wichura (1996); Thomas & Foken (2002) |

Reading, timestamp parsing and the period cut happen before step 1, in `read.py`. Nothing is
carried between periods: no running means, no fitted planes, no tables.

## Quickstart

From a checkout, with nothing installed:

```sh
cd miniflux
python -m unittest discover -s tests -t .          # 478 tests, about 4 s
python examples/make_sample.py                     # writes ./sample_20hz.dat, 36000 samples
python -m miniflux check examples/miniflux.ini     # validate and echo the config, read no data
python -m miniflux run examples/miniflux.ini       # -> ./miniflux.csv + ./miniflux_units.csv
```

`make_sample.py` prints what it planted in the synthetic file, and the run has to reproduce
it. On the shipped seed:

```
PLANTED            -> COMPUTED
N_IN     36000        36000
N_SPIKE_U    3        3           (also V 2, CO2 4, and zero elsewhere)
CO2_LAG  +0.75 s      0.75        (15 samples at 20 Hz)
THETA    -7.035997    -7.036 deg
COV_W_TS 0.159552027  0.159552 K m s-1
```

The generator computes those predictions with its own forty lines of arithmetic rather than
by calling the program, so agreement means something.

`ITC_W`, `ITC_U` and `ITC_T` come out as `-9999` on that run, and the log says why: the
example config sits at `latitude = 0.0`, where the Coriolis parameter vanishes and the
neutral model would take `log(0)`. Write your real latitude.

Then a real file:

```sh
cp examples/miniflux.ini mine.ini
# edit [files], [timestamp], [variables], [units], [gases], [site] to describe your data
python -m miniflux check mine.ini                  # exits 2 and names the key if it is wrong
python -m miniflux run mine.ini --limit 1          # one period, as a smoke run
python -m miniflux run mine.ini
```

A Campbell TOA5 file needs `header_line = 2` and `first_data_line = 5`; the quoting and the
skipped preamble lines are handled. `--pure` forces the standard-library kernels for a run.
Exit codes are `0` success, `2` a config error, `3` an input error, `1` anything else, so a
wrapper script can tell "you wrote the config wrong" from "the data is not what the config
says it is" without parsing the log.

## Configuration

One `.ini` file, every key documented in [`examples/miniflux.ini`](examples/miniflux.ini) and
specified in `CONTRACT.md` §6. The dozen that matter:

```ini
[files]
input_glob      = ./*.dat        # matches are read in the order of their first sample
header_line     = 1              # 1-based; TOA5 is header 2, data 5
first_data_line = 2

[period]
averaging_minutes     = 30
acquisition_frequency = 20.0     # [Hz], authoritative; the observed rate is only logged
closed                = right    # which side of a boundary a sample on it belongs to

[timestamp]
column = TIMESTAMP
format = iso                     # iso | numeric — parsed, never inferred

[variables]                      # canonical name = exact column header; empty = absent
u = Ux
w = Uz
co2 = CO2
p_air = Press

[units]                          # the unit AS WRITTEN IN THE FILE
ts  = K                          # K | degC
co2 = ppm                        # ppm | umol/mol | mmol/m3 | mol/m3 | mg/m3 | g/m3

[gases]
co2_measure_type = mixing_ratio  # mixing_ratio (per mole of dry air) | molar_density

[site]
measurement_height = 3.0
latitude           = 45.0        # used only by the ITC test
```

**Nothing is guessed.** Where a wrong-looking-right number is the alternative, miniflux
refuses and names the key:

```
$ python -m miniflux check nop.ini
ERROR miniflux.cli: configuration: [variables] p_air and [site] pressure_pa are both unset:
miniflux never invents a pressure. Name the pressure column in [variables] p_air, or write a
constant in [site] pressure_pa.

$ python -m miniflux run numts.ini
ERROR miniflux.cli: input: ./sample_20hz.dat line 2: timestamp '2025-09-08 19:30:00.050000'
does not have the shape declared by [timestamp] format, which is YYYYMMDDHHMMSS[.ffffff];
miniflux parses timestamps and never infers their format
```

The same rule runs through the program: a closed-path analyser is refused rather than given
an ambient density correction, a wet mole fraction is refused rather than half-corrected, and
a flux that is owed WPL but did not get it is written as `-9999` rather than under the
corrected name. Everything else degenerate is NaN in the meta, `-9999` in the table, and a
`WARNING` in the log saying which input was not finite.

## Extending it

A step is a function `(period, cfg) -> period`. `period` is a plain dict: `'meta'`, `'t'`,
and nine `array('d')` series of equal length (`u v w ts co2 h2o ta p_air`), where missing is
always NaN. Adding one means writing the function and putting its name in `pipeline.STEPS`.
There is nothing else to update.

Say you want the skewness of the vertical wind, the first half of a Vickers & Mahrt
instrument test:

```python
# miniflux/skewness.py
"""Per-period skewness of the vertical wind, a high-frequency sanity diagnostic."""

import logging
import math

from . import kernels

logger = logging.getLogger(__name__)


def skewness(period, cfg):
    """Write meta['skew_w'], the skewness of w over its finite samples [-].

    period: the dict of CONTRACT section 1. cfg: the Config of section 5.
    Returns the same period. Never raises: a degenerate period gives NaN.
    """
    w = period['w']
    mean = kernels.nanmean(w)
    sd = kernels.nanstd(w)
    if not (math.isfinite(mean) and math.isfinite(sd)) or sd == 0.0:
        logger.warning('skew_w is NaN: mean=%r, sd=%r', mean, sd)
        period['meta']['skew_w'] = float('nan')
        return period
    total = 0.0
    n = 0
    for value in w:
        if math.isfinite(value):
            total += ((value - mean) / sd) ** 3
            n += 1
    period['meta']['skew_w'] = total / n if n else float('nan')
    return period
```

Two lines in `miniflux/pipeline.py`, after `moments` because it wants the despiked, rotated,
detrended `w`:

```python
from . import despike, detrend, flux, lag, qc, read, rotate, skewness, wpl
...
    ('moments', flux.moments),
    ('skewness', skewness.skewness),
```

and one line in `miniflux/write.py` to publish it, if you want it in the table:

```python
    ('SKEW_W', 'skew_w', '-'),
```

That is the whole change. Re-run the sample and the new column is there:

```
TIMESTAMP_END  VAR_W     SKEW_W     USTAR
202509082000   0.131553  0.0161974  0.32693
```

The rules a step has to keep (`CONTRACT.md` §1.1): it must not raise on physically degenerate
data — it writes `float('nan')` and logs at `WARNING`, so one bad period never costs the run
its other rows; it must not read a config section other than its own; and it must not import
another step module. Anything shared lives in `kernels.py` (the only module that knows about
array layout, NaN bookkeeping and numpy), `constants.py`, or the helpers in `flux.py`.

If you are adding physics rather than a diagnostic, write it into `ALGORITHMS.md` first, with
its formula and citation, and pin it with a test in the same commit. That is the convention
the repo is kept under, and it is the reason the code can be checked at all.

## What it deliberately does not do

`ALGORITHMS.md` §13 gives the full list with the cost of each. In short:

* **No spectral / frequency-response correction.** The largest known bias in a miniflux
  number: fluxes are underestimated by roughly 2–15 % depending on height, tube length and
  stability, and nothing here estimates it.
* **No planar fit, no triple rotation.** Double rotation over-rotates individual low-wind
  periods; a planar fit needs cross-period state that a one-period-at-a-time program does not
  have.
* **No closed-path handling.** A cell needs a per-sample conversion to a dry mixing ratio
  with fast cell temperature and pressure, not an ambient density correction. Closed-path
  input is refused rather than mis-corrected.
* **No gap filling, no u\* filtering, no night-time partitioning.** The output has holes in
  it, and annual sums cannot be computed from it without a downstream step.
* **No storage term.** Night-time NEE is too low at tall towers.
* **No footprint model.** Nothing in a row says how much of the flux came from the target
  ecosystem; screening by `WD`, `ZL` and `USTAR` is left to you.
* **One gas pair only** — CO2 and H2O. No CH4, no N2O, no multi-analyser stream assembly, no
  resampling.
* **No random-uncertainty estimate.** A miniflux flux has no error bar.

For work that has to stand up, use a real processing chain:
[ONEFlux_preproc](https://github.com/pedrohenriquecoimbra/ONEFlux-preproc) (the reference
miniflux was extracted from, which carries the spectral corrections, the provenance record
and the engine comparisons), **EddyPro** (LI-COR), or **GEddySoft** (B. Heinesch, ULiège —
Gembloux Agro-Bio Tech). miniflux is the small thing you read, fork, or run where those
cannot go.

## Speed

A 36,000-sample period takes a couple of seconds end to end on the pure-Python path and about
half that with numpy on the machine this was written on — seconds per period, not minutes;
measured numbers, on real files and both paths, are in
[`docs/BENCHMARK.md`](docs/BENCHMARK.md). numpy is a speed switch and nothing else: every
decision (spike mask, selected lag, QC flag) is bit-for-bit identical between the two paths
and reductions agree within 8 ULP, which the suite asserts at 1e-12 relative. On the shipped
sample, `--pure` and the numpy path write byte-identical tables.

## Validation

The suite pins every formula in `ALGORITHMS.md` against hand-computed test vectors, and the
eight end-to-end self-tests of §14 (rotation invariants, lag sign, WPL magnitude on a known
parcel, Schotanus contrast, numpy parity) are part of it. What has and has not been
compared against a reference implementation on real data is written up in
[`docs/VALIDATION.md`](docs/VALIDATION.md) — read it before you report a miniflux number.
Where miniflux and `ONEFlux_preproc` disagree numerically, `ONEFlux_preproc` is right and
miniflux has a bug.

## Licence and provenance

EUPL-1.2. See [`LICENCE`](LICENCE); copyright (c) 2026 Pedro Henrique Herig Coimbra.

miniflux is a minimal reimplementation of the flux path of `oneflux_preproc` v0.10.0, by the
same author, also under the EUPL-1.2. The physics, the constants and the numerical
conventions come from that reference implementation, which is itself a reconciliation of
EddyPro 7.0.9 (LI-COR) and GEddySoft v4.1 (B. Heinesch, ULiège — Gembloux Agro-Bio Tech). The
code here is written from scratch against the standard library, and the pipeline is a plain
ordered list of functions rather than a registry. `oneflux_preproc` carries the engine
comparisons, the provenance record, the spectral corrections and the published parity against
EddyPro and GEddySoft; miniflux carries none of that on purpose. Full method citations are in
[`NOTICE`](NOTICE) and §15 of `ALGORITHMS.md`.
