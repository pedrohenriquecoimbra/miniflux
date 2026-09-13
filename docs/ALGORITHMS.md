# miniflux — ALGORITHMS

The scientific reference for `miniflux`. Every formula the program implements is written
here in plain ASCII mathematics, with its symbols, units, exact numerical literals,
defaults, edge cases and citation.

This document is normative. If the code and this document disagree, one of them is a bug;
fix the disagreement, do not paper over it. A reader should be able to check the code
line by line against these formulas, and a porter to another language should need nothing
except this file and `CONTRACT.md`.

Provenance: every formula is transcribed from ONEFlux_preproc v0.10
(`D:/_gitRepo/ONEFlux_preproc`), which is itself a reconciliation of EddyPro 7.0.9
(LI-COR) and GEddySoft v4.1 (B. Heinesch, ULiège–Gembloux Agro-Bio Tech). Where miniflux
departs from that parent, the departure is marked **DIVERGENCE** and justified on the spot.

### How to check these numbers

Every check value printed in this document — `es(293.15 K)`, the reference state of §8.4,
the midday parcel of §10.8, the Schotanus contrast of §14.7 — is **reproducible from the
formulas above it, with the constants of §1 and nothing else**. Save the single exception
named at the end of this section, none of them is a measurement, a remembered figure or a
value carried over from another program; a reader with a calculator must be able to land
on each one. Two consequences, both binding:

* If a printed value and the formula above it disagree, the printed value is the bug.
  Recompute it and restamp it; do not adjust the formula to meet it.
* If the formula and *physical reality* disagree — a fit that is a known fraction of a
  percent off the true quantity — that is a **code** problem, not a stamping problem. It
  is called out where it occurs (§8.3) and left in place until the code changes with it.

`tests/` pins these values, and each test carries the arithmetic in a comment:
`tests/test_flux.py` holds §8.3's three temperature conventions, the §8.4 pieces (`es`,
`cp_d`, `cp_v`, `lambda_v`) and §14.7's Schotanus contrast; `tests/test_wpl.py` holds
§10.8 and §14.6, in both of the parcel states §10.8 discusses; `tests/test_cell.py` holds
§2A's round trip and its per-sample property; `tests/test_spectral.py` holds §10A.2's two
check values and both stability branches. A restamp here that the suite does not already
agree with is a restamp that has not been checked.

The values printed here that are **not** of this kind each say so in place, and they are:

* §9.3's `H_L0 / H_EddyPro = 1.0387`, *measured* on the bundled sample against another
  program;
* §2A.8's agreement between the two closed-path declarations, and §10A.1's 10.2 %
  attenuation, both *measured* on the bundled FR-Gri day — the first against the
  instrument's own arithmetic, the second against EddyPro through the parent;
* §10A.4's `tau` table, which is §10A.2 evaluated over that day's 48 wind speeds and so is
  reproducible only with the data in hand.

Everything else stands on the formula above it.

---

## 0. Conventions

### 0.1 The averaging period

The program works one *averaging period* at a time. A period is 30 minutes by default.
Nothing is carried between periods: no running means, no fitted planes, no tables.

    period_end   = ceil(t, A)        (closed = right, the default)
    period_start = period_end - A
    offset(t)    = t - period_start                       [s], in (0, A]

    period_start = floor(t, A)       (closed = left)
    period_end   = period_start + A
    offset(t)    = t - period_start                       [s], in [0, A)

where `A` is the averaging interval [s] (1800 by default) and `t` is the sample clock.
A period is *labelled* by its end instant in both conventions.

`closed` is not cosmetic. A logger that stamps the whole second (`...:00.00`) and then
19 more samples with fractions is a left-closed stream; a logger whose first sample of a
half hour is `...:00.05` and whose last is exactly `...:30:00.00` is a right-closed
stream. Choosing the wrong side displaces the whole period by one sample, which no
per-period statistic can reveal.

### 0.2 Missing data

Missing is `NaN` (IEEE 754 quiet not-a-number). There is no numeric sentinel inside the
program; `-9999` exists only in the input tokens that are read *as* NaN and in the output
column that is written *from* NaN.

Every statistic is NaN-aware by construction:

* a mean, median or variance skips non-finite samples;
* a covariance uses the **joint** finite mask of its two series;
* a comparison with NaN is False, so NaN is never flagged as a spike and never selected
  as an extremum.

Series length is preserved everywhere. Nothing is ever compacted, interpolated or refilled.

### 0.3 Units

SI internally, with three carried exceptions that match the instruments and the FLUXNET
output convention:

| quantity | internal unit |
|---|---|
| temperature | K |
| pressure | Pa |
| gas reported as a dry mixing ratio (CO2, H2O) | mol mol-1 of dry air (a 420 ppm signal is `4.20e-4`) |
| gas reported as a molar density (CO2, H2O) | mol m-3 |
| published CO2 flux | umol m-2 s-1 (scaled at write time) |
| published E | g m-2 s-1 (scaled at write time) |

A gas series carries the unit implied by how it was measured, and only that: `ppm`, `ppt`,
`mmol/m3`, `mg/m3` and `g/m3` are *input* units, converted once by the reader. No scale factor
survives past the reader, and the published scalings (`1e6` for a molar flux, `1e3` for E)
are applied once, at write time.

Conversion from the file's own units happens once, at read time (`read.py`), as an affine
map `x_canonical = a * x_file + b`. Everything downstream sees canonical units only.

Note on despiking and units: the MAD test is equivariant under any affine change of unit
(median and MAD both scale by `|a|`, the offset `b` cancels), so converting units before
despiking rather than after — as the parent does — leaves the spike flags unchanged.

### 0.4 Estimator conventions, fixed once

* **ddof = 1** for every variance and covariance in the program. At N = 36000 the
  difference from `/N` is 2.8e-5 relative, larger than several published agreement claims.
* **Two-pass covariance** (subtract the means, then accumulate products) for every
  reported second moment. The one-pass raw-sums form cancels catastrophically on a series
  with a large mean and a small fluctuation — exactly the CO2 case, 400 ppm mean over a
  1 ppm fluctuation.
* **Joint mask**: `cov(x,y)` uses the samples where *both* x and y are finite, and takes
  both means over that same set.
* **Ties** in any argmax/argmin go to the lowest index. Scan ascending, compare strictly.
* **Summation** is naive left-to-right, not compensated. This is what a numpy fast path
  can be made to agree with to within a few ULP; a Kahan sum could not.

---

## 1. Physical constants

Exact literals. These are not free choices; several are deliberately *not* the value you
would derive from the others, and the derivations are given so the difference is visible.

| symbol | value | unit | meaning / source |
|---|---|---|---|
| `R`    | 8.314462618 | J mol-1 K-1 | universal gas constant, CODATA 2018 (exact since the 2019 SI) |
| `Rd`   | 287.04      | J kg-1 K-1  | specific gas constant, dry air — **EddyPro's rounded value**, not `R/Md` = 287.0025 (+0.0131 %) |
| `Rv`   | 461.5       | J kg-1 K-1  | specific gas constant, water vapour — **EddyPro's rounded value**, not `R/Mv` = 461.4019 (+0.0213 %) |
| `Md`   | 0.02897     | kg mol-1    | molar mass, dry air |
| `Mv`   | 0.01802     | kg mol-1    | molar mass, water vapour |
| `Mco2` | 0.04401     | kg mol-1    | molar mass, CO2 |
| `mu`   | `Md / Mv` = 1.6076581576026636 | - | computed, never written as a literal |
| `Cpd`  | 1005        | J kg-1 K-1  | **intercept of the cp_d(T) fit**, paired with the `(Tc+23.12)^2/3364` term. Substituting 1004.67 from a textbook leaves cp_d low and H with it |
| `g`    | 9.81        | m s-2       | gravitational acceleration (not 9.80665) |
| `kappa`| 0.4         | -           | von Karman constant. EddyPro's raw-processing L uses 0.41; miniflux keeps 0.40 |
| `P0a`  | 100000      | Pa          | reference pressure for potential temperature |
| `MAD_SCALE` | 0.6745 | -           | 0.75 quantile of the standard normal; the code **divides** by it (not `*1.4826`) |
| `T0`   | 273.15      | K           | degC -> K offset |

Inline fit coefficients (each belongs to its formula and may not be mixed with another
parameterisation):

* `cp_d`:      `1005`, `23.12`, `3364`
* `cp_v`:      `1859`, `0.13`, `0.193`, `5.6e-3`, `1e-3`, `5e-5`
* `lambda_v`:  `1e3`, `3147.5`, `2.37`
* `es`:        `77.345`, `0.0057`, `7235`, `8.2`
* sonic -> air temperature: `0.32`
* Schotanus / virtual temperature: `0.51`
* Poisson exponent: `0.286`
* canopy displacement ratio: `2/3`
* `(z-d)` floor: `1e-4` m

---

## 2. Stage 1 — reading, timestamp parsing, period cut

Not a scientific method (the parent registers it as "a file format, not a method"), but
three of its rules change the numbers, so they are specified here.

### 2.1 File cut

Configuration gives two 1-based line numbers: `header_line` (the row holding the column
names) and `first_data_line`. Lines before `first_data_line` other than `header_line` are
discarded. A Campbell TOA5 file is `header_line = 2`, `first_data_line = 5`; a plain CSV
with one header row is `1, 2`.

### 2.2 Timestamp parsing — parse, never infer

Two shapes are supported and each is parsed by explicit field extraction:

    ISO      "YYYY-MM-DD HH:MM:SS"  or  "YYYY-MM-DD HH:MM:SS.ffffff"   (also 'T' separator)
    NUMERIC  "YYYYMMDDHHMMSS"       or  "YYYYMMDDHHMMSS.ff"

Anything else is a hard error naming the file and line number.

**DIVERGENCE (a defect miniflux refuses to inherit).** The parent calls
`pandas.to_datetime(col, errors='coerce')` as a fallback. On pandas >= 2.0 that infers a
single format from the first element; a TOA5 stream writes the whole second without a
fractional part and every other sample with one, so 19 of every 20 samples at 20 Hz are
silently coerced to `NaT`. Measured on the parent with the FR-Jus file: 33833 of 35633
timestamps lost. Format inference on a high-frequency timestamp column is never safe.

A numeric timestamp must be read as a **string**, not through float64: `20220512233000.05`
carries 16 significant digits and a double carries ~15.95.

### 2.3 Period assignment

`ceil` / `floor` per §0.1 on the parsed instant. A sample landing exactly on a boundary
belongs to the *earlier* period under `closed = right` and to the *later* one under
`closed = left`.

The instant and the interval `A` are both carried as **whole microseconds since
1970-01-01**, as exact Python ints: `floor` is then one integer remainder and "exactly on a
boundary" is an exact equality, not a comparison of two rounded quantities. The two
boundaries of a period become datetimes when it closes; the samples themselves never do,
and what leaves this stage carries no per-sample time at all (CONTRACT §1.2).

File boundaries are irrelevant: a logger file may straddle a period boundary (the FR-Jus
TOA5 file spans 19:32:00.00 to 20:01:59.95). Periods are cut on the sample clock after the
input files have been concatenated in timestamp order.

Duplicate timestamps within a period: keep the first, count the rest, report the count.
Do not drop the period.

### 2.4 What is refused rather than guessed

* a timestamp matching neither shape;
* a canonical variable declared in the config whose column is absent from the header;
* a required canonical variable (u, v, w, ts, co2, h2o, p_air) with no source at all.

In particular, **miniflux never fabricates an air pressure**. The parent substitutes
99767.5 Pa for a missing pressure column and then produces finite WPL numbers from an
invented pressure. A constant pressure may be used, but only if the user writes the number
into the configuration themselves, and it is echoed in the log.

---

## 2A. Stage 1b — the closed-path cell conversion (Ibrom et al. 2007)

**Citation.** Ibrom, A., Dellwik, E., Larsen, S. E., Pilegaard, K. (2007). On the use of
the Webb-Pearman-Leuning theory for closed-path eddy correlation measurements. *Tellus B*
59, 937-946.

Lettered rather than numbered because it is a stage not every run has, and renumbering
§§3-15 would break every `ALGORITHMS §N` reference in the code. It runs **first**, before
despiking, and is a no-op for every open-path run.

### 2A.1 What a closed-path analyser actually measures

An LI-7200 does not measure the air at the tower. It draws air down a tube into a cell the
instrument has warmed and the pump has dropped in pressure. Measured over the 48 half hours
of the bundled FR-Gri day (medians, `T_CELL` / `P_CELL` against `TA` / `PA`):

    cell      293.2 K    99.29 kPa
    ambient   290.1 K   100.65 kPa        cell is +3.5 K (up to +7.6) and -1.33 kPa

Everything the analyser reports is a property of *that* air:

| what it reports | what the number is | conserved? |
|---|---|---|
| `CO2_DRY`, `H2O_DRY` | mixing ratio per mole of **dry** air, in the cell | **yes** |
| `CO2`, `H2O` | mole fraction of **moist** cell air | no (dilution) |
| `CO2_CONC`, `H2O_CONC` | molar density **in the cell** | no (expansion + dilution) |
| `CO2_MASS`, `H2O_MASS` | mass density in the cell | no (same) |

A dry mixing ratio is conserved along the tube: warming the sample, dropping its pressure
and letting water in or out changes none of it, because it is counted per mole of dry air
and the dry air is what came in. That is the whole reason analysers report one.

### 2A.2 The clean case: declare the dry mixing ratio

**If the file carries `CO2_DRY` / `H2O_DRY`, declare those, and this stage does nothing at
all.** No cell state is read, no conversion is done, no density correction is owed, and
the answer is exact rather than reconstructed. This is the recommended way to run a
closed-path analyser, and `examples/fr_gri_closedpath_dry.ini` is it.

### 2A.3 The conversion, when only the density is in the file

    V_cell  = R * T_cell / P_cell                      [m3 mol-1]
    chi_h2o = n_h2o * V_cell                           [mol mol-1 of moist cell air]
    r_gas   = n_gas * V_cell / (1 - chi_h2o)           [mol mol-1 of DRY air]

`n_gas * V_cell` is the sample's mole fraction; dividing by `1 - chi_h2o` re-expresses it
per mole of dry air. When the water is itself reported as a dry mixing ratio `r_h2o`
(reading an LI-7200 as `CO2_CONC` + `H2O_DRY`), the divisor is built from that instead,
`chi_h2o = r_h2o / (1 + r_h2o)`; it is a *moist* fraction either way. Applied to the water
itself the formula reduces to `r_h2o = chi_h2o / (1 - chi_h2o)`, which is the same
statement.

Dropping the `1 - chi_h2o` leaves a moist mole fraction wearing a dry name: on the FR-Gri
parcel that is 1 %, which is the size of a whole day's CO2 signal.

**Per sample, never on the means.** It is precisely the fluctuations of `T_cell`, `P_cell`
and `chi_h2o` that carry the spurious part of the density signal; the same conversion
applied to a block-averaged period removes exactly none of it and produces a correction
indistinguishable from no correction. The two effects are coupled — a warmer sample has a
larger `V_cell`, which raises the numerator *and* `chi_h2o` in the denominator — so the
per-sample ratio is not simply proportional to `T_cell`.

### 2A.4 Cell state or nothing

`T_cell` and `P_cell` must be declared, and **the ambient ones are not a fallback**. That
is a refusal (CONTRACT 5.2.12), not a warning, and after the missing pressure it is the
strongest one in the program. The difference is not a matter of degree: the ambient state
describes air that has not been through the tube or the pump. On the FR-Gri day the two
substitutions push the same way — the cell is warmer *and* at lower pressure — so

    V_ambient / V_cell  =  (Ta/T_cell) * (P_cell/Pa)

is **2.6 % from 1 at the median and 4.0 % at the worst half hour**, and that lands whole on
every gas flux of that period. It would look entirely right.

### 2A.5 Why no ambient WPL follows, and why that is the answer

Once the gas is a dry mixing ratio it is conserved, so there is nothing for Webb's equation
to remove. That is the *correct* treatment and not a missing feature, and it is worth
stating flatly because it is the part a reader assumes is a bug:

> **Ambient WPL is never applied to a closed-path gas.**

Webb, Pearman & Leuning derive the correction for the expansion and dilution of *ambient*
air at the sampling point. The temperature and humidity fluctuations inside a cell at the
end of a tube are not those fluctuations — the tube and the cell have damped some of them
and imposed others — so applying the ambient correction to a cell quantity is wrong **in
kind**, not merely in magnitude. It would describe air that never existed.

The magnitude, measured on the synthetic parcel of `tests/test_cell.py`: declare the same
cell densities as open-path ambient ones, let WPL run, and FC moves by **-15.3 %** and LE
by **+0.8 %**. Both are entirely plausible-looking numbers, which is the point.

Nothing in `wpl.py` or `flux.py` tests for a cell. `config.py` resolves a converted gas's
**effective** `measure_type` to `mixing_ratio` once, at load time, and the flux factor
(§9.1), the mean densities (§8.2) and "is a density correction owed?" (§10.1) all read
that one word. A conversion therefore retires the correction by itself.

### 2A.6 Where it sits, and why before despiking

`cell` is the only step allowed to run before `despike`, for two reasons:

* the MAD threshold should see the conserved quantity. A cell density carries the cell's
  own temperature and pressure fluctuations, which are not gas signal and widen the
  distribution the threshold is calibrated against;
* a spike in `T_cell` is screened **nowhere else** in miniflux. Converting first turns it
  into a spike in the gas, where the MAD test catches it.

Everything else stays where §14 puts it: the conversion changes the values, not the shape,
so the lag search, the covariances and the means are unaffected in kind.

### 2A.7 What this does *not* fix

Nothing here touches the frequency response. A closed-path flux is still attenuated in the
tube and is still an **underestimate** — see §10A, and the warning the run logs once.

### 2A.8 Verification against the instrument

Over the 36000 samples of one FR-Gri half hour, converting `CO2_CONC` / `H2O_CONC` through
§2A.3 and comparing with the `CO2_DRY` / `H2O_DRY` the same analyser wrote:

    CO2_DRY   converted/reported - 1 :  mean +0.0194 %  sd 0.0239 %  range -0.10 .. +0.12 %
    H2O_DRY   converted/reported - 1 :  mean +0.0194 %  sd 0.0239 %  range -0.10 .. +0.12 %

The two agree to five decimal places *with each other*, which puts the whole residual in
`V_cell = R T_cell / P_cell` — the analyser's internal cell state against the one it logs
— and none of it in the dilution algebra. Carried through the pipeline over the 48 half
hours of 2022-05-14, the CO2 flux from the two declarations agrees to **+0.033 % on
average (median +0.024 %, worst 0.49 %)** across the 43 periods with
`|FC| >= 1 µmol m-2 s-1`; LE and E to **+0.015 %** (worst 0.44 %). On synthetic data,
where the two shapes are built from one another exactly, the agreement is **2e-16
relative** — floating-point rounding and nothing else.

---

## 3. Stage 2 — despiking: Mauder et al. (2013) MAD test

**Citation.** Mauder, M., Cuntz, M., Drüe, C., Graf, A., Rebmann, C., Schmid, H. P.,
Schmidt, M., Steinbrecher, R. (2013). A strategy for quality and uncertainty assessment of
long-term eddy-covariance measurements. *Agric. For. Meteorol.* 169, 122-135.
Parent registration name: `mad_perperiod`. EddyPro selects it with
`RawProcess_ParameterSettings.despike_vm = 1`.

Applied **per averaging period and per variable, independently**, on the raw series before
rotation, before lag, before detrending.

### 3.1 The whole method

    F   = { i : x_i is finite }                      (the finite index set)
    m   = median({ x_i : i in F })
    MAD = median({ |x_i - m| : i in F })

    lower = m - (q * MAD) / 0.6745
    upper = m + (q * MAD) / 0.6745

    flag_i = (x_i < lower) OR (x_i > upper)          (STRICT on both sides)

    # clear the run that is still open at the end of the period
    if flag_{N-1}:
        s = 1 + max{ i : flag_i is False }           (s = 0 if every sample is flagged)
        flag_i := False for all i >= s

    x_i := NaN  where flag_i

| symbol | meaning | unit |
|---|---|---|
| `x_i` | i-th sample of one variable in one period | that variable's unit |
| `N`   | samples in the period | - |
| `m`   | period median | unit of x |
| `MAD` | median absolute deviation | unit of x |
| `q`   | threshold multiplier | robust standard deviations |
| `0.6745` | MAD -> sigma divisor | - |

**Median rule.** Drop the non-finite values *first*, then take the ordinary median of the
`k` survivors: `v[k//2]` for odd k, `(v[k//2 - 1] + v[k//2]) / 2` for even k. The parity is
that of `k`, not of `N`. `statistics.median` agrees; `statistics.median_low` does not.

**Evaluation order is load-bearing.** Compute `q * MAD` first, then divide by `0.6745`.
`(q*MAD)/0.6745` and `q*(MAD/0.6745)` and `q*MAD*1.4826` are three different doubles.
With `q = 7` the half-width is `7/0.6745 = 10.37805782...` MAD.

**The trailing-run rule** is asymmetric on purpose: a spike is an excursion that *returns*,
so a run still outside the bounds when the record ends may be a real step or a drift and is
left alone. A run at the *start* of the period is **not** cleared. This matches the EddyPro
binary on the FR-Gri sample.

### 3.2 Defaults

    q = 7                       (never configured by any bundled adapter; EddyPro's
                                 sr_lim_* thresholds steer a different method)
    variables = u, v, w, ts, co2, h2o
    one pass, one window over the whole period, no iteration

### 3.3 Edge cases

* **All-NaN or empty series**: median and MAD are NaN, every comparison is False, nothing
  is flagged. No crash, no special case needed.
* **MAD = 0** (a stuck sensor, or >50 % identical values): the bounds collapse to `(m, m)`
  and, because the comparisons are strict, every sample differing from the median at all is
  flagged. A perfectly constant series flags nothing. There is deliberately **no guard**.
* **Length 1**: MAD = 0, nothing flagged.
* NaN is never flagged (both strict comparisons are False on NaN) and simply stays NaN.

### 3.4 What is deliberately not implemented

* **No moving window.** The method is named after a paper that describes one, but the
  registered path runs a single window over the whole period, and the parent's windowing
  code is dead: a window shorter than the series raises, and a window equal to the series
  misplaces every flag by `n/2` because the data is rolled and the flags are not rolled
  back. Both are pinned as defects in the parent's own test suite.
* **No iteration / multi-pass, no linear interpolation of flagged runs.** Those belong to
  the EddyPro *binary* and to the Vickers & Mahrt (1997) sibling method, not here. The
  measured consequence against EddyPro is that EddyPro replaces flagged runs whole (many
  more samples replaced than spikes counted) while miniflux flags to NaN and fills nothing.
* **No consecutive-run cap.** Mauder et al. state no such limit and EddyPro applies none on
  this path; on the FR-Gri sample EddyPro's replaced runs reach thousands of samples.
  Applying a cap of 3 would suppress most of what should be flagged.
* **No pooling across periods.** Stacking a whole day would set the MAD by the spread
  *between* periods rather than within one, and the resulting threshold is far too wide to
  flag anything.

### 3.5 Downstream consequence

Flagged samples become NaN and nothing refills them. Every statistic downstream must
therefore be NaN-aware, and the covariance must use a pairwise-complete (joint) mask, or
the NaNs poison the flux silently.

---

## 4. Stage 3 — coordinate rotation: double rotation (Wilczak et al. 2001)

**Citation.** Wilczak, J. M., Oncley, S. P., Stage, S. A. (2001). Sonic anemometer tilt
correction algorithms. *Boundary-Layer Meteorol.* 99, 127-150. doi:10.1023/A:1018966204465

Applied to the **sample series**, per period, after despiking and before the lag search.

### 4.1 The two angles

    theta = atan2( nanmean(v), nanmean(u) )                      [rad]

    u1_i  =  u_i*cos(theta) + v_i*sin(theta)
    v1_i  = -u_i*sin(theta) + v_i*cos(theta)
    w1_i  =  w_i

    phi   = atan2( nanmean(w1), nanmean(u1) )                    [rad]

    u2_i  =  u1_i*cos(phi) + w1_i*sin(phi)
    v2_i  =  v1_i
    w2_i  = -u1_i*sin(phi) + w1_i*cos(phi)

| symbol | meaning | unit |
|---|---|---|
| `u, v, w` | sonic-frame wind components, despiked | m s-1 |
| `theta` | yaw angle, first rotation | rad |
| `phi` | pitch angle, second rotation | rad |
| `u2, v2, w2` | streamline-frame wind | m s-1 |

Equivalent single-matrix form (algebraically identical; the angles must still be computed
in the two-step order above):

    [u2]   [  cos(phi)cos(theta)   cos(phi)sin(theta)   sin(phi) ] [u]
    [v2] = [ -sin(theta)           cos(theta)           0        ] [v]
    [w2]   [ -sin(phi)cos(theta)  -sin(phi)sin(theta)   cos(phi) ] [w]

R = Ry(phi) . Rz(theta) is orthogonal with det = +1, so the instantaneous wind speed is
invariant.

### 4.2 Invariants — which hold exactly, and on what

Three of these are identities in the *angles*, true to rounding on any data. The other two
are statements about the rotated **series**, and they are exact only when u, v and w carry
the **same NaN mask**. Read the difference before using any of them as a self-test.

**Exact to rounding, always** (the angles are built from these very means, so the identity
is algebra, not a property of the data):

    -sin(theta)*nanmean(u)  + cos(theta)*nanmean(v) = 0            the YAW identity
    -sin(phi)*nanmean(u1)   + cos(phi)*nanmean(w)   = 0            the PITCH identity
    sqrt(u2^2 + v2^2 + w2^2) = sqrt(u^2 + v^2 + w^2)   sample by sample, to 1e-9 relative

Only the first of these three is checkable from outside. The pitch angle's reference is
`nanmean(u1)` — the mean of the **intermediate once-rotated** wind (`w1` is just `w`, so
that half is fine). `u1` is materialised inside `kernels.rotate_double` and is not
returned; once the masks differ it is not recoverable from `theta` and the unrotated means
either, because `nanmean(u1)` averages over the u-and-v mask while
`cos(theta)*nanmean(u) + sin(theta)*nanmean(v)` averages over two separate ones. So the
pitch identity is exact and unavailable at the same time.

**Exact only under a shared mask** — the streamline property every flux below assumes:

    nanmean(v2) = 0
    nanmean(w2) = 0
    nanmean(u1) = sqrt(mean(u)^2 + mean(v)^2)                >= 0
    nanmean(u2) = sqrt(mean(u)^2 + mean(v)^2 + mean(w)^2)    >= 0

`theta` comes from two independent nan-means and `phi` from two more (§4.3), while `v2`
exists only where u and v are both finite and `w2` only where all three are. Despiking
flags each component separately, so those masks routinely differ, and `nanmean(v2)` then
averages the rotated series over a *different* sample set from the one the angle was built
on. **On healthy despiked data this leaves ~1e-5 m s-1**, more on a short period. That is
the arithmetic working correctly, not a rotation that failed.

Tolerances, accordingly:

| quantity | bound | scope |
|---|---|---|
| yaw identity | `1e-9` relative to the wind speed | every period, at runtime |
| `nanmean(v2)`, `nanmean(w2)` | `1e-9` m s-1 | tests only, on a **shared-mask** fixture |
| `nanmean(v2)`, `nanmean(w2)` | `1e-3` relative to the wind speed | every period, at runtime |
| instantaneous speed | `1e-9` relative | tests, elementwise |

The runtime self-check (`rotate.py`, DEBUG only) is the source of the two runtime rows: it
is the yaw identity at `1e-9` that pins the *arithmetic*, and the rotated means only at
`1e-3`, which is two orders above the mask noise above and still an order below the
smallest real error worth catching — the §4.3 trap of taking `phi` from `mean(u)` instead
of `mean(u1)` leaves 1.2e-2 m s-1 in a 3 m s-1 wind. Both bounds scale with the wind speed
(floored at 1 m s-1): a fixed absolute bound is a different test at 0.1 and at 10 m s-1.

A port that wants the strict `1e-9` on `nanmean(v2)` must assert it on a fixture whose
three components share one mask. `tests/test_rotate.py` does exactly that, and that is
where the exactness belongs.

### 4.3 Rules that are easy to get wrong

* **`atan2`, never `atan`.** With `atan`, a period whose raw `mean(u) < 0` rotates to
  `u = -|wind|` and flips the sign of `<u'w'>` on exactly those periods and no others —
  invisible in any aggregate.
* **Argument order is `atan2(y, x)`**: `atan2(mean_v, mean_u)` and `atan2(mean_w, mean_u1)`.
  Swapping gives the complement angle and silently wrong fluxes.
* **`phi` is computed from `u1`, not from `u`.** `atan2(mean_w, mean_u)` is a different,
  wrong angle whenever `mean(v) != 0`.
* **Three independent nan-means.** u, v and w can carry different NaN masks (despiking
  flags each series separately). Taking three independent nan-means is what the parent does;
  the closed form `phi = atan2(mean(w), hypot(mean(u), mean(v)))` is only equivalent when
  the three masks agree.
* **Rotate the series, not the covariance matrix.** For the momentum flux alone the two are
  algebraically equivalent, but the lag search runs on the rotated `w`, so a
  covariance-matrix shortcut changes which lag is found and therefore changes `<w'c'>`.
* **Rotate before detrending.** Block-average detrending makes all three means zero, and
  both `atan2` calls then reduce to numerical noise.
* Angles are in **radians**; only the wind direction converts to degrees.
* No gating of any kind: no minimum wind speed, no maximum tilt, no sector logic.

### 4.4 NaN behaviour

`u2` and `w2` mix all three components, `v2` mixes u and v, so the output NaN mask is the
**union** of the input masks. An all-NaN period gives `theta = phi = NaN` and an all-NaN
output. A period with `mean(u) = mean(v) = 0` gives `atan2(0,0) = 0.0`, not an exception.

### 4.5 Wind speed and direction (computed here, from the *unrotated* means)

    wind_speed = sqrt( mean_u_unrot^2 + mean_v_unrot^2 )                    [m s-1]
    wind_dir   = ( 180.0 - degrees(atan2(mean_v_unrot, mean_u_unrot))
                   + north_offset ) mod 360.0                               [deg]

`north_offset` is the azimuth of the sonic's +u axis [deg], default 0.0. It enters **only**
here; it never touches `theta`, `phi` or any flux.

The parent's docstring calls u "zonal, positive eastward" and v "meridional, positive
northward". **Believe the formula, not the prose**: it is the sonic-frame convention with
u forward, v to the left, w up, a right-handed frame. Check: `u=1, v=0 -> 180` (a wind
blowing forward comes from behind); `u=0, v=1 -> 90`; `u=-1, v=0 -> 0`.

`theta` is **not** the wind direction. Do not reuse one for the other.

### 4.6 Not implemented

Triple rotation and planar fit. See §13.

---

## 5. Stage 4 — time-lag maximisation

**Citation.** Aubinet, M., Vesala, T., Papale, D. (eds.) (2012). *Eddy Covariance: A
Practical Guide to Measurement and Data Analysis.* Springer. Implementation follows
EddyPro's `CovMax` (G. Fratini, LI-COR; `src/src_rp/timelag_handle.f90`) with the
`maxcov&default` selection rule (GEddySoft: `MAX_WITH_DEFAULT`).

Runs after rotation, on the rotated `w`, before detrending.

### 5.1 Sign convention — one convention, no negations

`L` is the **physical lag in samples**: positive means the scalar arrives *after* the wind,
so the sample recorded `L` steps later belongs with the wind now. Everything in miniflux
stays in this convention; there is no internal "shift" variable and nothing is negated.

(The parent flips the sign twice — once into xarray's `.shift` convention on the way in and
once back out — which is where its window-ordering and boundary-test complications come
from.)

### 5.2 Window

    L_lo = round(tl_min * f)
    L_hi = round(tl_max * f)
    if L_hi < L_lo: swap
    Lags = [L_lo, L_lo+1, ..., L_hi]            BOTH ENDS INCLUSIVE
    L_nom = round(tl_nom * f)

| symbol | meaning | unit |
|---|---|---|
| `tl_min, tl_max, tl_nom` | lower / upper / nominal lag | s |
| `f` | acquisition frequency | Hz |
| `L_*` | the same in samples | samples (integer) |

A symmetric window `L_nom +- I` therefore holds **2I + 1** lags. This is EddyPro's loop
(`do i = lagmin, lagmax`, inclusive in Fortran). GEddySoft slices `2I` lags and is one
short at the top — do not copy it.

Rounding: EddyPro's `nint` is half-away-from-zero, Python's `round` is half-to-even. They
part only on a lag that is exactly half a sample. miniflux uses Python's `round`.

### 5.3 Covariance at a candidate lag

    for L >= 0:   a = w[0 : N-L],   b = c[L : N]
    for L <  0:   a = w[-L : N],    b = c[0 : N+L]

    P = { i : a_i and b_i both finite } ;  k = |P|
    if k < 2:  cov(L) = NaN
    else:      cov(L) = ( SUM_P a_i*b_i - (SUM_P a_i)(SUM_P b_i)/k ) / (k - 1)

This is `cov(w_i, c_{i+L})`. Three things are load-bearing:

1. **Overlap only, never wraparound.** No `np.roll`, no circular shift. GEddySoft's `xcov`
   rolls and then slices the wrap off only for positive lags; the wrapped fraction grows
   with `|L|`, which tilts the covariance across the window and moves the peak as well as
   biasing its height. The parent reproduces that defect on purpose, as a measurement. We
   do not.
2. **Means and count are recomputed on the overlap at every lag.** Subtracting one
   whole-period mean gives a different (and wrong) curve away from L = 0.
3. **Normalise by the per-lag `k`.** Maximising a raw lagged sum of products is a
   triangularly weighted correlation and pulls the peak toward L = 0.

**Conditioning note (miniflux-specific).** The one-pass raw-sums form above is used here —
it is what both reference engines compute and it is what makes a prefix-sum implementation
possible — but it is applied to *pre-centred* series: before the scan, `w` and `c` are each
reduced by their whole-period nan-mean. Subtracting a constant from either series leaves
every `cov(L)` unchanged in exact arithmetic, and it removes the cancellation that would
otherwise hit a 400 ppm CO2 series. The selected lag is unaffected; only the last bits of
the reported covariance value move.

`ddof`: miniflux uses `k-1`, as the parent and GEddySoft do. EddyPro uses `k`. The ratio
`k/(k-1) = 1.0000278` at k = 36000 and varies by < 0.2 % across a +-60 sample window — it
cannot move a peak, but it does change a printed covariance.

### 5.4 Peak selection

    MaxCov = 0 ;  L_star = L_nom
    for L in Lags, ascending:
        if isfinite(cov(L)) and abs(cov(L)) > MaxCov:
            MaxCov = abs(cov(L)) ;  L_star = L

Strict `>` keeps the **first** (lowest) lag on a tie. Maximising `|cov|` needs no
per-variable sign expectation: `w'T'` peaks positive by day, `w'CO2'` negative by day and
positive by night, `w'u'` negative — one criterion covers all of them.

EddyPro initialises its row-lag variable nowhere, so a period in which no `|cov|` exceeds
zero returns garbage. miniflux initialises `L_star = L_nom` explicitly.

**Not implemented**: the sign-aware alternative (`argmax` if the window-mean covariance is
>= 0 else `argmin`) used by the parent and GEddySoft. It agrees with `|cov|` on a clean
single-peak window and can pick the wrong branch when a strong opposite-sign lobe sits
inside the window.

### 5.5 Boundary fallback (`maxcov&default`)

    if L_star == L_lo or L_star == L_hi:
        L_applied = L_nom ;  default_used = True
    else:
        L_applied = L_star ; default_used = False

    reported:  lag_applied_s = L_applied / f
               lag_opt_s     = L_star    / f
               default_used

A peak pinned to the rim means the true peak is most likely outside the window: the search
failed, and applying a rim lag corrupts the flux. Both numbers are reported; they are
separate columns and must not be confused.

(GEddySoft tests the wrong edges — `lag == center-I+1 or lag == center+I` against a search
range of `center-I .. center+I-1` — so neither reachable edge is flagged. Do not copy it.)

### 5.6 Applying the lag — truncation, never wraparound

    for L >= 0:  c_aligned[i] = c[i+L]  for i = 0 .. N-1-L ;  NaN for i = N-L .. N-1
    for L <  0:  c_aligned[i] = NaN     for i = 0 .. |L|-1 ;  c[i+L] otherwise

Only the scalar moves; `w` is the fixed reference. Each lagged scalar gets its own `L` and
its own NaN edge, and that edge is excluded from every later statistic by the joint-mask
covariance. The period is `|L|` samples shorter for that variable.

### 5.7 Edge cases

* Fewer than 2 finite pairs at a lag: that lag is NaN and is skipped, not counted as zero.
* Every `cov(L)` NaN: `L_star = L_nom`, `default_used = True`.
* No window configured (`tl_min = tl_max = 0`): the search is skipped and `L = 0` is
  applied — a lag of zero, not the nominal.
* `f` non-finite or non-positive: report NaN seconds rather than a raw sample count. "22
  read as 22 s when it means 22 samples is not a diagnostic."
* The scan runs before detrending; this is harmless because the covariance removes the
  overlap means itself.

### 5.8 Do not reuse the scan's covariance as the flux covariance

The scan's value is computed on an overlap with its own count; the flux covariance is
computed afterwards on the aligned, detrended, full period with the two-pass estimator.
Compute it separately.

---

## 6. Stage 5 — detrending

**Citations.** Block average: Moncrieff, J., Clement, R., Finnigan, J., Meyers, T. (2004),
in *Handbook of Micrometeorology*. Linear: Rannik, Ü. and Vesala, T. (1999),
*Boundary-Layer Meteorol.* 91, 259-280.

### 6.1 Block average (default)

    xbar     = nanmean(x)
    xprime_i = x_i - xbar        (NaN stays NaN)

**This step changes no covariance by a single bit**, because the covariance demeans
internally. Its only effects are (a) to make the fluctuation series available and (b) to
destroy the mean. miniflux therefore implements the block average as *bookkeeping*: the
period mean of every series is computed and stored, and the raw series is left in place for
the covariance to demean itself.

The mean must be kept. In the parent, reading the detrended `u` instead of its pre-detrend
snapshot produced 9.5e-17 m s-1 where the wind was 3.15 m s-1, and a spectral correction
factor of 1.2e24 in place of 1.14.

Verified against the EddyPro binary at relative RMS < 1e-6 on all 144 periods of the
FR-Gri sample (level 6 -> level 7).

### 6.2 Linear detrend (opt-in)

This one *is* arithmetic: it removes a real part of the covariance and is a different
high-pass transfer function. `ba` and `ld` give different fluxes.

    F = { i : x_i finite } ;  m = |F|
    if m < 3:  fall back to mean removal (slope = 0)
    ibar  = (1/m) * SUM_{i in F} i
    xbar  = (1/m) * SUM_{i in F} x_i
    slope = SUM_{i in F} (i - ibar)(x_i - xbar) / SUM_{i in F} (i - ibar)^2
    offset = xbar - slope * ibar
    xprime_i = x_i - (slope*i + offset)         for every i = 0 .. N-1

| symbol | meaning | unit |
|---|---|---|
| `slope` | fitted trend | unit of x per sample |
| `offset` | fitted intercept at i = 0 | unit of x |

**DIVERGENCE, deliberate and the only one recommended in this stage.** The parent fits
against the index of the *compacted* (NaN-deleted) array and then evaluates the line on the
*full-length* index. With `k` missing samples the line fitted over `m = N-k` points is
stretched across `N` points, so the removed trend is too large by a factor `N/m` at the end
of the window. miniflux fits and evaluates on the **original sample index**, as written
above. The two agree exactly when `k = 0`.

The pre-detrend mean is still recorded, exactly as for the block average.

### 6.3 Two-dimensional traps that do not apply here

The parent's `block_average` reduces over every dimension and its `linear_detrend` stacks
all dimensions before fitting. miniflux series are strictly one-dimensional, so neither can
occur; do not generalise.

---

## 7. Stage 6 — second moments and friction velocity

### 7.1 The one covariance recipe

    M = { i : x_i finite AND y_i finite }        (JOINT mask)
    N_p = |M|
    if N_p < 2:  cov = NaN
    xbar = (1/N_p) SUM_{i in M} x_i
    ybar = (1/N_p) SUM_{i in M} y_i
    cov(x,y) = ( SUM_{i in M} (x_i - xbar)(y_i - ybar) ) / (N_p - 1)

    var(x) = cov(x, x)
    sd(x)  = sqrt(var(x))

This is exactly `xarray.cov(..., ddof=1)`, which is the parent's registered default.
`N_p = 0` must be handled explicitly (0/0).

Units: `cov(x,y)` carries `unit(x) * unit(y)`.

Pairs computed every period: `(w,ts)`, `(w,co2)`, `(w,h2o)`, `(u,v)`, `(u,w)`, `(v,w)`.
Variances: `u, v, w, ts`.

Zero variance (a stuck sensor) gives `cov = 0.0` exactly, not NaN; `ustar` then becomes 0
and every ratio built on it blows up. The QC guards downstream, not here.

### 7.2 Friction velocity

    ustar = ( cov(u,w)^2 + cov(v,w)^2 )^(1/4)                 [m s-1]

The full stress vector, **not** `sqrt(|cov(u,w)|)`. The two agree only once the rotation
has driven `<v'w'>` to zero; where it has not, the one-component form collapses toward
zero, and since `L` goes as `ustar^3` that error is cubed into every stability-dependent
result.

---

## 8. Stage 7 — thermodynamic mean state

**Computed per sample, then averaged over the period.** The flux factor is therefore
`1/<Vd>` (reciprocal of the mean molar volume), not `<1/Vd>`, and `cp` is
`<cp(T_i, RH_i)>`, not `cp(<T>, <RH>)`. The Jensen gap for the linear-in-P, linear-in-1/T
quantities is ~1e-6 relative.

Position in the chain: after despiking and lag alignment, before detrending, so that every
period mean is taken over exactly the samples that enter the covariances and before the
means are destroyed.

### 8.1 Pass A — seed at the sonic temperature

Inputs per sample: `Ts` [K], `P` [Pa], the H2O column, optionally a measured `Ta_meas` [K].

    if h2o is a dry mixing ratio r [mol mol-1]:
        chi_v = r / (1 + r)
        rho_v = chi_v * Mv * P / (R * Ts)
    if h2o is a molar density n_v [mol m-3]:
        rho_v = n_v * Mv
        chi_v = rho_v * R * Ts / (Mv * P)

    e  = rho_v * Rv * Ts
    Ta = Ta_meas          where finite
         Ts / (1 + 0.32 * e / P)    where that is finite
         Ts                          otherwise

The fallback is applied **per sample**, so one period can mix all three sources.

Note the coefficient: the sonic -> air temperature step is `0.32 * e/P`, **not** `0.51 * q`.
Two different coefficients live in this subsystem and they are not interchangeable.

Note also that `Ts` cancels out of `e`: substituting `rho_v` gives
`e = chi_v * P * (Mv*Rv/R)` for a mixing ratio, so the first-pass and second-pass `e` are
identical when the conversion temperature is `Ta`. The two-pass shape is kept because it is
what the parent does and because it is the hook for an EddyPro-compatibility option.

### 8.2 Pass B — the state, at Ta

    Va    = R * Ta / P                               [m3 mol-1]   moist-air molar volume
    if h2o is a dry mixing ratio:  rho_v = chi_v * Mv / Va
    if h2o is a molar density:     rho_v unchanged ; chi_v = rho_v * R * Ta / (Mv * P)
    e     = rho_v * Rv * Ta                          [Pa]
    es    = exp(77.345 + 0.0057*Ta - 7235/Ta) / Ta^8.2     [Pa]   -- Ta in KELVIN
    RH    = 100 * e / es                             [%]
    Pd    = P - e                                    [Pa]
    Vd    = R * Ta / Pd                              [m3 mol-1]   dry-air molar volume
    rho_d = Pd / (Rd * Ta)                           [kg m-3]
    rho_m = rho_d + rho_v                            [kg m-3]
    q     = rho_v / rho_m                            [kg kg-1]
    sigma = rho_v / rho_d                            [-]
    Tc    = Ta - 273.15                              [degC]
    cp_d  = 1005 + (Tc + 23.12)^2 / 3364             [J kg-1 K-1]
    cp_v  = 1859 + 0.13*RH + (0.193 + 5.6e-3*RH)*Tc + (1e-3 + 5e-5*RH)*Tc^2   [J kg-1 K-1]
    cp    = (1 - q)*cp_d + q*cp_v                    [J kg-1 K-1]
    lambda_v = 1e3 * (3147.5 - 2.37*Ta)              [J kg-1]     -- Ta in KELVIN
    rho_c = n_co2 * Mco2                             [kg m-3]
       with n_co2 = co2 * Pd/(R*Ta)  for a dry mixing ratio [mol mol-1]
            n_co2 = co2              for a molar density [mol m-3]

Period means are then taken (nan-means) of: `Ta, Ts, P, Pd, chi_v, e, es, RH, Va, Vd,
rho_v, rho_d, rho_m, rho_c, q, sigma, cp, lambda_v`.

### 8.3 Three temperature conventions in adjacent formulas — the classic trap

| formula | argument |
|---|---|
| `es` | **Kelvin** |
| `cp_d`, `cp_v` | **Celsius** |
| `lambda_v` | **Kelvin** |

In the parent, `saturation_vapor_pressure` names its local `Tc` and obtains it with
`.to('delta_degC')`; pint's `delta_degC` is the offset-free difference unit, so the value
fed to the formula is the **Kelvin** magnitude. The docstring one line above quotes a
Celsius formula from Campbell & Norman and points the wrong way. Sanity check:
`es(293.15 K) = 2331.215 Pa`. Feeding Celsius returns approximately zero, which drives RH
to infinity — that is what this check is for, and a factor of ~1000 is not a subtle
failure.

> **The `es` formulation itself is low by ~0.3 %, and that is a code question, not a
> stamping one.** The fit of §8.2 is the parent's, transcribed exactly, and it runs
> consistently below the true saturation vapour pressure across its whole range
> (reference: Buck 1981, over water; IAPWS gives 2339.3 Pa at 20 degC, against which the
> error is -0.35 %):
>
> | T | this formula | Buck (1981) | error |
> |---|---|---|---|
> | 273.15 K | 609.42 Pa | 611.21 Pa | -0.29 % |
> | 293.15 K | **2331.215 Pa** | 2338.34 Pa | **-0.31 %** |
> | 313.15 K | 7355.59 Pa | 7382.36 Pa | -0.36 % |
> | 373.15 K | 100985 Pa | 101308 Pa | -0.32 % |
>
> This is a property of the fit, not a transcription slip: it is the same sign and the
> same order at every temperature, and at 100 degC it misses the boiling point by 340 Pa.
> `es` is the denominator of `RH`, and `RH` is a **published column** (CONTRACT §18), so
> every reported relative humidity is ~0.3 % high. Downstream of `RH` the bias is harmless
> — `RH` enters only `cp_v`, which enters `cp` weighted by `q ~ 0.0074`, for a ~3e-7
> relative effect on `cp` and therefore on H — but the published `RH` itself carries it
> whole. Replacing the fit is a change to `flux.py` and to the parent-comparability claim
> that justifies it; it is **not** something to fix by editing this document. Until that
> commit lands, the number above is the number the code produces, and ~2339 Pa is the
> number nature produces.

An earlier revision of this section printed `2343 Pa` here "(true value 2339 Pa)". That
value is neither: it is not what the formula returns and it is not the true one. It is
gone.

`RH` is not clipped: supersaturation gives RH > 100 and feeds straight into `cp_v`.

### 8.4 Things that are *not* what a comment says

* `cp_moist = cp_d (1 + 0.84 q)` appears as a comment in the parent. The code computes the
  mixture `(1-q) cp_d + q cp_v`, which is `cp_d (1 + ~0.87 q)` at 20 degC / 50 % RH,
  because `cp_v` is itself T- and RH-dependent. Implement the mixture.
* `e` is not exactly Dalton's `chi_v * P`: `Mv*Rv/R = 1.00021`, so `e` is 0.021 % above it.
  This is purely the `Rv` rounding and it is kept for comparability with EddyPro.
* `Ta_refined = Ta / (1 + 0.51 q)` exists in the parent, is written to the dataset and is
  **never read**. It is not the air-temperature rule. Do not implement it.

Reference state, 20 degC / 50 % RH / 99 kPa, by the chain of §8.2 from `RH` back to `e`
(`e = 0.50 * es(293.15) = 1165.608 Pa`, `rho_v = e/(Rv*Ta)`, `Pd = P - e`,
`rho_d = Pd/(Rd*Ta)`, `rho_m = rho_d + rho_v`, `q = rho_v/rho_m`):

| quantity | value |
|---|---|
| `es(293.15 K)` | 2331.215 Pa |
| `e` | 1165.608 Pa |
| `rho_v` | 0.0086157 kg m-3 |
| `rho_d` | 1.162677 kg m-3 |
| `rho_m` | 1.171293 kg m-3 |
| `q` | **0.0073557** kg kg-1 |
| `cp_d` | 1005.553 J kg-1 K-1 |
| `cp_v` | 1876.36 J kg-1 K-1 |
| `cp` | **1011.958** J kg-1 K-1 |
| `lambda_v(293.15 K)` | **2452734.5** J kg-1 (`= 2.4527345e6`, and it is exact: the fit is linear in Ta, so no state assumption enters) |

`cp / cp_d = 1 + 0.866 q` here, which is where the `~0.87 q` above comes from.

(Earlier revisions printed `q = 0.00722`, `cp = 1011.84` and `lambda_v = 2.45271e6`. All
three were wrong against this chain — the last one is a plain rounding error on a closed
form, `1e3 * (3147.5 - 2.37 * 293.15)`, which no state assumption can move.)

---

## 9. Stage 8 — flux assembly and the Schotanus correction

### 9.1 Covariance -> molar flux factor

    f_x = 1 / <Vd>      if x is a dry mixing ratio   [mol m-3]
    f_x = 1             if x is a molar density      [-]

`<Vd>` is the **period mean** of the per-sample dry-air molar volume (§8.2). Evaluating
`Pd_mean/(R*Ta_mean)` instead is a slightly different number.

miniflux supports exactly these two measure types. The parent's third, a *wet* mole
fraction (`f = 1/<Va>`, dilution-only WPL), is not implemented — see §13.

### 9.2 The `_L0` fluxes

    FC_L0   = cov(w, co2) * f_co2                            [mol m-2 s-1]
    FH2O_L0 = cov(w, h2o) * f_h2o                            [mol m-2 s-1]
    E_L0    = FH2O_L0 * Mv                                   [kg m-2 s-1]
    LE_L0   = FH2O_L0 * Mv * lambda_v                        [W m-2]
    H_L0    = cov(w, ts) * rho_m * cp                        [W m-2]

`H_L0` is the **buoyancy** (sonic) heat flux, built on the **moist** air density and the
**moist** heat capacity. Both matter.

Arithmetic check in the *input* units, useful when comparing against another program that
never normalises: with CO2 in ppm, `FC[umol m-2 s-1] = cov_w_co2[ppm m s-1] * Pd/(R*Ta)
[mol m-3]` — the 1e-6 of ppm and the 1e6 of umol cancel exactly, no extra factor. With H2O
in ppt, `FH2O[mmol m-2 s-1] = cov_w_h2o[ppt m s-1] * Pd/(R*Ta)`. miniflux gets the same
numbers by the other route: it normalises to mol mol-1 at read time, computes in
mol m-2 s-1, and multiplies by 1e6 at write time.

### 9.3 Schotanus / van Dijk — the exact division form

**Citation.** Schotanus, P., Nieuwstadt, F. T. M., de Bruin, H. A. R. (1983). Temperature
measurement with a sonic anemometer and its application to heat and moisture fluxes.
*Boundary-Layer Meteorol.* 26, 81-93. Revision: van Dijk, A., Moene, A. F., de Bruin,
H. A. R. (2004), Eq. 3.53.

The sonic measures a virtual temperature, `Ts = Ta (1 + 0.51 q)`, whose covariance
decomposes as

    <w'Ts'> = (1 + 0.51 q) <w'T'> + 0.51 * Ta * <w'q'>

Inverting that exactly:

    H = ( H_L0 - 0.51 * cp * Ta * E_L0 ) / ( 1 + 0.51 * q )            [W m-2]

Magnitudes at the reference state of §8.4 (`q = 0.0073557`, `cp = 1011.958`,
`Ta = 293.15`, `lambda_v = 2452734.5`), where the whole numerator term scales as
`0.51 * cp * Ta / lambda_v / Bowen = 0.06168 / Bowen`:

| term | magnitude |
|---|---|
| numerator correction | 5.8 % of H_L0 at Bowen 1; 3.5 % at Bowen 1.7 |
| denominator (dilution) | 0.375 % — this is `0.51 q`, and it does not move with Bowen |

miniflux uses the **division** form, as the parent does. EddyPro uses the linearised
subtraction `H_lin = H_L0 - 0.51 cp Ta E_L0`; the two differ by exactly `1/(1 + 0.51 q)`
and agree to second order in `0.51 q`. Measured contrast on the bundled sample:
`H_L0 / H_EddyPro = 1.0387`.

Rules:

* The latent term takes the **uncorrected** `E_L0`, never a WPL-corrected or spectrally
  scaled E. Rebuilding H from corrected components inflates that term by the whole
  correction.
* The crosswind correction is **not** applied: it is assumed done in the sonic firmware
  (the EddyPro project used for comparison sets `cross_wind = 0`).
* If any of `H_L0`, `E_L0`, `Ta`, `cp`, `q` is missing, `H` is simply not produced. Do not
  alias `H = H_L0`.

### 9.4 Obukhov length and stability

    poisson = ( P0a / <P> )^0.286                                    [-]
    theta_s = <Ts> * poisson                                         [K]
    L       = -( theta_s * ustar^3 ) / ( kappa * g * cov(w,ts) )     [m]
    d       = declared_d          if declared and > 0
              (2/3) * h_canopy    if a canopy height is declared
              0.0                 otherwise (with a warning)
    z_L     = ( zm - max(1e-4, d) ) / L                              [-]

| symbol | meaning | unit |
|---|---|---|
| `<Ts>` | period mean **sonic** temperature, pre-detrend | K |
| `<P>` | period mean air pressure | Pa |
| `theta_s` | sonic **potential** temperature | K |
| `zm` | measurement height above ground | m |
| `d` | zero-plane displacement | m |

The temperature is the **sonic potential** one and the flux in the denominator is the
**sonic** covariance `cov(w,ts)`, not the Schotanus-corrected H: the virtual temperature
must be paired with the buoyancy flux it belongs to. The leading minus makes an upward heat
flux give `L < 0` = unstable.

`kappa = 0.4`. EddyPro's raw-processing L uses 0.41; the difference is a documented,
deliberate fork.

(GEddySoft omits the Poisson factor and subtracts no displacement; the parent measures the
resulting gap on z/L at 0.5-3 %.)

Degeneracies, unguarded by design, guarded in the QC consumer: `cov(w,ts) = 0` gives
`L = +-inf` and `z_L = 0`; `ustar = 0` gives `L = 0` and `z_L = +-inf`.

---

## 10. Stage 9 — WPL density correction (Webb, Pearman & Leuning 1980)

**Citation.** Webb, E. K., Pearman, G. I., Leuning, R. (1980). Correction of flux
measurements for density effects due to heat and water vapour transfer. *Q. J. R.
Meteorol. Soc.* 106, 85-100. Eq. 24 (gas) and Eq. 25 (vapour).

Removes from an open-path gas-density covariance the part that is not gas transport — the
thermal expansion of the air and the displacement of dry air by water vapour — using the
constraint that the mean dry-air flux through the surface is zero.

### 10.1 When it is owed

| how the gas was reported | correction owed |
|---|---|
| **ambient** molar density (mol m-3), i.e. open path | **full** — thermal + dilution |
| dry mixing ratio (per mole of dry air) | **none** |
| **cell** molar density, i.e. closed path | **none** — §2A converts it first; see below |
| wet mole fraction | dilution only — *not implemented in miniflux*, see §13 |

A dry mixing ratio is per mole of dry air, which is what makes it conserved; correcting it
for dry-air fluctuations again subtracts the same effect twice. miniflux runs the WPL block
only when at least one of CO2 / H2O is a molar density; otherwise the reported fluxes are
exact copies of the `_L0` fluxes.

**A closed-path gas is never given this correction** (§2A.5). Webb's derivation is about
ambient air at the sampling point; a cell at the end of a tube has already damped some of
those fluctuations and imposed others, so the ambient correction applied to a cell quantity
is wrong in kind. The closed-path treatment is the per-sample conversion of §2A instead,
after which the gas is conserved and owes nothing. `wpl.py` contains no test for a cell:
`config.py` has already resolved such a gas's **effective** measure type to
`mixing_ratio`, so the `molar_density` branches simply never see it.

If a gas *is* owed a correction and WPL is switched off, **no reported flux is written for
it at all** — `FC` is left missing while `FC_L0` is kept. An uncorrected number must never
carry a corrected name.

### 10.2 Inputs

    mu    = Md / Mv                      [-]
    sigma = <rho_v> / <rho_d>            [-]
    wT    = H / (<rho_m> * cp)           [K m s-1]    the AIR-temperature kinematic flux

`wT` is read back out of the **Schotanus-corrected** H. **Do not use `cov(w,ts)`**: the
sonic covariance runs high by the latent contribution, by an amount that grows as the Bowen
ratio falls, and since the thermal term is comparable to the CO2 flux itself the excess
would land more or less whole in the answer. At the parcel of §10.8 (`H = 180`,
`LE_L0 = 186.6`, so Bowen 0.96) the sonic flux is 6.8 % high by §9.3's ratio, and 6.8 % of
a term that is 43-58 % of FC is a 3-4 % error on the flux.

`cp` cancels identically between `H_L0 = cov(w,ts) rho_m cp` and `wT = H/(rho_m cp)`; a port
that computes `cp` badly still gets FC exactly right, and only H itself is wrong.

### 10.3 The canonical vapour flux and the vapour density covariance

    f_chi = FH2O_L0 * (1 - <chi_v>)                  if h2o is a dry mixing ratio
          = FH2O_L0 + (<rho_v>/Mv) * wT / <Ta>       if h2o is a molar density

    w_rho_v = Mv * f_chi - <rho_v> * wT / <Ta>       [kg m-2 s-1]

| symbol | meaning | unit |
|---|---|---|
| `f_chi` | vapour flux per mole of moist air | mol m-2 s-1 |
| `w_rho_v` | `<w' rho_v'>`, the vapour mass-density covariance | kg m-2 s-1 |

The reporting form is absorbed into `f_chi` *first*; feeding the raw `FH2O_L0` of a
mixing-ratio or mole-fraction analyser straight into Webb's equation double-counts the part
of the thermal expansion that form has already shed.

### 10.4 Webb Eq. 25 — water

    E  = (1 + mu*sigma) * ( w_rho_v + (<rho_v> / <Ta>) * wT )        [kg m-2 s-1]
    LE = lambda_v * E                                                [W m-2]

Write the bracket out rather than cancelling it to `Mv * f_chi`, so the equation on the page
is the equation in the code. (The two are equal algebraically but not in floating point.)

Self-consistency check: for a dry mixing ratio, `1 + mu*sigma = 1/(1 - chi_v)` exactly when
`rho_v` and `rho_d` come from one ideal-gas state, so `E` reduces to `Mv * FH2O_L0`. With
the rounded `Rd` and `Rv` the identity holds to ~3e-4. Assert this in a test.

### 10.5 Webb Eq. 24 — the gas

    heat = (1 + mu*sigma)          for a molar-density gas
    raw_mass       = FC_L0 * Mco2
    corrected_mass = raw_mass + mu * (<rho_c>/<rho_d>) * w_rho_v
                              + heat * (<rho_c>/<Ta>) * wT
    FC = corrected_mass / Mco2                                       [mol m-2 s-1]

The multiply-by-`Mco2` / divide-by-`Mco2` round trip cancels; it is written this way so the
code reads as Webb's mass-density equation.

Equivalent molar form (exact, no approximation — useful as a cross-check):

    FC = FC_L0 + c_c * f_chi / n_d + c_c * wT / <Ta>
    with c_c = <rho_c>/Mco2 [mol m-3], n_d = <rho_d>/Md [mol m-3]

### 10.6 The landmine: which mean density

`<rho_c>` and `<rho_v>` must be the means of the **raw** series, taken before detrending.
The parent's own regression was a version that rebuilt the mean from the block-averaged CO2
series, read ~1e-16 mmol m-3, and produced two vanishing Webb terms — a silent no-op
indistinguishable from a correction that ran. With block-average detrending this costs
nothing: the mean the covariance already needs *is* the mean WPL needs.

### 10.7 Other rules

* **Constant pressure** is assumed, per Webb's derivation. There is no pressure term and no
  pressure-fluctuation covariance; P enters only through the mean densities.
* **Ordering is fixed and non-iterative**: covariances -> `H_L0` -> Schotanus (on the
  *uncorrected* `E_L0`) -> WPL (on the resulting `H`) -> report. There is no second
  Schotanus pass on the corrected E.
* **`Ta` in Kelvin.** It appears as `rho/Ta`; a degC slip is silent in a stdlib port.
* **Refuse rather than approximate.** If any required mean or covariance is non-finite,
  emit NaN for the corrected flux. An absent input here does not make the correction
  smaller, it makes it wrong.
* **Sign**: the correction carries the sign of `wT` and of `E`, so by day it is a positive
  (upward) increment that makes an apparent CO2 sink smaller.
* The correction is first order; its residual is second order and grows with the
  correction/flux ratio — the parent measures 0.021 % at a ratio of 0.7 and 0.30 % at a
  ratio of 11.

### 10.8 Sanity magnitudes (midday parcel: Ta = 293 K, P = 99 kPa, chi_v = 0.012, 420 ppm CO2, H = 180 W m-2, LE = 200 W m-2)

The parcel's mean state is built **strictly by §8.2**, with `rho_d` at the dry partial
pressure `Pd = P - e` (not at `P`), which is the whole point of that line:

    e     = 1188.25 Pa      chi_v = 0.012, so rho_v = chi_v*Mv/Va and e = rho_v*Rv*Ta
    Pd    = 97811.75 Pa     = P - e
    rho_v = 0.0087876 kg m-3 ;  rho_d = 1.163003 kg m-3 ;  rho_m = 1.171791 kg m-3
    q     = 0.0074993 ;  cp = 1012.081 J kg-1 K-1 ;  lambda_v = 2453090 J kg-1
    n_co2 = 0.01686314 mol m-3  (= c_c) ;  n_d = 40.14509 mol m-3
    wT    = H / (rho_m * cp) = 0.1517773 K m s-1

    thermal term   c_c * wT / Ta      = +8.74 umol m-2 s-1    (43-58 % of a raw -15..-20 flux)
    dilution term  c_c * f_chi / n_d  = +1.88 umol m-2 s-1    (~9-12 %)
    total          +10.61 on a raw -20.00  ->  -9.39 umol m-2 s-1
    water:  (1 + mu*sigma) = 1.0121 (+1.2 %), thermal expansion adds +6.0 %,
            LE 186.6 -> 200.2 W m-2  (+7.3 %)

The `LE = 200 W m-2` in the heading is the parcel's round design figure; 200.2 is what the
chain actually returns from `LE_L0 = 186.6`, and it is the second number that has to be
reproduced.

**Where the old `-9.49` came from.** Earlier revisions printed `+8.63 / +1.88 / +10.51 ->
-9.49` and `LE ... 200.0`. Those are what you get from `rho_m = P/(Rd*Ta) + rho_v`, i.e.
the dry-air density evaluated at the **total** pressure with the vapour never subtracted.
That is 1.21 % high on `rho_m`, hence 1.21 % low on `wT` and on the thermal term, hence a
total correction 1.0 % smaller and an FC 1.1 % more negative. It is the single most attractive slip in §8.2 and it produces a number
that looks entirely reasonable, which is why both parcels are kept in the suite:
`tests/test_wpl.py::MiddayMagnitudes` pins the `-9.39` this section now prints (to
`-9.386`, its full precision) and `::PublishedParcel` pins the `-9.49` that the
total-pressure state gives, so a port that lands on the second knows exactly which line it
got wrong.

Rule of thumb worth asserting in a test: by day the CO2 correction is tens of percent to
about one times the flux. **If your WPL term comes out at a few percent of FC, you have
almost certainly used `cov(w,ts)` for `wT`, or a detrended (near-zero) mean density.**

---

## 10A. Stage 9b — first-order low-pass attenuation (Horst 1997)

**Citation.** Horst, T. W. (1997). A simple formula for attenuation of eddy fluxes measured
with first-order-response scalar sensors. *Boundary-Layer Meteorol.* 82, 219-233. Eq. 11,
with the peak frequency of his Eqs. 8 and 10.

**Default: off.** Lettered for the same reason as §2A.

### 10A.1 The bias this is about, stated loudly

A tube, a finite optical path, a sensor separation and a finite sensor response all
low-pass the gas signal. Whatever fraction of the flux lives above the resulting cut-off
is simply **missing from the covariance**. The error is one-signed:

> **A closed-path FC or LE from miniflux with `[spectral] enabled = false` is an
> UNDERESTIMATE — typically a few per cent to tens of per cent, depending on tube, flow
> and measurement height.**

This is the largest known bias in a miniflux gas flux. On the bundled FR-Gri sample the
parent's full multi-term analytic method measures it at **10.2 % for both LE and CO2**
against EddyPro. `H` is not affected: the sonic does not sample through the tube.

A run whose analyser is closed-path and whose spectral section is off logs that sentence
once, at `WARNING`, so a reader who has only the log and the table can still see it.

### 10A.2 The model

Horst treats the analyser as one first-order system with time constant `tau` and
integrates its transfer function against a similarity cospectrum. The integral has a closed
form, which is why this is forty lines and not four hundred:

    A   = 1 / (1 + (2*pi*nm*tau*u/z)**alpha)        the attenuated fraction, 0 < A <= 1
    SCF = 1/A = 1 + (2*pi*nm*tau*u/z)**alpha        the recovery factor, >= 1

with `u` the period's mean wind speed [m s-1] and `z` the measurement height above the
displacement, `z - d` [m] — the same height §9.4 puts under `z/L`. The cospectral shape
enters through two constants that depend only on stability:

| | `nm` | `alpha` |
|---|---|---|
| `z/L <= 0` | `0.085` | `7/8` |
| `z/L >  0` | `2.0 - 1.915 / (1 + 0.5 z/L)` | `1` |

`2.0 - 1.915 = 0.085` exactly, so the **peak frequency is continuous** at `z/L = 0` and
only the exponent steps; the factor itself therefore does jump there, and a port should
check the two facts separately rather than reading the jump as a bug. A stable period
loses more, because its cospectrum peaks at higher natural frequency and more of it sits
inside the roll-off.

Check value, on the FR-Gri geometry (`u = 2`, `z - d = 1.9`, `z/L = -0.5`, `tau = 0.13`):
`SCF = 1.1013541`, i.e. a flux measured 9.2 % low. The same geometry at `z/L = +0.5` gives
`SCF = 1.4023884`.

### 10A.3 What it is not

**This is a first-order approximation and NOT a substitute for an in-situ method.** It is
not EddyPro's, and it is not `oneflux_preproc`'s. It knows nothing about:

* tube sorption on the water channel, or its RH dependence (H2O attenuates more than CO2
  in a tube, and by an amount that varies with humidity — here it can only be given a
  larger `tau`, fixed for the run);
* sensor separation, which on a short tower is often the largest single term;
* the actual measured cospectra, which the in-situ methods fit a cut-off frequency to.

What it gives is **one defensible number per period, from geometry the user declares**,
instead of a silent zero.

### 10A.4 miniflux does not guess `tau`

`[spectral] co2_tau_s` and `h2o_tau_s` have **no default** and the run is refused if the
section is enabled without them (CONTRACT 5.2.13). The time constant belongs to *this*
tube at *this* flow rate, not to the analyser model; a shipped value would put a plausible
few per cent on every flux of every user who switched the section on without reading it.

Determine it from a cospectral ratio against the sonic temperature, from the instrument's
documented response — or, pragmatically, by fitting the loss you already know you have.
Worked on FR-Gri (`z - d = 1.9 m`, mean wind 0.73 m s-1 over the 48 half hours):

| `tau` [s] | median `SCF` | median flux recovered |
|---|---|---|
| 0.13 | 1.048 | 4.5 % |
| 0.30 | 1.099 | 9.0 % |
| 0.35 | ~1.113 | ~10.2 % |
| 0.40 | 1.127 | 11.3 % |

So ~0.35 s is what reproduces the 10.2 % the parent's full method measures against EddyPro
here — an order larger than the LI-7200's own declared 0.1 s response, because one
first-order constant has to absorb the tube, the optical path and the 17 cm sensor
separation as well. That gap is the honest measure of how much this model leaves out.

### 10A.5 What it publishes, and the naming rule

It **never rewrites a flux**. The factors and the scaled fluxes are their own columns
beside the measured ones:

    SCF_CO2, SCF_H2O                  the recovery factors [-], >= 1
    FC_SPEC = FC * SCF_CO2            [µmol m-2 s-1]
    E_SPEC  = E  * SCF_H2O            [g m-2 s-1]
    LE_SPEC = LE * SCF_H2O            [W m-2]

`FC`, `LE` and `E` stay exactly what they were — measured, and attenuated. No number in the
table can be read as corrected when it is not, or the other way round. With the section
off, all five columns are `na_value`: a correction that was not made is absent, not 1.0.

`H` is deliberately not scaled. A gas time constant does not describe the sonic.

### 10A.6 Order: it scales the reported flux, after WPL

The step runs **after** `wpl`, so `FC_SPEC` is one multiplication away from `FC` and a
reader can check it by hand. For a closed-path run — the case this exists for — that is
unambiguous, because no WPL ran: `FC = FC_L0`.

For an **open-path molar-density** run, EddyPro's order is the other one (correct the
covariance, then apply WPL), and WPL is not linear in the covariance, so the two differ.
The factor is applied where it is applied, and this paragraph is the whole of the caveat.

### 10A.7 Guards

* `wind_speed` or `z/L` non-finite, or `z - d <= 0` → the factor is `NaN` and a `WARNING`
  says which. A correction is a measured quantity here; a missing one is reported missing.
* `SCF > 2` → a `WARNING`. Above that, more than half the reported flux would come from
  the model rather than from the data, and Horst's first-order form is derived for modest
  attenuation. It is not clipped — clipping would invent a number — but it is not silent.
* `u = 0` gives `SCF = 1` exactly: no wind, no advected eddies, nothing to attenuate.

---

## 11. Stage 10 — quality flags

### 11.1 Steady-state test, Foken & Wichura (1996)

**Citation.** Foken, T. and Wichura, B. (1996). Tools for quality assessment of
surface-based flux measurements. *Agric. For. Meteorol.* 78, 83-105.
doi:10.1016/0168-1923(95)02248-1. Thresholds: Foken, T. et al. (2004), *Handbook of
Micrometeorology*, ch. 9.

    N = period length
    m = floor(N / 5)
    C_j   = cov( w[(j-1)m : j*m], c[(j-1)m : j*m] )    for j = 1..5
    C_tot = cov( w[0:N], c[0:N] )

    if C_tot == 0 or not finite:  S = NaN
    else:  S = | ( nanmean(C_1..C_5) - C_tot ) / C_tot |        [fraction]
    SST_pct = 100 * S                                            [%]

    flag = 2   if SST_pct is not finite
         = 0   if SST_pct <= 30
         = 1   if SST_pct <= 100
         = 2   otherwise

Structural details that are part of the test and must be kept verbatim:

* five sub-intervals, `m = floor(N/5)`, slices `[(j-1)m, j*m)` (exclusive end);
* the remainder `N - 5m` is excluded from the sub-intervals but **included** in `C_tot`;
* each sub-interval covariance is taken about its **own** mean — that is what makes the
  test sensitive to a trend, and it holds even though the series has already been
  block-average detrended;
* `nanmean` over the five, so a sub-interval that came back NaN is skipped and the
  statistic can rest on fewer than five with no record that it did;
* the statistic is **not capped**: a period can print thousands of per cent. 30 and 100 are
  flag bounds, not clamps;
* a non-finite statistic flags **2** (discard), not "missing".

Default pair: `(w, co2)`.

Degenerate `N`: `N < 5` gives `m = 0`, every sub-interval is empty, every `C_j` is NaN, so
`SST = NaN` and `flag = 2`.

**DIVERGENCE, decided.** The parent (following GEddySoft `xcov`) computes all six of these
covariances with a kernel that **drops the last sample of every slice** (`x[0 : M-1]` in its
zero-lag branch). miniflux uses the single covariance recipe of §7.1 for both the
sub-intervals and the whole period. The dropped sample changes the statistic by
`O(1/n) = 3e-5` relative, far below the 30 % / 100 % thresholds, and carrying a second
estimator only to reproduce a transcription artefact is not worth it. Anyone comparing
miniflux against ONEFlux_preproc or GEddySoft at more than four significant figures on this
statistic should expect this difference and no other.

**Not implemented**: the Mahrt (1998) and Dutaur (1999) steady-state variants, and the
parent's deprecated `sta` variant (6 sub-intervals, a floored percentage and a 99999
sentinel). See §13.

### 11.2 Integral turbulence characteristics (ITC)

**Citation.** Foken & Wichura (1996); Thomas, C. and Foken, T. (2002). Re-evaluation of
integral turbulence characteristics and their parameterisations. The parent's
implementation carries the note "implementation from EddyPro v7.0.4".

    omega = 2*pi / (24*60*60) = 7.2722052166430395e-05       [rad s-1]
    Fcor  = | 2 * omega * sin(lat * pi/180) |                [s-1]
    zplus = 1.0                                              [-]   (hard-coded, NOT the height)

Models:

    if z_L < -0.2:                              # unstable
        sw_model = 1.3  * (1 - 2*z_L)^(1/3)
        su_model = 4.15 * |z_L|^(1/8)
    else:                                       # neutral / stable
        sw_model = 0.21 * ln(Fcor * zplus / ustar) + 3.1
        su_model = 0.44 * ln(Fcor * zplus / ustar) + 6.3

    if            z_L <  -1.0    : sT_model = |z_L|^(-1/3)
    elif -1.0  <= z_L <  -0.0625 : sT_model = |z_L|^(-1/4)
    elif -0.0625 <= z_L <  0.02  : sT_model = 0.5 * |z_L|^(-0.5)
    else                         : sT_model = 1.4 * |z_L|^(-1/4)

The last branch is `else`, not `elif z_L > 0.02`: `z_L` exactly 0.02 and `z_L` non-finite
reach it. (Upstream GEddySoft leaves the variable unassigned in those two cases.)

Measured characteristics and deviations:

    Tstar   = -cov(w,ts) / ustar                             [K]
    sw_meas = sd(u2_w) / ustar                               [-]
    su_meas = sd(u2_u) / ustar                               [-]
    sT_meas = sd(ts)   / |Tstar|                             [-]
    ITC_w = | (sw_model - sw_meas) / sw_model |              [fraction]
    ITC_u = | (su_model - su_meas) / su_model |              [fraction]
    ITC_T = | (sT_model - sT_meas) / sT_model |              [fraction]

`sd` is `sqrt(var)` with ddof = 1 over the whole period; there are no sub-intervals. The
wind components are the **rotated** ones; `ts` is in Kelvin.

**No flag is derived from the ITC deviations.** The parent derives none either; there is no
0-9 Foken grade and no combined steady-state/ITC class anywhere in this program. The nine
grades of Foken (2003), the three of Mauder & Foken (2004) and the five of Göckede et al.
(2006) are described in EddyPro's documentation, not implemented here. Adding one is new
code, not a port.

**DIVERGENCE, decided.** The parent and GEddySoft divide by the **signed** `Tstar`. Under
unstable conditions `cov(w,ts) > 0` makes `Tstar < 0`, so `sT_meas < 0` while
`sT_model > 0` and `ITC_T = (model + |meas|)/model > 1` always. Foken's definition is
`sigma_T / |T*|`. miniflux uses `|Tstar|`, so its `ITC_T` is **not** comparable with the
parent's or with GEddySoft's on unstable periods. This is stated in the output header.

Guards (both reference codebases leave these open; miniflux returns NaN instead of `inf` or
a domain error):

* `ustar <= 0` — the neutral branch would take `ln(x/0)` and `Tstar` would divide by zero;
* `lat == 0` — `Fcor = 0` and the neutral branch takes `ln(0)`;
* `z_L == 0` exactly — branch 3 evaluates `0.5 * 0^(-0.5)`;
* `z_L` non-finite — the models are meaningless.

ITC also requires the period to have reached the stability stage; running it before `z_L`
exists returns the neutral wind models computed on an absent stability, which is two
plausible numbers and no error. In miniflux the QC step is the last step in the pipeline,
after the flux step that creates `z_L`.

---

## 12. Numerical policy and the numpy fast path

1. Reductions are naive left-to-right sums. Medians are exact order statistics with the
   even/odd rule of §3.1.
2. The optional numpy path must reproduce every **decision** bit-for-bit: the despike flag
   mask, the selected lag, the QC flag. Those are comparisons, not sums, and they are
   exactly reproducible.
3. Elementwise transformations (rotation, lag shift, line subtraction, unit conversion) are
   bit-identical between the two paths, because they are the same sequence of IEEE-754
   double operations per sample.
4. Reductions (sums, means, covariances) may differ in the last bits: numpy sums pairwise,
   the pure path sums left to right. The contract is agreement within 8 ULP, i.e. relative
   `2e-15`, and the test suite asserts `1e-12` relative. Anything larger is a bug in the
   kernel, not an acceptable variant.
5. Do not "improve" the pure path with a compensated sum: that would make the two paths
   disagree by more than the tolerance above.

---

## 13. What miniflux deliberately does not do

Each line says what the omission costs the user.

* **No in-situ spectral / frequency-response correction** (empirical cut-off frequencies
  fitted to measured cospectra, Moncrieff, Massman, Ibrom's low-pass method, RH-dependent
  H2O attenuation, sensor separation, the high-pass block-average term). Fluxes are
  underestimated by roughly 2-15 % depending on measurement height, tube length and
  stability — most severely for LE on a closed-path system and for any flux in stable,
  low-wind conditions. This is the single largest known bias in a miniflux gas flux.
  What miniflux *does* offer is §10A: one first-order time constant per gas against an
  analytic transfer function, off by default, published in its own columns and explicitly
  labelled an approximation. It does not replace the methods above, and §10A.4 measures
  the gap on the bundled sample.
* **No planar fit and no triple rotation.** Double rotation over-rotates individual
  low-wind periods (it forces `<w> = 0` even when the true streamline is tilted), which a
  planar fit estimated once over weeks of data avoids; the cost is a period-to-period
  scatter in the momentum flux and a small bias on sloping terrain. The planar fit needs
  cross-period state that a one-period-at-a-time program does not have. Triple rotation is
  omitted because its third angle depends on `<v'w'>` and `<v'^2>-<w'^2>`, EddyPro guards it
  at `|psi| > 10 deg` and the reference implementation deliberately does not, so the two
  disagree on essentially every period with a large roll angle.
* **No multi-instrument closed-path setups.** Closed-path analysers *are* accepted (§2A),
  but `[gases] analyser_path` is one word for the run: miniflux is a one-instrument, one-
  clock, one-file-series program, and a CO2 cell on one tube beside an open-path water
  channel is two instruments with two lags and two frames of reference. Run them
  separately, or use an engine built for it.
* **No wet-mole-fraction gas input** (dilution-only WPL). A user whose analyser reports a
  wet mole fraction must convert to a dry mixing ratio or a molar density before the file
  reaches miniflux; using the wrong branch is a ~12 % error on FC.
* **No gap filling, no u\* filtering, no night-time partitioning, no storage term.** The
  output is a half-hourly flux table with holes in it. Annual sums cannot be computed from
  it without a downstream gap-filling step (ONEFlux, REddyProc), and the missing storage
  term makes night-time NEE too low at tall towers.
* **No footprint model.** There is no way to tell, from a miniflux row, how much of the flux
  came from the target ecosystem. Quality screening by wind direction and stability must be
  done by hand from `WD`, `ZL` and `USTAR`.
* **No random-uncertainty estimate** (neither Finkelstein & Sims nor the Mann & Lenschow
  approach). A miniflux flux has no error bar.
* **No Vickers & Mahrt (1997) despiking, no instrument-diagnostic / skewness-kurtosis
  flags, no absolute-limits or drop-out tests.** The only high-frequency screening is the
  MAD test of §3. A stuck sensor that reports a plausible constant is not detected here.
* **No clock-drift or RH-dependent lag windows.** Those are table-driven, engine-specific
  window-centring steps; without them a badly drifting acquisition clock will simply push
  the peak to a window edge, where the boundary fallback (§5.5) catches it.
* **No Mahrt (1998) or Dutaur (1999) stationarity variants, and no combined Foken flag.**
  Both variants carry transcribed index and sign errors in the reference implementation and
  their thresholds are not Foken's; the combined 0-9 grade would have to be written from
  scratch. The user gets one stationarity number, one 0/1/2 flag and three ITC deviations,
  and must combine them themselves.
* **No multi-instrument / asynchronous stream assembly, no resampling, no regridding.** One
  instrument, one clock, one file series. A stream that is irregular in time is reported,
  not repaired.

---

## 14. End-to-end self-tests a port must pass

1. **Rotation invariants** (§4.2), each on the data it actually holds for: the yaw
   identity at `1e-9` relative on **every** period; `nanmean(v2) = nanmean(w2) = 0` at
   `1e-9` on a **shared-mask** fixture only, and at `1e-3` relative elsewhere. Asserting
   the tight bound on ragged masks fails on healthy despiked data.
2. **Despiking on a synthetic series**: 1000 samples of `N(0,1)` with one sample set to
   `+50` -> exactly one flag; the same spike placed at the last index -> zero flags
   (trailing-run rule); a constant series -> zero flags.
3. **Lag sign**: build `co2 = roll(w, 15)` at 10 Hz; the detected lag must be `+1.5 s`.
4. **Covariance parity**: `cov(x, x) == var(x)`; `cov(x + 1e6, y) == cov(x, y)` to within
   1e-9 relative (translation invariance is the two-pass form's whole point).
5. **WPL water identity**: with a dry mixing ratio for H2O, `E` from §10.4 equals
   `Mv * FH2O_L0` to within 5e-4 relative (the `Rd`/`Rv` rounding).
6. **WPL magnitude**: the parcel of §10.8, with its state built strictly per §8.2, must
   reproduce `-20.00 -> -9.39 umol m-2 s-1` to within 0.5 %. Feeding the same parcel a
   `rho_m` built on the **total** pressure gives `-9.49` instead; both are pinned, and
   which one a port hits says which line of §8.2 it read.
7. **Schotanus contrast**: at the reference state of §8.4, `H_L0 / H` is
   `(1 + 0.51 q) + 0.51 cp Ta / lambda_v / Bowen = 1.00375 + 0.06168 / Bowen`. At a Bowen
   ratio of **1** that is **1.0654**, not 1.04; 1.04 is the Bowen = 1.70 period. Assert
   whichever you like, but assert it against the Bowen ratio you actually built.
8. **numpy parity**: identical decisions, reductions within 1e-12 relative (§12).
9. **Closed-path round trip** (§2A.3): take a dry mixing ratio, express it as the cell
   molar density it would be reported as at a given `T_cell` / `P_cell`, convert it back —
   the same number returns. Synthesise the density from the ratio, not the other way
   round, so a shared algebra error would have to be made twice in opposite directions to
   pass. Carried through the whole pipeline, the two declarations of one synthetic half
   hour agree to `2e-16` relative; on the real FR-Gri half hours, to 0.03 % (§2A.8).
10. **The dilution divisor is load-bearing**: dropping `1/(1 - chi_h2o)` leaves a moist
    mole fraction under a dry name, ~1 % on an ordinary parcel.
11. **The conversion is per sample**: two periods with the same *mean* cell temperature,
    one steady and one swinging, must give different converted series — and the
    sample-to-sample ratio must carry the dilution coupling of §2A.3, not just `T+/T-`.
12. **Horst Eq. 11** (§10A.2): `SCF(u=2, z-d=1.9, z/L=-0.5, tau=0.13) = 1.1013541` and
    `SCF(z/L=+0.5) = 1.4023884`; the peak frequency is continuous at `z/L = 0` while the
    exponent is not; `SCF >= 1` everywhere; it rises with `tau` and with `u`, and falls
    with height.

---

## 15. References

* Aubinet, M., Vesala, T., Papale, D. (eds.) (2012). *Eddy Covariance: A Practical Guide to
  Measurement and Data Analysis.* Springer.
* Fleagle, R. G. and Businger, J. A. (1980). *An Introduction to Atmospheric Physics.*
  Academic Press. (the `lambda_v(T)` fit, via EddyPro `flux_params.f90`)
* Foken, T. and Wichura, B. (1996). Tools for quality assessment of surface-based flux
  measurements. *Agric. For. Meteorol.* 78, 83-105. doi:10.1016/0168-1923(95)02248-1
* Foken, T., Göckede, M., Mauder, M., Mahrt, L., Amiro, B., Munger, W. (2004). Post-field
  data quality control. In *Handbook of Micrometeorology*, Springer, 181-208.
  doi:10.1007/1-4020-2265-4_9
* Horst, T. W. (1997). A simple formula for attenuation of eddy fluxes measured with
  first-order-response scalar sensors. *Boundary-Layer Meteorol.* 82, 219-233.
  doi:10.1023/A:1000229130034 (§10A, Eq. 11 and the peak frequency of Eqs. 8 and 10)
* Ibrom, A., Dellwik, E., Larsen, S. E., Pilegaard, K. (2007). On the use of the
  Webb-Pearman-Leuning theory for closed-path eddy correlation measurements.
  *Tellus B* 59, 937-946. (§2A, the cell conversion)
* Mauder, M. et al. (2013). A strategy for quality and uncertainty assessment of long-term
  eddy-covariance measurements. *Agric. For. Meteorol.* 169, 122-135.
* Moncrieff, J., Clement, R., Finnigan, J., Meyers, T. (2004). Averaging, detrending and
  filtering of eddy covariance time series. In *Handbook of Micrometeorology*, Springer.
* Monteith, J. L. and Unsworth, M. H. (2013). *Principles of Environmental Physics*, 4th
  ed., Sect. 9.2. (the `d = 2h/3` closure)
* Rannik, Ü. and Vesala, T. (1999). Autoregressive filtering versus linear detrending in
  estimation of fluxes by the eddy covariance method. *Boundary-Layer Meteorol.* 91,
  259-280.
* Schotanus, P., Nieuwstadt, F. T. M., de Bruin, H. A. R. (1983). Temperature measurement
  with a sonic anemometer and its application to heat and moisture fluxes.
  *Boundary-Layer Meteorol.* 26, 81-93.
* Thomas, C. and Foken, T. (2002). Re-evaluation of integral turbulence characteristics and
  their parameterisations. *Proc. 15th Symposium on Boundary Layers and Turbulence*.
* van Dijk, A., Moene, A. F., de Bruin, H. A. R. (2004). The principles of surface flux
  physics. Meteorology and Air Quality Group, Wageningen University. (Eq. 3.53)
* Vickers, D. and Mahrt, L. (1997). Quality control and flux sampling problems for tower and
  aircraft data. *J. Atmos. Oceanic Technol.* 14, 512-526. (cited for what miniflux does
  *not* do)
* Webb, E. K., Pearman, G. I., Leuning, R. (1980). Correction of flux measurements for
  density effects due to heat and water vapour transfer. *Q. J. R. Meteorol. Soc.* 106,
  85-100.
* Wilczak, J. M., Oncley, S. P., Stage, S. A. (2001). Sonic anemometer tilt correction
  algorithms. *Boundary-Layer Meteorol.* 99, 127-150. doi:10.1023/A:1018966204465
* CODATA 2018 / SI 2019 for `R = 8.314462618 J mol-1 K-1`.

Upstream implementations transcribed or compared against: ONEFlux_preproc v0.10
(P. H. Coimbra et al.); EddyPro 7.0.9 (LI-COR Biosciences, G. Fratini); GEddySoft v4.1
(B. Heinesch, Université de Liège — Gembloux Agro-Bio Tech).
