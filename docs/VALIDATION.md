# miniflux — VALIDATION

**miniflux is not validated against a certified reference.** None exists for an eddy-covariance
flux. The references are two programs the community uses and that miniflux was extracted from:
**ONEFlux_preproc**, the parent this reimplements, and **EddyPro 7.0.9** (LI-COR), whose output
ships with the FR-Gri sample. **This document is not a reference**, and **every deviation below is
miniflux's to explain, not theirs** — each is traced to the convention behind it, or named a
FINDING and left standing. Every run below reproduces byte-identically on a pristine `72def39`
extract, and **no setting was chosen to make the agreement look better**; §1.4(b) is the
unflattering case. Suite there: **542 tests OK in 3.0 s**, numpy present, so both kernel paths run.

## 1. FR-Gri, 2022-05-14 — closed path, against EddyPro 7.0.9

`python -m miniflux run examples/fr_gri_closedpath_dry.ini` over the 48 `FR-Gri_EC_20220514*.csv` —
a full day, `N_IN = 36000` every period, none missing, none skipped, 2 min 16 s, against
`eddypro_..._full_output_...adv.csv` plus `_fluxnet_` for spike counts. An LI-7200 on 71.1 m of
tube: the first run to put a gas flux against a reference at all. `[spectral]` is `false`, the
default — **these gas fluxes are attenuated, and the run says so once, at WARNING.** Site values
are verbatim from the metadata (`latitude = 48.844243`, `measurement_height = 2.00`, `north_offset
= 250.0`, `canopy_height = 0.15` → `z − d = 1.90 m`, matching EddyPro's own 2.54698 × 0.745786) and
the methods match its `.eddypro` one for one: double rotation, block average, Mauder 2013 MAD at q
= 7.0, `covmax_default`, 30 min.

### 1.1 A finding first: the reference's rows are labelled 30 minutes late

`FR-Gri_EC_202205141200_L05_F01.csv` holds 11:30:00.05 to 12:00:00.00 — the name is the **end** of
its half hour, but the metadata declares `tstamp_end = 0` and EddyPro never reads `TIMESTAMP`
(`col_ts = 0`), so it cannot notice. Found, not assumed: RMSE/RMS over offsets {−30, 0, +30, +60}
min on four independent quantities gives, at +30 min, `T_SONIC` **0.00001**, `w/ts_cov` **0.0045**,
`co2_mixing_ratio` **0.0010**, `h2o_mixing_ratio` **0.0006**, against 0.023–0.41 elsewhere. **Every
comparison below pairs miniflux's period ending at T with EddyPro's row labelled T + 30 min.**

### 1.2 The comparison

48 periods, no exclusions. Deviation is `100 × (miniflux − EddyPro) / EddyPro`; `n` counts periods
whose EddyPro value clears zero enough to form a ratio (`|H|,|LE| ≥ 5 W m-2`, `|FC| ≥ 0.5`, `|E| ≥
0.2 mmol m-2 s-1`, `|u*| ≥ 0.02`). Last column: the same against EddyPro's separately published
uncorrected output.

| quantity | miniflux | EddyPro | n | med % | [p16, p84] | min … max % | vs `un_*` med % (n) |
|---|---|---|---|---|---|---|---|
| friction velocity | `USTAR` | `u*` | 48 | **−0.83** | [−1.61, −0.61] | −4.39 … −0.12 | **+0.00** (48) |
| sonic heat covariance | `COV_W_TS` | `w/ts_cov` | 33 | +0.003 | [−0.044, +0.003] | −5.10 … +1.42 | — |
| sensible heat | `H` | `H` | 36 | **−1.42** | [−3.83, −1.18] | −12.04 … +3.15 | **+0.77** (36, `H_L0`) |
| latent heat | `LE` | `LE` | 29 | **−8.80** | [−10.57, −8.41] | −33.78 … −5.98 | **+1.08** (27) |
| evaporation | `E` | `h2o_flux` | 26 | **−8.74** | [−9.34, −8.37] | −16.38 … −5.95 | **+1.08** (26) |
| CO2 flux | `FC` | `co2_flux` | 44 | **−4.97** | [−23.78, −1.08] | −60.13 … +72.28 | +5.68 (44) |
| CO2 flux, same lag | `FC` | `co2_flux` | 5 | **−9.82** | [−21.49, −8.91] | −24.65 … −8.38 | **+0.90** (5) |

| quantity, as a difference (angles circular) | n | med | [p16, p84] | min … max |
|---|---|---|---|---|
| `WD` − `wind_dir` | 48 | **−1.360°** | [−5.269, +3.349] | −22.166 … +11.320° |
| `THETA` − `yaw` | 48 | **+0.0000°** | [−0.0002, +0.0001] | −0.0165 … +0.0051° |
| `PHI` − `pitch` | 48 | **+0.0000°** | [−0.0001, +0.0000] | −0.3141 … +0.0475° |
| `CO2_LAG` − `co2_time_lag` | 48 | **−0.250 s** | [−0.300, +0.000] | −1.800 … +1.350 s |
| `H2O_LAG` − `h2o_time_lag` | 48 | **+0.000 s** | [−0.400, +0.624] | −2.650 … +1.100 s |

**Mean state,** per cent, median (|max|): `WS` **−0.127** [−0.329, −0.020] (5.587), `TA` −0.002
(0.032), `T_SONIC` +0.000 (0.004), `CP` +0.002 (0.004), `CO2_MEAN` +0.0000 (0.564), `H2O_MEAN`
+0.0000 (0.348), `PA` **+0.889** [+0.544, +0.974], `RHO_A` **+0.872**, `RH` **+1.080** [+0.925,
+1.305], `Q` **+0.319** — see §1.4(b), (c). **Spikes,** day totals (`N_SPIKE_*` / `*_SPIKE_NREX`):
u 3/3, v 10/10, w **918/918**, ts 1039/1038, co2 5958/5957, h2o **5900/5900**; per period identical
on 48/48 for u, v, w, h2o and 47/48 for ts, co2.

### 1.3 The sign verdict: does miniflux read low on the closed-path gas fluxes?

EddyPro spectrally corrects (`hf_meth = 1`, Moncrieff 1997, from the declared tube) and miniflux
does not, so the error is one-signed: miniflux **must** read low. This is worth more than the
table.

| flux | periods low | well developed | lag also matched | verdict |
|---|---|---|---|---|
| `LE` | **29 / 29** | 25 / 25 (`\|LE\| ≥ 20`) | 21 / 21 | **PASS** |
| `E` | **26 / 26** | 25 / 25 | — | **PASS** |
| `FC` | 38 / 44 | 31 / 34 (`\|FC\| ≥ 5`) | **5 / 5** | **PASS**, with §1.4(d) |
| `H` | 35 / 36 | — | — | pass — sonic path, not the tube |
| `USTAR` | **48 / 48** | — | — | pass — sonic path |

`LE` and `E` are low on every period there is, no exception, tight spread. **`FC` is low on every
period where both applied the same lag, and all six exceptions are lag disagreements** (§1.4d): the
well-developed ones are 12:30 (+0.49 %), 13:00 (+2.03 %) and 16:00 (+0.56 %), where a better lag
outweighed the missing correction. **No flux here reads high for any other reason.**

### 1.4 Every deviation, explained

**(a) The spectral correction is the whole of it, and EddyPro's own factors predict it** — a
missing feature, not an error (`ALGORITHMS §10A`). Deviation against `(PA_mf / PA_ep) / scf − 1`,
nothing fitted:

| | observed med % | predicted med % | residual med | residual [p16, p84] |
|---|---|---|---|---|
| `USTAR` (n=48) | −0.83 | −0.83 | **+0.001 pp** | [−0.205, +0.002] |
| `H` (n=22, `\|H\| ≥ 20`) | −1.31 | −1.31 | **+0.031 pp** | [−0.047, +0.069] |
| `LE` (n=25, `\|LE\| ≥ 20`) | −8.78 | −9.01 | +0.297 pp | [+0.262, +0.312] |
| `FC` (n=5, lag-matched) | −9.82 | −10.13 | +0.274 pp | [−4.591, +0.345] |

`Tau_scf` med 1.0168 (max 1.0607), `H_scf` med 1.0240 (max 1.1213), `LE_scf` = `co2_scf` =
`h2o_scf` med **1.1126** [1.1062, 1.2454], max 1.6422 — a **10.1 % median loss** on both gases, the
figure `ALGORITHMS §10A.1` states; the +0.3 pp left over is (c). **A miniflux closed-path `FC` or
`LE` used for science is an under-corrected flux**, here by ~10 %.

**(b) `PA` +0.89 %: a pressure choice that costs agreement, kept anyway.** Both closed-path configs
read `p_air = PRESS_BOX`, the LI-7200's own near-ambient barometer (100.65 kPa that day); EddyPro
had no ambient-pressure input (`col_air_p = 0`) and fell back to a constant **99767.5 Pa** from the
station altitude. `PRESS_BOX` is the better measurement and the worse agreement, moving `PA`,
`RHO_A`, `H_L0`, `H`, `RH`, `LE`, `E` and `FC` by the same +0.89 %, and was **not** changed: with
`fr_gri_wind_sonic.ini`, declaring EddyPro's 99767.5 Pa instead, `H_L0` sits **+0.0052 %** from
`un_H` rather than +0.774 %, `USTAR` +0.0013 %, `RH` +0.316 %. **So 0.9 % of every density-scaled
flux here is a choice about barometry — more than the whole turbulence processing, and the better
measurement is the one that disagrees.**

**(c) FINDING — EddyPro builds two adjacent quantities on the sonic temperature: the +0.3 pp on the
gases, and all of `RH` and `Q`.** `Ts − Ta` is 0.95 K on 290.1 K, 0.325 %. Its `air_molar_volume`
matches `R·Ts/P` to **−0.006 %** and `R·Ta/P` to **+0.320 %**, where miniflux converts a dry mixing
ratio through the dry-air molar volume at the **ambient** temperature (`ALGORITHMS §9.1`);
EddyPro's gas fluxes are 0.33 % smaller, and adding that to (a) takes the residual to **−0.003 pp**
on `LE` and lag-matched `FC`. Its `e` likewise: against `chi_wet · P` at EddyPro's own pressure the
deviation is **+0.2991 %**, with a `Ta/Ts` put back **−0.0257 %**. So `RH` = pressure (+0.897) +
this (+0.299) − 0.043 % of `es`, leaving **+0.003 pp**; `Q`, pressure-free, shows +0.32 % alone;
humidity reaches `H` only through `cp` and the Schotanus denominator, moving them 0.002 %. Which is
right is not decided here (`ALGORITHMS §8.3` is about this trap), but they cannot both be.

**(d) FINDING — the two lag searches differ, and that dominates `FC`.** EddyPro reports
`co2_def_timelag = 1`, its default instead of a found peak, on **38 of 48** periods (0.400 s) and
19 of 48 for water (0.700 s); miniflux fell back to its nominal 0.300 s on 5 and 6 of 48. The
metadata declares `nom/min/max_timelag = 0.00`, so that 0.4 s is EddyPro's own; its output does not
say which. The curves differ in shape: `w`–`CO2_DRY` peaks sharply at 0.15 s and loses 8.5 % by
0.40 s; `w`–`H2O_DRY` is flat-topped from 0.20 to 0.50 s, losing under 3 % in it. **Water agrees
whatever the lag and CO2 does not**, hence `FC`'s spread of [−24, −1] % against `LE`'s [−10.6,
−8.4] %. The ±2.0 s window was **not** narrowed to match — that would be buying agreement — and the
price shows: on the 5 night periods where miniflux took a peak beyond ±1.0 s (03:00, 04:30, 20:30,
22:00, 23:00), all `|FC| < 4`, it disagrees with `un_co2_flux` by +0.3, +8, +60, +91 and **+1845
%**. On a weak flux the maximisation finds noise, and miniflux publishes it.

**(e) Despiking: the same samples flagged, different things done with them.** miniflux NaNs a
flagged sample and masks it pairwise, so it is **dropped**; EddyPro (`filter_sr = 1`) **replaces
each flagged run by a straight line**. That is every deviation above 0.1 % in `COV_W_TS`, `PHI` and
the variances, and both `CO2_MEAN` outliers, −0.564 % at 02:30 (2156 flagged samples) and −0.364 %
at 21:00 (1702). With nothing flagged the residual is a constant **+0.00278 %** = `N/(N−1) − 1` at
`N = 36000`, EddyPro dividing the second moments by `Nact` where `ALGORITHMS §7.1` prescribes `N_p
− 1`. The one-sample ts and co2 difference fits its `do i = 2, N-1`, which skips each period's end
samples.

**(f) `WD` differs by up to 22°: the two average directions differently.** `WS` agrees to −0.127 %,
so the mean *vector* is the same and only the estimator differs: miniflux takes the direction of
the mean vector (`ALGORITHMS §4.5`), EddyPro a unit-vector angular mean of the 36000 sample
directions (`src_common/wind_direction.f90`) — recomputed on four periods, that reproduces
`wind_dir` to **0.000°** and the vector mean reproduces `WD`. They diverge in wandering wind, and
this day is exceptionally calm.

### 1.5 The same measurement, declared twice

`examples/fr_gri_closedpath_cell.ini` reads the **same 48 half hours** from `CO2_CONC`/`H2O_CONC`,
cell molar densities, with `T_CELL` and `PRESS_CELL`, converted per sample to a dry mixing ratio
(`ALGORITHMS §2A.3`), `N_CELL_CONV = 36000` every period. **One measurement written down twice: the
two tables must agree, and by how much is the test of the conversion.** Cell relative to dry:

| quantity | n | med % | [p16, p84] | \|max\| % | \|max\| absolute |
|---|---|---|---|---|---|
| `FC` (`\|FC\| ≥ 1 µmol`) | 43 | **+0.0244** | [−0.0034, +0.0357] | 0.490 | 0.013 µmol m-2 s-1 |
| `LE` and `E` (`\|LE\| ≥ 5`) | 27 | **+0.0221** | [+0.0180, +0.0250] | 0.033 | 0.046 W m-2 |
| `CO2_MEAN` / `H2O_MEAN` | 48 | +0.0197 | [+0.0190, +0.0201] | 0.021 | 0.099 µmol mol-1 |
| `H` | 35 | −0.0011 | [−0.0028, +0.0004] | 0.020 | 0.003 W m-2 |

`USTAR`, `COV_W_TS` and the variances come out **bit-identical**, as they must — no wind or sonic
quantity passes through the conversion. The **+0.0197 %** offset is one constant on both
concentrations, which puts the whole residual in `V_cell = R·T_cell/P_cell`, the analyser's cell
state against the one it logs, and none of it in the dilution algebra (`ALGORITHMS §2A.8` measures
+0.0194 % sample by sample). **The conversion is right to about two parts in ten thousand.**
Without the `|FC| ≥ 1` cut the worst over all 48 is −4.75 % at 02:00 — 0.006 µmol m-2 s-1 on a flux
of 0.13, a deviation of a number that is not there. The lag differs between the shapes on 3 periods
for CO2 and 6 for water, the conversion having changed what the MAD test flags.

**Both kernel paths agree on these runs.** `CONTRACT §3.8` promises identical elementwise results
from the numpy and pure-Python kernels; the first 12 periods of both configs re-run with `--pure`
give **1488 of 1488 cells byte-identical** at `%.6g`, `miniflux/cell.py` included.

## 2. FR-Jus, 2025-09-08 — open path, end to end, no reference

`python -m miniflux run examples/fr_jus_openpath.ini` over 7 TOA5 files, 20 Hz, Campbell IRGASON
(open path, collocated). `co2` and `h2o` are declared `molar_density`, so **WPL is owed and runs**;
everything else is default. A plausibility run, not a measurement: no reference output, no
radiation, no closure. `measurement_height = 30.0` and `latitude = 48.85` (Paris, from the `FR-`
prefix and the logger program) are **assumptions** entering `ZL` and the ITC test alone;
`north_offset = 0.0`, so **`WD` is in the sonic frame, not geographic.**

Seven periods, all but the first after sunset: `WS` 2.10–3.62, `USTAR` 0.357–0.548 m s-1, `TA`
294.5 → 292.0 K, `RH` 42 → 53 %, `CO2_MEAN` 415.5–416.9 ppm, `H` +26 to +46 W m-2 (one negative),
`FC` −3.17 to +3.66 µmol m-2 s-1, `LE` −20 to +21 W m-2, `L` −449 to +190 m. **Sensible for an
urban site and not a rural one:** positive `H` at night with `z/L` negative would be a red flag
over grassland and is normal where anthropogenic heat holds the layer near-neutral; `u*/WS` of
0.11–0.19 against a rural 0.05–0.10 is the built-up fetch.

**The honest caveat on `FC`: WPL is the larger term.** `FC − FC_L0` is +1.28 to +2.35 µmol m-2 s-1
against an `FC` of order 1 to 4, **and it flips the sign of the flux on four of the seven periods**
— the known open-path situation when `H` is large and the CO2 flux small, and the right size: Webb
Eq. 24's thermal part predicts +1.96 for 20:30 against +1.78 observed. So `FC`'s accuracy rests on
`H`, not on `w'c'`; Schotanus is tiny beside it, `H − H_L0` at most 3.0 %.

**What went wrong, and what miniflux refused.** The run **exits 3 (`ReadError`), by design**, on
the last file, whose final line the logger cut off mid-write; the seven complete periods were
already written, and the config still lists all seven rather than dodging the bad file. The 21:00
period holds **2375 samples, not 36000**, a 50-minute hole, and is the outlier in every column,
reported not dropped (`min_samples = 0`) so `N_IN` disqualifies it. Every file arrives at **19.8 Hz
against the declared 20 Hz** (~1 % dropped) and miniflux uses the declared rate, so the lag search
shifts by sample index; it hits the window edge on four of seven periods and falls back to the
nominal 0 s, right for a collocated head and a statement that the weak flux has no lag structure.
`SST_FLAG` is 2 on three periods and 1 on two more, the Foken–Wichura denominator vanishing near
zero covariance; `ITC_T` reaches 1.53, `CONTRACT §20.5` using `|T*|` for the parent's signed `T*`.

## 3. What this does not prove

1. **Not validated against a certified reference, because none exists.** One day at one site
against EddyPro 7.0.9, seven half hours end to end at another; nothing licenses using miniflux
where EddyPro or `oneflux_preproc` is called for. Where §1.4(c) names a convention miniflux departs
from, that is a stated divergence with its arithmetic shown, not a claim to be the better program.
And agreement with an implementation is not agreement with the atmosphere: both could share a wrong
convention unseen.

2. **The closed-path gas fluxes are ~10 % low, and that is the design, not a bug.** miniflux
applies at most a first-order factor, in its own column (`ALGORITHMS §10A`), and `[spectral]` was
**off** above; the 10.1 % belongs to this tube, flow, tower and calm day.

3. **`FC` here compares two lag searches as much as two flux calculations.** The same CO2 lag was
applied on 5 of 48 periods, so §1.3's verdict rests on those 5, on the 34 well-developed periods,
and on a covariance curve from the raw file. §1.4(d) also shows miniflux's own search taking noise
for a peak on 5 nights.

4. **The WPL correction is unvalidated against a reference.** §2 shows it dominating `FC` and
flipping the flux sign on four of seven periods, and FR-Gri is closed path, so nothing in §1
touches `miniflux/wpl.py`. If WPL is wrong, every open-path `FC` miniflux produces is wrong and
nothing here would catch it.

5. **§1.5 proves the two declarations agree, not that either is right.** Both come from one
instrument and one firmware: if the LI-7200's own cell-state arithmetic is wrong they are wrong
together, and the 0.024 % would not move.

6. **Two days is two days**, two instruments, two sites, one season each: no cold, no high wind, no
strong stability, no rain, no sensor failure beyond one truncated record. FR-Gri is exceptionally
calm (`WS` ≤ 1.5 m s-1), which inflates the spectral correction and the §1.4(f) gap. FR-Jus is a
plausibility argument: sign and order of magnitude would not detect a 10 % error, and two of its
site values are invented, so `ZL` and its ITC columns are conditional on them.

7. **A comparison this close can hide a shared mistake in the *inputs*.** §1.1's label offset was
found only because four independent quantities agreed about it; a column mapped to the wrong
variable would look like a processing difference and be explained away. The defence is that §1.4
accounts for every deviation, down to a named convention.

8. **Nothing here tests the refusals.** The unit tests do (`CONTRACT §21`); this document exercised
one in the field, FR-Jus's truncated final record, and it behaved as documented.
