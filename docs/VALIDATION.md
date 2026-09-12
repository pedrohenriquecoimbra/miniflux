# miniflux — VALIDATION

What miniflux has been run on, what it was compared against, what agreed, what did not, and
what none of it proves.

**Read this first.** miniflux is **not validated against a certified reference.** There is no
certified reference for an eddy-covariance flux. What exists here is a comparison against two
programs that the community uses as references and that miniflux was extracted from:

* **ONEFlux_preproc** (the parent this program is a minimal reimplementation of), and
* **EddyPro 7.0.9** (LI-COR), whose output is shipped with the FR-Gri sample.

Those are the references. **This document is not.** Where miniflux and EddyPro disagree, the
prior is that EddyPro is right and miniflux is the thing that needs explaining — and every
disagreement below is either explained down to the line of source that causes it, or named as
unexplained.

Companion documents: `ALGORITHMS.md` (the mathematics) and `CONTRACT.md` (the module API).
Both are law; nothing in this document may be used to argue for changing them.

Everything below was produced with the repository exactly as committed. `miniflux/` and
`tests/` were not touched. Test suite at the time of the runs: `python -m unittest discover
-s tests -t .` → **478 tests, OK, 3.5 s**, with numpy present, so both kernel paths are
exercised.

---

## 1. Summary

| run | data | reference | headline |
|---|---|---|---|
| §2 FR-Jus | IRGASON open path, 7 half hours, 2025-09-08 evening | none | fluxes are physically sensible for an urban night; WPL is the dominant term in `FC` |
| §3 FR-Gri | Gill HS-50 sonic, 48 half hours, 2022-05-14 | EddyPro 7.0.9 | after the reference's own spectral correction is undone, `cov_w_ts` agrees to a median **+0.003 %**, `u*` to **+0.001 %**, `H_L0` to **+0.005 %** |

The configurations used are committed beside this document:

* `examples/fr_jus_openpath.ini`
* `examples/fr_gri_wind_sonic.ini`
* `examples/fr_gri_wind_sonic_pbox.ini` (pressure-sensitivity variant of the previous one)

**No setting was chosen to make the agreement look better.** Every setting that is not a
documented default is justified in the `.ini` comment and again below. Three settings were
taken *from the reference's own metadata* rather than guessed — the site latitude, the
instrument height and the sonic north offset for FR-Gri — and one, the FR-Gri ambient
pressure, was taken from the reference's own *output*; §3.6 measures exactly how much that
last choice is worth, with a second run that does not make it.

---

## 2. FR-Jus, 2025-09-08 — end-to-end plausibility run, open path

### 2.1 What was run

```
python -m miniflux run examples/fr_jus_openpath.ini
```

Input: `ONEFlux_preproc/data/sample/FR-Jus_20250908/FLUX/*.csv` — 7 TOA5 files, 20 Hz,
Campbell IRGASON (open path, sonic and analyser collocated in one head). Header on line 2,
data from line 5. 4.2 s wall clock for the whole run.

Declared: `u,v,w = RAW_IRGASON_Ux/Uy/Uz`, `ts = RAW_IRGASON_Ts` (degC),
`co2 = RAW_IRGASON_CO2` (mg m-3), `h2o = RAW_IRGASON_H2O` (g m-3), both as
`molar_density`, so **WPL is owed and runs**. Everything else is the documented default:
double rotation, block detrending, MAD despiking at q = 7, `covmax_default` lag over
±2.0 s, `wpl = auto`, 30-minute periods closed on the right.

Two site values are **assumptions, not data** — the sample ships no tower geometry:

* `measurement_height = 30.0 m`, `displacement = 0.0 m`. These enter `ZL` and the ITC test
  and **nothing else**; `MO_LENGTH` does not depend on them at all, and no flux does.
  `ZL` below is therefore conditional and scales linearly with whatever `(z - d)` really is.
* `latitude = 48.85` (Paris, inferred from the `FR-` prefix and the `ICOSCitiesV031` logger
  program name). Enters the ITC Coriolis term only.

`north_offset` is left at 0.0 because the sonic's azimuth is unknown, so **`WD` below is in
the sonic frame and is not a geographic wind direction.**

### 2.2 The pressure decision, and why

The file carries `RAW_IRGASON_CellPrs` in kPa. On a *closed-path* analyser a cell pressure is
not ambient and using it would be wrong — the pump holds the cell below ambient. This is an
**open-path** IRGASON: the measurement volume is the open gap between the transducers, and
the EC100 pressure sensor reads the ambient pressure at the head. The numbers say the same
thing: the column runs 100.905–100.932 kPa, which is an ordinary September barometric
pressure for a low-altitude site, not a pumped cell (the FR-Gri closed-path cell in §3 sits
1.4 kPa *below* its own box). **I used it**, and left `[site] pressure_pa` empty.

It matters little either way. A control run replacing the column with a declared constant of
101325 Pa (+0.4 %) moves the outputs by:

| quantity | change, measured column → declared 101325 Pa |
|---|---|
| `PA` | +0.39 to +0.42 % (by construction) |
| `RHO_A` | +0.39 to +0.42 % |
| `H` | **+0.38 to +0.42 %** |
| `FC` | −0.11 to +0.04 % |
| `LE` | −0.01 to 0.00 % |
| `USTAR` | 0.000 % |

`H` tracks the density exactly, as it must. The momentum flux is untouched. So the pressure
choice here is worth a few tenths of a percent on `H` and nothing on `u*` — but miniflux
still refuses to run without one being *stated*, which is the point of `CONTRACT §20.11`.

### 2.3 What the run produced

Seven periods. The clock in the file is the logger's; whether it is local (CEST) or UTC is
not recorded. Either way everything from the second period on is after sunset.

Geometry and turbulence:

| period END | N_IN | WS | WD (sonic frame) | THETA | PHI | USTAR | u*/WS |
|---|---|---|---|---|---|---|---|
| 2025-09-08 20:00 | 33258 | 2.869 | 239.365 | −59.365 | 1.150 | 0.5067 | 0.177 |
| 2025-09-08 20:30 | 35640 | 2.504 | 248.876 | −68.876 | 2.402 | 0.4752 | 0.190 |
| 2025-09-08 21:00 | **2375** | 2.096 | 261.531 | −81.531 | 2.644 | 0.3569 | 0.170 |
| 2025-09-08 21:30 | 33265 | 3.619 | 252.159 | −72.160 | 0.109 | 0.3831 | 0.106 |
| 2025-09-08 22:00 | 35639 | 2.854 | 248.582 | −68.582 | 2.158 | 0.5483 | 0.192 |
| 2025-09-08 22:30 | 35639 | 2.944 | 248.837 | −68.837 | 1.623 | 0.5026 | 0.171 |
| 2025-09-08 23:00 | 35640 | 2.813 | 238.863 | −58.863 | 1.902 | 0.5182 | 0.184 |

Mean state and fluxes:

| period END | TA (K) | RH (%) | CO2 (ppm) | H2O (ppt) | H_L0 | H | FC_L0 | FC | LE_L0 | LE | L (m) | ZL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20:00 | 294.50 | 41.6 | 416.9 | 10.55 | 29.42 | **29.39** | 2.246 | **3.662** | −1.00 | **0.58** | −399.5 | −0.075 |
| 20:30 | 293.98 | 46.7 | 416.3 | 11.47 | 39.65 | **40.88** | −0.640 | **1.134** | −22.17 | **−20.01** | −244.5 | −0.123 |
| 21:00 | 293.63 | 49.0 | 416.0 | 11.77 | −21.57 | **−21.35** | −2.113 | **−3.172** | −2.40 | **−3.72** | 190.4 | 0.158 |
| 21:30 | 292.80 | 51.9 | 415.6 | 11.87 | 26.02 | **25.75** | −1.156 | **0.123** | 2.87 | **4.48** | −195.3 | −0.154 |
| 22:00 | 292.49 | 52.9 | 415.8 | 11.87 | 33.20 | **32.62** | 0.820 | **2.478** | 7.43 | **9.52** | −448.7 | −0.067 |
| 22:30 | 292.19 | 52.7 | 415.9 | 11.59 | 46.05 | **44.77** | −0.812 | **1.542** | 18.16 | **21.06** | −249.1 | −0.120 |
| 23:00 | 291.95 | 52.8 | 415.5 | 11.44 | 43.84 | **42.77** | 0.100 | **2.326** | 14.85 | **17.56** | −286.9 | −0.105 |

`H`, `LE` in W m-2; `FC` in µmol m-2 s-1. `WPL_APPLIED = 1` on every period.
Spike counts over the whole run: u 0, v 0, w 0, ts 0, co2 2, h2o 16 samples out of ~211 000.
`N_DUP = 0` everywhere.

### 2.4 Are these physically sensible?

**Yes, for an urban site — and they would not be for a rural one.** Read in that order:

* **`H` is positive at night, +26 to +46 W m-2, and `z/L` is negative (unstable).** At a
  grassland or forest site this would be a red flag: nocturnal `H` there is negative and the
  surface layer is stable. At a city site it is the expected signature — the urban fabric
  releases stored heat through the night and anthropogenic heat adds to it, keeping the
  surface layer near-neutral to weakly unstable long after sunset. The one negative `H`
  (−21 W m-2) is the 2375-sample fragment; see below.
* **`u*/WS` is 0.11–0.19.** A rural site sits near 0.05–0.10. Values near 0.18 mean a very
  rough surface, which is what a built-up fetch is. `u*` and `WS` are internally consistent
  across all seven periods.
* **CO2 is 415–417 ppm and falling slowly; H2O 10.5–11.9 ppt; RH 42→53 % as the air cools
  from 294.5 to 292.0 K.** Every one of those is an ordinary early-September evening, and RH
  rising while the absolute humidity barely moves is just the temperature falling.
* **`FC` is +1.1 to +3.7 µmol m-2 s-1 (uptake nowhere, emission everywhere except the
  fragment).** Correct sign for a city at night: no photosynthesis, plus traffic and
  respiration. The magnitude is at the low end of published urban fluxes, which is consistent
  with a high tower whose night-time footprint is large and not traffic-dominated — but see
  the caveat immediately below, because this number is *mostly WPL*.
* **`LE` is −20 to +21 W m-2, small and of inconsistent sign.** That is what a nocturnal
  latent heat flux of a few W m-2 looks like at 30-minute resolution. The −20 W m-2 period at
  46 % RH is a downward water vapour flux with no dew to explain it; at this magnitude I read
  it as noise plus large-scale advection of drier air, not as a defect — and not as evidence
  of correctness either.
* **`PHI` (pitch) is 0.1–2.6°.** A fixed sonic tilt would be constant; the variation is real
  mean vertical wind, which is exactly what the second rotation exists to remove.

**The honest caveat on `FC`: the WPL correction is the larger term.**

| period END | FC_L0 | FC | WPL contribution | as a share of FC |
|---|---|---|---|---|
| 20:00 | 2.246 | 3.662 | +1.416 | 39 % |
| 20:30 | −0.640 | 1.134 | +1.775 | 156 % (sign flip) |
| 21:00 | −2.113 | −3.172 | −1.059 | 33 % |
| 21:30 | −1.156 | 0.123 | +1.279 | 1037 % (sign flip) |
| 22:00 | 0.820 | 2.478 | +1.659 | 67 % |
| 22:30 | −0.812 | 1.542 | +2.354 | 153 % (sign flip) |
| 23:00 | 0.100 | 2.326 | +2.226 | 96 % |

On four of seven periods WPL **changes the sign of the CO2 flux**. That is not a miniflux
problem — it is the well-known open-path situation when the sensible heat flux is large and
the CO2 flux is small, and the term is the right size: the thermal part of Webb Eq. 24 is
`(1 + µσ)·(c̄/T̄)·w'T'` ≈ 1.011 × (0.0171 mol m-3 / 293 K) × 0.0332 K m s-1 ≈
**+1.96 µmol m-2 s-1** for the 20:30 period, against the +1.78 observed. It means the
accuracy of `FC` here rests on the accuracy of `H`, not on the accuracy of `w'c'`. The
Schotanus correction on `H` itself is tiny by comparison: `H − H_L0` is −1.3 to +1.2 W m-2,
at most 3.0 % of `H`.

### 2.5 What went wrong, and what miniflux refused

* **The run exits 3 (`ReadError`), by design, on the last file.** The final line of
  `20213_RAW_20Hz_2025-09-08-2330.csv` is `"2025-09-08 23:03:53.8` — a record the logger was
  cut off in the middle of writing, with no closing quote and no newline. miniflux reports
  `line 2258: the row holds 1 field(s) but the configuration reads column 12` and stops. The
  seven complete periods above were written before it stopped. This is the refusal principle
  working: the alternative is a period silently built from a half-parsed row. The config is
  left pointing at all seven files rather than dodging the bad one, because dodging it would
  hide the finding.
* **The 2025-09-08 21:00 period holds 2375 samples (2 minutes), not 36000.** There is a
  50-minute hole in the input: no file covers 20:32–21:01. Its flux is the outlier in every
  column — the only negative `H`, a positive (wrong-signed) `COV_U_W`, a `u*` carried
  entirely by `cov_v_w`. A 2-minute record cannot resolve the flux-carrying eddies. It is
  reported rather than dropped (`min_samples = 0`) so that `N_IN` is what disqualifies it;
  a user who wants it gone should set `min_samples` and say so.
* **Every file is logged as arriving at 19.8 Hz against the declared 20 Hz** (35 633–35 640
  rows per 30 min instead of 36000, ~1 % of records dropped by the logger). miniflux uses the
  declared 20 Hz, as documented. Two consequences worth stating: the lag search shifts by
  *sample index*, so on a gappy record a shift of `n` samples is not exactly `n/20` seconds;
  and `SST`/`ITC` sub-intervals are likewise index-based.
* **The lag search hits the window edge on four of seven periods** and the `covmax_default`
  rule falls back to the nominal 0 s, logging each time. For a collocated open path the true
  lag *is* ~0, so the fallback is doing the right thing — but it is also telling us the
  covariance curve has no interior peak, i.e. there is no detectable scalar-wind lag
  structure in a weak nocturnal flux. The optimum, where one was found, was −1.9 to +0.1 s,
  scattered; the ±2.0 s window is the documented default and was left alone.
* **`SST_FLAG` is 2 (non-stationary, >100 %) on three of seven periods** and 1 on two more.
  With `steady_state_pair = w,co2` and a CO2 covariance near zero, the Foken–Wichura ratio has
  a vanishing denominator. Expected; reported, not suppressed.
* **`ITC_T` reaches 1.53.** Near-neutral periods make the temperature ITC model ill-posed,
  and `CONTRACT §20.5` uses `|T*|` where the parent uses signed `T*`, so this column is not
  comparable with the parent's on unstable periods by construction.

---

## 3. FR-Gri, 2022-05-14 — comparison against EddyPro 7.0.9

### 3.1 What was run

```
python -m miniflux run examples/fr_gri_wind_sonic.ini
```

Input: the 48 files `FR-Gri_EC_20220514*.csv` — a **full day**, 48 half hours, 36000 samples
each, none missing, `N_IN = 36000` on every period. 38 s wall clock.

Reference:
`OUTPUT/EddyPro7/eddypro_FR-Gri_sample_full_output_2025-12-02T001726_adv.csv`, plus the
matching `_fluxnet_` file for the per-sample spike counts and the `_qc_details_` file for the
Foken–Wichura percentages.

Site values taken verbatim from the reference's own metadata (`eddypro/FR-Gri.metadata`):
`latitude = 48.844243`, `measurement_height = 2.00` (`instr_1_height`),
`north_offset = 250.0` (`instr_1_north_offset`), `canopy_height = 0.15`. That last one gives
`displacement = 0.10 m` and `(z − d) = 1.90 m`, which is exactly what EddyPro's own `L` and
`(z-d)/L` columns imply (2.54698 × 0.745786 = 1.8995) — EddyPro derives `d = 2h/3` too, even
though the metadata's `displacement_height` field says 0.

Method settings were matched to the reference's `.eddypro` processing file, one for one:

| EddyPro setting | value | miniflux |
|---|---|---|
| `rot_meth` | 1 (double rotation) | `[rotate] method = double` |
| `detrend_meth` | 0 (block average) | `[detrend] method = block` |
| `despike_vm` | 1 (Mauder et al. 2013) | `[despike]` MAD, `q = 7.0` |
| `tlag_meth` | 2 (maxcov & default) | `[lag] method = covmax_default` |
| `avrg_len` | 30 min | `[period] averaging_minutes = 30` |

So the two programs are being asked to do the *same* thing, not two different things. The
task brief anticipated "a different despiking and detrending default"; on this particular
reference run there is no such difference — detrending is block on both sides and the
despiking is the same paper at the same threshold. What remains is documented below, and it
is smaller and more specific than that.

### 3.2 What is deliberately not compared

The gas analyser is a **closed-path LI-7200 with 71.1 m of tube**. miniflux has no
tube-attenuation correction and the refusal on closed-path input stands. **`FC`, `LE`, `E`,
`FC_L0`, `LE_L0`, `E_L0` and the two gas lags from this run are discarded and appear nowhere
below.** They are numbers; they are not fluxes.

CO2 and H2O still had to be *declared*, for two reasons, both stated in the `.ini` header:
`[variables]` requires all six high-frequency columns (`CONTRACT §5.2.6`), and `H` needs a
water vapour flux for the Schotanus correction and a humidity for `rho` and `cp` — running
with no water at all would be a larger error than running with an attenuated one. `CO2_DRY`
and `H2O_DRY` are dry mole fractions, so `wpl = auto` correctly decides nothing is owed.
The size of the contamination this buys is bounded in §3.5(f): `H − H_L0` is the whole of it,
and the comparison of `H_L0` against EddyPro's `un_H` is free of it entirely.

### 3.3 A finding before the comparison: the reference's rows are labelled 30 minutes late

`FR-Gri_EC_202205141200_L05_F01.csv` contains samples timestamped 11:30:00.05 through
12:00:00.00 — the file name is the **end** of its half hour. The site metadata declares
`tstamp_end = 0`, i.e. *file name = start*, and EddyPro never reads the `TIMESTAMP` column
(`col_ts = 0`), so it cannot notice. EddyPro therefore labels that file's output row
**12:30**. miniflux labels by the sample clock and calls it **12:00**.

This was found, not assumed. Scanning the offset over {−30, 0, +30, +60} minutes and scoring
RMSE/RMS against three quantities:

| offset applied to the EddyPro key | T_SONIC | COV_W_TS | VAR_W |
|---|---|---|---|
| −30 min | 0.0041 | 0.4068 | 0.3544 |
| 0 | 0.0023 | 0.2649 | 0.2828 |
| **+30 min** | **0.0000** | **0.0045** | **0.0039** |
| +60 min | 0.0023 | 0.2647 | 0.2826 |

Unambiguous. **Every comparison below pairs miniflux's period ending at T with EddyPro's row
labelled T + 30 min.** This is a property of how the reference dataset was configured, not of
either program's arithmetic, but it is worth recording: without it every number in §3.4 is
wrong by an amount that looks like a processing difference (`cov_w_ts` would have shown a
median deviation of −10.7 % instead of +0.003 %) and would have been reported as one.

### 3.4 The comparison

48 periods, no exclusions. Deviation is `100 × (miniflux − EddyPro) / EddyPro`; "med" is the
median, `[p16, p84]` the 16th/84th-percentile spread, `|max|` the largest absolute deviation
over the 48. Where EddyPro's value is near zero the relative deviation is not formed (the
count is given); those periods still appear in the absolute-difference column and in the
per-period table.

**(a) Like for like — neither side spectrally corrected.** EddyPro publishes `un_Tau`,
`un_H` and the correction factors `Tau_scf`, `H_scf` separately, which makes this possible.

| quantity | miniflux column | EddyPro column | n (rel) | med % | [p16, p84] | \|max\| % |
|---|---|---|---|---|---|---|
| sonic heat covariance | `COV_W_TS` | `w/ts_cov` | 33 | **+0.003** | [−0.044, +0.003] | 5.103 |
| uncorrected H | `H_L0` | `un_H` | 36 | **+0.005** | [−0.045, +0.006] | 8.163 |
| uncorrected u* | `USTAR` | `u*/sqrt(Tau_scf)` | 41 | **+0.001** | [−0.138, +0.002] | 3.074 |
| variance of u | `VAR_U` | `u_var` | 48 | +0.003 | [−0.054, +0.003] | 1.707 |
| variance of v | `VAR_V` | `v_var` | 48 | +0.003 | [+0.002, +0.003] | 0.614 |
| variance of w | `VAR_W` | `w_var` | 48 | +0.003 | [−1.055, +0.003] | 10.827 |
| variance of Ts | `VAR_TS` | `ts_var` | 48 | +0.003 | [+0.002, +0.003] | 13.716 |

**(b) Geometry.**

| quantity | n | median | [p16, p84] | range |
|---|---|---|---|---|
| `WS` vs `wind_speed`, % | 48 | −0.127 | [−0.329, −0.020] | \|max\| 5.587 % |
| `WD` − `wind_dir`, deg (circular) | 48 | **−1.360** | [−5.269, +3.349] | −22.166 to +11.320 |
| `THETA` − `yaw`, deg (circular) | 48 | **0.000000** | [−0.000200, +0.000100] | \|max\| **0.0165** |
| `PHI` − `pitch`, deg | 48 | **0.000000** | [−0.000077, 0.000000] | \|max\| **0.3141** |

`THETA` is reported in (−180, 180] and `yaw` in [0, 360); they are the same angle on a
different branch, which is why the difference is taken circularly.

**(c) Mean state.**

| quantity | med % | [p16, p84] | \|max\| % |
|---|---|---|---|
| `TA` vs `air_temperature` | −0.002 | [−0.003, −0.002] | 0.032 |
| `T_SONIC` vs `sonic_temperature` | +0.000 | [+0.000, +0.000] | 0.004 |
| `PA` vs `air_pressure` | 0.000 | [0.000, 0.000] | 0.000 |
| `RHO_A` vs `air_density` | +0.001 | [+0.000, +0.001] | 0.093 |
| `CP` vs `air_heat_capacity` | +0.002 | [+0.001, +0.002] | 0.004 |
| `RH` vs `RH` | **+0.316** | [+0.229, +0.384] | 1.035 |
| `Q` vs `specific_humidity` | **+0.319** | [+0.290, +0.334] | 0.646 |
| `CO2_MEAN` vs `co2_mixing_ratio` | +0.0000 | [−0.0021, +0.0002] | — |
| `H2O_MEAN` vs `h2o_mixing_ratio` | +0.0000 | [−0.0001, +0.0005] | — |

**(d) Fully corrected — miniflux has no spectral correction and EddyPro does.**

| quantity | n | med % | [p16, p84] | \|max\| % |
|---|---|---|---|---|
| `H` vs `H` | 36 | **−2.228** | [−4.745, −1.995] | 12.830 |
| `USTAR` vs `u*` | 42 | **−0.795** | [−1.452, −0.606] | 4.391 |
| `MO_LENGTH` vs `L` | 38 | **−3.891** | [−15.024, +5.054] | 106.850 |

EddyPro's own correction factors over these 48 periods: `H_scf` median 1.0240 [1.0210,
1.0472], max 1.1213; `Tau_scf` median 1.0168 [1.0124, 1.0286], max 1.0607.

**(e) Undo the reference's own spectral correction and the residual disappears.**

| quantity | factor undone | n | med % | [p16, p84] | \|max\| % |
|---|---|---|---|---|---|
| `H` × `H_scf` vs `H` | `H_scf` | 36 | **+0.031** | [−0.215, +0.172] | 8.642 |
| `USTAR` × `sqrt(Tau_scf)` vs `u*` | `Tau_scf` | 42 | **+0.001** | [−0.113, +0.002] | 3.074 |

**(f) Spike counts, per sample, period by period.**

| variable | miniflux `N_SPIKE_*`, day total | EddyPro `*_SPIKE_NREX`, day total | EddyPro `*_NUM_SPIKES` (runs) |
|---|---|---|---|
| u | 3 | 3 | 3 |
| v | 10 | 10 | 8 |
| w | **918** | **918** | 121 |
| ts | **1039** | 1038 | 34 |

Not just the totals — the *per-period* counts match on 47 of 48 periods for w and ts and on
all 48 for u and v. The single exception is 2022-05-14 01:30, ts: miniflux 36, EddyPro 35.

### 3.5 Every deviation, explained

**(a) The +0.00278 % floor: `N` versus `N − 1`.** Take the 31 periods where *neither*
program flagged a single sample in u, v, w or ts, and the residual is not scattered — it is a
constant:

| quantity | n | median % | min % | max % |
|---|---|---|---|---|
| `COV_W_TS` | 31 | +0.00273 | +0.00129 | +0.00299 |
| `VAR_U` | 31 | +0.00286 | +0.00236 | +0.00420 |
| `VAR_V` | 31 | +0.00275 | +0.00189 | +0.00369 |
| `VAR_W` | 31 | +0.00278 | +0.00174 | +0.00319 |
| `VAR_TS` | 31 | +0.00290 | +0.00258 | +0.00330 |

`N/(N−1) − 1` for `N = 36000` is **+0.00278 %**. That is the whole of it. Source, EddyPro
`src_common/stats_operator_no_error.f90`, `CovarianceMatrixNoError`:
`Cov(i, j) = Cov(i, j) / dble(Nact)` — EddyPro divides the second moments by `Nact`, while
`ALGORITHMS §7.1` prescribes `/(N_p − 1)` for miniflux. (EddyPro's own `StDev` on line 308
uses `Nact − 1`, so the exported variances, which show the same +0.00278 %, come from the
covariance-matrix diagonal.) This is a documented convention difference, it is the smallest
disagreement in this document, and neither value is wrong.

**(b) The 15 periods that deviate more than 0.1 %: what is done with a spike.** The two
programs flag *the same samples* — 918 of 918 in w, 1038 of 1039 in ts (§3.4f). They then
part company:

* miniflux sets a flagged sample to NaN and the covariance uses a pairwise-complete mask, so
  the sample is **dropped**.
* EddyPro (`src_rp/test_spike_detection_mauder_13.f90`, with `filter_sr = 1`) **replaces each
  flagged run by a straight line** between the last good sample before it and the first good
  sample after it, then re-runs the whole test, up to 10 passes, until a pass finds nothing
  new.

An interpolated value is *in* the covariance; a dropped one is not. Every period whose
`cov_w_ts` deviates by more than 0.1 % is a period with a non-zero spike count, and every
period with a zero spike count sits on the +0.00278 % floor. The worst, +14.6 % at 01:00, is
66 replaced samples acting on a covariance of 0.0006 K m s-1 — a near-zero nocturnal flux,
where 66 samples out of 36000 are the whole signal. The absolute difference there is about
0.00009 K m s-1, or **0.11 W m-2**.

The one-sample disagreement at 01:30 has a candidate in the same file: EddyPro's detection
loop runs `do i = 2, N-1`, so it never tests the first or last sample of a period, while
miniflux tests all of them (and instead clears a run that is still open at the end, per
`ALGORITHMS §3.1`). One sample in 36000.

Two further notes on the spike counts. EddyPro's `*_NUM_SPIKES` column in `full_output`
counts spike **runs** (121 in w), not samples — comparing miniflux's per-sample
`N_SPIKE_W` against it produces a bogus 7.6× discrepancy. The right column is
`*_SPIKE_NREX` in the FLUXNET file. And `ALGORITHMS §3.4`'s claim that "EddyPro replaces
flagged runs whole (many more samples replaced than spikes counted)" is confirmed here: 918
samples in 121 runs, a mean run length of 7.6 samples.

**(c) `H` is 2.2 % low and `u*` is 0.8 % low because miniflux has no spectral correction.**
This is not an error, it is a missing feature, stated in `ALGORITHMS §13`. The size is
predictable from EddyPro's own factors: `1/1.0240 − 1 = −2.34 %` against an observed median
of −2.228 % for `H`, and `1/sqrt(1.0168) − 1 = −0.83 %` against an observed −0.795 % for
`u*`. Multiplying miniflux's values by those same factors closes the gap to a median of
+0.031 % and +0.001 % respectively (§3.4e). **Anyone using miniflux's `H` or `u*` for science
is using an under-corrected flux**, by about 2 % and 1 % on a 2 m sonic over a low crop, and
by more on a taller tower or a slower analyser.

**(d) `WD` differs by up to 22° because the two programs average directions differently — and
the difference is reproducible to 0.000°.** `WS` agrees to a median of −0.127 %, so the mean
wind *vector* is the same. What differs is the estimator:

* miniflux (`ALGORITHMS §4.5`): the direction **of the mean vector**,
  `180 − degrees(atan2(mean_v, mean_u)) + north_offset`.
* EddyPro (`src_common/wind_direction.f90`): the same formula applied **per sample**
  (`SingleWindDirection`), then a **unit-vector angular mean** of all 36000 instantaneous
  directions (`AngularAverageNoError`: mean the cosines and the sines, `atan2` the result).

The two coincide when the wind is steady and diverge when it wanders, because the angular
mean weights every sample equally regardless of its speed. To check rather than assert, I
recomputed EddyPro's estimator from the raw `U` and `V` columns for six periods:

| period | miniflux (vector direction) | EddyPro `wind_dir` | angular mean recomputed from raw U,V | difference |
|---|---|---|---|---|
| 00:30 | 61.419 | 83.482 | 83.482 | 0.000 |
| 05:00 | 235.132 | 224.756 | 224.756 | −0.000 |
| 06:30 | 208.270 | 230.436 | 230.436 | −0.000 |
| 13:00 | 46.387 | 53.700 | 53.700 | 0.000 |
| 19:30 | 117.509 | 114.281 | 114.281 | −0.000 |
| 23:30 | 112.296 | 100.976 | 100.976 | 0.000 |

Exact, on raw un-despiked data. The three lightest-wind periods of the day (0.121, 0.174 and
0.186 m s-1) carry three of the four largest deviations (+10.4°, −22.2°, −22.1°); the fourth,
+11.3° at 23:30, sits at 0.740 m s-1. Across all 48 periods the relationship is present but
not monotone — Spearman rank correlation between `|ΔWD|` and `WS` is **−0.28** — which is
what one should expect, since the gap depends on the *shape* of the directional distribution
and not on the wind speed alone. The whole day is unusually calm — `WS` never exceeds
1.5 m s-1 — which is why this gap is so visible here and would be a fraction of a degree on a
windy day. Neither estimator is wrong; they answer different questions, and
`ALGORITHMS §4.5` picks one on purpose.

**(e) `THETA` and `PHI` agree exactly, and the exceptions are spikes.** Median difference
0.000000° on both. `THETA`'s largest deviation, 0.0165°, is EddyPro's six-significant-figure
output rounding. Every `PHI` deviation above 0.01° — 0.3141°, 0.2790°, 0.0475°, 0.0438°,
0.0373°, 0.0249° — is on a period with flagged spikes, i.e. cause (b) again, since the second
rotation angle is taken from nan-means that a dropped sample moves and an interpolated one
does not. Six periods out of 48.

**(f) `RH` and `Q` are +0.32 % high: a temperature swapped between two adjacent formulas.**
The concentrations themselves agree to **0.0000 %** (`CO2_MEAN`, `H2O_MEAN` vs EddyPro's dry
mixing ratios), so the difference is downstream, in the conversion from a dry mole fraction to
a vapour pressure. Testing two candidate forms against EddyPro's published `e`, over all 48
periods:

| candidate for EddyPro's `e` | median deviation from EddyPro's `e` |
|---|---|
| `chi_wet · P` (the mole-fraction identity, what miniflux does) | **+0.2995 %** |
| `chi_wet · P · (Ta / Ts)` | **−0.0257 %** |

where `chi_wet = X/(1+X)` from EddyPro's own `h2o_mixing_ratio`. The reference's vapour
pressure carries a spurious `Ta/Ts` ratio: the ambient vapour molar density is formed at one
temperature and converted back to a partial pressure at the other. `Ts − Ta` is 0.83 K on
292 K, which is 0.28 % — the whole of the observed 0.32 %. This is precisely the trap
`ALGORITHMS §8.3` is written about, and miniflux keeps one temperature throughout.

Consequence for the quantities this section validates: **none measurable.** The humidity
reaches `H` through `cp` (which agrees to +0.002 %) and through the Schotanus denominator
`1 + 0.51q` (a 0.32 % change in q = 0.0068 moves it by 0.002 %). `H_L0` agrees to +0.005 %.

**(g) `MO_LENGTH` is 3.9 % low at the median and wanders to 107 % — three differences, all
documented, and the reference is reproducible to 0.0004 %.** EddyPro's `L` in this output is
not the raw-level `L`. Reconstructing it from EddyPro's own columns:

| reconstruction | median \|error\| vs EddyPro's `L` | max |
|---|---|---|
| `−Ta·(1e5/P)^0.286 · u*³ / (0.41·g·H/(ρ·cp))` | **0.0004 %** | 0.0015 % |
| same with `Ts` instead of `Ta` | 0.3255 % | 0.3494 % |
| same with `Ts` and the uncorrected `w'Ts'` | 4.8155 % | 105.28 % |

So EddyPro uses (i) `κ = 0.41`, (ii) the **ambient** temperature in the potential
temperature, and (iii) the **fully corrected** heat flux. miniflux uses (i) `κ = 0.40`
(`constants.py` says so, and says EddyPro's raw path uses 0.41), (ii) the **sonic**
temperature (`ALGORITHMS §9.4`), and (iii) the raw `cov_w_ts`. Those three are worth +2.5 %,
+0.29 % and roughly −0.1 % respectively, which is the right order for the observed median.
The 107 % excursion is at 19:00, where `L` passes through a pole as the flux crosses zero
(`L` = +277.7 vs +134.3 m) — near neutrality `L` is ill-conditioned and a ratio of two large
`L` values is not a meaningful statistic. `(z−d)` is the same 1.90 m on both sides, so `ZL`
carries exactly the `L` deviation and nothing extra.

**(h) `VAR_W` and `VAR_TS` reach 10.8 % and 13.7 %** — both on the low-flux night periods
with the heaviest spike replacement (cause b). `VAR_V`'s 0.614 % maximum is the 16:00 period
with 9 v-spikes; v-spikes propagate into `COV_W_TS` through the rotation, which is why that
period shows −0.029 % rather than the +0.00278 % floor.

### 3.6 How much of `H` rests on the pressure

`examples/fr_gri_wind_sonic.ini` declares `pressure_pa = 99767.5` because that is the
constant in EddyPro's `air_pressure` column for every half hour of the sample — EddyPro had
no ambient-pressure input (`col_air_p = 0`) and fell back to the 125 m station altitude. That
is the reference's own stated ambient pressure, so declaring it makes the comparison measure
turbulence processing and not a difference of barometry. It is *not* the best physical value:
the LI-7200's own box barometer, `PRESS_BOX`, reads **100.65 kPa** that day, 0.89 % higher,
and the cell reads 99.28 kPa, 0.49 % lower.

`examples/fr_gri_wind_sonic_pbox.ini` is the same run with `p_air = PRESS_BOX` and nothing
else changed:

| quantity | measured PRESS_BOX vs declared 99767.5 Pa |
|---|---|
| `PA`, `RHO_A` | +0.889 % (median) |
| `H_L0`, `H` | **+0.889 %** |
| `RH` | +0.889 % |
| `TA` | 0.000 % |
| `USTAR`, `COV_W_TS` | 0.000 % |

and against the reference, `H_L0` then sits **+0.774 %** above EddyPro's `un_H` instead of
+0.005 %. The reading: **0.9 % of `H` at this site is a choice about ambient pressure, and
the better measurement disagrees with the reference by more than the entire turbulence
processing does.** `TA` does not move because `e/p` is the mixing ratio, which is what the
sonic-temperature conversion uses. `u*` does not move at all.

### 3.7 Quality flags

Compared against EddyPro's `_qc_details_` file, which publishes the Foken–Wichura deviations
directly. This is a secondary comparison: the two use the same test but not the same
sub-interval bookkeeping, and `CONTRACT §20.1` records a deliberate estimator choice here.

| statistic | result |
|---|---|
| `SST_PCT` − EddyPro `dev(w/ts)`, on the 24 periods where EddyPro's value is 5–100 % | median **−0.90** percentage points, [p16 −12.89, p84 +9.47], \|max\| 23.18 pp |
| same, all 48 periods | median +0.04 pp, \|max\| 596 pp |
| `SST_FLAG` class vs EddyPro's 1–9 flag binned into three | **42 of 48 agree** |
| `ITC_W` − `dev(w)`/100 | median +0.0054, [+0.0010, +0.0091], \|max\| 0.0198 |
| `ITC_U` − `dev(u)`/100 | median +0.0051, [+0.0009, +0.0088], \|max\| 0.0189 |
| `ITC_T` − `dev(ts)`/100 | median −0.0092, [−0.0533, +0.0004], \|max\| 0.8926 |

The ITC deviations sit within about 0.01 of EddyPro's on `w` and `u`. `ITC_T` is **not
comparable by construction** — `CONTRACT §20.5` divides by `|T*|` where EddyPro and the
parent divide by a signed `T*` — and the 0.89 outlier is an unstable period, exactly where
that choice bites. The steady-state percentages track each other closely when the flux is
well developed and diverge wildly when the covariance approaches zero, because the statistic
is a ratio with a vanishing denominator; the 596 pp maximum is the 23:00 period where
EddyPro reports 2861 % and miniflux 2265 %, which are the same verdict.

---

## 4. The two kernel paths give the same numbers on real data

`CONTRACT §3.8` promises identical decisions and identical elementwise results between the
numpy fast path and the pure-Python path. Tested on real 36000-sample periods, not fixtures:
the first 12 FR-Gri periods were re-run with `--pure` and the output CSV compared cell by
cell against the numpy run.

**12 periods × 52 columns = 624 cells, 624 byte-identical (100.00 %), worst relative
difference 0.000e+00** at the `%.6g` output precision.

---

## 5. What this does not prove

Read this section as carefully as the tables.

1. **miniflux is not validated against a certified reference, because none exists.** It has
   been compared against EddyPro 7.0.9 on one day at one site, and run end to end on seven
   half hours at another. ONEFlux_preproc and EddyPro are the references; this document is
   not, and nothing here licenses using miniflux where one of those is called for.

2. **Agreement with EddyPro is agreement with an implementation, not with the atmosphere.**
   Both programs could share a wrong convention and the comparison would not see it. The
   places where they demonstrably *disagree* (§3.5) are the places this exercise has
   information about; the places where they agree to 0.003 % mostly demonstrate that two
   implementations of the same documented recipe produce the same number.

3. **§3 validates the wind and sonic-temperature chain only.** Despiking, double rotation,
   block detrending, the second moments, `u*`, the mean thermodynamic state and `H`. It says
   nothing about the gas fluxes, because the analyser was closed-path and those outputs were
   discarded. `FC`, `LE` and `E` have been compared against **no reference anywhere in this
   document.** The FR-Jus run exercises the WPL path end to end and shows it produces terms
   of the right size and sign, which is not the same as showing it produces the right number.

4. **The WPL correction is unvalidated against a reference.** §2.4 shows it dominating `FC`
   on an open-path site and flipping the flux sign on four of seven periods. If WPL is wrong,
   every open-path `FC` miniflux produces is wrong, and nothing in this document would have
   caught it.

5. **The lag search is barely exercised.** FR-Gri's gas lags were discarded; FR-Jus is a
   collocated open path whose true lag is zero, and the `covmax_default` fallback fired on
   four of seven periods. A separated open-path or a closed-path tube — the cases the lag
   search exists for — has not been tested against a reference here at all.

6. **Two days is two days.** 48 + 7 half hours, two instruments, two sites, one season each,
   one latitude band. No cold conditions, no high wind, no strong stability, no rain, no
   sensor failure beyond the one truncated record, no gap-riddled day beyond FR-Jus's 1 %
   dropout. FR-Gri is an exceptionally calm day (`WS` ≤ 1.5 m s-1), which magnifies the wind
   direction difference and suppresses the spectral correction.

7. **The FR-Jus assessment is a plausibility argument, not a measurement.** "H is positive at
   night and that is right for a city" is reasoning about sign and order of magnitude. It
   would not detect a 10 % error, or a 30 % one. There is no reference output for that site
   and no independent measurement — no radiation, no storage, no energy balance closure — to
   test against.

8. **Two FR-Jus site values are invented**: the 30 m measurement height and the Paris
   latitude. `ZL` and the ITC columns for that site are conditional on them. `MO_LENGTH` and
   every flux are not.

9. **miniflux's `H` and `u*` are systematically low by about 2 % and 1 % on this dataset**
   because it applies no spectral correction and never will (`ALGORITHMS §13`). On a taller
   tower, a slower sensor or a longer tube the deficit is larger — possibly much larger. That
   is a known, quantified, permanent limitation, not a bug to be fixed.

10. **A comparison this close can hide a shared mistake in the *inputs*.** The 30-minute
    label offset in §3.3 was found only because three independent quantities agreed about it.
    Something similar and smaller — a column mapped to the wrong variable, a unit declared
    twice — would show up as a processing difference and could be explained away. The defence
    is that §3.5 explains every deviation down to a named line of reference source, and no
    residual is left over that would hide one.

11. **Nothing here tests the refusals.** The unit tests do (`CONTRACT §21`); this document
    happened to exercise exactly one in the field — the truncated final record at FR-Jus,
    §2.5 — and it behaved as documented.


---

## Appendix A — FR-Gri per-period comparison, all 48 periods

The evidence behind §3.4 and §3.5. EddyPro rows are the `T + 30 min` ones (§3.3). `dev %` is
`100 x (miniflux - EddyPro) / EddyPro`. The last column is flagged **samples** on each side
(miniflux `N_SPIKE_W + N_SPIKE_TS`, EddyPro `W_SPIKE_NREX + T_SONIC_SPIKE_NREX` from the
FLUXNET file) -- the quantity that predicts which rows leave the +0.00278 % floor.

### A.1 Turbulence, neither side spectrally corrected

| period END | N_IN | cov_w_ts | EP w/ts_cov | dev % | H_L0 | EP un_H | dev % | USTAR | EP u*/sqrt(Tau_scf) | dev % | spikes w+ts (mf / EP nrex) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022-05-14 00:00 | 36000 | -0.02794 | -0.02944 | -5.103 | -34.067 | -35.898 | -5.101 | 0.1280 | 0.1283 | -0.216 | 70 / 70 |
| 2022-05-14 00:30 | 36000 | -0.00460 | -0.00436 | +5.352 | -5.599 | -5.314 | +5.350 | 0.1146 | 0.1174 | -2.434 | 233 / 233 |
| 2022-05-14 01:00 | 36000 | 0.00070 | 0.00062 | +14.578 | 0.858 | 0.749 | +14.580 | 0.0546 | 0.0540 | +1.188 | 66 / 66 |
| 2022-05-14 01:30 | 36000 | -0.00056 | -0.00062 | -9.727 | -0.682 | -0.756 | -9.725 | 0.0324 | 0.0328 | -1.418 | 126 / 125 |
| 2022-05-14 02:00 | 36000 | -0.00003 | -0.00003 | +0.001 | -0.038 | -0.038 | +0.003 | 0.0409 | 0.0409 | +0.001 | 0 / 0 |
| 2022-05-14 02:30 | 36000 | -0.00722 | -0.00712 | +1.416 | -8.814 | -8.699 | +1.325 | 0.0797 | 0.0804 | -0.796 | 469 / 469 |
| 2022-05-14 03:00 | 36000 | -0.00554 | -0.00566 | -2.133 | -6.798 | -6.947 | -2.136 | 0.0468 | 0.0468 | +0.001 | 129 / 129 |
| 2022-05-14 03:30 | 36000 | -0.00397 | -0.00397 | +0.003 | -4.886 | -4.885 | +0.005 | 0.0732 | 0.0732 | +0.001 | 0 / 0 |
| 2022-05-14 04:00 | 36000 | -0.00440 | -0.00480 | -8.161 | -5.412 | -5.894 | -8.163 | 0.0814 | 0.0816 | -0.200 | 586 / 586 |
| 2022-05-14 04:30 | 36000 | 0.00097 | 0.00097 | -0.244 | 1.187 | 1.190 | -0.242 | 0.0332 | 0.0332 | -0.244 | 11 / 11 |
| 2022-05-14 05:00 | 36000 | 0.00640 | 0.00639 | +0.028 | 7.859 | 7.857 | +0.030 | 0.0495 | 0.0495 | +0.095 | 5 / 5 |
| 2022-05-14 05:30 | 36000 | -0.00448 | -0.00448 | +0.003 | -5.510 | -5.509 | +0.005 | 0.0479 | 0.0479 | +0.001 | 0 / 0 |
| 2022-05-14 06:00 | 36000 | -0.00093 | -0.00093 | +0.008 | -1.140 | -1.140 | +0.011 | 0.0373 | 0.0373 | +0.093 | 13 / 13 |
| 2022-05-14 06:30 | 36000 | -0.00806 | -0.00809 | -0.403 | -9.912 | -9.952 | -0.400 | 0.0861 | 0.0888 | -3.074 | 184 / 184 |
| 2022-05-14 07:00 | 36000 | 0.01933 | 0.01933 | +0.003 | 23.731 | 23.730 | +0.006 | 0.0729 | 0.0729 | +0.001 | 0 / 0 |
| 2022-05-14 07:30 | 36000 | 0.04702 | 0.04701 | +0.003 | 57.351 | 57.348 | +0.005 | 0.1266 | 0.1265 | +0.002 | 0 / 0 |
| 2022-05-14 08:00 | 36000 | 0.03989 | 0.03989 | +0.003 | 48.568 | 48.565 | +0.006 | 0.1452 | 0.1452 | +0.001 | 0 / 0 |
| 2022-05-14 08:30 | 36000 | 0.05092 | 0.05092 | +0.003 | 61.738 | 61.735 | +0.005 | 0.2255 | 0.2255 | +0.002 | 0 / 0 |
| 2022-05-14 09:00 | 36000 | 0.05954 | 0.05954 | +0.003 | 72.007 | 72.004 | +0.005 | 0.2386 | 0.2386 | +0.001 | 0 / 0 |
| 2022-05-14 09:30 | 36000 | 0.07244 | 0.07244 | +0.003 | 87.406 | 87.402 | +0.005 | 0.2675 | 0.2675 | +0.001 | 0 / 0 |
| 2022-05-14 10:00 | 36000 | 0.09499 | 0.09499 | +0.003 | 114.413 | 114.407 | +0.005 | 0.2482 | 0.2481 | +0.002 | 0 / 0 |
| 2022-05-14 10:30 | 36000 | 0.09047 | 0.09047 | +0.003 | 108.771 | 108.765 | +0.006 | 0.2379 | 0.2379 | +0.001 | 0 / 0 |
| 2022-05-14 11:00 | 36000 | 0.09032 | 0.09108 | -0.842 | 108.488 | 109.406 | -0.839 | 0.1720 | 0.1726 | -0.356 | 13 / 13 |
| 2022-05-14 11:30 | 36000 | 0.09564 | 0.09564 | +0.003 | 114.791 | 114.785 | +0.005 | 0.2023 | 0.2023 | +0.001 | 0 / 0 |
| 2022-05-14 12:00 | 36000 | 0.09927 | 0.09915 | +0.124 | 118.887 | 118.736 | +0.127 | 0.2138 | 0.2149 | -0.476 | 2 / 2 |
| 2022-05-14 12:30 | 36000 | 0.12646 | 0.12645 | +0.002 | 151.165 | 151.156 | +0.006 | 0.2370 | 0.2370 | +0.001 | 0 / 0 |
| 2022-05-14 13:00 | 36000 | 0.11762 | 0.11762 | +0.003 | 140.265 | 140.257 | +0.006 | 0.3017 | 0.3017 | +0.001 | 0 / 0 |
| 2022-05-14 13:30 | 36000 | 0.15166 | 0.15166 | +0.003 | 180.664 | 180.653 | +0.006 | 0.2747 | 0.2747 | +0.002 | 0 / 0 |
| 2022-05-14 14:00 | 36000 | 0.13539 | 0.13539 | +0.002 | 160.920 | 160.910 | +0.006 | 0.3143 | 0.3143 | +0.002 | 0 / 0 |
| 2022-05-14 14:30 | 36000 | 0.10021 | 0.10020 | +0.003 | 119.071 | 119.064 | +0.006 | 0.2278 | 0.2278 | +0.002 | 0 / 0 |
| 2022-05-14 15:00 | 36000 | 0.06864 | 0.06863 | +0.003 | 81.466 | 81.462 | +0.005 | 0.2722 | 0.2722 | +0.001 | 0 / 0 |
| 2022-05-14 15:30 | 36000 | 0.04304 | 0.04304 | +0.003 | 51.075 | 51.072 | +0.005 | 0.2276 | 0.2276 | +0.002 | 0 / 0 |
| 2022-05-14 16:00 | 36000 | 0.03673 | 0.03674 | -0.029 | 43.529 | 43.541 | -0.027 | 0.3117 | 0.3118 | -0.004 | 0 / 0 |
| 2022-05-14 16:30 | 36000 | 0.01194 | 0.01194 | +0.003 | 14.151 | 14.151 | +0.005 | 0.2221 | 0.2221 | +0.001 | 0 / 0 |
| 2022-05-14 17:00 | 36000 | 0.01018 | 0.01019 | -0.046 | 12.067 | 12.072 | -0.044 | 0.2200 | 0.2201 | -0.016 | 0 / 0 |
| 2022-05-14 17:30 | 36000 | 0.01675 | 0.01675 | +0.002 | 19.840 | 19.839 | +0.005 | 0.2633 | 0.2632 | +0.002 | 0 / 0 |
| 2022-05-14 18:00 | 36000 | 0.00556 | 0.00556 | +0.003 | 6.591 | 6.591 | +0.005 | 0.2201 | 0.2201 | +0.001 | 0 / 0 |
| 2022-05-14 18:30 | 36000 | 0.01056 | 0.01056 | +0.003 | 12.529 | 12.528 | +0.006 | 0.1858 | 0.1858 | +0.001 | 0 / 0 |
| 2022-05-14 19:00 | 36000 | -0.00272 | -0.00272 | +0.002 | -3.225 | -3.225 | +0.004 | 0.2154 | 0.2154 | +0.001 | 0 / 0 |
| 2022-05-14 19:30 | 36000 | -0.01830 | -0.01829 | +0.003 | -21.763 | -21.762 | +0.005 | 0.1434 | 0.1434 | +0.001 | 0 / 0 |
| 2022-05-14 20:00 | 36000 | -0.02363 | -0.02364 | -0.050 | -28.262 | -28.276 | -0.047 | 0.1056 | 0.1056 | -0.045 | 1 / 1 |
| 2022-05-14 20:30 | 36000 | -0.00335 | -0.00335 | +0.003 | -4.020 | -4.019 | +0.005 | 0.0832 | 0.0832 | +0.001 | 0 / 0 |
| 2022-05-14 21:00 | 36000 | -0.01357 | -0.01356 | +0.047 | -16.366 | -16.363 | +0.019 | 0.1195 | 0.1195 | +0.002 | 49 / 49 |
| 2022-05-14 21:30 | 36000 | -0.00369 | -0.00369 | +0.003 | -4.456 | -4.455 | +0.005 | 0.0836 | 0.0836 | +0.001 | 0 / 0 |
| 2022-05-14 22:00 | 36000 | 0.00159 | 0.00159 | +0.002 | 1.925 | 1.924 | +0.005 | 0.1144 | 0.1144 | +0.001 | 0 / 0 |
| 2022-05-14 22:30 | 36000 | -0.00336 | -0.00336 | +0.003 | -4.064 | -4.064 | +0.005 | 0.0939 | 0.0939 | +0.001 | 0 / 0 |
| 2022-05-14 23:00 | 36000 | 0.00008 | 0.00008 | +0.002 | 0.096 | 0.096 | +0.004 | 0.0706 | 0.0706 | +0.001 | 0 / 0 |
| 2022-05-14 23:30 | 36000 | -0.01099 | -0.01099 | +0.003 | -13.337 | -13.336 | +0.005 | 0.1294 | 0.1294 | +0.001 | 0 / 0 |

`cov_w_ts` in K m s-1, `H_L0` / `un_H` in W m-2, `USTAR` in m s-1.

### A.2 Geometry

| period END | WS | EP wind_speed | WD | EP wind_dir | dWD deg | THETA | EP yaw | PHI | EP pitch |
|---|---|---|---|---|---|---|---|---|---|
| 2022-05-14 00:00 | 0.985 | 0.985 | 321.459 | 320.701 | +0.758 | 108.541 | 108.541 | 0.3633 | 0.3158 |
| 2022-05-14 00:30 | 0.186 | 0.186 | 61.419 | 83.482 | -22.062 | 8.581 | 8.581 | 1.5889 | 1.9029 |
| 2022-05-14 01:00 | 0.299 | 0.300 | 219.374 | 220.865 | -1.491 | -149.374 | 210.626 | 5.0568 | 5.0195 |
| 2022-05-14 01:30 | 0.425 | 0.426 | 252.542 | 251.622 | +0.920 | 177.458 | 177.458 | 3.4584 | 3.4503 |
| 2022-05-14 02:00 | 0.243 | 0.244 | 175.065 | 183.526 | -8.461 | -105.065 | 254.935 | 6.1130 | 6.1130 |
| 2022-05-14 02:30 | 0.581 | 0.582 | 145.653 | 142.172 | +3.481 | -75.653 | 284.347 | 2.2026 | 2.2041 |
| 2022-05-14 03:00 | 0.296 | 0.296 | 81.704 | 87.200 | -5.497 | -11.704 | 348.296 | -1.3339 | -1.3339 |
| 2022-05-14 03:30 | 0.635 | 0.635 | 112.884 | 105.502 | +7.382 | -42.884 | 317.116 | -0.8911 | -0.8911 |
| 2022-05-14 04:00 | 0.256 | 0.258 | 189.948 | 194.822 | -4.874 | -119.948 | 240.052 | 6.1102 | 6.1351 |
| 2022-05-14 04:30 | 0.265 | 0.265 | 227.616 | 228.845 | -1.229 | -157.616 | 202.384 | 3.1607 | 3.1655 |
| 2022-05-14 05:00 | 0.121 | 0.123 | 235.132 | 224.756 | +10.376 | -165.132 | 194.868 | 8.9651 | 8.9588 |
| 2022-05-14 05:30 | 0.508 | 0.508 | 223.805 | 233.663 | -9.858 | -153.805 | 206.195 | 1.5630 | 1.5630 |
| 2022-05-14 06:00 | 0.607 | 0.607 | 160.576 | 160.956 | -0.380 | -90.576 | 269.424 | 0.5511 | 0.5510 |
| 2022-05-14 06:30 | 0.174 | 0.174 | 208.270 | 230.436 | -22.166 | -138.270 | 221.730 | 4.3758 | 4.6548 |
| 2022-05-14 07:00 | 0.241 | 0.241 | 307.329 | 300.919 | +6.410 | 122.671 | 122.671 | 3.1459 | 3.1459 |
| 2022-05-14 07:30 | 0.211 | 0.224 | 75.622 | 79.318 | -3.696 | -5.622 | 354.378 | 19.2435 | 19.2435 |
| 2022-05-14 08:00 | 0.563 | 0.563 | 9.227 | 8.757 | +0.471 | 60.773 | 60.773 | -0.9402 | -0.9402 |
| 2022-05-14 08:30 | 0.690 | 0.690 | 74.471 | 76.128 | -1.656 | -4.471 | 355.529 | -0.0138 | -0.0138 |
| 2022-05-14 09:00 | 0.829 | 0.829 | 62.294 | 67.570 | -5.275 | 7.706 | 7.706 | -1.8538 | -1.8538 |
| 2022-05-14 09:30 | 0.926 | 0.926 | 57.153 | 61.684 | -4.531 | 12.847 | 12.847 | -1.1021 | -1.1021 |
| 2022-05-14 10:00 | 0.731 | 0.731 | 55.525 | 58.941 | -3.417 | 14.476 | 14.476 | -1.7524 | -1.7524 |
| 2022-05-14 10:30 | 0.748 | 0.748 | 82.460 | 84.019 | -1.558 | -12.460 | 347.540 | -1.1723 | -1.1723 |
| 2022-05-14 11:00 | 0.677 | 0.679 | 80.404 | 78.823 | +1.581 | -10.404 | 349.596 | -4.6855 | -4.6417 |
| 2022-05-14 11:30 | 0.481 | 0.485 | 62.874 | 70.871 | -7.996 | 7.126 | 7.126 | -7.7753 | -7.7753 |
| 2022-05-14 12:00 | 0.741 | 0.743 | 102.312 | 103.511 | -1.199 | -32.312 | 327.705 | -2.7838 | -2.7893 |
| 2022-05-14 12:30 | 1.294 | 1.295 | 12.883 | 16.166 | -3.283 | 57.117 | 57.117 | -3.0680 | -3.0680 |
| 2022-05-14 13:00 | 1.491 | 1.493 | 46.387 | 53.700 | -7.312 | 23.613 | 23.613 | -3.3793 | -3.3793 |
| 2022-05-14 13:30 | 1.115 | 1.118 | 30.148 | 34.931 | -4.783 | 39.852 | 39.852 | -3.7068 | -3.7068 |
| 2022-05-14 14:00 | 1.143 | 1.144 | 71.172 | 74.935 | -3.763 | -1.172 | 358.828 | -2.8729 | -2.8729 |
| 2022-05-14 14:30 | 0.582 | 0.583 | 104.992 | 98.920 | +6.072 | -34.992 | 325.008 | -4.1541 | -4.1541 |
| 2022-05-14 15:00 | 1.134 | 1.136 | 43.413 | 47.734 | -4.321 | 26.587 | 26.587 | -3.1218 | -3.1218 |
| 2022-05-14 15:30 | 0.794 | 0.795 | 88.249 | 91.733 | -3.485 | -18.249 | 341.751 | -1.5498 | -1.5498 |
| 2022-05-14 16:00 | 1.074 | 1.075 | 63.130 | 68.393 | -5.263 | 6.870 | 6.874 | -2.2157 | -2.2152 |
| 2022-05-14 16:30 | 0.784 | 0.784 | 71.303 | 74.080 | -2.777 | -1.303 | 358.697 | -2.3576 | -2.3576 |
| 2022-05-14 17:00 | 0.744 | 0.745 | 78.862 | 77.970 | +0.892 | -8.862 | 351.133 | -3.5841 | -3.5840 |
| 2022-05-14 17:30 | 1.256 | 1.258 | 56.829 | 61.717 | -4.887 | 13.171 | 13.171 | -3.2994 | -3.2994 |
| 2022-05-14 18:00 | 0.574 | 0.575 | 77.876 | 80.948 | -3.072 | -7.876 | 352.124 | 1.4888 | 1.4888 |
| 2022-05-14 18:30 | 0.558 | 0.558 | 92.595 | 89.903 | +2.692 | -22.595 | 337.405 | 0.1849 | 0.1849 |
| 2022-05-14 19:00 | 0.766 | 0.767 | 99.080 | 93.753 | +5.327 | -29.080 | 330.920 | -2.8915 | -2.8915 |
| 2022-05-14 19:30 | 1.326 | 1.329 | 117.509 | 114.281 | +3.228 | -47.509 | 312.491 | -3.5662 | -3.5662 |
| 2022-05-14 20:00 | 1.164 | 1.166 | 117.418 | 115.720 | +1.698 | -47.418 | 312.582 | -3.3114 | -3.3123 |
| 2022-05-14 20:30 | 1.134 | 1.136 | 112.986 | 111.834 | +1.152 | -42.986 | 317.014 | -2.9635 | -2.9635 |
| 2022-05-14 21:00 | 0.720 | 0.720 | 119.338 | 111.874 | +7.464 | -49.338 | 310.662 | -1.7172 | -1.7172 |
| 2022-05-14 21:30 | 1.108 | 1.109 | 117.269 | 116.320 | +0.949 | -47.270 | 312.731 | -1.8717 | -1.8717 |
| 2022-05-14 22:00 | 0.858 | 0.862 | 104.948 | 102.291 | +2.657 | -34.948 | 325.052 | -5.4389 | -5.4389 |
| 2022-05-14 22:30 | 1.110 | 1.111 | 118.109 | 117.215 | +0.894 | -48.109 | 311.891 | -1.3273 | -1.3273 |
| 2022-05-14 23:00 | 1.226 | 1.226 | 126.201 | 124.371 | +1.830 | -56.201 | 303.799 | -0.5451 | -0.5451 |
| 2022-05-14 23:30 | 0.740 | 0.740 | 112.296 | 100.976 | +11.320 | -42.296 | 317.704 | -1.7063 | -1.7063 |

`WS` in m s-1; `WD`, `THETA`, `PHI` in degrees. `THETA` is on (-180, 180] and `yaw` on
[0, 360) -- the same angle, a different branch. `dWD` is the circular difference.

### A.3 FR-Jus lags and quality flags, all 7 periods

| period END | CO2_LAG | CO2_LAG_OPT | default used | H2O_LAG | H2O_LAG_OPT | default used | SST_PCT | SST_FLAG | ITC_W | ITC_U | ITC_T |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-09-08 20:00 | 0 | 0 | 0 | 0 | 2 | **1** | 6.42 | 0 | 0.1209 | 0.0541 | 1.5344 |
| 2025-09-08 20:30 | 0 | -2 | **1** | 0.1 | 0.1 | 0 | 158.08 | **2** | 0.1084 | 0.0716 | 0.9636 |
| 2025-09-08 21:00 | -1.8 | -1.8 | 0 | -1.9 | -1.9 | 0 | 66.93 | 1 | 0.1557 | 0.2537 | 0.2385 |
| 2025-09-08 21:30 | -0.35 | -0.35 | 0 | 0 | 0 | 0 | 68.02 | 1 | 0.0933 | 0.2782 | 1.0699 |
| 2025-09-08 22:00 | 0 | 2 | **1** | 0 | 0 | 0 | 224.94 | **2** | 0.1345 | 0.0099 | 0.5400 |
| 2025-09-08 22:30 | 0.1 | 0.1 | 0 | 0 | 0 | 0 | 15.31 | 0 | 0.1120 | 0.0266 | 0.6263 |
| 2025-09-08 23:00 | 0 | 2 | **1** | 0 | 0 | 0 | 460.05 | **2** | 0.0832 | 0.0353 | 0.9745 |

Lags in seconds. `default used = 1` means the covariance peak sat on the window edge and the
`covmax_default` rule applied the nominal 0 s instead (§2.5). The 21:00 row is the
2375-sample fragment.
