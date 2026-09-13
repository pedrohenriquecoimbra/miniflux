# miniflux — VALIDATION

What miniflux has been run on, what it was compared against, what agreed, what did not, and what
none of it proves.

**miniflux is not validated against a certified reference.** There is no certified reference for
an eddy-covariance flux. What exists here is a comparison against two programs the community
uses as references and that miniflux was extracted from: **ONEFlux_preproc** (the parent this is
a minimal reimplementation of) and **EddyPro 7.0.9** (LI-COR), whose output ships with the FR-Gri
sample. Those are the references; **this document is not**. Where miniflux and EddyPro disagree
the prior is that EddyPro is right, and every disagreement below is either explained down to the
line of source that causes it or named as unexplained. `ALGORITHMS.md` (the mathematics) and
`CONTRACT.md` (the module API) are law; nothing here may be used to argue for changing them.

Everything below was produced with the repository exactly as committed; `miniflux/` and `tests/`
were not touched. Suite at the time of the runs: 478 tests, OK, 3.5 s, with numpy present, so
both kernel paths are exercised.

| run | data | reference | headline |
|---|---|---|---|
| §1 FR-Jus | IRGASON open path, 7 half hours, 2025-09-08 evening | none | fluxes are physically sensible for an urban night; WPL is the dominant term in `FC` |
| §2 FR-Gri | Gill HS-50 sonic, 48 half hours, 2022-05-14 | EddyPro 7.0.9 | with the reference's own spectral correction undone, `cov_w_ts` agrees to a median **+0.003 %**, `u*` to **+0.001 %**, `H_L0` to **+0.005 %** |

The configurations are committed beside this document: `examples/fr_jus_openpath.ini`,
`examples/fr_gri_wind_sonic.ini` and `examples/fr_gri_wind_sonic_pbox.ini` (a
pressure-sensitivity variant of the second). **No setting was chosen to make the agreement look
better;** every non-default is justified in the `.ini` comment and again below. Three settings
come from the reference's own metadata — the FR-Gri latitude, instrument height and sonic north
offset — and one, the FR-Gri ambient pressure, from the reference's own *output*; §2.4 measures
what that last choice is worth, with a second run that does not make it.

---

## 1. FR-Jus, 2025-09-08 — open path, end to end

`python -m miniflux run examples/fr_jus_openpath.ini` over
`ONEFlux_preproc/data/sample/FR-Jus_20250908/FLUX/*.csv` — 7 TOA5 files, 20 Hz, Campbell IRGASON
(open path, sonic and analyser in one head), 4.2 s wall clock. `co2` and `h2o` are declared as
`molar_density` (mg m-3, g m-3), so **WPL is owed and runs**; everything else is the documented
default — double rotation, block detrending, MAD despiking at q = 7, `covmax_default` lag over
±2.0 s, `wpl = auto`, 30-minute periods closed on the right.

Two site values are **assumptions, not data**, the sample shipping no tower geometry:
`measurement_height = 30.0 m` with `displacement = 0.0`, which enters `ZL` and the ITC test and
nothing else (`MO_LENGTH` does not depend on it and no flux does, so `ZL` below is conditional
and scales with whatever `(z − d)` really is), and `latitude = 48.85` — Paris, inferred from the
`FR-` prefix and the `ICOSCitiesV031` logger program — which enters only the ITC Coriolis term.
`north_offset` is left at 0.0 because the sonic azimuth is unknown, so **`WD` is in the sonic
frame and is not a geographic wind direction.**

**Pressure.** `RAW_IRGASON_CellPrs` was used and `[site] pressure_pa` left empty: a cell pressure
would be wrong on a closed path, but on this open path the EC100 reads ambient at the head, and
the column runs 100.905–100.932 kPa — an ordinary September barometric pressure, not a pumped
cell (the FR-Gri closed-path cell in §2 sits 1.4 kPa *below* its own box). Replacing it with a
declared 101325 Pa (+0.4 %) moves `PA` and `RHO_A` by +0.39 to +0.42 %, `H` by **+0.38 to
+0.42 %**, `FC` by −0.11 to +0.04 %, `LE` by ≤0.01 % and `USTAR` by 0.000 %: `H` tracks the
density exactly, as it must, and the momentum flux is untouched.

### 1.1 What came out

Seven periods; the logger clock is not documented as local or UTC, but either way everything from
the second period on is after sunset. `H`, `LE` in W m-2, `FC` in µmol m-2 s-1.

| period END | N_IN | WS | USTAR | TA (K) | RH % | CO2 ppm | H2O ppt | H | FC | LE | L (m) | ZL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20:00 | 33258 | 2.869 | 0.5067 | 294.50 | 41.6 | 416.9 | 10.55 | 29.39 | 3.662 | 0.58 | −399.5 | −0.075 |
| 20:30 | 35640 | 2.504 | 0.4752 | 293.98 | 46.7 | 416.3 | 11.47 | 40.88 | 1.134 | −20.01 | −244.5 | −0.123 |
| 21:00 | **2375** | 2.096 | 0.3569 | 293.63 | 49.0 | 416.0 | 11.77 | −21.35 | −3.172 | −3.72 | 190.4 | 0.158 |
| 21:30 | 33265 | 3.619 | 0.3831 | 292.80 | 51.9 | 415.6 | 11.87 | 25.75 | 0.123 | 4.48 | −195.3 | −0.154 |
| 22:00 | 35639 | 2.854 | 0.5483 | 292.49 | 52.9 | 415.8 | 11.87 | 32.62 | 2.478 | 9.52 | −448.7 | −0.067 |
| 22:30 | 35639 | 2.944 | 0.5026 | 292.19 | 52.7 | 415.9 | 11.59 | 44.77 | 1.542 | 21.06 | −249.1 | −0.120 |
| 23:00 | 35640 | 2.813 | 0.5182 | 291.95 | 52.8 | 415.5 | 11.44 | 42.77 | 2.326 | 17.56 | −286.9 | −0.105 |

`WPL_APPLIED = 1` everywhere; `N_DUP = 0` everywhere; spikes over the whole run are u 0, v 0,
w 0, ts 0, co2 2, h2o 16 samples out of ~211 000.

**Sensible for an urban site — and they would not be for a rural one.** `H` positive at night
(+26 to +46 W m-2) with `z/L` negative would be a red flag over grassland or forest; at a city
site stored and anthropogenic heat keep the surface layer near-neutral to weakly unstable long
after sunset (the one negative `H` is the 2375-sample fragment, §1.3). `u*/WS` of 0.11–0.19,
against 0.05–0.10 at a rural site, is the very rough surface a built-up fetch is. `FC` at +1.1 to
+3.7 µmol m-2 s-1 has the right sign for a city at night and sits at the low end of published
urban fluxes — but see §1.2, because it is mostly WPL. `LE` is −20 to +21 W m-2, small and of
inconsistent sign; the −20 W m-2 period at 46 % RH is a downward water vapour flux with no dew to
explain it, read here as noise plus advection of drier air — not as a defect, and not as evidence
of correctness either. The rest is an ordinary early-September evening: CO2 415–417 ppm falling
slowly, H2O 10.5–11.9 ppt, RH 42→53 % as the air cools 294.5 → 292.0 K because the temperature
falls and not because the absolute humidity moves; `PHI` (pitch) 0.1–2.6° and varying, which a
fixed sonic tilt would not be, so it is the real mean vertical wind the second rotation removes.

### 1.2 The honest caveat on `FC`: WPL is the larger term

| period END | FC_L0 | FC | WPL contribution | share of FC |
|---|---|---|---|---|
| 20:00 | 2.246 | 3.662 | +1.416 | 39 % |
| 20:30 | −0.640 | 1.134 | +1.775 | 156 % (sign flip) |
| 21:00 | −2.113 | −3.172 | −1.059 | 33 % |
| 21:30 | −1.156 | 0.123 | +1.279 | 1037 % (sign flip) |
| 22:00 | 0.820 | 2.478 | +1.659 | 67 % |
| 22:30 | −0.812 | 1.542 | +2.354 | 153 % (sign flip) |
| 23:00 | 0.100 | 2.326 | +2.226 | 96 % |

On four of seven periods **WPL changes the sign of the CO2 flux**. That is not a miniflux problem
— it is the known open-path situation when `H` is large and the CO2 flux small — and the term is
the right size: the thermal part of Webb Eq. 24 predicts +1.96 µmol m-2 s-1 for the 20:30 period
against the +1.78 observed. It means the accuracy of `FC` here rests on the accuracy of `H`, not
on `w'c'`. The Schotanus correction on `H` is tiny by comparison: `H − H_L0` is −1.3 to
+1.2 W m-2, at most 3.0 % of `H`.

### 1.3 What went wrong, and what miniflux refused

* **The run exits 3 (`ReadError`), by design, on the last file.** The final line of
  `20213_RAW_20Hz_2025-09-08-2330.csv` is `"2025-09-08 23:03:53.8` — a record the logger was cut
  off mid-write. miniflux reports `line 2258: the row holds 1 field(s) but the configuration reads
  column 12` and stops; the seven complete periods above were written before it did. The config
  still points at all seven files rather than dodging the bad one, because dodging it would hide
  the finding.
* **The 21:00 period holds 2375 samples (2 minutes), not 36000** — a 50-minute hole in the input.
  It is the outlier in every column: the only negative `H`, a positive (wrong-signed) `COV_U_W`, a
  `u*` carried entirely by `cov_v_w`. A 2-minute record cannot resolve the flux-carrying eddies.
  It is reported rather than dropped (`min_samples = 0`) so that `N_IN` is what disqualifies it.
* **Every file arrives at 19.8 Hz against the declared 20 Hz** (35 633–35 640 rows per 30 min,
  ~1 % dropped by the logger). miniflux uses the declared rate, as documented. So the lag search
  shifts by *sample index* — on a gappy record `n` samples is not exactly `n/20` s — and the
  `SST`/`ITC` sub-intervals are likewise index-based.
* **The lag search hits the window edge on four of seven periods**, where `covmax_default` falls
  back to the nominal 0 s and logs it. For a collocated open path the true lag *is* ~0, so the
  fallback does the right thing — but it also says the covariance curve has no interior peak, i.e.
  no detectable lag structure in a weak nocturnal flux. Where an optimum was found it was −1.9 to
  +0.1 s, scattered.
* **`SST_FLAG` is 2 (>100 %) on three of seven periods and 1 on two more.** With
  `steady_state_pair = w,co2` and a CO2 covariance near zero the Foken–Wichura ratio has a
  vanishing denominator. Expected; reported, not suppressed.
* **`ITC_T` reaches 1.53.** Near-neutral periods make the temperature ITC model ill-posed, and
  `CONTRACT §20.5` uses `|T*|` where the parent uses signed `T*`, so this column is not comparable
  with the parent's on unstable periods by construction.

---

## 2. FR-Gri, 2022-05-14 — against EddyPro 7.0.9

> **Provisional, and scheduled for replacement.** It compares the wind and sonic-temperature chain
> only, because the analyser is closed-path and miniflux currently refuses that input. When
> closed-path support lands, the gas fluxes become comparable and this section is rewritten
> against a fuller EddyPro output.

`python -m miniflux run examples/fr_gri_wind_sonic.ini` over the 48 files
`FR-Gri_EC_20220514*.csv` — a full day, `N_IN = 36000` on every period, none missing, 38 s wall
clock. Reference: `OUTPUT/EddyPro7/eddypro_FR-Gri_sample_full_output_2025-12-02T001726_adv.csv`,
with the matching `_fluxnet_` file for per-sample spike counts and `_qc_details_` for the
Foken–Wichura percentages.

Site values are verbatim from the reference's metadata (`eddypro/FR-Gri.metadata`):
`latitude = 48.844243`, `measurement_height = 2.00`, `north_offset = 250.0`,
`canopy_height = 0.15`, giving `displacement = 0.10 m` and `(z − d) = 1.90 m` — which is what
EddyPro's own columns imply (2.54698 × 0.745786 = 1.8995), so EddyPro derives `d = 2h/3` too even
though the metadata's `displacement_height` says 0. Method settings match the reference's
`.eddypro` file one for one: `rot_meth = 1` → double rotation, `detrend_meth = 0` → block average,
`despike_vm = 1` (Mauder 2013) → MAD at q = 7.0, `tlag_meth = 2` → `covmax_default`,
`avrg_len = 30` → 30 minutes. The two programs are asked to do the same thing.

**Not compared.** The analyser is a closed-path LI-7200 with 71.1 m of tube; miniflux has no
tube-attenuation correction and the refusal on closed-path input stands. **`FC`, `LE`, `E`,
`FC_L0`, `LE_L0`, `E_L0` and the two gas lags from this run are discarded and appear nowhere
below.** They are numbers; they are not fluxes. CO2 and H2O still had to be *declared* —
`[variables]` requires all six high-frequency columns (`CONTRACT §5.2.6`), and `H` needs a water
vapour flux for the Schotanus correction and a humidity for `rho` and `cp`, so running with no
water would be a larger error than running with an attenuated one. Both are dry mole fractions, so
`wpl = auto` correctly decides nothing is owed, and the contamination is confined to `H − H_L0`;
the `H_L0` vs `un_H` comparison is free of it.

### 2.1 A finding before the comparison: the reference's rows are labelled 30 minutes late

`FR-Gri_EC_202205141200_L05_F01.csv` holds samples from 11:30:00.05 to 12:00:00.00 — the file name
is the **end** of its half hour. The metadata declares `tstamp_end = 0` (file name = start) and
EddyPro never reads the `TIMESTAMP` column (`col_ts = 0`), so it cannot notice: it labels that row
**12:30** where miniflux, labelling by the sample clock, calls it **12:00**. Found, not assumed:
scoring RMSE/RMS against `T_SONIC` / `COV_W_TS` / `VAR_W` over offsets {−30, 0, +30, +60} min
gives 0.0041 / 0.4068 / 0.3544 at −30, 0.0023 / 0.2649 / 0.2828 at 0, **0.0000 / 0.0045 / 0.0039
at +30**, and 0.0023 / 0.2647 / 0.2826 at +60. **Every comparison below therefore pairs miniflux's
period ending at T with EddyPro's row labelled T + 30 min.** Without it `cov_w_ts` would show a
median deviation of −10.7 % instead of +0.003 %, and that would have been reported as a processing
difference.

### 2.2 The comparison

48 periods, no exclusions. Deviation is `100 × (miniflux − EddyPro) / EddyPro`; "med" is the
median, `[p16, p84]` the 16th/84th-percentile spread, `|max|` the largest absolute deviation over
the 48, and `n` counts the periods where EddyPro's value is far enough from zero to form a ratio.

**(a) Like for like — neither side spectrally corrected.** EddyPro publishes `un_Tau`, `un_H` and
the factors `Tau_scf`, `H_scf` separately, which makes this possible.

| quantity | miniflux | EddyPro | n | med % | [p16, p84] | \|max\| % |
|---|---|---|---|---|---|---|
| sonic heat covariance | `COV_W_TS` | `w/ts_cov` | 33 | **+0.003** | [−0.044, +0.003] | 5.103 |
| uncorrected H | `H_L0` | `un_H` | 36 | **+0.005** | [−0.045, +0.006] | 8.163 |
| uncorrected u* | `USTAR` | `u*/sqrt(Tau_scf)` | 41 | **+0.001** | [−0.138, +0.002] | 3.074 |
| variance of u | `VAR_U` | `u_var` | 48 | +0.003 | [−0.054, +0.003] | 1.707 |
| variance of v | `VAR_V` | `v_var` | 48 | +0.003 | [+0.002, +0.003] | 0.614 |
| variance of w | `VAR_W` | `w_var` | 48 | +0.003 | [−1.055, +0.003] | 10.827 |
| variance of Ts | `VAR_TS` | `ts_var` | 48 | +0.003 | [+0.002, +0.003] | 13.716 |

**(b) Fully corrected, and the same with the reference's correction undone.**

| quantity | n | med % | [p16, p84] | \|max\| % |
|---|---|---|---|---|
| `H` vs `H` | 36 | **−2.228** | [−4.745, −1.995] | 12.830 |
| `USTAR` vs `u*` | 42 | **−0.795** | [−1.452, −0.606] | 4.391 |
| `MO_LENGTH` vs `L` | 38 | **−3.891** | [−15.024, +5.054] | 106.850 |
| `H × H_scf` vs `H` | 36 | **+0.031** | [−0.215, +0.172] | 8.642 |
| `USTAR × sqrt(Tau_scf)` vs `u*` | 42 | **+0.001** | [−0.113, +0.002] | 3.074 |

EddyPro's own factors over these periods: `H_scf` median 1.0240 [1.0210, 1.0472], max 1.1213;
`Tau_scf` median 1.0168 [1.0124, 1.0286], max 1.0607.

**(c) Geometry.** `WS` vs `wind_speed`: med −0.127 %, [−0.329, −0.020], |max| 5.587 %.
`WD` − `wind_dir` (circular): med **−1.360°**, [−5.269, +3.349], range −22.166 to +11.320.
`THETA` − `yaw` (circular): med **0.000000°**, |max| 0.0165. `PHI` − `pitch`: med **0.000000°**,
|max| 0.3141. `THETA` is reported on (−180, 180] and `yaw` on [0, 360) — the same angle on a
different branch, which is why the difference is taken circularly.

**(d) Mean state.** `TA` med −0.002 % (|max| 0.032), `T_SONIC` +0.000 (0.004), `PA` 0.000
(0.000), `RHO_A` +0.001 (0.093), `CP` +0.002 (0.004), `CO2_MEAN` and `H2O_MEAN` both +0.0000
against EddyPro's dry mixing ratios. The two exceptions are `RH` at **+0.316 %** [+0.229, +0.384],
|max| 1.035 and `Q` at **+0.319 %** [+0.290, +0.334], |max| 0.646 — explained in §2.3(f).

**(e) Spike counts, per sample, day totals** (miniflux `N_SPIKE_*` / EddyPro `*_SPIKE_NREX`):
u 3/3, v 10/10, w **918/918**, ts **1039/1038**. Not just the totals — the per-period counts match
on 47 of 48 periods for w and ts and on all 48 for u and v, the one exception being 01:30, ts:
miniflux 36, EddyPro 35.

### 2.3 Every deviation, explained

**(a) The +0.00278 % floor: `N` versus `N − 1`.** On the 31 periods where neither program flagged
a sample in u, v, w or ts the residual is constant, not scattered: median +0.00273 to +0.00290 %
on `COV_W_TS` and all four variances (min +0.00129, max +0.00420), and `N/(N−1) − 1` at
`N = 36000` is **+0.00278 %**. EddyPro's `CovarianceMatrixNoError`
(`src_common/stats_operator_no_error.f90`) divides the second moments by `Nact` where
`ALGORITHMS §7.1` prescribes `/(N_p − 1)`; its own `StDev` uses `Nact − 1`, so the exported
variances come from the covariance-matrix diagonal. A convention difference; neither is wrong.

**(b) The 15 periods deviating more than 0.1 %: what is done with a spike.** Both programs flag
the same samples (§2.2e), then part company — miniflux sets a flagged sample to NaN and the
covariance uses a pairwise-complete mask, so it is **dropped**, while EddyPro
(`src_rp/test_spike_detection_mauder_13.f90`, `filter_sr = 1`) **replaces each flagged run by a
straight line** and re-runs the test, up to 10 passes. Every period deviating more than 0.1 % has
a non-zero spike count; every period with none sits on the floor above. The worst, **+14.6 % at
01:00**, is 66 replaced samples on a covariance of 0.0006 K m s-1 — a near-zero nocturnal flux
where those samples are the whole signal, and **0.11 W m-2** in absolute terms. The one-sample
count difference at 01:30 has a candidate in the same file: EddyPro's `do i = 2, N-1` never tests
a period's first or last sample, where miniflux tests all of them (`ALGORITHMS §3.1`). Note that
EddyPro's `*_NUM_SPIKES` counts spike **runs** (121 in w), not samples — use `*_SPIKE_NREX` from
the FLUXNET file — and that 918 samples in 121 runs, a mean run of 7.6, confirms `ALGORITHMS
§3.4`.

**(c) `H` is 2.2 % low and `u*` 0.8 % low: miniflux has no spectral correction.** Not an error, a
missing feature (`ALGORITHMS §13`), and predictable from EddyPro's own factors:
`1/1.0240 − 1 = −2.34 %` against the observed −2.228 % for `H`, `1/sqrt(1.0168) − 1 = −0.83 %`
against −0.795 % for `u*`. **Anyone using miniflux's `H` or `u*` for science is using an
under-corrected flux**, by about 2 % and 1 % on a 2 m sonic over a low crop and by more on a
taller tower or a slower analyser.

**(d) `WD` differs by up to 22°: the two average directions differently.** `WS` agrees to
−0.127 %, so the mean wind *vector* is the same; the estimator differs. miniflux
(`ALGORITHMS §4.5`) takes the direction **of the mean vector**, EddyPro
(`src_common/wind_direction.f90`) a **unit-vector angular mean of all 36000 per-sample
directions** — recomputed from the raw `U`, `V` columns on six periods, that reproduces its
published `wind_dir` to **0.000°** every time. The two coincide in steady wind and diverge when it
wanders, the angular mean weighting every sample equally regardless of speed: the three lightest
periods (0.121, 0.174, 0.186 m s-1) carry three of the four largest deviations (+10.4°, −22.2°,
−22.1°). Not monotone in speed though — Spearman `|ΔWD|` against `WS` is **−0.28** — because the
gap depends on the shape of the directional distribution. The whole day is unusually calm
(`WS` ≤ 1.5 m s-1), so on a windy day this would be a fraction of a degree. Neither estimator is
wrong; they answer different questions.

**(e) `THETA` and `PHI` agree exactly, and the exceptions are spikes.** `THETA`'s largest
deviation, 0.0165°, is EddyPro's six-significant-figure output rounding. Every `PHI` deviation
above 0.01° (0.3141, 0.2790, 0.0475, 0.0438, 0.0373, 0.0249°) is on a period with flagged spikes —
cause (b) again, the rotation angle coming from nan-means that a dropped sample moves and an
interpolated one does not. Six of 48.

**(f) `RH` and `Q` are +0.32 % high: a temperature swapped between two adjacent formulas.** The
concentrations agree to 0.0000 %, so the difference is downstream of them. Against EddyPro's
published `e` over all 48 periods, `chi_wet · P` (the mole-fraction identity, what miniflux does)
deviates **+0.2995 %** and `chi_wet · P · (Ta/Ts)` **−0.0257 %**: the reference's vapour pressure
carries a spurious `Ta/Ts`, the vapour molar density being formed at one temperature and converted
back to a partial pressure at the other. `Ts − Ta` is 0.83 K on 292 K, i.e. 0.28 % — the whole of
the gap. It is the trap `ALGORITHMS §8.3` is written about, and miniflux keeps one temperature
throughout. Consequence for what this section validates: **none measurable** — humidity reaches
`H` through `cp` (+0.002 %) and the Schotanus denominator `1 + 0.51q`, where 0.32 % of q = 0.0068
moves it by 0.002 %.

**(g) `MO_LENGTH` −3.9 % at the median, 107 % at worst: three documented differences.**
Reconstructed from EddyPro's own columns, `−Ta·(1e5/P)^0.286 · u*³ / (0.41·g·H/(ρ·cp))` matches
its `L` to a median |error| of **0.0004 %** (max 0.0015 %); with `Ts` for `Ta` it degrades to
0.3255 %, and with `Ts` plus the uncorrected `w'Ts'` to 4.8155 % (max 105.28 %). So EddyPro uses
`κ = 0.41`, the ambient temperature and the corrected heat flux where miniflux uses `κ = 0.40`,
the sonic temperature (`ALGORITHMS §9.4`) and the raw covariance — worth +2.5 %, +0.29 % and about
−0.1 %, the right order for the median. The 107 % is 19:00, where `L` passes through a pole as the
flux crosses zero (+277.7 vs +134.3 m) and a ratio of two large `L` values means nothing. `(z−d)`
is 1.90 m on both sides, so `ZL` carries the `L` deviation and nothing extra.

**(h) `VAR_W` and `VAR_TS` reach 10.8 % and 13.7 %** — both on low-flux night periods with the
heaviest spike replacement, cause (b). `VAR_V`'s 0.614 % maximum is the 16:00 period with 9
v-spikes, which propagate into `COV_W_TS` through the rotation: that period shows −0.029 % rather
than the floor.

### 2.4 How much of `H` rests on the pressure

`examples/fr_gri_wind_sonic.ini` declares `pressure_pa = 99767.5` because that is the constant in
EddyPro's `air_pressure` column for every half hour — EddyPro had no ambient-pressure input
(`col_air_p = 0`) and fell back to the 125 m station altitude. Declaring the reference's own
stated pressure makes the comparison measure turbulence processing and not barometry. It is *not*
the best physical value: the LI-7200's box barometer, `PRESS_BOX`, reads **100.65 kPa** that day,
0.89 % higher, and the cell reads 99.28 kPa, 0.49 % lower.
`examples/fr_gri_wind_sonic_pbox.ini` is the same run with `p_air = PRESS_BOX` and nothing else
changed: `PA`, `RHO_A`, `H_L0`, `H` and `RH` all move **+0.889 %** (median) while `TA`, `USTAR`
and `COV_W_TS` do not move at all, and against the reference `H_L0` then sits **+0.774 %** above
EddyPro's `un_H` instead of +0.005 %. The reading: **0.9 % of `H` at this site is a choice about
ambient pressure, and the better measurement disagrees with the reference by more than the entire
turbulence processing does.** `TA` does not move because `e/p` is the mixing ratio, which is what
the sonic-temperature conversion uses.

### 2.5 Quality flags

Against EddyPro's `_qc_details_` file, which publishes the Foken–Wichura deviations directly. A
secondary comparison: the two use the same test but not the same sub-interval bookkeeping, and
`CONTRACT §20.1` records a deliberate estimator choice.

| statistic | result |
|---|---|
| `SST_PCT` − EddyPro `dev(w/ts)`, on the 24 periods where EddyPro's value is 5–100 % | median **−0.90** pp, [p16 −12.89, p84 +9.47], \|max\| 23.18 pp |
| same, all 48 periods | median +0.04 pp, \|max\| 596 pp |
| `SST_FLAG` class vs EddyPro's 1–9 flag binned into three | **42 of 48 agree** |
| `ITC_W` − `dev(w)`/100 | median +0.0054, [+0.0010, +0.0091], \|max\| 0.0198 |
| `ITC_U` − `dev(u)`/100 | median +0.0051, [+0.0009, +0.0088], \|max\| 0.0189 |
| `ITC_T` − `dev(ts)`/100 | median −0.0092, [−0.0533, +0.0004], \|max\| 0.8926 |

`ITC_T` is **not comparable by construction** — `CONTRACT §20.5` divides by `|T*|` where EddyPro
and the parent divide by a signed `T*` — and the 0.89 outlier is an unstable period, exactly where
that bites. The steady-state percentages track each other when the flux is well developed and
diverge wildly when the covariance approaches zero, the statistic being a ratio with a vanishing
denominator; the 596 pp maximum is the 23:00 period where EddyPro reports 2861 % and miniflux
2265 %, which are the same verdict.

---

## 3. The two kernel paths give the same numbers on real data

`CONTRACT §3.8` promises identical decisions and identical elementwise results between the numpy
fast path and the pure-Python path. Tested on real 36000-sample periods, not fixtures: the first
12 FR-Gri periods re-run with `--pure` and the output CSV compared cell by cell. **12 periods ×
52 columns = 624 cells, 624 byte-identical (100.00 %), worst relative difference 0.000e+00** at
the `%.6g` output precision.

---

## 4. What this does not prove

Read this as carefully as the tables.

1. **Not validated against a certified reference, because none exists.** One day at one site
   against EddyPro 7.0.9, and seven half hours end to end at another. Nothing here licenses using
   miniflux where EddyPro or ONEFlux_preproc is called for.

2. **Agreement with EddyPro is agreement with an implementation, not with the atmosphere.** Both
   could share a wrong convention and this comparison would not see it. Where they demonstrably
   *disagree* (§2.3) is where this exercise has information; agreement to 0.003 % mostly shows
   that two implementations of the same documented recipe produce the same number.

3. **§2 validates the wind and sonic-temperature chain only** — despiking, double rotation, block
   detrending, the second moments, `u*`, the mean thermodynamic state and `H`. `FC`, `LE` and `E`
   are compared against **no reference anywhere in this document**, the FR-Gri gas outputs having
   been discarded as closed-path. FR-Jus exercises WPL end to end and shows terms of the right
   size and sign, which is not the same as the right number.

4. **The WPL correction is unvalidated against a reference.** §1.2 shows it dominating `FC` on an
   open-path site and flipping the flux sign on four of seven periods. If WPL is wrong, every
   open-path `FC` miniflux produces is wrong, and nothing here would have caught it.

5. **The lag search is barely exercised.** FR-Gri's gas lags were discarded; FR-Jus is a
   collocated open path whose true lag is zero, and the `covmax_default` fallback fired on four of
   seven periods. A separated open path or a closed-path tube — the cases the search exists for —
   is untested against a reference here.

6. **Two days is two days.** 48 + 7 half hours, two instruments, two sites, one season each, one
   latitude band. No cold, no high wind, no strong stability, no rain, no sensor failure beyond
   one truncated record, no gap-riddled day beyond FR-Jus's 1 % dropout. FR-Gri is exceptionally
   calm, which magnifies the wind-direction difference and suppresses the spectral correction.

7. **The FR-Jus assessment is a plausibility argument, not a measurement.** Reasoning about sign
   and order of magnitude would not detect a 10 % error, or a 30 % one, and that site has no
   reference output and no independent measurement — no radiation, no storage, no energy-balance
   closure. Two of its site values are invented, the 30 m height and the Paris latitude: `ZL` and
   the ITC columns there are conditional on them, `MO_LENGTH` and every flux are not.

8. **`H` and `u*` are systematically low by about 2 % and 1 % on this dataset** because miniflux
   applies no spectral correction and never will (`ALGORITHMS §13`). On a taller tower, a slower
   sensor or a longer tube the deficit is larger — possibly much larger. A known, quantified,
   permanent limitation, not a bug to be fixed.

9. **A comparison this close can hide a shared mistake in the *inputs*.** The 30-minute label
   offset in §2.1 was found only because three independent quantities agreed about it; something
   similar and smaller — a column mapped to the wrong variable, a unit declared twice — would look
   like a processing difference and could be explained away. The defence is that §2.3 explains
   every deviation down to a named line of reference source, with no residual left over.

10. **Nothing here tests the refusals.** The unit tests do (`CONTRACT §21`); this document
    exercised exactly one in the field — the truncated final record at FR-Jus, §1.3 — and it
    behaved as documented.
