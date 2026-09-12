# miniflux — MODULE CONTRACT

The integration contract. Every module, every public function, every key it reads and every
key it writes. This document is prescriptive: where it says "exactly", implement exactly
that. Where two of the extraction specs disagreed, §20 records the decision and why.

Companion document: `ALGORITHMS.md` holds the mathematics. This one holds the plumbing.

Rules that bind every module:

* **Standard library only.** `numpy` may be imported *only* inside `kernels.py`, inside a
  `try/except ImportError`, and only as a fast path (§3.9).
* **Python >= 3.8.** No walrus-in-comprehension cleverness, no `match`, no
  `from __future__ import annotations` requirement, no dataclass slots.
* **No classes except the four named here** (`Config` namespaces, `Writer`, and the three
  exceptions). No registry, no plugin discovery, no provenance layer, no units library.
* **Nothing crosses a module boundary except the `period` dict, the `cfg` object, and plain
  floats/ints/strings.**

---

## 1. The period dict

A period is a plain `dict`. It is created by `read.py` and mutated in place by each step.

```python
period = {
    'meta':  {},                 # dict[str, float|int|bool|str|datetime]
    't':     [datetime, ...],    # length N, strictly the parsed sample clock
    'u':     array('d'),         # length N
    'v':     array('d'),
    'w':     array('d'),
    'ts':    array('d'),
    'co2':   array('d'),
    'h2o':   array('d'),
    'ta':    array('d'),         # all-NaN when no air-temperature column exists
    'p_air': array('d'),         # constant-filled when [site] pressure_pa is used
}
```

All nine series keys are **always present** and always `array('d')` of the same length `N`.
A variable with no source in the file is an all-NaN array, never a missing key. `t` is a
plain `list` of `datetime.datetime` (naive, no timezone), length `N`, non-decreasing.

`period['meta']` is **flat**: string keys, scalar values. No nesting, no tuples as keys.
Every meta key is listed in §17.

### 1.1 The step signature — the one extension point

```python
def step(period, cfg):
    """period: the dict above. cfg: the Config of §5. Returns a period dict."""
    ...
    return period
```

A step mutates and returns the same dict (returning a *new* dict is allowed and must be
handled by the caller, which always rebinds). A step:

* must not raise on physically degenerate data — it writes `float('nan')` and logs at
  `WARNING`;
* must not read a config key outside its own section (plus `[period]`, which is global);
* must not import another step module (steps are independent; shared code lives in
  `kernels.py`, `constants.py` or `flux.py` helpers).

Adding a step means writing such a function and adding its name to `pipeline.STEPS`.
Nothing else.

---

## 2. Canonical variable vocabulary

One row per canonical name. These names are used as dict keys, as `[variables]` keys and in
log messages. Nothing else in the program has a name.

| name | meaning | internal unit | where it comes from |
|---|---|---|---|
| `u` | sonic wind, +x along the sonic axis (forward) | m s-1 | file column, `[variables] u` |
| `v` | sonic wind, +y to the left | m s-1 | file column, `[variables] v` |
| `w` | sonic wind, +z up | m s-1 | file column, `[variables] w` |
| `ts` | sonic (virtual) temperature | K | file column, converted from degC if declared |
| `co2` | CO2 dry mole fraction **or** molar density | mol mol-1 (dry) \| mol m-3 | file column, converted per `[units]` + `[gases]` |
| `h2o` | H2O dry mole fraction **or** molar density | mol mol-1 (dry) \| mol m-3 | file column, same |
| `ta` | measured ambient air temperature | K | file column if declared, else all-NaN |
| `p_air` | ambient (barometric) pressure | Pa | file column if declared, else the declared constant |

`co2` and `h2o` carry the unit implied by their `measure_type`: `mixing_ratio` -> dry mole
fraction in **mol mol-1** (a 420 ppm signal is `4.20e-4`); `molar_density` -> **mol m-3**.
`read.py` performs that normalisation; no scale factor survives past `read.py`.

Names that exist in the parent and deliberately **do not** exist here: `cell_t`,
`cell_pressure` (closed-path only — refused, see §20.4), `ch4` and any other gas, `sos`,
diagnostic/status columns.

---

## 3. `kernels.py` — the array layer

The only module that knows about `array('d')` layout, NaN bookkeeping and numpy. Every
other module goes through it for anything that touches more than one sample.

```python
NAN = float('nan')
HAVE_NUMPY  # bool, True when `import numpy` succeeded AND [runtime] use_numpy is not off
```

### 3.1 Construction

```python
new(n, fill=NAN)                 -> array('d')   # length n, every element = fill
from_values(values)              -> array('d')   # iterable of float|None; None -> NAN
copy(x)                          -> array('d')
```

Pure Python only; there is no numpy path (these are allocation, not arithmetic).

### 3.2 Reductions

```python
count_finite(x, start=0, stop=None)   -> int
nansum(x, start=0, stop=None)         -> (float, int)   # (sum of finite, count of finite)
nanmean(x, start=0, stop=None)        -> float          # NAN when count == 0
nanmedian(x)                          -> float          # NAN when no finite sample
mad(x, med)                           -> float          # median(|x_i - med|) over finite x_i
```

`nansum` accumulates **left to right**, no compensation. `nanmedian` collects the finite
values, sorts them, and returns `v[k//2]` for odd `k`, `(v[k//2-1] + v[k//2]) / 2` for even
`k` — the parity is that of the surviving count `k`, not of `len(x)`. `mad` is a second
`nanmedian` over the absolute deviations; it does **not** apply the 0.6745 scaling.

`start`/`stop` are half-open Python slice bounds; they exist so `qc.py` can take
sub-interval statistics without copying.

### 3.3 Second moments

```python
cov(x, y, start=0, stop=None)    -> float
variance(x, start=0, stop=None)  -> float   # == cov(x, x, start, stop)
nanstd(x, start=0, stop=None)    -> float   # sqrt(variance(...))
```

Exactly the recipe of ALGORITHMS §7.1: joint finite mask, both means over that mask,
two-pass, `/(N_p - 1)`, `NAN` when `N_p < 2`. `cov` must be translation-invariant to 1e-9
relative; a one-pass raw-sums implementation is forbidden here.

### 3.4 Despiking primitives

```python
mad_spike_mask(x, q)             -> bytearray   # length len(x), 1 = outside the bounds
apply_mask_nan(x, mask)          -> None        # in place: x[i] = NAN where mask[i]
```

`mad_spike_mask` computes `m = nanmedian(x)`, `d = mad(x, m)`,
`lower = m - (q*d)/0.6745`, `upper = m + (q*d)/0.6745` **in that evaluation order**, then
sets `mask[i] = 1` where `x[i] < lower or x[i] > upper` (strict; NaN gives 0). It does
**not** apply the trailing-run rule — that is `despike.py`'s job.

### 3.5 Series transforms

```python
shift_truncate(x, lag)           -> array('d')  # out[i] = x[i+lag] where in range, else NAN
sub_const(x, c)                  -> array('d')
sub_const_inplace(x, c)          -> None
sub_line_inplace(x, slope, offset) -> None      # x[i] -= slope*i + offset
linfit(x)                        -> (slope, offset)
rotate_double(u, v, w)           -> (u2, v2, w2, theta, phi)
```

`shift_truncate` never wraps. `linfit` is ALGORITHMS §6.2 over the **original** index,
returning `(0.0, nanmean(x))` when fewer than 3 finite samples remain and
`(NAN, NAN)` when none do. `rotate_double` computes the two nan-means, the two angles, then
applies the **two sequential rotations per sample** (u1/v1/w1 then u2/v2/w2) in one pass —
not the composite matrix, so the arithmetic matches the reference implementation
operation for operation.

### 3.6 Lag search

```python
cov_at_lag(w, c, lag)            -> float
cov_curve(w, c, lo, hi)          -> list[float]   # one entry per lag in [lo, hi] inclusive
```

Overlap-only slices, joint finite mask, per-lag count `k`, one-pass form
`(Sab - Sa*Sb/k)/(k-1)`, `NAN` when `k < 2` (ALGORITHMS §5.3). **The caller is responsible
for pre-centring**: `lag.py` subtracts each series' whole-period nan-mean before calling.
`cov_curve` is the hot loop; it may use prefix sums of the finite-masked series so that
`Sa` and `Sb` are O(1) per lag and only the dot product is O(n).

`cov_curve(w, c, lo, hi)` returns `hi - lo + 1` values. Both ends inclusive. If
`hi < lo` the caller has already swapped them; `cov_curve` raises `ValueError` rather than
returning an empty list.

### 3.7 Which functions get a numpy fast path

Fast path (zero-copy view: `np.frombuffer(arr, dtype=np.float64)` over the `array('d')`
buffer, which is writable, so in-place kernels stay in place):

`count_finite`, `nansum`, `nanmean`, `nanmedian`, `mad`, `cov`, `variance`, `nanstd`,
`mad_spike_mask`, `apply_mask_nan`, `shift_truncate`, `sub_const`, `sub_const_inplace`,
`sub_line_inplace`, `linfit`, `rotate_double`, `cov_at_lag`, `cov_curve`.

Pure Python only: `new`, `from_values`, `copy`.

### 3.8 Parity contract between the two paths

Restating ALGORITHMS §12 as a test obligation:

* **bit-identical**: `mad_spike_mask`, `apply_mask_nan`, `shift_truncate`, `sub_const`,
  `sub_line_inplace`, `rotate_double` (elementwise, same IEEE operations in the same order),
  `nanmedian` (exact order statistics), `count_finite`.
* **within 8 ULP (assert 1e-12 relative)**: `nansum`, `nanmean`, `mad`, `cov`, `variance`,
  `nanstd`, `linfit`, `cov_at_lag`, `cov_curve`. numpy sums pairwise; the pure path sums
  left to right.
* **decisions must match exactly**: the despike mask, the lag selected by `lag.find_lag`,
  and the QC flag are identical on both paths for every test fixture.

`tests/test_kernels_parity.py` skips itself when numpy is absent.

### 3.9 Disabling numpy

`[runtime] use_numpy = auto | on | off`. `on` with numpy absent is a `ConfigError`. The CLI
flag `--pure` forces `off` for a single run. `HAVE_NUMPY` is resolved once, at
`config.load()` time, by calling `kernels.set_numpy(enabled)`.

---

## 4. `constants.py`

Module-level floats, nothing else. No pint, no units, no functions.

```python
R    = 8.314462618      # J mol-1 K-1  universal gas constant, CODATA 2018
RD   = 287.04           # J kg-1 K-1   dry air  -- EddyPro's rounded value, NOT R/MD
RV   = 461.5            # J kg-1 K-1   vapour   -- EddyPro's rounded value, NOT R/MV
MD   = 0.02897          # kg mol-1
MV   = 0.01802          # kg mol-1
MCO2 = 0.04401          # kg mol-1
MU   = MD / MV          # 1.6076581576026636, computed
CPD  = 1005.0           # J kg-1 K-1   intercept of the cp_d fit (paired with 23.12/3364)
G    = 9.81             # m s-2
KAPPA = 0.4             # von Karman
P0A  = 1.0e5            # Pa           reference pressure for potential temperature
MAD_SCALE = 0.6745      # -            0.75 quantile of the standard normal
T0   = 273.15           # K            degC offset
CANOPY_DISPLACEMENT_RATIO = 2.0 / 3.0
Z_MINUS_D_FLOOR = 1.0e-4   # m
```

Every one of these carries the comment shown. `RD`, `RV` and `CPD` additionally carry the
one-line warning from ALGORITHMS §1 about why they are not derived values.

---

## 5. `config.py`

```python
class ConfigError(MinifluxError): ...
def load(path)                 -> Config      # path: str
def describe(cfg)              -> str         # every resolved key, one per line, for the log
```

`Config` is a plain object whose attributes are `types.SimpleNamespace` instances, one per
`.ini` section: `cfg.files`, `cfg.site`, `cfg.period`, `cfg.timestamp`, `cfg.variables`,
`cfg.units`, `cfg.gases`, `cfg.despike`, `cfg.rotate`, `cfg.lag`, `cfg.detrend`, `cfg.wpl`,
`cfg.qc`, `cfg.output`, `cfg.runtime`. Values are already typed (float, int, bool, str,
list of str, dict).

`load` parses with `configparser.ConfigParser(inline_comment_prefixes=('#', ';'))`, applies
the defaults of §6, coerces types, validates, and raises `ConfigError` with a message naming
the section and key on the first problem. It never guesses.

### 5.1 Derived fields `config.py` computes (so no other module repeats the logic)

| attribute | type | rule |
|---|---|---|
| `cfg.period.seconds` | float | `averaging_minutes * 60` |
| `cfg.period.dt` | float | `1.0 / acquisition_frequency` |
| `cfg.site.displacement` | float | declared if non-empty and > 0; else `2/3 * canopy_height` if that is > 0; else 0.0 with a `WARNING` |
| `cfg.units.convert[name]` | `(a, b)` | affine `x_canonical = a*x_file + b`, per §6.4 |
| `cfg.lag.windows[scalar]` | `(nominal_s, min_s, max_s)` | one entry per scalar in `[lag] scalars` |
| `cfg.gases.measure_type[gas]` | str | `'mixing_ratio'` or `'molar_density'` |

### 5.2 Refusals (all `ConfigError`)

1. `closed` not in `{left, right}`; `timestamp.format` not in `{iso, numeric}`;
   `detrend.method` not in `{block, linear}`; `rotate.method` not in `{double, none}`;
   `lag.method` not in `{covmax_default, covmax, fixed, none}`; `wpl.enabled` not in
   `{auto, on, off}`.
2. A `measure_type` of `mole_fraction` (or anything else): *"miniflux supports
   `mixing_ratio` (dry) and `molar_density` only; convert a wet mole fraction upstream"*.
3. A `[units]` value inconsistent with the gas's `measure_type` (e.g. `co2 = ppm` with
   `co2_measure_type = molar_density`).
4. Both `[variables] p_air` and `[site] pressure_pa` empty.
5. Both non-empty (ambiguous — say which one to delete).
6. Any of `u, v, w, ts, co2, h2o` with an empty `[variables]` entry.
7. `acquisition_frequency <= 0`, `averaging_minutes <= 0`, `despike.q <= 0`.
8. `lag.min_s > lag.max_s` for any scalar.
9. A scalar in `[lag] scalars` or `[despike] variables` or `[detrend] variables` that is not
   a canonical name.
10. `runtime.use_numpy = on` with numpy unavailable.
11. `averaging_minutes` not a **whole number of minutes** — tested as
    `abs(m - round(m)) > 1e-9`, so `30`, `30.0` and `60` pass and `0.5` or `7.5` do not.
    `TIMESTAMP_START` and `TIMESTAMP_END` are written at minute resolution
    (`YYYYMMDDHHMM`, §18), so a sub-minute period would print two consecutive rows under
    one label — two different numbers wearing the same name, with nothing in the file to
    tell them apart. Refused rather than emitted. (Numbered last to keep the `5.2.N`
    references in `config.py` and `tests/test_config.py` pointing where they did.)

`describe(cfg)` prints every resolved value including the derived ones, so a run's log
records exactly what it used — including a fabricated-free statement of where the pressure
came from.

---

## 6. The `.ini` file — every key, every default

This is the complete file. Every key is shown with its default; a key may be omitted from a
real file. An **empty value means "not set"** and is never equivalent to `0`.

```ini
# miniflux configuration.

[files]
input_glob      = ./*.dat        # shell glob; matches are read in sorted() order and concatenated
output_csv      = ./miniflux.csv # the per-period table; a sidecar <stem>_units.csv is written beside it
header_line     = 1              # 1-based line number holding the column names
first_data_line = 2              # 1-based line number of the first data row (TOA5: header 2, data 5)
delimiter       = ,              # one character; write \t for tab
quotechar       = "              # csv quoting character
na_values       = -9999,NAN,NaN,nan,   # raw tokens read as NaN; the trailing empty item makes "" NaN
encoding        = utf-8

[runtime]
use_numpy = auto                 # auto | on | off   (fast path only; identical decisions either way)
log_level = INFO                 # DEBUG | INFO | WARNING | ERROR

[site]
site_id            = XX-Xxx      # free text, echoed in the log only
latitude           = 0.0         # degrees north; used ONLY by the ITC test. 0.0 makes Fcor = 0 -> ITC = NaN
measurement_height = 3.0         # zm [m] above ground
canopy_height      = 0.0         # [m]; used only when displacement_height is empty
displacement_height =            # d [m]; empty -> 2/3*canopy_height if that is > 0, else 0.0 (warned)
north_offset       = 0.0         # azimuth of the sonic +u axis [deg]; affects WD only, never a flux
pressure_pa        =             # constant ambient pressure [Pa]; used ONLY if [variables] p_air is empty.
                                 # miniflux never invents a pressure: leave both empty and it refuses to run.

[period]
averaging_minutes     = 30       # the averaging interval A; a WHOLE number of minutes --
                                 # a fractional one is refused (§5.2.11), because the two
                                 # timestamps are written to the minute and a sub-minute
                                 # period would label two rows identically
closed                = right    # right | left -- which side of the boundary a sample on it belongs to
acquisition_frequency = 20.0     # [Hz], authoritative; the observed rate is logged for comparison only
min_samples           = 0        # periods with fewer rows are skipped and logged; 0 keeps every period

[timestamp]
column = TIMESTAMP               # exact column header
format = iso                     # iso      -> YYYY-MM-DD[ T]HH:MM:SS[.fff]
                                 # numeric  -> YYYYMMDDHHMMSS[.ff]  (read as text, never as float)

[variables]                      # canonical name = exact column header. Empty = the column is absent.
u     = Ux
v     = Uy
w     = Uz
ts    = Ts
co2   = CO2
h2o   = H2O
ta    =                          # measured air temperature; empty -> derived from ts per sample
p_air = Press                    # ambient pressure; empty -> [site] pressure_pa

[units]                          # the unit of each column AS WRITTEN IN THE FILE
ts    = K                        # K | degC
ta    = K                        # K | degC
p_air = Pa                       # Pa | hPa | kPa
co2   = ppm                      # ppm | umol/mol | mmol/m3 | mol/m3 | mg/m3 | g/m3
h2o   = ppt                      # ppt | mmol/mol | mmol/m3 | mol/m3 | mg/m3 | g/m3

[gases]
co2_measure_type = mixing_ratio  # mixing_ratio (per mole of DRY air) | molar_density
h2o_measure_type = mixing_ratio  # mole_fraction (wet) is refused -- see CONTRACT section 20.4

[despike]
enabled   = true
variables = u,v,w,ts,co2,h2o     # each is despiked independently, within each period
q         = 7.0                  # threshold in robust standard deviations; half-width = q/0.6745 MADs

[rotate]
method = double                  # double (Wilczak et al. 2001) | none

[lag]
method  = covmax_default         # covmax_default | covmax | fixed (apply the nominal) | none (no shift)
scalars = co2,h2o                # which series are shifted against w
co2_nominal_s = 0.0              # nominal / fallback lag [s], positive = scalar arrives after the wind
co2_min_s     = -2.0             # search window, lower bound [s]
co2_max_s     = 2.0              # search window, upper bound [s]  (both ends inclusive: 2I+1 lags)
h2o_nominal_s = 0.0
h2o_min_s     = -2.0
h2o_max_s     = 2.0

[detrend]
method    = block                # block (mean removal, bookkeeping only) | linear (least-squares line)
variables = u,v,w,ts,co2,h2o     # which series the trend is removed from

[wpl]
enabled = auto                   # auto -> run only when a gas is a molar_density
                                 # on   -> run whenever the inputs exist
                                 # off  -> never; a gas that is OWED a correction then gets NO reported flux

[qc]
steady_state_pair = w,co2        # the covariance the FW96 test is run on
itc               = true         # compute the three ITC deviations

[output]
na_value     = -9999             # written for any non-finite value
float_format = %.6g              # applied to every float column
```

### 6.1 `[units]` -> canonical affine conversions

| declared unit | canonical | `a` | `b` |
|---|---|---|---|
| `K` | K | 1.0 | 0.0 |
| `degC` | K | 1.0 | 273.15 |
| `Pa` | Pa | 1.0 | 0.0 |
| `hPa` | Pa | 100.0 | 0.0 |
| `kPa` | Pa | 1000.0 | 0.0 |
| `ppm`, `umol/mol` (mixing_ratio) | mol mol-1 | 1e-6 | 0.0 |
| `ppt`, `mmol/mol` (mixing_ratio) | mol mol-1 | 1e-3 | 0.0 |
| `mmol/m3` (molar_density) | mol m-3 | 1e-3 | 0.0 |
| `mol/m3` (molar_density) | mol m-3 | 1.0 | 0.0 |
| `mg/m3` (molar_density) | mol m-3 | `1e-6 / M` | 0.0 |
| `g/m3` (molar_density) | mol m-3 | `1e-3 / M` | 0.0 |

`M` is the molar mass of the gas the key names: `MCO2` for `co2`, `MV` for `h2o`. A mass
density is accepted because the conversion is exact and the molar mass is a constant of the
gas, not of the site -- open-path analysers report `mg/m3` and `g/m3`, and refusing them
would mean asking every user to pre-convert their own instrument's native output. A mass
density unit is only legal when that gas's `measure_type` is `molar_density`; declaring
`mg/m3` against `mixing_ratio` is refusal 5.3.

---

## 7. `read.py`

```python
class ReadError(MinifluxError): ...

def periods(cfg)                      -> iterator of period dicts   # chronological
def parse_timestamp(token, kind)      -> datetime                   # kind: 'iso' | 'numeric'
def period_bounds(t, seconds, closed) -> (datetime, datetime)       # (start, end)
```

### 7.1 `periods(cfg)`

Streams. It opens each file matched by `cfg.files.input_glob` in `sorted()` order, skips to
`first_data_line` (keeping `header_line`), and reads rows with `csv.reader`. For each row it
parses the timestamp, converts the declared columns with their affine map, and appends to the
period currently being accumulated. A period is yielded as soon as a sample with a later
period key arrives, and the last one is yielded at end of input.

Reads from config: `[files] *`, `[timestamp] *`, `[variables] *`, `[units] *`,
`[gases] *_measure_type`, `[period] averaging_minutes/closed/min_samples`,
`[site] pressure_pa`.

Writes to the period: all nine series keys, `t`, and these meta keys:

| meta key | type | unit | meaning |
|---|---|---|---|
| `period_start` | datetime | - | |
| `period_end` | datetime | - | the period's label |
| `n_in` | int | samples | rows accumulated into this period |
| `n_dup` | int | samples | duplicate timestamps dropped (first kept) |
| `freq_hz` | float | Hz | `cfg.period.acquisition_frequency`, verbatim |

### 7.2 Refusals (`ReadError`, each naming file and line number)

* a timestamp token matching neither supported shape;
* a header missing a column named in `[variables]`;
* a row with fewer fields than the highest column index used;
* a non-numeric token in a data column that is not in `na_values`;
* zero files matched by the glob.

### 7.3 Non-fatal behaviour (logged, never fatal)

* a duplicated `(period, offset)`: keep the first, increment `n_dup`, log once per period at
  `WARNING` with the count;
* a period with `n_in < cfg.period.min_samples`: **not yielded**, logged at `WARNING`;
* an observed sample rate differing from `acquisition_frequency` by more than 1 %: logged
  once per file at `WARNING`. The configured value is still used.

### 7.4 Guarantees to downstream modules

`t` is non-decreasing. All nine series have equal length. A series whose column is absent is
all-NaN (`ta`) or constant (`p_air` from `[site] pressure_pa`). Units are canonical (§2).

---

## 8. `despike.py`

```python
def despike(period, cfg) -> period
```

Config: `[despike] enabled`, `[despike] variables`, `[despike] q`.

For each name in `cfg.despike.variables`, in the order written:

1. `mask = kernels.mad_spike_mask(period[name], cfg.despike.q)`
2. trailing-run rule: `if mask[N-1]:` walk backwards to the last index with `mask[i] == 0`
   and zero every entry from `i+1` to `N-1` inclusive; if no such index exists, zero the
   whole mask.
3. `kernels.apply_mask_nan(period[name], mask)`
4. `period['meta']['n_spike_' + name] = sum(mask)` (the count **after** step 2).

Writes: `n_spike_u`, `n_spike_v`, `n_spike_w`, `n_spike_ts`, `n_spike_co2`, `n_spike_h2o`
(int, samples) — one per despiked variable; variables not in the list get no key, and
`write.py` emits `na_value` for them.

`enabled = false` writes `n_spike_* = 0` for the configured variables and returns unchanged.

No return value other than the period. Never raises.

---

## 9. `rotate.py`

```python
def rotate(period, cfg) -> period
```

Config: `[rotate] method`, `[site] north_offset`.

Always, **even when `method = none`**, it first records the unrotated means, because
`WS`/`WD` and the parent's `_as_measured` problem depend on them:

```
u_unrot_mean = kernels.nanmean(period['u'])
v_unrot_mean = kernels.nanmean(period['v'])
w_unrot_mean = kernels.nanmean(period['w'])
wind_speed   = hypot(u_unrot_mean, v_unrot_mean)
wind_dir     = (180.0 - degrees(atan2(v_unrot_mean, u_unrot_mean)) + north_offset) % 360.0
```

Then, when `method = double`, it replaces `period['u']`, `['v']`, `['w']` with the arrays
returned by `kernels.rotate_double` and records `theta`, `phi`. When `method = none`,
`theta = phi = 0.0` and the series are untouched.

Writes: `u_unrot_mean`, `v_unrot_mean`, `w_unrot_mean` [m s-1], `wind_speed` [m s-1],
`wind_dir` [deg], `theta` [rad], `phi` [rad].

`write.py` publishes `THETA` and `PHI` in **degrees**; the meta values stay in radians.

Self-check, run at `DEBUG` level only (it costs two extra passes). It logs `theta`, `phi`,
the yaw residual and the two rotated sample means, and checks **two different quantities at
two different bounds** (ALGORITHMS §4.2). Both bounds are relative to `wind_speed`, floored
at 1 m s-1:

| checked | bound | why that bound |
|---|---|---|
| `-sin(theta)*u_unrot_mean + cos(theta)*v_unrot_mean` | `1e-9` | an identity in the two means `theta` was built from, so it is exact to rounding on any data |
| `nanmean(v)`, `nanmean(w)` after rotation | `1e-3` | these average over the **joint** NaN mask while the angles came from separate ones; ~1e-5 m s-1 on healthy despiked data is ordinary |

The tight bound deliberately does **not** apply to the rotated means: `1e-9` there fires on
correct arithmetic as soon as despiking gives u, v and w different masks, which is the
normal case. The tight bound also cannot be extended to the pitch angle from out here —
its reference is `nanmean(u1)`, the intermediate once-rotated wind, which does not survive
`kernels.rotate_double`.

`1e-3` is not slack: it is two orders above the mask noise and still an order below the
1.2e-2 m s-1 that ALGORITHMS §4.3's named trap (`phi` from `mean(u)` instead of
`mean(u1)`) leaves in a 3 m s-1 wind. Exactness of the rotated means under a **shared**
mask is pinned in `tests/test_rotate.py`, not asserted at runtime.

A violation of either bound is logged at `ERROR` and does not raise: this step reports, it
never refuses.

---

## 10. `lag.py`

```python
def apply_lags(period, cfg) -> period
def find_lag(w, c, lo, hi, nominal, method) -> (applied, opt, default_used, cov_peak)
```

`find_lag` arguments: `w`, `c` are `array('d')` (already pre-centred by the caller);
`lo`, `hi`, `nominal` are **integer sample lags**; `method` is the string from
`[lag] method`. Returns `(int, int, bool, float)`.

Config: `[lag] method`, `[lag] scalars`, and per scalar `<s>_nominal_s`, `<s>_min_s`,
`<s>_max_s`; `[period] acquisition_frequency` via `meta['freq_hz']`.

`apply_lags` for each scalar `s` in `cfg.lag.scalars`:

1. `L_lo = round(min_s * f)`, `L_hi = round(max_s * f)`, `L_nom = round(nominal_s * f)`;
   swap `L_lo`/`L_hi` if inverted (already refused at config time, so this is defensive).
2. Pre-centre: `wc = kernels.sub_const(period['w'], nanmean(w))`,
   `cc = kernels.sub_const(period[s], nanmean(period[s]))`.
3. `curve = kernels.cov_curve(wc, cc, L_lo, L_hi)`; scan ascending for the strict maximum of
   `abs(value)`, initialising `L_star = L_nom` and `MaxCov = 0.0`.
4. Boundary fallback per `method`:
   * `covmax_default`: `L_star in (L_lo, L_hi)` -> `L_applied = L_nom`, `default_used = True`
   * `covmax`: `L_applied = L_star`, `default_used = False`
   * `fixed`: no scan at all; `L_applied = L_star = L_nom`, `default_used = True`, and
     `<s>_lag_cov = NaN`
   * `none`: no scan, no shift; `L_applied = L_star = 0`, `default_used = False`, and
     `<s>_lag_cov = NaN`

   **The two non-scanning methods report `<s>_lag_cov` as `NaN`, never `0.0`.** No scan
   ran, so there is no covariance to report, and a `0.0` would read as a measured
   covariance that happened to vanish — which is a real and different condition (a stuck
   sensor). The same rule reaches `covmax_default` and `covmax` by the side door: a scalar
   whose window is unconfigured (`min_s = max_s = 0`, ALGORITHMS §5.7) degrades to `none`
   for that scalar alone, and reports `NaN` too. `<s>_lag_s` and `<s>_lag_opt_s` are `0.0`
   under `none` only when the rate is usable; an unusable `freq_hz` makes them `NaN` (§5.7
   again), and it is the *seconds* that go NaN, never the sample counts.
5. `period[s] = kernels.shift_truncate(period[s], L_applied)` — **the scalar only**; `w` is
   never shifted.

Writes, per scalar `s`:

| meta key | type | unit |
|---|---|---|
| `<s>_lag_samples` | int | samples |
| `<s>_lag_s` | float | s (`= L_applied / f`, NaN if `f` is not finite and positive) |
| `<s>_lag_opt_samples` | int | samples |
| `<s>_lag_opt_s` | float | s |
| `<s>_lag_default_used` | bool | - |
| `<s>_lag_cov` | float | m s-1 * unit(s) — the covariance at `L_star`, diagnostic only; `NaN` whenever no scan ran |

`<s>_lag_cov` must **never** be reused as the flux covariance (ALGORITHMS §5.8).

Degenerate: an all-NaN curve gives `L_star = L_nom`, `default_used = True`, `cov = NaN`, and
a `WARNING`.

---

## 11. `flux.py`

Three steps plus pure helpers. This module holds the thermodynamics because the modules list
is fixed; there is no separate `thermo.py`.

### 11.1 Helpers (pure functions, scalars in and out, no period, no cfg)

```python
saturation_vapour_pressure(ta_k)   -> float   # Pa;  ta_k in KELVIN
cp_dry(ta_k)                       -> float   # J kg-1 K-1; internally uses Celsius
cp_vapour(ta_k, rh_pct)            -> float   # J kg-1 K-1; internally uses Celsius
latent_heat(ta_k)                  -> float   # J kg-1; ta_k in KELVIN
air_temperature_from_sonic(ts, e, p) -> float # K
```

Each carries, as a one-line comment, the temperature unit it expects. This is the single
most error-prone corner of the program (ALGORITHMS §8.3).

### 11.2 `thermodynamics(period, cfg) -> period`

Runs **after** `lag.apply_lags` and **before** `detrend.detrend`.

Config: `[gases] *_measure_type`.
Reads: `ts`, `p_air`, `h2o`, `co2`, `ta`.

Computes the per-sample chain of ALGORITHMS §8 (two passes, A then B), accumulating
nan-means without materialising the intermediate series where possible. It also computes the
plain series means `ta_mean`, `ts_mean`, `p_air_mean`.

Writes (all floats, all period means over the finite samples):

| meta key | unit | meta key | unit |
|---|---|---|---|
| `ta_mean` | K | `q_mean` | kg kg-1 |
| `ts_mean` | K | `sigma_mean` | - |
| `p_air_mean` | Pa | `cp_mean` | J kg-1 K-1 |
| `pd_mean` | Pa | `cp_d_mean` | J kg-1 K-1 |
| `e_mean` | Pa | `cp_v_mean` | J kg-1 K-1 |
| `es_mean` | Pa | `lambda_v_mean` | J kg-1 |
| `rh_mean` | % | `va_mean` | m3 mol-1 |
| `chi_v_mean` | mol mol-1 | `vd_mean` | m3 mol-1 |
| `rho_v_mean` | kg m-3 | `n_d_mean` | mol m-3 |
| `rho_d_mean` | kg m-3 | `co2_dry_ppm` | umol mol-1 (dry) |
| `rho_m_mean` | kg m-3 | `h2o_dry_ppt` | mmol mol-1 (dry) |
| `rho_c_mean` | kg m-3 | `n_ta_source_measured` | samples |

`co2_dry_ppm` and `h2o_dry_ppt` are published diagnostics with a **fixed** unit regardless of
`measure_type`: for a mixing ratio they are `1e6 * mean(co2)` and `1e3 * mean(h2o)`; for a
molar density they are `1e6 * mean(n_co2)/n_d_mean` and `1e3 * mean(n_v)/n_d_mean`.

`n_ta_source_measured` counts the samples where the measured `ta` was finite and therefore
used; it is the honest record of which temperature the period's thermodynamics rests on.

### 11.3 `moments(period, cfg) -> period`

Runs **after** `detrend.detrend`.

Computes, with `kernels.cov` / `kernels.variance`:

| meta key | unit |
|---|---|
| `var_u`, `var_v`, `var_w` | m2 s-2 |
| `var_ts` | K2 |
| `cov_u_v`, `cov_u_w`, `cov_v_w` | m2 s-2 |
| `cov_w_ts` | K m s-1 |
| `cov_w_co2` | mol mol-1 m s-1 \| mol m-3 m s-1 |
| `cov_w_h2o` | mol mol-1 m s-1 \| mol m-3 m s-1 |
| `n_cov_w_co2`, `n_cov_w_h2o`, `n_cov_w_ts` | samples (joint finite pairs) |
| `ustar` | m s-1 |

`ustar = (cov_u_w**2 + cov_v_w**2) ** 0.25`. The pair list is fixed; there is no
configurable `select`.

### 11.4 `assemble(period, cfg) -> period`

Reads the meta written by 11.2 and 11.3. Writes:

| meta key | unit | formula (ALGORITHMS §9) |
|---|---|---|
| `f_co2`, `f_h2o` | mol m-3 or - | `1/vd_mean` for a mixing ratio, `1.0` for a molar density |
| `fc_l0` | mol m-2 s-1 | `cov_w_co2 * f_co2` |
| `fh2o_l0` | mol m-2 s-1 | `cov_w_h2o * f_h2o` |
| `e_l0` | kg m-2 s-1 | `fh2o_l0 * MV` |
| `le_l0` | W m-2 | `e_l0 * lambda_v_mean` |
| `h_l0` | W m-2 | `cov_w_ts * rho_m_mean * cp_mean` |
| `h` | W m-2 | `(h_l0 - 0.51*cp_mean*ta_mean*e_l0) / (1 + 0.51*q_mean)` |
| `theta_s` | K | `ts_mean * (P0A / p_air_mean) ** 0.286` |
| `mo_length` | m | `-(theta_s * ustar**3) / (KAPPA * G * cov_w_ts)` |
| `z_l` | - | `(measurement_height - max(Z_MINUS_D_FLOOR, displacement)) / mo_length` |

Config read: `[site] measurement_height`, and `cfg.site.displacement` (already derived).

`h` is not written when any of `h_l0`, `e_l0`, `ta_mean`, `cp_mean`, `q_mean` is non-finite;
the key is then absent and `write.py` emits `na_value`. **`h` is never aliased to `h_l0`.**

Degenerate inputs produce `inf`/`nan` without a guard here (ALGORITHMS §9.4); `qc.py` and
`write.py` handle non-finite values.

---

## 12. `wpl.py`

```python
def correct(period, cfg) -> period
```

Config: `[wpl] enabled`, `[gases] *_measure_type`.

Decision table, evaluated once per period:

| `enabled` | any gas is `molar_density` | action |
|---|---|---|
| `auto` | yes | run the correction |
| `auto` | no | copy: `fc = fc_l0`, `e = e_l0`, `le = le_l0`; `wpl_applied = False` |
| `on` | — | run the correction |
| `off` | no | copy as above |
| `off` | yes | **write nothing** for the owed gas: `fc` absent if CO2 is a molar density; `e`/`le` absent if H2O is; log at `WARNING` |

When it runs (ALGORITHMS §10):

```
wt      = h / (rho_m_mean * cp_mean)
f_chi   = fh2o_l0 * (1 - chi_v_mean)                     # h2o mixing_ratio
        = fh2o_l0 + (rho_v_mean/MV) * wt / ta_mean       # h2o molar_density
w_rho_v = MV * f_chi - rho_v_mean * wt / ta_mean
e       = (1 + MU*sigma_mean) * (w_rho_v + (rho_v_mean/ta_mean) * wt)
le      = lambda_v_mean * e
heat    = (1 + MU*sigma_mean)                            # co2 molar_density
fc      = (fc_l0*MCO2 + MU*(rho_c_mean/rho_d_mean)*w_rho_v + heat*(rho_c_mean/ta_mean)*wt) / MCO2
```

When CO2 is a dry mixing ratio, `fc = fc_l0` (nothing is owed) even though the E/LE branch
still runs — see ALGORITHMS §10.4 for why that is self-consistent.

Writes: `fc` [mol m-2 s-1], `e` [kg m-2 s-1], `le` [W m-2], `wt` [K m s-1],
`wpl_applied` [bool].

**Refusal rule**: if any of `h`, `rho_m_mean`, `cp_mean`, `rho_v_mean`, `rho_d_mean`,
`ta_mean`, `lambda_v_mean`, `chi_v_mean`, `sigma_mean` is non-finite, `correct` writes `nan`
for `fc`/`e`/`le` and logs which input was missing. It never produces a partial correction.
`rho_c_mean` non-finite disables `fc` only; `e` and `le` are still written.

`wpl.py` must **never** use `cov_w_ts` as `wt`, and must **never** recompute a mean density
from a series (the means it reads were taken before detrending).

---

## 13. `qc.py`

```python
def quality(period, cfg) -> period
def steady_state(w, c)                       -> (float, int)   # (percent, flag)
def itc(var_w, var_u, var_ts, ustar, cov_w_ts, z_l, latitude) -> (float, float, float)
```

Config: `[qc] steady_state_pair`, `[qc] itc`, `[site] latitude`.

`steady_state(w, c)` implements ALGORITHMS §11.1 using `kernels.cov` with `start`/`stop` for
the five sub-intervals — one estimator, no second kernel. Returns
`(nan, 2)` when the statistic cannot be formed.

`itc(...)` implements ALGORITHMS §11.2 and returns `(nan, nan, nan)` when any guard fires
(`ustar <= 0`, `latitude == 0`, `z_l == 0` exactly, `z_l` non-finite). `sd` comes from the
variances already in meta, not from a fresh pass.

Writes: `sst_pct` [%], `sst_flag` [0/1/2], `itc_w`, `itc_u`, `itc_t` [fraction],
`tstar` [K].

`qc.quality` is the **last** step, because ITC needs `z_l`, which `flux.assemble` creates.
Running it earlier silently returns the neutral wind models computed on an absent stability.

---

## 14. `pipeline.py`

```python
STEPS = [
    ('despike',        despike.despike),
    ('rotate',         rotate.rotate),
    ('lag',            lag.apply_lags),
    ('thermodynamics', flux.thermodynamics),
    ('detrend',        detrend.detrend),
    ('moments',        flux.moments),
    ('assemble',       flux.assemble),
    ('wpl',            wpl.correct),
    ('qc',             qc.quality),
]

def run_period(period, cfg) -> period          # applies STEPS in order
def run(cfg, writer=None)   -> dict            # summary counters
```

The order is normative and every position in it is load-bearing:

| position | why it must be there |
|---|---|
| `despike` first | the median/MAD threshold must see raw, un-rotated, un-detrended values |
| `rotate` before `lag` | the lag search runs on the **rotated** w |
| `lag` before `thermodynamics` | so every period mean is taken over the samples that actually enter the covariances |
| `thermodynamics` before `detrend` | detrending destroys exactly the means WPL and the flux factor need |
| `detrend` before `moments` | block average changes nothing; linear detrend does |
| `assemble` before `wpl` | WPL consumes the Schotanus-corrected `h` |
| `qc` last | ITC needs `z_l` |

`run(cfg, writer)` iterates `read.periods(cfg)`, calls `run_period`, hands each finished
period to the writer, and returns
`{'periods': int, 'skipped_short': int, 'rows': int, 'warnings': int}`.

A step that raises is caught in `run_period`: the period is logged at `ERROR` with the step
name and the period label, the remaining steps are skipped, and the row is still written with
whatever meta exists (so a failure is visible in the output rather than a hole in it). That
is the only place an exception is swallowed.

---

## 15. `write.py`

```python
COLUMNS   # list of (column_name, meta_key, unit) tuples -- the table of section 18
class Writer:
    def __init__(self, path, cfg): ...
    def __enter__(self) / __exit__(self, *exc): ...
    def write(self, period): ...     # one row
    def close(self): ...
```

* Opens `cfg.files.output_csv` with `newline=''`, writes the header row of column names on
  construction, and writes a sidecar `<stem>_units.csv` with two columns, `name,unit`, once.
* `TIMESTAMP_START` / `TIMESTAMP_END` are `strftime('%Y%m%d%H%M')` of `period_start` /
  `period_end`.
* Any missing key, `None`, `nan` or `inf` is written as `cfg.output.na_value` (default
  `-9999`).
* Floats use `cfg.output.float_format` (default `%.6g`); integers and flags are written as
  plain integers; booleans as `0`/`1`.
* Unit scaling to the published units of §18 happens **here and only here** (`1e6 *` for
  molar fluxes, `1e3 *` for E, `degrees()` for the rotation angles). Meta stays SI.
* No whole-year padding, no reindexing, no gap rows. miniflux writes the periods it computed
  and nothing else.

---

## 16. `cli.py`

```python
def main(argv=None) -> int
```

```
python -m miniflux run   CONFIG.ini [--pure] [--limit N] [--log-level LEVEL]
python -m miniflux check CONFIG.ini            # parse + validate + echo describe(cfg), read nothing
```

* `--pure` forces `runtime.use_numpy = off`.
* `--limit N` stops after N periods (for a smoke run).
* Exit codes: `0` success, `2` `ConfigError`, `3` `ReadError`, `1` any other exception
  (traceback logged).
* Logging goes to stderr via `logging`, logger name `miniflux`, format
  `%(levelname)s %(name)s: %(message)s`.
* On success it logs the summary dict from `pipeline.run` and the output path.

---

## 17. Complete meta key index

| key | type | unit | written by |
|---|---|---|---|
| `period_start`, `period_end` | datetime | - | read |
| `n_in`, `n_dup` | int | samples | read |
| `freq_hz` | float | Hz | read |
| `n_spike_<var>` | int | samples | despike |
| `u_unrot_mean`, `v_unrot_mean`, `w_unrot_mean` | float | m s-1 | rotate |
| `wind_speed` | float | m s-1 | rotate |
| `wind_dir` | float | deg | rotate |
| `theta`, `phi` | float | rad | rotate |
| `<s>_lag_samples`, `<s>_lag_opt_samples` | int | samples | lag |
| `<s>_lag_s`, `<s>_lag_opt_s` | float | s | lag |
| `<s>_lag_default_used` | bool | - | lag |
| `<s>_lag_cov` | float | mixed | lag |
| the 24 `*_mean` / `*_dry_*` keys of §11.2 | float | see §11.2 | thermodynamics |
| `n_ta_source_measured` | int | samples | thermodynamics |
| `<var>_mean` for each detrended variable | float | unit of var | detrend |
| `var_*`, `cov_*`, `n_cov_*`, `ustar` | float/int | see §11.3 | moments |
| `f_co2`, `f_h2o`, `fc_l0`, `fh2o_l0`, `e_l0`, `le_l0`, `h_l0`, `h`, `theta_s`, `mo_length`, `z_l` | float | see §11.4 | assemble |
| `fc`, `e`, `le`, `wt` | float | see §12 | wpl |
| `wpl_applied` | bool | - | wpl |
| `sst_pct`, `sst_flag`, `itc_w`, `itc_u`, `itc_t`, `tstar` | float/int | see §13 | qc |

`detrend` writes `ts_mean`, `co2_mean` and `h2o_mean` for variables `thermodynamics` has
already summarised, over the same samples. **The two are not bit-identical, and the
guarantee is weaker than that**: they agree to the reduction tolerance of §3.8 — 8 ULP in
principle, `1e-12` relative as asserted.

They are two different summations of one array. `detrend` calls `kernels.nanmean`, which on
the numpy path sums **pairwise**; `flux.thermodynamics` accumulates its own left-to-right
running total inside the per-sample loop that builds the whole §8.2 state, because it is
carrying twenty-one sums at once and cannot call a kernel per quantity. On the **pure** path
both are left-to-right and the two do come out bit-identical; on the **numpy** path they
differ in the last bits — order 1e-15 on a mixing ratio and up to ~1.5e-14 on a 36000-sample
temperature series in Kelvin, where the mean is ~295 and the fluctuation is ~0.5.

The overwrite is still harmless, and for a reason that does not depend on the size of that
difference: every consumer of those three means (`moments`, `assemble`, `wpl`, `qc`) runs
**after** `detrend`, so they all read the same one of the two values. Nothing in the program
ever sees the pair. It is documented here so nobody "fixes" it — and so nobody writes a test
asserting bit-equality, which would fail under numpy and pass without it.

---

## 18. Output CSV — exact column list

Order is normative. `unit` is the **published** unit, which differs from the meta unit where
§15 applies a scale.

| # | column | meta key | published unit |
|---|---|---|---|
| 1 | `TIMESTAMP_START` | `period_start` | YYYYMMDDHHMM |
| 2 | `TIMESTAMP_END` | `period_end` | YYYYMMDDHHMM |
| 3 | `N_IN` | `n_in` | samples |
| 4 | `N_DUP` | `n_dup` | samples |
| 5 | `N_SPIKE_U` | `n_spike_u` | samples |
| 6 | `N_SPIKE_V` | `n_spike_v` | samples |
| 7 | `N_SPIKE_W` | `n_spike_w` | samples |
| 8 | `N_SPIKE_TS` | `n_spike_ts` | samples |
| 9 | `N_SPIKE_CO2` | `n_spike_co2` | samples |
| 10 | `N_SPIKE_H2O` | `n_spike_h2o` | samples |
| 11 | `WS` | `wind_speed` | m s-1 |
| 12 | `WD` | `wind_dir` | deg from north |
| 13 | `THETA` | `theta` | deg (scaled here) |
| 14 | `PHI` | `phi` | deg (scaled here) |
| 15 | `USTAR` | `ustar` | m s-1 |
| 16 | `MO_LENGTH` | `mo_length` | m |
| 17 | `ZL` | `z_l` | - |
| 18 | `TA` | `ta_mean` | K |
| 19 | `T_SONIC` | `ts_mean` | K |
| 20 | `PA` | `p_air_mean` | Pa |
| 21 | `RH` | `rh_mean` | % |
| 22 | `CO2_MEAN` | `co2_dry_ppm` | umol mol-1 (dry air) |
| 23 | `H2O_MEAN` | `h2o_dry_ppt` | mmol mol-1 (dry air) |
| 24 | `RHO_A` | `rho_m_mean` | kg m-3 (moist air) |
| 25 | `Q` | `q_mean` | kg kg-1 |
| 26 | `CP` | `cp_mean` | J kg-1 K-1 |
| 27 | `LAMBDA_V` | `lambda_v_mean` | J kg-1 |
| 28 | `VAR_U` | `var_u` | m2 s-2 |
| 29 | `VAR_V` | `var_v` | m2 s-2 |
| 30 | `VAR_W` | `var_w` | m2 s-2 |
| 31 | `VAR_TS` | `var_ts` | K2 |
| 32 | `COV_U_W` | `cov_u_w` | m2 s-2 |
| 33 | `COV_V_W` | `cov_v_w` | m2 s-2 |
| 34 | `COV_W_TS` | `cov_w_ts` | K m s-1 |
| 35 | `CO2_LAG` | `co2_lag_s` | s |
| 36 | `CO2_LAG_OPT` | `co2_lag_opt_s` | s |
| 37 | `CO2_LAG_DEFAULT` | `co2_lag_default_used` | 0/1 |
| 38 | `H2O_LAG` | `h2o_lag_s` | s |
| 39 | `H2O_LAG_OPT` | `h2o_lag_opt_s` | s |
| 40 | `H2O_LAG_DEFAULT` | `h2o_lag_default_used` | 0/1 |
| 41 | `H_L0` | `h_l0` | W m-2 |
| 42 | `H` | `h` | W m-2 |
| 43 | `FC_L0` | `fc_l0` | umol m-2 s-1 (scaled 1e6) |
| 44 | `FC` | `fc` | umol m-2 s-1 (scaled 1e6) |
| 45 | `LE_L0` | `le_l0` | W m-2 |
| 46 | `LE` | `le` | W m-2 |
| 47 | `E_L0` | `e_l0` | g m-2 s-1 (scaled 1e3) |
| 48 | `E` | `e` | g m-2 s-1 (scaled 1e3) |
| 49 | `WPL_APPLIED` | `wpl_applied` | 0/1 |
| 50 | `SST_PCT` | `sst_pct` | % |
| 51 | `SST_FLAG` | `sst_flag` | 0/1/2 |
| 52 | `ITC_W` | `itc_w` | fraction |
| 53 | `ITC_U` | `itc_u` | fraction |
| 54 | `ITC_T` | `itc_t` | fraction (uses \|T*\|, see §20.5) |

The sidecar `<stem>_units.csv` reproduces columns 1 and 4 of this table.

---

## 19. Errors and refusals

```python
class MinifluxError(Exception): ...
class ConfigError(MinifluxError): ...      # cli exit 2
class ReadError(MinifluxError): ...        # cli exit 3
```

Only `config.py` raises `ConfigError`; only `read.py` raises `ReadError`. **No step module
raises.** A step meets degeneracy by writing `nan` and logging.

The refusal principle, stated once: miniflux refuses when continuing would produce a number
that looks right and is not. That is the whole list — a fabricated pressure, an inferred
timestamp format, an unsupported measure type, a closed-path input, an uncorrected flux under
a corrected name. Everything else is reported as NaN and flagged.

---

## 20. Decisions where the extraction specs disagreed

**20.1 FW96 covariance estimator.** The despiking/covariance spec says to use the one
two-pass estimator everywhere; the QC spec says to keep GEddySoft's `xcov`, which drops the
last sample of every slice, for parity. **Decision: one estimator** (`kernels.cov`). The
dropped sample moves the statistic by `O(1/n) ~ 3e-5` relative, far below the 30 %/100 %
thresholds, and a second covariance kernel that exists only to reproduce a transcription
artefact is exactly the kind of thing miniflux exists to not have. Recorded in
ALGORITHMS §11.1 so a comparison against the parent is not a surprise.

**20.2 Linear detrend abscissa.** The parent fits on the compacted (NaN-deleted) index and
evaluates on the full index, stretching the trend by `N/m`. **Decision: fit and evaluate on
the original sample index.** Identical when there are no gaps; correct when there are.

**20.3 Covariance form in the lag scan.** The covariance spec forbids the one-pass raw-sums
form; the lag spec prescribes it (it is what both engines compute and what makes a prefix-sum
scan possible). **Decision: two-pass for every reported moment (`kernels.cov`), one-pass for
the scan (`kernels.cov_at_lag` / `cov_curve`) on series that `lag.py` has pre-centred.**
Pre-centring removes the cancellation the covariance spec warns about and changes no
covariance in exact arithmetic.

**20.4 Which gas measure types.** The WPL and micromet specs recommend molar density only;
the flux-assembly spec recommends a two-branch `if` covering dry mixing ratios (which is what
the bundled example configs actually use). **Decision: support `mixing_ratio` and
`molar_density`; refuse `mole_fraction` (wet) and refuse closed-path input.** Two branches,
one `if`, both example-config shapes readable, and the one branch that is genuinely a partial
correction is left out with a message telling the user what to do instead.

**20.5 ITC temperature scale.** The parent and GEddySoft divide by a signed `T*`, which makes
`ITC_T > 1` on every unstable period. **Decision: use `|T*|`**, per Foken's definition, and
say so in this contract, in ALGORITHMS §11.2 and in the output column note. miniflux's
`ITC_T` is therefore not comparable with the parent's on unstable periods; every other column
is.

**20.6 Block-average detrending.** **Decision: bookkeeping, not arithmetic.** The block mean
of every series is computed and stored; the series is left alone, because the covariance
demeans internally and materialising `x - xbar` cannot change a covariance by one ulp. Linear
detrending *is* materialised, because it changes the numbers.

**20.7 Where the thermodynamics runs.** The parent computes it per sample *before* the whole
high-frequency chain, i.e. on un-despiked, un-aligned data. **Decision: after `lag`, before
`detrend`**, so every period mean is taken over despiked, lag-aligned samples — the same ones
that enter the covariances — and before the means are destroyed.

**20.8 Where units are converted.** The parent despikes in raw analyser units and converts
later. **Decision: convert once, in `read.py`.** The MAD test is equivariant under an affine
change of unit, so the spike flags are unchanged, and every module downstream sees one
vocabulary.

**20.9 `wT` for WPL.** One spec gives the algebraically cancelled form
`wT = (cov_w_ts - 0.51*Ta*covWRhoV/rho_m)/(1 + 0.51 q)`. **Decision: follow the parent's
actual path** — build `h_l0`, apply Schotanus, then `wt = h / (rho_m * cp)`. Same quantity to
within rounding, and it keeps `H` (which we publish) and `wt` (which WPL consumes) provably
consistent with each other.

**20.10 numpy "identical numbers".** Bit-identity across a pairwise sum and a left-to-right
sum is not achievable. **Decision: identical decisions and identical elementwise results,
reductions within 8 ULP, tests asserting 1e-12 relative** (§3.8).

**20.11 Missing air pressure.** The parent substitutes 99767.5 Pa. **Decision: never.**
Either a column or a user-written constant; otherwise the run refuses.

**20.12 Output shape.** The parent's exporter pads to a whole calendar year and truncates to
17520 rows. **Decision: write the periods computed, nothing else.** Padding is a packaging
step for a different pipeline.

---

## 21. Test contract

`tests/` uses `unittest` only, no fixtures beyond small text files generated in
`setUp`/`tempfile`.

| file | what it pins |
|---|---|
| `test_kernels.py` | every kernel's pure semantics, including NaN and short-series edges |
| `test_kernels_parity.py` | §3.8; skipped when numpy is absent |
| `test_config.py` | every refusal in §5.2, and that every default in §6 round-trips |
| `test_read.py` | both timestamp shapes, `closed` left/right boundary placement, duplicate handling, each `ReadError` |
| `test_despike.py` | ALGORITHMS §14.2 |
| `test_rotate.py` | ALGORITHMS §14.1, plus the negative-mean-u quadrant guard |
| `test_lag.py` | ALGORITHMS §14.3, window inclusivity (`2I+1` evaluations), boundary fallback, truncation fill |
| `test_detrend.py` | block mean stored and series untouched; linear detrend on a gapped series matches the closed form on the original index |
| `test_flux.py` | ALGORITHMS §14.4, §14.7; the three temperature-unit conventions of §8.3 |
| `test_wpl.py` | ALGORITHMS §14.5, §14.6; the `off`-but-owed refusal |
| `test_qc.py` | the three flag bounds, non-finite -> flag 2, every ITC guard |
| `test_pipeline.py` | the step order of §14, a synthetic end-to-end period, and that a raising step still produces a row |

A pull request that changes a number in `ALGORITHMS.md` must change the corresponding test in
the same commit.
