"""Write the per-period flux table (CONTRACT sections 15 and 18).

One row per averaging period, in the column order of CONTRACT section 18, plus a sidecar
``<stem>_units.csv`` naming the published unit of every column. Two rules live here and
nowhere else:

* **the meta dict stays SI**; the scaling to the published units of section 18 (1e6 for the
  molar fluxes, 1e3 for E, radians to degrees for the rotation angles) is applied in this
  module at the moment of formatting, so no step ever has to remember which unit its
  neighbour publishes;
* **a value that is missing, ``None``, ``nan`` or ``inf`` is written as the configured
  ``na_value``**, because a flux that could not be computed must be visibly absent rather
  than quietly plausible.

No padding to a calendar year, no reindexing, no gap rows: miniflux writes the periods it
computed and nothing else (CONTRACT section 20.12).
"""

import csv
import logging
import math
import os

logger = logging.getLogger(__name__)

#: ``strftime`` pattern of TIMESTAMP_START / TIMESTAMP_END.
TIMESTAMP_FORMAT = '%Y%m%d%H%M'

#: The output table: ``(column_name, meta_key, published_unit)``, in normative order.
COLUMNS = [
    ('TIMESTAMP_START', 'period_start', 'YYYYMMDDHHMM'),
    ('TIMESTAMP_END', 'period_end', 'YYYYMMDDHHMM'),
    ('N_IN', 'n_in', 'samples'),
    ('N_DUP', 'n_dup', 'samples'),
    ('N_SPIKE_U', 'n_spike_u', 'samples'),
    ('N_SPIKE_V', 'n_spike_v', 'samples'),
    ('N_SPIKE_W', 'n_spike_w', 'samples'),
    ('N_SPIKE_TS', 'n_spike_ts', 'samples'),
    ('N_SPIKE_CO2', 'n_spike_co2', 'samples'),
    ('N_SPIKE_H2O', 'n_spike_h2o', 'samples'),
    ('WS', 'wind_speed', 'm s-1'),
    ('WD', 'wind_dir', 'deg from north'),
    ('THETA', 'theta', 'deg'),
    ('PHI', 'phi', 'deg'),
    ('USTAR', 'ustar', 'm s-1'),
    ('MO_LENGTH', 'mo_length', 'm'),
    ('ZL', 'z_l', '-'),
    ('TA', 'ta_mean', 'K'),
    ('T_SONIC', 'ts_mean', 'K'),
    ('PA', 'p_air_mean', 'Pa'),
    ('RH', 'rh_mean', '%'),
    ('CO2_MEAN', 'co2_dry_ppm', 'umol mol-1 (dry air)'),
    ('H2O_MEAN', 'h2o_dry_ppt', 'mmol mol-1 (dry air)'),
    ('RHO_A', 'rho_m_mean', 'kg m-3 (moist air)'),
    ('Q', 'q_mean', 'kg kg-1'),
    ('CP', 'cp_mean', 'J kg-1 K-1'),
    ('LAMBDA_V', 'lambda_v_mean', 'J kg-1'),
    ('VAR_U', 'var_u', 'm2 s-2'),
    ('VAR_V', 'var_v', 'm2 s-2'),
    ('VAR_W', 'var_w', 'm2 s-2'),
    ('VAR_TS', 'var_ts', 'K2'),
    ('COV_U_W', 'cov_u_w', 'm2 s-2'),
    ('COV_V_W', 'cov_v_w', 'm2 s-2'),
    ('COV_W_TS', 'cov_w_ts', 'K m s-1'),
    ('CO2_LAG', 'co2_lag_s', 's'),
    ('CO2_LAG_OPT', 'co2_lag_opt_s', 's'),
    ('CO2_LAG_DEFAULT', 'co2_lag_default_used', '0/1'),
    ('H2O_LAG', 'h2o_lag_s', 's'),
    ('H2O_LAG_OPT', 'h2o_lag_opt_s', 's'),
    ('H2O_LAG_DEFAULT', 'h2o_lag_default_used', '0/1'),
    ('H_L0', 'h_l0', 'W m-2'),
    ('H', 'h', 'W m-2'),
    ('FC_L0', 'fc_l0', 'umol m-2 s-1'),
    ('FC', 'fc', 'umol m-2 s-1'),
    ('LE_L0', 'le_l0', 'W m-2'),
    ('LE', 'le', 'W m-2'),
    ('E_L0', 'e_l0', 'g m-2 s-1'),
    ('E', 'e', 'g m-2 s-1'),
    ('WPL_APPLIED', 'wpl_applied', '0/1'),
    ('SST_PCT', 'sst_pct', '%'),
    ('SST_FLAG', 'sst_flag', '0/1/2'),
    ('ITC_W', 'itc_w', 'fraction'),
    ('ITC_U', 'itc_u', 'fraction'),
    # ITC_T is a deviation from the Thomas & Foken model taken against |T*|, not the signed
    # T* the parent engines use, so it is not comparable with theirs on unstable periods
    # (CONTRACT section 20.5).
    ('ITC_T', 'itc_t', 'fraction'),
]

# The two datetime columns. Everything else in the table is a number.
_TIMESTAMPS = frozenset(('period_start', 'period_end'))

# Counts and flags: written as plain integers, never through float_format, so a sample count
# is never rendered as 1.2e+04.
_INTEGERS = frozenset(('n_in', 'n_dup', 'n_spike_u', 'n_spike_v', 'n_spike_w', 'n_spike_ts',
                       'n_spike_co2', 'n_spike_h2o', 'sst_flag'))

# Booleans, published as 0/1.
_FLAGS = frozenset(('co2_lag_default_used', 'h2o_lag_default_used', 'wpl_applied'))

# meta unit -> published unit of section 18. The angles are stored in radians because every
# formula that consumes them is trigonometric; degrees are for the reader of the table.
_SCALE = {
    'theta': 180.0 / math.pi,
    'phi': 180.0 / math.pi,
    'fc_l0': 1e6,          # mol m-2 s-1 -> umol m-2 s-1
    'fc': 1e6,
    'e_l0': 1e3,           # kg m-2 s-1 -> g m-2 s-1
    'e': 1e3,
}


def units_path(output_csv):
    """The sidecar path for an output table: ``run.csv`` -> ``run_units.csv``.

    output_csv: str. Returns str.
    """
    stem, _extension = os.path.splitext(output_csv)
    return stem + '_units.csv'


class Writer:
    """Append one CSV row per finished period, in the fixed column order of section 18.

    Opening the writer truncates the output table, writes its header, and writes the units
    sidecar once. Use it as a context manager so the file is closed on the way out of a run,
    successful or not.
    """

    def __init__(self, path, cfg):
        """path: str, the output table. cfg: the Config of CONTRACT section 5."""
        self.path = path
        self.rows = 0
        self._na = cfg.output.na_value
        self._float_format = cfg.output.float_format
        # newline='' is the csv module's requirement: it writes its own line terminator, and
        # without this the file would get CR CR LF on Windows.
        self._handle = open(path, 'w', newline='', encoding='utf-8')
        self._writer = csv.writer(self._handle)
        self._writer.writerow([name for name, _key, _unit in COLUMNS])
        self._write_units()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False        # an exception on the way out is the caller's to handle

    def write(self, period):
        """Write one row for one finished period. period: the dict of CONTRACT section 1."""
        meta = period.get('meta', {})
        row = [_cell(meta.get(key), key, self._na, self._float_format)
               for _name, key, _unit in COLUMNS]
        self._writer.writerow(row)
        self.rows += 1

    def close(self):
        """Close the output table. Calling it twice is harmless."""
        if self._handle is not None and not self._handle.closed:
            self._handle.close()
            logger.info('wrote %d row(s) to %s', self.rows, self.path)

    def _write_units(self):
        """Write the sidecar: one `name,unit` line per column, published units."""
        path = units_path(self.path)
        with open(path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['name', 'unit'])
            for name, _key, unit in COLUMNS:
                writer.writerow([name, unit])
        logger.debug('wrote the units sidecar %s', path)


def _cell(value, key, na, float_format):
    """Render one meta value as its published column. Returns str.

    A value that is absent, None, nan or inf becomes `na`: the table says "not computed"
    rather than showing a number that survived a degenerate period.
    """
    if key in _TIMESTAMPS:
        try:
            return value.strftime(TIMESTAMP_FORMAT)
        except AttributeError:
            return na
    number = _finite(value)
    if number is None:
        return na
    if key in _FLAGS:
        return '1' if number else '0'
    if key in _INTEGERS:
        return str(int(number))
    return float_format % (number * _SCALE.get(key, 1.0))


def _finite(value):
    """The value as a float, or None when it is missing or not finite. Returns float|None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
