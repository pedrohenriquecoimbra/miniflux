"""Physical constants, as module-level floats. No units library, no functions.

Several of these are deliberately *not* the value you would derive from the others.
Where that is so, the derived value is given in the comment so the difference is
visible rather than accidental. See ALGORITHMS.md section 1.

Fit coefficients that belong to one formula (cp_d, cp_v, lambda_v, es, the 0.32 sonic
conversion, the 0.51 virtual-temperature factor, the 0.286 Poisson exponent) are
written inline in the module that uses them: they may not be mixed with another
parameterisation, so they are not shared constants.
"""

R = 8.314462618         # J mol-1 K-1  universal gas constant, CODATA 2018 (exact since the 2019 SI)

# Rd and Rv are EddyPro's rounded values, kept so miniflux's thermodynamics matches the
# engine it was extracted from. Deriving them instead moves every density by ~0.02 %.
RD = 287.04             # J kg-1 K-1   dry air    -- NOT R/MD = 287.0025 (+0.0131 %)
RV = 461.5              # J kg-1 K-1   vapour     -- NOT R/MV = 461.4019 (+0.0213 %)

MD = 0.02897            # kg mol-1     molar mass, dry air
MV = 0.01802            # kg mol-1     molar mass, water vapour
MCO2 = 0.04401          # kg mol-1     molar mass, CO2
MU = MD / MV            # 1.6076581576026636, computed -- never written as a literal

# CPD is the intercept of the cp_d(T) fit and is paired with that fit's (Tc+23.12)^2/3364
# term. Substituting the textbook 1004.67 leaves cp_d low and H with it.
CPD = 1005.0            # J kg-1 K-1   dry-air heat capacity at constant pressure

G = 9.81                # m s-2        gravitational acceleration (not 9.80665)
KAPPA = 0.4             # -            von Karman constant (EddyPro's raw path uses 0.41)
P0A = 1.0e5             # Pa           reference pressure for potential temperature
MAD_SCALE = 0.6745      # -            0.75 quantile of the standard normal; the code divides by it
T0 = 273.15             # K            degC -> K offset

CANOPY_DISPLACEMENT_RATIO = 2.0 / 3.0   # -   d = 2h/3 (Monteith & Unsworth 2013, Sect. 9.2)
Z_MINUS_D_FLOOR = 1.0e-4                # m   keeps (z-d) out of a division by zero
