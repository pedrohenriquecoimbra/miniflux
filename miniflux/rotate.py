"""Stage 3: coordinate rotation into the mean streamline, and the wind vector.

Double rotation, Wilczak, J. M., Oncley, S. P., Stage, S. A. (2001). Sonic anemometer
tilt correction algorithms. *Boundary-Layer Meteorol.* 99, 127-150.
doi:10.1023/A:1018966204465

The first rotation (yaw, ``theta``) turns the horizontal axes until the mean wind lies
along +u; the second (pitch, ``phi``) tilts them until the mean vertical wind is zero.
What comes out is the streamline frame the flux definitions assume: ``mean(v2)`` and
``mean(w2)`` are zero by construction, so ``cov(w, c)`` is the vertical transport of ``c``
and nothing else. The arithmetic lives in :func:`kernels.rotate_double`; this module
decides whether it runs, and records what it did.

Runs after despiking and before the lag search, because the lag search runs on the
rotated ``w`` -- and before detrending, which would drive all three means to zero and
leave both ``atan2`` calls reading numerical noise.
"""

import logging
import math

from . import kernels

logger = logging.getLogger(__name__)

# The relative residual above which the DEBUG self-check complains, per ALGORITHMS.md
# section 4.2. It bounds the rotation identity on the two means the yaw angle was built
# from, which is exact to rounding.
INVARIANT_TOL = 1e-9    # relative to the wind speed

# The same bound for the mean of the *rotated* series, which is the check CONTRACT
# section 9 asks for. It cannot be 1e-9: v2 exists only where u and v are both finite, so
# it averages over the joint mask while theta came from two separate ones, and despiking
# makes that routine -- ~1e-5 m s-1 on a healthy 36000-sample period, more on a short one.
# 1e-3 sits two orders above that noise and still an order below the smallest rotation
# error worth catching: the pitch angle taken from mean(u) instead of mean(u1)
# (ALGORITHMS 4.3's named trap) leaves 1.2e-2 m s-1 in a 3 m s-1 wind.
ROTATED_MEAN_TOL = 1e-3     # relative to the wind speed


def rotate(period, cfg):
    """Rotate the wind into the mean streamline and record the wind vector.

    Reads and replaces ``period['u']``, ``['v']``, ``['w']`` [m s-1] when
    ``[rotate] method = double``; leaves them alone when it is ``none``. Reads
    ``[site] north_offset`` [deg].

    Writes to ``period['meta']``: ``u_unrot_mean``, ``v_unrot_mean``, ``w_unrot_mean``
    and ``wind_speed`` [m s-1], ``wind_dir`` [deg], ``theta`` and ``phi`` [rad].
    ``write.py`` publishes the two angles in degrees; the meta stays in radians.

    Returns the same period dict. Never raises: an undefined rotation is NaN angles, an
    all-NaN wind and a warning.
    """
    meta = period['meta']

    # The unrotated means come first and unconditionally, even under method = none: WS
    # and WD are defined in the sonic frame, and a double rotation is about to overwrite
    # the series they are taken from. Three independent nan-means, because despiking
    # flags each component separately and their NaN masks routinely differ.
    u_mean = kernels.nanmean(period['u'])
    v_mean = kernels.nanmean(period['v'])
    w_mean = kernels.nanmean(period['w'])
    meta['u_unrot_mean'] = u_mean
    meta['v_unrot_mean'] = v_mean
    meta['w_unrot_mean'] = w_mean
    meta['wind_speed'] = math.hypot(u_mean, v_mean)
    meta['wind_dir'] = _wind_direction(u_mean, v_mean, cfg.site.north_offset)

    if cfg.rotate.method == 'double':
        u2, v2, w2, theta, phi = kernels.rotate_double(
            period['u'], period['v'], period['w'])
        period['u'] = u2
        period['v'] = v2
        period['w'] = w2
        if not (math.isfinite(theta) and math.isfinite(phi)):
            logger.warning('%s: the wind means are not finite, so the rotation angles '
                           'are NaN and the rotated wind is all-NaN',
                           _label(meta))
        elif logger.isEnabledFor(logging.DEBUG):
            _self_check(period, meta, theta, phi)
    else:
        # method = none leaves the series in the sonic frame, and the angles say so: a
        # zero angle is the honest record of a rotation that did not run.
        theta = 0.0
        phi = 0.0

    meta['theta'] = theta
    meta['phi'] = phi
    return period


def _wind_direction(mean_u, mean_v, north_offset):
    """Meteorological wind direction [deg] from the unrotated sonic-frame means [m s-1].

    The direction the wind comes FROM, clockwise from north:

        WD = (180 - degrees(atan2(mean_v, mean_u)) + north_offset) mod 360

    The frame is the sonic's own -- u forward, v to the left, w up, right-handed -- so
    the 180 turns "blowing towards" into "coming from": u = 1, v = 0 gives 180 (a wind
    blowing forward comes from behind), u = 0, v = 1 gives 90, u = -1, v = 0 gives 0.

    ``north_offset`` is the azimuth of the sonic's +u axis and enters here and nowhere
    else: it never touches theta, phi or any flux. ``theta`` is not the wind direction
    either; do not reuse one for the other.
    """
    return (180.0 - math.degrees(math.atan2(mean_v, mean_u)) + north_offset) % 360.0


def _self_check(period, meta, theta, phi):
    """Log what the double rotation zeroed, and complain when it did not.

    Two extra passes over the period, so the caller runs this at DEBUG only. A violation
    is logged at ERROR and nothing more: this step reports, it never refuses.

    Two checks, at two tolerances, because they are two different quantities:

    * the yaw identity ``-sin(theta)*mean(u) + cos(theta)*mean(v) = 0``, exact to
      rounding because theta was built from those same two means -- but blind to
      everything after it, since the angles the kernel returns are self-consistent with
      whatever series it returns alongside them;
    * the means of the rotated ``v`` and ``w``, which is what the rotation is *for*
      (CONTRACT section 9) and the only thing that sees a wrong pitch angle or a wrongly
      applied rotation. They average over the joint NaN mask while the angles came from
      three separate ones, so they are zero only to that mismatch -- hence the far looser
      ROTATED_MEAN_TOL. Exactness under a shared mask is pinned in tests/test_rotate.py,
      where it belongs.
    """
    ubar = meta['u_unrot_mean']
    vbar = meta['v_unrot_mean']
    yaw_residual = -math.sin(theta) * ubar + math.cos(theta) * vbar
    mean_v = kernels.nanmean(period['v'])
    mean_w = kernels.nanmean(period['w'])
    # No cause is claimed for the two sample means here: on a period whose three
    # components carry the same mask they are the arithmetic, and the checks below are
    # what tells the two apart.
    logger.debug('%s: theta %.6g rad, phi %.6g rad; yaw identity %.3g m s-1; sample means '
                 'of the rotated v and w %.3g / %.3g m s-1',
                 _label(meta), theta, phi, yaw_residual, mean_v, mean_w)

    # Scaled by the wind speed: 1e-9 absolute is a different test at 0.1 and at 10 m s-1,
    # and it is the relative error that says the arithmetic went wrong.
    speed = meta['wind_speed']
    scale = speed if math.isfinite(speed) and speed > 1.0 else 1.0
    if math.isfinite(yaw_residual) and abs(yaw_residual) > INVARIANT_TOL * scale:
        logger.error('%s: the double rotation left mean(v) = %.3g m s-1 in the very '
                     'means its yaw angle was built from, above the %g relative '
                     'tolerance; this is the rotation arithmetic, not the data',
                     _label(meta), yaw_residual, INVARIANT_TOL)
    for name, residual in (('v', mean_v), ('w', mean_w)):
        if math.isfinite(residual) and abs(residual) > ROTATED_MEAN_TOL * scale:
            logger.error('%s: the double rotation left mean(%s) = %.3g m s-1 in the '
                         'rotated wind, above the %g relative tolerance -- far above what '
                         'mismatched NaN masks explain. The streamline frame every flux '
                         'below assumes does not hold for this period.',
                         _label(meta), name, residual, ROTATED_MEAN_TOL)


def _label(meta):
    """The period's label for a log line, or a placeholder before read.py has set one."""
    return meta.get('period_end', 'period')
