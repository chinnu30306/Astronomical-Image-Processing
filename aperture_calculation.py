import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


def aperture_radius(x_centre, y_centre, image, window=20, fraction=0.5):
    x0 = max(0, x_centre - window)
    x1 = min(image.shape[1], x_centre + window + 1)

    row = image[y_centre, x0:x1]

    # If the whole window is NaN, bail out
    if np.all(np.isnan(row)):
        return None

    peak_idx = int(np.nanargmax(row))
    peak_val      = row[peak_idx]

    local_median   = np.nanmedian(row)
    level = local_median + (peak_val - local_median) * fraction
    
    left = None
    for i in range(peak_idx, 0, -1):
        if row[i-1] <= level:
            y1, y2 = row[i-1], row[i]
            frac = (level - y1) / (y2 - y1) if y2 != y1 else 0.5
            left = (i - 1) + frac
            break
    # Right
    right = None
    for i in range(peak_idx, len(row) - 1):
        if row[i+1] <= level:
            y1, y2 = row[i], row[i+1]
            frac = (level - y1) / (y2 - y1) if y2 != y1 else 0.5
            right = i + frac
            break

    if left is None:
        radius = right - peak_idx
    elif right is None:
        radius = peak_idx - left
    else:
        radius = (right - left) / 2.0
    
    return radius


def aperture_radius_noise(image, x_c, y_c, background_mean, background_sigma,
                          k=5, profile_max=25):
    """
    Aperture radius = first radius where the mean radial profile
    drops to within k * sigma of the background.

    If no bin crosses the threshold, falls back to the radius where
    the profile is closest to it (data-driven fallback, never returns None).

    Parameters
    ----------
    image           : 2D array
    x_c, y_c        : int — source centre
    background_mean : float
    background_sigma: float — noise level (from your Gaussian fit)
    k               : float — noise threshold multiplier
    profile_max     : int  — max radius to scan

    Returns
    -------
    r_ap : float — always a valid float
    """
    # Local window large enough to contain the profile
    win = profile_max + 5
    y0 = max(0, y_c - win)
    y1 = min(image.shape[0], y_c + win + 1)
    x0 = max(0, x_c - win)
    x1 = min(image.shape[1], x_c + win + 1)

    patch = image[y0:y1, x0:x1]
    yy_loc, xx_loc = np.ogrid[y0:y1, x0:x1]
    r_local = np.sqrt((xx_loc - x_c)**2 + (yy_loc - y_c)**2)

    # Build the radial profile
    bins = np.arange(0, profile_max + 1, 1.0)
    digitised = np.digitize(r_local.ravel(), bins)
    flat_patch = patch.ravel()

    radial_mean = np.full(len(bins) - 1, np.nan)
    for i in range(1, len(bins)):
        sel = (digitised == i)
        if sel.sum() > 0:
            vals = flat_patch[sel]
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                radial_mean[i - 1] = vals.mean()

    # Threshold: background + k * sigma
    threshold_level = background_mean + k * background_sigma

    # ---- First pass: look for the first crossing ----
    for i in range(1, len(radial_mean)):
        if np.isnan(radial_mean[i]):
            continue
        if radial_mean[i] <= threshold_level:
            return float(bins[i])

    # ---- Fallback: closest-to-threshold radius ----
    valid = ~np.isnan(radial_mean)
    if not np.any(valid):
        # Whole profile is NaN — pathological
        return float(profile_max)

    diffs        = np.abs(radial_mean[valid] - threshold_level)
    idx_in_valid = np.argmin(diffs)
    idx_full     = np.where(valid)[0][idx_in_valid]
    return float(bins[idx_full + 1])
