# ============================================================
# ASTRONOMICAL IMAGE PROCESSING — B2 Lab 3
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from background_noise import compute_background, gaussian
from aperture_calculation import aperture_radius, aperture_radius_noise
from scipy import ndimage
import csv
from scipy.optimize import curve_fit

# ============================================================
# 1. LOAD AND SUBSET
# ============================================================

hdulist = fits.open("Astro/Fits_Data/mosaic_clean.fits")
image_full = hdulist[0].data.astype(np.float32)
header = hdulist[0].header

pixel_scale = 0.258 # given
valid_pixels = np.isfinite(image_full)
number_of_valid_pixels = np.sum(valid_pixels)
pixel_area_deg2 = (pixel_scale / 3600)**2
survey_area_deg2 = number_of_valid_pixels * pixel_area_deg2
print(survey_area_deg2)

x_min, x_max = 1722, 2210
y_min, y_max = 1374, 1723

image = image_full[y_min:y_max, x_min:x_max]

print(f"Subset shape: {image.shape}")

plt.figure(figsize=(8, 8))
plt.imshow(image, origin="lower", cmap="gray", vmin=np.nanpercentile(image, 5), vmax=np.nanpercentile(image, 97))
plt.colorbar(label="Pixel counts")
plt.title(f"Subset ({x_min},{y_min}) → ({x_max},{y_max})")
plt.show()

# ============================================================
# 2. BACKGROUND FIT
# ============================================================

background_mean, background_sigma = compute_background(image, bins=1000, range=(3350, 3500))
detection_threshold = background_mean + 2.5 * background_sigma

print(f"Background mean:  {background_mean:.2f}")
print(f"Background sigma: {background_sigma:.2f}")
print(f"Threshold (2.5σ):   {detection_threshold:.2f}")

# ============================================================
# 3. CONNECTED-COMPONENT PRE-LABELING
# ============================================================

binary_image = (image > detection_threshold)
labeled_image, num_initial = ndimage.label(binary_image)

print(f"Initial connected regions above threshold: {num_initial}")

# Precompute region sizes for all labels
all_sizes = ndimage.sum(binary_image, labeled_image, index=np.arange(1, num_initial + 1))

# Minimum pixels for a valid source
min_pixels = 5

# ============================================================
# 4. SETUP
# ============================================================

source_mask = np.zeros(image.shape, dtype=bool)

default_r_ap = 10
default_inner_radius = default_r_ap + 5
default_outer_radius = default_inner_radius + 5

catalogue = []
mag_zpt  = header['MAGZPT']
mag_err_zpt = header['MAGZRR']

# ============================================================
# 5. ITERATIVE DETECTION + APERTURE PHOTOMETRY
# ============================================================

working_image = image.copy()
working_image[source_mask] = np.nan

iteration = 0
n_rejected = 0
while True:
    iteration += 1

    # ---- 5a. Brightest remaining unmasked pixel ----
    max_flat_index = np.nanargmax(working_image)
    y_source, x_source = np.unravel_index(max_flat_index, working_image.shape)
    source_pixel_value = float(image[y_source, x_source])

    # ---- 5b. Threshold check ----
    if source_pixel_value < detection_threshold:
        print(f"\nBrightest remaining pixel = {source_pixel_value:.2f} "
              f"< threshold {detection_threshold:.2f} — stopping.")
        break

    # ---- 5c. Size filter ----
    # Look up which connected region this pixel belongs to, and check whether it has enough pixels.
    region_id = labeled_image[y_source, x_source]

    if region_id == 0:
        # Pixel is above threshold but not in any labelled region
        working_image[y_source, x_source] = np.nan
        source_mask[y_source, x_source] = True
        continue

    region_size = all_sizes[region_id - 1]

    if region_size < min_pixels:
        # Too small — noise spike. Mask it out anyway and skip.
        n_rejected += 1
        small_region = (labeled_image == region_id)
        working_image[small_region] = np.nan
        source_mask[small_region] = True
        continue

    # ---- 5d. Measure aperture radius ----
    def _valid_radius(r):
        return r is not None and np.isfinite(r) and r > 0

    # Default window size
    current_window = 50
    is_bright_star = False

    # Check if this is the bright star in the middle
    if abs(x_source - 1435) <= 5:
        is_bright_star = True
        current_window = 500
        print(f"Bright star detected at x={x_source}, using hardcoded radii.")
        
        # --- HARDCODE THE RADII ---
        r_ap = 160
        inner_radius = 370
        outer_radius = 450
    else:
        # Normal aperture logic for all other sources
        r_ap = aperture_radius(x_source, y_source, image, window=current_window, fraction=0.001) * 1.5
        inner_radius = aperture_radius(x_source, y_source, image, window=current_window, fraction=0.0001) * 1.5
        outer_radius = inner_radius + 5

    if not _valid_radius(r_ap):
        r_ap = default_r_ap
    if not _valid_radius(inner_radius):
        inner_radius = default_inner_radius
    if not _valid_radius(outer_radius):
        outer_radius = default_outer_radius

    # ---- 5e. Build local window ----
    window = int(outer_radius) + 5
    y0 = max(0, y_source - window)
    y1 = min(image.shape[0], y_source + window + 1)
    x0 = max(0, x_source - window)
    x1 = min(image.shape[1], x_source + window + 1)

    patch = image[y0:y1, x0:x1]
    patch_mask = source_mask[y0:y1, x0:x1]
    yy_loc, xx_loc = np.ogrid[y0:y1, x0:x1]
    r_local = np.sqrt((xx_loc - x_source)**2 + (yy_loc - y_source)**2)

    # ---- 5f. Source aperture ----
    aperture   = (r_local <= r_ap) & ~patch_mask
    n_aperture = np.sum(aperture)

    if n_aperture == 0:
        working_image[y_source, x_source] = np.nan
        source_mask[y_source, x_source] = True
        continue

    aperture_counts = np.sum(patch[aperture])

    # ---- 5g. Background annulus ----
    annulus = ((r_local >= inner_radius) &
               (r_local <= outer_radius) &
               ~patch_mask)
    ann_pixels = patch[annulus]
    ann_pixels = ann_pixels[np.isfinite(ann_pixels)]

    if len(ann_pixels) == 0:
        local_background = background_mean
    else:
        local_background = np.median(ann_pixels)

    # ------------------------------------------------------------
    # DEBUG: histogram of annulus pixels
    # ------------------------------------------------------------
    if len(ann_pixels) > 0 and x_source == 210:
        median_bg = np.median(ann_pixels)
        mad_bg    = np.median(np.abs(ann_pixels - median_bg))
        sigma_bg  = 1.4826 * mad_bg

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(ann_pixels, bins=40, color='steelblue',
                alpha=0.8, edgecolor='black', linewidth=0.3)

        # Median line
        ax.axvline(median_bg, color='red', linestyle='--',
                   linewidth=2, label=f'Median = {median_bg:.1f}')

        # ±1σ band
        ax.axvline(median_bg + sigma_bg, color='orange',
                   linestyle=':', linewidth=2,
                   label=f'+1σ = {median_bg + sigma_bg:.1f}')
        ax.axvline(median_bg - sigma_bg, color='orange',
                   linestyle=':', linewidth=2,
                   label=f'-1σ = {median_bg - sigma_bg:.1f}')

        # Global background reference
        ax.axvline(background_mean, color='green',
                   linestyle='-.', linewidth=1.5,
                   label=f'Global bg = {background_mean:.1f}')

        ax.set_xlabel('Pixel value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Background annulus histogram — '
                     f'({x_source}, {y_source}), '
                     f'r ∈ [{inner_radius:.1f}, {outer_radius:.1f}]')
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.show()

        print(f"  Annulus stats: n={len(ann_pixels)}, "
              f"median={median_bg:.1f}, "
              f"MAD-sigma={sigma_bg:.2f}, "
              f"min={ann_pixels.min():.1f}, "
              f"max={ann_pixels.max():.1f}")

    # ---- 5g. Background annulus (with outlier masking) ----
    annulus = ((r_local >= inner_radius) &
               (r_local <= outer_radius) &
               ~patch_mask)
    
    # Extract the pixels in the annulus
    ann_pixels = patch[annulus]
    ann_pixels = ann_pixels[np.isfinite(ann_pixels)]

    if len(ann_pixels) == 0:
        local_background = background_mean
    else:
        # 1. First pass: get a robust initial median
        med = np.median(ann_pixels)
        
        # 2. Calculate the noise using MAD
        mad = np.median(np.abs(ann_pixels - med))
        sigma = 1.4826 * mad
        
        # 3. Mask out anything that is > 3-sigma above the median
        # (We only clip positive outliers because faint galaxies are positive)
        clean_pixels = ann_pixels[ann_pixels < (med + 3 * sigma)]
        
        # 4. Calculate the final median from the clean pixels
        if len(clean_pixels) > 0:
            local_background = np.median(clean_pixels)
        else:
            local_background = med # Fallback

    # ------------------------------------------------------------
    # DEBUG: histogram of annulus pixels
    # ------------------------------------------------------------
    if len(clean_pixels) > 0 and x_source == 210:
        median_bg = np.median(clean_pixels)
        mad_bg    = np.median(np.abs(clean_pixels - median_bg))
        sigma_bg  = 1.4826 * mad_bg

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(clean_pixels, bins=40, color='steelblue',
                alpha=0.8, edgecolor='black', linewidth=0.3)

        # Median line
        ax.axvline(median_bg, color='red', linestyle='--',
                   linewidth=2, label=f'Median = {median_bg:.1f}')

        # ±1σ band
        ax.axvline(median_bg + sigma_bg, color='orange',
                   linestyle=':', linewidth=2,
                   label=f'+1σ = {median_bg + sigma_bg:.1f}')
        ax.axvline(median_bg - sigma_bg, color='orange',
                   linestyle=':', linewidth=2,
                   label=f'-1σ = {median_bg - sigma_bg:.1f}')

        # Global background reference
        ax.axvline(background_mean, color='green',
                   linestyle='-.', linewidth=1.5,
                   label=f'Global bg = {background_mean:.1f}')

        ax.set_xlabel('Pixel value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'"Cleaned" pixels annulus histogram — '
                     f'({x_source}, {y_source}), '
                     f'r ∈ [{inner_radius:.1f}, {outer_radius:.1f}]')
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.show()

        print(f"  Annulus stats: n={len(clean_pixels)}, "
              f"median={median_bg:.1f}, "
              f"MAD-sigma={sigma_bg:.2f}, "
              f"min={clean_pixels.min():.1f}, "
              f"max={clean_pixels.max():.1f}")

    # ---- 5h. Background subtraction ----
    background_counts = local_background * n_aperture
    source_counts = aperture_counts - background_counts

    if source_counts <= 0 or np.isnan(source_counts):
        working_image[y_source, x_source] = np.nan
        source_mask[y_source, x_source] = True
        continue

    mag = mag_zpt - 2.5 * np.log10(source_counts)

    # ---- 5i. Catalogue ----
    catalogue.append({
        "x":             int(x_source),
        "y":             int(y_source),
        "peak":          source_pixel_value,
        "region_size":   int(region_size),
        "r_ap":          float(r_ap),
        "inner_radius":  float(inner_radius),
        "outer_radius":  float(outer_radius),
        "background":    float(local_background),
        "raw_counts":    float(aperture_counts),
        "source_counts": float(source_counts),
        "mag":           float(mag),
    })

    # ---- 5j. Mask the source ----
    mask_window = int(r_ap) + 5
    y0m = max(0, y_source - mask_window)
    y1m = min(image.shape[0], y_source + mask_window + 1)
    x0m = max(0, x_source - mask_window)
    x1m = min(image.shape[1], x_source + mask_window + 1)

    yy_m, xx_m = np.ogrid[y0m:y1m, x0m:x1m]
    r_m = np.sqrt((xx_m - x_source)**2 + (yy_m - y_source)**2)
    aperture_local = (r_m <= r_ap)

    source_mask[y0m:y1m, x0m:x1m] |= aperture_local
    working_image[y0m:y1m, x0m:x1m][aperture_local] = np.nan

    # ---- 5k. Progress ----
    if len(catalogue) % 100 == 0:
        print(f"  ...{len(catalogue):5d} sources | "
              f"iter {iteration:6d} | "
              f"rejected {n_rejected} | "
        )

print()
print(f"Detection complete. {len(catalogue)} sources in catalogue.")
print(f"Rejected {n_rejected} small regions (size < {min_pixels} px).")

# ============================================================
# 5b. POST-FILTER: REMOVE SOURCES AROUND A BRIGHTER APERTURE
# ============================================================

if len(catalogue) > 0:
    # 1. Sort by peak brightness (brightest first)
    catalogue_sorted = sorted(catalogue, key=lambda s: -s['peak'])

    kept_sources = []
    discarded_count = 0

    # 2. Iterate through each source, brightest first
    for s in catalogue_sorted:
        is_duplicate = False

        # 3. Check against all already-kept (brighter) sources
        for keeper in kept_sources:
            dx = s['x'] - keeper['x']
            dy = s['y'] - keeper['y']
            distance = np.hypot(dx, dy)

            # If this source is within the aperture of a brighter one, discard it
            if distance < keeper['r_ap'] + 2:
                is_duplicate = True
                break

        if is_duplicate:
            discarded_count += 1
        else:
            kept_sources.append(s)

    # 4. Update the catalogue
    catalogue = kept_sources
    
    print()
    print(f"Post-filter: discarded {discarded_count} sources "
          f"that fell inside a brighter source's aperture.")
    print(f"Remaining sources: {len(catalogue)}")

# ============================================================
# 6. CATALOGUE SUMMARY
# ============================================================

print()
print("======================================")
print("         CATALOGUE SUMMARY")
print("======================================")
print(f"Background mean:          {background_mean:.2f}")
print(f"Background sigma:         {background_sigma:.2f}")
print(f"Detection threshold:      {detection_threshold:.2f}")
print(f"Total sources detected:   {len(catalogue)}")
print(f"Rejected small regions:   {n_rejected}")

if len(catalogue) > 0:
    mags  = np.array([s['mag'] for s in catalogue])
    raps  = np.array([s['r_ap'] for s in catalogue])
    valid = np.isfinite(mags)
    if np.any(valid):
        print(f"Magnitude range:          {mags[valid].min():.2f} "
              f"to {mags[valid].max():.2f}")
        print(f"Aperture radius range:    {raps.min():.1f} "
              f"to {raps.max():.1f} pixels")
        print(f"Median aperture radius:   {np.median(raps):.1f} pixels")
print("======================================")

# ============================================================
# 7. SAVE CATALOGUE
# ============================================================

with open("catalogue.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "x", "y", "peak", "region_size",
        "r_ap", "inner_radius", "outer_radius",
        "background", "raw_counts", "source_counts", "mag"
    ])
    writer.writeheader()
    writer.writerows(catalogue)

print(f"Saved {len(catalogue)} sources to catalogue.csv")

# ============================================================
# 8. VISUAL CHECK
# ============================================================

fig, ax = plt.subplots(figsize=(8, 10))

ax.imshow(
    image,
    origin='lower',
    cmap='gray',
    vmin=np.nanpercentile(image, 5),
    vmax=np.nanpercentile(image, 97),
    extent=[0, image.shape[1], 0, image.shape[0]],
    aspect='equal'
)

xs = [s['x'] for s in catalogue]
ys = [s['y'] for s in catalogue]

ax.scatter(
    xs, ys,
    marker='+', s=60,
    color='red', linewidths=0.8,
    label=f'{len(catalogue)} detections'
)

for s in catalogue:
    ax.add_patch(plt.Circle(
        (s['x'], s['y']), s['r_ap'],
        fill=False, color='yellow',
        linewidth=1.0, alpha=0.7
    ))
    ax.add_patch(plt.Circle(
        (s['x'], s['y']), s['inner_radius'],
        fill=False, color='blue',
        linewidth=1.0, alpha=0.7
    ))
    ax.add_patch(plt.Circle(
        (s['x'], s['y']), s['outer_radius'],
        fill=False, color='green',
        linewidth=1.0, alpha=0.7
    ))

ax.set_xlabel('X pixel')
ax.set_ylabel('Y pixel')
ax.set_title('Galaxy detection with global background threshold')
ax.legend(loc='upper right')
plt.show()

# ============================================================
# 8. VISUAL CHECK - MAG 16
# ============================================================

# fig, ax = plt.subplots(figsize=(8, 10))

# ax.imshow(
#     image,
#     origin='lower',
#     cmap='gray',
#     vmin=np.nanpercentile(image, 5),
#     vmax=np.nanpercentile(image, 97),
#     extent=[0, image.shape[1], 0, image.shape[0]],
#     aspect='equal'
# )

# # Split detections by magnitude
# bright = [s for s in catalogue if np.isfinite(s['mag']) and s['mag'] < 16]
# faint  = [s for s in catalogue if not (np.isfinite(s['mag']) and s['mag'] < 16)]

# # Faint sources — small red +
# xs_faint = [s['x'] for s in faint]
# ys_faint = [s['y'] for s in faint]
# ax.scatter(
#     xs_faint, ys_faint,
#     marker='+', s=40,
#     color='red', linewidths=0.6,
#     label=f'{len(faint)} faint (mag ≥ 16)'
# )

# # Bright sources — large yellow star
# xs_bright = [s['x'] for s in bright]
# ys_bright = [s['y'] for s in bright]
# if len(bright) > 0:
#     ax.scatter(
#         xs_bright, ys_bright,
#         marker='*', s=200,
#         color='gold', edgecolors='black', linewidths=0.5,
#         label=f'{len(bright)} bright (mag < 16)'
#     )

# # Aperture circles — still drawn for every source, but with different styles
# for s in catalogue:
#     ax.add_patch(plt.Circle(
#         (s['x'], s['y']), s['r_ap'],
#         fill=False, color='yellow', linewidth=0.6, alpha=0.5
#     ))
#     ax.add_patch(plt.Circle(
#         (s['x'], s['y']), s['inner_radius'],
#         fill=False, color='blue', linewidth=0.6, alpha=0.5
#     ))
#     ax.add_patch(plt.Circle(
#         (s['x'], s['y']), s['outer_radius'],
#         fill=False, color='green', linewidth=0.6, alpha=0.5
#     ))

# ax.set_xlabel('X pixel (subset)')
# ax.set_ylabel('Y pixel (subset)')
# ax.set_title(f'Detections: {len(bright)} bright (mag < 16) + '
#              f'{len(faint)} faint')
# ax.legend(loc='upper right', fontsize=9)
# plt.tight_layout()
# plt.show()

# ============================================================
# 9. r_ap VS MAGNITUDE
# ============================================================

if len(catalogue) > 0:
    mags = np.array([s['mag'] for s in catalogue])
    raps = np.array([s['r_ap'] for s in catalogue])
    valid = np.isfinite(mags) & np.isfinite(raps)

    plt.figure(figsize=(8, 5))
    plt.scatter(mags[valid], raps[valid], s=5, alpha=0.4)
    plt.xlabel('Magnitude')
    plt.ylabel('Measured aperture radius (pixels)')
    plt.title('Aperture radius vs magnitude')
    plt.gca().invert_xaxis()
    plt.show()