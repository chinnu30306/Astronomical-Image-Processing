# ============================================================
# MASK BLEED COLUMN (LEAVING CENTRAL STAR UNTOUCHED)
# ============================================================

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

# ============================================================
# 1. LOAD THE IMAGE
# ============================================================

hdulist = fits.open("Astro/Fits_Data/mosaic_clean.fits")
data    = hdulist[0].data.astype(float)
header  = hdulist[0].header

H, W = data.shape
print(f"Loaded: shape = ({H}, {W})")


# ============================================================
# 2. BUILD THE MASK
# ============================================================

mask = np.zeros(data.shape, dtype=bool)

# ------------------------------------------------------------
# 2a. Rectangles around bright stars
# ------------------------------------------------------------
# (x_centre, y_centre, bleed_halfwidth, star_top, bleed_top, star_bottom, bleed_bottom)
bright_stars = [
    (558, 4098, 2.5, 4106, 4113, 4085, 4080),
    (1456, 4033, 2, 4041, 4048, 4021, 4016),
    (2134, 3759, 4, 3771, 3800, 3745, 3707),
    (775, 3320, 5, 3336, 3417, 3302, 3202),
    (2466, 3414, 3, 3424, 3439, 3402, 3384),
    (973, 2773, 6, 2787, 2834, 2758, 2703),
    (904, 2286, 5.5, 2300, 2355, 2270, 2222),
    (2132, 2130, 3.5, 2319, 2334, 2297, 2285),
    (2088, 1426, 3, 1437, 1450, 1414, 1399),
    (1433, 3212, 20, 3263, H, 3156, 0),
]

# ------------------------------------------------------------
# 2b. Bleed column (split into two segments)
# ------------------------------------------------------------
for (xc, yc, bleed_halfwidth, star_top, bleed_top, star_bottom, bleed_bottom) in bright_stars:
    col_x0 = int(round(xc - bleed_halfwidth))
    col_x1 = int(round(xc + bleed_halfwidth + 1))
    col_x0 = max(0, col_x0)
    col_x1 = min(W, col_x1)

    # Top segment: between star_top and bleed_top
    y_top0 = max(0, int(star_top))
    y_top1 = min(H, int(bleed_top))
    if y_top0 < y_top1:
        mask[y_top0:y_top1, col_x0:col_x1] = True

    # Bottom segment: between bleed_bottom and star_bottom
    y_bot0 = max(0, int(bleed_bottom))
    y_bot1 = min(H, int(star_bottom))
    if y_bot0 < y_bot1:
        mask[y_bot0:y_bot1, col_x0:col_x1] = True


# ------------------------------------------------------------
# 2c. Rectangle at the bottom (x 1018→1703, y 0→461)
# ------------------------------------------------------------
bottom_x0, bottom_x1 = 1018, 1703
bottom_y0, bottom_y1 = 0, 461

mask[bottom_y0:bottom_y1, bottom_x0:bottom_x1] = True
print(f"Bottom rect x∈[{bottom_x0},{bottom_x1}), y∈[{bottom_y0},{bottom_y1})")


# ------------------------------------------------------------
# 2d. Border masks
# ------------------------------------------------------------
border_x = 84    # width of vertical borders on left and right
border_y = 92    # width of horizontal borders on top and bottom

# Left and right
mask[:, :border_x]         = True
mask[:, W - border_x:]     = True

# Top and bottom
mask[:border_y, :]         = True
mask[H - border_y:, :]     = True

print(f"Borders: x-width {border_x} on each side, "
      f"y-width {border_y} on each edge")


# ============================================================
# 3. APPLY THE MASK
# ============================================================

masked_data = data.copy()
masked_data[mask] = np.nan

n_masked = np.sum(mask)
total    = mask.size
print(f"\nTotal masked pixels: {n_masked} / {total} "
      f"({100 * n_masked / total:.2f}%)")


# ============================================================
# 4. PLOT
# ============================================================

plt.figure(figsize=(10, 16))
plt.imshow(masked_data, cmap='gray', origin='lower',
           vmin=3350, vmax=3500)
plt.colorbar(label='Pixel value')
plt.title('Mosaic with bleed column segmented (star visible)')
plt.xlabel('X pixel')
plt.ylabel('Y pixel')
plt.show()


# ============================================================
# 5. SAVE
# ============================================================

hdu_masked = fits.PrimaryHDU(masked_data, header=header)
hdu_masked.writeto("Astro/Fits_Data/mosaic_masked.fits", overwrite=True)

print("Saved Astro/Fits_Data/mosaic_masked.fits")