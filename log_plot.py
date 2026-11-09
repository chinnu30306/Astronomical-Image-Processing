# ============================================================
# 10. NUMBER COUNTS: Differential log N(m) vs magnitude
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import csv

# 1. Load the catalogue
with open("catalogue.csv", "r") as f:
    reader = csv.DictReader(f)
    catalogue = list(reader)
    for s in catalogue:
        s['mag'] = float(s['mag'])

# ============================================================
# 1. DEFINE AREA AND EXTRACT DATA
# ============================================================
area_sq_deg = (2570 * 4611 * (0.258 / 3600)**2) 

mags = np.array([s['mag'] for s in catalogue if np.isfinite(s['mag'])])
mags.sort()

# ============================================================
# 2. CALCULATE DIFFERENTIAL COUNTS
# ============================================================
# Create 0.5-magnitude bins from 14 to 26
bin_edges = np.arange(14.0, 26.5, 0.5)
bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Count how many sources fall in each bin
N_raw, _ = np.histogram(mags, bins=bin_edges)
N_norm = N_raw / area_sq_deg

# Filter out bins where N_raw == 0 (can't take log of 0)
valid = N_raw > 0
x_plot = bin_centres[valid]
y_plot = np.log10(N_norm[valid])
N_valid = N_raw[valid]

# ============================================================
# 3. CALCULATE POISSON ERROR BARS
# ============================================================
# Error on N is sqrt(N). 
# Error on log10(N) is (1 / ln(10)) * (error_N / N)
err_N = np.sqrt(N_valid)
err_logN = (1.0 / np.log(10)) * (err_N / N_valid)

# ============================================================
# 4. FIT THE LINEAR REGION
# ============================================================
# Fit the bright and intermediate magnitude ranges where counts
# are rising steadily and completeness is good.
fit_mask_1 = (x_plot >= 14) & (x_plot <= 18)
fit_mask_2 = (x_plot >= 18) & (x_plot <= 24)

if np.sum(fit_mask_1) > 2:
    coeffs_1 = np.polyfit(x_plot[fit_mask_1], y_plot[fit_mask_1], 1, w=1/err_logN[fit_mask_1])
    slope_1, intercept_1 = coeffs_1
else:
    slope_1, intercept_1 = np.nan, np.nan

if np.sum(fit_mask_2) > 2:
    coeffs_2 = np.polyfit(x_plot[fit_mask_2], y_plot[fit_mask_2], 1, w=1/err_logN[fit_mask_2])
    slope_2, intercept_2 = coeffs_2
else:
    slope_2, intercept_2 = np.nan, np.nan

# Generate line data for plotting
fit_x_1 = np.linspace(14, 18, 100)
fit_y_1 = slope_1 * fit_x_1 + intercept_1

fit_x_2 = np.linspace(18, 24, 100)
fit_y_2 = slope_2 * fit_x_2 + intercept_2

# ============================================================
# 5. PLOT
# ============================================================
plt.figure(figsize=(10, 7))

# Plot the data points with error bars
plt.errorbar(x_plot, y_plot, yerr=err_logN, 
             fmt='o', color='#3b75af', ecolor='#3b75af',
             capsize=4, markersize=6, label='Differential Counts')

# Plot the best-fit lines
plt.plot(fit_x_1, fit_y_1, 'r-', linewidth=2.5, 
         label=f'Fit (14–18 mag, slope = {slope_1:.3f})')
# plt.plot(fit_x_2, fit_y_2, 'g-', linewidth=2.5, 
#          label=f'Fit (18–24 mag, slope = {slope_2:.3f})')

# Aesthetics
plt.title(r'Differential Number Counts: \(\log_{10} N(m)\) vs \(m\)', fontsize=16)
plt.xlabel('Apparent magnitude, \(m\)', fontsize=14)
# Note the y-axis label change to remove "(<m)" and add "per 0.5 mag"
plt.ylabel(r'\(\log_{10} N(m)\) [sources deg\(^{-2}\) per 0.5 mag]', fontsize=14)
plt.grid(True, linestyle='-', alpha=0.7)
plt.legend(fontsize=12, loc='upper left')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Optional: set axis limits to zoom in
# plt.xlim(14.5, 26)
# plt.ylim(0, 5.0)

plt.tight_layout()
plt.show()

# Print the result
print(f"Fitted slope (14-18 mag): {slope_1:.3f}")
print(f"Fitted slope (18-24 mag): {slope_2:.3f}")