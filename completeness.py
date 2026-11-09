import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

def sigmoid(m, m_50, w):
    return 1 / (1 + np.exp((m - m_50) / w))

mag, n_rec, n_inj, compl = np.loadtxt('Catalogues/completeness_data.csv', dtype=float, delimiter=',', skiprows=1, unpack=True)

compl_err = np.sqrt(compl * (1 - compl) / n_inj)
popt, pcov = curve_fit(sigmoid, mag, compl, p0=[24.6, 0.15], sigma=compl_err, absolute_sigma=True, maxfev=20000)
m50, width = popt
m50_err = np.sqrt(pcov[0, 0])
width_err = np.sqrt(pcov[1, 1])
print("m50 = ", m50, "p/m", m50_err)
print("w = ", width, "p/m", width_err)

m_fit = np.linspace(min(mag)-0.5, max(mag)+0.5, 500)
c_fit = sigmoid(m_fit, *popt)

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(mag, compl, yerr=compl_err, fmt='o', color='b', capsize=4, markersize=6, label='Injected & Recovered Sources')
ax.plot(m_fit, c_fit, 'r-', linewidth=2.5, label='Sigmoid Fit')
ax.set_title('Detection Completeness Curve')
ax.set_xlabel('Apparent Magnitude, m')
ax.set_ylabel('Completeness Fraction, C(m)')
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='lower left')
plt.show()



# 1. Load the catalogue
catalogue = pd.read_csv("MaskedData_CATALOGUE.csv")
xs = catalogue["x"].to_numpy()
ys = catalogue["y"].to_numpy()
magnitudes = catalogue["mag"].to_numpy(dtype=float)
magnitudes = magnitudes[np.isfinite(magnitudes)]

# 2. Define Survey Area
survey_area_deg2 = 0.05988678847777778

# 3. Set up magnitude bins
m_limits = np.arange(np.floor(magnitudes.min()), np.ceil(magnitudes.max()) + 0.5, 0.5)
bin_width = 0.5
m_min = np.floor(magnitudes.min() / bin_width) * bin_width
m_max = np.ceil(magnitudes.max() / bin_width) * bin_width
m_edges = np.arange(m_min, m_max + bin_width, bin_width)
m_centres = (m_edges[:-1] + m_edges[1:]) / 2

# ==========================================
# 5. DIFFERENTIAL COUNTS: N(m)
# ==========================================
N_m_raw, _ = np.histogram(magnitudes, bins=m_edges)
N_m_raw_error = np.sqrt(N_m_raw)
N_m_all = N_m_raw / survey_area_deg2
N_m_error_all = N_m_raw_error / survey_area_deg2

valid_1 = N_m_all > 0
m_Nm = m_centres[valid_1]
N_m = N_m_all[valid_1]
N_m_error = N_m_error_all[valid_1]
log_N_m = np.log10(N_m)
log_N_m_error = N_m_error / (N_m * np.log(10))

# --- APPLY COMPLETENESS CORRECTION ---
C_m = sigmoid(m_Nm, m50, width)
valid_corr = C_m > 0.5
m_Nm_corr = m_Nm[valid_corr]
C_m_valid = C_m[valid_corr]

N_m_corr = N_m[valid_corr] / C_m_valid
N_m_corr_error = N_m_error[valid_corr] / C_m_valid

log_N_m_corr = np.log10(N_m_corr)
log_N_m_corr_error = N_m_corr_error / (N_m_corr * np.log(10))

# --- SINGLE FIT FOR CORRECTED DIFFERENTIAL DATA (18 to 24) ---
fit_mask_Nm_corr = ((m_Nm_corr >= 18) & (m_Nm_corr <= 24)) 
m_fit_Nm_corr = m_Nm_corr[fit_mask_Nm_corr]
log_N_m_fit_corr = log_N_m_corr[fit_mask_Nm_corr]
slope_Nm_corr, intercept_Nm_corr = np.polyfit(m_fit_Nm_corr, log_N_m_fit_corr, 1)
log_N_m_line_corr = (slope_Nm_corr * m_fit_Nm_corr + intercept_Nm_corr)

# ==========================================
# 6. CUMULATIVE COUNTS: N(<m)
# ==========================================
N_less_than_m_raw = np.array([np.sum(magnitudes <= m) for m in m_limits])
N_less_than_m_error_raw = np.sqrt(N_less_than_m_raw)
N_less_than_m_all = N_less_than_m_raw / survey_area_deg2
N_less_than_m_error_all = (N_less_than_m_error_raw / survey_area_deg2)

valid_2 = N_less_than_m_all > 0
m_Nlessthanm = m_limits[valid_2]
N_less_than_m = N_less_than_m_all[valid_2]
N_less_than_m_error = N_less_than_m_error_all[valid_2]
log_N_less_than_m = np.log10(N_less_than_m)
log_N_less_than_m_error = N_less_than_m_error / (N_less_than_m * np.log(10))

# --- APPLY COMPLETENESS CORRECTION TO CUMULATIVE COUNTS ---
C_m_cum = sigmoid(m_Nlessthanm, m50, width)
valid_corr_cum = C_m_cum > 0.7 

m_Nlessthanm_corr = m_Nlessthanm[valid_corr_cum]
C_m_cum_valid = C_m_cum[valid_corr_cum]

N_less_than_m_corr = N_less_than_m[valid_corr_cum] / C_m_cum_valid
N_less_than_m_corr_error = N_less_than_m_error[valid_corr_cum] / C_m_cum_valid

log_N_less_than_m_corr = np.log10(N_less_than_m_corr)
log_N_less_than_m_corr_error = N_less_than_m_corr_error / (N_less_than_m_corr * np.log(10))

# --- SINGLE FIT FOR CORRECTED CUMULATIVE COUNTS (18 to 24) ---
fit_mask_cum = ((m_Nlessthanm_corr >= 18) & (m_Nlessthanm_corr <= 24))
m_fit_cum = m_Nlessthanm_corr[fit_mask_cum]
log_N_fit_cum = log_N_less_than_m_corr[fit_mask_cum]
slope_cum, intercept_cum = np.polyfit(m_fit_cum, log_N_fit_cum, 1)
log_N_line_cum = (slope_cum * m_fit_cum + intercept_cum)

# ==========================================
# 7. PLOTS
# ==========================================
# Plot 1: Differential Counts
plt.figure(figsize=(9, 6))
plt.errorbar(m_Nm, log_N_m, yerr=log_N_m_error, fmt='o', color='gray', alpha=0.5, markersize=4, capsize=3, label="Observed (Raw)")
plt.errorbar(m_Nm_corr, log_N_m_corr, yerr=log_N_m_corr_error, fmt='s', color='forestgreen', markersize=4, capsize=3, label="Corrected for Incompleteness")
plt.plot(m_fit_Nm_corr, log_N_m_line_corr, 'k--', linewidth=2, label=rf"Corrected Fit ($18 \leq m \leq 24$) | Slope = {slope_Nm_corr:.3f}")

plt.xlabel(r"Apparent magnitude, $m$")
plt.ylabel(r"$\log_{10} N(m)$")
plt.title("Differential Number Counts with Completeness Correction")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# Plot 2: Cumulative Counts (Corrected)
plt.figure(figsize=(9, 6))
# Raw Data (Gray)
plt.errorbar(m_Nlessthanm, log_N_less_than_m, yerr=log_N_less_than_m_error, fmt='o', color='gray', alpha=0.5, markersize=4, capsize=3, label="Observed (Raw)")

# Corrected Data (Green)
plt.errorbar(m_Nlessthanm_corr, log_N_less_than_m_corr, yerr=log_N_less_than_m_corr_error, fmt='s', color='forestgreen', markersize=4, capsize=3, label="Corrected for Incompleteness")

# Single Fit for Corrected Data
plt.plot(m_fit_cum, log_N_line_cum, 'r-', linewidth=2, 
         label=rf"Corrected Fit ($18 \leq m \leq 24$) | Slope = {slope_cum:.3f}")

plt.xlabel(r"Apparent magnitude, $m$")
plt.ylabel(r"$\log_{10} N(<m)$")
plt.title("Cumulative Number Counts with Completeness Correction")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()