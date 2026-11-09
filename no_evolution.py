import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
import pandas as pd

# ==========================================
# 1. SIMULATED CATALOGUE LOAD (REPLACE THIS)
# ==========================================
# Replace this random seed generation with your actual array of magnitudes:
# e.g., magnitudes = np.loadtxt("my_catalogue.txt")
catalogue = pd.read_csv("Catalogues/MaskedData_GALAXY_CATALOGUE.csv")
magnitudes = catalogue["mag"].to_numpy(dtype=float)
magnitudes = magnitudes[np.isfinite(magnitudes)]
# Clean out saturation/incomplete extremes for the fit range
magnitudes = magnitudes[(magnitudes >= 20.0) & (magnitudes <= 24.0)]

# ==========================================
# 2. BINNING AND EUCLIDEAN NORMALIZATION
# ==========================================
bin_width = 0.5
bins = np.arange(20.0, 24.01, bin_width)
bin_centers = bins[:-1] + bin_width / 2.0

# Calculate raw differential counts per bin
counts, _ = np.histogram(magnitudes, bins=bins)

# Calculate Poisson noise (as a baseline for statistical weights)
# Note: Yasuda points out that cosmic variance from clustering adds extra error,
# but Poisson error acts as your foundational counting uncertainty indicator.
errors = np.sqrt(counts)
errors = np.where(errors == 0, 1.0, errors)  # Avoid dividing by zero

# Normalize data to build Yasuda's Figure 6 style coordinates
# Y-Axis = Counts / 10^(0.6 * (m - 16))
euclidean_scale = 10**(0.6 * (bin_centers - 16.0))
y_observed_normalized = counts / euclidean_scale
y_errors_normalized = errors / euclidean_scale

# ==========================================
# 3. RELATIVISTIC COSMOLOGY MODEL PREDICTION
# ==========================================
# Cosmological Parameters (Flat Lambda-CDM setup)
H0 = 70.0          # km/s/Mpc
c = 299792.458     # Speed of light in km/s
DH = c / H0        # Hubble distance in Mpc

# Yasuda Table 5 Schechter values for r* band
M_star = -20.83    
alpha = -1.20      
phi_star = 1.40e-2 # Value in h^3 Mpc^-3

def E(z, Omega_M=0.3, Omega_L=0.7):
    """Hubble parameter evolution function."""
    return np.sqrt(Omega_M * (1 + z)**3 + Omega_L)

def comoving_distance(z, Omega_M=0.3, Omega_L=0.7):
    """Calculates transverse comoving distance D_M in Mpc."""
    integral, _ = quad(lambda x: 1.0 / E(x, Omega_M, Omega_L), 0, z)
    return DH * integral

def schechter_lf(M):
    """Standard Schechter Luminosity Function."""
    return phi_star * (10**(0.4 * (M_star - M)))**(alpha + 1) * np.exp(-10**(0.4 * (M_star - M)))

def predict_counts_per_bin(m_start, m_end, Omega_M=0.3, Omega_L=0.7):
    """
    Integrates Schechter functions across volume shells to project
    the total relativistic galaxy counts inside an apparent magnitude bin.
    """
    def integrand(z, m):
        if z <= 0: return 0
        DM = comoving_distance(z, Omega_M, Omega_L)
        DL = (1 + z) * DM # Luminosity distance
        
        # Approximate K-Correction for r-band: K(z) ~ 1.0 * z (rough standard approximation)
        K_corr = 1.0 * z 
        
        # Absolute Magnitude conversion
        M = m - 5 * np.log10(DL * 1e6 / 10.0) - K_corr
        
        # Comoving differential volume element dV/dz
        dV_dz = (DH * (DM**2)) / E(z, Omega_M, Omega_L)
        
        # Convert total volume space over the field of view solid angle fraction
        # (Adjust solid_angle_deg2 based on your survey limits, e.g., 230 for Yasuda North)
        solid_angle_deg2 = 0.05988678847777778
        solid_angle_rad = solid_angle_deg2 * (np.pi / 180.0)**2
        
        return schechter_lf(M) * dV_dz * solid_angle_rad

    # Double integration loop across magnitude slices and redshift paths
    bin_total = 0
    # Average across the discrete bin width window
    m_mid = (m_start + m_end) / 2.0
    # Redshift search window limits
    z_min, z_max = 0.001, 1.5 
    
    integrated_volume_slice, _ = quad(lambda z: integrand(z, m_mid), z_min, z_max)
    return integrated_volume_slice * (m_end - m_start)

# Compute theoretical projection curve across all bins
print("Computing theoretical ΛCDM galaxy counts...")
model_raw_counts = []
for i in range(len(bins)-1):
    m_start, m_end = bins[i], bins[i+1]
    model_raw_counts.append(predict_counts_per_bin(m_start, m_end))

model_raw_counts = np.array(model_raw_counts)

# ==========================================
# 4. CHI-SQUARED TESTING & OPTIMIZATION
# ==========================================
# Scale the model amplitude to map to your specific survey surface coverage area
def chi_squared_loss(scale_factor):
    scaled_model = scale_factor * model_raw_counts
    return np.sum(((counts - scaled_model) / errors) ** 2)

# Minimize loss to find the best amplitude alignment scaling parameter
res = minimize(chi_squared_loss, x0=[1.0], method='Nelder-Mead')
best_scale = res.x[0]

# Compute statistical validation values
final_chi2 = chi_squared_loss(best_scale)
dof = len(counts) - 1 # Degrees of Freedom
reduced_chi2 = final_chi2 / dof

print("\n=== Chi-Squared Test Results ===")
print(f"Total Chi-Squared (χ²): {final_chi2:.4f}")
print(f"Degrees of Freedom (dof): {dof}")
print(f"Reduced Chi-Squared (χ²/dof): {reduced_chi2:.4f}")

# Map scaled model predictions to Yasuda Normalized formatting
y_model_normalized = (best_scale * model_raw_counts) / euclidean_scale

# ==========================================
# 5. GENERATING YASUDA'S NORMALIZED PLOT
# ==========================================
# Optional verification printout if running in environments without graphic frameworks
print("\n=== Yasuda Figure 6 Plot Coordinates ===")
print("Mag_Center | Normalized_Observed | Normalized_Model")
for m, obs, mod in zip(bin_centers, y_observed_normalized, y_model_normalized):
    print(f"{m:10.2f} | {obs:21.4e} | {mod:16.4e}")

# If using matplotlib locally, uncomment this block to render your figure visually:

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.errorbar(bin_centers, y_observed_normalized, yerr=y_errors_normalized, fmt='go', label='Your Data (Normalized)')
plt.plot(bin_centers, y_model_normalized, 'k--', label='Flat ΛCDM Cosmology Model')
plt.yscale('log')
plt.xlabel('Apparent magnitude, m')
plt.ylabel('N × 10^(-0.6 * (m - 16))')
plt.title("Reproduction of Yasuda's Figure 6")
plt.legend()
plt.grid(True, which="both", ls="-")
plt.show()

