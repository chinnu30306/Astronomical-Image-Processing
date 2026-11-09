import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.io import fits


def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

def compute_background(image, bins=1000, range=(3350, 3500)):
    """
    Compute the background mean and standard deviation of the image
    by fitting a Gaussian to the histogram of pixel values.
    """
    counts, bin_edges = np.histogram(image, bins=bins, range=range)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Fit a Gaussian to the histogram
    p0 = [np.max(counts), bin_centres[np.argmax(counts)], 5]  # Initial guess
    fit_rng = (bin_centres > 3395) & (bin_centres < 3435)
    popt, _ = curve_fit(gaussian, bin_centres[fit_rng], counts[fit_rng], p0=p0)

    background_mean = popt[1]
    background_sigma = popt[2]
    # detection_threshold = background_mean + 5 * background_sigma

    return background_mean, background_sigma


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.io import fits

# --- Load Data ---
hdulist = fits.open("Astro/Fits_Data/mosaic_masked.fits")
image_full = hdulist[0].data.astype(np.float32)
header = hdulist[0].header

x_min, x_max = 0, image_full.shape[1]
y_min, y_max = 0, image_full.shape[0]
image = image_full[y_min:y_max, x_min:x_max]

# --- Histogram & Fit ---
bin_width = 1
bins = np.arange(3349.5, 3500.5, bin_width)
counts, bin_edges = np.histogram(image, bins=bins)
bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
counts_masked, _ = np.histogram(image[~np.isnan(image)], bins=bins)

fit_rng = (bin_centres > 3395) & (bin_centres < 3435)

def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

p0 = [np.max(counts_masked[fit_rng]), bin_centres[fit_rng][np.argmax(counts_masked[fit_rng])], 5]
popt, _ = curve_fit(gaussian, bin_centres[fit_rng], counts_masked[fit_rng], p0=p0, maxfev=10000)

# Extract fitted parameters
amp_fit, mu_fit, sigma_fit = popt
sigma_fit = abs(sigma_fit) # ensure positive

# --- Plotting ---

plt.figure(figsize=(10, 5))

# Histogram bars
plt.bar(bin_centres, counts_masked, width=bin_width, color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.3, label='Counts')

# Fitted Gaussian line
plt.plot(bin_centres, gaussian(bin_centres, *popt), 'r-', linewidth=2, label='Fitted Gaussian')

# Mean line
plt.axvline(mu_fit, color='green', linestyle='--', linewidth=2, label=rf'Mean = {mu_fit:.1f}')

# +/- 1 Sigma lines
plt.axvline(mu_fit + sigma_fit, color='orange', linestyle=':', linewidth=2, label=rf'$\pm 1\sigma$ = {sigma_fit:.1f}')
plt.axvline(mu_fit - sigma_fit, color='orange', linestyle=':', linewidth=2)

# Aesthetics
plt.xlabel('Pixel value')
plt.ylabel('Frequency')
plt.title('Histogram of Clean FITS Data')
plt.legend()
plt.show()