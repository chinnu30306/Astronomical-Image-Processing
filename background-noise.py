import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.optimize import curve_fit
from astropy.io import fits
import csv

def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

hdulist = fits.open("/Users/chinmayib/git/Astro/Fits_Data/mosaic.fits")
image = hdulist[0].data.astype(float)
header = hdulist[0].header

bin_width = 1
bins = np.arange(3349.5, 3500.5, bin_width)
counts, bin_edges = np.histogram(image, bins=bins)
bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

masked_image = np.where(image == 3421, np.nan, image)
counts_masked, _ = np.histogram(masked_image[~np.isnan(masked_image)], bins=bins)

fit_rng = (bin_centres > 3395) & (bin_centres < 3435)
p0 = [np.max(counts_masked[fit_rng]), bin_centres[fit_rng][np.argmax(counts_masked[fit_rng])], 5]
popt, _ = curve_fit(gaussian, bin_centres[fit_rng], counts_masked[fit_rng], p0=p0)

background_mean = popt[1]
background_sigma = popt[2]
detection_threshold = background_mean + 5 * background_sigma

plt.figure(figsize=(10, 5))
plt.bar(bin_centres, counts_masked, width=bin_width, color='red', alpha=0.7)
plt.plot(bin_centres, gaussian(bin_centres, *popt), 'b-', linewidth=2)
plt.xlabel('Pixel value')
plt.ylabel('Frequency')
plt.title('Histogram of FITS Data')
plt.show()