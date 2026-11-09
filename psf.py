# ============================================================
# STAR / GALAXY SEPARATION — Yasuda et al. (2001) PSF-vs-model method
# Complete, standalone script
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import least_squares
from scipy.signal import fftconvolve
from scipy.special import gamma

# ------------------------------------------------------------
# 0. CONFIG & LOADING
# ------------------------------------------------------------
FITS_PATH = "Astro/Fits_Data/mosaic_clean.fits"
CAT_IN    = "Catalogues/MaskedData_STAR_CATALOGUE.csv"
CAT_OUT   = "catalogue_classified.csv"

HS, OS = 12, 3       # Stamp half-size, Oversampling
RATIO_CUT = 0.875    # Yasuda eq. (4) (0.145 mag)
MIN_SNR   = 3.0      # Lowered threshold for initial PSF candidate list
SAT       = 65000    # 16-bit hard limit

hdu = fits.open(FITS_PATH)
image = hdu[0].data.astype(np.float32)
zpt = hdu[0].header['MAGZPT']
hdu.close()

cat = pd.read_csv(CAT_IN)
# Ensure all necessary columns are float
for col in ['x', 'y', 'source_counts', 'background', 'r_ap']:
    cat[col] = pd.to_numeric(cat[col], errors='coerce')

cat_xy = cat[['x', 'y']].values

# Robust background estimation (bypassing compute_background to be safe)
valid_pixels = image[np.isfinite(image)]
med = np.median(valid_pixels)
bg_sigma = 1.4826 * np.median(np.abs(valid_pixels - med))
print(f"{len(cat)} sources, robust background sigma = {bg_sigma:.2f}")

# ------------------------------------------------------------
# 1. MODEL-RENDERING MACHINERY
# ------------------------------------------------------------
N = 2 * HS + 1
sub = (np.arange(N * OS) + 0.5) / OS - HS - 0.5
XS, YS = np.meshgrid(sub, sub)

def bin_down(a): return a.reshape(N, OS, N, OS).mean(axis=(1, 3))

def psf_unit(x, y, s1, s2, b):
    g1 = np.exp(-(x**2 + y**2) / (2 * s1**2)) / (2 * np.pi * s1**2)
    g2 = np.exp(-(x**2 + y**2) / (2 * s2**2)) / (2 * np.pi * s2**2)
    return (g1 + b * g2) / (1 + b)

def psf_image(dx, dy, s1, s2, b): 
    return bin_down(psf_unit(XS - dx, YS - dy, s1, s2, b))

def psf_kernel(s1, s2, b):
    m = int(np.ceil(4 * s2 * OS))
    u = np.arange(-m, m + 1) / OS
    U, V = np.meshgrid(u, u)
    k = psf_unit(U, V, s1, s2, b)
    return k / k.sum()

def gal_image(F, size, q, phi, dx, dy, kind, kernel):
    x, y = XS - dx, YS - dy
    c, s = np.cos(phi), np.sin(phi)
    xr, yr = c * x + s * y, -s * x + c * y
    r = np.sqrt(xr**2 + (yr / q)**2)
    if kind == 'exp':
        prof = np.exp(-r / size) / (2 * np.pi * size**2 * q)
    else:
        b = 7.669
        norm = 2 * np.pi * size**2 * q * np.exp(b) * 4 * gamma(8) / b**8
        prof = np.exp(-b * ((r / size)**0.25 - 1)) / norm
    return bin_down(fftconvolve(F * prof, kernel, mode='same'))

# ------------------------------------------------------------
# 2. STAMP EXTRACTION
# ------------------------------------------------------------
def get_stamp(row):
    x, y = int(row.x), int(row.y)
    if x - HS < 0 or y - HS < 0 or x + HS >= image.shape[1] or y + HS >= image.shape[0]:
        return None
    
    # Subtract local background
    data = image[y-HS:y+HS+1, x-HS:x+HS+1].astype(float) - row.background
    w = np.ones_like(data)
    w[~np.isfinite(data)] = 0
    data = np.nan_to_num(data)
    
    # Mask neighbours
    d = np.hypot(cat_xy[:, 0] - x, cat_xy[:, 1] - y)
    yy, xx = np.mgrid[-HS:HS+1, -HS:HS+1]
    for j in np.where((d > 0.5) & (d < 2 * HS))[0]:
        ox, oy = cat_xy[j, 0] - x, cat_xy[j, 1] - y
        w[np.hypot(xx - ox, yy - oy) <= cat.r_ap.values[j]] = 0
    return data, w

# ------------------------------------------------------------
# 3. MEASURE THE PSF
# ------------------------------------------------------------
def moment_size(data, w):
    yy, xx = np.mgrid[-HS:HS+1, -HS:HS+1]
    m = (np.hypot(xx, yy) <= 8) & (w > 0)
    f = np.clip(data, 0, None) * m
    if f.sum() <= 0: return np.nan
    xc, yc = (f * xx).sum() / f.sum(), (f * yy).sum() / f.sum()
    return np.sqrt(0.5 * (f * ((xx-xc)**2 + (yy-yc)**2)).sum() / f.sum())

# DIAGNOSTIC
print("\n=== PSF CANDIDATE SELECTION ===")
print(f"bg_sigma = {bg_sigma:.2f}  |  threshold = {MIN_SNR * bg_sigma:.2f}")
print(f"source_counts range: [{cat['source_counts'].min():.1f}, {cat['source_counts'].max():.1f}]")

# Select candidates using source_counts (flux above background)
cand = cat[(cat['source_counts'] > MIN_SNR * bg_sigma) & (cat['source_counts'] < SAT)]

if len(cand) == 0:
    print("\nBreakdown of filter conditions:")
    print(f"  source_counts > {MIN_SNR * bg_sigma:.1f}: {(cat['source_counts'] > MIN_SNR * bg_sigma).sum()} pass")
    print(f"  source_counts < {SAT}: {(cat['source_counts'] < SAT).sum()} pass")
    raise RuntimeError("0 PSF candidates. Check the diagnostic output above.")

print(f"→ {len(cand)} candidates out of {len(cat)} total")

sizes, rows = [], []
for row in cand.itertuples():
    st = get_stamp(row)
    if st is None: continue
    s = moment_size(*st)
    if np.isfinite(s):
        sizes.append(s)
        rows.append(row)

sizes = np.array(sizes)
print(f"→ {len(sizes)} valid candidate stamps after size calculation")

if len(sizes) == 0:
    raise RuntimeError("0 valid stamps. Check image extent and get_stamp().")

# Select compact sources (likely stars)
s_ref = np.percentile(sizes, 15)
is_star_like = sizes < 1.15 * s_ref
print(f"→ {is_star_like.sum()} compact sources selected as PSF stars")

def fit_psf_star(row):
    data, w = get_stamp(row)
    def resid(p):
        F, dx, dy, s1, s2, b = p
        return ((data - F * psf_image(dx, dy, s1, s2, b)) * w).ravel() / bg_sigma
    p0 = [max(row.source_counts, 10), 0, 0, 1.5, 3.5, 0.1]
    lo = [0, -2, -2, 0.8, 1.5, 0.0]
    hi = [np.inf, 2, 2, 3.0, 8.0, 1.0]
    r = least_squares(resid, p0, bounds=(lo, hi))
    return r.x[3:]

pp = []
for r, ok in zip(rows, is_star_like):
    if ok:
        try:
            pp.append(fit_psf_star(r))
        except Exception:
            pass

if len(pp) == 0:
    print("WARNING: PSF fit failed. Using default Gaussian approximation.")
    S1, S2, B = 1.5, 3.0, 0.1
else:
    pp = np.array(pp)
    med = np.median(pp, axis=0)
    dev = np.abs(pp - med) / (1.4826 * np.median(np.abs(pp - med), axis=0) + 1e-9)
    good = np.all(dev < 3, axis=1)
    if np.sum(good) == 0:
        S1, S2, B = np.median(pp, axis=0)
    else:
        S1, S2, B = np.median(pp[good], axis=0)

KERNEL = psf_kernel(S1, S2, B)
print(f"PSF: sigma1={S1:.2f}px, sigma2={S2:.2f}px, b={B:.3f}")

# ------------------------------------------------------------
# 4. FIT EVERY SOURCE
# ------------------------------------------------------------
def fit_source(row):
    st = get_stamp(row)
    if st is None: return None
    data, w = st
    F0 = max(row.source_counts, 10.0) 

    # 1. PSF Model
    def res_psf(p): return ((data - p[0] * psf_image(p[1], p[2], S1, S2, B)) * w).ravel() / bg_sigma
    rp = least_squares(res_psf, [F0, 0, 0], bounds=([0, -2, -2], [np.inf, 2, 2]))
    F_psf, chi_psf = rp.x[0], np.sum(rp.fun**2)

    # 2. Galaxy Models (Exp, DeV)
    out = {}
    for kind in ('exp', 'dev'):
        def res_gal(p):
            F, size, q, phi, dx, dy = p
            return ((data - gal_image(F, size, q, phi, dx, dy, kind, KERNEL)) * w).ravel() / bg_sigma
        p0 = [F0, 1.5, 0.8, 0.0, rp.x[1], rp.x[2]]
        lo = [0.1 * F0, 0.3, 0.2, -np.pi/2, -2, -2]
        hi = [10 * F0, 20.0, 1.0,  np.pi/2,  2,  2]
        rg = least_squares(res_gal, p0, bounds=(lo, hi))
        out[kind] = (rg.x[0], np.sum(rg.fun**2), rg.x[1])

    best = min(out, key=lambda k: out[k][1])
    F_mod, chi_mod, size_mod = out[best]
    return F_psf, F_mod, chi_psf, chi_mod, best, size_mod

records = []
for i, row in enumerate(cat.itertuples()):
    res = fit_source(row)
    if res is None or row.source_counts >= SAT:
        records.append(dict(F_psf=np.nan, F_model=np.nan, chi2_psf=np.nan, chi2_model=np.nan, best_model='none', size_model=np.nan))
    else:
        records.append(dict(F_psf=res[0], F_model=res[1], chi2_psf=res[2], chi2_model=res[3], best_model=res[4], size_model=res[5]))
    if (i+1) % 100 == 0:
        print(f"  fitted {i+1}/{len(cat)}")

cat = pd.concat([cat, pd.DataFrame(records)], axis=1)

# ------------------------------------------------------------
# 5. CLASSIFY (Yasuda eq. 4)
# ------------------------------------------------------------
cat['mag_psf']   = zpt - 2.5 * np.log10(cat.F_psf.where(cat.F_psf > 0))
cat['mag_model'] = zpt - 2.5 * np.log10(cat.F_model.where(cat.F_model > 0))
cat['dmag']  = cat.mag_psf - cat.mag_model
cat['ratio'] = cat.F_psf / cat.F_model

cat['class'] = np.where(cat.ratio >= RATIO_CUT, 'star', 'galaxy')
cat.loc[cat.ratio.isna(), 'class'] = 'unclassified'
cat.loc[cat.mag_model < 16.0, 'class'] = 'unclassified' # Yasuda bright-end cut

print("\nClassification counts:")
print(cat['class'].value_counts())
cat.to_csv(CAT_OUT, index=False)

# ------------------------------------------------------------
# 6. DIAGNOSTICS
# ------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for c, col in [('galaxy', 'tab:blue'), ('star', 'tab:red')]:
    m = cat['class'] == c
    ax[0].scatter(cat.mag_model[m], cat.dmag[m], s=4, alpha=0.5, c=col, label=c)
ax[0].axhline(0.145, color='k', ls='--', lw=0.8)
ax[0].set_ylim(-0.5, 3); ax[0].invert_xaxis()
ax[0].set_xlabel('Model magnitude'); ax[0].set_ylabel('m_PSF − m_model')
ax[0].legend(); ax[0].set_title('Star/galaxy separation')

ax[1].imshow(image, origin='lower', cmap='gray', vmin=np.nanpercentile(image, 5), vmax=np.nanpercentile(image, 97))
for c, col in [('galaxy', 'cyan'), ('star', 'red')]:
    m = cat['class'] == c
    ax[1].scatter(cat.x[m], cat.y[m], s=25, facecolors='none', edgecolors=col, lw=0.6)
ax[1].set_title('Red = stars, cyan = galaxies')
plt.tight_layout(); plt.show()