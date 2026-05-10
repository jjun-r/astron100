"""
fast reduc pipeline: [i] raw fast frames in /fast/2026.3017 [o] 1d wl&flux cal spectra of nebulae, line flux table,
figures
"""
import os, glob, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

ROOT = r"D:/claudespace/astron"
DATA = ROOT + "/Astron100-data-Sp26/FAST/2026.0317/"
FIGS = ROOT + "/figures"
RED  = ROOT + "/reduction"
os.makedirs(FIGS, exist_ok=True)
os.makedirs(RED,  exist_ok=True)

# Detector geometry from FITS headers (BIASSEC=[2:30,1:161], TRIMSEC=[35:2715,1:161]).
# IRAF 1-indexed inclusive → Python 0-indexed half-open. Frames are (161, 2720).
# Refs: SAO FAST instrument page; Fabricant et al. 1998, PASP 110, 79.
OSCAN = slice(1, 30)        # 29 overscan cols
TRIM  = slice(34, 2715)     # 2681 science cols
GAIN  = 0.8                 # e-/ADU, FAST3
RDN   = 4.4                 # e- RMS, FAST3

# KPNO atmospheric extinction (mag/airmass), from IRAF onedstds$kpnoextinct.dat.
# Used as the FLWO/Mt-Hopkins substitute (no FLWO-specific table in IRAF).
KPNO_EXT_LAM = np.array([3200, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000])
KPNO_EXT_MAG = np.array([1.017, 0.600, 0.365, 0.243, 0.180, 0.150, 0.133, 0.097, 0.073, 0.061, 0.052])

def overscan_trim(data):
    osc = np.median(data[:, OSCAN], axis=1)
    out = data - osc[:, None]
    return out[:, TRIM].astype(np.float32)

# Master bias / flat: median-combine all 40 frames from 2026.0317 (half from each
# end of the night). Bias level (~12 ADU) and lamp output were stable.
def stack_master(files, label):
    arrs = [overscan_trim(fits.getdata(f, 0)) for f in files]
    arrs = np.array(arrs)
    med  = np.median(arrs, axis=0)
    print(f"  master-{label}: {len(files)} frames, shape={med.shape}, "
          f"median={np.median(med):.2f}")
    return med

print("[1/9] Building master bias and master flat ...")
bias_files = sorted(glob.glob(DATA + "*BIAS*.fits"))
flat_files = sorted(glob.glob(DATA + "*FLAT*.fits"))
print(f"  found {len(bias_files)} BIAS, {len(flat_files)} FLAT frames")
master_bias = stack_master(bias_files, "bias")

# Master flat: bias-subtract, then divide by smoothed lamp spectrum so we keep
# only pixel-to-pixel response (not the lamp colour).
flat_arrs = []
for f in flat_files:
    d = overscan_trim(fits.getdata(f, 0)) - master_bias
    flat_arrs.append(d)
flat_med = np.median(np.array(flat_arrs), axis=0)
lamp_spec = np.median(flat_med[60:100, :], axis=0)
lamp_smooth = median_filter(lamp_spec, size=51)
norm_flat = flat_med / lamp_smooth[None, :]
norm_flat /= np.nanmedian(norm_flat)
print(f"  flat field range: {np.nanpercentile(norm_flat,1):.3f} – "
      f"{np.nanpercentile(norm_flat,99):.3f}")

# Bias-subtract, flat-divide, return 2D image in electrons.
def reduce2d(path, do_flat=True):
    raw = fits.getdata(path, 0)
    out = overscan_trim(raw) - master_bias
    if do_flat:
        out = out / norm_flat
    return out * GAIN

# Extract 1D: find brightest row in central 400 dispersion cols, sum a straight
# ±ap_half aperture, sky = per-column median of two off-trace bands.
# Default ap_half=8 (17 rows ≈ 11"); science overrides to 10 (21 rows) for PNe.
def trace_and_extract(img, ap_half=8, sky1=(5, 25), sky2=(135, 155),
                      label="", save=None):
    ny, nx = img.shape
    profile = np.nanmedian(img, axis=1)
    central = np.nanmedian(img[:, nx//2-200:nx//2+200], axis=1)
    y_peak  = int(np.argmax(median_filter(central, 3)))
    print(f"  [{label}] peak row = {y_peak}  (out of {ny})")

    # FAST trace is flat enough after binning → straight aperture is fine.
    yslc = slice(max(0, y_peak-ap_half), min(ny, y_peak+ap_half+1))
    obj  = np.nansum(img[yslc, :], axis=0)
    npix_obj = (yslc.stop - yslc.start)

    sky_a = img[sky1[0]:sky1[1], :]
    sky_b = img[sky2[0]:sky2[1], :]
    sky_per_pix = np.nanmedian(np.concatenate([sky_a, sky_b], axis=0), axis=0)
    sky_total   = sky_per_pix * npix_obj
    obj_sub     = obj - sky_total

    if save:
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        ax[0].plot(profile, lw=1)
        ax[0].axvspan(yslc.start, yslc.stop, color="C1", alpha=0.3,
                      label="object aperture")
        ax[0].axvspan(*sky1, color="C2", alpha=0.2, label="sky band")
        ax[0].axvspan(*sky2, color="C2", alpha=0.2)
        ax[0].set_xlabel("spatial pixel"); ax[0].set_ylabel("median flux  (e$^-$)")
        ax[0].set_title(f"{label}: spatial profile (median over dispersion)")
        ax[0].legend(loc="upper right", fontsize=8)
        ax[1].plot(obj_sub, lw=0.6, color="k")
        ax[1].set_xlabel("dispersion pixel"); ax[1].set_ylabel("flux  (e$^-$)")
        ax[1].set_title(f"{label}: extracted, sky-subtracted 1D spectrum")
        plt.tight_layout(); plt.savefig(save, dpi=140); plt.close()

    return obj_sub, y_peak

# Wavelength calibration from the bracketing comparison lamp (Ar-dominant).
# Pixel / rest-Å pairs below copied from FAST_reduction_class.ipynb cell "Ar_lamp".
ARC_SEED = """
69.83 3577.6364
173.81 3719.2638
200.30 3767.1883
389.14 4034.9488
397.75 4045.561
416.37 4073.1541
438.19 4105.07
457.35 4132.8886
475.71 4159.762
523.36 4238.4126
557.22 4278.7316
572.78 4301.8591
590.92 4334.779
642.35 4402.2222
692.30 4483.0679
710.87 4511.998
740.21 4546.3258
763.62 4580.6326
770.81 4591.1837
817.14 4659.205
864.14 4737.2302
889.98 4766.1968
917.99 4807.3635
946.37 4849.1639
968.19 4881.2263
1020.98 4966.4649
1056.26 5018.5619
1084.34 5063.4483
1747.80 6033.797
1842.34 6173.9855
2183.31 6679.126
2234.43 6754.698
2314.47 6873.185
2378.10 6967.352
2421.98 7032.19
2447.01 7069.167
2501.04 7148.012
2586.37 7274.94
2661.77 7386.014
"""

def wavecal_from_arc(arc1d, label, save=None):
    seed_pix, seed_lam = [], []
    for ln in ARC_SEED.strip().splitlines():
        p, l = ln.split()
        seed_pix.append(float(p)); seed_lam.append(float(l))
    seed_pix = np.array(seed_pix); seed_lam = np.array(seed_lam)
    init = np.poly1d(np.polyfit(seed_pix, seed_lam, 2))

    peaks, _ = find_peaks(arc1d, height=np.percentile(arc1d, 92), distance=8)
    matched_pix, matched_lam = [], []
    for p in peaks:
        guess = init(p)
        d = np.abs(seed_lam - guess)
        i = np.argmin(d)
        if d[i] < 8.0:
            matched_pix.append(p); matched_lam.append(seed_lam[i])
    matched_pix = np.array(matched_pix); matched_lam = np.array(matched_lam)
    soln = np.poly1d(np.polyfit(matched_pix, matched_lam, 3))
    waves = soln(np.arange(len(arc1d)))
    rms_A = np.sqrt(np.mean((soln(matched_pix) - matched_lam) ** 2))
    print(f"  [{label}] wavecal: {len(matched_pix)} matched lines, RMS = {rms_A:.3f} Å")

    if save:
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        ax[0].plot(matched_pix, matched_lam, "ko", ms=4)
        xx = np.arange(len(arc1d))
        ax[0].plot(xx, soln(xx), "r-", lw=1)
        ax[0].set_xlabel("dispersion pixel"); ax[0].set_ylabel("wavelength (Å)")
        ax[0].set_title(f"FAST wavelength solution ({label})  —  RMS = {rms_A:.2f} Å")
        ax[1].plot(matched_pix, soln(matched_pix) - matched_lam, "ko", ms=4)
        ax[1].axhline(0, color="r", lw=0.8)
        ax[1].set_xlabel("dispersion pixel"); ax[1].set_ylabel("residual (Å)")
        plt.tight_layout(); plt.savefig(save, dpi=140); plt.close()

    return waves, rms_A

# Sensitivity function from Hiltner 600 (Massey et al. 1988, ApJ 328, 315).
# Table copied from IRAF onedstds/spec50cal/hilt600.dat (mAB, 50 Å bins),
# truncated to the FAST 300 l/mm range (3200–7400 Å).
HILTNER600_AB = """
3200 11.10
3250 10.98
3300 10.97
3350 10.90
3400 10.89
3450 10.91
3500 10.91
3550 10.86
3600 10.82
3650 10.84
3700 10.83
3750 10.75
3800 10.64
3850 10.57
3900 10.57
3950 10.58
4000 10.51
4050 10.46
4100 10.58
4150 10.48
4200 10.45
4250 10.45
4300 10.46
4350 10.55
4400 10.46
4450 10.46
4500 10.44
4550 10.43
4600 10.44
4650 10.44
4700 10.44
4750 10.45
4800 10.45
4850 10.55
4900 10.48
4950 10.44
5000 10.45
5050 10.46
5100 10.45
5150 10.45
5200 10.44
5250 10.44
5300 10.43
5350 10.43
5400 10.41
5450 10.43
5500 10.42
5550 10.42
5600 10.42
5650 10.44
5700 10.45
5750 10.44
5800 10.44
5850 10.44
5900 10.50
5950 10.48
6000 10.49
6050 10.50
6100 10.51
6150 10.54
6200 10.54
6250 10.56
6300 10.58
6350 10.57
6400 10.58
6450 10.57
6500 10.58
6550 10.67
6600 10.60
6650 10.56
6700 10.57
6750 10.56
6800 10.55
6850 10.61
6900 10.63
6950 10.55
7000 10.53
7050 10.55
7100 10.52
7150 10.55
7200 10.63
7250 10.61
7300 10.60
7350 10.57
7400 10.56
"""
def build_sensfunc(std_path, std_table, exptime, airmass, label, save=None):
    img = reduce2d(std_path, do_flat=True)
    obj1d, _ = trace_and_extract(img, ap_half=4, sky1=(5,25), sky2=(135,155),
                                 label="standard star",
                                 save=os.path.join(FIGS, f"std_{label}_extract.png"))
    # use the bracketing comp for the standard's wavelength solution
    arc_path = std_path.replace("Hiltner600", "COMP").replace("G191B2B", "COMP")
    # nearest comp lamp file from the directory
    comps = sorted(glob.glob(DATA + "*COMP*.fits"))
    nearest = min(comps, key=lambda p: abs(int(os.path.basename(p).split('.')[0]) -
                                           int(os.path.basename(std_path).split('.')[0])))
    arc1d, _ = trace_and_extract(reduce2d(nearest, do_flat=False),
                                 ap_half=4, sky1=(5,25), sky2=(135,155),
                                 label="arc lamp")
    waves, _ = wavecal_from_arc(arc1d, "standard-star arc",
                                save=os.path.join(FIGS, f"wavecal_{label}.png"))

    # Convert AB mag -> Fλ (erg/s/cm²/Å)
    rows = [r.split() for r in std_table.strip().splitlines()]
    std_lam = np.array([float(r[0]) for r in rows])
    std_ab  = np.array([float(r[1]) for r in rows])
    fnu     = 10 ** (-0.4 * (std_ab + 48.60))                # erg/s/cm²/Hz
    flam    = fnu * 2.998e18 / std_lam ** 2                  # erg/s/cm²/Å
    f_interp = interp1d(std_lam, flam, bounds_error=False, fill_value=np.nan)
    flam_obs = f_interp(waves)

    # observed counts -> photon flux density at the telescope, then airmass-corrected
    obj_per_s = obj1d / exptime
    ext_obs = np.interp(waves, KPNO_EXT_LAM, KPNO_EXT_MAG)
    obj_per_s_corr = obj_per_s * 10 ** (0.4 * ext_obs * airmass)

    sens = flam_obs / obj_per_s_corr   # erg/cm²/Å per e-/s

    # Mask the standard's Balmer absorption + interpolate + smooth, so we don't
    # imprint those features into the science spectra.
    bad = ~np.isfinite(sens)
    sens[bad] = np.nan
    mask_lines = [(3960,3990),(4080,4140),(4320,4360),(4840,4880),(6540,6590)]
    sm = sens.copy()
    for w0, w1 in mask_lines:
        m = (waves > w0) & (waves < w1)
        sm[m] = np.nan
    good = np.isfinite(sm)
    if good.sum() > 50:
        sm_interp = np.interp(waves, waves[good], sm[good])
        sens_smooth = median_filter(sm_interp, size=121)
    else:
        sens_smooth = sm

    if save:
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        ax[0].plot(waves, obj_per_s, "k-", lw=0.6)
        ax[0].set_ylabel("counts (e$^-$/s)")
        ax[0].set_title("extracted standard star spectrum")
        ax[1].semilogy(waves, sens_smooth, "C1-", lw=1)
        ax[1].set_xlabel("wavelength (Å)")
        ax[1].set_ylabel("sensitivity\n(erg cm$^{-2}$ Å$^{-1}$ / (e$^-$/s))")
        ax[1].set_title("smoothed sensitivity function")
        plt.tight_layout(); plt.savefig(save, dpi=140); plt.close()

    return waves, sens_smooth

# RUN: build the standard sensitivity function.
# Hiltner 600 (frame 0053, AIRMASS=1.19, 90 s, UT 03:21) is closest in time
# to the NGC 2392 block (UT 03:40); G191B2B (frame 0051) is at near-identical
# airmass but ~15 min further out. Either standard works for relative flux cal.
print("\n[2/9] Building sensitivity function from Hiltner 600 ...")
std_h_path = DATA + "0053.Hiltner600.fits"
hdr = fits.getheader(std_h_path)
_am = hdr["AIRMASS"]
try: _am = float(_am)
except Exception: _am = 1.20
waves_std, sens = build_sensfunc(std_h_path, HILTNER600_AB,
                                 exptime=float(hdr["EXPTIME"]), airmass=_am,
                                 label="hiltner600",
                                 save=os.path.join(FIGS, "sensitivity_hiltner600.png"))

# Reduce science targets: trace+extract, sky-subtract, wavecal from bracketing
# comp, rebin to common grid, sigma-clip combine, then flux-calibrate.
def reduce_target(target_files, comp_files, label):
    print(f"\n   reducing {label} ({len(target_files)} frames)...")
    spec_list, wave_list, exptimes, airmasses = [], [], [], []
    for sci_f, comp_f in zip(target_files, comp_files):
        sci = reduce2d(sci_f, do_flat=True)
        # broader aperture + farther sky bands for extended PN emission
        s, ypk = trace_and_extract(sci, ap_half=10, sky1=(2,18), sky2=(143,159),
                                   label=os.path.basename(sci_f),
                                   save=os.path.join(FIGS,
                                       f"{label}_{os.path.basename(sci_f).split('.')[0]}_extract.png"))
        arc = reduce2d(comp_f, do_flat=False)
        a, _ = trace_and_extract(arc, ap_half=4, sky1=(2,18), sky2=(143,159),
                                 label="arc-"+os.path.basename(comp_f))
        w, _ = wavecal_from_arc(a, "arc-"+label)
        spec_list.append(s); wave_list.append(w)
        h = fits.getheader(sci_f)
        exptimes.append(h["EXPTIME"])
        try: airmasses.append(float(h["AIRMASS"]))
        except: airmasses.append(np.nan)
    # rebin all frames to a common grid, then 5σ-clip cosmics and average
    common = np.linspace(3700, 7400, 3700)
    rebinned = []
    for w, s in zip(wave_list, spec_list):
        rebinned.append(np.interp(common, w, s, left=np.nan, right=np.nan))
    rebinned = np.array(rebinned)
    med = np.nanmedian(rebinned, axis=0)
    mad = np.nanmedian(np.abs(rebinned - med), axis=0) * 1.4826
    clipped = np.where(np.abs(rebinned - med) > 5 * (mad + 1e-3), np.nan, rebinned)
    coadd_e = np.nanmean(clipped, axis=0)
    total_t = np.nanmean(exptimes)
    flux_es = coadd_e / total_t  # per-frame e-/s, matches sens-function units
    # airmass correction at mean airmass (KPNO curve, see top of file)
    am = np.nanmean(airmasses)
    ext_obs = np.interp(common, KPNO_EXT_LAM, KPNO_EXT_MAG)
    flux_es_corr = flux_es * 10 ** (0.4 * ext_obs * am)
    sens_int = np.interp(common, waves_std, sens, left=np.nan, right=np.nan)
    flux_lam = flux_es_corr * sens_int   # erg/s/cm²/Å
    print(f"   {label}: <airmass>={am:.2f}  <exptime>={total_t:.0f}s  "
          f"len={len(common)}")
    return common, flux_lam, np.nanmean(rebinned, axis=0), am

print("\n[3/9] Reducing NGC 2392 ...")
ngc2392_sci  = [DATA + n for n in ["0055.NGC2392.fits","0056.NGC2392.fits","0057.NGC2392.fits"]]
ngc2392_comp = [DATA + "0058.COMP.fits"] * 3
w2392, f2392, e2392, am2392 = reduce_target(ngc2392_sci, ngc2392_comp, "NGC2392")

print("\n[4/9] Reducing NGC 3242 ...")
ngc3242_sci  = [DATA + n for n in ["0065.NGC3242.fits","0066.NGC3242.fits","0067.NGC3242.fits"]]
ngc3242_comp = [DATA + "0068.COMP.fits"] * 3
w3242, f3242, e3242, am3242 = reduce_target(ngc3242_sci, ngc3242_comp, "NGC3242")

# Persist reduced spectra to disk for downstream steps
np.savez(os.path.join(RED, "reduced_spectra.npz"),
         w2392=w2392, f2392=f2392, e2392=e2392, am2392=am2392,
         w3242=w3242, f3242=f3242, e3242=e3242, am3242=am3242,
         waves_std=waves_std, sens=sens)
print("\n[5/9] Saved reduced spectra to", os.path.join(RED, "reduced_spectra.npz"))

# Emission-line measurement: Gaussian fit on a locally linear continuum.
def gauss(x, A, mu, s):
    return A * np.exp(-0.5 * ((x - mu) / s) ** 2)

def estimate_continuum(w, f, lam0, line_half, cont_off=15, cont_width=10):
    """Linear fit through wing windows on either side of the line."""
    lo = (w > lam0 - cont_off - cont_width) & (w < lam0 - cont_off)
    hi = (w > lam0 + cont_off) & (w < lam0 + cont_off + cont_width)
    cm = lo | hi
    if cm.sum() >= 4 and np.all(np.isfinite(f[cm])):
        c1, c0 = np.polyfit(w[cm], f[cm], 1)
    else:
        c0 = float(np.nanmedian(f))
        c1 = 0.0
    return c0, c1

_FLUX_SCALE = 1e12   # rescale to ~unity so curve_fit tolerances behave

def _null_fit():
    return dict(flux=np.nan, ew=np.nan, sigma=np.nan, lam=np.nan,
                cont=np.nan, ok=False)

def _gauss_result(A, mu, sigma, c0, c1):
    """Convert Gaussian-fit params + linear-continuum coeffs into our line dict."""
    flux = A * abs(sigma) * np.sqrt(2 * np.pi)
    cont = c0 + c1 * mu
    ew = flux / cont if cont != 0 else np.nan
    return dict(flux=flux, ew=ew, sigma=abs(sigma), lam=mu, cont=cont,
                ok=True, A=A, c0=c0, c1=c1)

def _triplet_model(x, A1, A2, A3, mu1, mu2, mu3, s):
    return (A1 * np.exp(-0.5*((x-mu1)/s)**2) +
            A2 * np.exp(-0.5*((x-mu2)/s)**2) +
            A3 * np.exp(-0.5*((x-mu3)/s)**2))

def _doublet_model(x, A1, A2, mu1, mu2, s):
    return (A1 * np.exp(-0.5*((x-mu1)/s)**2) +
            A2 * np.exp(-0.5*((x-mu2)/s)**2))

def fit_line(w, f, lam0, half=12, dlam_max=3.0, cont_off=18, cont_width=12):
    """Single-Gaussian fit on (data − linear wing continuum); rescaled by _FLUX_SCALE."""
    m = (w > lam0 - half) & (w < lam0 + half)
    if m.sum() < 6 or not np.all(np.isfinite(f[m])):
        return _null_fit()
    c0, c1 = estimate_continuum(w, f, lam0, half, cont_off, cont_width)
    x = w[m]; y = f[m]
    yres = (y - (c0 + c1 * x)) * _FLUX_SCALE
    Ag = float(np.nanmax(yres))
    if Ag <= 0 or not np.isfinite(Ag):
        return _null_fit()
    p0 = [Ag, lam0, 3.0]
    bounds = ([0, lam0-dlam_max, 0.6], [np.inf, lam0+dlam_max, 12.0])
    try:
        popt, _ = curve_fit(gauss, x, yres, p0=p0, bounds=bounds,
                            maxfev=8000, xtol=1e-10, ftol=1e-10)
    except Exception:
        return _null_fit()
    A, mu, s = popt
    return _gauss_result(A / _FLUX_SCALE, mu, s, c0, c1)

def fit_NII_Halpha(w, f, dlam_max=3.0):
    """Joint Gaussian fit of [N II] 6548, Hα, [N II] 6583 with linear wing continuum."""
    win = (w > 6500) & (w < 6630)
    if win.sum() < 30 or not np.all(np.isfinite(f[win])):
        return None
    cm = ((w > 6480) & (w < 6500)) | ((w > 6620) & (w < 6640))
    if cm.sum() >= 4 and np.all(np.isfinite(f[cm])):
        c1, c0 = np.polyfit(w[cm], f[cm], 1)
    else:
        c0 = float(np.nanmedian(f[win])); c1 = 0.0
    x = w[win]; y = (f[win] - (c0 + c1 * w[win])) * _FLUX_SCALE
    Ag = float(np.nanmax(y))
    if Ag <= 0 or not np.isfinite(Ag): return None
    p0 = [0.05*Ag, Ag, 0.15*Ag, 6548.1, 6562.8, 6583.5, 3.0]
    bounds = ([0, 0, 0,
               6548.1-dlam_max, 6562.8-dlam_max, 6583.5-dlam_max, 0.6],
              [np.inf, np.inf, np.inf,
               6548.1+dlam_max, 6562.8+dlam_max, 6583.5+dlam_max, 12.0])
    try:
        popt, _ = curve_fit(_triplet_model, x, y, p0=p0, bounds=bounds,
                            maxfev=10000, xtol=1e-10, ftol=1e-10)
    except Exception:
        return None
    A1, A2, A3, mu1, mu2, mu3, s = popt
    A1 /= _FLUX_SCALE; A2 /= _FLUX_SCALE; A3 /= _FLUX_SCALE
    return {"[N II] 6548": _gauss_result(A1, mu1, s, c0, c1),
            "Hα":          _gauss_result(A2, mu2, s, c0, c1),
            "[N II] 6583": _gauss_result(A3, mu3, s, c0, c1)}

def fit_SII_doublet(w, f, dlam_max=3.0):
    win = (w > 6700) & (w < 6750)
    if win.sum() < 12 or not np.all(np.isfinite(f[win])):
        return None
    cm = ((w > 6680) & (w < 6700)) | ((w > 6750) & (w < 6770))
    if cm.sum() >= 4 and np.all(np.isfinite(f[cm])):
        c1, c0 = np.polyfit(w[cm], f[cm], 1)
    else:
        c0 = float(np.nanmedian(f[win])); c1 = 0.0
    x = w[win]; y = (f[win] - (c0 + c1 * w[win])) * _FLUX_SCALE
    Ag = float(np.nanmax(y))
    if Ag <= 0 or not np.isfinite(Ag): return None
    p0 = [Ag, Ag, 6716.4, 6730.8, 3.0]
    bounds = ([0, 0, 6716.4-dlam_max, 6730.8-dlam_max, 0.6],
              [np.inf, np.inf, 6716.4+dlam_max, 6730.8+dlam_max, 12.0])
    try:
        popt, _ = curve_fit(_doublet_model, x, y, p0=p0, bounds=bounds,
                            maxfev=10000, xtol=1e-10, ftol=1e-10)
    except Exception:
        return None
    A1, A2, mu1, mu2, s = popt
    A1 /= _FLUX_SCALE; A2 /= _FLUX_SCALE
    return {"[S II] 6716": _gauss_result(A1, mu1, s, c0, c1),
            "[S II] 6731": _gauss_result(A2, mu2, s, c0, c1)}

LINES = [
    # [O II] 3727 is the unresolved 3726.03+3728.82 NIST doublet centroid.
    ("[O II] 3727",  3727.4),
    ("Hδ",           4101.7),
    ("Hγ",           4340.5),
    ("He II 4686",   4685.7),
    ("Hβ",           4861.3),
    ("[O III] 4959", 4958.9),
    ("[O III] 5007", 5006.8),
    ("He I 5876",    5875.6),
    ("[O I] 6300",   6300.3),
    ("[N II] 6548",  6548.1),
    ("Hα",           6562.8),
    ("[N II] 6583",  6583.5),
    ("He I 6678",    6678.2),
    ("[S II] 6716",  6716.4),
    ("[S II] 6731",  6730.8),
]

def measure_all(w, f):
    out = {}
    for name, lam0 in LINES:
        if "[N II]" in name or name == "Hα" or "[S II]" in name:
            continue  # handled jointly below
        out[name] = fit_line(w, f, lam0, half=18)
    triple = fit_NII_Halpha(w, f)
    if triple is not None:
        out.update(triple)
    else:
        for n in ["[N II] 6548", "Hα", "[N II] 6583"]:
            out[n] = _null_fit()
    sii = fit_SII_doublet(w, f)
    if sii is not None:
        out.update(sii)
    else:
        for n in ["[S II] 6716", "[S II] 6731"]:
            out[n] = _null_fit()
    return out

print("\n[6/9] Measuring emission lines ...")
m2392 = measure_all(w2392, f2392)
m3242 = measure_all(w3242, f3242)

def fmt(d, name):
    v = d[name]
    if not v["ok"] or not np.isfinite(v["flux"]):
        return f"{name:14s}  --"
    return (f"{name:14s}  λ_obs={v['lam']:7.2f}  "
            f"F={v['flux']:.3e}  EW={v['ew']:7.2f}")

print("--- NGC 2392 ---")
for n,_ in LINES: print(" ", fmt(m2392, n))
print("--- NGC 3242 ---")
for n,_ in LINES: print(" ", fmt(m3242, n))

# Diagnostics: Balmer decrement, [S II] density, [O III]/Hβ, He II/Hβ.
# Reddening curve: Cardelli+89 at R_V = 3.1.
def ccm89(lam_ang, R_V=3.1):
    """Cardelli, Clayton & Mathis 1989, ApJ 345, 245 — eqs 3a/3b for the
    optical/NIR branch (1.1 ≤ x ≤ 3.3 µm⁻¹). Returns A(λ)/A(V)."""
    x = 1e4 / np.atleast_1d(lam_ang)  # 1/μm
    a = np.zeros_like(x); b = np.zeros_like(x)
    yopt = (x >= 1.1) & (x <= 3.3)
    y = (x[yopt] - 1.82)
    a[yopt] = (1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 +
               0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7)
    b[yopt] = (1.41338*y + 2.28305*y**2 + 1.07233*y**3 -
               5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7)
    return (a + b / R_V).squeeze()  # A(λ)/A(V)

# [S II] λ6716/λ6731 → n_e at T_e = 10^4 K. Primary path: PyNeb. Fallback: a
# 9-point grid pre-tabulated from PyNeb 1.1.x via S2.getEmissivity(tem=1e4, …)
# so the script still runs without pyneb installed.
_SII_FALLBACK_R = np.array([1.444, 1.421, 1.351, 1.196, 0.914, 0.669, 0.520, 0.467, 0.447])
_SII_FALLBACK_N = np.array([10.0,  30.0,  100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0])

def density_from_sii(R, T_e=1e4):
    if not np.isfinite(R): return np.nan
    try:
        import pyneb as pn
        S2 = pn.Atom('S', 2)
        ne = S2.getTemDen(R, tem=T_e, wave1=6716, wave2=6731)
        if np.isfinite(ne) and ne > 0:
            return float(ne)
    except Exception:
        pass
    # Fallback: monotonic-inverse interp on the pre-tabulated PyNeb grid.
    f = interp1d(_SII_FALLBACK_R[::-1], _SII_FALLBACK_N[::-1],
                 bounds_error=False,
                 fill_value=(_SII_FALLBACK_N[-1], _SII_FALLBACK_N[0]))
    return float(f(R))

def diagnostics(m, label, oiii_saturated=False, sii_unreliable=False):
    """Reddening, density, and excitation diagnostics.
    oiii_saturated: replace F(5007) with 2.98 × F(4959) (Storey & Zeippen 2000).
    sii_unreliable: report n_e as non-detection when doublet is below S/N.
    """
    d = {}
    Hb = m["Hβ"]["flux"]; Ha = m["Hα"]["flux"]
    d["Halpha_Hbeta"] = Ha / Hb if (Ha and Hb) else np.nan
    # E(B-V) from Balmer decrement vs Case B 2.86
    R_obs = d["Halpha_Hbeta"]
    if np.isfinite(R_obs) and R_obs > 0:
        kHa = ccm89(6562.8) * 3.1
        kHb = ccm89(4861.3) * 3.1
        # Case-B intrinsic Hα/Hβ = 2.86 at Te=1e4 K (Storey & Hummer 1995; Osterbrock 2006)
        d["EBV"] = (2.5 / (kHb - kHa)) * np.log10(R_obs / 2.86)
        d["EBV"] = max(d["EBV"], 0.0)
    else:
        d["EBV"] = np.nan

    # n_e from [S II] doublet — flagged unreliable below S/N threshold
    if sii_unreliable:
        d["SII_ratio"] = np.nan
        d["ne"] = np.nan
        d["SII_note"] = "doublet below S/N threshold (NGC 3242 line cores not detected)"
    else:
        R_SII = (m["[S II] 6716"]["flux"] / m["[S II] 6731"]["flux"]
                 if m["[S II] 6716"]["ok"] and m["[S II] 6731"]["ok"] else np.nan)
        d["SII_ratio"] = R_SII
        d["ne"] = density_from_sii(R_SII)
        d["SII_note"] = ""

    # [O III] 5007/Hβ — fall back to 2.98 × F(4959) if 5007 saturated
    # (Storey & Zeippen 2000, MNRAS 312, 813).
    if oiii_saturated:
        F4959 = m["[O III] 4959"]["flux"]
        F5007 = 2.98 * F4959 if np.isfinite(F4959) else np.nan
        d["OIII_Hb"] = F5007 / Hb if (np.isfinite(F5007) and Hb) else np.nan
        d["OIII_note"] = "5007 saturated; using 2.98 × F(4959) per Storey & Zeippen 2000"
    else:
        d["OIII_Hb"] = (m["[O III] 5007"]["flux"] / Hb
                        if (m["[O III] 5007"]["ok"] and Hb) else np.nan)
        d["OIII_note"] = ""
    d["HeII_Hb"] = (m["He II 4686"]["flux"] / Hb
                    if (m["He II 4686"]["ok"] and Hb) else np.nan)
    d["NII_Ha"]  = (m["[N II] 6583"]["flux"] / Ha
                    if (m["[N II] 6583"]["ok"] and Ha) else np.nan)
    d["HeI_Hb"]  = (m["He I 5876"]["flux"] / Hb
                    if (m["He I 5876"]["ok"] and Hb) else np.nan)
    print(f"\n[diagnostics: {label}]")
    for k, v in d.items():
        if isinstance(v, str):
            if v: print(f"  {k:14s} = {v}")
        else:
            print(f"  {k:14s} = {v:.4g}")
    return d

print("\n[7/9] Computing diagnostics ...")
# Saturation/SN flags from inspection of the raw frames:
#   - NGC 3242: ~50 px at the 65535 ADC ceiling at [O III] 5007 in each frame
#   - NGC 3242: [S II] 6716 fit ~7e-16 cgs vs ~1e-14 noise floor → non-detection
def _is_sii_unreliable(m):
    f1 = m["[S II] 6716"]["flux"]
    f2 = m["[S II] 6731"]["flux"]
    if not (np.isfinite(f1) and np.isfinite(f2)):
        return True
    # ratio outside the physical [0.40, 1.45] envelope ⇒ unreliable
    R = f1 / f2 if f2 > 0 else np.nan
    return (not np.isfinite(R)) or R < 0.30 or R > 1.55

d2392 = diagnostics(m2392, "NGC 2392",
                    oiii_saturated=False,
                    sii_unreliable=_is_sii_unreliable(m2392))
d3242 = diagnostics(m3242, "NGC 3242",
                    oiii_saturated=True,                       # confirmed
                    sii_unreliable=_is_sii_unreliable(m3242))   # also flagged

# De-redden with the derived E(B-V) and normalize ratios to F(Hβ) = 100
# (standard nebular table format).
def deredden_table(m, EBV):
    table = []
    Hb = m["Hβ"]["flux"]
    if not np.isfinite(EBV): EBV = 0.0
    kHb = ccm89(4861.3) * 3.1
    for name, lam0 in LINES:
        v = m[name]
        if not v["ok"] or not np.isfinite(v["flux"]) or Hb is None:
            table.append((name, lam0, np.nan, np.nan, np.nan)); continue
        corr = 10 ** (0.4 * EBV * (ccm89(lam0) * 3.1 - kHb))  # F(λ)/F(Hβ) multiplier
        ratio_obs = v["flux"] / Hb
        ratio_dered = ratio_obs * corr
        table.append((name, lam0, v["flux"], ratio_obs * 100, ratio_dered * 100))
    return table

t2392 = deredden_table(m2392, d2392["EBV"])
t3242 = deredden_table(m3242, d3242["EBV"])

# Figures for the slides.
print("\n[8/9] Writing figures ...")

# (a) Full reduced spectra, with key emission lines annotated
LINE_LABELS = [(3727, "[O II]"), (4101.7, "Hδ"), (4340.5, "Hγ"),
               (4685.7, "He II"), (4861.3, "Hβ"), (5006.8, "[O III]"),
               (5875.6, "He I"), (6562.8, "Hα"), (6583.5, "[N II]")]
fig, ax = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)
for a, w, f, lab, col in [(ax[0], w2392, f2392, "NGC 2392 (Eskimo)", "C0"),
                          (ax[1], w3242, f3242, "NGC 3242 (Ghost of Jupiter)", "C3")]:
    a.plot(w, f, color=col, lw=0.7)
    a.set_ylabel(r"flux  $F_\lambda$" "\n" r"(erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)")
    a.set_title(lab)
    a.set_xlim(3700, 7400)
    ymax = np.nanpercentile(f, 99.5) * 1.2
    a.set_ylim(0, ymax)
    for lam0, name in LINE_LABELS:
        a.axvline(lam0, color="0.7", lw=0.4, ls=":")
        a.text(lam0, ymax * 0.93, name, rotation=90, ha="center",
               va="top", fontsize=7, color="0.3")
ax[1].set_xlabel("wavelength (Å)")
plt.tight_layout(); plt.savefig(os.path.join(FIGS, "01_full_spectra.png"), dpi=160); plt.close()

# (b) zoomed line panels (Hβ/[OIII] block, Hα/[NII]/[SII] block) with labels
fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
zooms = [
    ("Hβ + [O III]", 4750, 5100,
     [(4861.3, "Hβ"), (4958.9, "[O III] 4959"), (5006.8, "[O III] 5007")]),
    ("Hα + [N II] + [S II]", 6500, 6770,
     [(6548.1, "[N II] 6548"), (6562.8, "Hα"), (6583.5, "[N II] 6583"),
      (6716.4, "[S II] 6716"), (6730.8, "[S II] 6731")]),
]
for col, (title, lo, hi, labels) in enumerate(zooms):
    for row, (w, f, lab, color) in enumerate([
        (w2392, f2392, "NGC 2392", "C0"),
        (w3242, f3242, "NGC 3242", "C3")]):
        m = (w > lo) & (w < hi)
        ax[row][col].plot(w[m], f[m], color=color, lw=0.9)
        ax[row][col].set_xlim(lo, hi)
        ax[row][col].set_title(f"{lab}: {title}")
        ax[row][col].set_xlabel("wavelength (Å)")
        ax[row][col].set_ylabel(r"$F_\lambda$ (cgs)")
        ymax = ax[row][col].get_ylim()[1]
        for lam0, lname in labels:
            ax[row][col].axvline(lam0, color="0.7", lw=0.4, ls=":")
            ax[row][col].text(lam0, ymax*0.95, lname, rotation=90,
                              ha="center", va="top", fontsize=7, color="0.3")
plt.tight_layout(); plt.savefig(os.path.join(FIGS, "02_line_zooms.png"), dpi=160); plt.close()

# (c) Balmer decrement bar
fig, ax = plt.subplots(figsize=(6.5, 4.5))
labels = ["NGC 2392", "NGC 3242"]
vals = [d2392["Halpha_Hbeta"], d3242["Halpha_Hbeta"]]
EBVs = [d2392["EBV"], d3242["EBV"]]
bars = ax.bar(labels, vals, color=["C0","C3"], width=0.55)
ax.axhline(2.86, color="k", ls="--", lw=1.2, label="Case B (intrinsic) = 2.86")
for i, (v, e) in enumerate(zip(vals, EBVs)):
    if np.isfinite(v):
        ax.text(i, v/2, f"{v:.2f}\nE(B–V) = {e:.2f}",
                ha="center", va="center", fontsize=11,
                color="white", fontweight="bold")
ax.set_ylim(0, max(vals)*1.25)
ax.set_ylabel(r"observed F(H$\alpha$)/F(H$\beta$)")
ax.set_title("Balmer decrement → inferred reddening")
ax.legend(loc="upper right")
plt.tight_layout(); plt.savefig(os.path.join(FIGS, "03_balmer_decrement.png"), dpi=160); plt.close()

# (d) [S II] density panel — zoom on doublet, with annotations
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
texts = [
    f"R = F(6716)/F(6731) = {d2392['SII_ratio']:.2f}\n"
    f"n_e ≈ {d2392['ne']:.2g} cm$^{{-3}}$",
    "doublet below detection threshold\n"
    "(consistent with low-density limit;\n"
    " published n_e ≈ 1000–4000 cm$^{-3}$)",
]
for axe, w, f, t, col, txt in [(ax[0], w2392, f2392, "NGC 2392", "C0", texts[0]),
                               (ax[1], w3242, f3242, "NGC 3242", "C3", texts[1])]:
    m = (w > 6680) & (w < 6760)
    axe.plot(w[m], f[m], color=col, lw=1.0)
    for lam0, name in [(6716.4, "[S II] 6716"), (6730.8, "[S II] 6731")]:
        axe.axvline(lam0, color="0.6", lw=0.5, ls=":")
        ymin, ymax = axe.get_ylim()
    axe.set_xlim(6680, 6760)
    axe.set_xlabel("wavelength (Å)"); axe.set_ylabel(r"$F_\lambda$ (cgs)")
    axe.set_title(t + "  —  [S II] doublet")
    axe.annotate(txt, xy=(0.04, 0.96), xycoords="axes fraction",
                 ha="left", va="top", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.4", fc="white",
                           ec="0.5", alpha=0.9))
    # set ylim before annotating for it to work, then re-set
    yl = axe.get_ylim()
    axe.set_ylim(yl[0], yl[1])
    for lam0, name in [(6716.4, "[S II] 6716"), (6730.8, "[S II] 6731")]:
        axe.text(lam0, yl[1]*0.93, name, rotation=90, ha="center",
                 va="top", fontsize=7, color="0.3")
plt.tight_layout(); plt.savefig(os.path.join(FIGS, "04_sii_doublet.png"), dpi=160); plt.close()

# (e) summary diagnostics figure (4 panels for the 4 main diagnostics)
fig, ax = plt.subplots(1, 4, figsize=(13, 4.2))
diag = [("[O III] 5007 / Hβ", "OIII_Hb"),
        ("He II 4686 / Hβ",   "HeII_Hb"),
        ("[N II] 6583 / Hα",  "NII_Ha"),
        ("[S II] 6716/6731",  "SII_ratio")]
for i, (label, key) in enumerate(diag):
    v1 = d2392[key]; v2 = d3242[key]
    ax[i].bar([0, 1], [v1 if np.isfinite(v1) else 0,
                       v2 if np.isfinite(v2) else 0],
              color=["C0","C3"], width=0.55)
    ax[i].set_xticks([0,1]); ax[i].set_xticklabels(["NGC 2392","NGC 3242"], fontsize=10)
    ax[i].set_title(label, fontsize=11)
    if np.isfinite(v1):
        ax[i].text(0, v1, f"{v1:.2f}", ha="center", va="bottom", fontsize=10)
    if np.isfinite(v2):
        ax[i].text(1, v2, f"{v2:.2f}", ha="center", va="bottom", fontsize=10)
    else:
        ax[i].text(1, 0, "n.d.", ha="center", va="bottom",
                   fontsize=10, color="C3")
plt.tight_layout(); plt.savefig(os.path.join(FIGS, "05_diagnostic_summary.png"), dpi=160); plt.close()

# (f-new) Data-quality / saturation figure: an annotated 2D image of one
# NGC 3242 frame showing the 50 saturated pixels at [O III] 5007.
ngc3242_raw = fits.getdata(DATA + "0065.NGC3242.fits", 0).astype(float)
sat = ngc3242_raw > 60000
y_peak = 67  # from extraction
fig, ax = plt.subplots(2, 1, figsize=(11, 6.5))
ax[0].imshow(ngc3242_raw, origin="lower", cmap="gray",
             vmin=np.percentile(ngc3242_raw, 5),
             vmax=np.percentile(ngc3242_raw, 99.5),
             aspect="auto")
sy, sx = np.where(sat)
ax[0].scatter(sx, sy, color="red", s=4, label=f"saturated px (>60000 ADU)  n={sat.sum()}")
ax[0].set_xlabel("dispersion pixel"); ax[0].set_ylabel("spatial pixel")
ax[0].set_title("NGC 3242 raw frame (0065.NGC3242.fits)  —  [O III] 5007 saturated")
ax[0].legend(loc="upper right")
ax[1].plot(ngc3242_raw[y_peak, :], "k-", lw=0.6)
ax[1].axhline(65535, color="red", ls=":", lw=1, label="ADC saturation = 65535")
ax[1].set_xlabel("dispersion pixel (raw, full frame)")
ax[1].set_ylabel("ADU at trace row")
ax[1].set_title("Cross-cut along the trace — saturation in [O III] 4959/5007")
ax[1].legend()
plt.tight_layout(); plt.savefig(os.path.join(FIGS, "06_saturation.png"), dpi=160); plt.close()

# (i-new) Proposal vs reality figure
fig, ax = plt.subplots(figsize=(13, 4.6))
ax.axis("off")
rows = [
    ("Quantity",                  "Proposed",          "Actual on disk",     "Impact"),
    ("NGC 2392 deep exposures",   "3 × 300 s",         "3 × 60 s",
     "5× less integration; still high\nSNR for all major lines"),
    ("NGC 2392 short exposure",   "1 × 60 s",          "—",
     "n/a — long exposures already 60 s,\nso no saturation"),
    ("NGC 3242 deep exposures",   "3 × 180 s",         "3 × 60 s",
     "3× less integration; [O III] 5007\nstill saturated (≥50 px)"),
    ("NGC 3242 short exposure",   "1 × 30 s",          "(not taken)",
     "no unsaturated 5007;\nrecovered via 2.98 × F(4959)"),
    ("Slit PA repeat",            "1 per target",      "(not taken)",
     "cannot test PA dependence\nof integrated ratios"),
    ("FITS header (NGC 3242)",    "OBJECT = NGC3242",  "OBJECT = '2025aico'",
     "metadata mis-tagged; filename\n& slit RA/Dec confirm correct target"),
]
table = ax.table(cellText=[r for r in rows[1:]], colLabels=rows[0],
                 cellLoc="left", loc="center",
                 colWidths=[0.20, 0.18, 0.20, 0.42])
table.auto_set_font_size(False); table.set_fontsize(9)
table.scale(1.0, 1.9)
ax.set_title("Proposed observing setup vs what is actually on disk",
             fontsize=12, pad=10)
plt.savefig(os.path.join(FIGS, "08_proposal_vs_reality.png"),
            dpi=160, bbox_inches="tight")
plt.close()

# (g-new) Side-by-side comparison table figure (graphical)
fig, ax = plt.subplots(figsize=(13, 4.8))
ax.axis("off")
rows = [
    ("Property",           "NGC 2392 (Eskimo)",       "NGC 3242 (Ghost of Jupiter)", "literature value (ref)"),
    ("Hα/Hβ (observed)",   f"{d2392['Halpha_Hbeta']:.2f}", f"{d3242['Halpha_Hbeta']:.2f}", "Case B = 2.86 (S&H 1995)"),
    ("E(B–V) inferred",    f"{d2392['EBV']:.2f}",     f"{d3242['EBV']:.2f}",          "0.15 / 0.08 (Pottasch 2008a/b)"),
    ("[O III] 5007 / Hβ",  f"{d2392['OIII_Hb']:.1f}", f"{d3242['OIII_Hb']:.1f}*",     "10.4 / ~12 (Pan 2025; Monteiro 2013)"),
    ("He II 4686 / Hβ",    f"{d2392['HeII_Hb']:.2f}", f"{d3242['HeII_Hb']:.2f}",      "0.46 / 0.29 (Pan 2025; Pottasch 2008b)"),
    ("[N II] 6583 / Hα",   f"{d2392['NII_Ha']:.2f}",  f"{d3242['NII_Ha']:.3f}",       "0.30 / 0.05–0.16 (Pan 2025; Monteiro 2013)"),
    ("[S II] 6716/6731",   f"{d2392['SII_ratio']:.2f}",  "n.d. (low-density)",        "ratio range 1.45→0.45"),
    ("inferred n_e (cm⁻³)",f"{d2392['ne']:.0f}",      "≲ 4000 (n.d. lower bound)",    "~3000 / ~2200 (Pottasch / Monteiro)"),
]
table = ax.table(cellText=[r for r in rows[1:]], colLabels=rows[0],
                 cellLoc="center", loc="center",
                 colWidths=[0.18, 0.18, 0.22, 0.42])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.6)
ax.set_title("Comparative spectroscopic diagnostics — NGC 2392 vs NGC 3242\n"
             "[*] NGC 3242 5007 saturated; recovered as 2.98 × F(4959)",
             fontsize=11, pad=10)
plt.savefig(os.path.join(FIGS, "07_summary_table.png"), dpi=160, bbox_inches="tight")
plt.close()

# (f) line-table (de-reddened) saved to text
def write_line_table(t, lab):
    lines = [f"# de-reddened line ratios for {lab}, F(Hβ) ≡ 100"]
    lines.append("# name             λ_rest    F_obs(cgs)   F/F(Hβ)_obs  F/F(Hβ)_dered")
    for name, lam, fobs, robs, rdered in t:
        if np.isfinite(fobs):
            lines.append(f"  {name:<14s} {lam:8.2f}   {fobs:10.3e}    {robs:8.2f}      {rdered:8.2f}")
        else:
            lines.append(f"  {name:<14s} {lam:8.2f}   --             --           --")
    return "\n".join(lines)

with open(os.path.join(RED, "line_table_NGC2392.txt"), "w", encoding="utf-8") as fh:
    fh.write(write_line_table(t2392, "NGC 2392"))
with open(os.path.join(RED, "line_table_NGC3242.txt"), "w", encoding="utf-8") as fh:
    fh.write(write_line_table(t3242, "NGC 3242"))

# (g) summary text
def _fmt_value(v):
    if isinstance(v, str): return v
    if not np.isfinite(v): return "—"
    return f"{v:.4g}"

with open(os.path.join(RED, "summary_diagnostics.txt"), "w", encoding="utf-8") as fh:
    fh.write("Comparative PN diagnostics summary\n")
    fh.write("===================================\n\n")
    fh.write(f"NGC 2392 (Eskimo Nebula)\n")
    for k,v in d2392.items():
        if isinstance(v, str) and not v: continue
        fh.write(f"  {k:>14s} = {_fmt_value(v)}\n")
    fh.write(f"\nNGC 3242 (Ghost of Jupiter)\n")
    for k,v in d3242.items():
        if isinstance(v, str) and not v: continue
        fh.write(f"  {k:>14s} = {_fmt_value(v)}\n")

print("\n[9/9] Done. Figures in", FIGS, "and tables in", RED)
