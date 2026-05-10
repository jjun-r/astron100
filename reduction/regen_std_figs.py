"""Regenerate only the three standard-star figures (sensitivity, std extract,
wavecal) with the updated (no "hiltner600") titles. Reuses the helpers from
reduce_fast.py so the figure contents are identical to the full pipeline."""
import os, sys, glob
import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importing reduce_fast runs the full pipeline at module import, which is
# expensive. We instead exec only the helper-definition block (everything up
# to but not including the "RUN: build the standard sensitivity function"
# comment), then call build_sensfunc ourselves.
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reduce_fast.py")
with open(SRC, "r", encoding="utf-8") as fh:
    text = fh.read()
cut = text.index("# RUN: build the standard sensitivity function")
# Execute only the setup + helpers in a fresh namespace
ns = {"__name__": "__main__"}
exec(compile(text[:cut], SRC, "exec"), ns)

build_sensfunc = ns["build_sensfunc"]
DATA = ns["DATA"]; FIGS = ns["FIGS"]
HILTNER600_AB = ns["HILTNER600_AB"]

print("\n[regen] Rebuilding the three standard-star figures with clean titles ...")
std_h_path = DATA + "0053.Hiltner600.fits"
hdr = fits.getheader(std_h_path)
_am = hdr["AIRMASS"]
try: _am = float(_am)
except Exception: _am = 1.20
build_sensfunc(std_h_path, HILTNER600_AB,
               exptime=float(hdr["EXPTIME"]), airmass=_am,
               label="hiltner600",
               save=os.path.join(FIGS, "sensitivity_hiltner600.png"))
print("[regen] Done. Updated figures:")
for fn in ["sensitivity_hiltner600.png",
           "std_hiltner600_extract.png",
           "wavecal_hiltner600.png"]:
    print("  ", os.path.join(FIGS, fn))
