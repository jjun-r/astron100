"""Regenerate only figures/07_summary_table.png with audit-corrected
literature references. The numerical values for the science columns are
loaded verbatim from reduction/summary_diagnostics.txt — the file written
by reduce_fast.py — so the figure stays in sync with the run."""
import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAG = os.path.join(ROOT, "reduction", "summary_diagnostics.txt")
OUT  = os.path.join(ROOT, "figures",   "07_summary_table.png")

# --- load diagnostics summary ----------------------------------------------
def load_diagnostics(path):
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    blocks = re.split(r"\n(?=NGC )", text.strip())
    out = {}
    for blk in blocks:
        if not blk.startswith("NGC"):
            continue
        name = blk.split("\n", 1)[0].split(" (")[0].strip()
        d = {}
        for line in blk.splitlines()[1:]:
            m = re.match(r"\s*(\S+)\s*=\s*(.+)\s*$", line)
            if not m: continue
            k, v = m.group(1), m.group(2).strip()
            d[k] = v
        out[name] = d
    return out

diag = load_diagnostics(DIAG)
n2392 = diag["NGC 2392"]
n3242 = diag["NGC 3242"]

def fnum(s, fmt):
    try:
        return format(float(s), fmt)
    except Exception:
        return "n.d."

# --- assemble table ---------------------------------------------------------
rows = [
    ("Property",            "NGC 2392 (Eskimo)",                     "NGC 3242 (Ghost of Jupiter)",                "literature value (ref)"),
    ("Hα/Hβ (observed)",    fnum(n2392["Halpha_Hbeta"], ".2f"),      fnum(n3242["Halpha_Hbeta"], ".2f"),           "Case B = 2.86 (Storey & Hummer 1995)"),
    ("E(B–V) inferred",     fnum(n2392["EBV"],          ".2f"),      fnum(n3242["EBV"],          ".2f"),           "0.115 / 0.083 (Singh 2025; Pottasch 2008b)"),
    ("[O III] 5007 / Hβ",   fnum(n2392["OIII_Hb"],      ".1f"),      fnum(n3242["OIII_Hb"],      ".1f") + "*",     "10.4 / 12.80 (Singh 2025; Pottasch 2008b)"),
    ("He II 4686 / Hβ",     fnum(n2392["HeII_Hb"],      ".2f"),      fnum(n3242["HeII_Hb"],      ".2f"),           "0.46 / 0.28 (Singh 2025; Pottasch 2008b)"),
    ("[N II] 6583 / Hα",    fnum(n2392["NII_Ha"],       ".2f"),      fnum(n3242["NII_Ha"],       ".3f"),           "0.30 / 0.009 (Singh 2025; Pottasch 2008b)"),
    ("[S II] 6716/6731",    fnum(n2392["SII_ratio"],    ".2f"),      "n.d. (low-density)",                          "ratio range 1.45 → 0.45"),
    ("inferred n_e (cm⁻³)", fnum(n2392["ne"],           ".0f"),      "n.d. (doublet < S/N)",                        "~2940 / ~1900 (Singh 2025; Pottasch 2008b)"),
]

fig, ax = plt.subplots(figsize=(13, 4.8))
ax.axis("off")
table = ax.table(
    cellText=[r for r in rows[1:]],
    colLabels=rows[0],
    cellLoc="center", loc="center",
    colWidths=[0.18, 0.18, 0.22, 0.42],
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.6)
ax.set_title(
    "Comparative spectroscopic diagnostics — NGC 2392 vs NGC 3242\n"
    "[*] NGC 3242 5007 saturated; recovered as 2.98 × F(4959). "
    "Literature column updated 2026-05-09 (citation audit).",
    fontsize=11, pad=10,
)
plt.savefig(OUT, dpi=160, bbox_inches="tight")
plt.close()
print(f"Wrote {OUT}")
