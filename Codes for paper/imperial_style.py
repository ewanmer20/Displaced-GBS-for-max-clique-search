"""Matplotlib style using the Imperial College fonts (Imperial Sans Text / Display).

Usage: `import imperial_style` at the top of a plotting script (after matplotlib is
imported, before any figure is created). Weights available through `fontweight=`:
extralight, light, regular (normal), medium, semibold, bold, extrabold.
"""
import os
import matplotlib as mpl
from matplotlib import font_manager as fm

FONT_DIRS = [
    r"C:\Windows\Fonts",
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Windows", "Fonts"),
]

for d in FONT_DIRS:
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.startswith("ImperialSans") and f.lower().endswith((".ttf", ".otf")):
                fm.fontManager.addfont(os.path.join(d, f))

TEXT = "Imperial Sans Text"
DISPLAY = "Imperial Sans Display"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [TEXT, DISPLAY, "DejaVu Sans"],
    # Math text (e.g. $\lambda_{max}$): Imperial Sans for letters, STIX Sans for missing glyphs
    "mathtext.fontset": "custom",
    "mathtext.rm": f"{TEXT}",
    "mathtext.it": f"{TEXT}:italic",
    "mathtext.bf": f"{TEXT}:bold",
    "mathtext.fallback": "stixsans",
    "svg.fonttype": "none",   # keep text as editable <text> (needs the font installed)
    "pdf.fonttype": 42,       # embed TrueType fonts in PDFs
    "axes.unicode_minus": False,
})
