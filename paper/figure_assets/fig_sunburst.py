#!/usr/bin/env python3
"""Sunburst of the LinguaGapAtlas corpus: all resources in the centre, resource
type in the inner ring, task class within each type in the outer ring, every
ring sized by its number of resources.

Writes fig_sunburst.pdf (vector, for the paper) and fig_sunburst_preview.png
next to this script:

    python paper/figure_assets/fig_sunburst.py

Text is set in Linux Biolinum, the sans face of the acmart template, through
pdflatex and the libertine package, so the figure matches the paper's type.
Without pdflatex or libertine (or with --no-latex) it falls back to DejaVu Sans.
The figure is drawn at its printed size, 6 in wide (the height follows the
drawing), so the font sizes are the printed ones: nothing below 7.5 pt.

The counts are the consensus labels of the released corpus (data/corpus.csv,
coder = consensus-5rounds). They are fixed here, not recomputed.
"""
import argparse
import math
import shutil
import subprocess
from pathlib import Path

import matplotlib

HERE = Path(__file__).resolve().parent

ROOT = "Multilingual AI Resources"
# (task class, resources), in the order of the consensus list.
GROUPS = [
    ("Benchmarks", [("QA", 13), ("Other", 9), ("Syntax", 9), ("NLU", 8), ("Summarization", 4),
                    ("Reasoning", 3), ("Multimodal", 3), ("Generation", 2), ("Math", 2), ("Code", 2),
                    ("RAG", 1), ("Long Context", 1), ("Embeddings", 1)]),
    ("Datasets", [("Pre-training", 21), ("Translation", 7), ("Instruction", 4), ("QA", 4),
                  ("NLU", 1), ("Syntax", 1)]),
]

# Colours: a grey centre, a blue and a green resource type, and steel, slate and
# periwinkle shades of each for its classes, alternating light and dark so that
# neighbours stay apart. Labels are white or ink, whichever contrasts with the fill.
ROOT_FILL = "#707070"
GROUP_FILL = {"Benchmarks": "#4e71be", "Datasets": "#567e2f"}
CLASS_FILL = {
    "Benchmarks": ["#bdcfea", "#b6bfc9", "#7a8fd0", "#9abfe5", "#546a8c", "#84b3de", "#5d6eb3",
                   "#bdcbe5", "#749bcc", "#c0d3e8", "#7084a8", "#b5c4e6", "#8195d3"],
    "Datasets": ["#c9dcb2", "#a4c483", "#5d7748", "#c1d3a1", "#8db177", "#c7d8b5"],
}
SURFACE = "#ffffff"
INK = "#1d1d1b"

# Geometry, in units of the outer radius: the centre disc and two rings.
R_ROOT, R_GROUP, R_CLASS = 0.26, 0.50, 1.0
GAP_PT = 1.0                           # white border between segments, in points
PAD = 0.014                            # space kept between a label and its segment's edges, units
WIDTH_IN = 6.0                         # printed width; the height follows the drawing
MARGIN_IN = 0.06
UNIT_IN = 1.95                         # inches per unit (outer radius)
FS_CLASS = 7.5                         # pt
FS_GROUP = 9.0
FS_ROOT = 9.0
LINESPACING = 1.1


def latex_ok():
    """pdflatex with the libertine and pgf packages (pgf typesets the figure)."""
    if not (shutil.which("pdflatex") and shutil.which("kpsewhich")):
        return False
    found = subprocess.run(["kpsewhich", "libertine.sty", "pgf.sty"], capture_output=True, text=True).stdout
    return len(found.split()) == 2


def setup(use_latex):
    if use_latex:
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
            # \pagestyle{empty}: no page number below the figure's page
            "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage{libertine}\pagestyle{empty}",
            "font.family": "sans-serif"})
    else:
        matplotlib.use("Agg")
        matplotlib.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})


def luminance(hex_colour):
    rgb = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def label_ink(fill):
    """White or ink, whichever has the higher contrast ratio with the fill."""
    on_white = (1.05) / (luminance(fill) + 0.05)
    on_ink = (luminance(fill) + 0.05) / (luminance(INK) + 0.05)
    return SURFACE if on_white >= on_ink else INK


class Canvas:
    """The axes, in units of the outer radius, and text measurement in those units."""

    def __init__(self):
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties
        self.FontProperties = FontProperties
        self.fig = plt.figure(figsize=(WIDTH_IN, WIDTH_IN))
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_axis_off()
        self.limits(0.0, 0.0, WIDTH_IN)
        self.renderer = self.fig.canvas.get_renderer()

    def limits(self, cx, cy, height_in):
        """Axes limits centred on (cx, cy) at UNIT_IN inches per unit, for a figure
        WIDTH_IN wide and height_in tall."""
        self.fig.set_size_inches(WIDTH_IN, height_in)
        w, h = WIDTH_IN / UNIT_IN, height_in / UNIT_IN
        self.ax.set_xlim(cx - w / 2, cx + w / 2)
        self.ax.set_ylim(cy - h / 2, cy + h / 2)

    def fit(self):
        """Centre the drawing on the page and make the page as tall as the drawing
        plus the margin; the scale, and so every font size, stays as drawn."""
        to_data = self.ax.transData.inverted()
        x0, y0, x1, y1 = -R_CLASS, -R_CLASS, R_CLASS, R_CLASS
        for t in self.ax.texts:
            (a, b), (c, d) = to_data.transform(t.get_window_extent(self.renderer).get_points())
            x0, y0, x1, y1 = min(x0, a), min(y0, b), max(x1, c), max(y1, d)
        need = (x1 - x0) * UNIT_IN + 2 * MARGIN_IN
        if need > WIDTH_IN + 1e-6:
            print(f"warning: the drawing is {need:.2f} in wide, more than the {WIDTH_IN} in page")
        self.limits((x0 + x1) / 2, (y0 + y1) / 2, (y1 - y0) * UNIT_IN + 2 * MARGIN_IN)
        return (x1 - x0) * UNIT_IN, (y1 - y0) * UNIT_IN

    def size(self, s, fs):
        """(width, height) of a bold label, in units: the widest line, and the
        lines' full height from the lowest descender to the highest ascender."""
        prop = self.FontProperties(size=fs, weight="bold")
        to_units = 1 / (self.fig.dpi * UNIT_IN)
        lines = s.split("\n")
        w = max(self.renderer.get_text_width_height_descent(line, prop, ismath=False)[0] for line in lines)
        h = self.renderer.get_text_width_height_descent("(Ag", prop, ismath=False)[1]
        h += (len(lines) - 1) * LINESPACING * fs * 1.2 * self.fig.dpi / 72
        return w * to_units, h * to_units

    def text(self, x, y, s, fs, color, rotation=0.0, ha="center"):
        self.ax.text(x, y, s, fontsize=fs, fontweight="bold", color=color, rotation=rotation, ha=ha,
                     va="center", rotation_mode="anchor", linespacing=LINESPACING, multialignment="center")


def polar(r, deg):
    a = math.radians(deg)
    return r * math.cos(a), r * math.sin(a)


def wedges():
    """Angles for every class, anticlockwise from 3 o'clock: the benchmarks fill
    the top and the datasets follow, as in a standard sunburst."""
    total = sum(n for _, kids in GROUPS for _, n in kids)
    per = 360 / total
    out, a = [], 0.0
    for group, kids in GROUPS:
        for i, (name, n) in enumerate(kids):
            out.append({"group": group, "name": name, "n": n, "lo": a, "hi": a + n * per,
                        "fill": CLASS_FILL[group][i]})
            a += n * per
    return out


def orientation(kind, deg):
    """Text rotation, in degrees, for a label at angle deg: horizontal, radial
    (along the radius) or tangential (across it), always turned to read upright."""
    if kind == "horizontal":
        return 0.0
    if kind == "radial":
        return deg if math.cos(math.radians(deg)) >= 0 else deg - 180
    return deg - 90 if math.sin(math.radians(deg)) >= 0 else deg + 90


def fits(cx, cy, w, h, rot, r_in, r_out, lo, hi):
    """Whether a w x h box centred on (cx, cy) and turned by rot degrees lies
    inside the ring segment, PAD from every edge (tested along its outline)."""
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    for k in range(40):
        t = k / 10
        u, v = [(t - 0.5, -0.5), (0.5, t - 1.5), (2.5 - t, 0.5), (-0.5, 3.5 - t)][k // 10]
        x, y = cx + u * w * c - v * h * s, cy + u * w * s + v * h * c
        r = math.hypot(x, y)
        rel = (math.degrees(math.atan2(y, x)) - lo) % 360
        margin = math.degrees(PAD / r)
        if not (r_in + PAD <= r <= r_out - PAD and margin <= rel <= hi - lo - margin):
            return False
    return True


def place(cv, text_lines, fs, r_in, r_out, lo, hi):
    """The first layout of a label that fits inside its segment: one line before
    two, horizontal before radial before tangential, and for each the fitting
    position nearest the segment's middle (within its central 70%).
    Returns (x, y, rotation, text) or None."""
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    r_mid = (r_in + r_out) / 2
    spots = sorted((abs(dr) / (r_out - r_in) + abs(da) / half, dr, da)
                   for dr in [0.0] + [s * k * 0.02 for k in range(1, 10) for s in (1, -1)]
                   for da in [0.0] + [s * k for k in range(1, int(0.7 * half) + 1) for s in (1, -1)])
    for text in text_lines:
        w, h = cv.size(text, fs)
        for kind in ("horizontal", "radial", "tangential"):
            for _, dr, da in spots:
                rot = orientation(kind, mid + da)
                x, y = polar(r_mid + dr, mid + da)
                if fits(x, y, w, h, rot, r_in, r_out, lo, hi):
                    return x, y, rot, text
    return None


def draw(cv):
    from matplotlib.patches import Circle, Wedge
    ax = cv.ax
    classes = wedges()
    total = sum(c["n"] for c in classes)

    # Centre: every resource.
    ax.add_patch(Circle((0, 0), R_ROOT, facecolor=ROOT_FILL, edgecolor=SURFACE, linewidth=GAP_PT))
    cv.text(0, 0, f"{ROOT.replace('AI ', 'AI' + chr(10))} ({total})", FS_ROOT, label_ink(ROOT_FILL))

    # Inner ring: resource type.
    for group, kids in GROUPS:
        mine = [c for c in classes if c["group"] == group]
        lo, hi = mine[0]["lo"], mine[-1]["hi"]
        fill = GROUP_FILL[group]
        ax.add_patch(Wedge((0, 0), R_GROUP, lo, hi, width=R_GROUP - R_ROOT, facecolor=fill,
                           edgecolor=SURFACE, linewidth=GAP_PT))
        n = sum(k for _, k in kids)
        spot = place(cv, [f"{group} ({n})", f"{group}\n({n})"], FS_GROUP, R_ROOT, R_GROUP, lo, hi)
        if spot is None:
            raise SystemExit(f"the label of {group} does not fit its ring segment")
        x, y, rot, text = spot
        cv.text(x, y, text, FS_GROUP, label_ink(fill), rot)

    # Outer ring: task class. A label that fits nowhere inside its segment is set
    # just outside it, along the radius; neighbouring outside labels are fanned
    # apart just enough to keep a line's height between them.
    outside = []
    for c in classes:
        ax.add_patch(Wedge((0, 0), R_CLASS, c["lo"], c["hi"], width=R_CLASS - R_GROUP, facecolor=c["fill"],
                           edgecolor=SURFACE, linewidth=GAP_PT))
        label = f"{c['name']} ({c['n']})"
        spot = place(cv, [label, f"{c['name']}\n({c['n']})"], FS_CLASS, R_GROUP, R_CLASS, c["lo"], c["hi"])
        if spot:
            x, y, rot, text = spot
            cv.text(x, y, text, FS_CLASS, label_ink(c["fill"]), rot)
        else:
            outside.append((label, (c["lo"] + c["hi"]) / 2))
    r_out = R_CLASS + 0.025
    min_gap = math.degrees(1.3 * FS_CLASS / 72 / UNIT_IN / r_out)
    angles = fan([deg for _, deg in outside], min_gap)
    for (label, _), deg in zip(outside, angles):
        x, y = polar(r_out, deg)
        right = math.cos(math.radians(deg)) >= 0
        cv.text(x, y, label, FS_CLASS, INK, orientation("radial", deg), ha="left" if right else "right")


def fan(angles, gap):
    """The angles moved apart, in order, until neighbours are at least `gap`
    degrees apart, each run of crowded labels staying centred where it was."""
    out = list(angles)
    for _ in range(200):
        moved = False
        for i in range(len(out) - 1):
            short = gap - (out[i + 1] - out[i])
            if 1e-9 < short and out[i + 1] - out[i] < 3 * gap:     # only near neighbours
                out[i] -= short / 2
                out[i + 1] += short / 2
                moved = True
        if not moved:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-latex", action="store_true", help="use DejaVu Sans instead of Biolinum through LaTeX")
    args = ap.parse_args()
    use_latex = not args.no_latex and latex_ok()
    if not use_latex and not args.no_latex:
        print("pdflatex with the libertine and pgf packages not found: falling back to DejaVu Sans")
    setup(use_latex)
    cv = Canvas()
    draw(cv)
    width, height = cv.fit()
    pdf, png = HERE / "fig_sunburst.pdf", HERE / "fig_sunburst_preview.png"
    cv.fig.savefig(pdf, facecolor=SURFACE)
    try:
        cv.fig.savefig(png, dpi=300, facecolor=SURFACE)
    except RuntimeError as e:      # through LaTeX, the PNG is converted from the PDF
        raise SystemExit(f"wrote {pdf}, but not the PNG preview, which needs pdftocairo or ghostscript: {e}")
    from PIL import Image          # flatten onto white: the page's last pixel row is fractional
    with Image.open(png) as im:
        flat = Image.new("RGB", im.size, SURFACE)
        flat.paste(im, mask=im.getchannel("A") if "A" in im.getbands() else None)
    flat.save(png, dpi=(300, 300))
    print(f"wrote {pdf} and {png}: drawing {width:.2f} x {height:.2f} in on a "
          f"{WIDTH_IN:.1f} x {height + 2 * MARGIN_IN:.2f} in page")


if __name__ == "__main__":
    main()
