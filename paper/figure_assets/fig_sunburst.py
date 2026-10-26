#!/usr/bin/env python3
"""Sunburst of the LinguaGapAtlas corpus: resource type (inner ring) and task
class within each type (outer ring), each sized by its number of resources.

Writes fig_sunburst.pdf (vector, for the paper) and fig_sunburst_preview.png
next to this script:

    python paper/figure_assets/fig_sunburst.py

Text is set in Linux Biolinum, the sans face of the acmart template, through
pdflatex and the libertine package, so the figure matches the paper's type.
Without pdflatex or libertine (or with --no-latex) it falls back to DejaVu Sans.
The figure is drawn at its printed size, 6 in wide (the height follows the
drawing), so the font sizes are the printed ones: 8 pt labels, nothing below 7 pt.

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

# (task class, resources), largest first; "Other", the catch-all, last.
GROUPS = [
    ("Benchmarks", [("QA", 13), ("Syntax", 9), ("NLU", 8), ("Summarization", 4), ("Reasoning", 3),
                    ("Multimodal", 3), ("Generation", 2), ("Math", 2), ("Code", 2), ("RAG", 1),
                    ("Long Context", 1), ("Embeddings", 1), ("Other", 9)]),
    ("Datasets", [("Pre-training", 21), ("Translation", 7), ("Instruction", 4), ("QA", 4),
                  ("NLU", 1), ("Syntax", 1)]),
]
CENTRE = ("Multilingual AI", "Resources")

# Benchmarks take steps 550 (inner) and 150 (outer) of the blue ramp of the
# dataviz reference palette; datasets take the same OKLCH lightness and chroma at
# that palette's aqua-green hue (h = 162), so the two groups carry equal weight.
# The inner pair passes the palette validator: CVD dE 17.3, normal-vision dE 18.1.
INNER = {"Benchmarks": "#1c5cab", "Datasets": "#056f4b"}
OUTER = {"Benchmarks": "#b7d3f6", "Datasets": "#afddc5"}
OTHER = "#dde2e8"          # the catch-all class: the blue tint, nearly desaturated
SURFACE = "#ffffff"
INK = "#1d1d1b"            # labels on the light outer ring and outside it (11:1 on the tints)
INK_SOFT = "#5c5b57"
LEADER = "#a3a19b"

# Geometry, in units of the outer radius.
HOLE, INNER_OUT = 0.41, 0.615          # inner ring
OUTER_IN, OUTER_OUT = 0.64, 1.0        # outer ring
GAP_PT = 1.2                           # white gap between segments, in points
LABEL_X = 1.15                         # inner edge of the outside label columns
ELBOW_X = 1.10
WIDTH_IN = 6.0                         # printed width; the height follows the content
MARGIN_IN = 0.08
UNIT_IN = 1.72                         # inches per unit (outer radius)
FS = 8.0                               # label size, pt
FS_RING = 7.0                          # group labels on the inner ring, pt
FS_HERO = 22.0


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


class Canvas:
    """The axes, in units of the outer radius, and text measurement in those units."""

    def __init__(self, use_latex):
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties
        self.latex, self.FontProperties = use_latex, FontProperties
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
        x0, y0, x1, y1 = -OUTER_OUT, -OUTER_OUT, OUTER_OUT, OUTER_OUT
        for t in self.ax.texts:
            (a, b), (c, d) = to_data.transform(t.get_window_extent(self.renderer).get_points())
            x0, y0, x1, y1 = min(x0, a), min(y0, b), max(x1, c), max(y1, d)
        for line in self.ax.lines:
            xs, ys = line.get_data()
            x0, y0, x1, y1 = min(x0, *xs), min(y0, *ys), max(x1, *xs), max(y1, *ys)
        need = (x1 - x0) * UNIT_IN + 2 * MARGIN_IN
        if need > WIDTH_IN + 1e-6:
            print(f"warning: the drawing is {need:.2f} in wide, more than the {WIDTH_IN} in page")
        self.limits((x0 + x1) / 2, (y0 + y1) / 2, (y1 - y0) * UNIT_IN + 2 * MARGIN_IN)
        return (x1 - x0) * UNIT_IN, (y1 - y0) * UNIT_IN

    def caps(self, s):
        """Small capitals in Biolinum; plain upper case in the fallback font."""
        return rf"\textsc{{{s}}}" if self.latex else s.upper()

    def size(self, s, fs, bold=False):
        """(width, height, descent) of a string, in units."""
        prop = self.FontProperties(size=fs, weight="bold" if bold else "normal")
        w, h, d = self.renderer.get_text_width_height_descent(s, prop, ismath=False)
        px_per_unit = self.fig.dpi * UNIT_IN
        return w / px_per_unit, h / px_per_unit, d / px_per_unit

    def text(self, x, y, s, fs, bold=False, color=INK, **kw):
        return self.ax.text(x, y, s, fontsize=fs, fontweight="bold" if bold else "normal", color=color, **kw)


def polar(r, deg):
    a = math.radians(deg)
    return r * math.cos(a), r * math.sin(a)


def segments():
    """Angles for every wedge. Benchmarks fill the top, centred on 12 o'clock, and
    datasets the bottom; both start at the left end of their arc and run to the
    right (benchmarks clockwise, datasets anticlockwise), so the small classes of
    both meet at the right, where the outside labels go."""
    total = sum(n for _, kids in GROUPS for _, n in kids)
    per = 360 / total
    spans = {g: sum(n for _, n in kids) * per for g, kids in GROUPS}
    start = 90 + spans["Benchmarks"] / 2        # left end of the benchmark arc
    out = []
    for group, kids in GROUPS:
        sign = -1 if group == "Benchmarks" else 1
        a = start
        for name, n in kids:
            b = a + sign * n * per
            out.append({"group": group, "name": name, "n": n, "lo": min(a, b), "hi": max(a, b)})
            a = b
    return out, start, spans


def curved_label(cv, parts, r_mid, centre_deg, top, color):
    """Letter-spaced text set along an arc, reading left to right: along the top
    of the ring clockwise, along the bottom anticlockwise (letters stay upright).
    parts: [(text, bold)], each laid out letter by letter."""
    glyphs = []
    for i, (s, bold) in enumerate(parts):
        if i:
            glyphs.append((None, 0.55 * FS_RING / 72 / UNIT_IN, False))    # word gap
        for ch in s:
            if ch == " ":
                glyphs.append((None, 0.28 * FS_RING / 72 / UNIT_IN, False))
                continue
            t = cv.caps(ch) if ch.islower() else ch
            glyphs.append((t, cv.size(t, FS_RING, bold)[0], bold))
    track = 0.07 * FS_RING / 72 / UNIT_IN
    cap = cv.size("H", FS_RING)[1] - cv.size("H", FS_RING)[2]
    r = r_mid - cap / 2 if top else r_mid + cap / 2              # baseline radius
    total = sum(w for _, w, _ in glyphs) + track * (len(glyphs) - 1)
    pos = 0.0
    for t, w, bold in glyphs:
        offset = math.degrees((pos + w / 2 - total / 2) / r)
        deg = centre_deg - offset if top else centre_deg + offset
        if t is not None:
            x, y = polar(r, deg)
            cv.text(x, y, t, FS_RING, bold, color, ha="center", va="baseline",
                    rotation=deg - 90 if top else deg + 90, rotation_mode="anchor")
        pos += w + track


LINE_GAP = 0.012     # between the name and the count of an inside label, units


def inside_spot(cv, seg, pad=0.02):
    """Where the two-line label (name over count) fits inside the segment's
    outer-ring wedge, with `pad` to spare on every side: the fitting point
    nearest the middle of the wedge, searched outwards in angle and radius.
    None if it fits nowhere."""
    lines = [(seg["name"], False), (str(seg["n"]), True)]
    w = max(cv.size(s, FS, b)[0] for s, b in lines)
    h = sum(cv.size(s, FS, b)[1] for s, b in lines) + LINE_GAP

    def fits(r0, deg):
        cx, cy = polar(r0, deg)
        for dx in (-w / 2, w / 2):
            for dy in (-h / 2, h / 2):
                x, y = cx + dx, cy + dy
                r = math.hypot(x, y)
                rel = (math.degrees(math.atan2(y, x)) - seg["lo"]) % 360
                margin = math.degrees(pad / r)
                if not (OUTER_IN + pad <= r <= OUTER_OUT - pad and margin <= rel <= seg["hi"] - seg["lo"] - margin):
                    return False
        return True

    mid, half = (seg["lo"] + seg["hi"]) / 2, (seg["hi"] - seg["lo"]) / 2
    r_mid = (OUTER_IN + OUTER_OUT) / 2
    spots = [(abs(dr) / 0.18 + abs(da) / max(half, 1e-9), r_mid + dr, mid + da)
             for dr in (0, 0.02, -0.02, 0.04, -0.04, 0.06)
             for da in [0] + [s * k for k in range(1, int(half) + 1) for s in (-1, 1)]]
    for _, r0, deg in sorted(spots):
        if fits(r0, deg):
            return polar(r0, deg)
    return None


def spread(desired, gap, lo, hi):
    """Positions for labels in top-to-bottom order, at least `gap` apart and as
    close as possible to their desired heights."""
    ys = list(desired)
    for _ in range(500):
        moved = False
        for i in range(len(ys) - 1):
            overlap = gap - (ys[i] - ys[i + 1])
            if overlap > 1e-9:
                ys[i] += overlap / 2
                ys[i + 1] -= overlap / 2
                moved = True
        if ys and ys[0] > hi:
            ys = [y - (ys[0] - hi) for y in ys]
            moved = True
        if ys and ys[-1] < lo:
            ys = [y + (lo - ys[-1]) for y in ys]
            moved = True
        if not moved:
            break
    # Within each run of labels pushed together, space them evenly between the
    # run's ends, so a column reads with one rhythm.
    runs, start = [], 0
    for i in range(1, len(ys) + 1):
        if i == len(ys) or ys[i - 1] - ys[i] > 1.5 * gap:
            runs.append((start, i))
            start = i
    for a, b in runs:
        if b - a > 2:
            step = (ys[a] - ys[b - 1]) / (b - a - 1)
            ys[a:b] = [ys[a] - k * step for k in range(b - a)]
    return ys


def draw(cv):
    from matplotlib.patches import Wedge
    ax = cv.ax
    segs, start, spans = segments()

    # Inner ring: the two resource types, labelled along the ring.
    for group, _ in GROUPS:
        lo, hi = (start - spans[group], start) if group == "Benchmarks" else (start, start + spans[group])
        ax.add_patch(Wedge((0, 0), INNER_OUT, lo, hi, width=INNER_OUT - HOLE, facecolor=INNER[group],
                           edgecolor=SURFACE, linewidth=GAP_PT))
        count = sum(n for _, n in dict(GROUPS)[group])
        curved_label(cv, [(group, False), (str(count), True)], (HOLE + INNER_OUT) / 2,
                     (lo + hi) / 2, top=group == "Benchmarks", color=SURFACE)

    # Outer ring: the task classes.
    outside = []
    for s in segs:
        fill = OTHER if s["name"] == "Other" else OUTER[s["group"]]
        ax.add_patch(Wedge((0, 0), OUTER_OUT, s["lo"], s["hi"], width=OUTER_OUT - OUTER_IN, facecolor=fill,
                           edgecolor=SURFACE, linewidth=GAP_PT))
        spot = inside_spot(cv, s)
        if spot:
            cx, cy = spot
            h_name, h_num = cv.size(s["name"], FS)[1], cv.size(str(s["n"]), FS, True)[1]
            block = h_name + LINE_GAP + h_num
            cv.text(cx, cy + block / 2 - h_name / 2, s["name"], FS, ha="center", va="center_baseline")
            cv.text(cx, cy - block / 2 + h_num / 2, str(s["n"]), FS, True, ha="center", va="center_baseline")
        else:
            outside.append(s)

    # Outside labels, in a column on each side, joined to their wedges by leader lines.
    top, bottom = OUTER_OUT + 0.1, -OUTER_OUT - 0.1      # keep the columns level with the ring
    gap = 1.32 * FS / 72 / UNIT_IN
    for side in (1, -1):
        group = [s for s in outside if side * math.cos(math.radians((s["lo"] + s["hi"]) / 2)) >= 0]
        group.sort(key=lambda s: -math.sin(math.radians((s["lo"] + s["hi"]) / 2)))
        mids = [(s["lo"] + s["hi"]) / 2 for s in group]
        ys = spread([math.sin(math.radians(m)) * 1.0 for m in mids], gap, bottom, top)
        for s, mid, y in zip(group, mids, ys):
            # leader: from just off the wedge, out radially, then across to the label
            (px, py), (qx, qy) = polar(OUTER_OUT + 0.012, mid), polar(OUTER_OUT + 0.05, mid)
            ax.plot([px, qx, side * ELBOW_X, side * (LABEL_X - 0.015)], [py, qy, y, y], color=LEADER,
                    linewidth=0.5, solid_joinstyle="round", solid_capstyle="round")
            w_name = cv.size(s["name"], FS)[0]
            space = 0.3 * FS / 72 / UNIT_IN
            if side == 1:
                cv.text(LABEL_X, y, s["name"], FS, ha="left", va="center_baseline")
                cv.text(LABEL_X + w_name + space, y, str(s["n"]), FS, True, ha="left", va="center_baseline")
            else:
                w_num = cv.size(str(s["n"]), FS, True)[0]
                cv.text(-LABEL_X - w_num - space, y, s["name"], FS, ha="right", va="center_baseline")
                cv.text(-LABEL_X, y, str(s["n"]), FS, True, ha="right", va="center_baseline")

    # Centre: the total, then what it counts.
    total = sum(s["n"] for s in segs)
    hero_h = cv.size(str(total), FS_HERO, True)[1] - cv.size(str(total), FS_HERO, True)[2]
    line = 1.25 * FS / 72 / UNIT_IN
    block = hero_h + 0.05 + 2 * line
    y_hero = block / 2 - hero_h
    cv.text(0, y_hero, str(total), FS_HERO, True, ha="center", va="baseline")
    for i, words in enumerate(CENTRE):
        cv.text(0, y_hero - 0.05 - (i + 0.8) * line, cv.caps(words), FS - 0.5, color=INK_SOFT,
                ha="center", va="baseline")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-latex", action="store_true", help="use DejaVu Sans instead of Biolinum through LaTeX")
    args = ap.parse_args()
    use_latex = not args.no_latex and latex_ok()
    if not use_latex and not args.no_latex:
        print("pdflatex with the libertine package not found: falling back to DejaVu Sans")
    setup(use_latex)
    cv = Canvas(use_latex)
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
