#!/usr/bin/env python3
"""Sunburst of the LinguaGapAtlas corpus, in the style of the reference figure
"Taxonomy of Multilingual AI Resources": all resources in the centre, resource
type in the inner ring, task class within each type in the outer ring.

Writes fig_sunburst.pdf (vector, for the paper) and fig_sunburst_preview.png
next to this script, both cropped to the drawing:

    pip install plotly==5.24.1 kaleido==0.2.1 pypdf pillow
    python paper/figure_assets/fig_sunburst.py

The figure is a Plotly sunburst with Plotly's defaults, as in the reference:
Arial Bold labels (12 px, shrunk to fit thin segments), an 18 px bold title,
the first segment at 3 o'clock, segments sorted by size, and branch values in
Plotly's default "remainder" mode, which gives the reference its fan shape. The
colours are those of the reference, sampled from it. Classes it does not have
take shades from the same families.

The counts are the consensus labels of the released corpus (data/corpus.csv,
coder = consensus-5rounds). They are fixed here, not recomputed.
"""
from io import BytesIO
from pathlib import Path

import plotly.graph_objects as go

HERE = Path(__file__).resolve().parent

TITLE = "Taxonomy of Multilingual AI Resources"
ROOT = ("Multilingual AI<br>Resources", "#808080")
# (resource type, colour, [(task class, resources, colour)]), in the order of the consensus list
GROUPS = [
    ("Benchmarks", "#4d71bd", [
        ("QA", 13, "#bfd3eb"), ("Other", 9, "#c2d3e5"), ("Syntax", 9, "#b3c2e3"), ("NLU", 8, "#7287b2"),
        ("Summarization", 4, "#6a829c"), ("Reasoning", 3, "#95b6dc"), ("Multimodal", 3, "#829bd1"),
        ("Generation", 2, "#a9bfe3"), ("Math", 2, "#ccd7eb"), ("Code", 2, "#d7e3f1"), ("RAG", 1, "#e5f0f8"),
        ("Long Context", 1, "#8fa4c7"), ("Embeddings", 1, "#c9d6ee")]),
    ("Datasets", "#7daa55", [
        ("Pre-training", 21, "#c8ddb4"), ("Translation", 7, "#a3c384"), ("Instruction", 4, "#8ca676"),
        ("QA", 4, "#bcd3a5"), ("NLU", 1, "#99b586"), ("Syntax", 1, "#d6e5c8")]),
]
WIDTH, HEIGHT = 1070, 1036     # px, as the reference
PNG_SCALE = 2
MARGIN = 12                    # px kept around the drawing when cropping


def figure():
    total = sum(n for _, _, kids in GROUPS for _, n, _ in kids)
    root = f"{ROOT[0]} ({total})"
    ids, labels, parents, values, colours = [root], [root], [""], [total], [ROOT[1]]
    for group, colour, kids in GROUPS:
        g = f"{group} ({sum(n for _, n, _ in kids)})"
        ids.append(g), labels.append(g), parents.append(root), values.append(sum(n for _, n, _ in kids))
        colours.append(colour)
        for name, n, c in kids:
            ids.append(f"{g}/{name}"), labels.append(f"{name} ({n})"), parents.append(g), values.append(n)
            colours.append(c)
    fig = go.Figure(go.Sunburst(
        ids=ids, labels=labels, parents=parents, values=values, texttemplate="<b>%{label}</b>",
        marker=dict(colors=colours), leaf=dict(opacity=1),
        textfont=dict(family="Arial", color="white"), insidetextfont=dict(family="Arial", color="white")))
    fig.update_layout(
        width=WIDTH, height=HEIGHT, font=dict(family="Arial"),
        title=dict(text=f"<b>{TITLE}</b>", x=0.5, y=1 - 66 / HEIGHT, yanchor="middle",
                   font=dict(family="Arial", size=18, color="black")))
    return fig


def content_box(png, scale):
    """The drawing's bounding box in layout px (left, top, right, bottom), plus MARGIN."""
    from PIL import Image, ImageChops
    with Image.open(BytesIO(png)) as im:
        rgb = im.convert("RGB")
        box = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white")).getbbox()
    left, top, right, bottom = (v / scale for v in box)
    return (max(0, left - MARGIN), max(0, top - MARGIN), min(WIDTH, right + MARGIN), min(HEIGHT, bottom + MARGIN))


def main():
    fig = figure()
    pdf, png = HERE / "fig_sunburst.pdf", HERE / "fig_sunburst_preview.png"
    png_bytes = fig.to_image(format="png", scale=PNG_SCALE)
    left, top, right, bottom = content_box(png_bytes, PNG_SCALE)

    from PIL import Image
    with Image.open(BytesIO(png_bytes)) as im:
        im.convert("RGB").crop(tuple(round(v * PNG_SCALE) for v in (left, top, right, bottom))).save(png)

    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import RectangleObject
    page = PdfReader(BytesIO(fig.to_image(format="pdf"))).pages[0]
    pt = float(page.mediabox.width) / WIDTH        # PDF points per layout px
    height = float(page.mediabox.height)
    page.mediabox = page.cropbox = RectangleObject([left * pt, height - bottom * pt, right * pt, height - top * pt])
    writer = PdfWriter()
    writer.add_page(page)
    with open(pdf, "wb") as f:
        writer.write(f)
    print(f"wrote {pdf} ({(right - left) * pt / 72:.2f} x {(bottom - top) * pt / 72:.2f} in) and {png}")


if __name__ == "__main__":
    main()
