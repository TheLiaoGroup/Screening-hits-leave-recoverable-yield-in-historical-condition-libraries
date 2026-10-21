"""Shared print typography and panel layout for every figure of the paper.

Every figure is drawn at its final printed width and embedded in Word at 100%, so the sizes below
are the printed sizes. Two sizes only, both Arial:
  LARGE 9 pt  panel letters (bold), panel titles and axis labels;
  SMALL 8 pt  tick labels, legends and in-plot notes.
Chemical structures (ChemDraw, ACS Document 1996 settings) are placed at STRUCTURE_SCALE, so their
atom labels print at LARGE as well (subscripts at 75% of that, the ChemDraw convention).

Panel letters and titles are placed by align_panels() from the rendered ink, not from nominal axes
positions: in each column the letters stand at the leftmost ink of that column's panels (tick labels
and axis titles included), every title starts the same distance right of its letter, and the
letters and titles of one row share their top edge.
"""
import os

from matplotlib import font_manager, pyplot as plt

# A fixed PDF creation date (2026-10-18 UTC), so a rebuilt figure is byte-identical to the deposited one.
os.environ.setdefault("SOURCE_DATE_EPOCH", "1792281600")

LARGE = 9.0
SMALL = 8.0
STRUCTURE_SCALE = LARGE / 10.0   # ChemDraw ACS 1996 labels are 10 pt at 100%
TITLE_GAP_PT = 5.0               # from the right edge of the widest letter to the title
TITLE_PAD_PT = 6.0               # from the top of a row's panel ink to the bottom of its titles


def configure_fonts():
    for weight, style in (("normal", "normal"), ("bold", "normal"), ("normal", "italic")):
        font_manager.findfont(font_manager.FontProperties(family="Arial", weight=weight, style=style),
                              fallback_to_default=False)
    plt.rcParams.update({"font.family": "Arial", "font.size": SMALL,
                         "axes.titlesize": LARGE, "axes.labelsize": LARGE,
                         "xtick.labelsize": SMALL, "ytick.labelsize": SMALL,
                         "legend.fontsize": SMALL, "legend.title_fontsize": SMALL,
                         "figure.titlesize": LARGE, "figure.labelsize": LARGE,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6, "pdf.fonttype": 42,
                         "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                         "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})


def panel_title(fig, ax, letter, title, row, col):
    """Register an axes panel; align_panels() places its letter and title."""
    ax.set_title(title, loc="left", fontsize=LARGE, y=1.0)   # y given: matplotlib stops moving it
    return add_panel(fig, letter, ax._left_title, [ax], row, col)


def add_panel(fig, letter, title, members, row, col):
    """Register a panel made of several artists (axes or texts) with a title Text already created."""
    text = fig.text(0, 0, letter, fontsize=LARGE, fontweight="bold", ha="left", va="top")
    title.set_fontsize(LARGE)
    fig._panel_letters = getattr(fig, "_panel_letters", []) + [(text, letter)]
    fig._panels = getattr(fig, "_panels", []) + [
        {"letter": text, "title": title, "members": members, "row": row, "col": col}]
    return text


def _ink(artist, renderer):
    if hasattr(artist, "get_tightbbox"):
        box = artist.get_tightbbox(renderer)
        if box is not None:
            return box
    return artist.get_window_extent(renderer)


def align_panels(fig):
    """Place every registered letter and title from the rendered ink (see the module docstring)."""
    panels = getattr(fig, "_panels", [])
    if not panels:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    titles = [p["title"] for p in panels]
    for t in titles:
        t.set_visible(False)
    for p in panels:
        boxes = [_ink(m, renderer) for m in p["members"]]
        p["x0"] = min(b.x0 for b in boxes)
        p["y1"] = max(b.y1 for b in boxes)
    for t in titles:
        t.set_visible(True)
    px = fig.dpi / 72
    column_x = {}
    for p in panels:
        column_x[p["col"]] = min(column_x.get(p["col"], p["x0"]), p["x0"])
    widest = max(p["letter"].get_window_extent(renderer).width for p in panels)
    title_heights = {}
    for p in panels:
        h = p["title"].get_window_extent(renderer).height
        title_heights[p["row"]] = max(title_heights.get(p["row"], 0), h)
    row_top = {}
    for p in panels:
        top = p["y1"] + TITLE_PAD_PT * px + title_heights[p["row"]]
        row_top[p["row"]] = max(row_top.get(p["row"], top), top)
    to_fig = fig.transFigure.inverted()
    for p in panels:
        x = column_x[p["col"]]
        y = row_top[p["row"]]
        p["letter"].set_position(to_fig.transform((x, y)))
        p["letter"].set_verticalalignment("top")
        title = p["title"]
        title.set_transform(fig.transFigure)
        title.set_position(to_fig.transform((x + widest + TITLE_GAP_PT * px, y)))
        title.set_horizontalalignment("left")
        title.set_verticalalignment("top")
    fig.canvas.draw()
    outside = [p["title"].get_text() for p in panels
               if p["title"].get_window_extent(renderer).y1 > fig.bbox.height + 0.5
               or p["letter"].get_window_extent(renderer).x0 < -0.5]
    assert not outside, "panel titles leave the canvas; give the figure more top margin: %s" % outside
