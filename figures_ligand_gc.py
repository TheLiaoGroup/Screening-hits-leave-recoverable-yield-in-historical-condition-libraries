"""Figure 5: exact ligand-spread expectation and out-of-fold history-selected panels."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from figure_style_gc import SMALL, align_panels, configure_fonts, panel_title
from presentation_gc import array_label

HERE = Path(__file__).resolve().parent
OUT, FIG = HERE / "out", HERE / "figures"
BLUE, DARK, GREY, TRIANGLE = "#0072B2", "#003f66", "#999999", "#555555"
W, H = 6.5, 3.85


def _axis(fig, x, y, width, height):
    return fig.add_axes([x / W, y / H, width / W, height / H])


def build(write_outputs=False):
    configure_fonts()
    recipes = pd.read_csv(OUT / "review_axes_recipes.csv")
    curves = pd.read_csv(OUT / "review_curve24.csv")
    scores = recipes.pivot(index="screen", columns="method", values="retention")
    scores = scores.sort_values("greedy6", ascending=False)
    assert len(scores) == 12
    below_random = scores.ligand_uniform6 < scores.random6
    assert int(below_random.sum()) == 8
    fig = plt.figure(figsize=(W, H), dpi=150)
    a = _axis(fig, 1.53, 0.86, 2.35, 2.49)
    b = _axis(fig, 4.75, 0.86, 1.62, 2.49)
    rules = [("ligand_uniform6", "ligand spread", TRIANGLE, "^", "white", "none"),
             ("random6", "random", GREY, "o", GREY, ":"),
             ("greedy6", "history-ranked", DARK, "o", DARK, "-"),
             ("topmean6", "historical mean", BLUE, "s", "white", (0, (3, 2)))]
    for i, (screen, row) in enumerate(scores.iterrows()):
        values = row[[rule[0] for rule in rules]]
        a.hlines(i, values.min() * 100, values.max() * 100,
                 lw=1.0, color="#D7D7D7", zorder=1)
        for method, label, color, marker, fill, style in rules:
            a.plot(row[method] * 100, i, marker=marker, linestyle="none",
                   ms=4.6, mec=color, mfc=fill, mew=1.0, zorder=3)
    a.set_yticks(np.arange(len(scores)))
    a.set_yticklabels([array_label(screen) for screen in scores.index])
    for text, screen in zip(a.get_yticklabels(), scores.index):
        text.set_fontweight("bold" if below_random.loc[screen] else "normal")
    a.set_ylim(len(scores) - 0.5, -0.5)
    a.set_xlim(0, 102)
    a.set_xticks([0, 25, 50, 75, 100])
    a.set_xlabel("library-best yield found by six (%)")
    a.tick_params(length=2)
    panel_title(fig, a, "a", "Ligand spread trails random in 8/12 libraries", row=0, col=0)
    handles = [Line2D([], [], marker=marker, mfc=fill, mec=color, mew=1.0,
                      color=color, linestyle=style, lw=1.1, ms=4.6, label=label)
               for method, label, color, marker, fill, style in rules]
    legend = fig.legend(handles=handles, frameon=False, ncol=2, loc="upper left",
                        bbox_to_anchor=(0.63 / W, 0.40 / H), handletextpad=0.4,
                        columnspacing=1.3, borderaxespad=0)
    median = curves.groupby("k")[["retention_greedy", "retention_meanyield", "retention_random"]].median()
    specs = [("retention_greedy", DARK, "-", 1.4, "o", DARK),
             ("retention_meanyield", BLUE, (0, (3, 2)), 1.1, "s", "white"),
             ("retention_random", GREY, ":", 1.3, "o", GREY)]
    for column, color, style, width, marker, fill in specs:
        b.plot(median.index, 100 * median[column], color=color, ls=style, lw=width)
    point_labels = []
    for (column, color, style, width, marker, fill), offset in zip(specs, (8, -7, -3)):
        value = 100 * median.loc[6, column]
        b.plot(6, value, marker=marker, mfc=fill, mec=color, mew=0.8,
               linestyle="none", ms=3, zorder=4)
        dx = -5 if column == "retention_greedy" else 5
        text = b.annotate("%.0f%%" % value, (6, value), xytext=(dx, offset),
                          textcoords="offset points", fontsize=SMALL, color=color,
                          ha="right" if dx < 0 else "left", va="bottom" if offset > 0 else "top")
        point_labels.append((column, (6, value), text))
    b.axvline(6, lw=0.6, color="#BBBBBB", zorder=0)
    b.set_xlim(1, 24)
    b.set_ylim(0, 102)
    b.set_xticks([1, 6, 12, 24])
    b.set_yticks([0, 50, 100])
    b.set_xlabel("conditions")
    b.set_ylabel("library-best yield (%)")
    b.tick_params(length=2)
    b.text(0.04, 0.04, "median of 18 libraries", transform=b.transAxes,
           fontsize=SMALL, va="bottom", color="#444444")
    panel_title(fig, b, "b", "Six history-ranked conditions\nfind 85%", row=0, col=1)
    align_panels(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    fig._fig5_six_condition_label_audit = []
    for column, point, text in point_labels:
        box = text.get_window_extent(renderer)
        assert not any(box.overlaps(other) for other in boxes), "Figure 5 six-condition labels overlap"
        assert not any(line.get_path().transformed(line.get_transform()).intersects_bbox(box, filled=False)
                       for line in b.lines if len(line.get_xdata()) > 5), "Figure 5 label touches a curve"
        boxes.append(box)
        fig._fig5_six_condition_label_audit.append({"method": column, "point": point, "text": text})
    top_titles = [p["title"].get_window_extent(renderer).y1 for p in fig._panels]
    a_ink = a.get_tightbbox(renderer)
    b_ink = b.get_tightbbox(renderer)
    legend_box = legend.get_window_extent(renderer)
    scale = 300 / fig.dpi
    errors = {
        "a_b_title_top_difference_px": float(max(top_titles) - min(top_titles)),
        "panel_gap_px_at_300dpi": float(b_ink.x0 - a_ink.x1) * scale,
        "legend_to_a_axis_ink_gap_px_at_300dpi": float(a_ink.y0 - legend_box.y1) * scale,
        "legend_bottom_margin_px_at_300dpi": float(legend_box.y0) * scale,
        "library_row_pitch_px_at_300dpi": float(a.bbox.height / len(scores)) * scale,
        "a_b_plot_height_difference_px_at_300dpi": float(abs(a.bbox.height - b.bbox.height)) * scale,
    }
    assert errors["a_b_title_top_difference_px"] <= 0.5
    assert errors["panel_gap_px_at_300dpi"] > 10
    assert errors["legend_to_a_axis_ink_gap_px_at_300dpi"] > 10
    assert errors["legend_bottom_margin_px_at_300dpi"] > 0
    fig._fig5_layout_audit = errors
    fig._fig5_method_data = scores.copy()
    fig._fig5_curves = median.copy()
    if write_outputs:
        write(fig)
    return fig


def write(fig):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "Figure5_rules.pdf")
    fig.savefig(FIG / "Figure5_rules.png", dpi=300)


def main():
    fig = build(write_outputs=True)
    print(json.dumps(fig._fig5_layout_audit, indent=2))
    plt.close(fig)


if __name__ == "__main__":
    main()
