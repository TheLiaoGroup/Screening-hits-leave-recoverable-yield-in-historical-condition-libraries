"""Shared print fonts and figure-coordinate panel labels."""
from matplotlib import font_manager, pyplot as plt


def configure_fonts():
    for weight, style in (("normal", "normal"), ("bold", "normal"), ("normal", "italic")):
        font_manager.findfont(font_manager.FontProperties(family="Arial", weight=weight, style=style),
                              fallback_to_default=False)
    plt.rcParams.update({"font.family": "Arial", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6, "pdf.fonttype": 42})


def panel_title(fig, ax, letter, title, fontsize=9.5, letter_size=10.5):
    ax.set_title(title, loc="left", fontsize=fontsize, pad=9)
    box = ax.get_position()
    line_offset = title.count("\n") * fontsize * 1.2 / 72 / fig.get_figheight()
    text = fig.text(box.x0 - 0.045, box.y1 + 0.022 + line_offset, letter,
                    fontsize=letter_size, fontweight="bold", va="baseline")
    fig._panel_letters = getattr(fig, "_panel_letters", []) + [(text, letter)]
    return text
