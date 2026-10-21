"""TOC: the measured Suzuki example and a history-directed second screen."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd

import figure1_gc

HERE = Path(__file__).resolve().parent
OUT, FIG = HERE / "out", HERE / "figures"
W, H, DPI = 3.25, 1.75, 300
BLUE, LIGHT, GREY, GAP = "#0072B2", "#56B4E9", "#999999", "#D55E00"
WORDS = {4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
ROLES = ["initial six (Greedy)", "six added from history (Greedy)", "rest of library"]


def example():
    """Read the deposited example and the complete below-80% cohort."""
    E = json.loads((OUT / "example_pair.json").read_text(encoding="utf-8"))
    D = pd.read_csv(OUT / "example_pair.csv")
    assert set(D.role) == set(ROLES) and len(D) == E["measured_conditions"]
    first = D[D.role == ROLES[0]]
    assert np.isclose(first["yield"].max(), E["m6"]) and np.isclose(D["yield"].max(), E["b"])
    twelve = D[D.greedy_order <= 12]
    assert len(first) == 6 and len(twelve) == 12
    assert np.isclose(twelve["yield"].max(), E["m12"], rtol=0, atol=1e-12)
    assert np.isclose(E["m12"], E["b"], rtol=0, atol=1e-12), \
        "The TOC's full blue recovery requires the measured m12 to equal the library best"
    summary = json.loads((OUT / "posthit_summary.json").read_text(encoding="utf-8"))["below_ceiling"]
    return E, D, {"hits": summary["pairs"], "severe": summary["severe_k6"],
                  "one_in": int(round(summary["pairs"] / summary["severe_k6"]))}


def scheme(fig, E, x, top, width):
    """Keep the native ChemDraw chemistry, scaled to the fixed TOC width."""
    from PIL import Image
    expected, master = figure1_gc.example_master(E)
    if master is not None:
        import pymupdf
        with pymupdf.open(master) as document:
            page = document[0]
            scale = width / (page.rect.width / 72)
            zoom = DPI * scale / 72
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=True)
            raster = Image.frombytes("RGBA", [pix.width, pix.height], pix.samples)
    else:
        from types import SimpleNamespace
        from chemical_schemes_gc import scheme_drawing
        row = SimpleNamespace(reactant1_smiles=expected[0], reactant2_smiles=expected[1],
                              product_smiles=expected[2])
        raster = scheme_drawing(row)[0]
        raster = raster.resize((int(width * DPI), int(width * DPI * raster.height / raster.width)))
    h = raster.height / DPI
    ax = fig.add_axes([x / W, (top - h) / H, width / W, h / H]); ax.axis("off")
    image = ax.imshow(raster, interpolation="nearest")
    fig._toc_master = master
    fig._vector_panels = [(image, master)] if master else []
    return ax


def graphic():
    plt.rcParams.update({"font.family": "Arial", "font.size": 8})
    E, D, cohort = example()
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    scheme_ax = scheme(fig, E, .08, 1.68, W-.16)
    endpoints=(E["m6"],E["b"])
    axes=[]
    artists={}
    for name,y,color in (("initial",.98,GAP),("recovery",.60,LIGHT)):
        ax=fig.add_axes([.25/W,(y-.10)/H,2.75/W,.20/H])
        ax.set(xlim=(endpoints[0]-.025,endpoints[1]+.025),ylim=(-.5,.5))
        ax.axis("off")
        stop=E["b"] if name=="initial" else E["m12"]
        segment,=ax.plot([E["m6"],stop],[0,0],color=color,lw=3,solid_capstyle="butt")
        hit,=ax.plot(E["m6"],0,"o",ms=6,mfc="white",mec="black",mew=.8,zorder=4)
        if name=="recovery":
            recovered,=ax.plot(E["m12"],0,"D",ms=9,mfc=LIGHT,mec=LIGHT,mew=.7,zorder=4)
            artists["recovered_endpoint"]=recovered
        best,=ax.plot(E["b"],0,"*",ms=8,mfc="black",mec="black",mew=.5,zorder=5)
        artists[name]={"axis":ax,"segment":segment,"hit":hit,"best":best}
        axes.append(ax)
    fig.text(.22/W,1.13/H,"hit %.1f%% (first six)"%(100*E["m6"]),fontsize=8,ha="left",va="center")
    fig.text(3.05/W,1.13/H,"library best %.1f%%"%(100*E["b"]),fontsize=8,ha="right",va="center")
    fig.text(.22/W,.78/H,"+ six more from the screening history",ha="left",va="center",fontsize=8,color=BLUE)
    fig.text(3.05/W,.78/H,"%.1f%%"%(100*E["m12"]),ha="right",va="center",fontsize=8)
    fig.text(.5,.24/H,"one hit in %s below 80%% sits >20 points short"%WORDS[cohort["one_in"]],
             ha="center",va="center",fontsize=8)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    texts=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    boxes=[t.get_window_extent(renderer) for t in texts]
    assert all(b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height for b in boxes), \
        "TOC text leaves the canvas"
    assert not any(a.overlaps(b) for i,a in enumerate(boxes) for b in boxes[i+1:]), "TOC labels overlap"
    scheme_box=scheme_ax.get_window_extent(renderer)
    assert scheme_box.y0>max(box.y1 for box in boxes), \
        "TOC reaction scheme overlaps the endpoint labels"
    rows=[]
    symbol_boxes=[]
    for name in ("initial","recovery"):
        row=artists[name]
        ax=row["axis"]
        segment=row["segment"]
        points=ax.transData.transform(np.column_stack((segment.get_xdata(),segment.get_ydata())))
        for role in ("hit","best"):
            symbol_boxes.append(row[role].get_window_extent(renderer))
        rows.append({"name":name,"color":segment.get_color(),
                     "yield_endpoints":np.asarray(segment.get_xdata(),dtype=float).tolist(),
                     "endpoints_px":points.tolist(),
                     "hit_marker":row["hit"].get_marker(),"best_marker":row["best"].get_marker()})
    symbol_boxes.append(artists["recovered_endpoint"].get_window_extent(renderer))
    assert np.allclose(np.array(rows[0]["endpoints_px"])[:,0],np.array(rows[1]["endpoints_px"])[:,0]), \
        "The TOC's two yield-shortfall rows must share their printed start and endpoint"
    diamond=artists["recovered_endpoint"]
    star=artists["recovery"]["best"]
    diamond_xy=axes[1].transData.transform((diamond.get_xdata()[0],diamond.get_ydata()[0]))
    star_xy=axes[1].transData.transform((star.get_xdata()[0],star.get_ydata()[0]))
    assert np.array_equal(diamond_xy,star_xy), "Complete recovery must place the diamond on the best-recipe star"
    assert not any(box.overlaps(symbol) for box in boxes for symbol in symbol_boxes), \
        "TOC endpoint symbols overlap text"
    assert all(symbol.x0>=0 and symbol.y0>=0 and symbol.x1<=fig.bbox.width and symbol.y1<=fig.bbox.height
               for symbol in symbol_boxes), "TOC endpoint symbols leave the canvas"
    fig._toc_example=E
    fig._toc_cohort=cohort
    fig._toc_artists=artists
    fig._toc_geometry={"size_in":[W,H],"dpi":DPI,"png_px":[round(W*DPI),round(H*DPI)],
        "text_bbox_px":[list(b.extents) for b in boxes],"scheme_bbox_px":list(scheme_box.extents),
        "motif":"orange initial shortfall / light-blue complete recovery; hit circle / added-result diamond / library-best star",
        "library_point_cloud":False,"rows":rows,
        "data":{"m6":E["m6"],"m12":E["m12"],"b":E["b"],"full_recovery":True},
        "recovered_diamond_px":diamond_xy.tolist(),"recovered_best_star_px":star_xy.tolist(),
        "symbol_bbox_px":[list(b.extents) for b in symbol_boxes]}
    return fig


if __name__=="__main__":
    fig=graphic()
    fig.savefig(FIG/"TOC_graphic.tif",dpi=DPI,pil_kwargs={"compression":"tiff_lzw"})
    fig.savefig(FIG/"TOC_graphic.png",dpi=DPI)
    (OUT/"toc_geometry.json").write_text(json.dumps(fig._toc_geometry,indent=2)+"\n",encoding="utf-8")
    plt.close(fig)
    print("TOC graphic written: 975 x 525 px; pair "+json.loads((OUT/"example_pair.json").read_text(encoding="utf-8"))["subpair"])
