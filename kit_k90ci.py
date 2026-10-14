"""How well determined is the kit size a transformation needs?

The kit size for 90% retention is a threshold crossing on a curve that is nearly flat near 0.90, so a
point estimate hides how much of it is sampling error. Substrate pairs are resampled with replacement,
the kit refitted on the resample and scored on the pairs the resample left out, and the crossing read
again. Writes out/kit_k90ci.csv: the interval for the kit size, and for retention at six.
"""
import sys, warnings, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import lib
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
OUT = Path(__file__).parent / "out"; OUT.mkdir(exist_ok=True)
KS = list(range(1, 13)); B = 400; OVER = 13
A, C = lib.load(); KC = pd.read_csv(OUT / "kit_curve.csv"); rows = []
for screen in sorted(A.screen.unique()):
    label = A[A.screen == screen].screen_label.iloc[0]
    Y = lib.screen_matrix(A, screen).values; S, n = Y.shape
    rng = np.random.RandomState(23); ks = []; r6 = []
    for _ in range(B):
        idx = rng.choice(S, S, replace=True); oob = np.setdiff1d(np.arange(S), np.unique(idx))
        if len(oob) < 5: continue
        kit = lib.greedy_kit(Y[idx], 12)
        f = lib.retention(Y[oob], list(range(n)))[1].mean()
        cur = {k: lib.retention(Y[oob], kit[:k])[0].mean() / f for k in KS}
        ks.append(next((k for k in KS if cur[k] >= 0.90), OVER)); r6.append(cur[6])
    ks = np.array(ks); r6 = np.array(r6)
    kc = KC[KC.screen == screen].set_index("k").retention_greedy
    point = int(kc.index[kc >= 0.90].min()) if (kc >= 0.90).any() else OVER
    lo, hi = np.percentile(ks, [2.5, 97.5])
    r6lo, r6hi = np.percentile(r6, [2.5, 97.5])
    rows.append({"screen": screen, "label": label, "pairs": S, "k90": point,
                 "k90_lo": float(lo), "k90_hi": float(hi), "share_over_12": float((ks == OVER).mean()),
                 "ret6": float(kc.loc[6]), "ret6_lo": float(r6lo), "ret6_hi": float(r6hi)})
    print(screen, point, lo, hi, flush=True)
T = pd.DataFrame(rows); T.to_csv(OUT / "kit_k90ci.csv", index=False)
T["width"] = T.k90_hi - T.k90_lo
print(T.round(2).to_string(index=False))
print("median 95%% interval width: %.1f conditions" % T.width.median())
print("arrays whose interval spans both 6 or fewer and more than twelve: %d" % int(((T.k90_lo <= 6) & (T.k90_hi >= 13)).sum()))
lo = T.set_index("label")
print("reductive amination %.1f-%.1f, Hiyama %.1f-%.1f, Ullmann %.1f-%.1f" % (
    lo.loc["Reductive amination", "k90_lo"], lo.loc["Reductive amination", "k90_hi"],
    lo.loc["Hiyama (Pd)", "k90_lo"], lo.loc["Hiyama (Pd)", "k90_hi"],
    lo.loc["Ullmann C\u2013O (Cu)", "k90_lo"], lo.loc["Ullmann C\u2013O (Cu)", "k90_hi"]))
