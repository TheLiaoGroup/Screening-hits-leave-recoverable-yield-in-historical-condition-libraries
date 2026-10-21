"""Condition-side ownership: which condition variable carries the substrate's condition preference.
All definitions live here. Data: the released DATASET_* files (BH_DATA_DIR or the sibling paper's 06_Data_and_Code)."""
import os, re, itertools
import numpy as np, pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
DATA = Path(os.environ["BH_DATA_DIR"]) if os.environ.get("BH_DATA_DIR") else HERE / "data"
# this paper's own release: BH_DATA_DIR if it holds arrays.csv, else data/ beside the code (the shipped package), else 02_CODE_AND_DATA/data/ (the working tree)
DATASET = next((d for d in (DATA, HERE / "data", HERE / "02_CODE_AND_DATA" / "data") if (d / "arrays.csv").exists()), None)
SHRINK = 4.0          # pseudo-count for shrunken level means (same as the partner paper)
MIN_SHARED = 10       # substrates two conditions must share for a pairwise r
NFOLD = 5

# ---------------------------------------------------------------- loading
def _from_dataset(which):
    """arrays.csv of the released dataset, renamed to the columns the code expects."""
    d = pd.read_csv(DATASET / "arrays.csv", low_memory=False)
    # measured cells only: a numeric yield or a measured zero; unmeasured cells carry no yield
    d = d[(d["set"] == which) & d["measurement_status"].isin(["quantified", "zero_by_rule"])].copy()
    assert d["yield"].notna().all()
    d = d.rename(columns={"array_id": "screen", "array": "screen_label"}); d["subpair"] = d.sub_1_smiles + " >> " + d.sub_2_smiles
    return d

def load(which="analysis"):
    """(rows, conditions). Rows are the measured cells of one set of the released arrays (analysis by
    default); conditions carry the source column names."""
    if DATASET is None:
        raise FileNotFoundError("arrays.csv not found: set BH_DATA_DIR or place data/ beside the code")
    c = pd.read_csv(DATASET / "conditions.csv", low_memory=False)
    ren = {"solvent": "Standardized Name", "temperature_C": "temp_id", "time_h": "time_id", "note": "note_en"}
    for i in range(1, 7):
        ren.update({f"reagent{i}_name": f"Reagent{i}_Standardized Name", f"reagent{i}_cas": f"Reagent{i}_HTE_CAS", f"reagent{i}_equiv": f"Reagent{i}_HTE_Equiv", f"reagent{i}_id": f"Reagent{i}_id"})
    return _from_dataset(which), c.rename(columns=ren)

# ---------------------------------------------------------------- roles
# slot -> variable name, per screen. Slots absent here are ignored.
_PD = {"Reagent1": "precatalyst", "Reagent2": "ligand", "Reagent3": "base", "Reagent4": "additive", "solvent": "solvent"}
ROLES = {
    "Acylation|1":                 {"Reagent1": "activator", "Reagent2": "additive", "Reagent3": "base", "solvent": "solvent"},
    "Buchwald-Hartwig(CMC)|1": _PD, "Buchwald-Hartwig(CMC)|2": _PD, "Buchwald-Hartwig(CMC)|3": _PD,
    "Buchwald-Hartwig|1": _PD, "Buchwald-Hartwig|2": _PD, "C-O competitive reactions|1": _PD, "Suzuki|1": _PD,
    "Heck|1": _PD, "Ullmann|1": _PD,
    "Chan-Lam|1":                  {"Reagent1": "precatalyst", "Reagent2": "ligand", "Reagent3": "base", "Reagent4": "oxidant", "solvent": "solvent"},
    "Esterification|1":            {"Reagent1": "promoter", "Reagent2": "additive", "Reagent3": "base", "Reagent4": "coupling reagent", "Reagent5": "additive", "solvent": "solvent"},
    "Grignard|1":                  {"Reagent1": "additive", "Reagent2": "additive", "solvent": "solvent"},
    "Hiyama|1":                    {"Reagent1": "precatalyst", "Reagent2": "ligand", "Reagent3": "base", "Reagent4": "fluoride", "solvent": "solvent"},
    "Kumada|1":                    {"Reagent1": "precatalyst", "Reagent2": "ligand", "Reagent3": "additive", "solvent": "solvent"},
    "Reductive-amination|1":       {"Reagent1": "Lewis acid", "Reagent3": "reductant", "Reagent4": "acid", "solvent": "solvent"},
    "SNAr|1":                      {"Reagent3": "base", "solvent": "solvent"},
    "Sonogashira|1":               {"Reagent1": "precatalyst", "Reagent2": "co-catalyst", "Reagent3": "ligand", "Reagent4": "base", "Reagent5": "additive", "solvent": "solvent"},
}
METAL_SCREENS = {s for s in ROLES if "precatalyst" in ROLES[s].values()}

_METALS = ["Pd", "Ni", "Cu", "Fe", "Au", "Ag", "Ti", "Sn", "Zn", "Co", "Ce", "Cd", "In", "Mn", "Ir", "Rh", "Ru"]
def metal_of(name):
    n = str(name)
    if "nickel" in n.lower(): return "Ni"
    if "PEPPSI" in n or "palladium" in n.lower(): return "Pd"
    for m in _METALS:
        if re.search(r'(^|[^A-Za-z])' + m + r'(?![a-z])', n): return m
    return "none"

_NONLIG = {"oac", "acac", "mecn", "phcn", "dba", "cod", "allyl", "cinnamyl", "methylallyl", "otf", "tfa", "no3", "glyme",
           "toluene", "benzene", "chcl3", "ch2cl2", "h2o", "oh", "cl", "br", "i", "so4", "cl2", "br2", "2", "3", "4", "ii", "c",
           "et2o", "me2s", "5h2o", "6h2o"}
def norm_lig(s):
    s = str(s).lower().replace("\u00b7", ".")
    s = re.sub(r'\.(hbf4|hcl|chcl3|ch2cl2|toluene|benzene|me2s)$', '', s)
    s = re.sub(r'[\s\-\.]', '', s)
    s = s.replace("tbu3)", "tbu)3")
    return s

def embedded_ligand(name):
    """Ligand carried inside a precatalyst name; 'none' if the precatalyst has no phosphine/NHC/amine ligand."""
    n = str(name).strip()
    if n == "nan" or not n: return "none"
    if "nickel(II) chloride" in n: return "nnn-pincer"
    if n.startswith("trans-di("): return "herrmann"          # Herrmann's palladacycle, P(o-tolyl)3-derived
    m = re.match(r'^(.+?)[\s\-]+Pd[\s\-]*(G\d)?$', n)
    if m: return norm_lig(m.group(1))
    m = re.match(r'^PEPPSI-(.+)$', n)
    if m: return norm_lig(m.group(1))
    m = re.match(r'^\((.+?)\)Pd\(allyl\)Cl$', n)
    if m: return norm_lig(m.group(1))
    m = re.match(r'^\[Cu\(OH\)(.+?)\]', n)
    if m: return norm_lig(m.group(1))
    if n.startswith("[PdP(tBu)3Br]2"): return norm_lig("P(tBu)3")
    found = []
    for g in re.findall(r'[\(\[]([^\(\)\[\]]*(?:\([^\(\)]*\))*[^\(\)\[\]]*)[\)\]]', n):
        g2 = norm_lig(g)
        if g2 in _NONLIG or not g2: continue
        if re.match(r'^(pd|ni|cu|fe)\(', g2): continue
        if re.fullmatch(r'[a-z0-9]{1,2}', g2): continue
        found.append(g2)
    found = [f for f in found if f not in _NONLIG]
    return "+".join(sorted(set(found))) if found else "none"

def condition_variables(screen, C, rxn_class, cond_ids):
    """DataFrame indexed by condition_id: one column per variable (levels as strings; missing slot -> 'none')."""
    cc = C[(C.rxn_class == rxn_class) & (C.condition_id.isin(cond_ids))].set_index("condition_id").reindex(cond_ids)
    roles = ROLES[screen]; V = {}
    for slot, role in roles.items():
        col = "Standardized Name" if slot == "solvent" else slot + "_Standardized Name"
        vals = cc[col].fillna("none").astype(str)
        if role == "ligand": vals = vals.map(norm_lig)
        if role in V: V[role] = [a if b == "none" else (b if a == "none" else a + "+" + b) for a, b in zip(V[role], vals)]
        else: V[role] = list(vals)
    if "precatalyst" in roles.values():
        pre = cc["Reagent1_Standardized Name"].astype(str)
        V["metal"] = list(pre.map(metal_of))
        emb = pre.map(embedded_ligand)
        add = V.get("ligand", ["none"] * len(cc))
        V["ligand"] = ["+".join(sorted(set([x for x in (a, e) if x != "none"]))) or "none" for a, e in zip(add, emb)]
    out = pd.DataFrame(V, index=cond_ids)
    out = out[[c for c in out.columns if out[c].nunique() > 1]]           # a variable with one level is not a variable
    return out

# ---------------------------------------------------------------- residuals
def twoway(M):
    R = M.copy(); mu = np.nanmean(R); a = np.zeros(R.shape[0]); b = np.zeros(R.shape[1])
    for _ in range(100):
        a = np.nanmean(R - mu - b[None, :], axis=1); b = np.nanmean(R - mu - a[:, None], axis=0)
    return R - mu - a[:, None] - b[None, :]

def screen_matrix(A, screen):
    d = A[A.screen == screen]
    return d.pivot_table(index="subpair", columns="condition_id", values="yield", aggfunc="mean")

# ---------------------------------------------------------------- pairwise agreement
def pairwise_table(R, V):
    """One row per condition pair with >= MIN_SHARED substrates: r and share_<var> indicators."""
    n = R.shape[1]; rows = []
    for i, j in itertools.combinations(range(n), 2):
        x, y = R[:, i], R[:, j]; ok = ~np.isnan(x) & ~np.isnan(y)
        if ok.sum() < MIN_SHARED: continue
        xs, ys = x[ok], y[ok]
        if xs.std() == 0 or ys.std() == 0: continue
        row = {"i": i, "j": j, "r": np.corrcoef(xs, ys)[0, 1]}
        for v in V.columns: row["share_" + v] = int(V[v].iloc[i] == V[v].iloc[j])
        rows.append(row)
    return pd.DataFrame(rows)

def agreement_regression(P, vars_):
    """OLS of pairwise r on share indicators. Returns dict var -> beta, plus intercept and the mean r of pairs sharing nothing."""
    X = np.column_stack([np.ones(len(P))] + [P["share_" + v].values for v in vars_]).astype(float)
    beta, *_ = np.linalg.lstsq(X, P["r"].values, rcond=None)
    out = {"intercept": beta[0]}
    for k, v in enumerate(vars_): out[v] = beta[k + 1]
    none = P[["share_" + v for v in vars_]].sum(1) == 0
    out["r_share_none"] = P.r[none].mean() if none.sum() >= 5 else np.nan
    out["n_share_none"] = int(none.sum())
    return out

def agreement_with_inference(R, V, n_perm=200, n_boot=300, seed=0):
    """beta per variable; permutation p (labels of that variable shuffled across conditions, others fixed);
    bootstrap CI over conditions."""
    rng = np.random.RandomState(seed); vars_ = list(V.columns)
    P = pairwise_table(R, V); obs = agreement_regression(P, vars_)
    res = {v: {"beta": obs[v]} for v in vars_}
    res["_intercept"] = obs["intercept"]; res["_r_share_none"] = obs["r_share_none"]; res["_n_share_none"] = obs["n_share_none"]
    for v in vars_:
        null = []
        for _ in range(n_perm):
            lab = rng.permutation(V[v].values)
            Pp = P.copy(); Pp["share_" + v] = (lab[P.i.values] == lab[P.j.values]).astype(int)
            null.append(agreement_regression(Pp, vars_)[v])
        null = np.array(null)
        res[v]["p_perm"] = (np.sum(null >= obs[v]) + 1) / (n_perm + 1)
        res[v]["null_mean"] = null.mean(); res[v]["null_sd"] = null.std()
    n = R.shape[1]; boots = {v: [] for v in vars_}
    for _ in range(n_boot):
        keep = rng.choice(n, n, replace=True); w = np.bincount(keep, minlength=n)
        Pw = P[(w[P.i.values] > 0) & (w[P.j.values] > 0)].copy()
        wt = w[Pw.i.values] * w[Pw.j.values]
        X = np.column_stack([np.ones(len(Pw))] + [Pw["share_" + v].values for v in vars_]).astype(float)
        sw = np.sqrt(wt); b, *_ = np.linalg.lstsq(X * sw[:, None], Pw.r.values * sw, rcond=None)
        for k, v in enumerate(vars_): boots[v].append(b[k + 1])
    for v in vars_:
        res[v]["ci_lo"], res[v]["ci_hi"] = np.percentile(boots[v], [2.5, 97.5])
    res["_n_pairs"] = len(P)
    # metal is a coarsening of precatalyst: report their sum as "catalyst" (same metal and same precursor) and use it as the candidate
    if "metal" in vars_ and "precatalyst" in vars_:
        boots["catalyst"] = list(np.array(boots["metal"]) + np.array(boots["precatalyst"]))
        res["catalyst"] = {"beta": res["metal"]["beta"] + res["precatalyst"]["beta"], "p_perm": min(res["metal"]["p_perm"], res["precatalyst"]["p_perm"]),
                           "null_mean": res["metal"]["null_mean"] + res["precatalyst"]["null_mean"], "null_sd": np.nan}
        res["catalyst"]["ci_lo"], res["catalyst"]["ci_hi"] = np.percentile(boots["catalyst"], [2.5, 97.5])
        cands = [v for v in vars_ if v not in ("metal", "precatalyst")] + ["catalyst"]
    else:
        cands = vars_
    # owner = largest observed beta among candidates; margin over the runner-up with its bootstrap CI
    order = sorted(cands, key=lambda v: -res[v]["beta"])
    res["_owner"] = order[0]
    if len(order) > 1:
        diff = np.array(boots[order[0]]) - np.array(boots[order[1]])
        res["_runner_up"] = order[1]; res["_margin"] = res[order[0]]["beta"] - res[order[1]]["beta"]
        res["_margin_lo"], res["_margin_hi"] = np.percentile(diff, [2.5, 97.5])
    else:
        res["_runner_up"] = None; res["_margin"] = res["_margin_lo"] = res["_margin_hi"] = np.nan
    return res, P

# ---------------------------------------------------------------- held-out R2 over conditions
def profile_predict(R, labels, train, test):
    """Predict held-out condition columns from substrate x level shrunken means learned on training conditions."""
    pred = np.zeros((R.shape[0], len(test)))
    for jj, j in enumerate(test):
        same = [t for t in train if labels[t] == labels[j]]
        if not same: continue
        sub = R[:, same]; cnt = np.sum(~np.isnan(sub), 1); sm = np.nansum(sub, 1)
        pred[:, jj] = np.where(cnt > 0, sm / (cnt + SHRINK), 0.0)
    return pred

def heldout_r2(R, V, vars_, folds):
    """R2 over held-out conditions of the additive sum of the named variables' profiles."""
    num = den = 0.0
    for f in folds:
        tr = np.setdiff1d(np.arange(R.shape[1]), f); pred = np.zeros((R.shape[0], len(f)))
        for v in vars_: pred += profile_predict(R, V[v].values, tr, f)
        y = R[:, f]; ok = ~np.isnan(y)
        num += np.sum((y[ok] - pred[ok]) ** 2); den += np.sum(y[ok] ** 2)
    return 1 - num / den

def make_folds(n, seed=20260909):
    rng = np.random.RandomState(seed); return np.array_split(rng.permutation(n), NFOLD)

def r2_table(R, V, n_perm=50, seed=1):
    folds = make_folds(R.shape[1]); vars_ = list(V.columns); rng = np.random.RandomState(seed); out = {}
    for v in vars_:
        obs = heldout_r2(R, V, [v], folds)
        null = np.array([heldout_r2(R, V.assign(**{v: rng.permutation(V[v].values)}), [v], folds) for _ in range(n_perm)])
        out[v] = {"r2": obs, "null_p95": np.percentile(null, 95), "levels": int(V[v].nunique())}
    out["_all"] = heldout_r2(R, V, vars_, folds)
    for v in vars_:
        out[v]["r2_drop"] = out["_all"] - heldout_r2(R, V, [w for w in vars_ if w != v], folds)
    return out

# ---------------------------------------------------------------- kit retention
def greedy_kit(Y, k, allowed=None, distinct=None):
    """Greedy k conditions maximising mean over substrates of best yield. Y: substrates x conditions (NaN allowed).
    allowed: boolean mask over conditions; distinct: label array -> chosen conditions must have distinct labels."""
    n = Y.shape[1]; allowed = np.ones(n, bool) if allowed is None else allowed.copy()
    best = np.full(Y.shape[0], -np.inf); chosen = []
    Yf = np.where(np.isnan(Y), -np.inf, Y)
    for _ in range(k):
        cand = np.where(allowed)[0]
        if len(cand) == 0: break
        gains = np.array([np.mean(np.where(np.isinf(np.maximum(best, Yf[:, c])), 0, np.maximum(best, Yf[:, c]))) for c in cand])
        c = cand[int(np.argmax(gains))]; chosen.append(c); best = np.maximum(best, Yf[:, c]); allowed[c] = False
        if distinct is not None: allowed &= (distinct != distinct[c])
    return chosen

def retention(Y, kit):
    Yf = np.where(np.isnan(Y), -np.inf, Y)
    b = np.max(Yf[:, kit], 1) if kit else np.full(Y.shape[0], -np.inf)
    b = np.where(np.isinf(b), 0, b); full = np.where(np.isinf(np.max(Yf, 1)), 0, np.max(Yf, 1))
    return b, full

def kit_analysis(M, V, ks=(3, 6), nfold=10, seed=3, n_random=100):
    """Cross-validated over substrates: retention of mean best yield for unconstrained, V-diverse and V-fixed kits."""
    Y = M.values; S = Y.shape[0]; rng = np.random.RandomState(seed); folds = np.array_split(rng.permutation(S), nfold)
    rows = []
    for k in ks:
        acc = {"free": [], "random": []}; full_all = []
        for v in V.columns: acc["diverse:" + v] = []; acc["fixed:" + v] = []
        for f in folds:
            tr = np.setdiff1d(np.arange(S), f); Ytr, Yte = Y[tr], Y[f]
            _, full = retention(Yte, list(range(Y.shape[1]))); full_all.append(full)
            acc["free"].append(retention(Yte, greedy_kit(Ytr, k))[0])
            acc["random"].append(np.mean([retention(Yte, list(rng.choice(Y.shape[1], k, replace=False)))[0] for _ in range(n_random)], 0))
            for v in V.columns:
                lab = V[v].values
                acc["diverse:" + v].append(retention(Yte, greedy_kit(Ytr, k, distinct=lab))[0])
                bestlev, bestval, bestkit = None, -1, []
                for lev in pd.unique(lab):
                    mask = lab == lev
                    if mask.sum() < k: continue
                    kit = greedy_kit(Ytr, k, allowed=mask); val = retention(Ytr, kit)[0].mean()
                    if val > bestval: bestlev, bestval, bestkit = lev, val, kit
                acc["fixed:" + v].append(retention(Yte, bestkit)[0] if bestkit else np.zeros(len(f)))
        full = np.concatenate(full_all).mean()
        for key, vals in acc.items():
            rows.append({"k": k, "kit": key, "mean_best": np.concatenate(vals).mean(), "retention": np.concatenate(vals).mean() / full, "full": full})
    return pd.DataFrame(rows)

# ---------------------------------------------------------------- first-screen design
def first_screen(R, V, n_rep=30, cap=10, seed=5):
    """Train the additive profile model (all variables) on m conditions and predict the rest.
    Spanning set for variable v: one condition per level of v (levels capped at `cap`, largest levels first).
    Baseline: m random conditions. Returns rows with R2_span, R2_random, delta, m per variable."""
    rng = np.random.RandomState(seed); n = R.shape[1]; vars_ = list(V.columns); rows = []
    def r2(train):
        test = np.setdiff1d(np.arange(n), train); pred = np.zeros((R.shape[0], len(test)))
        for v in vars_: pred += profile_predict(R, V[v].values, train, test)
        y = R[:, test]; ok = ~np.isnan(y)
        return 1 - np.sum((y[ok] - pred[ok]) ** 2) / np.sum(y[ok] ** 2)
    for v in vars_:
        levels = list(V[v].value_counts().index[:cap]); m = len(levels)
        if m < 2 or m >= n - 5: continue
        span, rand = [], []
        for _ in range(n_rep):
            tr = np.array([rng.choice(np.where(V[v].values == lev)[0]) for lev in levels])
            span.append(r2(tr)); rand.append(r2(rng.choice(n, m, replace=False)))
        rows.append({"variable": v, "m": m, "R2_span": np.mean(span), "R2_random": np.mean(rand), "delta": np.mean(span) - np.mean(rand),
                     "delta_sd": np.std(np.array(span) - np.array(rand)) / np.sqrt(n_rep)})
    return pd.DataFrame(rows)

def design_association(V):
    """Cramer's V between every pair of variables (how confounded the array's design is)."""
    rows = []
    for a, b in itertools.combinations(V.columns, 2):
        t = pd.crosstab(V[a], V[b]).values; n = t.sum()
        e = np.outer(t.sum(1), t.sum(0)) / n; chi2 = np.sum((t - e) ** 2 / np.where(e == 0, 1, e))
        k = min(t.shape) - 1
        rows.append({"a": a, "b": b, "cramers_v": np.sqrt(chi2 / (n * k)) if k > 0 else np.nan})
    return pd.DataFrame(rows)
