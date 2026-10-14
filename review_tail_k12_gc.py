"""Extend the existing fixed-outcome molecular reweighting to k12 only.

Reads deposited predictions; does not refit panels or rerun earlier analyses.
Reproduces k6 draws as a direct consistency check using the same random stream.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from review_tail_gc import B, SEED, seed_for, node_pair_weights, cohort

P = Path(__file__).resolve().parent
O = P / 'out'
SCHEME = 'role_merged_node_exponential'

def main():
    files = ['review_tail_pair_identity.csv', 'review_tail_pair_scores.csv',
             'review_tail_bootstrap_draws.npz', 'review_tail_bootstrap_summary.csv']
    sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    before = {f: sha(O/f) for f in files}
    pairs = pd.read_csv(O/files[0])
    scores = pd.read_csv(O/files[1])
    full = pairs.full.to_numpy()
    masks = np.column_stack([cohort(full, c) for c in ('full_gt0', 'full_ge20pct')]).astype(float)
    events = {}
    for k in (6, 12):
        x = pairs[['screen', 'subpair']].merge(scores[scores.k == k], on=['screen', 'subpair'], validate='one_to_one')
        assert len(x) == len(pairs) and np.allclose(x.full, full, rtol=0, atol=1e-12)
        events[k] = [(100*(full-x[m].to_numpy()) > 20).astype(float) for m in ('greedy', 'topmean')]
    nnode = int(max(pairs.node1.max(), pairs.node2.max())+1)
    rng = np.random.default_rng(seed_for(SCHEME))
    draws = {k: np.empty((B, 2, 2)) for k in (6, 12)}
    for start in range(0, B, 250):
        count = min(250, B-start)
        w = node_pair_weights(rng.exponential(size=(count, nnode)), pairs.node1.to_numpy(), pairs.node2.to_numpy())
        denominator = w @ masks
        for k in (6, 12):
            for j, e in enumerate(events[k]):
                draws[k][start:start+count, :, j] = (w @ (masks*e[:, None])) / denominator
    with np.load(O/'review_tail_bootstrap_draws.npz') as old:
        for i, c in enumerate(('full_gt0', 'full_ge20pct')):
            ix = np.flatnonzero((old['group_cohort'] == c) & (old['group_screen'] == 'ALL')).item()
            assert np.allclose(draws[6][:, i, :], old[SCHEME][:, ix, :], rtol=0, atol=1e-12)
    rows = []
    for i, c in enumerate(('full_gt0', 'full_ge20pct')):
        mask = masks[:, i].astype(bool)
        pg, pt = (e[mask].mean() for e in events[12])
        g, t = draws[12][:, i, 0], draws[12][:, i, 1]
        for metric, estimate, d in [('greedy_tail_rate', pg, g), ('topmean_tail_rate', pt, t),
            ('benefit_topmean_minus_greedy_rate', pt-pg, t-g), ('relative_tail_reduction', (pt-pg)/pt, (t-g)/t)]:
            lo, hi = np.quantile(d, [.025, .975])
            rows.append(dict(scheme=SCHEME, screen='ALL', cohort=c, k=12, metric=metric,
                estimate=estimate, interval_lo=lo, interval_hi=hi, draws=B, pairs=int(mask.sum()),
                conditional_on_fixed_oof_predictions=True))
    out = pd.DataFrame(rows)
    out.to_csv(O/'review_tail_k12_bootstrap_summary.csv', index=False)
    np.savez_compressed(O/'review_tail_k12_bootstrap_draws.npz', rates=draws[12], cohorts=np.array(['full_gt0', 'full_ge20pct']))
    assert before == {f: sha(O/f) for f in files}
    manifest = dict(seed=SEED, draws=B, nodes=nnode, k6_draws_reproduced=True,
        input_sha256=before, script_sha256=sha(Path(__file__)), original_inputs_unchanged=True,
        scope='k12 pooled rates; same canonical-molecule multiplier stream as k6; fixed existing OOF predictions')
    (O/'review_tail_k12_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(out[out.metric == 'benefit_topmean_minus_greedy_rate'].to_string(index=False))

if __name__ == '__main__':
    main()
