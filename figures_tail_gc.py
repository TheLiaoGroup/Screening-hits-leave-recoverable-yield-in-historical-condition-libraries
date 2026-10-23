"""Supplementary decision curves from held-out fixed-budget tail estimates."""
from pathlib import Path
import hashlib, json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figure_style_gc import SMALL, align_panels, configure_fonts, panel_title

P=Path(__file__).resolve().parent
data=P/'out/review_tail_curve_pooled.csv'
d=pd.read_csv(data)
configure_fonts()
fig,axes=plt.subplots(1,2,figsize=(7.0,3.8),sharey=True)
for ax,cohort,title in zip(axes,['full_gt0','full_ge20pct'],
                           ['Full-library maximum >0','Full-library maximum ≥20%']):
    for method,color,style,label in [('greedy','#0072B2','-','Greedy'),
                                    ('topmean','#666666','--','Historical mean')]:
        r=d[(d.cohort==cohort)&(d.method==method)].sort_values('k')
        ax.plot(r.k,100*r.tail_rate,color=color,ls=style,lw=1.7,label=label)
        marked=r[r.k.isin([6,9,12,24])]
        ax.scatter(marked.k,100*marked.tail_rate,color=color,s=22,zorder=3)
        if method=='greedy':
            for row in marked.itertuples():
                ax.annotate(f'{100*row.tail_rate:.1f}%',(row.k,100*row.tail_rate),
                            xytext=(0,8 if row.k==24 else -16),textcoords='offset points',
                            ha='center',fontsize=SMALL,color=color)
    ax._panel_title = title
    ax.set_xlabel('Conditions per substrate pair, k')
    ax.set_xticks([1,6,9,12,18,24]); ax.set_xlim(.5,25); ax.set_ylim(0,51)
    ax.grid(axis='y',color='#E3E3E3',lw=.6)
axes[0].set_ylabel('Pairs losing >20 yield percentage points (%)')
axes[1].legend(frameon=False,loc='upper right')
fig.tight_layout(rect=(0, 0, 1, .92))
for i,ax in enumerate(axes):
    panel_title(fig, ax, chr(ord('a')+i), ax._panel_title, row=0, col=i)
align_panels(fig)
stem=P/'figures/SupplementaryFigure1_tail'
fig.savefig(stem.with_suffix('.png'),dpi=300)
fig.savefig(stem.with_suffix('.pdf'))
plt.close(fig)
(P/'out/review_tail_figure_manifest.json').write_text(json.dumps({
    'source_csv':data.name,'source_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),
    'weighting':'pooled pair-level risk; same original CV folds; fixed library reference',
    'cohorts':['full_gt0','full_ge20pct'],'methods':['greedy','topmean'],
    'budgets':list(range(1,25))},indent=2),encoding='utf-8')
print('Wrote SupplementaryFigure1_tail.png/pdf')
