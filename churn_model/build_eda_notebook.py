"""
Build and execute notebooks/01_eda_churn_panel.ipynb (outputs embedded).

EDA on the churn panel: who churns, when, where, and which signals separate churners.
Uses only the panel (as-of features + label), never the ground-truth latent state.
Run: python -m churn_model.build_eda_notebook
"""
import nbformat as nbf
from nbclient import NotebookClient

from churn_model.common import ROOT

MD, CODE = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

CELLS = [
    MD("# EDA: churn panel\n\n"
       "One row per active subscriber per month-end cutoff; features are as-of the cutoff, the label is "
       "\"churns in the next calendar month\". **Simulated data** (hidden-process generator). Sep-Nov are the test months; "
       "this notebook only looks at the panel and never at the generator's latent state.\n\n"
       "Questions: how rare is churn, how does it move over time and by region, and which signals separate churners?"),
    CODE("import sys, warnings\nfrom pathlib import Path\nwarnings.filterwarnings('ignore')\n"
         "sys.path.insert(0, str(Path.cwd().parent))\n"
         "import numpy as np, pandas as pd, matplotlib.pyplot as plt\n"
         "from sklearn.metrics import roc_auc_score\n"
         "from churn_model.common import load_config, load_panel\nfrom churn_model.as_of_features import FEATURE_COLUMNS as F\n"
         "cfg = load_config(); P = load_panel(cfg)\n"
         "MONTH = {2:'Mar',3:'Apr',4:'May',5:'Jun',6:'Jul',7:'Aug',8:'Sep',9:'Oct',10:'Nov'}\n"
         "plt.rcParams.update({'figure.dpi':110,'axes.spines.top':False,'axes.spines.right':False})\n"
         "print(f'{len(P):,} rows | {P.customer_id.nunique():,} subscribers | {len(F)} features | base rate {P.y.mean():.2%}')"),
    MD("## 1. Churn is rare and moves with the calendar\n"
       "A class balance of about 2% means accuracy is meaningless; PR-AUC and lift are the honest metrics."),
    CODE("t = P.groupby('cutoff').agg(rows=('y','size'), churners=('y','sum'), rate=('y','mean'), split=('split','first'))\n"
         "t.index = [MONTH[i] for i in t.index]\ndisplay(t.assign(rate=t['rate'].map('{:.2%}'.format), rows=t['rows'].map('{:,}'.format)))\n"
         "fig, ax = plt.subplots(figsize=(7,3.2))\n"
         "colors = t['split'].map({'train':'#1f77b4','validation':'#ff7f0e','test':'#d62728'})\n"
         "ax.bar(t.index, t['rate']*100, color=colors); ax.set_ylabel('% churning next month')\n"
         "ax.set_title('Next-month churn by cutoff (blue train, orange validation, red test)'); plt.show()"),
    MD("## 2. Region and time: outages show up as churn hot spots\n"
       "The heatmap should light up where a regional outage drove dissatisfaction. Region is *not* a model feature."),
    CODE("h = P.pivot_table(index='region', columns='cutoff', values='y', aggfunc='mean')*100\n"
         "h = h.loc[h.mean(axis=1).sort_values(ascending=False).index]\n"
         "fig, ax = plt.subplots(figsize=(7.5,3.8)); im = ax.imshow(h.values, aspect='auto', cmap='Reds')\n"
         "ax.set_xticks(range(h.shape[1])); ax.set_xticklabels([MONTH[c] for c in h.columns])\n"
         "ax.set_yticks(range(h.shape[0])); ax.set_yticklabels(h.index)\n"
         "for i in range(h.shape[0]):\n    for j in range(h.shape[1]):\n        ax.text(j, i, f'{h.values[i,j]:.1f}', ha='center', va='center', fontsize=7)\n"
         "fig.colorbar(im, label='% churning next month'); ax.set_title('Churn rate by region and cutoff'); plt.show()"),
    MD("## 3. Plan and tenure"),
    CODE("P['tenure_band'] = pd.cut(P.tenure_days, [-1,89,180,365,730,10000], labels=['<90d','90-180d','180-365d','1-2y','>2y'])\n"
         "fig, axs = plt.subplots(1, 2, figsize=(9,3.2))\n"
         "(P.groupby(P.is_prepaid.map({1:'prepaid',0:'postpaid'})).y.mean()*100).plot.bar(ax=axs[0], color='#4c72b0', rot=0)\n"
         "axs[0].set_ylabel('% churning'); axs[0].set_title('By plan')\n"
         "(P.groupby('tenure_band', observed=True).y.mean()*100).plot.bar(ax=axs[1], color='#55a868', rot=0)\n"
         "axs[1].set_title('By tenure'); plt.tight_layout(); plt.show()"),
    MD("## 4. Do churners look different before they leave?\n"
       "Distributions of the strongest as-of signals for churners vs everyone else (log scale where skewed)."),
    CODE("cols = ['use_30d','use_trend','days_since_use','unpaid_last3','tix_30d','days_since_rech']\n"
         "fig, axs = plt.subplots(2, 3, figsize=(11,5.5))\n"
         "for ax, c in zip(axs.ravel(), cols):\n"
         "    for y, col, lab in [(0,'#1f77b4','stayed'),(1,'#d62728','churned next month')]:\n"
         "        v = P.loc[P.y==y, c].clip(upper=P[c].quantile(.99))\n"
         "        ax.hist(v, bins=30, alpha=.55, density=True, color=col, label=lab)\n"
         "    ax.set_title(c); ax.set_yscale('log')\n"
         "axs[0,0].legend(fontsize=7); plt.tight_layout(); plt.show()\n"
         "print(P.groupby('y')[cols].median().round(2).rename(index={0:'stayed',1:'churned'}))"),
    MD("## 5. How much signal is in each feature on its own?\n"
       "Direction-free ROC-AUC per feature on the test months. A single feature above 0.85 would be a leakage smell "
       "(`leakage_checks.py` fails the build if it happens)."),
    CODE("te = P[P.split=='test']\n"
         "a = pd.Series({f: roc_auc_score(te.y, te[f].fillna(0)) for f in F}); a = np.maximum(a, 1-a).sort_values()\n"
         "fig, ax = plt.subplots(figsize=(6.5,6)); a.plot.barh(ax=ax, color='#4c72b0'); ax.axvline(.85, color='r', ls='--', lw=1)\n"
         "ax.axvline(.5, color='grey', lw=1); ax.set_xlim(.45,.9); ax.set_xlabel('single-feature ROC-AUC (test)'); plt.tight_layout(); plt.show()\n"
         "print(f'strongest single feature: {a.idxmax()} = {a.max():.3f}')"),
    MD("## 6. Redundancy and data quality\n"
       "Usage features are correlated with each other, which is why SHAP values are grouped into reasons and why logistic and boosting end up close."),
    CODE("c = P[F].corr(method='spearman')\n"
         "fig, ax = plt.subplots(figsize=(7.5,6.5)); im = ax.imshow(c.values, cmap='coolwarm', vmin=-1, vmax=1)\n"
         "ax.set_xticks(range(len(F))); ax.set_xticklabels(F, rotation=90, fontsize=7); ax.set_yticks(range(len(F))); ax.set_yticklabels(F, fontsize=7)\n"
         "fig.colorbar(im); ax.set_title('Spearman correlation between features'); plt.tight_layout(); plt.show()\n"
         "pairs = c.where(np.triu(np.ones(c.shape), 1).astype(bool)).stack().abs().sort_values(ascending=False).head(5)\n"
         "print('most correlated pairs:'); print(pairs.round(2).to_string())\n"
         "print(f\"\\nnull cells in features: {int(P[F].isna().sum().sum())} | rows with zero usage in 30d: {(P.use_30d==0).mean():.1%}\")"),
    MD("## What this changes in the modeling\n"
       "- Rare positives, so PR-AUC, lift and calibration are reported, never accuracy.\n"
       "- Churn concentrates in regions during outages, so region is kept out of the main model and a regional ticket rate is used instead; "
       "a region one-hot is only an ablation.\n"
       "- Usage, recharge and payment signals separate churners from stayers, but no single feature is close to a leak, so the signal is real and noisy.\n"
       "- Features are correlated, so explanations are grouped into reasons rather than reported feature by feature."),
]


def main():
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(out)}}).execute()
    nbf.write(nb, out / "01_eda_churn_panel.ipynb")
    print("wrote notebooks/01_eda_churn_panel.ipynb (executed)")


if __name__ == "__main__":
    main()
