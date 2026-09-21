"""
Out-of-time evaluation on the test cutoffs (Sep-Nov), all models in one table.

Models : no-skill, rule heuristic (as-of thresholds), logistic, LightGBM, LightGBM+region
         ablation, ORACLE (true hazard, uses ground truth, validation only).
Metrics: PR-AUC vs base rate, ROC-AUC, top-decile lift, precision/recall at the
         "contact top 10%" threshold chosen on VALIDATION, Brier score (calibrated).
Uncertainty: 200 bootstrap resamples by customer -> 95% CIs.
Outputs: results table (json + docs/results.md), reliability plot, gains curve.

Run: python -m churn_model.evaluate
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from churn_model.common import ROOT, load_config, load_panel, out_dir, split

N_BOOT = 200
TOP_K = 0.10


def lift_at(y, s, k=TOP_K):
    n = max(int(len(y) * k), 1)
    top = np.argpartition(-s, n - 1)[:n]
    return y[top].mean() / y.mean() if y.mean() > 0 else np.nan


def rule_score(df):
    """The root-cause engine's thresholds applied as-of: silent 30+ d, tickets 3+, unpaid 2+."""
    return ((df["days_since_use"] >= 30).astype(int) + (df["tix_90d"] >= 3).astype(int)
            + (df["unpaid_last3"] >= 2).astype(int)).values + 0.0


def bootstrap_ci(y, scores, groups, seed, n_boot=N_BOOT):
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(groups)
    order = np.argsort(codes, kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(codes))]
    rows_of = [order[starts[i]:starts[i + 1]] for i in range(len(uniq))]
    out = {k: {"pr": [], "lift": []} for k in scores}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_of[i] for i in pick])
        yb = y[idx]
        if yb.sum() == 0:
            continue
        for k, s in scores.items():
            out[k]["pr"].append(average_precision_score(yb, s[idx]))
            out[k]["lift"].append(lift_at(yb, s[idx]))
    return {k: {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                for m, v in d.items()} for k, d in out.items()}


def main():
    cfg = load_config()
    od = out_dir(cfg)
    panel = load_panel(cfg)
    tr, va, te = split(panel, cfg)
    truth = pd.read_parquet(od / "churn_panel_truth.parquet")
    te = te.merge(truth, on=["customer_id", "cutoff"], how="left")
    y = te["y"].values
    m_lr, m_gb = (joblib.load(od / "models" / f"{n}.joblib") for n in ("logistic", "lightgbm"))
    abl = joblib.load(od / "models" / "lightgbm_region_ablation.joblib")

    Xte_r = pd.concat([te[m_gb.features].fillna(0),
                       pd.get_dummies(te["region"]).reindex(columns=abl["regions"], fill_value=0)
                       .add_prefix("reg_").astype(int)], axis=1)[abl["features"]]
    raw = {"logistic": m_lr.predict_raw(te), "lightgbm": m_gb.predict_raw(te)}
    cal = {"logistic": m_lr.predict_proba(te), "lightgbm": m_gb.predict_proba(te)}
    scores = {"no-skill": np.full(len(y), y.mean()) + np.random.default_rng(0).normal(0, 1e-9, len(y)),
              "rule heuristic": rule_score(te) + np.random.default_rng(1).normal(0, 1e-6, len(y)),
              "logistic": cal["logistic"], "lightgbm": cal["lightgbm"],
              "lightgbm + region (ablation)": abl["model"].predict_proba(Xte_r)[:, 1],
              "ORACLE true hazard": te["true_hazard"].values}
    ranks = {"logistic": raw["logistic"], "lightgbm": raw["lightgbm"]}   # rank on the raw score (calibration is monotonic)

    # threshold chosen on validation: contact the top 10% by score
    thr = {"logistic": np.quantile(m_lr.predict_proba(va), 1 - TOP_K),
           "lightgbm": np.quantile(m_gb.predict_proba(va), 1 - TOP_K)}

    rows, base = {}, float(y.mean())
    for name, s in scores.items():
        rk = ranks.get(name, s)
        r = {"pr_auc": average_precision_score(y, rk), "roc_auc": 0.5 if name == "no-skill" else roc_auc_score(y, rk),
             "top10_lift": 1.0 if name == "no-skill" else lift_at(y, rk)}
        if name in thr:
            flag = cal[name] >= thr[name]
            r.update(precision_at_thr=float(y[flag].mean()) if flag.any() else 0.0,
                     recall_at_thr=float(y[flag].sum() / y.sum()), flagged_share=float(flag.mean()),
                     brier=brier_score_loss(y, cal[name]),
                     brier_uncalibrated=brier_score_loss(y, raw[name]))
        rows[name] = {k: float(v) for k, v in r.items()}
    cis = bootstrap_ci(y, {k: ranks.get(k, v) for k, v in scores.items() if k != "no-skill"},
                       te["customer_id"].values, cfg["seed"])
    for k, v in cis.items():
        rows[k]["pr_auc_ci95"], rows[k]["top10_lift_ci95"] = v["pr"], v["lift"]
    rows["_meta"] = {"test_rows": int(len(y)), "test_positives": int(y.sum()), "base_rate": base,
                     "bootstrap_resamples": N_BOOT, "brier_base_rate_only": float(base * (1 - base))}

    # per-cutoff view (Kassala outage sits in the Sep-Oct cutoffs)
    per_cut = {}
    for c, g in te.groupby("cutoff"):
        per_cut[int(c)] = {"n": int(len(g)), "positives": int(g["y"].sum()),
                           "lightgbm_pr_auc": float(average_precision_score(g["y"], raw["lightgbm"][g.index.values])),
                           "lightgbm_roc_auc": float(roc_auc_score(g["y"], raw["lightgbm"][g.index.values]))}
    rows["_per_cutoff"] = per_cut
    (od / "evaluation_results.json").write_text(json.dumps(rows, indent=2))

    # ---- figures
    fig_dir = ROOT / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for n, c in [("logistic", "#1f77b4"), ("lightgbm", "#d62728")]:
        fr, mp = calibration_curve(y, cal[n], n_bins=10, strategy="quantile")
        ax.plot(mp, fr, "o-", color=c, label=f"{n} (Brier {rows[n]['brier']:.4f})")
    lim = max(0.12, float(ax.get_xlim()[1]))
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="perfect")
    ax.set_xlabel("Predicted probability"), ax.set_ylabel("Observed churn rate")
    ax.set_title("Reliability on test cutoffs (Sep-Nov)"), ax.legend(fontsize=8)
    fig.tight_layout(), fig.savefig(fig_dir / "reliability.png", dpi=150), plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for n, c in [("logistic", "#1f77b4"), ("lightgbm", "#d62728"), ("ORACLE true hazard", "#2ca02c")]:
        order = np.argsort(-ranks.get(n, scores[n]))
        ax.plot(np.arange(1, len(y) + 1) / len(y) * 100, np.cumsum(y[order]) / y.sum() * 100, label=n)
    ax.plot([0, 100], [0, 100], "k--", lw=1, label="random")
    ax.set_xlabel("% subscribers contacted (highest risk first)"), ax.set_ylabel("% of churners captured")
    ax.set_title("Cumulative gains, test cutoffs"), ax.legend(fontsize=8)
    fig.tight_layout(), fig.savefig(fig_dir / "gains.png", dpi=150), plt.close(fig)

    # ---- markdown table
    L = ["# Out-of-time results (simulated data)\n",
         f"Test cutoffs Sep-Nov, {len(y):,} rows, {int(y.sum())} churners, base rate {base:.2%}. "
         f"95% CIs from {N_BOOT} customer-level bootstrap resamples.\n",
         "| Model | PR-AUC (95% CI) | ROC-AUC | Top-10% lift (95% CI) | Precision @ top10% | Recall @ top10% | Brier |",
         "|---|---|---|---|---|---|---|"]
    for n in scores:
        r = rows[n]
        ci = lambda k: f"{r[k][0]:.3f}-{r[k][1]:.3f}" if k in r else "-"
        L.append(f"| {n} | {r['pr_auc']:.3f} ({ci('pr_auc_ci95')}) | {r['roc_auc']:.3f} | "
                 f"{r['top10_lift']:.2f} ({r['top10_lift_ci95'][0]:.2f}-{r['top10_lift_ci95'][1]:.2f})"
                 if "top10_lift_ci95" in r else f"| {n} | {r['pr_auc']:.3f} | {r['roc_auc']:.3f} | {r['top10_lift']:.2f}")
        L[-1] += (f" | {r.get('precision_at_thr', float('nan')):.3f} | {r.get('recall_at_thr', float('nan')):.3f}"
                  f" | {r.get('brier', float('nan')):.4f} |")
    (ROOT / "docs" / "results.md").write_text("\n".join(L).replace("nan", "-") + "\n", encoding="utf-8")
    print("\n".join(L))
    print("\nper cutoff (lightgbm):", json.dumps(per_cut))


if __name__ == "__main__":
    main()
