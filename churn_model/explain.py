"""
SHAP explanations + the answer-key test.

SHAP shows what the MODEL uses, not what causes churn. The answer-key test asks a
narrower question: does the model lean on the drivers the generator actually planted?

Run: python -m churn_model.explain
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from churn_model.common import ROOT, load_config, load_panel, out_dir, split

# Similar features are collapsed into one reason so codes never repeat.
FEATURE_GROUPS = {
    "usage_decline": ["use_30d", "use_prev60d", "use_trend", "data_mb_30d", "voice_min_30d"],
    "inactivity": ["days_since_use"],
    "recharge_gap": ["days_since_rech", "rech_30d", "rech_90d", "rech_amt_90d", "rech_gap_ratio"],
    "network_complaints": ["tix_30d", "tix_90d", "hard_tix_30d", "region_tix_rate_30d"],
    "unpaid_bills": ["unpaid_last3", "late_last3", "bill_ratio"],
    "new_subscriber": ["tenure_days", "tenure_lt90"],
    "plan_profile": ["is_prepaid", "monthly_arpu_sdg"],
}
# What phase1b_generate_data.py planted (answer_key.json "expected_top_drivers")
PLANTED = {"usage_decline", "inactivity", "recharge_gap", "network_complaints",
           "unpaid_bills", "new_subscriber", "plan_profile"}


def shap_values(model, X):
    """SHAP values (log-odds of the uncalibrated model) for the positive class."""
    if model.name == "lightgbm":
        sv = shap.TreeExplainer(model.base).shap_values(X[model.features].fillna(0))
    else:
        scaler, lr = model.base.steps[0][1], model.base.steps[-1][1]
        Xs = scaler.transform(X[model.features].fillna(0))
        sv = shap.LinearExplainer(lr, Xs).shap_values(Xs)
    if isinstance(sv, list):
        sv = sv[-1]
    sv = np.asarray(sv)
    return sv[..., 1] if sv.ndim == 3 else sv


def group_shap(sv, features):
    """Sum SHAP values within each reason group -> DataFrame (rows x groups)."""
    idx = {f: i for i, f in enumerate(features)}
    return pd.DataFrame({g: sv[:, [idx[f] for f in fs]].sum(axis=1)
                         for g, fs in FEATURE_GROUPS.items()})


def main():
    cfg = load_config()
    od = out_dir(cfg)
    model = joblib.load(od / "models" / "primary.joblib")
    _, _, te = split(load_panel(cfg), cfg)
    truth = pd.read_parquet(od / "churn_panel_truth.parquet")
    te = te.merge(truth, on=["customer_id", "cutoff"], how="left").reset_index(drop=True)

    sv = shap_values(model, te)
    gs = group_shap(sv, model.features)
    feat_imp = pd.Series(np.abs(sv).mean(0), index=model.features).sort_values(ascending=False)
    grp_imp = gs.abs().mean().sort_values(ascending=False)
    share = (grp_imp / grp_imp.sum()).round(3)

    fig_dir = ROOT / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    feat_imp.head(15)[::-1].plot.barh(ax=ax, color="#1f77b4")
    ax.set_xlabel("mean |SHAP| (log-odds)"), ax.set_title(f"Global feature importance ({model.name})")
    fig.tight_layout(), fig.savefig(fig_dir / "shap_bar.png", dpi=150), plt.close(fig)
    samp = te.sample(6000, random_state=cfg["seed"]).index
    shap.summary_plot(sv[samp], te.loc[samp, model.features].fillna(0), show=False, max_display=15)
    plt.tight_layout(), plt.savefig(fig_dir / "shap_summary.png", dpi=150), plt.close()

    # ---- answer-key test
    p = model.predict_proba(te)
    rho = spearmanr(p, te["true_hazard"]).correlation
    # sign check: usage decline should push risk UP when usage fell (use_trend low)
    sign_checks = {
        "use_trend (lower usage -> higher risk)": float(spearmanr(te["use_trend"], sv[:, model.features.index("use_trend")]).correlation),
        "unpaid_last3 (more unpaid -> higher risk)": float(spearmanr(te["unpaid_last3"], sv[:, model.features.index("unpaid_last3")]).correlation),
        "tenure_days (shorter tenure -> higher risk)": float(spearmanr(te["tenure_days"], sv[:, model.features.index("tenure_days")]).correlation),
        "days_since_use (longer -> higher risk)": float(spearmanr(te["days_since_use"], sv[:, model.features.index("days_since_use")]).correlation),
    }
    ev = json.loads((od / "evaluation_results.json").read_text())
    oracle_pr, model_pr = ev["ORACLE true hazard"]["pr_auc"], ev[model.name]["pr_auc"]
    report = {
        "model": model.name,
        "spearman_pred_vs_true_hazard": float(rho),
        "group_share_of_mean_abs_shap": share.to_dict(),
        "planted_drivers_present": sorted(PLANTED),
        "top_groups": list(share.index[:4]),
        "sign_checks_spearman_feature_vs_shap": sign_checks,
        "oracle_gap": {"model_pr_auc": model_pr, "oracle_pr_auc": oracle_pr,
                       "share_of_oracle_pr_auc": model_pr / oracle_pr,
                       "model_roc_auc": ev[model.name]["roc_auc"],
                       "oracle_roc_auc": ev["ORACLE true hazard"]["roc_auc"]},
        "note": "SHAP shows what the model uses, not what causes churn. The generator's "
                "dissatisfaction state is hidden, so behaviour is a noisy proxy.",
    }
    (od / "answer_key_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
