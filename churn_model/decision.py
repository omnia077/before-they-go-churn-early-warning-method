"""
Decision layer: turn calibrated churn probabilities into "who do we contact, and what does it buy?".

    ev = p * value * SAVE_RATE - COST_PER_CONTACT          (all assumptions in config.yaml)

value = average monthly spend x horizon. Prepaid: mean monthly recharge over 90 days
(rech_amt_90d * rech_90d / 3). Postpaid: static monthly ARPU (bills are not in the panel).
`value` is what the subscriber is worth IF RETAINED: it is NOT multiplied by survival probability.

Outputs (latest test cutoff = the "this month" decision, plus pooled test months):
  budget_curve.png, budget_summary.json, docs/figures/budget_curve.png, dashboard data.
Run: python -m churn_model.decision
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from churn_model.common import ROOT, load_config, load_panel, out_dir, split


MONTH = {2: "Mar", 3: "Apr", 4: "May", 5: "Jun", 6: "Jul", 7: "Aug", 8: "Sep", 9: "Oct", 10: "Nov"}


def attach_value(df, cfg):
    d = cfg["decision"]
    spend = np.where(df["is_prepaid"] == 1, df["rech_amt_90d"] * df["rech_90d"] / 3.0, df["monthly_arpu_sdg"])
    out = df.copy()
    out["monthly_spend_sdg"] = spend
    out["value_sdg"] = spend * d["value_horizon_months"]
    return out


def add_ev(df, p, cfg):
    d = cfg["decision"]
    out = df.copy()
    out["p"] = p
    out["ev"] = out["p"] * out["value_sdg"] * d["save_rate"] - d["cost_per_contact_sdg"]
    return out


def curve(df, rank_col, cfg, realised=True):
    """Cumulative net value contacting subscribers in `rank_col` order."""
    d = cfg["decision"]
    o = df.sort_values(rank_col, ascending=False)
    exp = np.cumsum(o["ev"].values)
    out = {"n": np.arange(1, len(o) + 1), "expected_net": exp}
    if realised and "y" in o:
        out["realised_net"] = np.cumsum(o["y"].values * o["value_sdg"].values * d["save_rate"]
                                        - d["cost_per_contact_sdg"])
    return out


def main():
    cfg = load_config()
    od = out_dir(cfg)
    model = joblib.load(od / "models" / "primary.joblib")
    _, _, te = split(load_panel(cfg), cfg)
    te = add_ev(attach_value(te, cfg), model.predict_proba(te), cfg).reset_index(drop=True)
    latest = te["cutoff"].max()
    cur = te[te["cutoff"] == latest].reset_index(drop=True)

    summ = {"assumptions": cfg["decision"], "cutoff_index": int(latest),
            "subscribers_scored": int(len(cur)), "value_note": "simulated; assumptions, not findings"}
    by_ev, by_p = curve(cur, "ev", cfg), curve(cur, "p", cfg)
    n_star = int(np.argmax(by_ev["expected_net"]) + 1)
    n_star_p = int(np.argmax(by_p["expected_net"]) + 1)
    ev_top = cur.sort_values("ev", ascending=False).head(n_star)
    p_top = cur.sort_values("p", ascending=False).head(n_star_p)
    k10 = int(len(cur) * 0.10)
    top10 = cur.sort_values("p", ascending=False).head(k10)
    summ.update({
        "recommended_contacts_by_ev": n_star,
        "recommended_share_by_ev": n_star / len(cur),
        "expected_net_value_at_optimum_sdg": float(by_ev["expected_net"][n_star - 1]),
        "realised_net_value_at_optimum_sdg": float(by_ev["realised_net"][n_star - 1]),
        "expected_churners_reached_at_optimum": float(ev_top["p"].sum()),
        "actual_churners_reached_at_optimum": int(ev_top["y"].sum()),
        "actual_churners_total": int(cur["y"].sum()),
        "prob_ranking": {"peak_contacts": n_star_p,
                         "expected_net_sdg": float(by_p["expected_net"][n_star_p - 1]),
                         "realised_net_sdg": float(by_p["realised_net"][n_star_p - 1])},
        "ev_vs_prob_ranking_same_budget": {
            "budget": n_star,
            "expected_net_ev_rank": float(by_ev["expected_net"][n_star - 1]),
            "expected_net_prob_rank": float(by_p["expected_net"][n_star - 1]),
            "realised_net_ev_rank": float(by_ev["realised_net"][n_star - 1]),
            "realised_net_prob_rank": float(by_p["realised_net"][n_star - 1])},
        "overlap_top_n_ev_vs_prob": float(len(set(ev_top["customer_id"]) & set(
            cur.sort_values("p", ascending=False).head(n_star)["customer_id"])) / n_star),
        "top10pct_capture_rate": float(top10["y"].sum() / cur["y"].sum()),
        "top10pct_contacts": k10,
        "top10pct_expected_net_sdg": float(top10["ev"].sum()),
        "top10pct_realised_net_sdg": float((top10["y"] * top10["value_sdg"] * cfg["decision"]["save_rate"]
                                            - cfg["decision"]["cost_per_contact_sdg"]).sum()),
    })
    # pooled test months, ranked within each month
    pooled = []
    for c, g in te.groupby("cutoff"):
        cv = curve(g.reset_index(drop=True), "ev", cfg)
        pooled.append({"cutoff": int(c), "n": int(len(g)), "optimum_contacts": int(np.argmax(cv["expected_net"]) + 1),
                       "expected_net_sdg": float(cv["expected_net"].max()),
                       "realised_net_at_optimum_sdg": float(cv["realised_net"][int(np.argmax(cv["expected_net"]))])})
    summ["by_test_month"] = pooled
    (od / "budget_summary.json").write_text(json.dumps(summ, indent=2))

    # what-if data for the dashboard (latest cutoff)
    cur[["sub_hash", "region", "p", "value_sdg", "y"]].assign(
        p=lambda x: x["p"].round(5), value_sdg=lambda x: x["value_sdg"].round(0)
    ).to_parquet(od / "budget_whatif_latest.parquet", index=False)

    fig_dir = ROOT / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(by_ev["n"] / 1000, by_ev["expected_net"] / 1000, label="ranked by expected value", color="#d62728")
    ax.plot(by_p["n"] / 1000, by_p["expected_net"] / 1000, label="ranked by probability", color="#1f77b4", ls="--")
    ax.plot(by_ev["n"] / 1000, by_ev["realised_net"] / 1000, label="realised (EV rank, simulated outcomes)",
            color="#2ca02c", alpha=.8)
    ax.axvline(n_star / 1000, color="grey", ls=":")
    ax.set_xlabel("Subscribers contacted (thousands)"), ax.set_ylabel("Cumulative net value (thousand SDG)")
    ax.set_title(f"Budget curve, {MONTH[int(latest)]} cutoff (simulated; save rate "
                 f"{cfg['decision']['save_rate']:.0%}, cost {cfg['decision']['cost_per_contact_sdg']} SDG)", fontsize=9)
    ax.legend(fontsize=8), fig.tight_layout(), fig.savefig(fig_dir / "budget_curve.png", dpi=150), plt.close(fig)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
