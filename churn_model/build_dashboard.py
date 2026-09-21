"""
Generate dashboard/churn_early_warning.html (offline, no external scripts) from pipeline outputs.

Pages: Overview | Risk list (EN/AR toggle) | Budget curve (what-if sliders) | Model performance |
       Network hand-off | Monitoring.
Only hashed IDs appear. Re-run after each scoring run: python -m churn_model.build_dashboard
"""
import json

import pandas as pd

from churn_model.common import ROOT, load_config, load_panel, out_dir

TEMPLATE = (ROOT / "churn_model" / "dashboard_template.html").read_text(encoding="utf-8")


def main():
    cfg = load_config()
    od = out_dir(cfg)
    j = lambda n: json.loads((od / n).read_text(encoding="utf-8"))
    ev, budget, ans = j("evaluation_results.json"), j("budget_summary.json"), j("answer_key_report.json")
    scoring, netv, train = j("scoring_summary.json"), j("network_handoff_validation.json"), j("training_results.json")

    panel = load_panel(cfg)
    trend = (panel.groupby("cutoff").agg(rate=("y", "mean"), n=("y", "size"), split=("split", "first"))
             .reset_index().to_dict("records"))

    scores = pd.read_parquet(od / "churn_scores.parquet")
    risk = (scores[scores["reason_1_en"].notna()].sort_values("probability", ascending=False).head(200)
            .assign(probability=lambda x: x["probability"].round(4),
                    expected_value_sdg=lambda x: x["expected_value_sdg"].round(0))
            .drop(columns=["cutoff_date"]))
    risk = risk.astype(object).where(risk.notna(), None).to_dict("records")

    wi = pd.read_parquet(od / "budget_whatif_latest.parquet")
    whatif = {"p": wi["p"].round(5).tolist(), "value": wi["value_sdg"].round(0).astype(int).tolist(),
              "y": wi["y"].astype(int).tolist()}
    net = pd.read_parquet(od / "network_handoff.parquet")
    net = net.assign(network_share=net["network_share"].round(4),
                     network_attributable_ev_sdg=net["network_attributable_ev_sdg"].round(0)).to_dict("records")
    hist = pd.read_csv(od / "drift_history.csv").to_dict("records") if (od / "drift_history.csv").exists() else []

    data = {"eval": ev, "budget": budget, "answer": ans, "scoring": scoring, "network_validation": netv,
            "training": train, "trend": trend, "risk": risk, "whatif": whatif, "network": net,
            "drift_history": hist, "assumptions": cfg["decision"]}
    html = TEMPLATE.replace("/*__DATA__*/null", json.dumps(data, default=float, ensure_ascii=False))
    out = ROOT / "dashboard" / "churn_early_warning.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
