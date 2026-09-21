"""
Refresh dashboard/churn_dashboard.html (root-cause view) from the current analytics mart.

The page embeds its data as JavaScript constants. This script rewrites RC_DATA, REGION_DATA and
PLAN_DATA and the four KPI tiles from src/data/processed/mart/*.parquet, so the page always matches
the data the pipeline last produced.
Run: python dashboard/refresh_churn_dashboard.py
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MART = ROOT / "src" / "data" / "processed" / "mart"
PAGE = ROOT / "dashboard" / "churn_dashboard.html"


def js(records):
    return "[\n" + ",\n".join("  " + json.dumps(r) for r in records) + "\n]"


def main():
    rc = pd.read_parquet(MART / "mart_root_cause_summary.parquet")
    reg = pd.read_parquet(MART / "mart_churn_by_region.parquet").sort_values("churn_rate_pct", ascending=False)
    plan = pd.read_parquet(MART / "mart_churn_by_plan.parquet")
    total = int(reg["total_customers"].sum())
    churned = int(reg["churned_customers"].sum())
    rc_churned = rc[rc["root_cause_category"] != "active"][["root_cause_category", "customer_count", "pct_of_churned_only"]]
    top = reg.iloc[0]

    html = PAGE.read_text(encoding="utf-8")

    def swap(name, records):
        nonlocal html
        pat = re.compile(rf"const {name} = \[.*?\];", re.S)
        assert pat.search(html), name
        html = pat.sub(lambda m: f"const {name} = {js(records)};", html, count=1)

    swap("RC_DATA", rc_churned.round(2).to_dict("records"))
    swap("REGION_DATA", reg[["region_name", "total_customers", "churned_customers", "churn_rate_pct",
                             "avg_monthly_arpu_sdg", "avg_days_silent_churned"]].round(2).to_dict("records"))
    swap("PLAN_DATA", plan[["plan_type", "total_customers", "churned_customers", "churn_rate_pct",
                            "avg_monthly_arpu_sdg"]].round(2).to_dict("records"))

    tiles = [("Total Customers", f"{total:,}", "Simulation cohort (hidden-process v2 data)"),
             ("Churned", f"{churned:,}", f"{churned / total:.2%} churned during 2024"),
             ("Active", f"{total - churned:,}", f"{(total - churned) / total:.2%} retained"),
             ("Highest-Risk Region", top["region_name"], f"{top['churn_rate_pct']:.2f}% churn rate")]
    for label, value, sub in tiles:
        pat = re.compile(rf'(<div class="label">{re.escape(label)}</div>\s*<div class="value"[^>]*>)[^<]*(</div>\s*<div class="sub">)[^<]*(</div>)')
        assert pat.search(html), label
        html = pat.sub(lambda m: f"{m.group(1)}{value}{m.group(2)}{sub}{m.group(3)}", html, count=1)

    note = ('<p style="color:#f59e0b;font-size:12px;margin-top:6px">Root-cause view (rule engine, explains churn after it '
            'happens). The predictive early-warning view is <code>churn_early_warning.html</code>. '
            'Simulated data. On this data the rule engine labels almost every churner behavioral.</p>')
    if "Root-cause view (rule engine" not in html:
        html = html.replace("</header>", note + "\n</header>", 1)
    PAGE.write_text(html, encoding="utf-8")
    print(f"refreshed {PAGE.name}: {total:,} customers, {churned:,} churned, top region {top['region_name']}")


if __name__ == "__main__":
    main()
