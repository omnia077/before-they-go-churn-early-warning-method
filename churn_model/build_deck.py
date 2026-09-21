"""Build docs/churn_early_warning_deck.pptx (5 slides) from the pipeline outputs. Run: python -m churn_model.build_deck"""
import json

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

from churn_model.common import ROOT, load_config, out_dir

NAVY, GREY, RED = RGBColor(0x0F, 0x17, 0x2A), RGBColor(0x55, 0x5F, 0x73), RGBColor(0xC0, 0x39, 0x2B)


def text(slide, x, y, w, h, lines, size=16, bold_first=False, color=None):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size = Pt(size)
        p.font.bold = bold_first and i == 0
        p.font.color.rgb = color or NAVY
        p.space_after = Pt(6)
    return tb


def slide(prs, title, subtitle=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    text(s, 0.5, 0.3, 12.3, 0.8, [title], size=28, bold_first=True)
    if subtitle:
        text(s, 0.5, 1.05, 12.3, 0.5, [subtitle], size=14, color=GREY)
    text(s, 0.5, 7.0, 12.3, 0.4, ["Simulated data: demonstrates the method, not real-world performance."], size=10, color=GREY)
    return s


def main():
    cfg = load_config()
    od = out_dir(cfg)
    ev = json.loads((od / "evaluation_results.json").read_text())
    bd = json.loads((od / "budget_summary.json").read_text())
    ps = json.loads((od / "pilot_sample_sizes.json").read_text())
    ans = json.loads((od / "answer_key_report.json").read_text())
    fig = ROOT / "docs" / "figures"
    g = ev["lightgbm"]
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.33), Inches(7.5)

    s = slide(prs, "Churn early-warning and retention budget", "Should we spend this month's budget on N subscribers, and which ones?")
    text(s, 0.5, 1.8, 6.2, 4.5, [
        "Predict next-month churn, explain it in English and Arabic, and turn scores into a contact decision.",
        "The rule engine explains churn after it happens; this model predicts it before. They are complementary.",
        f"Answer: contact about {bd['recommended_share_by_ev']:.1%} of the base, ranked by expected value, and stop where the next contact no longer pays.",
        "20,000 simulated subscribers, hidden churn process, validated out of time and against a known answer key."], size=16)
    s.shapes.add_picture(str(fig / "budget_curve.png"), Inches(7.0), Inches(1.9), width=Inches(5.9))

    s = slide(prs, "Method: no leakage, out-of-time, checked against truth")
    text(s, 0.5, 1.4, 6.3, 5.4, [
        "Data: hidden dissatisfaction drives usage, recharge, tickets and bills; churn is a random draw from a hazard.",
        "22 as-of features per subscriber per month-end; label = churn in the next month.",
        "Train Mar-Jul, validate Aug, test Sep-Nov (Kassala outage inside the test window).",
        "Logistic vs LightGBM, Platt calibration on pooled out-of-fold scores, 200 customer-level bootstrap resamples.",
        "Excluded: persona, label-anchored fields, gender, age, msisdn; hashed IDs only."], size=15)
    s.shapes.add_picture(str(fig / "shap_bar.png"), Inches(7.2), Inches(1.5), width=Inches(5.6))

    s = slide(prs, "Results: about 5x lift, and we say how far from the ceiling")
    rows = [("Model", "PR-AUC", "ROC-AUC", "Top-10% lift")] + [
        (n, f"{ev[n]['pr_auc']:.3f}", f"{ev[n]['roc_auc']:.3f}", f"{ev[n]['top10_lift']:.2f}")
        for n in ("no-skill", "rule heuristic", "logistic", "lightgbm", "ORACLE true hazard")]
    tbl = s.shapes.add_table(len(rows), 4, Inches(0.5), Inches(1.5), Inches(6.4), Inches(2.6)).table
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            c = tbl.cell(i, j)
            c.text = v
            c.text_frame.paragraphs[0].font.size = Pt(13)
    text(s, 0.5, 4.4, 6.4, 2.5, [
        f"Boosting ties logistic: the planted signal is near-linear (CIs overlap).",
        f"Recovers about half of the oracle PR-AUC; Spearman with true hazard {ans['spearman_pred_vs_true_hazard']:.2f}.",
        "SHAP leans on the planted drivers, with the right signs. It shows what the model uses, not causes.",
        "Region is uneven (Kassala over-flagged); age bands are even."], size=13)
    s.shapes.add_picture(str(fig / "gains.png"), Inches(7.2), Inches(1.5), width=Inches(5.6))

    s = slide(prs, "Decision: rank by expected value, not probability")
    ev_r = bd["ev_vs_prob_ranking_same_budget"]
    text(s, 0.5, 1.5, 6.3, 5.3, [
        f"Nov backtest, {bd['recommended_contacts_by_ev']:,} contacts: net {ev_r['realised_net_ev_rank']:,.0f} SDG (value ranking) vs {ev_r['realised_net_prob_rank']:,.0f} SDG (probability ranking).",
        f"Only {bd['overlap_top_n_ev_vs_prob']:.0%} of the two lists overlap: some high-risk subscribers are low value.",
        f"Expected net ({ev_r['expected_net_ev_rank']:,.0f} SDG) overstates realised in a quiet month: plan on the realised figure.",
        "Assumptions (labelled): save rate 20%, cost 20 SDG per contact, 6-month value horizon.",
        "Network hand-off ranks Nyala and Kassala first at every planted-outage cutoff (Kassala is out of sample)."], size=15)
    s.shapes.add_picture(str(fig / "network_handoff.png"), Inches(7.2), Inches(1.6), width=Inches(5.6))

    s = slide(prs, "Deployment, pilot and risks")
    n = ps["n"]
    text(s, 0.5, 1.5, 6.3, 5.3, [
        "Monthly Airflow DAG: features, validation, registered model, scores with reasons, PSI drift check, retrain gate.",
        "Monitor already alerts on regional ticket rate once the outages end.",
        f"Pilot: 50/50 randomised top-decile prepaid holdout; baseline churn {ps['p1']:.1%}.",
        f"Sample per arm for a 15 / 20 / 25 / 30% relative reduction: {n['0.15']:,} / {n['0.2']:,} / {n['0.25']:,} / {n['0.3']:,}."], size=15)
    text(s, 7.0, 1.5, 5.8, 5.3, [
        "Risks",
        "Simulated data; real performance unknown.",
        "Expected value still overstates realised (about 17% in the Nov backtest).",
        "Regional outages distort scores.",
        "Only ticket signals for the network, no cell KPIs.",
        "Next: pilot, real operator data, native review of Arabic text."], size=15, bold_first=True, color=RED)
    prs.save(ROOT / "docs" / "churn_early_warning_deck.pptx")
    print("saved docs/churn_early_warning_deck.pptx")


if __name__ == "__main__":
    main()
