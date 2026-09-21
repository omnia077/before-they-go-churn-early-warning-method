"""Unit tests for drift, decision, reasons (no Spark, no data needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from churn_model.decision import add_ev, attach_value, curve
from churn_model.drift import psi, status
from churn_model.reasons import REASON_GROUPS, ar_days, build_reasons

CFG = {"decision": {"save_rate": 0.2, "cost_per_contact_sdg": 20, "value_horizon_months": 6}}


def test_psi_identical_is_zero_and_shift_is_flagged():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x) < 1e-3
    assert psi(x, x + 1.5) > 0.25
    assert status(0.05) == "stable" and status(0.15) == "watch" and status(0.3) == "investigate"


def test_expected_value_and_curve_peak():
    df = pd.DataFrame({"is_prepaid": [1, 1, 0], "rech_amt_90d": [100.0, 100.0, 0.0],
                       "rech_90d": [3, 3, 0], "monthly_arpu_sdg": [0.0, 0.0, 1000.0],
                       "y": [1, 0, 0]})
    df = attach_value(df, CFG)
    assert df["monthly_spend_sdg"].tolist() == [100.0, 100.0, 1000.0]
    assert df["value_sdg"].tolist() == [600.0, 600.0, 6000.0]
    out = add_ev(df, np.array([0.5, 0.01, 0.1]), CFG)
    assert np.allclose(out["ev"], [0.5 * 600 * .2 - 20, 0.01 * 600 * .2 - 20, 0.1 * 6000 * .2 - 20])
    c = curve(out, "ev", CFG)
    assert c["n"][np.argmax(c["expected_net"])] == 2          # third subscriber has negative EV


def test_arabic_day_agreement():
    assert ar_days(1) == "يوم واحد" and ar_days(2) == "يومين"
    assert ar_days(5) == "5 أيام" and ar_days(30) == "30 يوماً"


def test_reasons_top3_positive_only_and_no_repeats():
    df = pd.DataFrame([{"use_30d": 2, "use_prev60d": 20, "days_since_rech": 45, "tix_30d": 3,
                        "unpaid_last3": 2, "tenure_days": 40, "days_since_use": 12}])
    gs = pd.DataFrame([{g: v for g, v in zip(REASON_GROUPS, [0.9, 0.5, 0.3, -0.2, 0.1, -0.5])}])
    gs["plan_profile"] = 5.0                                   # must never become a reason
    r = build_reasons(df, gs).iloc[0]
    groups = [r["reason_1_group"], r["reason_2_group"], r["reason_3_group"]]
    assert groups == ["usage_decline", "recharge_gap", "network_complaints"]
    assert "Usage fell 80%" in r["reason_1_en"] and "45 days" in r["reason_2_en"]
    assert "3 network complaints" in r["reason_3_en"]
    assert all(len(r[f"reason_{k}_en"].split()) <= 12 for k in (1, 2, 3))


def test_platt_calibrator_is_monotonic_and_matches_base_rate():
    from churn_model.model import PlattCalibrator
    rng = np.random.default_rng(0)
    true_p = rng.beta(1, 40, 20000)                      # base rate about 2.4%
    y = (rng.random(20000) < true_p).astype(int)
    raw = np.clip(true_p * 1.8, 1e-6, 0.99)              # a badly over-confident raw score
    cal = PlattCalibrator().fit(raw, y)
    grid = np.linspace(0.001, 0.5, 200)
    assert (np.diff(cal.predict(grid)) > 0).all(), "must be strictly increasing (never ties or reorders)"
    assert abs(cal.predict(raw).mean() - y.mean()) < 0.003
    assert abs(raw.mean() - y.mean()) > 0.01             # the raw score really was off
