"""
Population Stability Index (PSI) drift check on the score distribution and the top-5 features.

PSI < 0.1 stable, 0.1-0.25 watch, > 0.25 investigate.
Retrain trigger: score PSI > 0.25, OR score PSI > 0.1 for two consecutive months,
OR PR-AUC dropping materially once labels mature (checked separately, not here).
"""
import numpy as np
import pandas as pd


def psi(expected, actual, bins=10):
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                                   # near-constant feature: no drift signal
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected) + 1e-6
    a = np.histogram(actual, edges)[0] / len(actual) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))


def status(v):
    return "stable" if v < 0.1 else "watch" if v < 0.25 else "investigate"


# tenure_days advances by construction (every subscriber ages), so PSI on it always alarms
CLOCK_FEATURES = {"tenure_days"}


def top_features(model, k=5):
    imp = getattr(model.base, "feature_importances_", None)
    if imp is None:                                      # logistic: |coef|
        imp = np.abs(model.base.steps[-1][1].coef_[0])
    ranked = [model.features[i] for i in np.argsort(-imp)]
    return [f for f in ranked if f not in CLOCK_FEATURES][:k]


def drift_report(model, reference_df, current_df, history=None):
    """reference/current: DataFrames with model.features. history: list of prior score-PSI values."""
    ref_p, cur_p = model.predict_proba(reference_df), model.predict_proba(current_df)
    score_psi = psi(ref_p, cur_p)
    feats = {f: psi(reference_df[f].fillna(0), current_df[f].fillna(0)) for f in top_features(model)}
    prior = (history or [])[-1:]
    trigger = score_psi > 0.25 or (score_psi > 0.1 and bool(prior) and prior[-1] > 0.1)
    return {"score_psi": score_psi, "score_status": status(score_psi),
            "feature_psi": feats, "feature_status": {k: status(v) for k, v in feats.items()},
            "retrain_trigger": bool(trigger)}
