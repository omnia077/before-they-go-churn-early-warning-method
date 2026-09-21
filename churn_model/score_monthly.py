"""
Monthly scoring pipeline (the body of the Airflow DAG `churn_scoring_monthly`).

  1 build_features   PySpark as-of features for the cutoff
  2 validate_inputs  row count, null rates, schema
  3 score            registered MLflow model (falls back to the joblib copy), calibrated probability
  4 explain          SHAP for the top 20% by probability only, bilingual top-3 reasons
  5 write            churn_scores -> parquet + DuckDB mart (idempotent per cutoff)
  6 drift            PSI on score + top-5 features vs the training reference
  7 log              MLflow run + dashboard export

Only hashed IDs are written. Run: python -m churn_model.score_monthly --as-of 2024-12-31
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

from churn_model.as_of_features import FEATURE_COLUMNS, build_as_of_features, get_spark, load_raw
from churn_model.common import ROOT, hash_id, load_config, load_panel, out_dir, path, split
from churn_model.decision import add_ev, attach_value
from churn_model.drift import drift_report
from churn_model.explain import group_shap, shap_values
from churn_model.reasons import build_reasons

ACTIVE_MAX_DAYS_SINCE_USE = 60      # production proxy for "not already deactivated"
EXPLAIN_TOP_SHARE = 0.20


def build_features(cfg, as_of):
    spark = get_spark("churn_scoring_monthly")
    spark.sparkContext.setLogLevel("ERROR")
    try:
        X = build_as_of_features(load_raw(spark, str(path(cfg, "raw"))), as_of, cfg).toPandas()
    finally:
        spark.stop()
    return X


def validate_inputs(X, cfg, min_rows=5000, max_null=0.01):
    problems = []
    if len(X) < min_rows:
        problems.append(f"row count {len(X)} < {min_rows}")
    missing = set(FEATURE_COLUMNS) - set(X.columns)
    if missing:
        problems.append(f"missing columns {sorted(missing)}")
    else:
        nulls = X[FEATURE_COLUMNS].isna().mean()
        bad = nulls[nulls > max_null]
        if len(bad):
            problems.append(f"null rate above {max_null:.0%}: {bad.round(3).to_dict()}")
    if X["customer_id"].duplicated().any():
        problems.append("duplicate customer_id rows")
    if problems:
        raise ValueError("input validation failed: " + "; ".join(problems))
    return {"rows": int(len(X)), "max_null_rate": float(X[FEATURE_COLUMNS].isna().mean().max())}


def load_model(cfg):
    od = out_dir(cfg)
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{(od / 'mlflow.db').as_posix()}")
        return mlflow.sklearn.load_model("models:/churn_model/latest"), "mlflow_registry:churn_model/latest"
    except Exception as e:
        print("registry load failed, using joblib copy:", str(e)[:120])
        return joblib.load(od / "models" / "primary.joblib"), "joblib:primary"


def score(X, model, cfg, as_of):
    df = X[X["days_since_use"] <= ACTIVE_MAX_DAYS_SINCE_USE].copy().reset_index(drop=True)
    df["cutoff"] = int(pd.Timestamp(as_of).month - 1)
    df = add_ev(attach_value(df, cfg), model.predict_proba(df), cfg)
    df["risk_rank"] = df["p"].rank(ascending=False, method="first")
    df["decile"] = (np.ceil(df["risk_rank"] / len(df) * 10)).astype(int).clip(1, 10)   # 1 = highest risk
    return df


def explain(df, model):
    n = int(len(df) * EXPLAIN_TOP_SHARE)
    top = df.nsmallest(n, "risk_rank")
    gs = group_shap(shap_values(model, top), model.features)
    reasons = build_reasons(top, gs.set_index(top.index))
    return df.join(reasons)


def write(df, cfg, as_of):
    import duckdb
    ids = pd.read_parquet(path(cfg, "raw").parent / "sftp_drop" / "customers.parquet")[["customer_id", "msisdn"]]
    ids["sub_hash"] = hash_id(ids["msisdn"])
    out = df.merge(ids[["customer_id", "sub_hash"]], on="customer_id", how="left")
    out["cutoff_date"] = pd.Timestamp(as_of)
    cols = (["sub_hash", "cutoff_date", "region", "p", "decile", "ev"]
            + [f"reason_{k}_{s}" for k in (1, 2, 3) for s in ("group", "en", "ar")])
    scores = out[cols].rename(columns={"p": "probability", "ev": "expected_value_sdg"})
    assert "customer_id" not in scores.columns and "msisdn" not in scores.columns
    od = out_dir(cfg)
    scores.to_parquet(od / "churn_scores.parquet", index=False)
    mart = ROOT / "src" / "data" / "processed" / "analytics_mart.duckdb"
    mart.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(mart))
    con.execute("CREATE TABLE IF NOT EXISTS churn_scores AS SELECT * FROM scores WHERE 1=0")
    con.execute("DELETE FROM churn_scores WHERE cutoff_date = ?", [pd.Timestamp(as_of)])
    con.execute("INSERT INTO churn_scores SELECT * FROM scores")
    n = con.execute("SELECT count(*) FROM churn_scores").fetchone()[0]
    con.close()
    return scores, n


def drift(model, cfg, current_df):
    tr, va, _ = split(load_panel(cfg), cfg)
    ref = pd.concat([tr, va])
    hist_path = out_dir(cfg) / "drift_history.csv"
    hist = pd.read_csv(hist_path)["score_psi"].tolist() if hist_path.exists() else []
    rep = drift_report(model, ref, current_df, hist)
    return rep, hist_path, hist


# ---- stage functions: each Airflow task calls one; data moves between them as parquet files
def stage_build_features(as_of):
    cfg = load_config()
    build_features(cfg, as_of).to_parquet(out_dir(cfg) / "stage_features.parquet", index=False)


def stage_validate_inputs(as_of):
    cfg = load_config()
    return validate_inputs(pd.read_parquet(out_dir(cfg) / "stage_features.parquet"), cfg)


def stage_score_explain_write(as_of):
    cfg = load_config()
    od = out_dir(cfg)
    X = pd.read_parquet(od / "stage_features.parquet")
    model, source = load_model(cfg)
    df = explain(score(X, model, cfg, as_of), model)
    _, n_rows = write(df, cfg, as_of)
    df.to_parquet(od / "stage_scored.parquet", index=False)
    return {"model_source": source, "model": model.name, "rows_scored": int(len(df)),
            "dropped_inactive": int(len(X) - len(df)), "rows_in_mart_table": int(n_rows),
            "explained_rows": int(df["reason_1_en"].notna().sum()), "mean_probability": float(df["p"].mean())}


def stage_drift_and_log(as_of, prior=None):
    cfg = load_config()
    od = out_dir(cfg)
    model, source = load_model(cfg)
    df = pd.read_parquet(od / "stage_scored.parquet")
    rep, hist_path, _ = drift(model, cfg, df)
    row = pd.DataFrame([{"cutoff_date": as_of, "score_psi": rep["score_psi"]}])
    hist = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()
    hist = pd.concat([hist[hist["cutoff_date"] != as_of] if len(hist) else hist, row])
    hist.to_csv(hist_path, index=False)                         # idempotent per cutoff
    summary = {"as_of": as_of, **(prior or {}), "drift": rep}
    (od / "scoring_summary.json").write_text(json.dumps(summary, indent=2))
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{(od / 'mlflow.db').as_posix()}")
        mlflow.set_experiment("churn_scoring_monthly")
        with mlflow.start_run(run_name=f"score_{as_of}"):
            mlflow.log_params({"as_of": as_of, "model_source": source})
            mlflow.log_metrics({"rows_scored": len(df), "score_psi": rep["score_psi"],
                                "mean_probability": float(df["p"].mean())})
            mlflow.log_dict(summary, "scoring_summary.json")
    except Exception as e:
        print("MLflow logging skipped:", e)
    return summary


def run(as_of):
    stage_build_features(as_of)
    val = stage_validate_inputs(as_of)
    scored = stage_score_explain_write(as_of)
    summary = stage_drift_and_log(as_of, {"validation": val, **scored})
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default="2024-12-31")
    run(ap.parse_args().as_of)
