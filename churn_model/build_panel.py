"""
Build `churn_panel`: one row per subscriber per month-end cutoff, as-of features + label.

Label  : churn_date falls in the calendar month after the cutoff.
Exclude: subscribers already churned on or before the cutoff.
Ground truth (churn_labels) is used ONLY for the label and that exclusion.
`true_hazard` goes to a separate churn_panel_truth file, for validation only.

Run:  set CHURN_HASH_SALT=<secret>  &&  python -m churn_model.build_panel
"""
import numpy as np
import pandas as pd

from churn_model.as_of_features import FEATURE_COLUMNS, build_as_of_features, get_spark, load_raw
from churn_model.common import ROOT, hash_id, load_config, month_end, out_dir, path


def main():
    cfg = load_config()
    td = cfg["time_design"]
    split_of = {c: s for s in ("train", "validation", "test") for c in td[s]}
    gt = path(cfg, "ground_truth")
    labels = pd.read_parquet(gt / "churn_labels.parquet")
    labels["churn_date"] = pd.to_datetime(labels["churn_date"])
    latent = pd.read_parquet(gt / "latent_state.parquet")[["customer_id", "month", "true_hazard"]]

    msisdn = pd.read_parquet(path(cfg, "raw").parent / "sftp_drop" / "customers.parquet")[["customer_id", "msisdn"]]
    msisdn["sub_hash"] = hash_id(msisdn["msisdn"])          # salted hash, raw number dropped
    id_map = msisdn[["customer_id", "sub_hash"]]

    spark = get_spark("build_panel")
    spark.sparkContext.setLogLevel("ERROR")
    tables = load_raw(spark, str(path(cfg, "raw")))

    frames, truth = [], []
    for c in sorted(split_of):
        C = month_end(cfg, c)
        X = build_as_of_features(tables, C.strftime("%Y-%m-%d"), cfg).toPandas()
        X = X.merge(labels[["customer_id", "churn_date"]], on="customer_id", how="left")
        X = X[~(X["churn_date"].notna() & (X["churn_date"] <= C))].copy()
        nxt = month_end(cfg, c + 1)
        X["y"] = (X["churn_date"].notna() & (X["churn_date"] > C) & (X["churn_date"] <= nxt)).astype(int)
        X["cutoff"], X["cutoff_date"], X["split"] = c, C, split_of[c]
        hz = latent[latent["month"] == c + 2][["customer_id", "true_hazard"]]
        truth.append(X[["customer_id", "cutoff"]].merge(hz, on="customer_id", how="left"))
        frames.append(X.drop(columns=["churn_date"]))
        print(f"cutoff {c:2d} {C.date()} | rows {len(X):6,} | positives {int(X['y'].sum()):4d} "
              f"| rate {X['y'].mean():.2%} | split {split_of[c]}")
    spark.stop()

    panel = pd.concat(frames, ignore_index=True).merge(id_map, on="customer_id", how="left")
    cols = ["customer_id", "sub_hash", "cutoff", "cutoff_date", "split", "region", "y"] + FEATURE_COLUMNS
    panel = panel[cols]
    assert panel["sub_hash"].notna().all()
    assert not panel.duplicated(["customer_id", "cutoff"]).any()
    assert "msisdn" not in panel.columns
    od = out_dir(cfg)
    panel.to_parquet(od / "churn_panel.parquet", index=False)
    pd.concat(truth).to_parquet(od / "churn_panel_truth.parquet", index=False)
    print(f"\nwrote {len(panel):,} rows -> {od / 'churn_panel.parquet'}")


if __name__ == "__main__":
    main()
