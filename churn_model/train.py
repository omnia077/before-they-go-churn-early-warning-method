"""
Train, tune, calibrate and register the churn models.

  logistic  : StandardScaler + LogisticRegression
  lightgbm  : random search on the August validation cutoff (early stopping),
              with and without scale_pos_weight
  calibrate : Platt scaling fitted on pooled out-of-fold predictions over the train+validation
              cutoffs (not on one month), applied to test
  cv        : GroupKFold by customer on the training cutoffs (no customer in two folds)
  ablation  : LightGBM + region one-hot (kept out of the main model on purpose)

Everything is logged to MLflow (SQLite backend) and the primary model is registered.
Run: python -m churn_model.train
"""
import json

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from churn_model.as_of_features import FEATURE_COLUMNS
from churn_model.common import ROOT, load_config, load_panel, out_dir, split
from churn_model.model import CalibratedChurnModel, PlattCalibrator


def region_dummies(df, regions):
    return pd.get_dummies(df["region"]).reindex(columns=regions, fill_value=0).add_prefix("reg_").astype(int)


def make_lgbm(params, seed, n_estimators=2000):
    return lgb.LGBMClassifier(objective="binary", n_estimators=n_estimators, random_state=seed,
                              deterministic=True, force_row_wise=True, n_jobs=2, verbose=-1, **params)


def sample_params(rng):
    return {"learning_rate": float(rng.choice([0.02, 0.03, 0.05, 0.08])),
            "num_leaves": int(rng.choice([4, 8, 15, 31])),
            "min_child_samples": int(rng.choice([20, 50, 100, 200])),
            "subsample": float(rng.choice([0.7, 0.85, 1.0])), "subsample_freq": 1,
            "colsample_bytree": float(rng.choice([0.6, 0.8, 1.0])),
            "reg_lambda": float(rng.choice([0.0, 1.0, 5.0]))}


def main():
    cfg = load_config()
    seed = cfg["seed"]
    rng = np.random.default_rng(seed)
    tr, va, te = split(load_panel(cfg), cfg)
    feats = list(FEATURE_COLUMNS)
    Xtr, ytr = tr[feats].fillna(0), tr["y"].values
    Xva, yva = va[feats].fillna(0), va["y"].values
    od = out_dir(cfg)
    (od / "models").mkdir(exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{(od / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("churn_early_warning")
    results = {}

    with mlflow.start_run(run_name="train_all"):
        mlflow.log_params({"seed": seed, "n_features": len(feats),
                           "train_cutoffs": cfg["time_design"]["train"],
                           "validation_cutoffs": cfg["time_design"]["validation"],
                           "test_cutoffs": cfg["time_design"]["test"], "train_rows": len(tr)})
        mlflow.log_dict({"features": feats}, "features.json")

        # ---- logistic baseline
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=seed))
        lr.fit(Xtr, ytr)
        p_lr = lr.predict_proba(Xva)[:, 1]
        results["logistic"] = {"val_pr_auc": average_precision_score(yva, p_lr),
                               "val_roc_auc": roc_auc_score(yva, p_lr)}

        # ---- LightGBM random search on validation, with/without scale_pos_weight
        pos_ratio = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        best = None
        trials = []
        for spw_name, spw in [("none", 1.0), ("sqrt", float(np.sqrt(pos_ratio))), ("full", float(pos_ratio))]:
            for _ in range(8):
                params = sample_params(rng)
                m = make_lgbm({**params, "scale_pos_weight": spw}, seed)
                m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="binary_logloss",
                      callbacks=[lgb.early_stopping(50, verbose=False)])
                p = m.predict_proba(Xva)[:, 1]
                ap = average_precision_score(yva, p)
                trials.append({"spw": spw_name, "pr_auc": ap, "iters": m.best_iteration_})
                if best is None or ap > best["pr_auc"]:
                    best = {"pr_auc": ap, "params": {**params, "scale_pos_weight": spw},
                            "spw_name": spw_name, "iters": int(m.best_iteration_ or 100), "model": m}
        by_spw = pd.DataFrame(trials).groupby("spw")["pr_auc"].max().round(4).to_dict()
        print(f"best validation PR-AUC by scale_pos_weight setting: {by_spw}")
        gbm = best["model"]
        p_gb = gbm.predict_proba(Xva)[:, 1]
        results["lightgbm"] = {"val_pr_auc": best["pr_auc"], "val_roc_auc": roc_auc_score(yva, p_gb),
                               "params": best["params"], "n_trees": best["iters"],
                               "best_by_scale_pos_weight": by_spw}
        mlflow.log_params({f"lgbm_{k}": v for k, v in best["params"].items()})
        mlflow.log_param("lgbm_n_trees", best["iters"])

        # ---- grouped CV (customer never in two folds), fixed tree count
        gkf = GroupKFold(n_splits=5)
        cv = {"logistic": [], "lightgbm": []}
        for a, b in gkf.split(Xtr, ytr, groups=tr["customer_id"]):
            l2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=seed))
            l2.fit(Xtr.iloc[a], ytr[a])
            cv["logistic"].append(average_precision_score(ytr[b], l2.predict_proba(Xtr.iloc[b])[:, 1]))
            g2 = make_lgbm(best["params"], seed, n_estimators=best["iters"])
            g2.fit(Xtr.iloc[a], ytr[a])
            cv["lightgbm"].append(average_precision_score(ytr[b], g2.predict_proba(Xtr.iloc[b])[:, 1]))
        for k, v in cv.items():
            results[k]["cv_pr_auc_mean"], results[k]["cv_pr_auc_sd"] = float(np.mean(v)), float(np.std(v))
            mlflow.log_metric(f"cv_pr_auc_{k}", float(np.mean(v)))

        # ---- calibration: pooled out-of-fold predictions over train+validation (Mar-Aug)
        # Fitting on the August validation month alone over-predicted in quieter months (Aug churn 2.8%,
        # test 1.95%). Pooling all matured months, out of fold and grouped by subscriber, averages regimes.
        pool = pd.concat([tr, va]).reset_index(drop=True)
        Xp, yp = pool[feats].fillna(0), pool["y"].values
        oof = {"logistic": np.zeros(len(pool)), "lightgbm": np.zeros(len(pool))}
        for a, b in GroupKFold(n_splits=5).split(Xp, yp, groups=pool["customer_id"]):
            l3 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=seed)).fit(Xp.iloc[a], yp[a])
            oof["logistic"][b] = l3.predict_proba(Xp.iloc[b])[:, 1]
            g3 = make_lgbm(best["params"], seed, n_estimators=best["iters"]).fit(Xp.iloc[a], yp[a])
            oof["lightgbm"][b] = g3.predict_proba(Xp.iloc[b])[:, 1]
        m_lr = CalibratedChurnModel(lr, PlattCalibrator().fit(oof["logistic"], yp), feats, "logistic")
        m_gb = CalibratedChurnModel(gbm, PlattCalibrator().fit(oof["lightgbm"], yp), feats, "lightgbm")

        # ---- region ablation (region one-hot in a LightGBM)
        regions = sorted(tr["region"].unique())
        Xtr_r = pd.concat([Xtr, region_dummies(tr, regions)], axis=1)
        Xva_r = pd.concat([Xva, region_dummies(va, regions)], axis=1)
        abl = make_lgbm(best["params"], seed, n_estimators=best["iters"]).fit(Xtr_r, ytr)
        results["lightgbm_region_ablation"] = {
            "val_pr_auc": average_precision_score(yva, abl.predict_proba(Xva_r)[:, 1])}
        joblib.dump({"model": abl, "regions": regions, "features": list(Xtr_r.columns)},
                    od / "models" / "lightgbm_region_ablation.joblib")

        # ---- primary model = higher validation PR-AUC
        primary = m_gb if results["lightgbm"]["val_pr_auc"] >= results["logistic"]["val_pr_auc"] else m_lr
        results["primary_model"] = primary.name
        joblib.dump(m_lr, od / "models" / "logistic.joblib")
        joblib.dump(m_gb, od / "models" / "lightgbm.joblib")
        joblib.dump(primary, od / "models" / "primary.joblib")
        for k in ("logistic", "lightgbm"):
            mlflow.log_metric(f"val_pr_auc_{k}", results[k]["val_pr_auc"])
        try:
            mlflow.sklearn.log_model(primary, name="model", registered_model_name="churn_model",
                                     serialization_format="cloudpickle")
        except Exception as e:                       # registry must not block training
            print("MLflow registry step skipped:", e)
        mlflow.log_dict(json.loads(json.dumps(results, default=float)), "training_results.json")

    (od / "training_results.json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps({k: (v if not isinstance(v, dict) else
                          {a: b for a, b in v.items() if a != "params"}) for k, v in results.items()},
                     indent=2, default=float))


if __name__ == "__main__":
    main()
