"""Pickle-safe model wrapper: base estimator + probability calibrator + feature list."""
import numpy as np
from sklearn.linear_model import LogisticRegression


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """Platt scaling: logistic regression on the logit of the raw score.

    Two parameters and strictly monotonic, so it never creates ties and never changes the ranking.
    Fitted on pooled out-of-fold predictions over all train and validation cutoffs, so the calibration
    reflects the average base rate across regimes instead of one busy month.
    """

    def __init__(self):
        self.lr = LogisticRegression(C=1e6, max_iter=1000)

    def fit(self, raw, y):
        self.lr.fit(_logit(raw).reshape(-1, 1), y)
        return self

    def predict(self, raw):
        return self.lr.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]


class CalibratedChurnModel:
    def __init__(self, base, calibrator, features, name):
        self.base, self.iso, self.features, self.name = base, calibrator, list(features), name   # `iso` kept for old pickles

    def _x(self, df):
        return df[self.features].fillna(0)

    def predict_raw(self, df):
        return self.base.predict_proba(self._x(df))[:, 1]

    def predict_proba(self, df):
        """Calibrated churn probability for next month (1-D array)."""
        return self.iso.predict(self.predict_raw(df))

    def predict(self, df):                       # sklearn/mlflow pyfunc compatibility
        return self.predict_proba(df)
