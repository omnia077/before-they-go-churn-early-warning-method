"""Two-proportion sample sizes for the pilot, from the measured top-decile churn on the test months.
Run: python -m churn_model.pilot_sizes   (writes src/data/ml/pilot_sample_sizes.json)"""
import json
import math

from scipy.stats import norm

from churn_model.common import load_config, out_dir


def sample_per_arm(p1, relative_reduction, alpha=0.05, power=0.80):
    p2 = p1 * (1 - relative_reduction)
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    return math.ceil((za + zb) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p1 - p2) ** 2)


def main():
    od = out_dir(load_config())
    p1 = json.loads((od / "evaluation_results.json").read_text())["lightgbm"]["precision_at_thr"]
    out = {"p1": p1, "n": {str(r): sample_per_arm(p1, r) for r in (0.15, 0.20, 0.25, 0.30)}}
    (od / "pilot_sample_sizes.json").write_text(json.dumps(out))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
