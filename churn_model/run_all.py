"""
Run the whole ML pipeline in order. Stops at the first failing step.

    python -m churn_model.run_all                 # everything, from the panel onwards
    python -m churn_model.run_all --from train    # resume from a step
    python -m churn_model.run_all --list          # show the steps

Prerequisite: the v2 data in src/data and the Phase 2-7 outputs (run_pipeline.ps1 -SkipGeneration).
CHURN_HASH_SALT must be set; if it is not, a gitignored `.churn_salt` file in the repo root is used.
"""
import argparse
import os
import subprocess
import sys
import time

from churn_model.common import ROOT

STEPS = [
    ("panel", ["-m", "churn_model.build_panel"]),
    ("leakage", ["-m", "churn_model.leakage_checks"]),
    ("train", ["-m", "churn_model.train"]),
    ("evaluate", ["-m", "churn_model.evaluate"]),
    ("explain", ["-m", "churn_model.explain"]),
    ("decision", ["-m", "churn_model.decision"]),
    ("network", ["-m", "churn_model.network_handoff"]),
    ("fairness", ["-m", "churn_model.fairness"]),
    ("pilot", ["-m", "churn_model.pilot_sizes"]),
    ("score-nov", ["-m", "churn_model.score_monthly", "--as-of", "2024-11-30"]),   # builds drift history
    ("score-dec", ["-m", "churn_model.score_monthly", "--as-of", "2024-12-31"]),
    ("dashboard", ["-m", "churn_model.build_dashboard"]),
    ("root-cause-dashboard", ["dashboard/refresh_churn_dashboard.py"]),
    ("deck", ["-m", "churn_model.build_deck"]),
]


def environment():
    env = dict(os.environ, PYTHONUTF8="1", PYTHONPATH=str(ROOT), MLFLOW_DISABLE_AGENT_HINT="1")
    if not env.get("CHURN_HASH_SALT"):
        salt_file = ROOT / ".churn_salt"
        if not salt_file.exists():
            sys.exit("Set CHURN_HASH_SALT (or create a gitignored .churn_salt file) before running.")
        env["CHURN_HASH_SALT"] = salt_file.read_text().strip()
    if os.name == "nt" and os.path.exists("C:/hadoop/bin"):
        env["HADOOP_HOME"] = "C:\\hadoop"
        env["PATH"] = "C:\\hadoop\\bin;" + env["PATH"]
    env.setdefault("PYSPARK_PYTHON", sys.executable)
    env.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=STEPS[0][0], choices=[n for n, _ in STEPS])
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        print("\n".join(n for n, _ in STEPS))
        return
    env = environment()
    names = [n for n, _ in STEPS]
    for name, cmd in STEPS[names.index(a.start):]:
        t = time.time()
        print(f"\n=== {name} ===", flush=True)
        r = subprocess.run([sys.executable, "-W", "ignore", *cmd], cwd=ROOT, env=env)
        if r.returncode != 0:
            sys.exit(f"step '{name}' failed (exit {r.returncode}); fix it and resume with --from {name}")
        print(f"--- {name} done in {time.time() - t:.0f}s", flush=True)
    print("\nAll ML steps completed.")


if __name__ == "__main__":
    main()
