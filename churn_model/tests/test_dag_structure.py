"""
Structure test for src/03_orchestration/dags/churn_scoring_monthly.py against a minimal stub of Airflow.

Airflow does not install on Windows, so this cannot prove the DAG runs on a scheduler. It does
prove that the file imports, that the task graph is the intended one, that every task calls a
real stage function, and that the branch and XCom helpers behave as written.
"""
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class _Op:
    dag = None

    def __init__(self, task_id, python_callable=None, op_args=None, **kw):
        self.task_id, self.python_callable, self.op_args = task_id, python_callable, op_args or []
        self.downstream = []
        _Op.dag.tasks[task_id] = self

    def __rshift__(self, other):
        for o in (other if isinstance(other, list) else [other]):
            self.downstream.append(o.task_id)
        return other


class _DAG:
    def __init__(self, dag_id, **kw):
        self.dag_id, self.kw, self.tasks = dag_id, kw, {}

    def __enter__(self):
        _Op.dag = self
        return self

    def __exit__(self, *a):
        return False


def _load_dag():
    airflow = types.ModuleType("airflow")
    airflow.DAG = _DAG
    ops = types.ModuleType("airflow.operators")
    py = types.ModuleType("airflow.operators.python")
    py.PythonOperator = py.BranchPythonOperator = _Op
    empty = types.ModuleType("airflow.operators.empty")
    empty.EmptyOperator = _Op
    sys.modules.update({"airflow": airflow, "airflow.operators": ops,
                        "airflow.operators.python": py, "airflow.operators.empty": empty})
    spec = importlib.util.spec_from_file_location("dag_under_test", ROOT / "src" / "03_orchestration" / "dags" / "churn_scoring_monthly.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_task_graph_and_callables():
    mod = _load_dag()
    dag = mod.dag
    t = dag.tasks
    assert dag.dag_id == "churn_scoring_monthly"
    assert dag.kw["schedule"] == "0 3 1 * *" and dag.kw["catchup"] is False and dag.kw["max_active_runs"] == 1
    assert set(t) == {"build_features", "validate_inputs", "score_explain_write", "drift_check_log",
                      "retrain_gate", "retrain_alert", "scores_published"}
    assert t["build_features"].downstream == ["validate_inputs"]
    assert t["validate_inputs"].downstream == ["score_explain_write"]
    assert t["score_explain_write"].downstream == ["drift_check_log"]
    assert t["drift_check_log"].downstream == ["retrain_gate"]
    assert t["retrain_gate"].downstream == ["retrain_alert", "scores_published"]
    from churn_model import score_monthly as sm
    assert t["build_features"].python_callable is sm.stage_build_features
    for tid in ("build_features", "validate_inputs", "score_explain_write", "drift_check_log"):
        assert "data_interval_end" in t[tid].op_args[0], "cutoff must come from the run's interval"


def test_branch_and_xcom_helpers():
    mod = _load_dag()

    class TI:
        def __init__(self, store):
            self.store = store

        def xcom_push(self, key, value):
            self.store[key] = value

        def xcom_pull(self, key):
            return self.store.get(key)

    assert mod._gate(ti=TI({"retrain": True})) == "retrain_alert"
    assert mod._gate(ti=TI({"retrain": False})) == "scores_published"
