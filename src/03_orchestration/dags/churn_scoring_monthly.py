"""
Airflow DAG: churn_scoring_monthly

Runs on the 1st of each month for the previous month-end cutoff ({{ ds }} minus 1 day).
Each task calls one stage of churn_model/score_monthly.py; intermediate data is handed over
as parquet files under src/data/ml (never through XCom).

    build_features -> validate_inputs -> score_explain_write -> drift_check_log -> retrain_gate

Deploy: copy this file to $AIRFLOW_HOME/dags/ and set the repo root on PYTHONPATH.
Secrets: CHURN_HASH_SALT must be set in the worker environment (never in the repo).
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.empty import EmptyOperator

from churn_model import score_monthly as sm

AS_OF = "{{ (data_interval_end - macros.timedelta(days=1)).strftime('%Y-%m-%d') }}"

default_args = {"owner": "data_science", "retries": 2, "retry_delay": timedelta(minutes=10),
                "depends_on_past": False}


def _validate(as_of, ti=None):
    ti.xcom_push(key="validation", value=sm.stage_validate_inputs(as_of))      # small dict only


def _score(as_of, ti=None):
    ti.xcom_push(key="scored", value=sm.stage_score_explain_write(as_of))


def _drift(as_of, ti=None):
    prior = {"validation": ti.xcom_pull(key="validation"), **ti.xcom_pull(key="scored")}
    summary = sm.stage_drift_and_log(as_of, prior)
    ti.xcom_push(key="retrain", value=summary["drift"]["retrain_trigger"])


def _gate(ti=None):
    return "retrain_alert" if ti.xcom_pull(key="retrain") else "scores_published"


with DAG("churn_scoring_monthly", default_args=default_args, start_date=datetime(2024, 4, 1),
         schedule="0 3 1 * *", catchup=False, max_active_runs=1,
         tags=["churn", "ml", "monthly"]) as dag:
    build = PythonOperator(task_id="build_features", python_callable=sm.stage_build_features,
                           op_args=[AS_OF])
    validate = PythonOperator(task_id="validate_inputs", python_callable=_validate, op_args=[AS_OF])
    score = PythonOperator(task_id="score_explain_write", python_callable=_score, op_args=[AS_OF])
    drift = PythonOperator(task_id="drift_check_log", python_callable=_drift, op_args=[AS_OF])
    gate = BranchPythonOperator(task_id="retrain_gate", python_callable=_gate)
    alert = EmptyOperator(task_id="retrain_alert")          # wire to e-mail/Slack in production
    ok = EmptyOperator(task_id="scores_published")

    build >> validate >> score >> drift >> gate >> [alert, ok]
