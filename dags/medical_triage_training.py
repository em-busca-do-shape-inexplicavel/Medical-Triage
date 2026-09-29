from datetime import datetime, timedelta, timezone
from hashlib import sha256

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG


def training_run_dir(run_id: str) -> str:
    """Return a stable, filesystem-safe directory for each DAG run."""
    run_key = sha256(run_id.encode("utf-8")).hexdigest()
    return f"/opt/airflow/training_runs/{run_key}"


with DAG(
    dag_id="medical_triage_training",
    description="Prepare medical abstracts and train the selected classifier.",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "medical-triage",
        "retries": 0,
        "execution_timeout": timedelta(minutes=20),
    },
    user_defined_macros={
        "training_run_dir": training_run_dir,
    },
    tags=["medical-triage", "training"],
) as dag:
    prepare_data = BashOperator(
        task_id="prepare_data",
        env={
            "RUN_DIR": "{{ training_run_dir(run_id) }}",
        },
        append_env=True,
        bash_command="""
            set -euo pipefail

            /opt/medical-venv/bin/python \
                /opt/medical-triage/scripts/prepare_data.py \
                --raw-dir /opt/medical-triage/data/raw \
                --output-dir "$RUN_DIR/processed"
        """,
    )

    train_model = BashOperator(
        task_id="train_model",
        env={
            "RUN_DIR": "{{ training_run_dir(run_id) }}",
        },
        append_env=True,
        bash_command="""
            set -euo pipefail

            /opt/medical-venv/bin/python \
                /opt/medical-triage/scripts/train_model.py \
                --data-dir "$RUN_DIR/processed" \
                --output "$RUN_DIR/model.joblib"
        """,
    )

    prepare_data >> train_model