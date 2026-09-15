"""Skyscrapper building-segmentation MLOps pipeline.

DAG: ``skyscrapper_mlops_pipeline``

Workflow (mirrors ``coder/mlops2/mlops.ipynb``)
----------------------------------------------
1. ``fetch_dataset`` / ``fetch_hlp`` pull the satellite ``dataset`` and the
   human-labelled ``hlp`` tiles from MinIO into the run workspace.
2. ``prepare_data`` inspects the images and extracts hand-crafted features.
3. ``train_segmentation`` trains an SVM on HLP red masks and segments every
   dataset tile, producing masks, contours, metrics and the model.
4. ``finalize_results`` enriches the outputs and writes final CSVs / plots.
5. ``compare_hlp`` compares detected masks with the HLP annotations
   (RMS / IoU / precision / recall / F1).
6. ``log_mlflow`` records params, metrics and artifacts to the MLflow server.
7. ``upload_outputs`` stores everything under ``s3://<bucket>/runs/<dag_run_id>/``.
8. ``report`` prints the run summary (experiment, MLflow run id, metrics).

Every DAG execution is identified by its Airflow ``dag_run_id``; all local and
S3 paths are derived from it, so consecutive runs never overwrite each other.

The ML logic lives in ``coder/mlops2/src/``. Each task shells out to the small
stage runner ``src/run_cli.py`` (running a fresh Python process avoids Airflow's
task-runner sys.path restrictions and keeps the DAG a pure orchestrator).

This DAG is manually triggered (``schedule=None``, ``catchup=False``).
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys

from airflow import DAG
from airflow.sdk import task
from airflow.sdk.exceptions import AirflowFailException

logger = logging.getLogger(__name__)

DAG_ID = "skyscrapper_mlops_pipeline"
_RUN_CLI = "/opt/airflow/mlops2/src/run_cli.py"

doc_md = __doc__ or ""

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
}


def _current_run_id() -> str:
    """Return the Airflow ``dag_run_id`` of the running task instance."""
    run_id = os.environ.get("AIRFLOW_CTX_DAG_RUN_ID")
    if run_id:
        return run_id
    try:
        from airflow.sdk import get_current_context

        ctx = get_current_context()
        run_id = ctx.get("run_id") or os.environ.get("AIRFLOW_CTX_DAG_RUN_ID")
    except Exception:  # pragma: no cover - defensive fallback
        run_id = None
    if not run_id:
        raise RuntimeError("Could not determine the Airflow DAG run id.")
    return run_id


def _run_stage(stage: str) -> str:
    """Run one pipeline stage in a dedicated Python process."""
    run_id = _current_run_id()
    proc = subprocess.run(
        [sys.executable, _RUN_CLI, stage, run_id],
        text=True,
    )
    if proc.returncode != 0:
        raise AirflowFailException(
            f"Stage '{stage}' failed for DAG run '{run_id}' "
            f"(exit code {proc.returncode}). See task logs for details."
        )
    logger.info("Stage %s completed for DAG run %s", stage, run_id)
    return stage


# ---------------------------------------------------------------------------
# 1. Fetch inputs from MinIO
# ---------------------------------------------------------------------------
@task
def fetch_dataset() -> str:
    """Download the ``dataset/`` prefix of MinIO into the run workspace."""
    return _run_stage("fetch_dataset")


@task
def fetch_hlp() -> str:
    """Download the ``hlp/`` prefix of MinIO into the run workspace."""
    return _run_stage("fetch_hlp")


# ---------------------------------------------------------------------------
# 2. Prepare data (inspect images + extract features)
# ---------------------------------------------------------------------------
@task
def prepare_data() -> str:
    """Inspect the dataset (metadata CSV) and extract hand-crafted features."""
    return _run_stage("prepare_data")


# ---------------------------------------------------------------------------
# 3. Train the SVM and segment every dataset image
# ---------------------------------------------------------------------------
@task
def train_segmentation() -> str:
    """Train an SVM on the HLP red masks and segment all dataset tiles."""
    return _run_stage("train_segmentation")


# ---------------------------------------------------------------------------
# 4. Finalize results (final CSVs + visual comparisons)
# ---------------------------------------------------------------------------
@task
def finalize_results() -> str:
    """Enrich stage-3 outputs and write the final contours/metrics/plots."""
    return _run_stage("finalize_results")


# ---------------------------------------------------------------------------
# 5. Compare detected masks with the HLP ground truth
# ---------------------------------------------------------------------------
@task
def compare_hlp() -> str:
    """Pixel-wise comparison of detected masks vs HLP red masks."""
    return _run_stage("compare_hlp")


# ---------------------------------------------------------------------------
# 6. Log everything to MLflow
# ---------------------------------------------------------------------------
@task
def log_mlflow() -> str:
    """Log parameters, metrics, artifacts and tags to the MLflow server."""
    return _run_stage("log_mlflow")


# ---------------------------------------------------------------------------
# 7. Upload the run outputs to MinIO under runs/<dag_run_id>/
# ---------------------------------------------------------------------------
@task
def upload_outputs() -> str:
    """Upload the per-run output tree to ``s3://<bucket>/runs/<dag_run_id>/``."""
    return _run_stage("upload_outputs")


# ---------------------------------------------------------------------------
# 8. Report (human-readable summary visible in the Airflow logs)
# ---------------------------------------------------------------------------
@task
def report() -> str:
    """Print the run summary (experiment, MLflow run id, metrics, artifacts)."""
    return _run_stage("report")


with DAG(
    dag_id=DAG_ID,
    description="Building segmentation MLOps pipeline (SVM) with MinIO inputs and MLflow tracking.",
    doc_md=doc_md,
    schedule=None,
    start_date=None,
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "skyscrapper", "segmentation", "minio", "mlflow"],
    default_args=default_args,
) as dag:

    dataset = fetch_dataset()
    hlp = fetch_hlp()
    prepared = prepare_data()
    trained = train_segmentation()
    finalized = finalize_results()
    compared = compare_hlp()
    mlflow_log = log_mlflow()
    uploaded = upload_outputs()
    final_report = report()

    [dataset, hlp] >> prepared >> trained >> finalized >> compared >> mlflow_log >> uploaded >> final_report
