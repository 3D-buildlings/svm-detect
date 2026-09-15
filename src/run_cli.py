#!/usr/bin/env python3
"""Stage runner used by the Airflow DAG.

Each Airflow task shells out to this script (``python run_cli.py <stage>
<dag_run_id>``). Running as a separate process makes the pipeline independent of
the import/sys.path restrictions of the Airflow task runner: the script inserts
its own project root on ``sys.path`` (``sys.path[0]`` is always the script's
directory), exactly like ``seed_minio.py``.

Usage:
    python run_cli.py <stage> <run_id>

Stages:
    fetch_dataset | fetch_hlp | prepare_data | train_segmentation |
    finalize_results | compare_hlp | log_mlflow | upload_outputs | report
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# sys.path[0] == directory of this script == coder/mlops2 (project root)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config, data, evaluate, pipeline, storage, track, train  # noqa: E402

logger = logging.getLogger("run_cli")


def _run(run_id: str) -> config.RunPaths:
    run = config.RunPaths.build(run_id)
    run.create_dirs()
    return run


def fetch_dataset(run: config.RunPaths) -> None:
    client = storage.default_client()
    try:
        files = client.download_prefix(config.DATASET_PREFIX, run.inputs_dataset)
    except storage.StorageError as exc:
        print(f"ERROR: {exc}")
        print("Upload the initial data first (seed the 'dataset/' prefix in MinIO).")
        sys.exit(1)
    print(f"Fetched {len(files)} dataset objects -> {run.inputs_dataset}")


def fetch_hlp(run: config.RunPaths) -> None:
    client = storage.default_client()
    try:
        files = client.download_prefix(config.HLP_PREFIX, run.inputs_hlp)
    except storage.StorageError as exc:
        print(f"ERROR: {exc}")
        print("Upload the initial data first (seed the 'hlp/' prefix in MinIO).")
        sys.exit(1)
    print(f"Fetched {len(files)} hlp objects -> {run.inputs_hlp}")


def prepare_data(run: config.RunPaths) -> None:
    metadata_csv = data.stage_load_and_inspect(run)
    features_csv = data.stage_preprocess(run)
    print(f"metadata: {metadata_csv}")
    print(f"features: {features_csv}")


def train_segmentation(run: config.RunPaths) -> None:
    paths = train.stage_train_and_segment(run)
    print(f"train outputs: {paths}")


def finalize_results(run: config.RunPaths) -> None:
    paths = evaluate.stage_finalize(run)
    print(f"finalize outputs: {paths}")


def compare_hlp(run: config.RunPaths) -> None:
    paths = evaluate.stage_hlp_comparison(run)
    print(f"hlp comparison outputs: {paths}")


def log_mlflow(run: config.RunPaths) -> None:
    summary = track.log_run_to_mlflow(run)
    print(f"mlflow_run_id: {summary['mlflow_run_id']}")
    print(f"experiment: {summary['experiment_name']}")
    print(f"tracking_uri: {summary['mlflow_tracking_uri']}")


def upload_outputs(run: config.RunPaths) -> None:
    prefix = pipeline.upload_outputs_to_minio(run)
    print(f"uploaded outputs -> s3://{config.minio_bucket()}/{prefix}")


def report(run: config.RunPaths) -> None:
    summary = track.load_summary(run)
    metrics = summary.get("metrics", {})
    metrics_lines = "\n".join(f"    {k}: {v}" for k, v in metrics.items()) or "    (none)"
    block = "\n".join([
        "=" * 60,
        "MLOps Pipeline Completed Successfully",
        "=" * 60,
        f"Airflow DAG Run ID: {run.run_id}",
        "",
        f"MLflow Experiment: {summary.get('experiment_name', config.mlflow_experiment_name())}",
        f"MLflow Run ID: {summary.get('mlflow_run_id', 'N/A')}",
        f"MLflow Tracking URI: {summary.get('mlflow_tracking_uri', config.mlflow_tracking_uri())}",
        "",
        "Metrics:",
        metrics_lines,
        "",
        f"Artifacts: {summary.get('s3_runs_uri', config.s3_runs_uri(run.run_id))}",
        "=" * 60,
    ])
    print(block)


STAGES = {
    "fetch_dataset": fetch_dataset,
    "fetch_hlp": fetch_hlp,
    "prepare_data": prepare_data,
    "train_segmentation": train_segmentation,
    "finalize_results": finalize_results,
    "compare_hlp": compare_hlp,
    "log_mlflow": log_mlflow,
    "upload_outputs": upload_outputs,
    "report": report,
}


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: run_cli.py <stage> <dag_run_id>")
        return 2
    stage, run_id = sys.argv[1], sys.argv[2]
    if stage not in STAGES:
        print(f"unknown stage: {stage}. Valid: {sorted(STAGES)}")
        return 2
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    run = _run(run_id)
    print(f"--- stage: {stage} | run_id: {run_id} ---")
    STAGES[stage](run)
    print(f"--- stage {stage} complete ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
