"""End-to-end pipeline runner.

Orchestrates every stage in the same order as the original notebook:

    fetch inputs (MinIO) -> load/inspect -> preprocess -> train/segment
        -> finalize results -> HLP comparison -> MLflow logging -> upload outputs

The Airflow DAG splits these steps into individual tasks; this module keeps a
single-process ``run_pipeline()`` so the same code can be executed from the
notebook (``mlops.ipynb``) and from tests.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

from src import config, data, evaluate, storage, track, train

logger = logging.getLogger("skyscrapper.pipeline")


def run_pipeline(
    run_id: str,
    base_dir: str | None = None,
    *,
    fetch_from_minio: bool = True,
    upload_outputs: bool = True,
) -> Dict[str, Any]:
    """Run the whole skyscrapper MLOps pipeline for ``run_id``.

    When ``fetch_from_minio`` is True the ``dataset`` and ``hlp`` inputs are
    downloaded from MinIO into the per-run workspace. When ``upload_outputs``
    is True the final outputs are uploaded to ``runs/<run_id>/``.
    """
    run = config.RunPaths.build(run_id, base_dir)
    run.create_dirs()

    if fetch_from_minio:
        client = storage.default_client()
        client.download_prefix(config.DATASET_PREFIX, run.inputs_dataset)
        client.download_prefix(config.HLP_PREFIX, run.inputs_hlp)

    data.stage_load_and_inspect(run)
    data.stage_preprocess(run)
    train.stage_train_and_segment(run)
    evaluate.stage_finalize(run)
    evaluate.stage_hlp_comparison(run)
    summary = track.log_run_to_mlflow(run)

    if upload_outputs:
        s3_prefix = upload_outputs_to_minio(run)
        summary["s3_uploaded_prefix"] = s3_prefix

    return summary


def upload_outputs_to_minio(run: config.RunPaths, extra: Dict[str, Any] | None = None) -> str:
    """Upload the whole per-run ``outputs`` tree under ``runs/<run_id>/``."""
    client = storage.default_client()
    prefix = config.s3_run_prefix(run.run_id)

    client.upload_dir(run.outputs_dir, prefix)
    objects = client.list_objects(prefix)
    manifest = {
        "dag_run_id": run.run_id,
        "local_outputs_dir": run.outputs_dir,
        "s3_runs_uri": config.s3_runs_uri(run.run_id),
        "object_count": len(objects),
        "objects": sorted(objects),
    }
    if extra:
        manifest["extra"] = extra
    storage.write_run_manifest(client, run.run_id, manifest)
    logger.info("All outputs uploaded to %s (%d objects)",
                config.s3_runs_uri(run.run_id), len(objects))
    return prefix


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if len(sys.argv) < 2:
        sys.exit("usage: python -m src.pipeline <run_id>")
    summary = run_pipeline(sys.argv[1])
    print(summary)
