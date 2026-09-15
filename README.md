# Skyscrapper MLOps pipeline (GitHub execution)

Building-segmentation MLOps pipeline (pixel-level **SVM**) that runs on
**MinIO** inputs and logs to **MLflow**.

This project is **executed by an Apache Airflow GitHub CI runner**: Airflow
clones this repository into a run-specific workspace, builds an isolated
`.venv`, optionally installs `requirements.txt`, and runs
`skyscrapper_mlops_pipeline.py`. The repository is the source of truth for the
application code; Airflow is only the runner.

## Layout

```
.
├── skyscrapper_mlops_pipeline.py   # entrypoint executed by Airflow (plain Python, no Airflow imports)
├── requirements.txt                # project dependencies (installed into the per-run .venv)
├── README.md
└── src/
    ├── config.py                   # environment-driven configuration
    ├── storage.py                  # MinIO (S3) read/write helpers
    ├── data.py                     # image inspection + feature extraction
    ├── train.py                    # SVM training + segmentation
    ├── evaluate.py                 # finalization + HLP comparison
    ├── track.py                    # MLflow logging
    ├── pipeline.py                 # single-process end-to-end runner
    └── run_cli.py                  # optional per-stage runner
```

## How Airflow runs it

```
GitHub (this repo)  ->  Airflow GitHub CI runner  ->  isolated workspace + .venv  ->  run entrypoint
```

Airflow's `skyscrapper_mlops_pipeline_github_ci` DAG performs:

```
prepare_workspace -> clone_repository -> create_virtualenv
    -> install_requirements -> run_project -> cleanup
```

The entrypoint is invoked with the project venv:

```bash
.venv/bin/python skyscrapper_mlops_pipeline.py
```

and receives `DAG_RUN_ID` (and `GITHUB_CI_*` metadata) through the environment.

## Configuration (environment variables)

| Variable | Purpose | Example |
|---|---|---|
| `MINIO_ENDPOINT` | MinIO/S3 endpoint | `http://minio:9000` |
| `MINIO_ACCESS_KEY` | MinIO access key | `admin` |
| `MINIO_SECRET_KEY` | MinIO secret key | `…` |
| `MINIO_BUCKET` | bucket holding inputs/outputs | `mlops` |
| `MLFLOW_TRACKING_URI` | MLflow tracking server | `http://mlflow:5000` |
| `MLFLOW_EXPERIMENT_NAME` | experiment name | `skyscrapper_building_segmentation` |
| `MLOPS_RUNS_DIR` | local scratch dir for this run | injected by the runner |

No credentials are stored in this repository.

## Inputs

The pipeline reads its inputs from MinIO:

* `s3://<bucket>/dataset/` - satellite tiles
* `s3://<bucket>/hlp/` - human-labelled red-mask tiles

Outputs are written to `s3://<bucket>/runs/<run_id>/` and to MLflow.

## Running locally (optional)

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
export MINIO_ENDPOINT=http://localhost:9500
export MINIO_ACCESS_KEY=admin
export MINIO_SECRET_KEY=<your-minio-secret>
export MLFLOW_TRACKING_URI=http://localhost:5050
.venv/bin/python skyscrapper_mlops_pipeline.py --run-id local__demo
```

Useful flags:

```bash
--run-id <id>     # explicit run identifier
--stage <name>    # run a single stage (fetch_dataset, prepare_data, train_segmentation, ...)
--no-fetch        # skip downloading inputs from MinIO
--no-upload       # skip uploading outputs back to MinIO
```
