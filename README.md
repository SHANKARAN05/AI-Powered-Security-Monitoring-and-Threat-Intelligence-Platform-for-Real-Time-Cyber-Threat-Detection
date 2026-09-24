# AI-Powered Security Monitoring & Threat Intelligence System (AICS)

A machine-learning-based network intrusion detection and classification system that moves beyond signature-based rules to catch novel attack patterns in real time. The system ingests network flow records, runs them through a **two-phase AI engine** (unsupervised anomaly **Detection** followed by supervised attack-type **Prediction**), enriches results with local threat-intelligence lookups, computes a **0–100 risk score**, and surfaces everything through a live Flask + Chart.js **security dashboard**.

It is built and evaluated on two real, independent network-traffic datasets — **CICIDS2017** (primary) and **UNSW-NB15** (secondary) — run as two structurally identical, parallel pipelines.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Run with Docker (recommended)](#run-with-docker-recommended)
  - [Local development](#local-development)
- [Training the Models](#training-the-models)
- [Running the Real-Time Demo (Replay Simulator)](#running-the-real-time-demo-replay-simulator)
- [REST API Reference](#rest-api-reference)
- [Dashboard](#dashboard)
- [Database Schema](#database-schema)
- [Model Performance Summary](#model-performance-summary)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Further Documentation](#further-documentation)

---

## Overview

Traditional signature-based intrusion detection systems can only catch attacks they already have a rule for. AICS instead:

1. **Detects** anomalous network flows using models trained *only on benign traffic* (so it can flag attack patterns it has never seen before).
2. **Classifies** every flagged flow into a specific attack category using supervised multiclass models.
3. **Scores risk** (0–100) by combining attack severity, model confidence, anomaly strength, and threat-intel (blacklist) matches.
4. **Persists and visualizes** every alert in PostgreSQL and a live dashboard, so an analyst can triage, filter, and update alert status.

A **replay simulator** feeds held-out test rows through the pipeline at a configurable interval to simulate a live traffic feed for demo purposes — architecturally identical to how a live capture tool (e.g. CICFlowMeter) would feed the same schema in production.

## Key Features

- **Two-phase ML engine**: unsupervised Detection (Isolation Forest, One-Class SVM, Keras Autoencoder, ensemble) gates a supervised Prediction phase (Random Forest, XGBoost, Keras NN classifier).
- **Dual-dataset support**: CICIDS2017 (13-class, demo/primary) and UNSW-NB15 (10-class, generalization/secondary), via one shared preprocessing/model base-class skeleton.
- **Risk scoring engine**: severity-tier lookup table × model confidence, with bonuses for anomaly-gate agreement and threat-intel blacklist hits.
- **Threat intelligence correlation**: local IP blacklist and attack-category → CVE/CWE lookup tables.
- **PostgreSQL persistence**: alerts, an append-only prediction log, and a model registry table, using JSONB for nested detector scores / raw features / threat-intel matches.
- **REST API**: predict, alert CRUD/filtering, replay control, dashboard aggregates, health check.
- **Live dashboard**: KPI tiles, severity/attack-type charts (Chart.js), a live recent-alerts feed, a filterable/paginated alerts page, and a model-metrics comparison page — with light/dark theme support.
- **Fully containerized**: `docker-compose` spins up the Flask app + PostgreSQL together.

## Architecture

```
                         ┌─────────────────────────────┐
  Network flow record ─▶│   Shared Preprocessor        │  (per-dataset fitted
  (or replay simulator)  │   clean → encode → scale →   │   encoders/scaler/
                         │   feature-select             │   feature list)
                         └──────────────┬───────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │  DETECTION PHASE (runs on    │
                         │  every flow, benign-only     │  Isolation Forest
                         │  trained, semi-supervised)   │  One-Class SVM
                         │  → anomaly score + threshold │  Autoencoder (+ ensemble)
                         └──────────────┬───────────────┘
                                        │ anomaly_flagged?
                             ┌──────────┴───────────┐
                             ▼ yes                   ▼ no
                 ┌───────────────────────┐   label = BENIGN/Normal
                 │  PREDICTION PHASE     │   confidence = 1 - anomaly ratio
                 │  (supervised          │
                 │  multiclass attack    │   Random Forest / XGBoost /
                 │  classifier)          │   Keras NN classifier
                 └───────────┬───────────┘
                             ▼
                 ┌───────────────────────┐
                 │  Risk Scoring         │   severity table × confidence
                 │  + Threat Intel match │   + anomaly/blacklist bonuses
                 └───────────┬───────────┘
                             ▼
              ┌───────────────────────────────┐
              │ PostgreSQL (alerts,            │
              │ predictions_log, model_registry)│
              └───────────────┬───────────────┘
                              ▼
              ┌───────────────────────────────┐
              │ Flask REST API + Jinja2/Chart.js│
              │ dashboard (KPIs, charts, alerts) │
              └───────────────────────────────┘
```

## Datasets

| Dataset | Role | Rows | Features | Classes | Split |
|---|---|---|---|---|---|
| **CICIDS2017** (`Data/archive/Week_filtered.csv`) | Primary / demo | ~543.7k (post-dedup) | 78 CICFlowMeter flow features | 13 (BENIGN + 12 attack types) | 70/15/15 stratified (created in-pipeline) |
| **UNSW-NB15** (`Data/UNSW_NB15_*-set.csv`) | Secondary / generalization | 175,341 train + 82,332 test | 45 raw (numeric + `proto`/`service`/`state`) | 10 (Normal + 9 attack categories) | Author-provided train/test, val carved from train |

Both datasets are curated, pre-labeled public network-intrusion benchmarks; they use **incompatible feature schemas** and are therefore trained and served as two independent pipelines that share only code structure, not data or models.

Known dataset caveats (see [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) for full detail): CICIDS2017 had 23,792 exact duplicate rows and a mangled dash in `Web Attack` labels (both cleaned in preprocessing); UNSW-NB15 has several rare/near-unlearnable classes (`Worms` = 130 training rows) and known class-overlap issues in `Analysis`/`Backdoor`/`DoS` documented in the literature.

## Tech Stack

| Layer | Technology |
|---|---|
| ML / data | scikit-learn, TensorFlow/Keras, XGBoost, pandas, numpy, joblib, pyarrow |
| API / backend | Flask, Flask-SQLAlchemy, gunicorn |
| Database | PostgreSQL 16 (via SQLAlchemy + psycopg2, JSONB columns) |
| Frontend | Server-rendered Jinja2 templates + vanilla JS + Chart.js 4 (no SPA/build step) |
| Containerization | Docker, docker-compose (`python:3.11-slim` app image + `postgres:16-alpine`) |
| Notebooks (planned) | Jupyter, matplotlib, seaborn |

> **Why Docker is required, not optional**: the host environment has only Python 3.14 available, for which no TensorFlow wheel exists. All ML training and serving is designed to run inside the Python-3.11 Docker image (see `docker/Dockerfile`).

## Project Structure

```
AICS Project/
├── Data/                          # Raw datasets (untouched) + processed splits + threat-intel tables
│   ├── UNSW_NB15_training-set.csv / -testing-set.csv
│   ├── archive/Week_filtered.csv  # curated CICIDS2017 file (+ raw daily CSVs)
│   ├── processed/{cicids,unsw}/   # train/val/test parquet splits (generated)
│   └── threat_intel/              # blacklist_ips.csv, cve_lookup.json
├── models/{cicids,unsw}/          # fitted encoders/scaler + trained model artifacts (generated)
│   ├── scaler.pkl, onehot_encoder.pkl, multiclass_encoder.pkl, feature_metadata.json
│   ├── detection/                 # isolation_forest.pkl, one_class_svm.pkl, autoencoder.keras, metrics.json
│   └── prediction/                # random_forest.pkl, xgboost.pkl, nn_classifier.keras, metrics.json, class_names.json, feature_importance.json
├── src/
│   ├── common/                    # config.py, base_preprocessor.py, base_detector.py, base_predictor.py
│   ├── preprocessing/             # cicids_preprocessor.py, unsw_preprocessor.py
│   ├── models/
│   │   ├── detection/             # isolation_forest.py, one_class_svm.py, autoencoder.py
│   │   └── prediction/            # random_forest.py, xgboost_model.py, nn_classifier.py, risk_scoring.py
│   ├── pipeline/                  # prepare_data.py, train_detection.py, train_prediction.py, inference_pipeline.py, replay_simulator.py
│   ├── threat_intel/              # blacklist_lookup.py, cve_lookup.py
│   ├── db/                        # models.py (SQLAlchemy), postgres_client.py
│   ├── api/                       # app.py, pipelines.py, routes/{health,predict,alerts,replay,dashboard}.py
│   └── dashboard/                 # views.py, templates/*.html, static/{app.js,style.css}
├── docker/                        # Dockerfile, docker-compose.yml
├── notebooks/                     # (reserved for EDA notebooks — currently empty)
├── tests/                         # (reserved — currently empty package)
├── PROJECT_PLAN.md                # original design/decision document
├── PROJECT_DOCUMENTATION.md       # full technical analysis (this repo)
└── requirements.txt
```

## Getting Started

### Prerequisites

- Docker & Docker Compose (recommended path — required for TensorFlow, since the host Python is 3.14)
- *or* Python 3.11 locally if you want to run outside Docker
- The two dataset files already present under `Data/` (already included in this repo)

### Run with Docker (recommended)

```bash
cd docker
docker-compose up --build
```

This starts:
- `postgres` — PostgreSQL 16 on port `5432` (user/pass/db: `aics` / `aics` / `aics`)
- `app` — the Flask app (via gunicorn) on port `5000`, with `src/`, `Data/`, `models/`, and `notebooks/` bind-mounted for live iteration

Once running, visit **http://localhost:5000** — it redirects to the dashboard.

> Note: `docker-compose.yml` mounts `../models` from the host, so model artifacts must already exist (see [Training the Models](#training-the-models)) or be trained inside a running container before the API can serve predictions — `src/api/pipelines.py` lazily loads a dataset's `InferencePipeline` (and therefore its model files) on first use.

### Local development

```bash
python -m venv .venv && source .venv/bin/activate   # or the Windows equivalent
pip install -r requirements.txt
export DATABASE_URL=postgresql+psycopg2://aics:aics@localhost:5432/aics   # point at a running Postgres
python -m src.api.app
```

(TensorFlow-dependent code — the autoencoder and NN classifier — requires Python ≤3.12 locally; use Docker if your host Python is 3.13+.)

## Training the Models

Run these once (inside the container, or locally with the dependencies installed) before the API/dashboard has anything real to serve. Each step reads `Data/`, writes to `Data/processed/` and `models/`:

```bash
# 1. Clean, encode, scale, select features, split -> Data/processed/{cicids,unsw}/*.parquet
python -m src.pipeline.prepare_data

# 2. Train the Detection Phase (Isolation Forest, One-Class SVM, Autoencoder) per dataset
python -m src.pipeline.train_detection cicids unsw

# 3. Train the Prediction Phase (Random Forest, XGBoost, NN classifier) per dataset
python -m src.pipeline.train_prediction cicids unsw
```

Each step prints per-model metrics to stdout and writes a `metrics.json` consumed by the `/api/models` endpoint and the dashboard's Models page.

## Running the Real-Time Demo (Replay Simulator)

Either from the CLI:

```bash
python -m src.pipeline.replay_simulator cicids --n-rows 300 --delay 0.5
```

or via the dashboard's **Start replay** control (top of the Dashboard page), which calls `POST /api/replay/start`. This streams held-out test rows through the live inference pipeline at the chosen interval, generating real alerts in Postgres that immediately appear on the dashboard.

## REST API Reference

All endpoints are prefixed `/api` (registered in `src/api/app.py`).

| Method & Path | Purpose |
|---|---|
| `GET /api/health` | Reports which datasets have trained artifacts on disk and which pipelines are currently loaded in memory. |
| `POST /api/predict` | Run one flow (`{"dataset": "cicids"\|"unsw", "features": {...}, "src_ip"?, "dst_ip"?, "flow_ref"?}`) through the full detect→predict→risk-score pipeline; persists an alert if flagged. |
| `GET /api/alerts` | Paginated, filterable (`page`, `per_page`, `severity`, `dataset`, `status`) alert listing. |
| `GET /api/alerts/<id>` | Full detail for one alert, including raw features. |
| `PATCH /api/alerts/<id>` | Update analyst status (`new` / `investigating` / `resolved`). |
| `POST /api/replay/start` | Start a background replay thread (`dataset`, `n_rows`, `delay`). |
| `POST /api/replay/stop` | Signal the running replay thread to stop. |
| `GET /api/replay/status` | Whether a replay is currently running, and for which dataset. |
| `GET /api/dashboard-data` | Aggregates for the dashboard: total alerts, severity/attack-type breakdowns, top source IPs, 20 most recent alerts. |
| `GET /api/models` | Detection- and prediction-phase metrics for both datasets (from the saved `metrics.json` files). |

## Dashboard

Three server-rendered pages (Flask + Jinja2 + Chart.js, no build step):

- **Dashboard** (`/dashboard`) — replay controls, KPI tiles (total/critical/high/avg risk score), a severity bar chart, an attack-type bar chart, and a live recent-alerts table. Auto-refreshes every 4 seconds.
- **Alerts** (`/alerts`) — full paginated, filterable alert table with inline status updates.
- **Models** (`/models`) — side-by-side detection-phase (precision/recall/F1/ROC-AUC/PR-AUC/FPR) and prediction-phase (macro-F1/weighted-F1) comparison tables for both datasets.

## Database Schema

PostgreSQL via SQLAlchemy (`src/db/models.py`), JSONB for nested fields:

- **`alerts`** — one row per flagged flow: timestamp, dataset, flow/src/dst identifiers, predicted attack category, severity, risk score, anomaly score, `detector_scores` (JSONB), `threat_intel_match` (JSONB), analyst `status`, `raw_features` (JSONB).
- **`predictions_log`** — append-only log of *every* scored flow (not just flagged ones), for later drift/audit analysis.
- **`model_registry`** — model name/dataset/version/trained-at/metrics/artifact path (reserved for future model-versioning use; not yet written to by the current pipelines).

## Model Performance Summary

Headline numbers from the currently trained artifacts (full per-class breakdown in [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)):

**Detection Phase** (test set, F1-optimal threshold):

| Dataset | Best detector | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| CICIDS2017 | Autoencoder | 0.876 | 0.923 | **0.899** | 0.945 |
| UNSW-NB15 | Autoencoder | 0.902 | 0.799 | **0.847** | 0.887 |

**Prediction Phase** (test set, attack-type classifier):

| Dataset | Best model | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| CICIDS2017 | XGBoost | 0.910 | 0.998 |
| UNSW-NB15 | XGBoost | 0.522 | 0.752 |

The large macro-F1 gap on UNSW-NB15 (vs. its high weighted-F1) reflects several genuinely hard, overlapping classes (`Analysis`, `Backdoor`, `DoS`) — a known characteristic of this dataset, not a pipeline defect.

## Known Limitations

- **No automated tests** — `tests/` is currently an empty package.
- **No EDA notebooks** — `notebooks/` is reserved but empty; the dataset analysis lives in `PROJECT_PLAN.md`/`PROJECT_DOCUMENTATION.md` instead.
- **Detection-phase FPR is high at the F1-optimal operating point** (up to ~57% on CICIDS ensemble), a real precision/recall trade-off — see the Models page and the documentation's evaluation section.
- **No live email/SMS alerting** — `flask-mail` is listed in `requirements.txt` but not wired up yet.
- **No live packet capture** — the "real-time" pipeline is a replay simulator over held-out test data, documented as architecturally equivalent to a live CICFlowMeter feed.
- **No cloud deployment** — intentionally out of scope for this phase; the system is "containerized and deployment-ready," not deployed.
- Synthetic IPs are generated by the replay simulator (neither dataset retains real IPs), so IP-based threat-intel correlation is demo-grade by construction.

## Roadmap

- Wire up `Flask-Mail` alerting for Critical-severity alerts.
- Add automated unit tests for preprocessors, detectors, and the risk-scoring formula.
- Populate `model_registry` from the training pipelines for real model-version tracking.
- Add a `Destination Port` ablation run (flagged as a high-leakage feature in `PROJECT_PLAN.md` but not yet run).
- AWS deployment (ECS/RDS or equivalent) — deferred phase.

## Further Documentation

See **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)** for the complete file-by-file technical analysis (every module's purpose and internals, full preprocessing steps, full per-class metrics tables, database/API/dashboard detail, and a documented list of every gap between the original plan and the current implementation) and **[PROJECT_PLAN.md](PROJECT_PLAN.md)** for the original design rationale and decisions.
