# AICS — Full Technical Documentation

This document is a complete, file-by-file technical analysis of the AI-Powered Security Monitoring & Threat Intelligence System (AICS). It is written to support producing a project report: it explains *what exists*, *why it was built that way*, *what the trained models actually achieved*, and *where the implementation currently stands relative to the original plan*.

Companion documents: [`README.md`](README.md) (quick-start/reference) and [`PROJECT_PLAN.md`](PROJECT_PLAN.md) (the original design/decision document written before implementation).

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement & Motivation](#2-problem-statement--motivation)
3. [System Architecture](#3-system-architecture)
4. [Datasets — Detailed Analysis](#4-datasets--detailed-analysis)
5. [Repository Structure (Annotated)](#5-repository-structure-annotated)
6. [Module-by-Module Breakdown](#6-module-by-module-breakdown)
7. [Preprocessing Pipeline](#7-preprocessing-pipeline)
8. [Machine Learning Models](#8-machine-learning-models)
9. [Trained Model Results](#9-trained-model-results)
10. [Database Schema](#10-database-schema)
11. [REST API — Full Detail](#11-rest-api--full-detail)
12. [Dashboard / Frontend](#12-dashboard--frontend)
13. [Threat Intelligence Module](#13-threat-intelligence-module)
14. [Docker & Deployment Setup](#14-docker--deployment-setup)
15. [Known Limitations, Risks & Design Tradeoffs](#15-known-limitations-risks--design-tradeoffs)
16. [Plan vs. Implementation — Gap Analysis](#16-plan-vs-implementation--gap-analysis)
17. [Testing Status](#17-testing-status)
18. [Future Work / Roadmap](#18-future-work--roadmap)
19. [Appendix](#19-appendix)

---

## 1. Executive Summary

AICS is a network-intrusion detection and classification system built around a **two-phase AI/ML engine**:

- **Detection Phase** — three semi-supervised/unsupervised anomaly detectors (Isolation Forest, One-Class SVM, Keras Autoencoder), each trained *only on benign traffic*, so they can flag attack patterns never seen in training (the core defense against novel/zero-day-style attacks that pure signature systems miss). Runs on every incoming flow.
- **Prediction Phase** — three supervised multiclass classifiers (Random Forest, XGBoost, a small Keras neural network) that only run on flows the Detection Phase has already flagged as anomalous, predicting a specific attack category.
- **Risk Scoring** — a deterministic formula (severity-tier base score × model confidence, plus bonuses for anomaly-gate agreement and threat-intel blacklist hits) converts every prediction into a 0–100 actionable risk score.

The system is implemented twice, in parallel, for two independent public network-traffic datasets — **CICIDS2017** (primary/demo, 13 classes) and **UNSW-NB15** (secondary/generalization, 10 classes) — sharing one codebase via abstract base classes (`BasePreprocessor`, `BaseDetector`, `BasePredictor`).

Results, alerts, and predictions are persisted to **PostgreSQL** (JSONB for nested data) and surfaced through a **Flask REST API** and a **server-rendered dashboard** (Jinja2 + Chart.js). A **replay simulator** streams held-out test rows through the live pipeline to demonstrate real-time behavior without requiring a live packet-capture integration. The whole stack is **containerized** with Docker Compose (Flask app + PostgreSQL), with cloud deployment explicitly deferred.

All models described here have already been trained; their artifacts (`.pkl`/`.keras` files and `metrics.json` result files) are present in `models/cicids/` and `models/unsw/` and are analyzed in detail in [Section 9](#9-trained-model-results).

## 2. Problem Statement & Motivation

Signature-based intrusion detection can only catch attacks matching a known rule — it is blind to novel attack patterns, variants, and zero-day-style traffic. The goal of this project is a working system that uses machine learning to detect and classify network-based cyberattacks, catching patterns that have no existing signature, and to surface the result as actionable, prioritized alerts rather than raw model output.

The core design insight (from `PROJECT_PLAN.md`) is to **separate detection from classification**:
- A model trained *exclusively on benign traffic* learns what "normal" looks like and flags deviations — this is what allows it to catch attack types it never saw during training.
- A *second*, supervised model — which only ever needs to distinguish among **known** attack categories — is then used to name the specific attack type, but only for flows the first stage has already flagged. This keeps the expensive/precise classification step off of the (vast majority) of clearly-benign traffic and cleanly separates "is this suspicious?" from "what is this?".

## 3. System Architecture

### 3.1 End-to-end data flow

```
Flow record (dict of feature_name -> value, + optional src_ip/dst_ip/flow_ref)
        │
        ▼
BasePreprocessor.transform()          [per-dataset fitted scaler/encoder/feature-selection]
        │  -> numpy feature vector, in the exact column order used at training time
        ▼
Detector.score(X)                     [isolation_forest | one_class_svm | autoencoder]
        │  -> anomaly_score (higher = more anomalous)
        ▼
anomaly_flagged = anomaly_score >= threshold   [threshold picked at training time, F1-optimal]
        │
        ├── NOT flagged ──▶ predicted_label = benign class name
        │                    confidence = 1 - min(anomaly_score/threshold, 1)
        │
        └── flagged (or force_predict=True) ──▶ Predictor.predict_proba(X)
                                                   -> predicted_label = argmax class
                                                   -> confidence = max class probability
        │
        ▼
Threat-intel blacklist lookup on src_ip / dst_ip
        │
        ▼
score_risk(attack_label, confidence, anomaly_flagged, blacklist_match)
        │  -> severity tier + 0-100 risk_score
        ▼
Persist: PredictionLog row (always) + Alert row (only if anomaly_flagged)
        │
        ▼
Flask REST API  ──▶  Dashboard (Chart.js) / Alerts table / Models comparison page
```

This exact flow is implemented in `InferencePipeline.process()` (`src/pipeline/inference_pipeline.py`), which is the single most important piece of runtime logic in the system — everything else (API routes, replay simulator) is a thin wrapper around it.

### 3.2 Two parallel, structurally-identical pipelines

Because CICIDS2017 and UNSW-NB15 have incompatible feature schemas (different columns, different attack taxonomies), the project deliberately does **not** try to merge them into one feature space. Instead, it runs two independent pipelines that share code structure through three abstract base classes:

- `BasePreprocessor` (`src/common/base_preprocessor.py`) — shared clean → log-transform → one-hot-encode → scale → feature-select pipeline; subclasses (`CICIDSPreprocessor`, `UNSWPreprocessor`) implement only dataset-specific loading/cleaning/splitting and column lists.
- `BaseDetector` (`src/common/base_detector.py`) — common `fit(X_benign)` / `score(X)` / `save(path)` contract plus shared evaluation/threshold-picking utilities, implemented by `IsolationForestDetector`, `OneClassSVMDetector`, `AutoencoderDetector`.
- `BasePredictor` (`src/common/base_predictor.py`) — common `fit`/`predict`/`predict_proba`/`save` contract plus shared evaluation utilities, implemented by `RandomForestPredictor`, `XGBoostPredictor`, `NNPredictor`.

This bounds the "two datasets" duplication cost to dataset-specific column lists and cleaning logic — every downstream step (scaling, feature selection, model training, evaluation, inference, risk scoring, persistence, API, dashboard) is identical code parameterized by a `dataset` string (`"cicids"` or `"unsw"`).

### 3.3 Model/artifact directory layout at runtime

```
models/
├── cicids/
│   ├── scaler.pkl                 # StandardScaler, fit on train
│   ├── onehot_encoder.pkl         # OneHotEncoder (empty for CICIDS — no categorical cols)
│   ├── multiclass_encoder.pkl     # LabelEncoder for the 13 attack-category strings
│   ├── feature_metadata.json      # selected_features, correlated_dropped, scaled_columns
│   ├── detection/
│   │   ├── isolation_forest.pkl
│   │   ├── one_class_svm.pkl
│   │   ├── autoencoder.keras
│   │   └── metrics.json           # per-detector + ensemble metrics, both threshold strategies
│   └── prediction/
│       ├── random_forest.pkl
│       ├── xgboost.pkl
│       ├── nn_classifier.keras
│       ├── class_names.json       # ordered list of the 13 class label strings
│       ├── feature_importance.json# Random Forest feature importances
│       └── metrics.json           # per-classifier per-class/macro/weighted metrics
└── unsw/                          # identical layout, 10 classes instead of 13
```

`InferencePipeline.__init__` reads `detection/metrics.json` purely to recover the *saved F1-optimal threshold* for the chosen detector — it does not re-derive the threshold at inference time, so training and serving always agree on the operating point.

## 4. Datasets — Detailed Analysis

### 4.1 CICIDS2017 (primary)

- **Source file used**: `Data/archive/Week_filtered.csv` — a pre-curated combination file, **not** the raw per-day captures.
- **Verified size**: 543,735 lines (≈543,734 data rows) × 79 raw columns (CICFlowMeter flow features — packet/byte statistics, inter-arrival-time statistics, TCP flag counts, subflow/window statistics).
- **Curation applied by the dataset's original authors** (confirmed by cross-referencing the 8 raw daily CSVs under `Data/archive/MachineLearningCSV/.../MachineLearningCVE/`):
  - `BENIGN` downsampled to 201,666 rows.
  - The three largest attack classes (`DDoS`, `PortScan`, `DoS Hulk`) each capped at exactly 100,833 rows.
  - All smaller attack classes kept at full original count.
  - Two extremely rare classes, `Infiltration` (36 rows) and `Heartbleed` (11 rows), were **dropped entirely** as unmodelable — they do not appear anywhere in this file.
  - Net result: **13 classes** (`BENIGN` + 12 attack types).
- **Data-quality issues found and handled in code** (`src/preprocessing/cicids_preprocessor.py`):
  - **23,792 exact duplicate rows** — removed via `drop_duplicates()`.
  - **Mangled dash character** in `Web Attack` label strings (an encoding artifact in the original dataset) — cleaned via a regex canonicalization dict (`_LABEL_CLEANUP`) into `"Web Attack - Brute Force"` / `"Web Attack - XSS"` / `"Web Attack - Sql Injection"`.
  - A defensive guard strips a duplicated `Fwd Header Length.1` column and sanitizes `Flow Bytes/s` / `Flow Packets/s` Inf/NaN values (a known quirk of the raw per-day files; not actually present in `Week_filtered.csv`, but guarded anyway in case a raw file is ever substituted in).
- **The 13 target classes**: `BENIGN`, `Bot`, `DDoS`, `DoS GoldenEye`, `DoS Hulk`, `DoS Slowhttptest`, `DoS slowloris`, `FTP-Patator`, `PortScan`, `SSH-Patator`, `Web Attack - Brute Force`, `Web Attack - Sql Injection`, `Web Attack - XSS`.
- **Known literature caveat**: CICIDS2017's label quality has documented issues in later academic work (e.g. Engelen et al.) — treated here as an accepted dataset limitation, not a pipeline bug.

### 4.2 UNSW-NB15 (secondary)

- **Source files**: `Data/UNSW_NB15_training-set.csv` (175,341 rows) / `Data/UNSW_NB15_testing-set.csv` (82,332 rows) — the dataset authors' **official pre-split**, respected as-is (never repooled) so that the code's own validation split can never leak into the official test set.
- **45 columns**: mostly numeric; 3 categorical (`proto`, `service` — uses `"-"` for unknown, remapped to the literal string `"unknown"` — and `state`); plus `attack_cat` (10-class label) and binary `label`.
- **Cleaning applied**: drop the `id` column; remap `service == "-"` to `"unknown"`; rename the raw `label` column to the shared `binary_label` column name.
- **Class distribution (training set)**: `Normal` 56,000 vs. 119,341 attack rows spread across `Generic` (40k), `Exploits` (33.4k), `Fuzzers` (18.2k), `DoS` (12.3k), `Reconnaissance` (10.5k), `Analysis` (2k), `Backdoor` (1.7k), `Shellcode` (1.1k), and **`Worms` (only 130 rows — near-unlearnable)**.
- **The 10 target classes**: `Analysis`, `Backdoor`, `DoS`, `Exploits`, `Fuzzers`, `Generic`, `Normal`, `Reconnaissance`, `Shellcode`, `Worms`.

### 4.3 Why two pipelines instead of one

The two datasets have disjoint feature schemas (79 CICFlowMeter columns vs. 45 UNSW columns with different semantics) and different attack taxonomies (13 vs. 10 classes, almost no 1:1 category overlap beyond a rough DoS/Normal correspondence). Merging them into one feature space would require lossy feature engineering and would conflate two different labeling philosophies. The project instead treats this as a **scope decision**: run both as independent pipelines sharing only code structure, with CICIDS2017 carrying the full demo (dashboard, live replay) and UNSW-NB15 scoped down to a detection-and-classification generalization check (no second dashboard).

## 5. Repository Structure (Annotated)

```
AICS Project/
├── Data/                                        # inputs — see §4
│   ├── UNSW_NB15_training-set.csv
│   ├── UNSW_NB15_testing-set.csv
│   ├── archive/
│   │   ├── Week_filtered.csv                    # curated CICIDS2017 file actually used
│   │   └── MachineLearningCSV/.../MachineLearningCVE/*.csv   # raw daily captures (reference only)
│   ├── processed/{cicids,unsw}/{train,val,test}.parquet      # generated by prepare_data.py
│   └── threat_intel/
│       ├── blacklist_ips.csv                    # 6 RFC5737 placeholder IPs
│       └── cve_lookup.json                      # attack_cat -> description/CWE/CVE
├── models/{cicids,unsw}/...                     # generated artifacts — see §3.3
├── notebooks/                                   # reserved, currently empty (see §16)
├── tests/__init__.py                            # reserved, currently empty (see §17)
├── docker/
│   ├── Dockerfile                                # python:3.11-slim, gunicorn entrypoint
│   └── docker-compose.yml                        # app + postgres services
├── src/
│   ├── common/
│   │   ├── config.py                             # path constants, RANDOM_SEED, split sizes
│   │   ├── base_preprocessor.py                  # shared preprocessing pipeline (see §6.1, §7)
│   │   ├── base_detector.py                      # shared detector contract + eval/threshold utils
│   │   └── base_predictor.py                     # shared predictor contract + eval utils
│   ├── preprocessing/
│   │   ├── cicids_preprocessor.py                # CICIDS-specific load/clean/split
│   │   └── unsw_preprocessor.py                  # UNSW-specific load/clean/split
│   ├── models/
│   │   ├── detection/
│   │   │   ├── isolation_forest.py
│   │   │   ├── one_class_svm.py
│   │   │   └── autoencoder.py
│   │   └── prediction/
│   │       ├── random_forest.py
│   │       ├── xgboost_model.py
│   │       ├── nn_classifier.py
│   │       └── risk_scoring.py                   # severity table + risk formula
│   ├── pipeline/
│   │   ├── prepare_data.py                       # orchestrates preprocessing -> parquet splits
│   │   ├── train_detection.py                    # orchestrates detector training + eval
│   │   ├── train_prediction.py                   # orchestrates classifier training + eval
│   │   ├── inference_pipeline.py                 # InferencePipeline — the runtime core (see §3.1)
│   │   └── replay_simulator.py                   # streams test rows through InferencePipeline
│   ├── threat_intel/
│   │   ├── blacklist_lookup.py                   # IP -> reason lookup
│   │   └── cve_lookup.py                          # attack label -> CVE/CWE lookup
│   ├── db/
│   │   ├── models.py                              # SQLAlchemy models: Alert, PredictionLog, ModelRegistry
│   │   └── postgres_client.py                     # engine/session/init_db
│   ├── api/
│   │   ├── app.py                                 # Flask app factory, blueprint registration
│   │   ├── pipelines.py                           # process-wide InferencePipeline registry/cache
│   │   └── routes/{health,predict,alerts,replay,dashboard}.py
│   └── dashboard/
│       ├── views.py                               # page routes (dashboard/alerts/models)
│       ├── templates/{base,dashboard,alerts,models}.html
│       └── static/{app.js,style.css}
├── requirements.txt
├── PROJECT_PLAN.md                                # original design document
├── PROJECT_DOCUMENTATION.md                       # this file
└── README.md
```

## 6. Module-by-Module Breakdown

### 6.1 `src/common/config.py`

Defines all shared path constants (`PROJECT_ROOT`, `DATA_DIR`, `PROCESSED_DIR`, `THREAT_INTEL_DIR`, `MODELS_DIR`, raw file paths), `RANDOM_SEED = 42` (used everywhere a model or split needs determinism), and split sizes (`VAL_SIZE = 0.15`, `TEST_SIZE = 0.15`, the latter used only for CICIDS's from-scratch split since UNSW keeps its official test file). Also eagerly creates all required output directories on import.

### 6.2 `src/common/base_preprocessor.py`

The shared preprocessing engine (detailed step-by-step in [§7](#7-preprocessing-pipeline)). Key design points:

- `ProcessedSplits` is a dataclass bundling the final train/val/test DataFrames plus the selected feature-column list and the two label-column names.
- `BasePreprocessor` is an `ABC` with two abstract hooks (`load_and_clean`, `categorical_columns`, `log_transform_columns`) that subclasses implement, and one concrete, fully shared method, `run()`, that chains: log-transform → one-hot-encode categoricals → scale → feature-select, fitting each transform on `train` only and applying it to `val`/`test`.
- `transform()` is the **inference-time counterpart** — applies already-fitted transforms (loaded via `load_artifacts()`) to a single new row, used by `InferencePipeline`. *(Note: in the current runtime path, `InferencePipeline` actually reconstructs the feature vector directly from `feature_metadata.json`'s `selected_features` list rather than calling `transform()` — see §16 for this nuance.)*
- `encode_multiclass_labels` fits a `LabelEncoder` on the training split's raw label strings only, and safely maps any label unseen in training (in val/test) to a fallback class rather than raising.
- Feature selection (`_apply_feature_selection`) is itself a three-step process: drop zero-variance columns (`VarianceThreshold`), drop one column from every pair with `|correlation| > 0.95`, then rank the survivors by a `RandomForestClassifier(class_weight="balanced")`'s feature importances and keep the top `n_features` (default 40).
- `save_artifacts()` / `load_artifacts()` persist/restore the one-hot encoder, scaler, multiclass label encoder, and a `feature_metadata.json` capturing the selected features, the correlated-and-dropped list, the one-hot output column names, and the scaled column list.

### 6.3 `src/common/base_detector.py`

Defines the `BaseDetector` contract (`fit(X_benign)`, `score(X)` where **higher = more anomalous**, `save(path)`), plus shared, dataset-agnostic evaluation/threshold-picking utilities used identically by all three detectors:

- `evaluate_scores(y_true, scores, threshold)` — thresholds the anomaly scores into binary predictions and computes precision/recall/F1/ROC-AUC/PR-AUC/false-positive-rate/confusion-matrix, returned as a `DetectionMetrics` dataclass.
- `pick_threshold_by_benign_percentile(benign_scores, percentile=99.0)` — a **label-free** threshold choice: the 99th percentile of benign-only scores (a purely unsupervised operating point).
- `pick_threshold_by_f1(y_true, scores, n_steps=200)` — sweeps candidate thresholds and picks the one maximizing F1 **using labels**. The code explicitly documents that this is standard practice for semi-supervised anomaly detection — training itself never sees labels, only this one threshold-selection step does.

Both threshold strategies are computed and saved for every detector (see [§9.1](#91-detection-phase-results)); the **F1-optimal threshold is what `InferencePipeline` actually loads and uses at runtime**.

### 6.4 `src/common/base_predictor.py`

Defines the `BasePredictor` contract (`fit`, `predict`, `predict_proba`, `save`), plus:

- `evaluate_predictions(y_true, y_pred, class_names)` — full `sklearn.classification_report` (per-class precision/recall/F1/support) plus macro-F1, weighted-F1, and confusion matrix, all JSON-serializable.
- `balanced_class_weight(y)` — computes `{class: len(y) / (n_classes * count)}` inverse-frequency weights, used by XGBoost's per-sample weighting and the NN classifier's `class_weight`.

### 6.5 `src/preprocessing/cicids_preprocessor.py` / `unsw_preprocessor.py`

Concrete `BasePreprocessor` subclasses. See [§4](#4-datasets--detailed-analysis) and [§7](#7-preprocessing-pipeline) for what each one actually does; structurally each implements exactly three methods (`load_and_clean`, `categorical_columns`, `log_transform_columns`) and nothing else — all downstream logic is inherited.

### 6.6 `src/models/detection/*.py`

- **`isolation_forest.py`** — `IsolationForestDetector` wraps `sklearn.ensemble.IsolationForest(n_estimators=200, contamination="auto")`. Score = `-decision_function(X)` (the library's convention is higher-=-more-normal, so it's negated to match this project's higher-=-more-anomalous convention).
- **`one_class_svm.py`** — `OneClassSVMDetector` wraps `sklearn.svm.OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")`. Because an RBF-kernel SVM is roughly O(n²)–O(n³) to train, `fit()` **subsamples to at most `MAX_TRAIN_SAMPLES = 30,000` benign rows** (deterministic via `RANDOM_SEED`) before fitting — it is documented in-code as unable to fit on the full 100k+-row benign set otherwise.
- **`autoencoder.py`** — `AutoencoderDetector` is a Keras dense autoencoder: encoder dims default to `(32, 16, 8)`, mirrored back up for the decoder, linear output activation, MSE loss, Adam optimizer. `fit()` trains with early stopping (`patience=5`, `restore_best_weights=True`) monitoring `val_loss` when a validation set is supplied. `score(X)` returns the per-row mean squared reconstruction error — the standard autoencoder anomaly score.

### 6.7 `src/models/prediction/*.py`

- **`random_forest.py`** — `RandomForestPredictor` wraps `RandomForestClassifier(n_estimators=300, class_weight="balanced")`; also exposes `feature_importances()` (used to write `feature_importance.json`).
- **`xgboost_model.py`** — `XGBoostPredictor` wraps `XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, tree_method="hist")`; imbalance is handled via **explicit per-sample weights** derived from `balanced_class_weight()` (XGBoost has no built-in `class_weight` for multiclass) rather than `class_weight="balanced"`.
- **`nn_classifier.py`** — `NNPredictor` is a small Keras MLP: default hidden layers `(64, 32)` each followed by `Dropout(0.2)`, softmax output, `sparse_categorical_crossentropy` loss. `fit()` accepts a `class_weight` dict and uses the same early-stopping pattern as the autoencoder.
- **`risk_scoring.py`** — see [§8.3](#83-risk-scoring-engine) below; this is pure logic with no ML model behind it.

### 6.8 `src/pipeline/*.py`

- **`prepare_data.py`** — the top-level entry point for preprocessing: for each of `{"cicids": CICIDSPreprocessor, "unsw": UNSWPreprocessor}`, instantiates the preprocessor, calls `.run()`, and writes the three resulting splits to `Data/processed/<dataset>/{train,val,test}.parquet`.
- **`train_detection.py`** — for each dataset: loads the parquet splits, isolates benign-only training rows, trains all three detectors, computes both threshold strategies' metrics on the held-out test set for each, builds a min-max-normalized mean **ensemble** score across all three detectors and evaluates that too, saves every model artifact plus a consolidated `metrics.json`.
- **`train_prediction.py`** — for each dataset: loads the parquet splits and the fitted multiclass `LabelEncoder`, trains all three classifiers on the **full** training set (all classes, not just anomalous ones — the gating happens only at inference time), evaluates each on the test set, saves every model artifact plus `metrics.json`, `class_names.json`, and (for Random Forest) `feature_importance.json`.
- **`inference_pipeline.py`** — `InferencePipeline`, the runtime core described in [§3.1](#31-end-to-end-data-flow). Loads one detector + one predictor (defaults: `autoencoder` + `xgboost`) plus the saved F1-optimal threshold, the selected feature-column list, and the class-name list for one dataset. `process(row, persist=True)` runs one flow through the full detect→predict→risk-score→persist flow and returns a JSON-serializable result dict. A `force_predict` constructor flag allows running the Prediction Phase on 100% of rows regardless of the anomaly gate (for offline benchmark-style evaluation, as called out in `PROJECT_PLAN.md`).
- **`replay_simulator.py`** — `replay(dataset, n_rows, delay_seconds, shuffle, seed, stop_event, pipeline)` reads the dataset's held-out `test.parquet`, optionally shuffles it deterministically, and feeds up to `n_rows` rows into a live (or supplied) `InferencePipeline` one at a time with a real `time.sleep(delay_seconds)` between rows, printing true-vs-predicted-label progress and honoring a `threading.Event` for cooperative early stop. Also synthesizes demo source/destination IPs per row (see [§13.3](#133-synthetic-ip-generation-in-the-replay-simulator)).

### 6.9 `src/db/models.py` / `postgres_client.py`

SQLAlchemy `declarative_base()` models (`Alert`, `PredictionLog`, `ModelRegistry` — see [§10](#10-database-schema)) and a thin `postgres_client.py` providing `engine`/`SessionLocal` (reading `DATABASE_URL` from the environment, defaulting to `postgresql+psycopg2://aics:aics@localhost:5432/aics`), `init_db()` (creates all tables if missing — called once at Flask app startup), and a `get_session()` context manager that commits on success and rolls back on any exception.

### 6.10 `src/threat_intel/blacklist_lookup.py` / `cve_lookup.py`

Two tiny, `lru_cache`-memoized, file-backed lookup functions — see [§13](#13-threat-intelligence-module).

### 6.11 `src/api/*`

- **`app.py`** — Flask application factory (`create_app()`): configures template/static folders to point at `src/dashboard/`, calls `init_db()`, and registers six blueprints (`health`, `predict`, `alerts`, `replay`, `dashboard_api`, `views`). Module-level `app = create_app()` is what gunicorn's `src.api.app:app` target imports.
- **`pipelines.py`** — a process-wide dict cache (`_PIPELINES`) so that `get_pipeline(dataset)` constructs an `InferencePipeline` (which loads model files from disk) **once per dataset per process**, not once per request; `loaded_datasets()` exposes which datasets are currently warm, used by `/api/health`.
- **`routes/health.py`**, **`predict.py`**, **`alerts.py`**, **`replay.py`**, **`dashboard.py`** — see [§11](#11-rest-api--full-detail) for full endpoint-by-endpoint detail.

### 6.12 `src/dashboard/*`

- **`views.py`** — three simple page routes (`/dashboard`, `/alerts`, `/models`) that just render a template; all real data loading happens client-side via `fetch()` calls to the `/api/*` JSON endpoints (see [§12](#12-dashboard--frontend)).
- **`templates/base.html`** — shared layout: top navigation bar, a `Chart.js` `<script>` tag pulled from a CDN, and includes for `style.css` / `app.js`.
- **`templates/dashboard.html`**, **`alerts.html`**, **`models.html`** — page-specific content and inline `<script>` blocks (see §12).
- **`static/app.js`** — shared, dependency-free helpers used by all three page scripts: a light/dark-aware severity color/ordering scheme, a fixed 8-slot categorical color palette (never cycled — "fold to Other" past 7 categories), a `severityBadge()` HTML helper, and a `fetchJSON()` wrapper that throws on non-2xx responses using the server's `{"error": ...}` body when present.
- **`static/style.css`** — a small design-token-based stylesheet (CSS custom properties for light/dark via `prefers-color-scheme` plus an explicit `data-theme` override), KPI tiles, panels, tables, badges, and form controls. No CSS framework/build step.

## 7. Preprocessing Pipeline

The exact step order executed by `BasePreprocessor.run()` for **both** datasets (dataset-specific behavior only in steps 1 and — for UNSW — step 3):

1. **`load_and_clean()`** (dataset-specific): read raw CSV(s), drop known duplicate/junk columns, deduplicate rows (CICIDS only), clean/canonicalize label strings, handle Inf/NaN (CICIDS only, defensive), split into train/val/test (CICIDS: fresh stratified 70/15/15 split; UNSW: keep the official train/test files, carve val from train only), and encode the multiclass label column via a `LabelEncoder` fit on train only.
2. **Log transform** (`_apply_log_transform`): `np.log1p(x.clip(lower=0))` applied to a per-dataset list of heavy-tailed columns (e.g. CICIDS's `Flow Duration`, `Flow Bytes/s`; UNSW's `dur`, `sbytes`, `sload`, `rate`, etc.) — compresses long-tailed distributions before scaling.
3. **Categorical encoding** (`_apply_categorical`, UNSW only — CICIDS has none): `OneHotEncoder(handle_unknown="ignore")` fit on train's `proto`/`service`/`state` columns; unseen categories at val/test/inference time are safely encoded as all-zero rather than raising.
4. **Scaling** (`_apply_scaling`): `StandardScaler` fit on **all** numeric columns of the train split (post log-transform/one-hot), applied to val/test.
5. **Feature selection** (`_apply_feature_selection`), fit on train only:
   a. Drop zero-variance columns (`VarianceThreshold(threshold=0.0)`).
   b. Drop one column from every pair whose absolute Pearson correlation exceeds `corr_threshold` (default **0.95**).
   c. Fit a `RandomForestClassifier(n_estimators=200, class_weight="balanced")` on the remaining candidates against the multiclass label, rank by `feature_importances_`, keep the top `n_features` (default **40** — both datasets ended up with **39** selected features after the drop-if-tied behavior of `.head(n_features)` on a shorter-than-40 candidate list is accounted for).
6. **Persist artifacts** (`save_artifacts`): the fitted one-hot encoder, scaler, multiclass label encoder, and a `feature_metadata.json` recording exactly which features were selected, which were dropped for correlation, the one-hot output column names, and which columns were scaled — this is what lets the API and replay simulator reproduce the *exact* training-time transform at inference time.

A separate, lighter-weight **`transform()`** method exists for applying already-fitted artifacts to brand-new inference rows (loads via `load_artifacts()` rather than fitting).

### 7.1 High-leakage feature flagged but not yet ablated

`PROJECT_PLAN.md` explicitly flags `Destination Port` as a high-leakage feature (it can near-perfectly separate some CICIDS attack types on its own, which inflates offline metrics relative to what a deployed system would see) and calls for running one ablation without it. **`Destination Port` is in fact the #1-ranked selected feature** for the CICIDS Random-Forest feature-importance ranking (see `models/cicids/feature_metadata.json`), and **no ablation run currently exists in the codebase** — this is a known, called-out gap; see [§16](#16-plan-vs-implementation--gap-analysis).

## 8. Machine Learning Models

### 8.1 Detection Phase (semi-supervised anomaly detection)

All three detectors share the same contract and training regime: **fit on benign-only training rows**, **score every row** (higher = more anomalous), **threshold** the score into a binary flag. Two threshold strategies are computed for every detector and persisted in `metrics.json`:

| Strategy | How it's chosen | Characteristic |
|---|---|---|
| `benign_99th_percentile_threshold` | 99th percentile of benign-only validation scores — **no labels used** | Very low false-positive rate (~1–2%), but poor recall — this is the "purely unsupervised" operating point. |
| `f1_optimal_threshold` | Sweep thresholds, maximize F1 on labeled validation data | High recall, but a much higher false-positive rate — **this is the threshold `InferencePipeline` actually loads and uses.** |

An **ensemble** score (element-wise mean of each detector's min-max-normalized score) is also computed and evaluated at its own F1-optimal threshold, as a fourth row in the comparison table.

### 8.2 Prediction Phase (supervised multiclass classification)

Trained on the **full** training set (not gated), so the classifiers themselves see all classes including benign — the anomaly gate is a purely runtime/inference-time filter, not part of how these models are trained or evaluated. All three handle class imbalance, but by different mechanisms:

| Model | Imbalance handling |
|---|---|
| Random Forest | `class_weight="balanced"` (sklearn built-in) |
| XGBoost | Explicit per-sample weights via `balanced_class_weight()` (XGBoost has no native multiclass `class_weight`) |
| NN classifier | `class_weight` dict passed to `keras.Model.fit()` |

The project deliberately chose **class weighting over SMOTE/synthetic oversampling** — per `PROJECT_PLAN.md`, synthetic oversampling of flow-level features can produce physically unrealistic flows, whereas class weighting only reweights the loss, not the data.

### 8.3 Risk Scoring Engine

Pure deterministic logic (`src/models/prediction/risk_scoring.py`), no trained model. A hand-built `SEVERITY_TABLE` maps every known attack label (across **both** dataset taxonomies, so it works unmodified for either pipeline) to an `(severity_tier, base_score)` pair, e.g.:

| Severity tier | Example labels | Base score range |
|---|---|---|
| Critical | `DDoS` (95), `Worms` (98), `Backdoor` (93), `DoS Hulk` (92), UNSW `DoS` (90) | 90–100 |
| High | `Exploits` (85), `Infiltration` (88), `Web Attack - Sql Injection` (85), `Bot` (80) | 70–89 |
| Medium | `PortScan` (55), `Reconnaissance` (50), `Web Attack - Brute Force` (65) | 40–69 |
| Low | `Fuzzers` (30), `Web Attack - XSS` (35), `Generic` (20) | 10–39 |
| Informational | `BENIGN` / `Normal` | 0 |

Any label not in the table falls back to `("Medium", 50)`. Benign/Normal predictions always score exactly `0` regardless of confidence. For everything else:

```
risk_score = clamp( base_score × confidence + (10 if anomaly_flagged) + (15 if blacklist_match), 0, 100 )
```

i.e. the attack-type's inherent severity is scaled by how confident the classifier is, then boosted slightly if the detection-phase anomaly gate also agreed and/or if either endpoint IP matched the local threat-intel blacklist.

## 9. Trained Model Results

All numbers below are read directly from the `metrics.json` files currently checked into `models/cicids/` and `models/unsw/` (i.e., these are the actual results of the models as trained, not illustrative numbers).

### 9.1 Detection Phase results

**CICIDS2017** — test set:

| Detector | Threshold strategy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR |
|---|---|---|---|---|---|---|---|
| Isolation Forest | benign-99th-pct | 0.832 | 0.033 | 0.064 | 0.817 | 0.830 | 0.97% |
| Isolation Forest | **F1-optimal** | 0.719 | 0.955 | **0.820** | 0.817 | 0.830 | 53.26% |
| One-Class SVM | benign-99th-pct | 0.794 | 0.028 | 0.053 | 0.703 | 0.739 | 1.02% |
| One-Class SVM | **F1-optimal** | 0.705 | 0.941 | **0.806** | 0.703 | 0.739 | 56.25% |
| Autoencoder | benign-99th-pct | 0.979 | 0.305 | 0.466 | 0.945 | 0.950 | 0.95% |
| Autoencoder | **F1-optimal** | 0.876 | 0.923 | **0.899** | 0.945 | 0.950 | 18.61% |
| Ensemble (mean) | F1-optimal | 0.705 | 0.960 | 0.813 | 0.785 | 0.810 | 57.49% |

**UNSW-NB15** — test set:

| Detector | Threshold strategy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR |
|---|---|---|---|---|---|---|---|
| Isolation Forest | benign-99th-pct | 0.958 | 0.217 | 0.354 | 0.824 | 0.851 | 1.18% |
| Isolation Forest | **F1-optimal** | 0.742 | 0.827 | **0.783** | 0.824 | 0.851 | 35.19% |
| One-Class SVM | benign-99th-pct | 0.904 | 0.150 | 0.258 | 0.687 | 0.749 | 1.96% |
| One-Class SVM | **F1-optimal** | 0.553 | 0.992 | **0.710** | 0.687 | 0.749 | **98.35%** |
| Autoencoder | benign-99th-pct | 0.972 | 0.595 | 0.738 | 0.887 | 0.914 | 2.06% |
| Autoencoder | **F1-optimal** | 0.902 | 0.799 | **0.847** | 0.887 | 0.914 | 10.61% |
| Ensemble (mean) | F1-optimal | 0.690 | 0.828 | 0.753 | 0.795 | 0.825 | 45.60% |

**Reading these results:**
- The **Autoencoder is the clear best detector on both datasets** — highest ROC-AUC/PR-AUC and by far the best precision/recall balance at its F1-optimal threshold, and it is (not coincidentally) the default detector loaded by `InferencePipeline`.
- **The F1-optimal threshold trades a large amount of false-positive rate for recall.** On CICIDS, every detector's F1-optimal FPR is 18–57%; on UNSW, One-Class SVM's F1-optimal threshold has a **98.35% false-positive rate** — at that operating point the model is flagging almost everything as anomalous, which is a real, documented weakness of One-Class SVM on this dataset rather than a useful detector. This is an important finding for a report: **the choice of threshold strategy matters enormously**, and a production deployment would likely need a middle-ground threshold (or a precision-recall-curve-based operating point) rather than either extreme currently computed.
- The **ensemble does not beat the Autoencoder alone** on either dataset — averaging in the weaker One-Class SVM/Isolation-Forest scores drags the ensemble's precision down without a compensating recall gain.

### 9.2 Prediction Phase results

**CICIDS2017** — test set (13 classes, weighted by class support = 70,887 rows):

| Model | Macro-F1 | Weighted-F1 | Accuracy |
|---|---|---|---|
| Random Forest | 0.884 | 0.998 | 0.998 |
| **XGBoost** | **0.910** | **0.998** | 0.998 |
| NN classifier | 0.728 | 0.968 | 0.962 |

Weakest classes across all three models (support in test set): `Web Attack - Sql Injection` (only 3 test rows — F1 ranges 0.03–0.67 depending on model and is statistically meaningless at n=3), `Web Attack - XSS` (98 rows, F1 0.03–0.45), `Web Attack - Brute Force` (221 rows, F1 0.05–0.75). All other classes (including `DDoS`, `PortScan`, the `DoS` variants, and both `*-Patator` brute-force categories) achieve F1 > 0.93 for both tree-based models. The NN classifier is noticeably worse on the rare Web-Attack classes and on `Bot` (293 rows) than either tree-based model — consistent with neural nets generally needing more data per class than tree ensembles to learn minority classes well.

**UNSW-NB15** — test set (10 classes, 82,332 rows):

| Model | Macro-F1 | Weighted-F1 | Accuracy |
|---|---|---|---|
| Random Forest | 0.510 | 0.786 | 0.752 |
| **XGBoost** | **0.522** | 0.752 | 0.704 |
| NN classifier | 0.380 | 0.698 | 0.643 |

The much lower macro-F1 here (vs. CICIDS) reflects genuinely hard classes: `Analysis` (F1 0.03–0.07 across all three models), `Backdoor` (F1 0.07–0.10), and `DoS` (F1 0.05–0.27) are consistently the weakest — these three categories are known in the UNSW-NB15 literature to have substantial feature-space overlap with other categories (particularly `Exploits` and `Generic`), so this is best reported as a **documented dataset characteristic**, not a defect in these particular models. Interestingly, `Worms` (only 44 test rows) actually scores reasonably (F1 0.43–0.68) for the tree-based models — a result that should be read cautiously given the tiny support rather than presented as a genuine strength.

### 9.3 Selected features per dataset

Both datasets converged on **39 selected features** (of a `n_features=40` target — one fewer than the cap because a duplicate/tied importance rank was absorbed). Full lists are in `models/{cicids,unsw}/feature_metadata.json` and reproduced in the [Appendix](#19-appendix).

## 10. Database Schema

Three SQLAlchemy models (`src/db/models.py`), created automatically on Flask startup via `init_db()` → `Base.metadata.create_all(engine)`:

### `alerts` — one row per flow the Detection Phase flagged as anomalous
| Column | Type | Notes |
|---|---|---|
| `id` | Integer, PK | |
| `timestamp` | DateTime(tz) | |
| `dataset_source` | String(20) | `"cicids"` or `"unsw"` |
| `flow_ref` | String(64), nullable | e.g. `"cicids-replay-42"` from the replay simulator |
| `src_ip` / `dst_ip` | String(45), nullable | IPv4/IPv6-length; synthetic in the current demo (see §13.3) |
| `predicted_attack_cat` | String(64) | Prediction-phase output, or the benign label if not flagged (though only flagged rows are ever inserted here) |
| `severity` | String(20) | One of `Critical`/`High`/`Medium`/`Low`/`Informational` |
| `risk_score` | Float | 0–100 |
| `anomaly_score` | Float | Raw detector score |
| `detector_scores` | **JSONB** | `{detector_name: score}` — currently only the one active detector, structured to support multiple |
| `threat_intel_match` | **JSONB**, nullable | `{ip, reason}` if a blacklist hit occurred |
| `status` | String(20), default `"new"` | Analyst workflow: `new` / `investigating` / `resolved` |
| `raw_features` | **JSONB**, nullable | The selected feature values for this flow, for later inspection |

### `predictions_log` — append-only log of **every** scored flow (not just flagged ones)
`id`, `timestamp`, `dataset_source`, `predicted_attack_cat`, `binary_flag` (1 if the anomaly gate fired, else 0), `anomaly_score`, `confidence`, `risk_score`. Intended for later drift/audit analysis — every flow that passes through `InferencePipeline.process()` gets one row here regardless of outcome.

### `model_registry` — reserved for model-version tracking
`id`, `model_name`, `dataset`, `version`, `trained_at`, `metrics` (JSONB), `artifact_path`. **Not currently written to by any pipeline script** — the table exists in the schema but is not yet populated; model metadata currently lives only in the `metrics.json` files on disk. See [§16](#16-plan-vs-implementation--gap-analysis).

JSONB was chosen specifically (per `PROJECT_PLAN.md`) to get Mongo-like flexibility for nested/variable-shape data (detector scores, raw features, threat-intel matches) while staying inside a single relational Postgres database rather than introducing a second datastore.

## 11. REST API — Full Detail

All routes are Flask blueprints registered in `src/api/app.py`, all under the `/api` prefix except the dashboard page routes (`/`, `/dashboard`, `/alerts`, `/models`) which are unprefixed HTML views.

### `GET /api/health`
Returns, per dataset, whether trained model artifacts exist on disk (`detection/metrics.json` and `prediction/metrics.json` both present) and whether an `InferencePipeline` for that dataset is currently loaded in the process-wide cache. Does **not** currently report loaded model *versions* despite `PROJECT_PLAN.md`'s original endpoint description calling for that — see §16.

### `POST /api/predict`
Body: `{"dataset": "cicids"|"unsw", "features": {feature_name: value, ...}, "src_ip"?, "dst_ip"?, "flow_ref"?}`. Validates `dataset` is one of the two known values and `features` is an object; on success, builds a row dict, fetches (or lazily creates) that dataset's cached `InferencePipeline`, calls `.process(row)` (which persists a `PredictionLog` row always and an `Alert` row if flagged), and returns the full result JSON (attack label, severity, confidence, anomaly score/flag, risk score, threat-intel match).

### `GET /api/alerts`
Query params: `page` (default 1), `per_page` (default 25, capped at 200), `severity`, `dataset`, `status` — all optional filters, ANDed together. Returns `{total, page, per_page, alerts: [...]}`, ordered by timestamp descending.

### `GET /api/alerts/<id>`
Full single-alert detail including `raw_features`; 404 JSON error if not found.

### `PATCH /api/alerts/<id>`
Body: `{"status": "new"|"investigating"|"resolved"}`. Validates against the allowed set; 400 if invalid, 404 if the alert doesn't exist; otherwise updates and returns the serialized alert.

### `POST /api/replay/start`
Body: `{"dataset"?: default "cicids", "n_rows"?: default 500, "delay"?: default 1.0}`. Refuses with `409` if a replay thread is already running. Otherwise spins up a **daemon thread** running `replay_simulator.replay(...)` with a fresh `threading.Event` for cooperative stop, reusing the cached `InferencePipeline` for that dataset (so replayed alerts use the same warm model as `/api/predict` would).

### `POST /api/replay/stop`
Sets the stop event if a replay is running; the replay loop checks it between rows and exits early. Returns `{"status": "not running"}` if nothing is active.

### `GET /api/replay/status`
`{"running": bool, "dataset": str|null}`.

### `GET /api/dashboard-data`
Loads the most recent 2,000 alerts, computes severity/attack-type/top-10-source-IP breakdowns via `collections.Counter`, and returns the 20 most recent alerts in summary form alongside the total alert count. This single endpoint powers every widget on the Dashboard page.

### `GET /api/models`
Reads and returns both `detection/metrics.json` and `prediction/metrics.json` for both datasets directly from disk (no caching) — this is what powers the Models comparison page and is the source of all numbers in [§9](#9-trained-model-results).

**Note**: there is no authentication/authorization layer on any endpoint — appropriate for a local demo/coursework deployment, but a gap that would need addressing before any real internet-facing deployment.

## 12. Dashboard / Frontend

A deliberately simple, dependency-light frontend: Flask + Jinja2 templates, vanilla JavaScript, and Chart.js loaded from a CDN — explicitly chosen over a separate React/SPA build (per `PROJECT_PLAN.md`) to avoid a second build toolchain and CORS configuration.

- **Dashboard page** (`/dashboard`): replay controls (dataset selector, row-count and delay inputs, start/stop buttons wired to `/api/replay/*`), four KPI tiles (total alerts, critical count, high count, average risk score of the 20 most recent alerts), a severity bar chart (fixed 5-tier ordinal color ramp, always shows all 5 tiers even at zero), an attack-type bar chart (horizontal, folds anything past the top 7 categories into an "Other" bucket so the fixed 8-slot categorical palette is never stretched or cycled), and a live recent-alerts table. Polls `/api/dashboard-data` and `/api/replay/status` every 4 seconds via `setInterval`.
- **Alerts page** (`/alerts`): dataset/severity/status filter dropdowns, a paginated (25/page) alerts table, and an inline per-row status `<select>` that PATCHes `/api/alerts/<id>` and reloads on change.
- **Models page** (`/models`): renders the detection-phase comparison table (precision/recall/F1/ROC-AUC/PR-AUC/FPR per detector) and the prediction-phase comparison table (macro-F1/weighted-F1 per classifier) for both datasets from `/api/models`, plus an inline note explaining that a low macro-F1 relative to weighted-F1 indicates rare-class weakness.
- **Theming**: `style.css` defines light-mode CSS custom properties on `:root`, overridden for dark mode both via `prefers-color-scheme` and an explicit `data-theme="dark"` attribute override; `app.js` mirrors the same light/dark split for chart colors so Chart.js canvases match the surrounding page theme.

## 13. Threat Intelligence Module

Deliberately **static, local lookup tables** rather than a live feed integration (per `PROJECT_PLAN.md` scope decision) — described in-code as what a production version would instead sync from a live feed (e.g. abuse.ch, AlienVault OTX).

### 13.1 IP blacklist (`Data/threat_intel/blacklist_ips.csv`, `src/threat_intel/blacklist_lookup.py`)
6 rows, using **RFC 5737 documentation/example IP ranges** (`192.0.2.0/24`, `198.51.100.0/24`, `203.0.113.0/24`) explicitly labeled as placeholder/sample data, not real threat data. `check_ip(ip)` does an O(1) dict lookup (the CSV is loaded once and memoized via `functools.lru_cache`) and returns `{"ip": ..., "reason": ...}` or `None`.

### 13.2 CVE/CWE lookup (`Data/threat_intel/cve_lookup.json`, `src/threat_intel/cve_lookup.py`)
A hand-built JSON mapping from attack-category label (spanning **both** dataset taxonomies) to a short human-readable description plus `cwe_refs`/`cve_refs` lists. Most entries have an empty `cve_refs` (these are attack *categories*, not specific vulnerabilities); the one exception, `Heartbleed`, correctly cites **CVE-2014-0160**. `lookup(attack_label)` falls back to a generic "Unclassified/uncommon attack category" entry for anything not in the table. **Note**: this lookup function exists but is **not currently called anywhere in `InferencePipeline` or the API routes** — see §16.

### 13.3 Synthetic IP generation in the replay simulator
Neither CICIDS2017 nor UNSW-NB15 retains real source/destination IP addresses in their published feature sets (stripped upstream by the datasets' authors specifically to prevent identity-based shortcuts in ML models trained on them). Since the replay simulator needs *something* to populate `src_ip`/`dst_ip` for the dashboard/alerts UI and to exercise the blacklist-lookup code path live, it generates synthetic `10.x.x.x` addresses per row, occasionally drawing from the blacklist pool instead (5% of rows by default) specifically so a demo run visibly exercises threat-intel correlation. This is explicitly commented in-code as a demo mechanism, not a claim about real traffic.

## 14. Docker & Deployment Setup

### `docker/Dockerfile`
`python:3.11-slim` base (chosen because the host machine's only available Python, 3.14, has no published TensorFlow wheel — Docker is the only way to get a working TensorFlow install for this project, not merely a convenience). Installs `build-essential` and `libpq-dev` (needed to build `psycopg2` and any native wheels), installs `requirements.txt`, copies the full repo in, sets `PYTHONUNBUFFERED=1` and `PYTHONPATH=/app` (so `src.*` imports resolve without an installed package), exposes port 5000, and runs via **gunicorn** (`gunicorn -b 0.0.0.0:5000 src.api.app:app`) rather than Flask's development server.

### `docker/docker-compose.yml`
Two services:
- **`app`** — built from the Dockerfile with build context set to the repo root (`context: ..`), port `5000:5000`, `DATABASE_URL` pointed at the `postgres` service by its Compose network hostname, `FLASK_ENV=development`, and **bind mounts** for `src/`, `Data/`, `models/`, and `notebooks/` (so code and model-artifact changes on the host are reflected inside the container without a rebuild) — `depends_on: postgres: condition: service_healthy` so the app doesn't start racing an unready database.
- **`postgres`** — `postgres:16-alpine`, credentials `aics`/`aics`/`aics`, port `5432:5432` exposed to the host for local debugging/psql access, a named volume `pgdata` for persistence across restarts, and a `pg_isready`-based healthcheck (5s interval, 10 retries) that gates the `app` service's startup.

### Deployment status
Explicitly scoped as **"containerized and deployment-ready," not actually deployed** — cloud deployment (AWS) is called out in `PROJECT_PLAN.md` as a deferred, later-phase item, not part of the current implementation. There is no CI/CD configuration, no cloud IaC (Terraform/CloudFormation/etc.), and no production-hardening (TLS, secrets management, authentication) anywhere in the repo.

## 15. Known Limitations, Risks & Design Tradeoffs

Carried forward from `PROJECT_PLAN.md`'s risk list, and cross-checked against what actually shipped:

- **CICIDS2017 label-quality issues** are a documented dataset limitation in later literature (e.g. Engelen et al.), not a pipeline bug — accepted as-is.
- **`Destination Port` leakage** — flagged as able to near-perfectly separate some CICIDS attack types (and indeed ranks #1 in feature importance); the planned "with/without" ablation to quantify its effect has **not** been run (see §16).
- **Rare classes are unlearnable regardless of tuning** — UNSW `Worms` (130 train rows) and CICIDS `Web Attack - Sql Injection` (21 total rows across the dataset, 3 in the test split) are accepted limitations, reported via macro-F1 rather than papered over with more tuning. Confirmed in the actual results (§9.2): both are among the lowest-F1 classes in their respective datasets.
- **One-Class SVM must be trained on a benign subsample**, not the full benign set, due to its O(n²)–O(n³) training cost — implemented as a hard 30,000-row cap.
- **Class weighting was chosen over SMOTE/oversampling** specifically because synthetic oversampling of flow-level numeric features can produce physically unrealistic flows.
- **Two-pipeline duplication is bounded** by the shared `Base*` classes — confirmed in the actual code, where each dataset's preprocessor/detector/predictor subclasses are only tens of lines each, with all real logic living once in the shared base classes.
- **The F1-optimal detection threshold trades a large false-positive rate for recall** — this is real and visible in the actual results (§9.1), most dramatically as a 98.35% FPR for One-Class SVM on UNSW-NB15. A production deployment would likely want a different, less recall-maximizing operating point (or a cost-sensitive threshold choice) to avoid alert fatigue — this exact tradeoff (framed as "alert fatigue") was anticipated in `PROJECT_PLAN.md`'s evaluation plan.
- **No authentication on any API endpoint** — acceptable for a local/coursework demo, not for any internet-facing deployment.
- **Synthetic demo IPs** — the replay simulator's src/dst IPs are generated, not real, by construction (see §13.3), so IP-based dashboard/alert data should be read as illustrative of the *mechanism*, not as real observed traffic origins.

## 16. Plan vs. Implementation — Gap Analysis

This section exists specifically because the user asked for "a note of each and every thing" — including where the shipped system differs from `PROJECT_PLAN.md`'s original design. None of these are necessarily problems; they are simply worth calling out explicitly in a report so the plan and the delivered system are both represented accurately.

| Planned (per `PROJECT_PLAN.md`) | Actual state |
|---|---|
| Notebooks `01_eda_cicids`, `02_eda_unsw`, `03_detection`, `04_prediction` | `notebooks/` directory exists but is **empty** — no notebooks were produced; equivalent analysis lives in this document and `PROJECT_PLAN.md`'s prose instead. |
| `Destination Port` with/without ablation | **Not implemented** — no ablation script or flag exists in `src/pipeline/train_detection.py` or `train_prediction.py`. |
| Simple email alerting via `Flask-Mail` as an "easy stretch" | `flask-mail` is listed in `requirements.txt` but **no code imports or uses it anywhere** in `src/`. |
| `model_registry` table for model_name/dataset/version/metrics tracking | Table is defined in `src/db/models.py` but **no pipeline script writes to it** — model metadata currently lives only in on-disk `metrics.json` files, not the database. |
| `GET /api/health` reports "loaded model versions" | Actual implementation reports artifact-presence and pipeline-loaded booleans per dataset, but **not a version string** (there is no versioning scheme in play yet, consistent with the `model_registry` gap above). |
| CVE/CWE threat-intel lookup (`cve_lookup.py`) wired into alert enrichment | The lookup module is fully implemented and correct, but **is not currently called from `InferencePipeline` or any API route** — alerts carry `threat_intel_match` (IP blacklist only), not a CVE/CWE annotation, despite the module existing and being ready to use. |
| Unit-level verification ("assert row counts, no NaN/Inf, expected column count post-selection") | **No automated tests exist** — `tests/` contains only an empty `__init__.py` (see §17). |
| A short CICIDS-vs-UNSW generalization table "if time allows" | Not produced as a standalone artifact, though the side-by-side structure of this document's §9 effectively provides the same comparison. |

Everything else in `PROJECT_PLAN.md` — the two-phase engine, the specific detector/predictor algorithm choices, PostgreSQL + JSONB, the replay-simulator "real-time" framing, the static threat-intel tables, the Flask+Jinja2+Chart.js dashboard, and the Docker Compose setup — **matches the shipped implementation closely**, including most specific hyperparameter and architecture choices called out in the plan (e.g. One-Class SVM's subsampling cap, the exact severity-table structure, `class_weight="balanced"` over SMOTE).

## 17. Testing Status

`tests/` currently contains only an empty `__init__.py` — there are **no automated tests** anywhere in the repository (no unit tests for the preprocessors, detectors, predictors, risk-scoring formula, API routes, or database models). All verification to date has been manual/observational: running the training pipelines and inspecting the printed metrics and the resulting `metrics.json` files (which is exactly what this document's §9 is built from). A report should treat this as an explicit, currently-open gap rather than assume test coverage exists.

## 18. Future Work / Roadmap

In rough priority order for closing the gaps identified in §16:

1. **Wire up alert enrichment**: call `cve_lookup.lookup()` from `InferencePipeline.process()` and store the result alongside `threat_intel_match`, since the lookup logic already exists and just isn't connected.
2. **Add automated tests**: at minimum, unit tests for `score_risk()` (pure function, easy to test exhaustively), the preprocessing pipeline's row-count/NaN/Inf invariants (as originally planned), and the API route contracts.
3. **Populate `model_registry`** from `train_detection.py`/`train_prediction.py` so model versioning and the `/api/health` "loaded model version" claim become real.
4. **Run the `Destination Port` ablation** flagged in the original plan, to quantify how much of CICIDS's very high prediction-phase accuracy is attributable to that one leaky feature.
5. **Wire up `Flask-Mail`** for Critical-severity alert notifications, as originally scoped as an "easy stretch."
6. **Re-tune the detection-phase operating point** away from the pure F1-optimal threshold (which produces very high false-positive rates, up to 98% for UNSW's One-Class SVM) toward something that better balances analyst alert fatigue against recall — e.g. picking a threshold at a fixed target FPR from the ROC curve.
7. **Cloud deployment** (AWS or equivalent) — explicitly deferred in the original plan and still out of scope today.

## 19. Appendix

### 19.1 CICIDS2017 — 13 target classes
`BENIGN`, `Bot`, `DDoS`, `DoS GoldenEye`, `DoS Hulk`, `DoS Slowhttptest`, `DoS slowloris`, `FTP-Patator`, `PortScan`, `SSH-Patator`, `Web Attack - Brute Force`, `Web Attack - Sql Injection`, `Web Attack - XSS`.

### 19.2 UNSW-NB15 — 10 target classes
`Analysis`, `Backdoor`, `DoS`, `Exploits`, `Fuzzers`, `Generic`, `Normal`, `Reconnaissance`, `Shellcode`, `Worms`.

### 19.3 CICIDS2017 — 39 selected features (post feature-selection)
`Destination Port`, `Init_Win_bytes_backward`, `Fwd Packet Length Max`, `Bwd Header Length`, `Flow Duration`, `Flow IAT Std`, `Fwd IAT Mean`, `Total Length of Bwd Packets`, `Fwd IAT Total`, `Packet Length Variance`, `min_seg_size_forward`, `Bwd Packet Length Max`, `Fwd IAT Std`, `Fwd Header Length`, `Init_Win_bytes_forward`, `Total Length of Fwd Packets`, `Subflow Fwd Bytes`, `Fwd Packet Length Std`, `Fwd Packet Length Mean`, `Flow Bytes/s`, `Bwd Packets/s`, `Fwd IAT Min`, `Bwd Packet Length Min`, `Flow IAT Min`, `Total Fwd Packets`, `Active Min`, `PSH Flag Count`, `Bwd IAT Mean`, `Bwd IAT Min`, `Bwd IAT Total`, `Fwd Packets/s`, `Bwd IAT Std`, `Bwd IAT Max`, `Active Mean`, `Active Max`, `ACK Flag Count`, `Min Packet Length`, `Fwd Packet Length Min`, `URG Flag Count`, `Down/Up Ratio`.

### 19.4 UNSW-NB15 — 39 selected features (post feature-selection)
`sbytes`, `smean`, `ct_srv_src`, `sttl`, `dur`, `dbytes`, `proto_udp`, `rate`, `dload`, `ct_src_ltm`, `ct_dst_sport_ltm`, `dmean`, `service_unknown`, `sinpkt`, `ct_dst_ltm`, `service_dns`, `sjit`, `dinpkt`, `ct_state_ttl`, `ackdat`, `djit`, `tcprtt`, `service_http`, `dpkts`, `spkts`, `dtcpb`, `stcpb`, `ct_flw_http_mthd`, `dttl`, `trans_depth`, `response_body_len`, `proto_unas`, `swin`, `state_INT`, `state_CON`, `service_smtp`, `is_sm_ips_ports`, `proto_sctp`, `service_ftp-data`, `proto_ospf`.

### 19.5 Full severity table (`src/models/prediction/risk_scoring.py`)

| Attack label | Severity | Base score |
|---|---|---|
| Worms | Critical | 98 |
| DDoS | Critical | 95 |
| Backdoor | Critical | 93 |
| DoS Hulk | Critical | 92 |
| DoS GoldenEye | Critical | 90 |
| DoS slowloris | Critical | 90 |
| DoS Slowhttptest | Critical | 90 |
| DoS (UNSW generic) | Critical | 90 |
| Infiltration | High | 88 |
| Exploits | High | 85 |
| Web Attack - Sql Injection | High | 85 |
| Shellcode | High | 82 |
| Bot | High | 80 |
| Web Attack - Brute Force | Medium | 65 |
| FTP-Patator | Medium | 60 |
| SSH-Patator | Medium | 60 |
| PortScan | Medium | 55 |
| Reconnaissance | Medium | 50 |
| *(any unrecognized label)* | Medium (default) | 50 |
| Web Attack - XSS | Low | 35 |
| Fuzzers | Low | 30 |
| Analysis | Low | 25 |
| Generic | Low | 20 |
| BENIGN / Normal | Informational | 0 |

Bonuses: **+10** if the detection-phase anomaly gate also flagged the flow; **+15** if either endpoint IP matched the local threat-intel blacklist. Final score is clamped to `[0, 100]`.

### 19.6 Environment note

The host machine for this project has only Python 3.14.1 installed, for which **no published TensorFlow wheel exists** (verified via `pip download` returning zero matching distributions). This is why all ML development and serving is designed to run inside the Docker image pinned to Python 3.11 (`docker/Dockerfile`) rather than requiring a second local Python installation — a decision that also happens to make the environment fully reproducible ahead of any future deployment phase.
