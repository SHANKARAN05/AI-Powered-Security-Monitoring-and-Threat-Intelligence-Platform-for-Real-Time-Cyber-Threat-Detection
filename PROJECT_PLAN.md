# AI-Powered Security Monitoring & Threat Intelligence System — Implementation Plan

## Context

The goal is a working system, matching the supplied architecture diagram, that uses ML to detect and classify network-based cyberattacks in real time and surface them through alerts/a dashboard — moving beyond signature-based detection to catch novel attack patterns. Two real datasets are already sitting in `Data/`, unused. The immediate priority is: **get the ML models right first** (detection + prediction), wrap them in a **simple frontend**, containerize with **Docker**, and store data in **PostgreSQL** (not MongoDB). Cloud deployment (AWS) is explicitly deferred to a later phase.

## Dataset Analysis (verified directly against the files in `Data/`)

Two structurally incompatible dataset families are present — they cannot be merged into one feature space, so this is a scope decision, not a data-quality one. **Decision: run both as parallel pipelines** sharing one code skeleton, with **CICIDS2017 as the primary/demo dataset** and **UNSW-NB15 as a secondary detection-only generalization track** (bounds effort — no second full dashboard).

**CICIDS2017** (`Data/archive/Week_filtered.csv`, primary):
- 543,735 rows × 79 columns (CICFlowMeter flow features: packet/byte stats, IAT stats, flag counts, subflow/window stats). Clean headers, no Inf/NaN found — but **23,792 exact duplicate rows** present, and `Web Attack` labels contain a mangled dash character needing cleanup.
- Cross-referencing against the 8 raw daily CSVs in `Data/archive/MachineLearningCSV/.../MachineLearningCVE/` shows this file is a deliberately curated combination: BENIGN downsampled to 201,666; the three largest attack classes (DDoS, PortScan, DoS Hulk) each capped at exactly 100,833; all smaller attack classes kept at full original count; the two extremely rare classes (Infiltration=36, Heartbleed=11) dropped entirely as unmodelable. Net: 13 classes (BENIGN + 12 attack types) — this is the intended ML-ready file; the raw per-day files are its untouched source.
- Raw daily files also have a known quirk (duplicated `Fwd Header Length`/`Fwd Header Length.1` column, inconsistent header whitespace) — not present in `Week_filtered.csv`, but the shared preprocessor should defensively handle it anyway.

**UNSW-NB15** (`Data/UNSW_NB15_training-set.csv` / `-testing-set.csv`, secondary):
- Pre-split by the dataset authors: 175,341 train rows / 82,332 test rows, 45 columns. 3 categorical fields (`proto`, `service` — uses `-` for unknown, `state`), rest numeric, plus `attack_cat` (10 classes) and binary `label`.
- Train distribution: Normal 56,000 vs 119,341 attack rows across Generic(40k)/Exploits(33.4k)/Fuzzers(18.2k)/DoS(12.3k)/Reconnaissance(10.5k)/Analysis(2k)/Backdoor(1.7k)/Shellcode(1.1k)/**Worms(130 — near-unlearnable)**.

## Critical environment blocker (verified, must fix first)

Only Python 3.14.1 is installed on this machine, and **TensorFlow has no published wheel for Python 3.14** (confirmed via `pip download` — zero matching distributions). Since Docker is already required for this project, the cleanest fix is to **do all ML development and serving inside a Docker image pinned to Python 3.11**, rather than installing a second Python locally. This also makes the environment reproducible for the eventual deployment phase.

## Architecture Decisions

1. **Two-phase AI/ML Engine**, matching the diagram exactly:
   - **Detection Phase** (unsupervised/semi-supervised, trained on benign-only data — this is what lets it catch *novel* attacks): Isolation Forest, One-Class SVM (subsampled — it's O(n²), can't fit on the full benign set), and a Keras Autoencoder (reconstruction-error based). Runs on every flow.
   - **Prediction Phase** (supervised): only flows the Detection Phase flags as anomalous are passed to a multiclass classifier (Random Forest primary, XGBoost comparison, small Keras NN comparison) that predicts attack type, then a **risk score** (0–100) is derived from a severity lookup table × model confidence × anomaly/blacklist bonuses.
   - A config flag also allows running Prediction on 100% of data for offline benchmark metrics separate from the gated "operational" metrics.
2. **Storage: PostgreSQL** via `Flask-SQLAlchemy`/`SQLAlchemy`, using JSONB columns for nested data (detector scores, raw features, CVE refs) — gives Mongo-like flexibility with a relational store, and needs no new tooling.
3. **"Real-time" demo**: a replay simulator that feeds held-out test rows into the pipeline at intervals, simulating a live feed — documented in the report as architecturally identical to a live capture pipeline (e.g., CICFlowMeter) feeding the same schema.
4. **Threat intel correlation**: static local lookup tables (a small blacklist-IP CSV snippet, a hand-built attack→CVE JSON mapping) — not a live feed integration.
5. **Frontend**: simple server-rendered Flask + Jinja2 + Chart.js dashboard (no separate React build/CORS setup) — KPI tiles, alerts-over-time, attack-type breakdown, live "recent alerts" table, model metrics page.
6. **Alerting**: log + dashboard visual highlight as baseline; simple email via Flask-Mail as an easy stretch. SMS/push explicitly out of scope for now.
7. **Docker**: `docker-compose` with services for the Flask app (Python 3.11 image) and Postgres. AWS/cloud deployment deferred — call it out in the report as "containerized and deployment-ready" rather than actually deployed.

## Preprocessing Plan

**CICIDS2017**: strip column whitespace defensively → drop duplicate `Fwd Header Length.1` if present → `drop_duplicates()` (removes the 23,792 known dupes) → clean mangled `Web Attack` label strings via a canonical mapping dict shared across train/replay code → defensively sanitize `Flow Bytes/s`/`Flow Packets/s` Inf/NaN (not observed here, but keep the guard) → log1p-transform heavy-tailed columns → `RobustScaler` → `VarianceThreshold` + correlation pruning (drop |r|>0.95 pairs) → Random Forest importance to pick ~30–40 features → stratified 70/15/15 split (safe here since `Week_filtered.csv` is already deduplicated/downsampled — note in the report that naive splits on the *raw* per-day files would leak, which is why the curated file is used). Flag `Destination Port` as a high-leakage feature and run one ablation without it.

**UNSW-NB15**: keep the authors' official train/test split intact (no repooling) → drop `id` → map `service`'s `-` to explicit `"unknown"`, one-hot/frequency-encode `proto`/`service`/`state` → log1p + `StandardScaler` (fit on train only) → variance/correlation pruning on the correlated `ct_*` group → bucket or explicitly accept near-zero recall on Worms (130 samples) → carve validation from the training set only, never touch the official test set until final reporting.

Persist every fitted encoder/scaler via `joblib`, one set per dataset, so the API and replay simulator apply identical transforms used at training time.

## Repo Structure

```
AICS Project/
├── Data/                          # existing raw data — untouched
├── data/processed/{cicids,unsw}/  # train/val/test splits + fitted encoders
├── data/threat_intel/             # blacklist_ips.csv, cve_lookup.json
├── notebooks/                     # 01_eda_cicids, 02_eda_unsw, 03_detection, 04_prediction
├── src/
│   ├── common/                    # base_preprocessor.py, base_detector.py, config.py
│   ├── preprocessing/             # cicids_preprocessor.py, unsw_preprocessor.py
│   ├── models/
│   │   ├── detection/             # isolation_forest.py, autoencoder.py, one_class_svm.py
│   │   └── prediction/            # random_forest.py, xgboost_model.py, risk_scoring.py
│   ├── pipeline/                  # train_pipeline.py, inference_pipeline.py, replay_simulator.py
│   ├── threat_intel/              # blacklist_lookup.py, cve_lookup.py
│   ├── db/                        # models.py (SQLAlchemy), postgres_client.py
│   ├── api/                       # app.py, routes/
│   └── dashboard/                 # templates/, static/
├── models/{cicids,unsw}/          # saved scaler.pkl, iforest.pkl, autoencoder.h5, rf.pkl, xgb.pkl
├── docker/                        # Dockerfile (python:3.11-slim), docker-compose.yml (app + postgres)
├── requirements.txt
└── README.md
```

`BasePreprocessor`/`BaseDetector` keep the two dataset pipelines structurally identical while isolating dataset-specific column lists/encoders, bounding the two-pipeline duplication cost.

## Postgres Schema (SQLAlchemy models, JSONB for nested fields)

- `alerts`: id, timestamp, dataset_source, flow_ref, src_ip, dst_ip, predicted_attack_cat, severity, risk_score, anomaly_score, detector_scores (JSONB), threat_intel_match (JSONB), status (new/investigating/resolved), raw_features (JSONB).
- `predictions_log`: append-only log of every scored flow, for later drift/audit analysis.
- `model_registry`: model_name, dataset, version, trained_at, metrics (JSONB), artifact_path.

## API Endpoints

`POST /api/predict` (preprocess → detect → gated predict + risk score → persist → return JSON) · `POST /api/replay/start` / `/stop` · `GET /api/alerts` (paginated/filterable) · `GET /api/alerts/<id>` · `PATCH /api/alerts/<id>` (analyst status, feeds retraining) · `GET /api/dashboard-data` (chart aggregates) · `GET /api/health`.

## Evaluation Plan

- **Detection**: Precision/Recall/F1 at chosen threshold, ROC-AUC, **PR-AUC as headline metric** (imbalance-appropriate), confusion matrix, false-positive rate framed in "alert fatigue" terms.
- **Prediction**: full per-class P/R/F1, **macro-F1 and weighted-F1 side by side** (macro exposes rare-class weakness), confusion matrix heatmap, feature-importance chart.
- Report inference latency (ms/flow) to support the real-time claim; comparison tables across the 3 detection algorithms and 3 prediction algorithms.
- If time allows, a short CICIDS-vs-UNSW generalization table (caveated as separately-trained models, not transfer learning).

## Phased Milestones

1. **Environment**: Docker image pinned to Python 3.11 with pandas/scikit-learn/tensorflow/xgboost/psycopg2/flask installed; confirm it builds before writing any model code.
2. **EDA & Preprocessing**: notebooks confirming the facts above in code; build base + dataset-specific preprocessors; persist processed splits/encoders.
3. **Detection Phase models**: Isolation Forest / Autoencoder / One-Class SVM per dataset, threshold tuning, comparison report.
4. **Prediction Phase models**: RF/XGBoost/NN multiclass classifiers, imbalance handling, severity table + risk scoring.
5. **Storage & threat intel**: Postgres schema/models, blacklist/CVE lookup tables, replay simulator.
6. **API & Dashboard**: Flask endpoints, Jinja2+Chart.js dashboard wired to replay simulator, log+email alerting.
7. **Containerization**: `docker-compose` (app + Postgres) running locally end-to-end. *(Cloud deployment: later phase, not in this scope.)*
8. **Report/demo prep**: consolidate metrics/plots, comparison discussion, live demo script.

## Risks / Tradeoffs to carry into execution

- **CICIDS2017 label-quality issues** are documented in later literature (e.g. Engelen et al.) — cite as a known dataset limitation, not a pipeline bug.
- **`Destination Port` leakage**: can near-perfectly separate some attack types; report both with/without ablations rather than relying on it silently.
- **Rare classes are unlearnable regardless of tuning** (UNSW Worms=130, CICIDS Web Attack-SqlInjection=21) — document as an accepted limitation via macro-F1, don't over-tune.
- **One-Class SVM** must be trained on a benign subsample (~20–50k rows), not the full set.
- Prefer `class_weight='balanced'` over SMOTE for imbalance — synthetic oversampling of flow features can produce physically unrealistic flows.
- Two-pipeline duplication is bounded by the shared base classes; if time gets tight, cut UNSW down to Detection-Phase-only (already the plan).

## Verification

- Unit-level: run each dataset's preprocessor against `Data/` and assert row counts, no NaN/Inf, expected column count post-selection.
- Model-level: evaluate each detector/classifier against its held-out test split, print the metrics tables from the Evaluation Plan section; sanity-check thresholds against the target FPR.
- End-to-end: `docker-compose up`, run the replay simulator against the API, confirm alerts populate Postgres and appear on the dashboard in near-real-time, and `GET /api/health` reports loaded model versions.
