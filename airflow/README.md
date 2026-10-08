# RetailPulse — Apache Airflow Automated Retraining Orchestration

## Overview
Phase 8 introduces production-grade workflow orchestration for the **Demand Forecasting** module using **Apache Airflow**. The automated retraining pipeline guarantees scheduled model updates, drift-aware decision gating, anti-leakage training partitions, and conservative model promotion safeguards.

---

## Workflow DAG: `demand_forecasting_retraining`

### Task Architecture & Chronological Flow
The DAG executes 8 discrete, idempotent tasks in strict chronological sequence:

```
[data_validation]
        │
        ▼
 [preprocessing]
        │
        ▼
[drift_monitoring]
        │
        ▼
[hyperparameter_tuning] (Drift-Aware Gating)
        │
        ▼
 [model_training]
        │
        ▼
[model_evaluation] (Conservative Promotion Check)
        │
        ▼
 [model_metadata]
        │
        ▼
 [mlflow_tracking]
```

### Task Descriptions
1. **`data_validation`**:
   - Inspects `data/daily_demand_clean.parquet` and `data/product_eligibility.csv`.
   - Validates schema (`Date`, `StockCode`, `Demand`), non-null constraints, and temporal bounds.
2. **`preprocessing`**:
   - Strictly enforces training cutoffs (`<= 2011-11-09`).
   - Asserts zero future/holdout leakage before data is passed downstream.
3. **`drift_monitoring`**:
   - Executes Phase 7 **Evidently AI** distribution drift audit on incoming demand.
   - Compares Reference baseline (`2011-01-01` to `2011-10-10`) vs. Current traffic (`2011-10-11` to `2011-12-09`).
4. **`hyperparameter_tuning`**:
   - Drift-Aware Gating: If drift is detected, runs **Optuna** Bayesian optimization on internal validation data.
   - If no drift is present, reuses proven production parameters to conserve computational resources.
5. **`model_training`**:
   - Trains candidate models (Facebook Prophet, PyTorch LSTM, Hybrid Ensemble) on historical training data.
6. **`model_evaluation`**:
   - Evaluates retrained candidates vs. current production baseline (Phase 5 Default Ensemble) on the untouched out-of-sample holdout test window (`2011-11-10` to `2011-12-09`).
   - Conservative Promotion Logic: Requires strictly superior holdout MAE (at least 2% improvement) to promote. Worse or equivalent models are rejected and the production baseline is preserved.
7. **`model_metadata`**:
   - Compiles comprehensive, auditable model metadata JSON including software package manifests.
8. **`mlflow_tracking`**:
   - Logs retraining runs, parameters, holdout metrics, promotion decisions, and artifacts to the local MLflow registry (`RetailPulse-Demand-Forecasting`).

---

## Local Development & Validation Setup

### 1. Requirements & Version Compatibility
- **Apache Airflow Version:** `2.11.2`
- **Python Version:** `3.10`
- **Dependencies:** Specified in `requirements.txt` (`apache-airflow>=2.8.0,<3.0.0`).

### 2. DAG Verification & Syntax Test (No Server Needed)
You can validate the DAG structure, task graph, and dependencies locally without launching background Airflow daemons:

```bash
# Verify DAG imports cleanly without syntax or dependency errors
python airflow/dags/demand_forecasting_retraining.py
```

### 3. Standalone Pipeline Execution
To execute the end-to-end retraining pipeline directly via Python:

```bash
python -c "
import sys; sys.path.insert(0, 'src')
from retraining_pipeline import (
    task_data_validation,
    task_preprocessing,
    task_drift_monitoring,
    task_hyperparameter_tuning,
    task_model_training,
    task_model_evaluation,
    task_model_metadata,
    task_mlflow_tracking
)

print('1. Validation:', task_data_validation())
print('2. Preprocessing:', task_preprocessing())
drift_res = task_drift_monitoring()
print('3. Drift Monitoring:', drift_res)
tune_res = task_hyperparameter_tuning(drift_detected=drift_res['dataset_drift_detected'], n_trials=3)
print('4. Tuning:', tune_res)
train_res = task_model_training(prophet_params=tune_res['prophet_params'], lstm_params=tune_res['lstm_params'])
print('5. Training:', train_res)
eval_res = task_model_evaluation(prophet_params=tune_res['prophet_params'], lstm_params=tune_res['lstm_params'])
print('6. Evaluation & Promotion:', eval_res)
meta_res = task_model_metadata(eval_result=eval_res, drift_result=drift_res)
print('7. Metadata:', meta_res)
mlflow_res = task_mlflow_tracking(eval_result=eval_res, drift_result=drift_res)
print('8. MLflow Tracking:', mlflow_res)
"
```

### 4. Running Airflow Locally (Optional Production Mode)
If running a full local Airflow webserver and scheduler:

```bash
# Set Airflow Home directory
export AIRFLOW_HOME="$(pwd)/airflow"

# Initialize SQLite database
airflow db init

# Create Admin User
airflow users create \
    --username admin \
    --firstname Retail \
    --lastname Admin \
    --role Admin \
    --email admin@retailpulse.com

# Start Airflow Standalone
airflow standalone
```

Then navigate to `http://localhost:8080`, unpause `demand_forecasting_retraining`, and trigger manually via the UI.

---

## Anti-Leakage Safeguards
- **Training Horizon Limit:** The DAG strictly caps training data at `2011-11-09`.
- **Holdout Test Quarantine:** The out-of-sample holdout test window (`2011-11-10` to `2011-12-09`) is never touched during preprocessing, Optuna tuning, or model fitting.
- **Automated Assertions:** All tasks execute automated boundary checks that fail immediately if future timestamps are detected.
