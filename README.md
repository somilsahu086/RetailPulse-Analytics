# RetailPulse-Analytics

AI-Powered Customer Analytics & Demand Forecasting Platform

## Project structure

```
RetailPulse-Analytics/
├── app.py                      # Streamlit dashboard (Phase 9 presentation layer)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── src/                        # Core ML, forecasting, and pipeline code
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── forecasting.py
│   ├── baseline_forecasting.py
│   ├── ensemble_forecasting.py
│   ├── lstm_forecasting.py
│   ├── evaluation.py
│   ├── inventory_planning.py
│   ├── drift_monitoring.py
│   ├── segmentation.py
│   ├── churn_model.py
│   ├── mlops_tracking.py
│   ├── optuna_tuning.py
│   ├── model_metadata.py
│   └── retraining_pipeline.py
├── data/                       # Input data (read-only for Streamlit layer)
│   ├── cleaned_data.xlsx
│   ├── customer_segments.csv
│   ├── daily_demand_clean.parquet
│   ├── online_retail_II.xlsx
│   └── product_eligibility.csv
├── model/                      # Trained model artifacts (read-only)
│   ├── scaler.pkl
│   ├── kmeans.pkl
│   └── churn_model.pkl
├── outputs/                    # Forecasting, drift, and retraining outputs
│   └── forecasting/
├── notebooks/                  # EDA and segmentation notebooks
│   ├── demand_forecasting_eda.ipynb
│   └── customer_segment.ipynb
├── airflow/                    # Apache Airflow DAGs for retraining (Phase 8)
│   └── dags/
└── tests/                      # Test suite
    ├── test_phase9_dashboard.py
    ├── test_phase7_drift.py
    ├── test_phase8_orchestration.py
    ├── test_phase6_mlops.py
    ├── test_phase5_ensemble.py
    ├── test_ensemble_forecasting.py
    ├── test_baseline_forecasting.py
    ├── test_advanced_forecasting.py
    ├── test_inventory_planning.py
    └── test_preprocessing_and_evaluation.py
```

## Phases

| Phase | Component | Location |
|-------|-----------|----------|
| Phase 1 | EDA & Catalog Selection | `notebooks/demand_forecasting_eda.ipynb` |
| Phase 2 | Preprocessing & Daily Demand Series | `src/preprocessing.py` |
| Phase 3 | Baseline Benchmarks (Naive, MA 28D) | `src/baseline_forecasting.py` |
| Phase 4 | Advanced Models (Prophet, LightGBM) | `src/forecasting.py`, `src/lstm_forecasting.py` |
| Phase 5 | Hybrid Prophet + PyTorch LSTM Ensemble & Inventory Planning | `src/ensemble_forecasting.py`, `src/inventory_planning.py` |
| Phase 6 | MLflow Tracking & Optuna Hyperparameter Governance | `src/mlops_tracking.py`, `src/optuna_tuning.py` |
| Phase 7 | Evidently AI Data Drift & Quality Monitoring | `src/drift_monitoring.py` |
| Phase 8 | Apache Airflow Automated Weekly Retraining | `airflow/dags/`, `src/retraining_pipeline.py` |
| Phase 9 | Interactive Streamlit Dashboard | `app.py` |

## Streamlit dashboard (app.py)

The Streamlit layer (`app.py`) is the presentation surface only. It consumes data, models, forecasts, drift reports, and retraining artifacts read-only. No retraining or data modification happens here.

### Pages
- **Overview** — Business filters (Country, date range), KPIs, monthly revenue trend, top countries and products, CSV downloads
- **Customers** — Segmentation KPIs, cluster distribution, CustomerID search, churn predictor
- **Forecast** — 30-day demand trajectory with confidence interval, horizon and model selectors, weekly aggregation, daily export
- **Inventory** — Safety stock, reorder point, order-up-to level, warehouse simulator, stockout risk pie chart
- **Models** — Holdout model comparison (MAE/RMSE), accuracy bar chart, operational accuracy notes
- **Drift** — Evidently AI dataset drift status, feature-level KS test breakdown
- **Retraining** — Airflow DAG status, model promotion decision, retraining metrics, MLOps manifest

### Run locally
```bash
streamlit run app.py --server.port 8501
```

Then open http://localhost:8501 in your browser.

### Verification
```bash
python -m py_compile app.py
pytest tests/test_phase9_dashboard.py -q
```

## Installation

```bash
pip install -r requirements.txt
```

## Data and models

Input data (`data/`) and trained models (`model/`) are treated as read-only by the Streamlit presentation layer. The `app.py` file loads them via existing loader functions and `joblib.load` only.

## Branch policy

Streamlit work happens on `feat/streamlit-business-filters`. Each task is committed separately so individual tasks can be reversed if needed. PR descriptions include only What/why and Testing sections.
