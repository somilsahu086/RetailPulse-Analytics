# RetailPulse Demand Forecasting — Phase 7 Drift Monitoring Report (Corrected)

**Generated:** 2026-10-18 07:52:19Z  
**Evidently AI Version:** 0.7.23  
**Monitoring Scope:** Development Period Baseline vs Development Monitoring Windows  
- **Reference Baseline Period:** `2011-01-01` to `2011-08-11` (11,150 SKU-day observations across 50 SKUs)  
- **Current Feature Monitoring Period:** `2011-08-12` to `2011-11-07` (4,400 SKU-day observations across 50 SKUs)  
- **Holdout Quarantine:** Quarantined period (`2011-11-10` to `2011-12-09`) strictly preserved and untouched.

---

## 1. Executive Summary & Status Scorecard

| Monitoring Pillar | Evaluated Metric | Status | Status Label | Detail |
|:---|:---|:---:|:---:|:---|
| **Data Schema & Quality** | Nulls, Non-finites, Column presence | 0 missing, 0 inf | **PASS** | Schema valid, 0 nulls across 100% of rows. |
| **Feature Distribution Drift** | Normalized Wasserstein distance > 0.05 | 12 / 12 features | **DRIFT_DETECTED** | Seasonal volume expansion during Autumn ramp. |
| **Dataset Drift** | Drifted feature share >= 50% | 100.0% | **DATASET_DRIFT** | Structural retail volume shift in Q3/Q4. |
| **Prediction Distribution Drift** | K-S test on model predictions (p <= 0.05) | p = 0.0102 | **DRIFT_DETECTED** | Forecasts shifted +3.1% following demand ramp. |
| **30-Day Aggregate Volume Error** | Net error across 30 days (2011-10-11 to 2011-11-09) | Vol Error = 5.92%, Bias = +5.92% | **PERFORMANCE_HEALTHY** | Net volume discrepancy across entire 30-day horizon. |
| **Official Weekly Portfolio MAPE** | Mean of 4 weekly portfolio MAPEs (2011-10-11 to 2011-11-07) | Mean MAPE = **14.81%** | **BASELINE_FROZEN** | Official evaluation metric; target (<=12%) not yet met. |

> [!IMPORTANT]
> **Key Methodological Distinction (Metric Naming & Date Boundaries):**
> 1. **30-Day Aggregate Volume Error (5.92%):** Evaluated over the full 30-day Window 3 horizon (`2011-10-11` to `2011-11-09`, 1,500 observations). In Weeks 1 and 2, the model over-forecasted (+24.93% and +27.03%), while in Weeks 3 and 4 demand surged and the model under-forecasted (-5.40% and -1.86%). When actuals and forecasts are pooled across all 30 days into one total volume ($98,045$ vs $92,569$), positive and negative errors cancel each other out, compressing the net volume discrepancy to **5.92%**.
> 2. **Official Mean Weekly Portfolio MAPE (14.81%):** Evaluated over the strict 4-week window (`2011-10-11` to `2011-11-07`, 1,400 observations). This is the official Phase 1–6 project benchmark. Errors are evaluated per 7-day week and do not cancel across weeks. **The baseline forecasting model has not yet met the <=12% acceptance target.**
> 3. **Both Periods Confined to Development Data:** Both the 28-day window (ending `2011-11-07`) and the 30-day window (ending `2011-11-09`) strictly precede the quarantined holdout (`2011-11-10` to `2011-12-09`).

---

## 2. Feature Drift Details & Statistical Semantics

> [!NOTE]
> **Drift Metric Semantics & Collinearity:**
> - For numerical features with sample sizes $N > 1,000$ ($N_{\text{ref}}=11,150$), Evidently AI automatically selects **Wasserstein distance (normed)** rather than Kolmogorov-Smirnov p-values. The scores below represent normalized distance metrics where **Score > Threshold (0.05)** triggers drift (they are NOT p-values).
> - All 12 monitored features are deterministic mathematical transformations of the **same univariate demand series**. Concurrent drift across all 12 features reflects **a single underlying seasonal demand surge** (Autumn volume expansion), rather than 12 independent evidence signals of model failure.

| Feature Name | Feature Type | Statistical Test Method | Distance Score | Threshold | Drift Detected | Mean Shift (%) |
|:---|:---:|:---:|---:|---:|:---:|---:|
| `Demand` | Target | Wasserstein distance (normed) | 0.057307 | 0.05 | YES | +15.1% |
| `lag_7` | Lag | Wasserstein distance (normed) | 0.062505 | 0.05 | YES | +14.9% |
| `lag_1` | Lag | Wasserstein distance (normed) | 0.063425 | 0.05 | YES | +16.5% |
| `lag_28` | Lag | Wasserstein distance (normed) | 0.077078 | 0.05 | YES | +19.8% |
| `lag_21` | Lag | Wasserstein distance (normed) | 0.081609 | 0.05 | YES | +21.3% |
| `lag_14` | Lag | Wasserstein distance (normed) | 0.082972 | 0.05 | YES | +21.3% |
| `rolling_std_7` | Rolling Stat | Wasserstein distance (normed) | 0.086510 | 0.05 | YES | +10.0% |
| `rolling_std_14` | Rolling Stat | Wasserstein distance (normed) | 0.119264 | 0.05 | YES | +12.6% |
| `rolling_std_28` | Rolling Stat | Wasserstein distance (normed) | 0.135407 | 0.05 | YES | +11.4% |
| `rolling_mean_7` | Rolling Stat | Wasserstein distance (normed) | 0.153709 | 0.05 | YES | +17.2% |
| `rolling_mean_14` | Rolling Stat | Wasserstein distance (normed) | 0.213242 | 0.05 | YES | +20.3% |
| `rolling_mean_28` | Rolling Stat | Wasserstein distance (normed) | 0.263024 | 0.05 | YES | +21.1% |

---

## 3. Prediction Drift vs Actual Forecast Performance

### A. Prediction-Distribution Monitoring
- **Metric Monitored:** `Prophet_Predicted_Demand`
- **Test:** Kolmogorov-Smirnov 2-sample test (statistic = 0.0593, p-value = 0.010164)
- **Reference Mean:** 70.76 units
- **Current Mean:** 72.98 units (+3.1% shift)
- **Status:** `DRIFT_DETECTED`

### B. Actual-Versus-Forecast Accuracy (Realized Monitoring Windows)

#### 1. Full 30-Day Window 3 Horizon (`2011-10-11` to `2011-11-09`):
- **Sample Count:** 1,500 SKU-day predictions
- **Actual Realized Demand:** 92,569.0 units
- **Predicted Forecast Demand:** 98,045.03 units
- **30-Day Aggregate Volume Error:** **5.92%** (Warning: 15.0%, Critical: 20.0%)
- **Signed Portfolio Bias:** **+5.92%** (Warning: +/- 10.0%)
- **Tracking Signal:** 85.16 | **MAE:** 64.30 units | **RMSE:** 181.65 units
- **Status Classification:** `PERFORMANCE_HEALTHY`

#### 2. Strict 28-Day 4-Week Horizon (`2011-10-11` to `2011-11-07`):
- **Sample Count:** 1,400 SKU-day predictions
- **Actual Realized Demand:** 82,830.0 units
- **Predicted Forecast Demand:** 89,801.97 units
- **28-Day Aggregate Volume Error:** **8.42%** (Signed Bias: **+8.42%**)
- **Official Mean Weekly Portfolio MAPE:** **14.81%** (Week 1: 24.93%, Week 2: 27.03%, Week 3: 5.40%, Week 4: 1.86%)

---

## 4. Leakage Safeguards & Governance

1. **Strict Chronological Separation:** Reference baseline strictly precedes Current monitoring (`2011-08-11 < 2011-08-12`).
2. **Holdout Quarantine Hardening:** Fail-safe quarantine checks are enforced by default (`enforce_holdout_quarantine=True`). Any request extending beyond `2011-11-09` raises an exception prior to loading or slicing data.
3. **Production Model Unchanged:** Production baseline in `src/` and dashboard scorecard in `app.py` remain frozen at **14.81%**.
4. **Reproducibility:** Updated monitoring artifacts exported to:
   - Report: `outputs/forecasting/phase7/phase7_drift_report.html`
   - Metadata: `outputs/forecasting/phase7/monitoring_metadata.json`
   - Data Quality: `outputs/forecasting/phase7/data_quality_summary.json`
   - Prediction & Performance: `outputs/forecasting/phase7/prediction_and_performance_drift.json`
