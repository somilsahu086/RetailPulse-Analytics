"""
RetailPulse — Demand Forecasting Module
Model Metadata & Governance Registry

Generates structured, reproducible metadata for demand forecasting models:
- Selected model architecture & configuration
- Tuned hyperparameters (Optuna search results)
- Component ensemble weights
- Internal validation metrics (2011-10-11 to 2011-11-09)
- Out-of-sample holdout metrics (2011-11-10 to 2011-12-09)
- Chronological time periods (train, validation, holdout)
- Random seeds and anti-leakage guarantees
- Comprehensive software package version manifest (PyTorch, Prophet, Optuna, MLflow, etc.)
"""

import sys
import json
import platform
from datetime import datetime
from typing import Any, Dict, Optional, Union
import numpy as np


def get_environment_package_versions() -> Dict[str, str]:
    """
    Collect versions of core machine learning, time-series, and MLOps libraries.
    """
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "os": platform.system()
    }

    # Packages to probe
    pkgs = [
        ("torch", "PyTorch"),
        ("prophet", "Prophet"),
        ("optuna", "Optuna"),
        ("mlflow", "MLflow"),
        ("lightgbm", "LightGBM"),
        ("sklearn", "scikit-learn"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("pyarrow", "pyarrow")
    ]

    for mod_name, disp_name in pkgs:
        try:
            mod = __import__(mod_name)
            versions[disp_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[disp_name] = "not_installed"

    return versions


def build_model_metadata(
    selected_model: str,
    tuned_parameters: Dict[str, Any],
    ensemble_weights: Union[Dict[str, float], float, str],
    validation_metrics: Dict[str, Any],
    holdout_metrics: Dict[str, Any],
    training_period: str = "2009-12-01 to 2011-10-10 (Pre-Train) / 2009-12-01 to 2011-11-09 (Full Train)",
    validation_period: str = "2011-10-11 to 2011-11-09 (30 Days)",
    holdout_period: str = "2011-11-10 to 2011-12-09 (30 Days)",
    forecast_horizon: int = 30,
    random_seed: int = 42,
    extra_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build a comprehensive, auditable model metadata dictionary.
    """
    metadata: Dict[str, Any] = {
        "metadata_version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "project": "RetailPulse Analytics",
        "module": "Demand Forecasting",
        "phase": "Phase 6 MLOps",
        "selected_model": selected_model,
        "forecast_horizon": forecast_horizon,
        "random_seed": random_seed,
        "temporal_timeline": {
            "training_period": training_period,
            "validation_period": validation_period,
            "holdout_period": holdout_period,
            "leakage_prevention": (
                "Hyperparameter tuning (Optuna) and ensemble weighting were performed "
                "strictly on the internal validation window (2011-10-11 to 2011-11-09). "
                "The final 30-day holdout test set (2011-11-10 to 2011-12-09) was strictly "
                "sequestered and untouched until post-tuning evaluation."
            )
        },
        "tuned_parameters": tuned_parameters,
        "ensemble_weights": ensemble_weights,
        "validation_metrics": validation_metrics,
        "holdout_metrics": holdout_metrics,
        "environment_packages": get_environment_package_versions()
    }

    if extra_metadata:
        metadata["extra_metadata"] = extra_metadata

    return metadata


def save_model_metadata(
    metadata: Dict[str, Any],
    output_filepath: str
) -> str:
    """
    Save model metadata dictionary to a JSON file.
    """
    # Custom serializer for numpy types
    def default_serializer(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=default_serializer)

    return output_filepath
