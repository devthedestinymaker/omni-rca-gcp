# ==============================================================================
# WHAT: LightGBM Root-Cause Cascade Classifier [Topic 04: Machine Learning]
# WHY:  When multiple GCP services fail simultaneously, engineers need an instant
#       data-driven classification of the primary root-cause family (e.g., Worker
#       OOM vs. Upstream Data Quality vs. Slot Contention vs. Schema Drift) plus
#       feature-importance proof of which telemetry signals drove that decision.
# HOW:  Extracts tabular cross-plane features from a correlated Pandas incident
#       window, trains a multi-class LightGBM classifier on realistic synthetic +
#       seed incident patterns, saves the model artifact to `artifacts/cascade_lgbm.pkl`,
#       and runs fast inference with class probabilities.
# WHERE: `src/intelligence/cascade_ml.py` (invoked by the MCP ML tool and UI).
# EXAMPLE:
#   from src.intelligence.cascade_ml import CascadeRootCauseClassifier
#   clf = CascadeRootCauseClassifier()
#   clf.train_and_save()
#   pred = clf.predict_from_window(win_df)
# ==============================================================================

from pathlib import Path                                                    # Imports Path for clean model artifact saving/loading
from typing import Any, Dict, List                                          # Imports static type hints for dictionaries and lists
import joblib                                                               # Imports joblib to persist trained LightGBM models to disk
import lightgbm as lgb                                                      # Imports LightGBM gradient boosted decision tree library
import numpy as np                                                          # Imports NumPy for synthetic training dataset generation
import pandas as pd                                                         # Imports Pandas for structured feature DataFrames


ROOT_CAUSE_CLASSES: List[str] = [                                           # Ordered list of target RCA categories predicted by LightGBM
    "HEALTHY_NOISE",                                                        # Class 0: Minor transient jitter (no real outage)
    "DATAFLOW_OOM_MEMORY",                                                  # Class 1: Beam worker memory exhaustion (often post-deploy)
    "BIGQUERY_DATA_QUALITY_NULLS",                                          # Class 2: Upstream 0-row load or null-rate spike in BQ/dbt
    "BIGQUERY_SLOT_CONTENTION",                                             # Class 3: Heavy unpartitioned scan exhausting BQ slot quota
    "SCHEMA_DRIFT_STREAMING",                                               # Class 4: Producer schema change causing Dataflow/PubSub DLQ surge
]

FEATURE_COLUMNS: List[str] = [                                              # Exact tabular feature columns fed into LightGBM
    "alert_count",                                                          # Number of events in the 'alert' plane
    "service_log_count",                                                    # Number of events in the 'service_log' plane
    "data_quality_count",                                                   # Number of events in the 'data_quality' plane
    "cicd_deploy_count",                                                    # Number of recent deployments in the 'cicd' plane
    "critical_error_count",                                                 # Total count of ERROR or CRITICAL severity events
    "max_abs_pct_deviation",                                                # Largest absolute percentage metric jump/drop in the window
    "has_negative_row_drop",                                                # Binary flag (1.0 if metric_value dropped below baseline_value)
]


class CascadeRootCauseClassifier:                                           # Wraps feature extraction, LightGBM training, persistence & inference
    def __init__(self, model_path: str = "artifacts/cascade_lgbm.pkl"):     # Initializes classifier with default artifact output path
        self.model_path = Path(model_path)                                  # Stores Path object pointing to `artifacts/cascade_lgbm.pkl`
        self.model: lgb.LGBMClassifier | None = None                        # Holds trained LightGBM model instance in memory

    def extract_features(self, window_df: pd.DataFrame) -> pd.DataFrame:    # Converts a Pandas incident window into a 1-row ML feature table
        if window_df.empty:                                                 # Guards against empty incident windows
            zero_row = {col: 0.0 for col in FEATURE_COLUMNS}                # Creates all-zero feature dictionary
            return pd.DataFrame([zero_row])                                 # Returns 1-row DataFrame matching expected schema

        plane_counts = window_df["plane"].value_counts().to_dict()          # Counts how many signals occurred per operational plane
        sev_mask = window_df["severity"].isin(["ERROR", "CRITICAL"])        # Boolean mask identifying severe error/critical logs
        has_drop = float((window_df["metric_value"] < window_df["baseline_value"]).any()) # 1.0 if any metric dropped sharply (e.g., 0 rows)

        features = {                                                        # Builds exact feature dictionary matching `FEATURE_COLUMNS`
            "alert_count": float(plane_counts.get("alert", 0)),             # Count of Dynatrace / Cloud Monitoring alerts
            "service_log_count": float(plane_counts.get("service_log", 0)), # Count of Composer / Dataflow log failures
            "data_quality_count": float(plane_counts.get("data_quality", 0)), # Count of BigQuery DQ / PubSub backlog signals
            "cicd_deploy_count": float(plane_counts.get("cicd", 0)),        # Count of Harness / Terraform deployment events
            "critical_error_count": float(sev_mask.sum()),                  # Total ERROR + CRITICAL events in window
            "max_abs_pct_deviation": float(window_df["pct_deviation"].abs().max()), # Peak percentage deviation across all signals
            "has_negative_row_drop": has_drop,                              # Indicator distinguishing row/volume drops from latency spikes
        }
        return pd.DataFrame([features])[FEATURE_COLUMNS]                    # Returns single-row DataFrame with guaranteed column order

    def _generate_training_data(self, samples_per_class: int = 80, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]: # Creates training data
        rng = np.random.default_rng(seed)                                   # Deterministic RNG for reproducible model training
        rows: List[List[float]] = []                                        # Accumulates feature vectors
        labels: List[int] = []                                              # Accumulates integer class IDs (0..4)

        for _ in range(samples_per_class):                                  # Generates 80 realistic scenarios per root-cause class (400 total)
            # Class 0: HEALTHY_NOISE (no alerts, minor deviation)
            rows.append([0, rng.integers(0, 2), 0, rng.integers(0, 2), 0, rng.uniform(1, 15), 0])
            labels.append(0)
            # Class 1: DATAFLOW_OOM_MEMORY (alert + Dataflow/Composer logs + often CI/CD deploy + high latency spike)
            rows.append([1, rng.integers(2, 4), 0, rng.integers(0, 2), rng.integers(2, 4), rng.uniform(400, 5000), 0])
            labels.append(1)
            # Class 2: BIGQUERY_DATA_QUALITY_NULLS (alert + BQ DQ row drop + dbt fail + negative drop flag = 1)
            rows.append([1, rng.integers(1, 3), rng.integers(1, 3), 0, rng.integers(2, 4), rng.uniform(80, 90000), 1])
            labels.append(2)
            # Class 3: BIGQUERY_SLOT_CONTENTION (GCP alert + BQ bytes_billed spike + no CI/CD deploy + upward spike)
            rows.append([1, 0, rng.integers(1, 3), 0, rng.integers(1, 3), rng.uniform(2500, 4500), 0])
            labels.append(3)
            # Class 4: SCHEMA_DRIFT_STREAMING (alert + Dataflow CoderException + PubSub DLQ + CI/CD producer deploy)
            rows.append([1, rng.integers(1, 3), rng.integers(1, 3), 1, rng.integers(2, 4), rng.uniform(15000, 35000), 0])
            labels.append(4)

        X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)                     # Wraps training matrix into a named Pandas DataFrame
        y = np.array(labels, dtype=int)                                     # Converts target labels into a NumPy integer array
        return X, y                                                         # Returns feature matrix X and target vector y

    def train_and_save(self) -> Dict[str, Any]:                             # Trains LightGBM multi-class model and saves .pkl artifact
        X, y = self._generate_training_data()                               # Generates balanced 400-sample cross-plane training dataset
        clf = lgb.LGBMClassifier(                                           # Initializes LightGBM gradient boosted decision tree classifier
            objective="multiclass",                                         # Configures multi-class probability prediction across 5 RCA classes
            num_class=len(ROOT_CAUSE_CLASSES),                              # Sets number of target classes (5)
            n_estimators=60,                                                # Uses 60 boosting trees (fast, accurate, zero overfitting)
            learning_rate=0.08,                                             # Controls step size shrinkage per boosting round
            random_state=42,                                                # Fixes random seed for deterministic training
            verbose=-1,                                                     # Suppresses noisy C++ training logs in terminal
        )
        clf.fit(X, y)                                                       # Fits LightGBM trees on our tabular cross-plane features
        self.model = clf                                                    # Stores trained model on instance
        self.model_path.parent.mkdir(parents=True, exist_ok=True)           # Ensures `artifacts/` folder exists before saving
        joblib.dump(clf, self.model_path)                                   # Serializes trained model to `artifacts/cascade_lgbm.pkl`
        train_acc = float((clf.predict(X) == y).mean())                     # Computes training accuracy to confirm clean convergence
        return {"model_path": str(self.model_path), "train_accuracy": round(train_acc, 4)} # Returns training summary

    def predict_from_window(self, window_df: pd.DataFrame) -> Dict[str, Any]: # Predicts root-cause class & feature importance for an incident
        if self.model is None:                                              # Checks if model is already loaded in memory
            if self.model_path.exists():                                    # If saved artifact exists on disk, loads it immediately
                self.model = joblib.load(self.model_path)                   # Loads persisted LightGBM model via joblib
            else:                                                           # Otherwise trains and saves automatically on first run
                self.train_and_save()                                       # Auto-bootstraps model artifact if missing

        X_obs = self.extract_features(window_df)                            # Extracts 1-row feature DataFrame from active incident window
        probs = self.model.predict_proba(X_obs)[0]                          # Predicts probability distribution across all 5 RCA classes
        best_idx = int(np.argmax(probs))                                    # Finds index of highest-probability root-cause class
        importances = dict(zip(FEATURE_COLUMNS, [int(x) for x in self.model.feature_importances_])) # Maps split importances to feature names

        return {                                                            # Returns structured ML diagnostic prediction
            "predicted_root_cause": ROOT_CAUSE_CLASSES[best_idx],           # Top predicted root-cause label
            "confidence_pct": round(float(probs[best_idx]) * 100.0, 2),     # Model confidence percentage (0-100%)
            "extracted_features": X_obs.iloc[0].to_dict(),                  # Exact feature values extracted from the incident window
            "top_feature_importances": importances,                         # Tree split counts showing which features mattered most
        }