import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from optitest.core.models import TelemetrySample
from optitest.ml.dataset import (
    FEATURE_COLUMNS,
    LABEL_MAP,
    REVERSE_LABEL_MAP,
    extract_features_from_history,
)

class OpticalAnomalyDetector:
    """
    Dual-engine AI detector:
    1. Unsupervised IsolationForest for general anomaly score [0.0 - 1.0].
    2. Supervised RandomForestClassifier for multi-class root-cause classification.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=4,
            random_state=random_state,
            n_jobs=-1
        )
        self.iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.25,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_trained = False

    def train(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Train both models on dataset DataFrame."""
        X = dataset[FEATURE_COLUMNS].fillna(0.0)
        y = dataset["label"].values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train Random Forest classifier
        self.classifier.fit(X_scaled, y)

        # Train Isolation Forest (trained on normal samples + full spectrum)
        self.iso_forest.fit(X_scaled)

        self.is_trained = True

        acc = self.classifier.score(X_scaled, y)
        importances = dict(zip(FEATURE_COLUMNS, self.classifier.feature_importances_))

        return {
            "training_samples": len(dataset),
            "train_accuracy": float(acc),
            "feature_importances": {k: float(v) for k, v in sorted(importances.items(), key=lambda item: item[1], reverse=True)}
        }

    def predict_history(self, history: List[TelemetrySample]) -> pd.DataFrame:
        """
        Process a full telemetry time series and return a DataFrame with:
        - raw telemetry
        - anomaly_score (0.0 to 1.0)
        - is_anomaly (bool)
        - predicted_category (string label)
        - confidence (probability of predicted category)
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train() or load() first.")

        df = extract_features_from_history(history)
        X = df[FEATURE_COLUMNS].fillna(0.0)
        X_scaled = self.scaler.transform(X)

        # Isolation Forest scoring: raw decision function maps normal > 0, outlier < 0
        raw_iso_scores = self.iso_forest.decision_function(X_scaled)
        # Normalize into [0, 1] anomaly score where 1 is highest anomaly
        # Typically raw scores range between -0.4 and +0.3
        anomaly_scores = 1.0 / (1.0 + np.exp(raw_iso_scores * 5.0))
        is_anomaly = anomaly_scores >= 0.55

        # Classifier predictions
        pred_labels = self.classifier.predict(X_scaled)
        pred_probs = self.classifier.predict_proba(X_scaled)
        confidences = np.max(pred_probs, axis=1)

        result_df = df.copy()
        result_df["anomaly_score"] = anomaly_scores
        result_df["is_anomaly"] = is_anomaly
        result_df["predicted_label_id"] = pred_labels
        result_df["predicted_category"] = [REVERSE_LABEL_MAP.get(l, "UNKNOWN") for l in pred_labels]
        result_df["confidence"] = confidences

        # Class probabilities columns
        for cat_name, cat_id in LABEL_MAP.items():
            if cat_id < pred_probs.shape[1]:
                result_df[f"prob_{cat_name}"] = pred_probs[:, cat_id]

        return result_df

    def get_feature_importances(self) -> Dict[str, float]:
        if not self.is_trained:
            return {}
        importances = dict(zip(FEATURE_COLUMNS, self.classifier.feature_importances_))
        return {k: float(v) for k, v in sorted(importances.items(), key=lambda item: item[1], reverse=True)}

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        bundle = {
            "scaler": self.scaler,
            "classifier": self.classifier,
            "iso_forest": self.iso_forest,
            "is_trained": self.is_trained,
            "random_state": self.random_state
        }
        joblib.dump(bundle, filepath)

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
        bundle = joblib.load(filepath)
        self.scaler = bundle["scaler"]
        self.classifier = bundle["classifier"]
        self.iso_forest = bundle["iso_forest"]
        self.is_trained = bundle["is_trained"]
        self.random_state = bundle.get("random_state", 42)
