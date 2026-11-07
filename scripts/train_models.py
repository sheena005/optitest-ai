#!/usr/bin/env python3
"""
Model training and evaluation script for OptiTest AI.
"""

import os
import sys
import json

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from optitest.ml.dataset import generate_telemetry_dataset
from optitest.ml.detector import OpticalAnomalyDetector
from optitest.ml.evaluate import evaluate_models_on_test_data

def main():
    print("=" * 60)
    print("OptiTest AI: Machine Learning Anomaly Detector Training")
    print("=" * 60)

    print("\n[1/4] Generating synthetic telemetry training dataset...")
    train_df = generate_telemetry_dataset(num_runs_per_fault=20, seed=42)
    print(f"Generated {len(train_df)} training samples across {train_df['run_id'].nunique()} link simulations.")

    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    dataset_path = os.path.join(data_dir, "training_dataset.csv")
    train_df.to_csv(dataset_path, index=False)
    print(f"Saved training dataset to: {dataset_path}")

    print("\n[2/4] Generating hold-out validation dataset...")
    test_df = generate_telemetry_dataset(num_runs_per_fault=8, seed=999)
    print(f"Generated {len(test_df)} validation samples.")

    print("\n[3/4] Training IsolationForest & RandomForestClassifier...")
    detector = OpticalAnomalyDetector(random_state=42)
    train_res = detector.train(train_df)
    print(f"Training Accuracy: {train_res['train_accuracy'] * 100:.2f}%")
    print("Top Feature Importances:")
    for feat, imp in list(train_res['feature_importances'].items())[:5]:
        print(f"  - {feat:25s}: {imp * 100:.2f}%")

    print("\n[4/4] Evaluating models on hold-out validation set...")
    eval_res = evaluate_models_on_test_data(detector, test_df)
    print(f"Hold-out Test Accuracy: {eval_res['accuracy'] * 100:.2f}%")
    print(f"Hold-out Macro F1:      {eval_res['macro_f1'] * 100:.2f}%")

    model_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "optical_detector.joblib")
    detector.save(model_path)
    print(f"\nSaved trained model artifact to: {model_path}")

    metrics_path = os.path.join(model_dir, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "accuracy": eval_res["accuracy"],
            "macro_f1": eval_res["macro_f1"],
            "target_names": eval_res["target_names"],
            "feature_importances": train_res["feature_importances"]
        }, f, indent=2)
    print(f"Saved evaluation metrics to: {metrics_path}")

    print("\nModel training and validation completed successfully!")

if __name__ == "__main__":
    main()
