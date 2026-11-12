#!/usr/bin/env python3
"""
Model training, group-level evaluation, baselines benchmarking,
OOD stress testing, and precursor lead-time evaluation for OptiTest AI.
"""

import os
import sys
import json

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from optitest.ml.dataset import generate_telemetry_dataset, generate_ood_dataset
from optitest.ml.detector import OpticalAnomalyDetector
from optitest.harness.rules import OpticalRuleEngine
from optitest.ml.evaluate import (
    group_train_test_split,
    evaluate_baselines,
    evaluate_models_on_test_data,
    evaluate_noise_robustness,
    evaluate_precursor_lead_times
)

def main():
    print("=" * 65)
    print("OptiTest AI: Machine Learning Rigor & Robustness Benchmark")
    print("=" * 65)

    print("\n[1/6] Synthesizing comprehensive multi-run telemetry dataset...")
    full_dataset = generate_telemetry_dataset(num_runs_per_fault=24, seed=42)
    print(f"Total dataset: {len(full_dataset)} steps across {full_dataset['run_id'].nunique()} simulation runs.")

    print("\n[2/6] Executing Run-Level Group Split (GroupShuffleSplit on run_id)...")
    train_df, test_df = group_train_test_split(full_dataset, group_col="run_id", test_size=0.25, random_state=42)
    print(f"Train set: {len(train_df)} steps ({train_df['run_id'].nunique()} runs, {train_df['run_id'].nunique() / full_dataset['run_id'].nunique() * 100:.1f}%)")
    print(f"Test set:  {len(test_df)} steps ({test_df['run_id'].nunique()} runs, {test_df['run_id'].nunique() / full_dataset['run_id'].nunique() * 100:.1f}%)")

    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    train_df.to_csv(os.path.join(data_dir, "train_dataset.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "test_dataset.csv"), index=False)

    print("\n[3/6] Training Dual-Engine Anomaly Detector (RandomForest + IsolationForest)...")
    detector = OpticalAnomalyDetector(random_state=42)
    train_res = detector.train(train_df)
    print(f"Training Accuracy: {train_res['train_accuracy'] * 100:.2f}%")
    print("Top Feature Importances:")
    for feat, imp in list(train_res['feature_importances'].items())[:5]:
        print(f"  - {feat:25s}: {imp * 100:.2f}%")

    print("\n[4/6] Evaluating Hold-Out Group Test Set vs Baselines...")
    eval_res = evaluate_models_on_test_data(detector, test_df)
    baselines = evaluate_baselines(train_df, test_df)

    print("\n-------------------------------------------------------------")
    print(f"{'Model / Baseline':<28} | {'Accuracy':<12} | {'Macro F1':<12}")
    print("-------------------------------------------------------------")
    print(f"{'Dummy (Majority Class)':<28} | {baselines['dummy_majority']['accuracy'] * 100:>10.2f}% | {baselines['dummy_majority']['macro_f1'] * 100:>10.2f}%")
    print(f"{'Rule-Based Static Baseline':<28} | {baselines['rule_based_baseline']['accuracy'] * 100:>10.2f}% | {baselines['rule_based_baseline']['macro_f1'] * 100:>10.2f}%")
    print(f"{'OptiTest ML Classifier':<28} | {eval_res['accuracy'] * 100:>10.2f}% | {eval_res['macro_f1'] * 100:>10.2f}%")
    print("-------------------------------------------------------------")

    print("\n[5/6] Stress Testing: Out-Of-Distribution (OOD) & Sensor Noise Jitter...")
    ood_df = generate_ood_dataset(num_runs_per_fault=6, seed=777)
    ood_res = evaluate_models_on_test_data(detector, ood_df)
    print(f"OOD Set (105-140 km spans, extreme power, 3x noise):")
    print(f"  -> OOD Accuracy: {ood_res['accuracy'] * 100:.2f}% (Macro F1: {ood_res['macro_f1'] * 100:.2f}%)")

    noise_sweep = evaluate_noise_robustness(detector, test_df)
    print("\nNoise Jitter Resilience Sweep:")
    for step in noise_sweep:
        print(f"  - Sigma = {step['noise_sigma_db']:.2f} dB: Acc = {step['accuracy']*100:.2f}%, F1 = {step['macro_f1']*100:.2f}%, Degradation = {step['degradation_pct']:.2f}%")

    print("\n[6/6] Quantifying Precursor Detection Lead Times...")
    rule_engine = OpticalRuleEngine()
    lead_times = evaluate_precursor_lead_times(detector, rule_engine)
    print("Scenario Lead-Time Analysis (ML warning before ITU-T alarm):")
    for lt in lead_times:
        print(f"  - {lt['scenario']:<32}: Lead Time = {lt['lead_time_steps']} steps ({lt['lead_time_seconds']}s) | Agreement = {lt['agreement_rate']}%")

    # Save artifacts
    model_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "optical_detector.joblib")
    detector.save(model_path)
    print(f"\nSaved trained model artifact to: {model_path}")

    metrics_path = os.path.join(model_dir, "eval_metrics.json")
    metrics_data = {
        "group_split_evaluation": {
            "test_accuracy": eval_res["accuracy"],
            "test_macro_f1": eval_res["macro_f1"],
            "test_weighted_f1": eval_res["weighted_f1"],
            "confusion_matrix": eval_res["confusion_matrix"],
            "target_names": eval_res["target_names"]
        },
        "baselines_comparison": {
            "dummy_majority": baselines["dummy_majority"],
            "rule_based_baseline": baselines["rule_based_baseline"],
            "ml_classifier": {
                "accuracy": eval_res["accuracy"],
                "macro_f1": eval_res["macro_f1"]
            }
        },
        "out_of_distribution": {
            "test_accuracy": ood_res["accuracy"],
            "test_macro_f1": ood_res["macro_f1"],
            "description": "Evaluated on spans 105-140 km, extreme launch powers, 3x sensor jitter"
        },
        "noise_robustness_sweep": noise_sweep,
        "precursor_lead_times": lead_times,
        "feature_importances": train_res["feature_importances"]
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved comprehensive evaluation metrics to: {metrics_path}")

    print("\nEvaluation and model benchmarking completed successfully!")

if __name__ == "__main__":
    main()
