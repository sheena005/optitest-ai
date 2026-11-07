import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import classification_report, confusion_matrix
from optitest.core.models import TelemetrySample
from optitest.harness.rules import OpticalRuleEngine
from optitest.ml.detector import OpticalAnomalyDetector
from optitest.ml.dataset import LABEL_MAP, REVERSE_LABEL_MAP

def evaluate_models_on_test_data(
    detector: OpticalAnomalyDetector,
    test_dataset: pd.DataFrame
) -> Dict[str, Any]:
    """
    Evaluates ML classifier metrics on hold-out synthetic dataset.
    """
    from optitest.ml.dataset import FEATURE_COLUMNS
    X_test = test_dataset[FEATURE_COLUMNS].fillna(0.0)
    y_test = test_dataset["label"].values

    X_scaled = detector.scaler.transform(X_test)
    y_pred = detector.classifier.predict(X_scaled)

    labels = sorted(list(LABEL_MAP.values()))
    target_names = [REVERSE_LABEL_MAP[i] for i in labels]

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    return {
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "target_names": target_names
    }

def compare_rule_vs_ml_on_history(
    history: List[TelemetrySample],
    detector: OpticalAnomalyDetector,
    rule_engine: OpticalRuleEngine,
    fault_start_step: int = 20
) -> Dict[str, Any]:
    """
    Side-by-side comparison of Rule-Based testing vs ML Anomaly Detector on a time series.
    Analyzes:
    - First detection step (latency)
    - Agreement rate
    - Early warning precursors
    """
    ml_results = detector.predict_history(history)
    rule_engine.reset()

    rule_alarms = []
    rule_states = []

    for s in history:
        eval_res = rule_engine.evaluate_sample(s)
        rule_alarms.append(len(eval_res.active_alarms) > 0)
        rule_states.append(eval_res.state)

    ml_anomalies = ml_results["is_anomaly"].tolist()
    ml_categories = ml_results["predicted_category"].tolist()
    ml_scores = ml_results["anomaly_score"].tolist()

    # Calculate first trigger step after fault start
    rule_trigger_step = None
    for step in range(fault_start_step, len(rule_alarms)):
        if rule_alarms[step]:
            rule_trigger_step = step
            break

    ml_trigger_step = None
    for step in range(fault_start_step, len(ml_anomalies)):
        if ml_anomalies[step]:
            ml_trigger_step = step
            break

    # Early detection lead time
    lead_time_steps = 0
    if rule_trigger_step is not None and ml_trigger_step is not None:
        lead_time_steps = rule_trigger_step - ml_trigger_step

    # Agreement rate
    agreement_count = sum(1 for r, m in zip(rule_alarms, ml_anomalies) if r == m)
    agreement_rate = agreement_count / len(history) if history else 1.0

    return {
        "total_steps": len(history),
        "fault_start_step": fault_start_step,
        "rule_trigger_step": rule_trigger_step,
        "ml_trigger_step": ml_trigger_step,
        "lead_time_steps": lead_time_steps,
        "agreement_rate": float(agreement_rate),
        "rule_alarms": rule_alarms,
        "ml_anomalies": ml_anomalies,
        "ml_categories": ml_categories,
        "ml_scores": ml_scores,
        "rule_states": rule_states
    }
