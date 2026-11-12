import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupShuffleSplit

from optitest.core.models import TelemetrySample
from optitest.harness.rules import OpticalRuleEngine
from optitest.ml.detector import OpticalAnomalyDetector
from optitest.ml.dataset import FEATURE_COLUMNS, LABEL_MAP, REVERSE_LABEL_MAP

def group_train_test_split(
    dataset: pd.DataFrame,
    group_col: str = "run_id",
    test_size: float = 0.25,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a strict group-level train/test split based on run_id.
    Ensures that all telemetry steps belonging to a given simulation run
    are placed exclusively into either the train or test set, preventing
    temporal and scenario leakage.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(dataset, groups=dataset[group_col]))
    train_df = dataset.iloc[train_idx].reset_index(drop=True)
    test_df = dataset.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df

def rule_classify_sample(row: pd.Series) -> int:
    """
    Deterministic rule-based baseline classifier using static optical thresholds.
    Mimics standard threshold-alarm routing without machine learning.
    """
    rx_power = row["rx_power_dbm"]
    bias_curr = row["laser_bias_current_ma"]
    temp = row["laser_temperature_c"]
    osnr = row["osnr_db"]
    rolling_std = row.get("rx_power_rolling_std", 0.0)
    rx_delta = row.get("rx_power_delta", 0.0)

    if rx_power <= -28.0 or row.get("q_factor", 10.0) <= 0.0:
        return LABEL_MAP["FIBER_CUT"]
    if bias_curr >= 55.0 or temp >= 50.0:
        return LABEL_MAP["LASER_AGING"]
    if rolling_std >= 0.8 or osnr < 20.0:
        return LABEL_MAP["OPTICAL_NOISE"]
    if rx_power <= -22.0:
        return LABEL_MAP["EXCESS_ATTENUATION"]
    if rx_delta <= -4.0:
        return LABEL_MAP["DIRTY_CONNECTOR"]
    return LABEL_MAP["NORMAL"]

def evaluate_baselines(
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark comparison against:
    1. DummyClassifier (majority class / most frequent)
    2. Deterministic Rule-Based static threshold baseline
    """
    y_train = train_dataset["label"].values
    y_test = test_dataset["label"].values

    # 1. Dummy Classifier
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(train_dataset[FEATURE_COLUMNS], y_train)
    dummy_pred = dummy.predict(test_dataset[FEATURE_COLUMNS])

    dummy_acc = accuracy_score(y_test, dummy_pred)
    dummy_f1 = f1_score(y_test, dummy_pred, average="macro", zero_division=0)

    # 2. Rule-Based Baseline
    rule_pred = np.array([rule_classify_sample(row) for _, row in test_dataset.iterrows()])
    rule_acc = accuracy_score(y_test, rule_pred)
    rule_f1 = f1_score(y_test, rule_pred, average="macro", zero_division=0)

    return {
        "dummy_majority": {
            "accuracy": float(dummy_acc),
            "macro_f1": float(dummy_f1)
        },
        "rule_based_baseline": {
            "accuracy": float(rule_acc),
            "macro_f1": float(rule_f1)
        }
    }

def evaluate_models_on_test_data(
    detector: OpticalAnomalyDetector,
    test_dataset: pd.DataFrame
) -> Dict[str, Any]:
    """
    Evaluates ML classifier metrics on hold-out dataset.
    """
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

def evaluate_noise_robustness(
    detector: OpticalAnomalyDetector,
    test_dataset: pd.DataFrame,
    noise_sigmas: Optional[List[float]] = None,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Evaluates classifier resilience against sensor telemetry noise jitter.
    Applies Gaussian perturbations with standard deviation sigma (in dB/mA/C)
    to physical measurements.
    """
    if noise_sigmas is None:
        noise_sigmas = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]

    np.random.seed(seed)
    results = []
    y_test = test_dataset["label"].values

    for sigma in noise_sigmas:
        X_perturbed = test_dataset[FEATURE_COLUMNS].copy().fillna(0.0)
        if sigma > 0:
            for col in ["rx_power_dbm", "tx_power_dbm", "osnr_db"]:
                if col in X_perturbed.columns:
                    X_perturbed[col] += np.random.normal(0.0, sigma, size=len(X_perturbed))
            for col in ["laser_bias_current_ma", "laser_temperature_c"]:
                if col in X_perturbed.columns:
                    X_perturbed[col] += np.random.normal(0.0, sigma * 2.0, size=len(X_perturbed))

        X_scaled = detector.scaler.transform(X_perturbed)
        y_pred = detector.classifier.predict(X_scaled)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        results.append({
            "noise_sigma_db": float(sigma),
            "accuracy": float(acc),
            "macro_f1": float(f1),
            "degradation_pct": float(max(0.0, (1.0 - acc / (results[0]["accuracy"] if results else 1.0)) * 100))
        })

    return results

def evaluate_precursor_lead_times(
    detector: OpticalAnomalyDetector,
    rule_engine: OpticalRuleEngine
) -> List[Dict[str, Any]]:
    """
    Empirically measure precursor detection lead time across standard scenarios.
    Returns table of scenario, ML trigger step, rule trigger step, and lead time.
    """
    from optitest.faults.scenarios import (
        scenario_excessive_attenuation,
        scenario_laser_aging,
        scenario_dirty_connector,
        scenario_noise_burst
    )
    from optitest.harness.runner import TestHarnessRunner

    runner = TestHarnessRunner(rule_engine=rule_engine)
    scenarios = [
        scenario_laser_aging(),
        scenario_excessive_attenuation(),
        scenario_dirty_connector(),
        scenario_noise_burst()
    ]

    lead_time_records = []
    for sc in scenarios:
        res_obj = runner.run_scenario(sc)
        fault_start_step = int(sc.fault_start_s / sc.dt_s)
        res = compare_rule_vs_ml_on_history(res_obj.history, detector, rule_engine, fault_start_step=fault_start_step)
        lead_time_records.append({
            "scenario": sc.name,
            "fault_start_step": fault_start_step,
            "rule_trigger_step": res["rule_trigger_step"],
            "ml_trigger_step": res["ml_trigger_step"],
            "lead_time_steps": res["lead_time_steps"],
            "lead_time_seconds": round(res["lead_time_steps"] * sc.dt_s, 2) if res["lead_time_steps"] else 0.0,
            "agreement_rate": round(res["agreement_rate"] * 100, 1)
        })

    return lead_time_records

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
