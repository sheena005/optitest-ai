import os
import pytest
from optitest.core.models import LinkParameters
from optitest.core.simulator import OpticalLinkWrapper
from optitest.ml.detector import OpticalAnomalyDetector
from optitest.ml.evaluate import compare_rule_vs_ml_on_history
from optitest.harness.rules import OpticalRuleEngine
from optitest.faults.scenarios import scenario_fiber_cut, scenario_normal_link, scenario_laser_aging

@pytest.fixture(scope="module")
def trained_detector():
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "optical_detector.joblib")
    assert os.path.exists(model_path), f"Pre-trained model not found at {model_path}"
    detector = OpticalAnomalyDetector()
    detector.load(model_path)
    return detector

def test_model_inference_normal_link(trained_detector):
    sc = scenario_normal_link()
    link = OpticalLinkWrapper(sc.link_params)
    history = [link.step(i * 0.1) for i in range(30)]

    df = trained_detector.predict_history(history)
    assert len(df) == 30
    assert "anomaly_score" in df.columns
    assert "predicted_category" in df.columns

    # For a completely normal link, predicted category should be NORMAL
    normal_pct = (df["predicted_category"] == "NORMAL").mean()
    assert normal_pct >= 0.90

def test_model_inference_fiber_cut(trained_detector):
    sc = scenario_fiber_cut()
    link = OpticalLinkWrapper(sc.link_params)
    history = []
    for i in range(40):
        t = i * 0.1
        history.append(link.step(t, **sc.apply_perturbation(t)))

    df = trained_detector.predict_history(history)
    # After step 20 (t=2.0s), fiber cut is active
    post_fault = df.iloc[22:]
    cut_detected = (post_fault["predicted_category"] == "FIBER_CUT").any()
    assert cut_detected
    assert post_fault["is_anomaly"].any()
    assert (post_fault["anomaly_score"] > 0.55).any()

def test_compare_rule_vs_ml(trained_detector):
    sc = scenario_laser_aging()
    link = OpticalLinkWrapper(sc.link_params)
    history = [link.step(i * 0.1, **sc.apply_perturbation(i * 0.1)) for i in range(50)]

    rule_engine = OpticalRuleEngine()
    comp = compare_rule_vs_ml_on_history(history, trained_detector, rule_engine, fault_start_step=20)

    assert "lead_time_steps" in comp
    assert "agreement_rate" in comp
    assert comp["agreement_rate"] > 0.5

def test_group_train_test_split_isolation():
    from optitest.ml.dataset import generate_telemetry_dataset
    from optitest.ml.evaluate import group_train_test_split

    df = generate_telemetry_dataset(num_runs_per_fault=2, seed=123)
    train_df, test_df = group_train_test_split(df, group_col="run_id", test_size=0.33, random_state=42)

    train_runs = set(train_df["run_id"].unique())
    test_runs = set(test_df["run_id"].unique())

    # Strict isolation: intersection of run_ids between train and test must be empty
    assert len(train_runs.intersection(test_runs)) == 0
    assert len(train_df) + len(test_df) == len(df)

def test_baselines_and_noise_robustness(trained_detector):
    from optitest.ml.dataset import generate_telemetry_dataset
    from optitest.ml.evaluate import evaluate_baselines, evaluate_noise_robustness

    df = generate_telemetry_dataset(num_runs_per_fault=2, seed=321)
    train_df = df.iloc[: len(df) // 2]
    test_df = df.iloc[len(df) // 2 :]

    baselines = evaluate_baselines(train_df, test_df)
    assert "dummy_majority" in baselines
    assert "rule_based_baseline" in baselines
    assert 0.0 <= baselines["dummy_majority"]["accuracy"] <= 1.0

    noise_res = evaluate_noise_robustness(trained_detector, test_df, noise_sigmas=[0.0, 0.1])
    assert len(noise_res) == 2
    assert noise_res[0]["accuracy"] >= 0.70

