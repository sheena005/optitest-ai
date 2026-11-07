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
