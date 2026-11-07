import pytest
from optitest.core.models import TelemetrySample
from optitest.harness.rules import OpticalRuleEngine, RuleThresholds
from optitest.harness.runner import TestHarnessRunner
from optitest.faults.scenarios import get_all_scenarios

def test_rule_threshold_detection():
    engine = OpticalRuleEngine()

    # Nominal sample
    s_nom = TelemetrySample(rx_power_dbm=-10.0, osnr_db=30.0, ber=1e-12, laser_bias_current_ma=45.0)
    res_nom = engine.evaluate_sample(s_nom)
    assert res_nom.state == "HEALTHY"
    assert len(res_nom.active_alarms) == 0

    # Under-power sample
    s_low = TelemetrySample(rx_power_dbm=-24.0, osnr_db=25.0, ber=1e-8, laser_bias_current_ma=45.0)
    res_low = engine.evaluate_sample(s_low)
    assert "ALARM_DEGRADED_POWER" in res_low.active_alarms

    # LOS sample
    s_los = TelemetrySample(rx_power_dbm=-30.0, osnr_db=0.0, ber=0.5, laser_bias_current_ma=45.0)
    res_los = engine.evaluate_sample(s_los)
    assert "ALARM_LOS" in res_los.active_alarms

def test_debounce_filtering():
    thresholds = RuleThresholds(debounce_steps=3, clear_steps=2)
    engine = OpticalRuleEngine(thresholds)

    s_healthy = TelemetrySample(rx_power_dbm=-10.0, osnr_db=30.0, ber=1e-12)
    s_glitch = TelemetrySample(rx_power_dbm=-24.0, osnr_db=25.0, ber=1e-8)  # Single glitch

    # Step 1: Healthy
    res1 = engine.evaluate_sample(s_healthy)
    assert res1.state == "HEALTHY"

    # Step 2: Glitch 1
    res2 = engine.evaluate_sample(s_glitch)
    assert res2.state == "HEALTHY"  # Debounce prevents immediate state flip!

    # Step 3: Back to healthy
    res3 = engine.evaluate_sample(s_healthy)
    assert res3.state == "HEALTHY"

def test_all_scenarios_pass_runner():
    """Verify that all standard scenarios pass the automated test harness."""
    runner = TestHarnessRunner()
    scenarios = get_all_scenarios()
    results = runner.run_all(scenarios)

    for r in results:
        assert r.status == "PASS", f"Scenario {r.scenario_name} failed: {r.failure_reasons}"
