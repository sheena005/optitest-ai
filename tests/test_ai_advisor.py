import pytest
from optitest.harness.runner import TestHarnessRunner
from optitest.faults.scenarios import scenario_fiber_cut, scenario_normal_link, scenario_laser_aging
from optitest.ai_advisor.diagnostic import OpticalDiagnosticAdvisor

def test_diagnostic_advisor_fiber_cut():
    runner = TestHarnessRunner()
    res = runner.run_scenario(scenario_fiber_cut())

    advisor = OpticalDiagnosticAdvisor()
    diag = advisor.diagnose(res)

    assert diag.scenario_name == "Complete Fiber Cut"
    assert diag.incident_severity == "P1_CRITICAL"
    assert "optical power extinction" in diag.root_cause_summary.lower() or "break" in diag.root_cause_summary.lower() or "severance" in diag.physical_root_cause.lower()
    assert len(diag.remediation_steps) >= 2
    assert "OTDR" in diag.remediation_steps[0]
    assert "dBm" in diag.markdown_report

def test_diagnostic_advisor_normal_link():
    runner = TestHarnessRunner()
    res = runner.run_scenario(scenario_normal_link())

    advisor = OpticalDiagnosticAdvisor()
    diag = advisor.diagnose(res)

    assert diag.incident_severity == "NOMINAL"
    assert "tolerances" in diag.root_cause_summary.lower() or "healthy" in diag.physical_root_cause.lower()

def test_grounding_prevents_hallucinated_status():
    runner = TestHarnessRunner()
    res = runner.run_scenario(scenario_laser_aging())

    advisor = OpticalDiagnosticAdvisor()
    diag = advisor.diagnose(res)

    # Status must match test result exactly
    assert diag.status == res.status
    # Max laser current in telemetry findings must match real measurements
    measured_max_bias = max(s.laser_bias_current_ma for s in res.history)
    assert diag.telemetry_findings["max_bias_current_ma"] == pytest.approx(measured_max_bias, abs=0.2)
