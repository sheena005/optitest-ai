import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from optitest.core.models import TelemetrySample
from optitest.core.simulator import OpticalLinkWrapper
from optitest.faults.scenarios import FaultScenario, get_all_scenarios
from optitest.harness.rules import OpticalRuleEngine, RuleEvaluation

@dataclass
class ScenarioTestResult:
    scenario_name: str
    description: str
    status: str  # "PASS" | "FAIL"
    expected_nominal: str
    observed_nominal: str
    expected_fault: str
    observed_fault: str
    history: List[TelemetrySample] = field(default_factory=list)
    evaluations: List[RuleEvaluation] = field(default_factory=list)
    alarms_triggered: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    failure_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "description": self.description,
            "status": self.status,
            "expected_nominal": self.expected_nominal,
            "observed_nominal": self.observed_nominal,
            "expected_fault": self.expected_fault,
            "observed_fault": self.observed_fault,
            "alarms_triggered": list(set(self.alarms_triggered)),
            "duration_ms": round(self.duration_ms, 2),
            "failure_reasons": self.failure_reasons,
            "sample_count": len(self.history),
        }

class TestHarnessRunner:
    """
    Automated test harness that executes fault injection scenarios against
    the optical link simulator and verifies deterministic behavior.
    """
    __test__ = False

    def __init__(self, rule_engine: Optional[OpticalRuleEngine] = None):
        self.rule_engine = rule_engine or OpticalRuleEngine()

    def run_scenario(self, scenario: FaultScenario) -> ScenarioTestResult:
        start_t = time.perf_counter()
        link = OpticalLinkWrapper(scenario.link_params)
        self.rule_engine.reset()

        history: List[TelemetrySample] = []
        evaluations: List[RuleEvaluation] = []
        all_alarms: List[str] = []

        nominal_states = []
        fault_states = []

        for step_idx in range(scenario.num_steps):
            t = step_idx * scenario.dt_s
            p = scenario.apply_perturbation(t) if scenario.apply_perturbation else {}

            sample = link.step(
                timestamp_s=t,
                injected_attenuation_db=p.get("injected_attenuation_db", 0.0),
                injected_noise_db=p.get("injected_noise_db", 0.0),
                bias_current_drift_ma=p.get("bias_current_drift_ma", 0.0),
                temp_drift_c=p.get("temp_drift_c", 0.0),
                force_los=p.get("force_los", False),
                fault_name=p.get("fault_name", "NONE")
            )
            history.append(sample)

            eval_res = self.rule_engine.evaluate_sample(sample)
            evaluations.append(eval_res)
            all_alarms.extend(eval_res.active_alarms)

            # Categorize state before/after fault trigger
            if t < scenario.fault_start_s:
                nominal_states.append(eval_res.state)
            else:
                # If fault has a duration and recovered
                if scenario.fault_duration_s > 0 and t >= (scenario.fault_start_s + scenario.fault_duration_s):
                    # In recovery window
                    pass
                else:
                    fault_states.append(eval_res.state)

        # Dominant nominal state
        observed_nominal = max(set(nominal_states), key=nominal_states.count) if nominal_states else "HEALTHY"
        # Dominant or worst fault state
        if fault_states:
            # Check if expected fault was reached
            if scenario.expected_fault_state in fault_states:
                observed_fault = scenario.expected_fault_state
            else:
                observed_fault = max(set(fault_states), key=fault_states.count)
        else:
            observed_fault = observed_nominal

        # Verification verdict
        failures = []
        if observed_nominal != scenario.expected_nominal_state:
            failures.append(f"Nominal state mismatch: expected {scenario.expected_nominal_state}, got {observed_nominal}")

        if observed_fault != scenario.expected_fault_state:
            failures.append(f"Fault state mismatch: expected {scenario.expected_fault_state}, got {observed_fault}")

        status = "PASS" if len(failures) == 0 else "FAIL"
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return ScenarioTestResult(
            scenario_name=scenario.name,
            description=scenario.description,
            status=status,
            expected_nominal=scenario.expected_nominal_state,
            observed_nominal=observed_nominal,
            expected_fault=scenario.expected_fault_state,
            observed_fault=observed_fault,
            history=history,
            evaluations=evaluations,
            alarms_triggered=sorted(list(set(all_alarms))),
            duration_ms=elapsed_ms,
            failure_reasons=failures,
        )

    def run_all(self, scenarios: Optional[List[FaultScenario]] = None) -> List[ScenarioTestResult]:
        scenarios = scenarios or get_all_scenarios()
        return [self.run_scenario(s) for s in scenarios]
