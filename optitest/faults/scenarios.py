from dataclasses import dataclass, field
from typing import List, Callable, Optional, Dict, Any
from optitest.core.models import LinkParameters, TelemetrySample
from optitest.core.simulator import OpticalLinkWrapper

@dataclass
class FaultScenario:
    name: str
    description: str
    link_params: LinkParameters
    expected_nominal_state: str
    expected_fault_state: str
    num_steps: int = 100
    dt_s: float = 0.1
    fault_start_s: float = 2.0
    fault_duration_s: float = -1.0
    apply_perturbation: Optional[Callable[[float], Dict[str, Any]]] = None

def scenario_normal_link() -> FaultScenario:
    return FaultScenario(
        name="Normal Link Operation",
        description="Standard 40km SMF-28 optical link with 0 dBm Tx power under nominal operating conditions.",
        link_params=LinkParameters(tx_power_dbm=0.0, fiber_length_km=40.0, attenuation_db_per_km=0.20, other_loss_db=2.0),
        expected_nominal_state="HEALTHY",
        expected_fault_state="HEALTHY",
        num_steps=100,
        dt_s=0.1,
        fault_start_s=999.0,
        apply_perturbation=lambda t: {
            "injected_attenuation_db": 0.0,
            "injected_noise_db": 0.0,
            "bias_current_drift_ma": 0.0,
            "temp_drift_c": 0.0,
            "force_los": False,
            "fault_name": "NONE"
        }
    )

def scenario_excessive_attenuation() -> FaultScenario:
    def perturbation(t: float) -> Dict[str, Any]:
        if t < 2.0:
            return {"injected_attenuation_db": 0.0, "injected_noise_db": 0.0, "bias_current_drift_ma": 0.0, "temp_drift_c": 0.0, "force_los": False, "fault_name": "NONE"}
        added_att = min(15.0, (t - 2.0) * 2.0)
        return {
            "injected_attenuation_db": added_att,
            "injected_noise_db": 0.0,
            "bias_current_drift_ma": 0.0,
            "temp_drift_c": 0.0,
            "force_los": False,
            "fault_name": "ATTENUATION_RAMP"
        }

    return FaultScenario(
        name="Excessive Attenuation",
        description="Simulates macro-bending / optical cable compression causing progressive attenuation ramp up to +15 dB.",
        link_params=LinkParameters(tx_power_dbm=0.0, fiber_length_km=40.0, attenuation_db_per_km=0.20, other_loss_db=2.0),
        expected_nominal_state="HEALTHY",
        expected_fault_state="DEGRADED",
        num_steps=100,
        dt_s=0.1,
        fault_start_s=2.0,
        apply_perturbation=perturbation
    )

def scenario_fiber_cut() -> FaultScenario:
    def perturbation(t: float) -> Dict[str, Any]:
        if t >= 2.0:
            return {
                "injected_attenuation_db": 99.0,
                "injected_noise_db": 0.0,
                "bias_current_drift_ma": 0.0,
                "temp_drift_c": 0.0,
                "force_los": True,
                "fault_name": "FIBER_CUT"
            }
        return {"injected_attenuation_db": 0.0, "injected_noise_db": 0.0, "bias_current_drift_ma": 0.0, "temp_drift_c": 0.0, "force_los": False, "fault_name": "NONE"}

    return FaultScenario(
        name="Complete Fiber Cut",
        description="Instantaneous physical severance of the optical cable triggering immediate Loss of Signal (LOS).",
        link_params=LinkParameters(tx_power_dbm=0.0, fiber_length_km=40.0, attenuation_db_per_km=0.20, other_loss_db=2.0),
        expected_nominal_state="HEALTHY",
        expected_fault_state="LINK_DOWN",
        num_steps=100,
        dt_s=0.1,
        fault_start_s=2.0,
        apply_perturbation=perturbation
    )

def scenario_dirty_connector() -> FaultScenario:
    def perturbation(t: float) -> Dict[str, Any]:
        if t >= 2.0:
            return {
                "injected_attenuation_db": 13.0,
                "injected_noise_db": 2.0,
                "bias_current_drift_ma": 0.0,
                "temp_drift_c": 0.0,
                "force_los": False,
                "fault_name": "CONNECTOR_CONTAMINATION"
            }
        return {"injected_attenuation_db": 0.0, "injected_noise_db": 0.0, "bias_current_drift_ma": 0.0, "temp_drift_c": 0.0, "force_los": False, "fault_name": "NONE"}

    return FaultScenario(
        name="Connector Dirt Contamination",
        description="Dust or oil on optical ferrule endface introducing a sudden 13.0 dB insertion loss and return loss penalty.",
        link_params=LinkParameters(tx_power_dbm=0.0, fiber_length_km=40.0, attenuation_db_per_km=0.20, other_loss_db=2.0),
        expected_nominal_state="HEALTHY",
        expected_fault_state="DEGRADED",
        num_steps=100,
        dt_s=0.1,
        fault_start_s=2.0,
        apply_perturbation=perturbation
    )

def scenario_laser_aging() -> FaultScenario:
    def perturbation(t: float) -> Dict[str, Any]:
        if t >= 2.0:
            elapsed = t - 2.0
            drift_ma = min(30.0, elapsed * 4.0)
            temp_rise = min(18.0, elapsed * 2.5)
            return {
                "injected_attenuation_db": 0.5,
                "injected_noise_db": 0.0,
                "bias_current_drift_ma": drift_ma,
                "temp_drift_c": temp_rise,
                "force_los": False,
                "fault_name": "LASER_AGING"
            }
        return {"injected_attenuation_db": 0.0, "injected_noise_db": 0.0, "bias_current_drift_ma": 0.0, "temp_drift_c": 0.0, "force_los": False, "fault_name": "NONE"}

    return FaultScenario(
        name="Laser Diode Aging",
        description="Internal degradation of the DFB laser diode causing runaway bias current and thermal elevation.",
        link_params=LinkParameters(tx_power_dbm=0.0, fiber_length_km=40.0, attenuation_db_per_km=0.20, other_loss_db=2.0),
        expected_nominal_state="HEALTHY",
        expected_fault_state="DEGRADED",
        num_steps=100,
        dt_s=0.1,
        fault_start_s=2.0,
        apply_perturbation=perturbation
    )

def scenario_noise_burst() -> FaultScenario:
    def perturbation(t: float) -> Dict[str, Any]:
        if t >= 2.0:
            return {
                "injected_attenuation_db": 0.0,
                "injected_noise_db": 9.5,
                "bias_current_drift_ma": 0.0,
                "temp_drift_c": 0.0,
                "force_los": False,
                "fault_name": "NOISE_BURST"
            }
        return {"injected_attenuation_db": 0.0, "injected_noise_db": 0.0, "bias_current_drift_ma": 0.0, "temp_drift_c": 0.0, "force_los": False, "fault_name": "NONE"}

    return FaultScenario(
        name="Amplifier Optical Noise (ASE)",
        description="High spontaneous emission optical noise from an EDFA causing OSNR and BER degradation without optical power drop.",
        link_params=LinkParameters(tx_power_dbm=0.0, fiber_length_km=40.0, attenuation_db_per_km=0.20, other_loss_db=2.0),
        expected_nominal_state="HEALTHY",
        expected_fault_state="DEGRADED",
        num_steps=100,
        dt_s=0.1,
        fault_start_s=2.0,
        apply_perturbation=perturbation
    )

def scenario_fault_recovery() -> FaultScenario:
    def perturbation(t: float) -> Dict[str, Any]:
        if 2.0 <= t < 5.0:
            return {
                "injected_attenuation_db": 99.0,
                "injected_noise_db": 0.0,
                "bias_current_drift_ma": 0.0,
                "temp_drift_c": 0.0,
                "force_los": True,
                "fault_name": "TEMPORARY_CUT"
            }
        return {"injected_attenuation_db": 0.0, "injected_noise_db": 0.0, "bias_current_drift_ma": 0.0, "temp_drift_c": 0.0, "force_los": False, "fault_name": "NONE"}

    return FaultScenario(
        name="Fault Recovery Cycle",
        description="Simulates a temporary optical disruption from t=2s to t=5s followed by full physical link restoration.",
        link_params=LinkParameters(tx_power_dbm=0.0, fiber_length_km=40.0, attenuation_db_per_km=0.20, other_loss_db=2.0),
        expected_nominal_state="HEALTHY",
        expected_fault_state="LINK_DOWN",
        num_steps=100,
        dt_s=0.1,
        fault_start_s=2.0,
        fault_duration_s=3.0,
        apply_perturbation=perturbation
    )

def get_all_scenarios() -> List[FaultScenario]:
    return [
        scenario_normal_link(),
        scenario_excessive_attenuation(),
        scenario_fiber_cut(),
        scenario_dirty_connector(),
        scenario_laser_aging(),
        scenario_noise_burst(),
        scenario_fault_recovery()
    ]
