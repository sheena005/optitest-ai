from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from optitest.core.models import TelemetrySample

@dataclass
class RuleThresholds:
    los_threshold_dbm: float = -28.0
    degraded_power_threshold_dbm: float = -22.0
    overpower_threshold_dbm: float = -1.0
    low_osnr_threshold_db: float = 18.0
    high_ber_threshold: float = 1.0e-4
    degraded_ber_threshold: float = 1.0e-7
    max_laser_temp_c: float = 55.0
    max_bias_current_ma: float = 65.0
    debounce_steps: int = 2
    clear_steps: int = 2

@dataclass
class RuleEvaluation:
    state: str
    active_alarms: List[str]
    rule_details: Dict[str, Any]
    is_passing: bool

class OpticalRuleEngine:
    """
    Deterministic rule-based optical validation engine implementing
    ITU-T / IEEE 802.3 optical transmission limits with debounce filtering.
    """
    def __init__(self, thresholds: Optional[RuleThresholds] = None):
        self.thresholds = thresholds or RuleThresholds()
        self.current_state: str = "HEALTHY"
        self._consecutive_degraded: int = 0
        self._consecutive_los: int = 0
        self._consecutive_healthy: int = 0
        self._history: List[RuleEvaluation] = []

    def reset(self):
        self.current_state = "HEALTHY"
        self._consecutive_degraded = 0
        self._consecutive_los = 0
        self._consecutive_healthy = 0
        self._history.clear()

    def evaluate_sample(self, sample: TelemetrySample) -> RuleEvaluation:
        alarms = []
        rule_details = {}

        # 1. Received Optical Power Checks
        if sample.rx_power_dbm <= self.thresholds.los_threshold_dbm:
            alarms.append("ALARM_LOS")
            rule_details["rx_power"] = f"CRITICAL_LOW ({sample.rx_power_dbm:.2f} dBm <= {self.thresholds.los_threshold_dbm} dBm)"
        elif sample.rx_power_dbm < self.thresholds.degraded_power_threshold_dbm:
            alarms.append("ALARM_DEGRADED_POWER")
            rule_details["rx_power"] = f"DEGRADED ({sample.rx_power_dbm:.2f} dBm < {self.thresholds.degraded_power_threshold_dbm} dBm)"
        elif sample.rx_power_dbm >= self.thresholds.overpower_threshold_dbm:
            alarms.append("ALARM_OVERPOWER")
            rule_details["rx_power"] = f"OVERPOWER ({sample.rx_power_dbm:.2f} dBm >= {self.thresholds.overpower_threshold_dbm} dBm)"
        else:
            rule_details["rx_power"] = f"NOMINAL ({sample.rx_power_dbm:.2f} dBm)"

        # 2. OSNR Check
        if sample.osnr_db < self.thresholds.low_osnr_threshold_db:
            alarms.append("ALARM_LOW_OSNR")
            rule_details["osnr"] = f"DEGRADED ({sample.osnr_db:.2f} dB < {self.thresholds.low_osnr_threshold_db} dB)"
        else:
            rule_details["osnr"] = f"NOMINAL ({sample.osnr_db:.2f} dB)"

        # 3. BER Check
        if sample.ber >= self.thresholds.high_ber_threshold:
            alarms.append("ALARM_HIGH_BER")
            rule_details["ber"] = f"CRITICAL ({sample.ber:.2e} >= {self.thresholds.high_ber_threshold:.2e})"
        elif sample.ber > self.thresholds.degraded_ber_threshold:
            alarms.append("ALARM_DEGRADED_BER")
            rule_details["ber"] = f"DEGRADED ({sample.ber:.2e} > {self.thresholds.degraded_ber_threshold:.2e})"
        else:
            rule_details["ber"] = f"NOMINAL ({sample.ber:.2e})"

        # 4. Laser Transceiver Health
        if sample.laser_bias_current_ma > self.thresholds.max_bias_current_ma:
            alarms.append("ALARM_LASER_BIAS_CURRENT")
            rule_details["bias_current"] = f"EXCESSIVE ({sample.laser_bias_current_ma:.1f} mA > {self.thresholds.max_bias_current_ma} mA)"
        else:
            rule_details["bias_current"] = f"NOMINAL ({sample.laser_bias_current_ma:.1f} mA)"

        if sample.laser_temperature_c > self.thresholds.max_laser_temp_c:
            alarms.append("ALARM_HIGH_TEMPERATURE")
            rule_details["temperature"] = f"HIGH ({sample.laser_temperature_c:.1f} C > {self.thresholds.max_laser_temp_c} C)"
        else:
            rule_details["temperature"] = f"NOMINAL ({sample.laser_temperature_c:.1f} C)"

        # Immediate raw candidate state
        if "ALARM_LOS" in alarms or sample.rx_power_dbm <= -28.0:
            candidate = "LINK_DOWN"
        elif "ALARM_HIGH_BER" in alarms or sample.rx_power_dbm < -25.0:
            candidate = "CRITICAL"
        elif len(alarms) > 0:
            candidate = "DEGRADED"
        else:
            candidate = "HEALTHY"

        # Apply Debounce Filtering
        if candidate == "LINK_DOWN":
            self._consecutive_los += 1
            self._consecutive_degraded += 1
            self._consecutive_healthy = 0
            if self._consecutive_los >= self.thresholds.debounce_steps:
                self.current_state = "LINK_DOWN"
        elif candidate in ("CRITICAL", "DEGRADED"):
            self._consecutive_los = 0
            self._consecutive_degraded += 1
            self._consecutive_healthy = 0
            if self._consecutive_degraded >= self.thresholds.debounce_steps:
                self.current_state = candidate
        else:
            self._consecutive_los = 0
            self._consecutive_degraded = 0
            self._consecutive_healthy += 1
            if self._consecutive_healthy >= self.thresholds.clear_steps:
                self.current_state = "HEALTHY"

        evaluation = RuleEvaluation(
            state=self.current_state,
            active_alarms=alarms,
            rule_details=rule_details,
            is_passing=(self.current_state == "HEALTHY")
        )
        self._history.append(evaluation)
        return evaluation
