from dataclasses import dataclass, asdict
import json
import math
from typing import Dict, Any, Optional

@dataclass
class LinkParameters:
    tx_power_dbm: float = 0.0
    fiber_length_km: float = 40.0
    attenuation_db_per_km: float = 0.20
    other_loss_db: float = 2.0
    dispersion_param_ps_nm_km: float = 17.0
    wavelength_nm: float = 1550.0
    initial_osnr_db: float = 38.0
    nominal_bias_current_ma: float = 45.0
    nominal_temp_c: float = 35.0
    rx_sensitivity_dbm: float = -24.0
    rx_overload_dbm: float = -1.0
    los_threshold_dbm: float = -28.0

    def validate(self):
        if self.fiber_length_km < 0:
            raise ValueError("Fiber length cannot be negative.")
        if self.attenuation_db_per_km < 0:
            raise ValueError("Attenuation coefficient cannot be negative.")
        if self.other_loss_db < 0:
            raise ValueError("Other loss cannot be negative.")
        if self.wavelength_nm <= 0:
            raise ValueError("Wavelength must be positive.")

@dataclass
class TelemetrySample:
    timestamp_s: float = 0.0
    tx_power_dbm: float = 0.0
    rx_power_dbm: float = -10.0
    fiber_loss_db: float = 8.0
    other_loss_db: float = 2.0
    total_loss_db: float = 10.0
    osnr_db: float = 32.0
    q_factor: float = 8.5
    ber: float = 1e-12
    chromatic_dispersion_ps_nm: float = 680.0
    laser_temperature_c: float = 35.0
    laser_bias_current_ma: float = 45.0
    supply_voltage_v: float = 3.30
    los_alarm: bool = False
    high_ber_alarm: bool = False
    degraded_warning: bool = False
    status_label: str = "HEALTHY"
    fault_active: str = "NONE"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
