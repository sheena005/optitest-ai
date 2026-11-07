import math
import random
from typing import List, Optional
from optitest.core.models import LinkParameters, TelemetrySample

# Attempt to load compiled C++ core
_NATIVE_AVAILABLE = False
try:
    import optitest_core
    _NATIVE_AVAILABLE = True
except ImportError:
    optitest_core = None

def is_native_available() -> bool:
    return _NATIVE_AVAILABLE

class PurePythonOpticalLink:
    """Pure Python implementation of OpticalLink, matching C++ physics equations exactly."""
    def __init__(self, params: Optional[LinkParameters] = None):
        self.params = params or LinkParameters()
        self.params.validate()
        self.rng = random.Random(42)

    def set_parameters(self, params: LinkParameters):
        params.validate()
        self.params = params

    def get_parameters(self) -> LinkParameters:
        return self.params

    def reset_noise_generator(self, seed: int = 42):
        self.rng = random.Random(seed)

    def calculate_fiber_loss(self) -> float:
        return self.params.attenuation_db_per_km * self.params.fiber_length_km

    def calculate_received_power(self, additional_attenuation_db: float = 0.0) -> float:
        total_loss = self.calculate_fiber_loss() + self.params.other_loss_db + additional_attenuation_db
        return self.params.tx_power_dbm - total_loss

    def calculate_osnr(self, total_loss: float, extra_noise_db: float = 0.0) -> float:
        degraded = self.params.initial_osnr_db - (0.35 * total_loss) - extra_noise_db
        return max(3.0, degraded)

    def calculate_ber(self, rx_power_dbm: float, osnr_db: float) -> float:
        if rx_power_dbm <= self.params.los_threshold_dbm:
            return 0.5

        q_nominal = 10.0 ** ((osnr_db - 12.0) / 20.0)
        q_eff = q_nominal
        if rx_power_dbm < self.params.rx_sensitivity_dbm:
            deficit = self.params.rx_sensitivity_dbm - rx_power_dbm
            q_eff = q_nominal / (1.0 + 0.6 * (deficit ** 1.8))

        # erfc calculation
        ber = 0.5 * math.erfc(q_eff / math.sqrt(2.0))
        return max(1.0e-15, min(0.5, ber))

    def calculate_q_factor(self, ber: float) -> float:
        if ber >= 0.5:
            return 0.0
        if ber <= 1.0e-15:
            return 9.5
        t = math.sqrt(-2.0 * math.log(2.0 * ber))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        q = t - ((c2 * t + c1) * t + c0) / (((d3 * t + d2) * t + d1) * t + 1.0)
        return max(0.0, q)

    def step(
        self,
        timestamp_s: float,
        injected_attenuation_db: float = 0.0,
        injected_noise_db: float = 0.0,
        bias_current_drift_ma: float = 0.0,
        temp_drift_c: float = 0.0,
        force_los: bool = False,
        fault_name: str = "NONE"
    ) -> TelemetrySample:
        jitter = self.rng.gauss(0.0, 0.04)
        if force_los:
            return TelemetrySample(
                timestamp_s=timestamp_s,
                tx_power_dbm=self.params.tx_power_dbm,
                rx_power_dbm=-50.0 + (jitter * 0.2),
                fiber_loss_db=99.0,
                other_loss_db=self.params.other_loss_db,
                total_loss_db=99.0,
                osnr_db=0.0,
                q_factor=0.0,
                ber=0.5,
                chromatic_dispersion_ps_nm=self.params.dispersion_param_ps_nm_km * self.params.fiber_length_km,
                laser_temperature_c=self.params.nominal_temp_c + temp_drift_c,
                laser_bias_current_ma=self.params.nominal_bias_current_ma + bias_current_drift_ma,
                supply_voltage_v=3.30 + (jitter * 0.01),
                los_alarm=True,
                high_ber_alarm=True,
                degraded_warning=True,
                status_label="LINK_DOWN",
                fault_active=fault_name,
            )

        fiber_loss = self.calculate_fiber_loss() + injected_attenuation_db
        total_loss = fiber_loss + self.params.other_loss_db
        rx_power = max(-50.0, (self.params.tx_power_dbm - total_loss) + jitter)
        osnr = max(0.0, self.calculate_osnr(total_loss, injected_noise_db) + (jitter * 0.2))
        ber = self.calculate_ber(rx_power, osnr)
        q_factor = self.calculate_q_factor(ber)

        los_alarm = rx_power <= self.params.los_threshold_dbm
        high_ber = ber >= 1.0e-4
        degraded = rx_power < self.params.rx_sensitivity_dbm or osnr < 18.0 or ber > 1.0e-7

        if los_alarm:
            status = "LINK_DOWN"
        elif high_ber or rx_power < -25.0:
            status = "CRITICAL"
        elif degraded:
            status = "DEGRADED"
        else:
            status = "HEALTHY"

        return TelemetrySample(
            timestamp_s=timestamp_s,
            tx_power_dbm=self.params.tx_power_dbm + (jitter * 0.05),
            rx_power_dbm=rx_power,
            fiber_loss_db=fiber_loss,
            other_loss_db=self.params.other_loss_db,
            total_loss_db=total_loss,
            osnr_db=osnr,
            q_factor=q_factor,
            ber=ber,
            chromatic_dispersion_ps_nm=self.params.dispersion_param_ps_nm_km * self.params.fiber_length_km,
            laser_temperature_c=self.params.nominal_temp_c + temp_drift_c + (jitter * 0.1),
            laser_bias_current_ma=self.params.nominal_bias_current_ma + bias_current_drift_ma + (jitter * 0.2),
            supply_voltage_v=3.30 + (jitter * 0.005),
            los_alarm=los_alarm,
            high_ber_alarm=high_ber,
            degraded_warning=degraded,
            status_label=status,
            fault_active=fault_name,
        )

    def simulate_profile(self, num_steps: int = 100, dt_s: float = 0.1) -> List[TelemetrySample]:
        return [self.step(i * dt_s) for i in range(num_steps)]


class OpticalLinkWrapper:
    """Unified wrapper around C++ OpticalLink (when available) or PurePythonOpticalLink."""
    def __init__(self, params: Optional[LinkParameters] = None, force_python: bool = False):
        self.params = params or LinkParameters()
        self.force_python = force_python

        if _NATIVE_AVAILABLE and not force_python:
            c_params = optitest_core.LinkParameters()
            for key, val in self.params.__dict__.items():
                if hasattr(c_params, key):
                    setattr(c_params, key, val)
            self._link = optitest_core.OpticalLink(c_params)
            self._is_native = True
        else:
            self._link = PurePythonOpticalLink(self.params)
            self._is_native = False

    @property
    def is_native(self) -> bool:
        return self._is_native

    def set_parameters(self, params: LinkParameters):
        self.params = params
        if self._is_native:
            c_params = optitest_core.LinkParameters()
            for key, val in self.params.__dict__.items():
                if hasattr(c_params, key):
                    setattr(c_params, key, val)
            self._link.set_parameters(c_params)
        else:
            self._link.set_parameters(params)

    def get_parameters(self) -> LinkParameters:
        return self.params

    def reset_noise_generator(self, seed: int = 42):
        self._link.reset_noise_generator(seed)

    def calculate_fiber_loss(self) -> float:
        return self._link.calculate_fiber_loss()

    def calculate_received_power(self, additional_attenuation_db: float = 0.0) -> float:
        if self._is_native:
            return self._link.calculate_received_power(additional_attenuation_db)
        return self._link.calculate_received_power(additional_attenuation_db)

    def calculate_osnr(self, total_loss: float, extra_noise_db: float = 0.0) -> float:
        return self._link.calculate_osnr(total_loss, extra_noise_db)

    def calculate_ber(self, rx_power_dbm: float, osnr_db: float) -> float:
        return self._link.calculate_ber(rx_power_dbm, osnr_db)

    def calculate_q_factor(self, ber: float) -> float:
        return self._link.calculate_q_factor(ber)

    def step(
        self,
        timestamp_s: float,
        injected_attenuation_db: float = 0.0,
        injected_noise_db: float = 0.0,
        bias_current_drift_ma: float = 0.0,
        temp_drift_c: float = 0.0,
        force_los: bool = False,
        fault_name: str = "NONE"
    ) -> TelemetrySample:
        if self._is_native:
            s = self._link.step(
                timestamp_s,
                injected_attenuation_db,
                injected_noise_db,
                bias_current_drift_ma,
                temp_drift_c,
                force_los,
                fault_name
            )
            return TelemetrySample(
                timestamp_s=s.timestamp_s,
                tx_power_dbm=s.tx_power_dbm,
                rx_power_dbm=s.rx_power_dbm,
                fiber_loss_db=s.fiber_loss_db,
                other_loss_db=s.other_loss_db,
                total_loss_db=s.total_loss_db,
                osnr_db=s.osnr_db,
                q_factor=s.q_factor,
                ber=s.ber,
                chromatic_dispersion_ps_nm=s.chromatic_dispersion_ps_nm,
                laser_temperature_c=s.laser_temperature_c,
                laser_bias_current_ma=s.laser_bias_current_ma,
                supply_voltage_v=s.supply_voltage_v,
                los_alarm=s.los_alarm,
                high_ber_alarm=s.high_ber_alarm,
                degraded_warning=s.degraded_warning,
                status_label=s.status_label,
                fault_active=s.fault_active,
            )
        else:
            return self._link.step(
                timestamp_s,
                injected_attenuation_db,
                injected_noise_db,
                bias_current_drift_ma,
                temp_drift_c,
                force_los,
                fault_name
            )

    def simulate_profile(self, num_steps: int = 100, dt_s: float = 0.1) -> List[TelemetrySample]:
        return [self.step(i * dt_s) for i in range(num_steps)]
