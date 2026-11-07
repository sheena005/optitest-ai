import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from optitest.core.models import LinkParameters, TelemetrySample
from optitest.core.simulator import OpticalLinkWrapper

LABEL_MAP = {
    "NORMAL": 0,
    "FIBER_CUT": 1,
    "EXCESS_ATTENUATION": 2,
    "DIRTY_CONNECTOR": 3,
    "LASER_AGING": 4,
    "OPTICAL_NOISE": 5
}

REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

FEATURE_COLUMNS = [
    "rx_power_dbm",
    "tx_power_dbm",
    "osnr_db",
    "log10_ber",
    "q_factor",
    "laser_temperature_c",
    "laser_bias_current_ma",
    "supply_voltage_v",
    "rx_power_delta",
    "bias_current_delta",
    "osnr_delta",
    "rx_power_rolling_std"
]

def extract_features_from_history(history: List[TelemetrySample]) -> pd.DataFrame:
    """Extract tabular features including temporal gradients and rolling statistics."""
    records = []
    for s in history:
        # Avoid log10(0)
        ber_val = max(1.0e-15, min(0.5, s.ber))
        log10_ber = float(np.log10(ber_val))
        records.append({
            "timestamp_s": s.timestamp_s,
            "rx_power_dbm": s.rx_power_dbm,
            "tx_power_dbm": s.tx_power_dbm,
            "osnr_db": s.osnr_db,
            "log10_ber": log10_ber,
            "q_factor": s.q_factor,
            "laser_temperature_c": s.laser_temperature_c,
            "laser_bias_current_ma": s.laser_bias_current_ma,
            "supply_voltage_v": s.supply_voltage_v,
            "fault_active": s.fault_active,
            "status_label": s.status_label
        })

    df = pd.DataFrame(records)
    # Temporal deltas
    df["rx_power_delta"] = df["rx_power_dbm"].diff().fillna(0.0)
    df["bias_current_delta"] = df["laser_bias_current_ma"].diff().fillna(0.0)
    df["osnr_delta"] = df["osnr_db"].diff().fillna(0.0)
    # Rolling standard deviation (5-step window)
    df["rx_power_rolling_std"] = df["rx_power_dbm"].rolling(window=5, min_periods=1).std().fillna(0.0)

    return df

def generate_telemetry_dataset(num_runs_per_fault: int = 15, seed: int = 42) -> pd.DataFrame:
    """
    Synthesize diverse optical link profiles across lengths, power levels, and fault types.
    """
    np.random.seed(seed)
    all_dfs = []

    lengths = [10.0, 25.0, 40.0, 60.0, 80.0]
    tx_powers = [-2.0, 0.0, 2.0, 4.0]

    run_id = 0
    fault_types = list(LABEL_MAP.keys())

    for fault in fault_types:
        for _ in range(num_runs_per_fault):
            length = float(np.random.choice(lengths)) + np.random.uniform(-3.0, 3.0)
            tx_pow = float(np.random.choice(tx_powers)) + np.random.uniform(-0.5, 0.5)
            other_loss = 2.0 + np.random.uniform(-0.5, 0.5)
            alpha = 0.20 + np.random.uniform(-0.02, 0.02)

            params = LinkParameters(
                tx_power_dbm=tx_pow,
                fiber_length_km=length,
                attenuation_db_per_km=alpha,
                other_loss_db=other_loss
            )
            link = OpticalLinkWrapper(params)
            link.reset_noise_generator(seed + run_id)

            history = []
            num_steps = 60
            dt = 0.1
            fault_start = 2.0  # at step 20

            for i in range(num_steps):
                t = i * dt
                # Default nominal perturbation
                p_att, p_noise, p_bias, p_temp, force_los = 0.0, 0.0, 0.0, 0.0, False
                current_label = "NORMAL"

                if t >= fault_start:
                    if fault == "FIBER_CUT":
                        force_los = True
                        p_att = 99.0
                        current_label = "FIBER_CUT"
                    elif fault == "EXCESS_ATTENUATION":
                        p_att = min(16.0, (t - fault_start) * 2.5)
                        current_label = "EXCESS_ATTENUATION"
                    elif fault == "DIRTY_CONNECTOR":
                        p_att = 12.0 + np.random.uniform(0.0, 2.0)
                        p_noise = 2.0
                        current_label = "DIRTY_CONNECTOR"
                    elif fault == "LASER_AGING":
                        elapsed = t - fault_start
                        p_bias = min(32.0, elapsed * 5.0)
                        p_temp = min(16.0, elapsed * 2.5)
                        p_att = 0.4
                        current_label = "LASER_AGING"
                    elif fault == "OPTICAL_NOISE":
                        p_noise = 10.0 + np.random.uniform(0.0, 3.0)
                        current_label = "OPTICAL_NOISE"

                sample = link.step(
                    timestamp_s=t,
                    injected_attenuation_db=p_att,
                    injected_noise_db=p_noise,
                    bias_current_drift_ma=p_bias,
                    temp_drift_c=p_temp,
                    force_los=force_los,
                    fault_name=current_label
                )
                sample.fault_active = current_label
                history.append(sample)

            run_df = extract_features_from_history(history)
            run_df["fault_category"] = [s.fault_active for s in history]
            run_df["label"] = run_df["fault_category"].map(LABEL_MAP)
            run_df["run_id"] = run_id
            all_dfs.append(run_df)
            run_id += 1

    dataset = pd.concat(all_dfs, ignore_index=True)
    return dataset
