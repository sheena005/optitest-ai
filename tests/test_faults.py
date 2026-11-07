import pytest
from optitest.core.models import LinkParameters
from optitest.core.simulator import OpticalLinkWrapper
from optitest.faults.scenarios import (
    scenario_normal_link,
    scenario_fiber_cut,
    scenario_excessive_attenuation,
    scenario_dirty_connector,
    scenario_laser_aging,
    scenario_noise_burst,
    scenario_fault_recovery
)

def test_fiber_cut_triggers_los():
    sc = scenario_fiber_cut()
    link = OpticalLinkWrapper(sc.link_params)

    # Before fault (t = 1.0s)
    s_pre = link.step(timestamp_s=1.0, **sc.apply_perturbation(1.0))
    assert not s_pre.los_alarm
    assert s_pre.rx_power_dbm > -15.0

    # During fault (t = 2.5s)
    s_fault = link.step(timestamp_s=2.5, **sc.apply_perturbation(2.5))
    assert s_fault.los_alarm
    assert s_fault.rx_power_dbm <= -28.0
    assert s_fault.status_label == "LINK_DOWN"

def test_attenuation_ramp_progressive_loss():
    sc = scenario_excessive_attenuation()
    link = OpticalLinkWrapper(sc.link_params)

    s1 = link.step(timestamp_s=1.0, **sc.apply_perturbation(1.0))
    s2 = link.step(timestamp_s=5.0, **sc.apply_perturbation(5.0))
    s3 = link.step(timestamp_s=10.0, **sc.apply_perturbation(10.0))

    assert s1.rx_power_dbm > s2.rx_power_dbm > s3.rx_power_dbm
    assert s3.degraded_warning or s3.high_ber_alarm

def test_laser_aging_bias_drift():
    sc = scenario_laser_aging()
    link = OpticalLinkWrapper(sc.link_params)

    s_pre = link.step(timestamp_s=1.0, **sc.apply_perturbation(1.0))
    s_post = link.step(timestamp_s=5.0, **sc.apply_perturbation(5.0))

    assert s_post.laser_bias_current_ma > s_pre.laser_bias_current_ma + 10.0
    assert s_post.laser_temperature_c > s_pre.laser_temperature_c + 5.0

def test_fault_recovery_cycle():
    sc = scenario_fault_recovery()
    link = OpticalLinkWrapper(sc.link_params)

    # 1. Before fault
    s1 = link.step(timestamp_s=1.0, **sc.apply_perturbation(1.0))
    assert s1.status_label == "HEALTHY"

    # 2. During fault (t=3.0s)
    s2 = link.step(timestamp_s=3.0, **sc.apply_perturbation(3.0))
    assert s2.status_label == "LINK_DOWN"
    assert s2.los_alarm

    # 3. After recovery (t=6.0s)
    s3 = link.step(timestamp_s=6.0, **sc.apply_perturbation(6.0))
    assert s3.status_label == "HEALTHY"
    assert not s3.los_alarm
