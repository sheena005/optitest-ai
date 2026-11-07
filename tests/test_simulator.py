import pytest
import math
from optitest.core.models import LinkParameters
from optitest.core.simulator import OpticalLinkWrapper, PurePythonOpticalLink, is_native_available

def test_nominal_power_calculation():
    """Verify standard formula: P_rx = P_tx - alpha*L - L_other."""
    params = LinkParameters(
        tx_power_dbm=0.0,
        fiber_length_km=40.0,
        attenuation_db_per_km=0.20,
        other_loss_db=2.0
    )
    link = OpticalLinkWrapper(params)
    assert link.calculate_fiber_loss() == pytest.approx(8.0, abs=1e-5)
    # 0 - (0.2 * 40) - 2 = -10.0 dBm
    assert link.calculate_received_power() == pytest.approx(-10.0, abs=1e-5)
    # Added attenuation of 5 dB
    assert link.calculate_received_power(5.0) == pytest.approx(-15.0, abs=1e-5)

def test_zero_length_boundary():
    """Link with 0km fiber should only experience other_loss_db."""
    params = LinkParameters(
        tx_power_dbm=3.0,
        fiber_length_km=0.0,
        attenuation_db_per_km=0.20,
        other_loss_db=1.5
    )
    link = OpticalLinkWrapper(params)
    assert link.calculate_fiber_loss() == pytest.approx(0.0, abs=1e-5)
    assert link.calculate_received_power() == pytest.approx(1.5, abs=1e-5)

def test_invalid_negative_length():
    """Negative fiber length must raise ValueError."""
    params = LinkParameters(fiber_length_km=-5.0)
    with pytest.raises((ValueError, Exception)):
        OpticalLinkWrapper(params)

def test_invalid_negative_attenuation():
    """Negative attenuation coefficient must raise ValueError."""
    params = LinkParameters(attenuation_db_per_km=-0.2)
    with pytest.raises((ValueError, Exception)):
        OpticalLinkWrapper(params)

def test_ber_and_q_factor_relationship():
    """Higher OSNR and healthy power must produce low BER and high Q-factor."""
    params = LinkParameters(tx_power_dbm=0.0, fiber_length_km=20.0)
    link = OpticalLinkWrapper(params)
    ber_good = link.calculate_ber(rx_power_dbm=-8.0, osnr_db=32.0)
    q_good = link.calculate_q_factor(ber_good)

    ber_bad = link.calculate_ber(rx_power_dbm=-26.0, osnr_db=15.0)
    q_bad = link.calculate_q_factor(ber_bad)

    assert ber_good < 1e-10
    assert q_good > 6.0
    assert ber_bad > 1e-5
    assert q_bad < q_good

def test_los_threshold_clamp():
    """When Rx power drops below LOS threshold, BER must be clamped to 0.5."""
    params = LinkParameters(los_threshold_dbm=-28.0)
    link = OpticalLinkWrapper(params)
    ber_los = link.calculate_ber(rx_power_dbm=-30.0, osnr_db=20.0)
    assert ber_los == pytest.approx(0.5, abs=1e-6)
    assert link.calculate_q_factor(ber_los) == 0.0

def test_native_vs_python_parity():
    """Verify pure-Python fallback produces identical physics to C++ engine."""
    params = LinkParameters(
        tx_power_dbm=2.0,
        fiber_length_km=50.0,
        attenuation_db_per_km=0.22,
        other_loss_db=2.5
    )
    py_link = OpticalLinkWrapper(params, force_python=True)
    assert not py_link.is_native

    if is_native_available():
        native_link = OpticalLinkWrapper(params, force_python=False)
        assert native_link.is_native
        assert native_link.calculate_fiber_loss() == pytest.approx(py_link.calculate_fiber_loss(), abs=1e-4)
        assert native_link.calculate_received_power() == pytest.approx(py_link.calculate_received_power(), abs=1e-4)
        assert native_link.calculate_osnr(13.5) == pytest.approx(py_link.calculate_osnr(13.5), abs=1e-4)
