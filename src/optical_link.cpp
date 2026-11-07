#include "optical_link.hpp"
#include <cmath>
#include <algorithm>

namespace optitest {

OpticalLink::OpticalLink(const LinkParameters& params)
    : params_(params), rng_(42), noise_dist_(0.0, 0.04) {
    params_.validate();
}

void OpticalLink::set_parameters(const LinkParameters& params) {
    params.validate();
    params_ = params;
}

const LinkParameters& OpticalLink::get_parameters() const {
    return params_;
}

void OpticalLink::reset_noise_generator(unsigned int seed) {
    rng_.seed(seed);
}

double OpticalLink::calculate_fiber_loss() const {
    return params_.attenuation_db_per_km * params_.fiber_length_km;
}

double OpticalLink::calculate_received_power() const {
    double total_loss = calculate_fiber_loss() + params_.other_loss_db;
    return params_.tx_power_dbm - total_loss;
}

double OpticalLink::calculate_received_power(double additional_attenuation_db) const {
    double total_loss = calculate_fiber_loss() + params_.other_loss_db + additional_attenuation_db;
    return params_.tx_power_dbm - total_loss;
}

double OpticalLink::calculate_osnr(double total_loss, double extra_noise_db) const {
    // Standard model: OSNR drops with loss & amplifier noise degradation
    double degraded_osnr = params_.initial_osnr_db - (0.35 * total_loss) - extra_noise_db;
    return std::max(3.0, degraded_osnr);
}

double OpticalLink::calculate_ber(double rx_power_dbm, double osnr_db) const {
    if (rx_power_dbm <= params_.los_threshold_dbm) {
        return 0.5; // Complete loss of signal / loss of framing
    }

    // Nominal Q-factor derived from OSNR
    double q_nominal = std::pow(10.0, (osnr_db - 12.0) / 20.0);

    // Power sensitivity penalty if power is below target receiver sensitivity
    double q_eff = q_nominal;
    if (rx_power_dbm < params_.rx_sensitivity_dbm) {
        double power_deficit = params_.rx_sensitivity_dbm - rx_power_dbm;
        q_eff = q_nominal / (1.0 + 0.6 * std::pow(power_deficit, 1.8));
    }

    // erfc-based BER calculation
    // BER = 0.5 * erfc(Q / sqrt(2))
    double ber = 0.5 * std::erfc(q_eff / std::sqrt(2.0));
    ber = std::clamp(ber, 1.0e-15, 0.5);
    return ber;
}

double OpticalLink::calculate_q_factor(double ber) const {
    if (ber >= 0.5) return 0.0;
    if (ber <= 1.0e-15) return 9.5;

    // Approximate inverse erfc for Q factor:
    // Q ~= sqrt(2) * inverfc(2 * ber)
    // Winitzki / standard rational approximation
    double t = std::sqrt(-2.0 * std::log(2.0 * ber));
    double c0 = 2.515517;
    double c1 = 0.802853;
    double c2 = 0.010328;
    double d1 = 1.432788;
    double d2 = 0.189269;
    double d3 = 0.001308;
    double q = t - ((c2 * t + c1) * t + c0) / (((d3 * t + d2) * t + d1) * t + 1.0);
    return std::max(0.0, q);
}

TelemetrySample OpticalLink::step(
    double timestamp_s,
    double injected_attenuation_db,
    double injected_noise_db,
    double bias_current_drift_ma,
    double temp_drift_c,
    bool force_los,
    const std::string& fault_name
) {
    TelemetrySample sample;
    sample.timestamp_s = timestamp_s;
    sample.fault_active = fault_name;

    double measurement_jitter = noise_dist_(rng_);

    if (force_los) {
        sample.tx_power_dbm = params_.tx_power_dbm;
        sample.fiber_loss_db = 99.0;
        sample.other_loss_db = params_.other_loss_db;
        sample.total_loss_db = 99.0;
        sample.rx_power_dbm = -50.0 + (measurement_jitter * 0.2); // Noise floor
        sample.osnr_db = 0.0;
        sample.q_factor = 0.0;
        sample.ber = 0.5;
        sample.chromatic_dispersion_ps_nm = params_.dispersion_param_ps_nm_km * params_.fiber_length_km;
        sample.laser_temperature_c = params_.nominal_temp_c + temp_drift_c;
        sample.laser_bias_current_ma = params_.nominal_bias_current_ma + bias_current_drift_ma;
        sample.supply_voltage_v = 3.30 + (measurement_jitter * 0.01);
        sample.los_alarm = true;
        sample.high_ber_alarm = true;
        sample.degraded_warning = true;
        sample.status_label = "LINK_DOWN";
        return sample;
    }

    double fiber_loss = calculate_fiber_loss() + injected_attenuation_db;
    double total_loss = fiber_loss + params_.other_loss_db;
    double raw_rx_power = params_.tx_power_dbm - total_loss;

    // Apply measurement jitter
    double rx_power = raw_rx_power + measurement_jitter;
    rx_power = std::max(-50.0, rx_power);

    double osnr = calculate_osnr(total_loss, injected_noise_db) + (measurement_jitter * 0.2);
    osnr = std::max(0.0, osnr);

    double ber = calculate_ber(rx_power, osnr);
    double q_factor = calculate_q_factor(ber);

    sample.tx_power_dbm = params_.tx_power_dbm + (measurement_jitter * 0.05);
    sample.fiber_loss_db = fiber_loss;
    sample.other_loss_db = params_.other_loss_db;
    sample.total_loss_db = total_loss;
    sample.rx_power_dbm = rx_power;
    sample.osnr_db = osnr;
    sample.q_factor = q_factor;
    sample.ber = ber;
    sample.chromatic_dispersion_ps_nm = params_.dispersion_param_ps_nm_km * params_.fiber_length_km;
    sample.laser_temperature_c = params_.nominal_temp_c + temp_drift_c + (measurement_jitter * 0.1);
    sample.laser_bias_current_ma = params_.nominal_bias_current_ma + bias_current_drift_ma + (measurement_jitter * 0.2);
    sample.supply_voltage_v = 3.30 + (measurement_jitter * 0.005);

    // Alarm determination
    sample.los_alarm = (rx_power <= params_.los_threshold_dbm);
    sample.high_ber_alarm = (ber >= 1.0e-4);
    sample.degraded_warning = (rx_power < params_.rx_sensitivity_dbm || osnr < 18.0 || ber > 1.0e-7);

    if (sample.los_alarm) {
        sample.status_label = "LINK_DOWN";
    } else if (sample.high_ber_alarm || rx_power < -25.0) {
        sample.status_label = "CRITICAL";
    } else if (sample.degraded_warning) {
        sample.status_label = "DEGRADED";
    } else {
        sample.status_label = "HEALTHY";
    }

    return sample;
}

std::vector<TelemetrySample> OpticalLink::simulate_profile(int num_steps, double dt_s) {
    std::vector<TelemetrySample> samples;
    samples.reserve(num_steps);
    for (int i = 0; i < num_steps; ++i) {
        samples.push_back(step(i * dt_s));
    }
    return samples;
}

} // namespace optitest
