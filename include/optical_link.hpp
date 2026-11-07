#pragma once

#include "telemetry.hpp"
#include <string>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <random>

namespace optitest {

struct LinkParameters {
    double tx_power_dbm{0.0};
    double fiber_length_km{40.0};
    double attenuation_db_per_km{0.20};
    double other_loss_db{2.0};
    double dispersion_param_ps_nm_km{17.0};
    double wavelength_nm{1550.0};
    double initial_osnr_db{38.0};
    double nominal_bias_current_ma{45.0};
    double nominal_temp_c{35.0};
    double rx_sensitivity_dbm{-24.0};
    double rx_overload_dbm{-1.0};
    double los_threshold_dbm{-28.0};

    void validate() const {
        if (fiber_length_km < 0.0) {
            throw std::invalid_argument("Fiber length cannot be negative.");
        }
        if (attenuation_db_per_km < 0.0) {
            throw std::invalid_argument("Attenuation coefficient cannot be negative.");
        }
        if (other_loss_db < 0.0) {
            throw std::invalid_argument("Other loss cannot be negative.");
        }
        if (wavelength_nm <= 0.0) {
            throw std::invalid_argument("Wavelength must be positive.");
        }
    }
};

class OpticalLink {
public:
    explicit OpticalLink(const LinkParameters& params = LinkParameters());

    void set_parameters(const LinkParameters& params);
    const LinkParameters& get_parameters() const;

    // Physical calculations
    double calculate_fiber_loss() const;
    double calculate_received_power() const;
    double calculate_received_power(double additional_attenuation_db) const;
    double calculate_osnr(double total_loss, double extra_noise_db = 0.0) const;
    double calculate_ber(double rx_power_dbm, double osnr_db) const;
    double calculate_q_factor(double ber) const;

    // Simulation stepping
    TelemetrySample step(
        double timestamp_s,
        double injected_attenuation_db = 0.0,
        double injected_noise_db = 0.0,
        double bias_current_drift_ma = 0.0,
        double temp_drift_c = 0.0,
        bool force_los = false,
        const std::string& fault_name = "NONE"
    );

    std::vector<TelemetrySample> simulate_profile(
        int num_steps,
        double dt_s = 0.1
    );

    void reset_noise_generator(unsigned int seed = 42);

private:
    LinkParameters params_;
    std::mt19937 rng_;
    std::normal_distribution<double> noise_dist_;
};

} // namespace optitest
