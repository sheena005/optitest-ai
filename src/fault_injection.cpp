#include "fault_injection.hpp"
#include <cmath>
#include <algorithm>

namespace optitest {

// --- FiberCutFault ---
FiberCutFault::FiberCutFault(double start_time_s, double duration_s)
    : start_time_s_(start_time_s), duration_s_(duration_s) {}

FaultPerturbation FiberCutFault::get_perturbation(double timestamp_s) {
    FaultPerturbation p;
    bool active = (timestamp_s >= start_time_s_);
    if (duration_s_ > 0.0 && timestamp_s >= (start_time_s_ + duration_s_)) {
        active = false;
    }
    if (active) {
        p.force_los = true;
        p.additional_attenuation_db = 99.0;
        p.fault_name = "FIBER_CUT";
    }
    return p;
}

// --- AttenuationRampFault ---
AttenuationRampFault::AttenuationRampFault(double start_time_s, double ramp_rate_db_per_s, double max_attenuation_db, double duration_s)
    : start_time_s_(start_time_s), ramp_rate_db_per_s_(ramp_rate_db_per_s), max_attenuation_db_(max_attenuation_db), duration_s_(duration_s) {}

FaultPerturbation AttenuationRampFault::get_perturbation(double timestamp_s) {
    FaultPerturbation p;
    bool active = (timestamp_s >= start_time_s_);
    if (duration_s_ > 0.0 && timestamp_s >= (start_time_s_ + duration_s_)) {
        active = false; // Recovered
    }
    if (active) {
        double elapsed = timestamp_s - start_time_s_;
        double added_att = std::min(max_attenuation_db_, elapsed * ramp_rate_db_per_s_);
        p.additional_attenuation_db = added_att;
        p.fault_name = "ATTENUATION_RAMP";
    }
    return p;
}

// --- ConnectorContaminationFault ---
ConnectorContaminationFault::ConnectorContaminationFault(double start_time_s, double added_loss_db, double duration_s)
    : start_time_s_(start_time_s), added_loss_db_(added_loss_db), duration_s_(duration_s) {}

FaultPerturbation ConnectorContaminationFault::get_perturbation(double timestamp_s) {
    FaultPerturbation p;
    bool active = (timestamp_s >= start_time_s_);
    if (duration_s_ > 0.0 && timestamp_s >= (start_time_s_ + duration_s_)) {
        active = false;
    }
    if (active) {
        p.additional_attenuation_db = added_loss_db_;
        p.additional_noise_db = 1.5; // slight return loss penalty
        p.fault_name = "CONNECTOR_CONTAMINATION";
    }
    return p;
}

// --- LaserAgingFault ---
LaserAgingFault::LaserAgingFault(double start_time_s, double current_drift_rate_ma_s, double max_drift_ma, double temp_drift_rate_c_s)
    : start_time_s_(start_time_s), current_drift_rate_ma_s_(current_drift_rate_ma_s), max_drift_ma_(max_drift_ma), temp_drift_rate_c_s_(temp_drift_rate_c_s) {}

FaultPerturbation LaserAgingFault::get_perturbation(double timestamp_s) {
    FaultPerturbation p;
    if (timestamp_s >= start_time_s_) {
        double elapsed = timestamp_s - start_time_s_;
        p.bias_current_drift_ma = std::min(max_drift_ma_, elapsed * current_drift_rate_ma_s_);
        p.temp_drift_c = elapsed * temp_drift_rate_c_s_;
        p.fault_name = "LASER_AGING";
    }
    return p;
}

// --- NoiseBurstFault ---
NoiseBurstFault::NoiseBurstFault(double start_time_s, double added_noise_db, double duration_s)
    : start_time_s_(start_time_s), added_noise_db_(added_noise_db), duration_s_(duration_s) {}

FaultPerturbation NoiseBurstFault::get_perturbation(double timestamp_s) {
    FaultPerturbation p;
    bool active = (timestamp_s >= start_time_s_);
    if (duration_s_ > 0.0 && timestamp_s >= (start_time_s_ + duration_s_)) {
        active = false;
    }
    if (active) {
        p.additional_noise_db = added_noise_db_;
        p.fault_name = "NOISE_BURST";
    }
    return p;
}

// --- TransientDropoutFault ---
TransientDropoutFault::TransientDropoutFault(double start_time_s, double period_s, double pulse_width_s, double duration_s)
    : start_time_s_(start_time_s), period_s_(period_s), pulse_width_s_(pulse_width_s), duration_s_(duration_s) {}

FaultPerturbation TransientDropoutFault::get_perturbation(double timestamp_s) {
    FaultPerturbation p;
    bool in_overall_window = (timestamp_s >= start_time_s_);
    if (duration_s_ > 0.0 && timestamp_s >= (start_time_s_ + duration_s_)) {
        in_overall_window = false;
    }
    if (in_overall_window) {
        double cycle_pos = std::fmod(timestamp_s - start_time_s_, period_s_);
        if (cycle_pos < pulse_width_s_) {
            p.force_los = true;
            p.additional_attenuation_db = 99.0;
            p.fault_name = "TRANSIENT_DROPOUT";
        }
    }
    return p;
}

// --- FaultEngine ---
FaultEngine::FaultEngine() = default;

void FaultEngine::add_injector(std::shared_ptr<FaultInjector> injector) {
    injectors_.push_back(injector);
}

void FaultEngine::clear_injectors() {
    injectors_.clear();
}

FaultPerturbation FaultEngine::evaluate(double timestamp_s) const {
    FaultPerturbation combined;
    for (const auto& injector : injectors_) {
        FaultPerturbation p = injector->get_perturbation(timestamp_s);
        combined.additional_attenuation_db += p.additional_attenuation_db;
        combined.additional_noise_db += p.additional_noise_db;
        combined.bias_current_drift_ma += p.bias_current_drift_ma;
        combined.temp_drift_c += p.temp_drift_c;
        if (p.force_los) {
            combined.force_los = true;
        }
        if (p.fault_name != "NONE") {
            if (combined.fault_name == "NONE") {
                combined.fault_name = p.fault_name;
            } else {
                combined.fault_name += "+" + p.fault_name;
            }
        }
    }
    return combined;
}

std::vector<TelemetrySample> FaultEngine::run_simulation(
    OpticalLink& link,
    int num_steps,
    double dt_s
) {
    std::vector<TelemetrySample> history;
    history.reserve(num_steps);

    for (int i = 0; i < num_steps; ++i) {
        double t = i * dt_s;
        FaultPerturbation p = evaluate(t);
        TelemetrySample s = link.step(
            t,
            p.additional_attenuation_db,
            p.additional_noise_db,
            p.bias_current_drift_ma,
            p.temp_drift_c,
            p.force_los,
            p.fault_name
        );
        history.push_back(s);
    }
    return history;
}

} // namespace optitest
