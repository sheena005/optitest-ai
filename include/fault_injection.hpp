#pragma once

#include "optical_link.hpp"
#include <string>
#include <memory>
#include <vector>

namespace optitest {

enum class FaultType {
    NONE,
    FIBER_CUT,
    ATTENUATION_RAMP,
    CONNECTOR_CONTAMINATION,
    LASER_AGING,
    NOISE_BURST,
    TRANSIENT_DROPOUT
};

struct FaultPerturbation {
    double additional_attenuation_db{0.0};
    double additional_noise_db{0.0};
    double bias_current_drift_ma{0.0};
    double temp_drift_c{0.0};
    bool force_los{false};
    std::string fault_name{"NONE"};
};

class FaultInjector {
public:
    virtual ~FaultInjector() = default;
    virtual FaultPerturbation get_perturbation(double timestamp_s) = 0;
    virtual void reset() {}
    virtual std::string get_name() const = 0;
};

class FiberCutFault : public FaultInjector {
public:
    FiberCutFault(double start_time_s = 2.0, double duration_s = -1.0);
    FaultPerturbation get_perturbation(double timestamp_s) override;
    std::string get_name() const override { return "FIBER_CUT"; }

private:
    double start_time_s_;
    double duration_s_;
};

class AttenuationRampFault : public FaultInjector {
public:
    AttenuationRampFault(double start_time_s = 2.0, double ramp_rate_db_per_s = 1.5, double max_attenuation_db = 15.0, double duration_s = -1.0);
    FaultPerturbation get_perturbation(double timestamp_s) override;
    std::string get_name() const override { return "ATTENUATION_RAMP"; }

private:
    double start_time_s_;
    double ramp_rate_db_per_s_;
    double max_attenuation_db_;
    double duration_s_;
};

class ConnectorContaminationFault : public FaultInjector {
public:
    ConnectorContaminationFault(double start_time_s = 2.0, double added_loss_db = 6.0, double duration_s = -1.0);
    FaultPerturbation get_perturbation(double timestamp_s) override;
    std::string get_name() const override { return "CONNECTOR_CONTAMINATION"; }

private:
    double start_time_s_;
    double added_loss_db_;
    double duration_s_;
};

class LaserAgingFault : public FaultInjector {
public:
    LaserAgingFault(double start_time_s = 2.0, double current_drift_rate_ma_s = 5.0, double max_drift_ma = 30.0, double temp_drift_rate_c_s = 2.0);
    FaultPerturbation get_perturbation(double timestamp_s) override;
    std::string get_name() const override { return "LASER_AGING"; }

private:
    double start_time_s_;
    double current_drift_rate_ma_s_;
    double max_drift_ma_;
    double temp_drift_rate_c_s_;
};

class NoiseBurstFault : public FaultInjector {
public:
    NoiseBurstFault(double start_time_s = 2.0, double added_noise_db = 12.0, double duration_s = -1.0);
    FaultPerturbation get_perturbation(double timestamp_s) override;
    std::string get_name() const override { return "NOISE_BURST"; }

private:
    double start_time_s_;
    double added_noise_db_;
    double duration_s_;
};

class TransientDropoutFault : public FaultInjector {
public:
    TransientDropoutFault(double start_time_s = 1.0, double period_s = 1.0, double pulse_width_s = 0.2, double duration_s = -1.0);
    FaultPerturbation get_perturbation(double timestamp_s) override;
    std::string get_name() const override { return "TRANSIENT_DROPOUT"; }

private:
    double start_time_s_;
    double period_s_;
    double pulse_width_s_;
    double duration_s_;
};

class FaultEngine {
public:
    FaultEngine();

    void add_injector(std::shared_ptr<FaultInjector> injector);
    void clear_injectors();

    FaultPerturbation evaluate(double timestamp_s) const;

    std::vector<TelemetrySample> run_simulation(
        OpticalLink& link,
        int num_steps = 100,
        double dt_s = 0.1
    );

private:
    std::vector<std::shared_ptr<FaultInjector>> injectors_;
};

} // namespace optitest
