#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

namespace optitest {

struct TelemetrySample {
    double timestamp_s{0.0};
    double tx_power_dbm{0.0};
    double rx_power_dbm{-10.0};
    double fiber_loss_db{8.0};
    double other_loss_db{2.0};
    double total_loss_db{10.0};
    double osnr_db{32.0};
    double q_factor{8.5};
    double ber{1e-12};
    double chromatic_dispersion_ps_nm{680.0};
    double laser_temperature_c{35.0};
    double laser_bias_current_ma{45.0};
    double supply_voltage_v{3.3};
    bool los_alarm{false};
    bool high_ber_alarm{false};
    bool degraded_warning{false};
    std::string status_label{"HEALTHY"};
    std::string fault_active{"NONE"};

    std::string to_json() const {
        std::ostringstream ss;
        ss << "{"
           << "\"timestamp_s\":" << std::fixed << std::setprecision(3) << timestamp_s << ","
           << "\"tx_power_dbm\":" << std::setprecision(2) << tx_power_dbm << ","
           << "\"rx_power_dbm\":" << rx_power_dbm << ","
           << "\"fiber_loss_db\":" << fiber_loss_db << ","
           << "\"other_loss_db\":" << other_loss_db << ","
           << "\"total_loss_db\":" << total_loss_db << ","
           << "\"osnr_db\":" << osnr_db << ","
           << "\"q_factor\":" << q_factor << ","
           << "\"ber\":" << std::scientific << std::setprecision(3) << ber << ","
           << "\"chromatic_dispersion_ps_nm\":" << std::fixed << std::setprecision(2) << chromatic_dispersion_ps_nm << ","
           << "\"laser_temperature_c\":" << laser_temperature_c << ","
           << "\"laser_bias_current_ma\":" << laser_bias_current_ma << ","
           << "\"supply_voltage_v\":" << supply_voltage_v << ","
           << "\"los_alarm\":" << (los_alarm ? "true" : "false") << ","
           << "\"high_ber_alarm\":" << (high_ber_alarm ? "true" : "false") << ","
           << "\"degraded_warning\":" << (degraded_warning ? "true" : "false") << ","
           << "\"status_label\":\"" << status_label << "\","
           << "\"fault_active\":\"" << fault_active << "\""
           << "}";
        return ss.str();
    }
};

} // namespace optitest
