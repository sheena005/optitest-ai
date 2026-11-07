#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "optical_link.hpp"
#include "fault_injection.hpp"
#include "telemetry.hpp"

namespace py = pybind11;
using namespace optitest;

PYBIND11_MODULE(optitest_core, m) {
    m.doc() = "OptiTest AI - Optical Link Simulation and Fault Injection C++ Core";

    // TelemetrySample
    py::class_<TelemetrySample>(m, "TelemetrySample")
        .def(py::init<>())
        .def_readwrite("timestamp_s", &TelemetrySample::timestamp_s)
        .def_readwrite("tx_power_dbm", &TelemetrySample::tx_power_dbm)
        .def_readwrite("rx_power_dbm", &TelemetrySample::rx_power_dbm)
        .def_readwrite("fiber_loss_db", &TelemetrySample::fiber_loss_db)
        .def_readwrite("other_loss_db", &TelemetrySample::other_loss_db)
        .def_readwrite("total_loss_db", &TelemetrySample::total_loss_db)
        .def_readwrite("osnr_db", &TelemetrySample::osnr_db)
        .def_readwrite("q_factor", &TelemetrySample::q_factor)
        .def_readwrite("ber", &TelemetrySample::ber)
        .def_readwrite("chromatic_dispersion_ps_nm", &TelemetrySample::chromatic_dispersion_ps_nm)
        .def_readwrite("laser_temperature_c", &TelemetrySample::laser_temperature_c)
        .def_readwrite("laser_bias_current_ma", &TelemetrySample::laser_bias_current_ma)
        .def_readwrite("supply_voltage_v", &TelemetrySample::supply_voltage_v)
        .def_readwrite("los_alarm", &TelemetrySample::los_alarm)
        .def_readwrite("high_ber_alarm", &TelemetrySample::high_ber_alarm)
        .def_readwrite("degraded_warning", &TelemetrySample::degraded_warning)
        .def_readwrite("status_label", &TelemetrySample::status_label)
        .def_readwrite("fault_active", &TelemetrySample::fault_active)
        .def("to_json", &TelemetrySample::to_json)
        .def("__repr__", [](const TelemetrySample& s) {
            return "<TelemetrySample t=" + std::to_string(s.timestamp_s) + 
                   "s rx=" + std::to_string(s.rx_power_dbm) + 
                   "dBm status=" + s.status_label + ">";
        });

    // LinkParameters
    py::class_<LinkParameters>(m, "LinkParameters")
        .def(py::init<>())
        .def_readwrite("tx_power_dbm", &LinkParameters::tx_power_dbm)
        .def_readwrite("fiber_length_km", &LinkParameters::fiber_length_km)
        .def_readwrite("attenuation_db_per_km", &LinkParameters::attenuation_db_per_km)
        .def_readwrite("other_loss_db", &LinkParameters::other_loss_db)
        .def_readwrite("dispersion_param_ps_nm_km", &LinkParameters::dispersion_param_ps_nm_km)
        .def_readwrite("wavelength_nm", &LinkParameters::wavelength_nm)
        .def_readwrite("initial_osnr_db", &LinkParameters::initial_osnr_db)
        .def_readwrite("nominal_bias_current_ma", &LinkParameters::nominal_bias_current_ma)
        .def_readwrite("nominal_temp_c", &LinkParameters::nominal_temp_c)
        .def_readwrite("rx_sensitivity_dbm", &LinkParameters::rx_sensitivity_dbm)
        .def_readwrite("rx_overload_dbm", &LinkParameters::rx_overload_dbm)
        .def_readwrite("los_threshold_dbm", &LinkParameters::los_threshold_dbm)
        .def("validate", &LinkParameters::validate)
        .def("__repr__", [](const LinkParameters& p) {
            return "<LinkParameters L=" + std::to_string(p.fiber_length_km) + 
                   "km alpha=" + std::to_string(p.attenuation_db_per_km) + 
                   "dB/km Ptx=" + std::to_string(p.tx_power_dbm) + "dBm>";
        });

    // OpticalLink
    py::class_<OpticalLink>(m, "OpticalLink")
        .def(py::init<const LinkParameters&>(), py::arg("params") = LinkParameters())
        .def("set_parameters", &OpticalLink::set_parameters)
        .def("get_parameters", &OpticalLink::get_parameters)
        .def("calculate_fiber_loss", &OpticalLink::calculate_fiber_loss)
        .def("calculate_received_power", py::overload_cast<>(&OpticalLink::calculate_received_power, py::const_))
        .def("calculate_received_power", py::overload_cast<double>(&OpticalLink::calculate_received_power, py::const_), py::arg("additional_attenuation_db"))
        .def("calculate_osnr", &OpticalLink::calculate_osnr, py::arg("total_loss"), py::arg("extra_noise_db") = 0.0)
        .def("calculate_ber", &OpticalLink::calculate_ber, py::arg("rx_power_dbm"), py::arg("osnr_db"))
        .def("calculate_q_factor", &OpticalLink::calculate_q_factor, py::arg("ber"))
        .def("step", &OpticalLink::step,
             py::arg("timestamp_s"),
             py::arg("injected_attenuation_db") = 0.0,
             py::arg("injected_noise_db") = 0.0,
             py::arg("bias_current_drift_ma") = 0.0,
             py::arg("temp_drift_c") = 0.0,
             py::arg("force_los") = false,
             py::arg("fault_name") = "NONE")
        .def("simulate_profile", &OpticalLink::simulate_profile, py::arg("num_steps") = 100, py::arg("dt_s") = 0.1)
        .def("reset_noise_generator", &OpticalLink::reset_noise_generator, py::arg("seed") = 42);

    // Fault Injectors Base
    py::class_<FaultInjector, std::shared_ptr<FaultInjector>>(m, "FaultInjector")
        .def("get_name", &FaultInjector::get_name);

    py::class_<FiberCutFault, FaultInjector, std::shared_ptr<FiberCutFault>>(m, "FiberCutFault")
        .def(py::init<double, double>(), py::arg("start_time_s") = 2.0, py::arg("duration_s") = -1.0);

    py::class_<AttenuationRampFault, FaultInjector, std::shared_ptr<AttenuationRampFault>>(m, "AttenuationRampFault")
        .def(py::init<double, double, double, double>(),
             py::arg("start_time_s") = 2.0,
             py::arg("ramp_rate_db_per_s") = 1.5,
             py::arg("max_attenuation_db") = 15.0,
             py::arg("duration_s") = -1.0);

    py::class_<ConnectorContaminationFault, FaultInjector, std::shared_ptr<ConnectorContaminationFault>>(m, "ConnectorContaminationFault")
        .def(py::init<double, double, double>(),
             py::arg("start_time_s") = 2.0,
             py::arg("added_loss_db") = 6.0,
             py::arg("duration_s") = -1.0);

    py::class_<LaserAgingFault, FaultInjector, std::shared_ptr<LaserAgingFault>>(m, "LaserAgingFault")
        .def(py::init<double, double, double, double>(),
             py::arg("start_time_s") = 2.0,
             py::arg("current_drift_rate_ma_s") = 5.0,
             py::arg("max_drift_ma") = 30.0,
             py::arg("temp_drift_rate_c_s") = 2.0);

    py::class_<NoiseBurstFault, FaultInjector, std::shared_ptr<NoiseBurstFault>>(m, "NoiseBurstFault")
        .def(py::init<double, double, double>(),
             py::arg("start_time_s") = 2.0,
             py::arg("added_noise_db") = 12.0,
             py::arg("duration_s") = -1.0);

    py::class_<TransientDropoutFault, FaultInjector, std::shared_ptr<TransientDropoutFault>>(m, "TransientDropoutFault")
        .def(py::init<double, double, double, double>(),
             py::arg("start_time_s") = 1.0,
             py::arg("period_s") = 1.0,
             py::arg("pulse_width_s") = 0.2,
             py::arg("duration_s") = -1.0);

    // FaultEngine
    py::class_<FaultEngine>(m, "FaultEngine")
        .def(py::init<>())
        .def("add_injector", &FaultEngine::add_injector)
        .def("clear_injectors", &FaultEngine::clear_injectors)
        .def("run_simulation", &FaultEngine::run_simulation,
             py::arg("link"),
             py::arg("num_steps") = 100,
             py::arg("dt_s") = 0.1);
}
