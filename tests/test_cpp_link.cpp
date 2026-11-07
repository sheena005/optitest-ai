#include "optical_link.hpp"
#include "fault_injection.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

void test_basic_power_calculation() {
    std::cout << "[TEST] Running test_basic_power_calculation..." << std::endl;
    optitest::LinkParameters params;
    params.tx_power_dbm = 0.0;
    params.fiber_length_km = 40.0;
    params.attenuation_db_per_km = 0.20;
    params.other_loss_db = 2.0;

    optitest::OpticalLink link(params);
    double fiber_loss = link.calculate_fiber_loss();
    assert(std::fabs(fiber_loss - 8.0) < 1e-6);

    double rx_power = link.calculate_received_power();
    // 0 - (0.2 * 40) - 2 = -10.0 dBm
    assert(std::fabs(rx_power - (-10.0)) < 1e-6);

    double rx_power_added = link.calculate_received_power(5.0);
    assert(std::fabs(rx_power_added - (-15.0)) < 1e-6);

    std::cout << "  -> PASS (Rx power = " << rx_power << " dBm)" << std::endl;
}

void test_invalid_parameters() {
    std::cout << "[TEST] Running test_invalid_parameters..." << std::endl;
    optitest::LinkParameters params;
    params.fiber_length_km = -10.0;
    bool caught = false;
    try {
        optitest::OpticalLink link(params);
    } catch (const std::invalid_argument& e) {
        caught = true;
    }
    assert(caught);
    std::cout << "  -> PASS (Successfully caught negative length exception)" << std::endl;
}

void test_fault_injection_engine() {
    std::cout << "[TEST] Running test_fault_injection_engine..." << std::endl;
    optitest::LinkParameters params;
    params.tx_power_dbm = 0.0;
    params.fiber_length_km = 40.0;
    params.attenuation_db_per_km = 0.20;
    params.other_loss_db = 2.0;

    optitest::OpticalLink link(params);
    optitest::FaultEngine engine;

    // Add Fiber Cut at t = 1.0s with duration 1.0s (recovers at t = 2.0s)
    engine.add_injector(std::make_shared<optitest::FiberCutFault>(1.0, 1.0));

    auto history = engine.run_simulation(link, 30, 0.1); // 0.0s to 2.9s
    assert(history.size() == 30);

    // At t = 0.5s: HEALTHY
    assert(history[5].status_label == "HEALTHY");
    assert(!history[5].los_alarm);

    // At t = 1.5s: LINK_DOWN
    assert(history[15].status_label == "LINK_DOWN");
    assert(history[15].los_alarm);
    assert(history[15].rx_power_dbm <= -28.0);

    // At t = 2.5s: RECOVERED (HEALTHY)
    assert(history[25].status_label == "HEALTHY");
    assert(!history[25].los_alarm);

    std::cout << "  -> PASS (Fiber cut injected and recovered cleanly)" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "OptiTest AI: Native C++ Verification Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        test_basic_power_calculation();
        test_invalid_parameters();
        test_fault_injection_engine();
        std::cout << "\nALL C++ TESTS PASSED SUCCESSFULLY!" << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "C++ Test Failed with error: " << ex.what() << std::endl;
        return 1;
    }
}
