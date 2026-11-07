cmake_minimum_required(VERSION 3.16)
project(OptiTestAI VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(include)

# Physics core library
add_library(optitest_physics STATIC
    src/optical_link.cpp
    src/fault_injection.cpp
)

# Standalone C++ verification test binary
add_executable(optitest_cpp_tests
    tests/test_cpp_link.cpp
)
target_link_libraries(optitest_cpp_tests PRIVATE optitest_physics)

# Optional Python bindings via pybind11
find_package(pybind11 QUIET)
if (pybind11_FOUND)
    message(STATUS "pybind11 found via CMake package")
    pybind11_add_module(optitest_core src/bindings.cpp)
    target_link_libraries(optitest_core PRIVATE optitest_physics)
else()
    # Check if pybind11 is available in python environment
    execute_process(
        COMMAND python3 -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if (pybind11_DIR)
        message(STATUS "pybind11 found at: ${pybind11_DIR}")
        find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})
        pybind11_add_module(optitest_core src/bindings.cpp)
        target_link_libraries(optitest_core PRIVATE optitest_physics)
    else()
        message(WARNING "pybind11 not found; skipping Python native module build in CMake")
    endif()
endif()
