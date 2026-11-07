#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================="
echo "OptiTest AI: Building C++ Physics Core"
echo "========================================="

mkdir -p "$PROJECT_ROOT/build"
cd "$PROJECT_ROOT/build"
cmake ..
make -j"$(nproc)"

echo ""
echo "Running standalone C++ test binary..."
./optitest_cpp_tests

echo ""
echo "Copying compiled shared module to project root..."
cp -f "$PROJECT_ROOT/build/optitest_core"*.so "$PROJECT_ROOT/"
cp -f "$PROJECT_ROOT/build/optitest_core"*.so "$PROJECT_ROOT/optitest/"

echo "Build and C++ verification completed successfully!"
