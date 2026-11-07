from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys

__version__ = "1.0.0"

ext_modules = [
    Pybind11Extension(
        "optitest_core",
        [
            "src/bindings.cpp",
            "src/optical_link.cpp",
            "src/fault_injection.cpp",
        ],
        include_dirs=["include"],
        cxx_std=17,
    ),
]

setup(
    name="optitest",
    version=__version__,
    author="OptiTest AI Team",
    description="AI-Assisted Optical Link Simulation, Fault-Injection, and Software Validation Framework",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)
