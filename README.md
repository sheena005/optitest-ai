# OptiTest AI: An AI-Assisted Optical Link Simulation, Fault-Injection, and Software Validation Framework

[![GitHub Repo](https://img.shields.io/badge/GitHub-sheena005%2Foptitest--ai-181717.svg?logo=github)](https://github.com/sheena005/optitest-ai)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B17)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Pytest](https://img.shields.io/badge/Tests-20%20Passed%20(100%25)-brightgreen.svg)](https://docs.pytest.org/)
[![ML Accuracy](https://img.shields.io/badge/ML%20Accuracy-99.55%25-success.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

> 🌐 **Live Application URL:** [https://drhji-2409-40f4-201b-b2f2-60d3-12a2-68b0-fa42.run.pinggy-free.link](https://drhji-2409-40f4-201b-b2f2-60d3-12a2-68b0-fa42.run.pinggy-free.link)  
> 📦 **GitHub Repository:** [https://github.com/sheena005/optitest-ai](https://github.com/sheena005/optitest-ai)

OptiTest AI is an engineering framework designed to simulate optical transmission links, inject controlled physical faults, automate software validation against ITU-T/IEEE standards, detect anomalous behavior via machine learning, and produce grounded root cause analysis (RCA) reports with Generative AI.

---

## 📌 Problem Statement

Optical transport products must operate reliably under nominal conditions and gracefully handle physical degradations (fiber bends, dirty connectors, laser aging, ASE amplifier noise, and fiber cuts). Validating optical management software on real equipment can be expensive, difficult to reproduce, or dangerous to physical transceivers. 

**OptiTest AI solves this by providing:**
1. **High-Performance Physics Simulation:** Deterministic C++ optical power budget, OSNR, Q-factor, and erfc-based Bit Error Rate (BER) calculations.
2. **Modular Fault-Injection Engine:** Decoupled perturbation model simulating fiber cuts, macro-bends, ferrule contamination, laser diode runaway, and transient dropouts with recovery cycles.
3. **Automated Rule Validation Harness:** Pytest-driven test engine implementing ITU-T G.692 and IEEE 802.3 optical thresholds with debounce state filtering.
4. **AI-Based Anomaly Detection:** Supervised Random Forest failure classifier (99.55% accuracy) and unsupervised Isolation Forest anomaly scoring.
5. **AI Diagnostic & RCA Advisor:** Grounded Root Cause Analysis via Google Gemini API (with deterministic optical expert heuristics for offline execution).
6. **Interactive Dashboard:** Complete 5-tab Streamlit engineering workbench.

---

## 🏗️ System Architecture

```
                    User Configuration / Web UI
                                |
                                v
               +----------------------------------+
               |      Optical Link Simulator      |
               |          (C++17 Core)            |
               |  - Power Budget: P_rx = P_tx -   |
               |    alpha*L - L_other             |
               |  - OSNR, Q-Factor, BER Modeling  |
               |  - Transceiver Telemetry Engine  |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |     Fault Injection Engine       |
               |  - Fiber Cut (LOS)               |
               |  - Progressive Attenuation/Bend  |
               |  - Connector Contamination       |
               |  - Laser Aging / Bias Drift      |
               |  - ASE Optical Noise Ripple      |
               |  - Transient Dropouts & Recovery |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |     Telemetry & Test Events      |
               +----------------------------------+
                                |
                    +-----------+-----------+
                    |                       |
                    v                       v
      +-------------------------+ +-------------------------+
      |  Rule-Based Test Engine | |   ML Anomaly Detector   |
      | - ITU-T / IEEE Limits   | | - Isolation Forest      |
      | - Alarm State Machine   | | - Random Forest Classif.|
      | - Debounce & Hysteresis | | - Telemetry Feature Eng.|
      +-------------------------+ +-------------------------+
                    |                       |
                    +-----------+-----------+
                                |
                                v
               +----------------------------------+
               |    Python Test Harness & Matrix  |
               |  - Pytest Automation Suite       |
               |  - Boundary, Invalid, Regression |
               |  - Rule vs ML Comparison Matrix  |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |  Interactive Streamlit Dashboard |
               |  + AI-Assisted Root Cause (RCA)  |
               +----------------------------------+
```

---

## 🔬 Optical Physics Model

### 1. Received Optical Power
\[
P_{\mathrm{rx}} = P_{\mathrm{tx}} - (\alpha L + L_{\mathrm{conn}} + L_{\mathrm{splice}} + L_{\mathrm{other}})
\]
- $P_{\mathrm{rx}}$: Received optical power in dBm
- $P_{\mathrm{tx}}$: Transmit launch power in dBm (nominal $0.0$ dBm)
- $\alpha$: Fiber attenuation coefficient (e.g. $0.20$ dB/km at $1550$ nm SMF-28, $0.35$ dB/km at $1310$ nm)
- $L$: Fiber length in km
- $L_{\mathrm{other}}$: Fixed link insertion losses (connectors, splices, patch panels)

### 2. Optical Signal-to-Noise Ratio (OSNR)
\[
\mathrm{OSNR}_{\mathrm{rx}} = \mathrm{OSNR}_{\mathrm{tx}} - (0.35 \times L_{\mathrm{total}}) - \Delta_{\mathrm{noise}}
\]

### 3. Bit Error Rate (BER) & Q-Factor
Nominal Gaussian optical channel Q-factor:
\[
Q_{\mathrm{nominal}} = 10^{(\mathrm{OSNR}_{\mathrm{dB}} - 12)/20}
\]
When $P_{\mathrm{rx}}$ drops below receiver sensitivity ($P_{\mathrm{sens}} = -24$ dBm):
\[
Q_{\mathrm{eff}} = \frac{Q_{\mathrm{nominal}}}{1 + 0.6 \cdot |P_{\mathrm{sens}} - P_{\mathrm{rx}}|^{1.8}}
\]
Bit Error Rate computed via complementary error function:
\[
\mathrm{BER} = \frac{1}{2} \mathrm{erfc}\left(\frac{Q_{\mathrm{eff}}}{\sqrt{2}}\right)
\]
When $P_{\mathrm{rx}} \le -28$ dBm (LOS threshold), $\mathrm{BER}$ clamps to $0.5$ (loss of frame synchronization).

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```bash
cd /home/sheena/.gemini/antigravity/scratch/optitest-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Build C++ Core and Python Bindings
```bash
./scripts/build.sh
pip install --no-build-isolation -e .
```

### 3. Run Verification Suite (C++ and Pytest)
```bash
./scripts/run_tests.sh
```

### 4. Train / Re-train ML Anomaly Models
```bash
python3 scripts/train_models.py
```

### 5. Launch Interactive Dashboard
```bash
./scripts/run_dashboard.sh
```
Open browser at `http://localhost:8501`.

---

## 🧪 Test Verification Suite

OptiTest AI includes 20 comprehensive unit, boundary, fault-injection, and regression tests:

| Test File | Focus Area | Status |
|:---|:---|:---:|
| `test_simulator.py` | Optical power budget, boundary lengths ($L=0$), invalid inputs, parity between C++ and Python | **PASS** |
| `test_faults.py` | Fiber cut LOS, progressive attenuation ramp, laser aging bias drift, fault recovery cycle | **PASS** |
| `test_rules.py` | ITU-T thresholds, debounce state filtering, full automated runner suite | **PASS** |
| `test_ml_detector.py` | Random Forest inference, Isolation Forest anomaly scoring, Rule vs ML comparison | **PASS** |
| `test_ai_advisor.py` | Telemetry-grounded RCA generation, field engineering action items, hallucination prevention | **PASS** |
| `test_cpp_link.cpp` | Standalone native C++ binary tests | **PASS** |

---

## 🤖 Rule-Based vs AI-Based Anomaly Detection

| Dimension | Deterministic Rule-Based Validator | Machine Learning Anomaly Detector |
|:---|:---|:---|
| **Detection Mechanism** | Static ITU-T thresholds ($P_{\mathrm{rx}} < -22$ dBm, $\text{BER} > 10^{-4}$) | Multi-variate gradient & rolling variance pattern analysis |
| **Early Warning Precursor** | Latent; waits until threshold is physically breached | **Proactive; flags anomaly up to 10 steps earlier** (e.g. subtle laser bias current drift) |
| **Root Cause Attribution** | Limited to specific threshold flags (`ALARM_LOS`) | Classifies failure modes (`FIBER_CUT`, `CONNECTOR_DIRT`, `LASER_AGING`, `ASE_NOISE`) |
| **False Positives** | Low (rigid limits) | Controlled via threshold tuning ($>0.55$ anomaly score) |
| **Zero-Day Faults** | Fails to detect novel unmodeled degradation | Unsupervised Isolation Forest flags anomalies without prior labels |

---

## 📁 Repository Structure

```
optitest-ai/
├── CMakeLists.txt              # Modern CMake build configuration
├── pyproject.toml              # Packaging and dependency declarations
├── setup.py                    # Pybind11Extension build script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── app.py                      # 5-Tab Streamlit Interactive Dashboard
├── include/
│   ├── telemetry.hpp           # Telemetry sample struct & JSON serializer
│   ├── optical_link.hpp        # Physical link equations and OpticalLink class
│   └── fault_injection.hpp     # Decoupled fault injectors & FaultEngine
├── src/
│   ├── optical_link.cpp        # C++ physics implementation
│   ├── fault_injection.cpp     # C++ fault implementations
│   └── bindings.cpp            # Pybind11 C++/Python bindings
├── tests/
│   ├── test_cpp_link.cpp       # Native C++ binary test program
│   ├── test_simulator.py       # Physics, boundary, and parity tests
│   ├── test_faults.py          # Fault scenario and recovery tests
│   ├── test_rules.py           # ITU-T rules and debounce tests
│   ├── test_ml_detector.py     # ML inference and comparison tests
│   └── test_ai_advisor.py      # Diagnostic grounding tests
├── optitest/
│   ├── core/
│   │   ├── models.py           # Data models (LinkParameters, TelemetrySample)
│   │   └── simulator.py        # Unified C++ wrapper + Python fallback
│   ├── faults/
│   │   └── scenarios.py        # Standardized optical test scenarios
│   ├── harness/
│   │   ├── rules.py            # ITU-T G.692 rule validation engine
│   │   └── runner.py           # Automated test harness runner
│   ├── ml/
│   │   ├── dataset.py          # Synthetic dataset generation & feature engineering
│   │   ├── detector.py         # Isolation Forest & Random Forest models
│   │   └── evaluate.py         # Comparative evaluation matrix
│   └── ai_advisor/
│       └── diagnostic.py       # Gemini API & optical expert heuristic RCA engine
├── scripts/
│   ├── build.sh                # CMake and pybind11 compilation script
│   ├── run_tests.sh            # Complete C++ and Pytest verification runner
│   ├── train_models.py         # Synthetic data generation and model trainer
│   └── run_dashboard.sh        # Streamlit dashboard launcher
└── models/
    ├── optical_detector.joblib # Serialized pre-trained ML models (99.55% Acc)
    └── eval_metrics.json       # Hold-out test accuracy and feature importances
```
