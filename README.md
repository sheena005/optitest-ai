# OptiTest AI: An AI-Assisted Optical Link Simulation, Fault-Injection, and Software Validation Framework

[![CI](https://github.com/sheena005/optitest-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/sheena005/optitest-ai/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B17)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Pytest](https://img.shields.io/badge/Tests-25%20Passed%20(100%25)-brightgreen.svg)](https://docs.pytest.org/)
[![Coverage](https://img.shields.io/badge/Coverage-75%25-brightgreen.svg)](https://coverage.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

> 📦 **GitHub Repository:** [https://github.com/sheena005/optitest-ai](https://github.com/sheena005/optitest-ai)  
> 🌐 **Interactive Demo:** Deployable to [Streamlit Community Cloud](https://share.streamlit.io/) in 1 click or runnable locally on `localhost:8501`.

OptiTest AI is an end-to-end optical engineering framework designed to simulate optical transmission links, inject controlled physical faults, automate software validation against ITU-T/IEEE standards, detect anomalous behavior via machine learning, and produce grounded root-cause analysis (RCA) reports with Generative AI.

---

## 📌 Problem Statement

Optical transport products must operate reliably under nominal conditions and gracefully handle physical degradations (fiber bends, dirty connectors, laser diode aging, ASE amplifier noise ripple, and fiber cuts). Validating optical management software on physical line cards and ROADMs is expensive, difficult to reproduce deterministically, and hazardous to physical transceivers.

**OptiTest AI solves this by delivering:**
1. **High-Performance Physics Simulation:** Deterministic C++17 optical power budget, OSNR, Q-factor, and erfc-based Bit Error Rate (BER) calculations.
2. **Modular Fault-Injection Engine:** Decoupled perturbation model simulating fiber cuts (LOS), progressive macro-bends, ferrule contamination, laser diode runaway, and transient dropouts with recovery cycles.
3. **Automated Rule Validation Harness:** Pytest-driven test engine implementing ITU-T G.692 and IEEE 802.3 optical thresholds with debounce state filtering.
4. **Rigorous Dual ML Anomaly Detection:** Group-split verified Random Forest failure classifier (99.63% test accuracy on unseen runs) and unsupervised Isolation Forest anomaly scoring.
5. **AI Diagnostic & RCA Advisor:** Grounded Root Cause Analysis via Google Gemini API (with deterministic optical expert heuristics for offline execution).
6. **Interactive Dashboard:** Complete 5-tab Streamlit engineering workbench.

---

## 🏗️ System Architecture

```
                    User Configuration / Streamlit Web UI
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
                |  - Pytest Automation Suite (25)  |
                |  - Boundary, Invalid, Regression |
                |  - Benchmark & Stress Evaluator  |
                +----------------------------------+
                                 |
                                 v
                +----------------------------------+
                |  Interactive Streamlit Dashboard |
                |  + AI-Assisted Root Cause (RCA)  |
                +----------------------------------+
```

---

## 🔬 Physical Modeling Assumptions & Limitations

To maintain engineering transparency, the physical layer model is scoped as follows:

### 1. Attenuation & Power Budget
\[
P_{\mathrm{rx}} = P_{\mathrm{tx}} - (\alpha L + L_{\mathrm{conn}} + L_{\mathrm{splice}} + L_{\mathrm{other}})
\]
- $P_{\mathrm{rx}}$: Received optical power in dBm.
- $P_{\mathrm{tx}}$: Transmit launch power in dBm (nominal $0.0$ dBm).
- $\alpha$: Fiber attenuation coefficient ($0.20$ dB/km at $1550$ nm SMF-28, $0.35$ dB/km at $1310$ nm O-band).
- $L$: Fiber length in km.
- $L_{\mathrm{other}}$: Fixed link insertion losses (connectors, splices, patch panels).

### 2. Optical Signal-to-Noise Ratio (OSNR)
\[
\mathrm{OSNR}_{\mathrm{rx}} = \mathrm{OSNR}_{\mathrm{tx}} - (0.35 \times L_{\mathrm{total}}) - \Delta_{\mathrm{noise}}
\]
- Simplified linear loss approximation representing accumulated Amplified Spontaneous Emission (ASE) noise over unamplified spans.

### 3. Bit Error Rate (BER) & Q-Factor
The nominal Gaussian optical channel Q-factor relates to OSNR:
\[
Q_{\mathrm{nominal}} = 10^{(\mathrm{OSNR}_{\mathrm{dB}} - 12)/20}
\]
When $P_{\mathrm{rx}}$ drops below receiver sensitivity ($P_{\mathrm{sens}} = -24$ dBm):
\[
Q_{\mathrm{eff}} = \frac{Q_{\mathrm{nominal}}}{1 + 0.6 \cdot |P_{\mathrm{sens}} - P_{\mathrm{rx}}|^{1.8}}
\]
Bit Error Rate computed via the complementary error function:
\[
\mathrm{BER} = \frac{1}{2} \mathrm{erfc}\left(\frac{Q_{\mathrm{eff}}}{\sqrt{2}}\right)
\]
When $P_{\mathrm{rx}} \le -28$ dBm (LOS threshold), $\mathrm{BER}$ clamps to $0.5$ (total loss of frame synchronization).

### 4. Explicit Physical Simplifications & Omissions
- **Kerr Non-Linearities**: Does not compute non-linear Schrödinger equation (NLSE) phase shifts (Self-Phase Modulation [SPM], Cross-Phase Modulation [XPM], Four-Wave Mixing [FWM]).
- **Dispersion Broadening**: Chromatic dispersion (CD) and Polarization Mode Dispersion (PMD) are modeled as linear signal degradation penalties rather than full pulse envelope convolutions.
- **Standards Scope**: ITU-T G.692 and IEEE 802.3 are applied specifically to the **alarm threshold validation logic** (power boundaries, BER limits, debouncing state machines), not as full optical transport layer emulators.

---

## 📊 Model Evaluation, Benchmarking & Robustness

### Addressing the Synthetic Evaluation Critique
Reporting high accuracy on synthetic data often risks **temporal leakage** (adjacent steps in the same simulation run leaking into train and test sets). OptiTest AI addresses this with strict engineering validation:

1. **Run-Level Group Splitting**: Using `GroupShuffleSplit` on `run_id`, an entire simulation run belongs exclusively to either the training set (108 runs, 6,480 samples) or the hold-out test set (36 runs, 2,160 samples). **Zero temporal or scenario leakage exists between splits.**
2. **Baselines Benchmarking**: Evaluated against a trivial `DummyClassifier` (majority class) and a deterministic `Rule-Based Threshold Baseline`.

### Benchmark Comparison Matrix (Hold-Out Runs)

| Model / Architecture | Test Accuracy | Macro F1 | Role |
|:---|:---:|:---:|:---|
| **OptiTest Dual ML (RF + IsoForest)** | **99.63%** | **99.66%** | Multi-channel temporal pattern recognition |
| Deterministic Rule-Based Baseline | 64.95% | 42.76% | Static optical thresholds |
| Dummy (Majority Class) Baseline | 48.15% | 10.83% | Trivial floor |

### Out-of-Distribution (OOD) Stress Test
To rigorously identify domain boundaries, the model was evaluated against an OOD test suite featuring:
- Ultra-long fiber spans (105–140 km) unseen during training (10–80 km).
- Extreme optical launch powers ($-5.0$ dBm starved, $+6.0$ dBm high power).
- Elevated $3\times$ sensor jitter and non-linear cubic degradation ramps.

**Result**: Under these extreme conditions, supervised classifier accuracy dropped to **19.72%** (Macro F1: 18.81%).  
*Engineering takeaway*: This degradation transparently demonstrates the physical domain boundaries of supervised classification on tabular data, validating why OptiTest AI pairs Random Forest with an **unsupervised Isolation Forest** to flag zero-day anomalies without requiring prior labels.

### Sensor Noise Jitter Resilience Sweep
Evaluated against Gaussian measurement noise applied to optical power, bias current, and temperature:

| Sensor Noise $\sigma$ (dB) | Test Accuracy | Macro F1 | Degradation |
|:---:|:---:|:---:|:---:|
| **0.00 dB** | 99.63% | 99.66% | Baseline |
| **0.05 dB** | 99.58% | 99.62% | 0.05% |
| **0.10 dB** | 98.15% | 98.39% | 1.49% |
| **0.20 dB** | 94.26% | 95.40% | 5.39% |
| **0.35 dB** | 90.83% | 92.98% | 8.83% |
| **0.50 dB** | 88.43% | 91.39% | 11.25% |

The model degrades monotonically and gracefully, retaining >90% accuracy even under severe 0.35 dB measurement jitter.

### Empirical Precursor Lead-Time Quantification

| Scenario | Fault Start | Rule Alarm Trigger | ML Anomaly Trigger | Lead Time |
|:---|:---:|:---:|:---:|:---:|
| **Laser Diode Aging** | Step 20 ($t=2.0$s) | Step 71 ($t=7.1$s) | **Step 39 ($t=3.9$s)** | **+32 steps (3.2 seconds earlier)** |
| **Excessive Attenuation Ramp** | Step 20 ($t=2.0$s) | Step 80 ($t=8.0$s) | Step 80 ($t=8.0$s) | Synchronous |
| **Connector Contamination** | Step 20 ($t=2.0$s) | Step 20 ($t=2.0$s) | Step 20 ($t=2.0$s) | Synchronous |
| **Amplifier Optical Noise (ASE)** | Step 20 ($t=2.0$s) | Step 20 ($t=2.0$s) | Step 20 ($t=2.0$s) | Synchronous |

*Finding*: In progressive degradation scenarios (like laser diode aging), the ML detector flags anomalies **3.2 seconds before static ITU-T alarms** by recognizing subtle multivariate drifts in bias current and temperature before any hard threshold is breached.

---

## 💡 Technical Deep-Dive & Key Design Decisions

### 1. Derivation of BER from Gaussian Q-Factor
In optical direct-detection receivers, the decision circuit samples received photocurrent in the presence of Gaussian thermal and shot noise:
\[
Q = \frac{I_1 - I_0}{\sigma_1 + \sigma_0}
\]
Assuming symmetric noise variances ($\sigma_0 \approx \sigma_1 = \sigma$) and optimal decision threshold $I_D = (I_1 + I_0)/2$:
\[
P_e = \frac{1}{\sqrt{2\pi}\sigma} \int_{I_D}^{\infty} e^{-(I - I_0)^2 / (2\sigma^2)} dI = \frac{1}{2} \mathrm{erfc}\left(\frac{Q}{\sqrt{2}}\right)
\]
OptiTest AI computes this via `std::erfc` in C++17, avoiding numerical lookup tables and providing continuous gradients down to machine precision ($10^{-15}$).

### 2. Dual ML Architecture (Random Forest + Isolation Forest)
- **Supervised Random Forest**: Excels on multi-variate tabular telemetry with complex decision boundaries, isolating known failure modes (`FIBER_CUT`, `LASER_AGING`, `DIRTY_CONNECTOR`).
- **Unsupervised Isolation Forest**: Isolates anomalies by randomly partitioning features; anomalous points require fewer splits. This enables zero-day anomaly detection on novel perturbations that do not match training labels.

### 3. Alarm Debouncing State Machine & Hysteresis
Raw optical telemetry exhibits transient dips caused by polarization fluctuations or vibrational micro-bends. Firing alarms instantaneously causes alarm flapping and network management fatigue:
- **Consecutive Sample Integration**: An alarm condition must persist for $N=3$ consecutive samples before transitioning `HEALTHY` $\to$ `DEGRADED` or `LINK_DOWN`.
- **Hysteresis Band**: Recovery requires optical power to exceed the clear threshold by $+1.0$ dB above the trip threshold, preventing oscillation around the boundary.

### 4. Why C++17 for the Physics Simulation Core?
1. **Deterministic Execution**: Optical simulation runs at microsecond latency without Python garbage collection pauses, enabling real-time emulation of 100+ optical spans.
2. **Memory Layout & Cache Locality**: Sequential contiguous memory for telemetry structures (`TelemetrySample`) optimizes CPU L1/L2 cache prefetching.
3. **GIL Bypass**: pybind11 releases the Python Global Interpreter Lock during bulk Monte Carlo simulation sweeps.
4. **Resilient Fallback**: A pure Python implementation mirrors the C++ API, ensuring portability when native compilation tools are unavailable.

### 5. Anomaly Threshold (0.55) Derivation
The Isolation Forest decision function yields scores $s \in [-0.4, 0.3]$. We map this to $[0, 1]$ via a calibrated sigmoid:
\[
S_{\mathrm{anomaly}} = \frac{1}{1 + e^{5 \cdot s}}
\]
At $s = 0$ (the boundary between inlier and outlier), $S_{\mathrm{anomaly}} = 0.50$. Setting the alarm threshold at $0.55$ introduces a slight buffer to eliminate false alarms from nominal noise while preserving rapid trigger latency.

---

## 🚀 Quick Start Guide

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/sheena005/optitest-ai.git
cd optitest-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Build C++ Core and Python Extension
```bash
chmod +x scripts/*.sh
./scripts/build.sh
pip install --no-build-isolation -e .
```

### 3. Run Full Verification Suite
```bash
./scripts/run_tests.sh
```

### 4. Train Models & Generate Rigorous Evaluation Metrics
```bash
python3 scripts/train_models.py
```

### 5. Launch Interactive Dashboard
```bash
./scripts/run_dashboard.sh
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Test Verification Suite

OptiTest AI contains **25 comprehensive tests** (22 Python tests + 3 Native C++ tests) with a **100% pass rate**:

| Test Suite | Focus Area | Test Count | Status |
|:---|:---|:---:|:---:|
| `test_cpp_link.cpp` | Native C++ power budget, parameter validation, fault injection | 3 | **PASS** |
| `test_simulator.py` | Power budget, boundary lengths ($L=0$), negative inputs, BER/Q relationship, C++/Python parity | 7 | **PASS** |
| `test_faults.py` | Fiber cut LOS, progressive attenuation ramp, laser aging drift, fault recovery cycle | 4 | **PASS** |
| `test_rules.py` | ITU-T thresholds, debounce state machine filtering, automated runner | 3 | **PASS** |
| `test_ml_detector.py` | RF inference, Isolation Forest scoring, lead time, group isolation, baselines & noise | 5 | **PASS** |
| `test_ai_advisor.py` | Grounded RCA generation, field engineering action items, hallucination prevention | 3 | **PASS** |
| **Total** | **Full Verification Suite** | **25** | **100% PASS** |

---

## 📁 Repository Structure

```
optitest-ai/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI workflow (C++ & Python)
├── CMakeLists.txt              # Modern CMake build configuration
├── LICENSE                     # MIT License
├── pyproject.toml              # Packaging and dependency declarations
├── setup.py                    # Pybind11Extension build script
├── requirements.txt            # Python dependencies
├── packages.txt                # System packages for Streamlit Cloud
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
│   ├── test_ml_detector.py     # ML inference, group isolation, and baselines
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
│   │   ├── dataset.py          # Synthetic dataset generation & OOD generator
│   │   ├── detector.py         # Isolation Forest & Random Forest models
│   │   └── evaluate.py         # Group split, baselines, noise sweep & lead time
│   └── ai_advisor/
│       └── diagnostic.py       # Gemini API & optical expert heuristic RCA engine
├── scripts/
│   ├── build.sh                # CMake and pybind11 compilation script
│   ├── run_tests.sh            # Complete C++ and Pytest verification runner
│   ├── train_models.py         # Group-level training, benchmark & OOD evaluator
│   └── run_dashboard.sh        # Streamlit dashboard launcher
└── models/
    ├── optical_detector.joblib # Serialized pre-trained ML models (Group-split)
    └── eval_metrics.json       # Group accuracy, baseline comparison & noise sweep
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
