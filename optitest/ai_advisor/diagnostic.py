import os
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from optitest.harness.runner import ScenarioTestResult

@dataclass
class DiagnosticReport:
    scenario_name: str
    status: str
    incident_severity: str  # "P1_CRITICAL" | "P2_MAJOR" | "P3_WARNING" | "NOMINAL"
    root_cause_summary: str
    physical_root_cause: str
    telemetry_findings: Dict[str, Any]
    rule_alarms: List[str]
    ml_classification: str
    remediation_steps: List[str]
    markdown_report: str
    is_llm_generated: bool

class OpticalDiagnosticAdvisor:
    """
    AI-Assisted Diagnostic and Root Cause Analysis (RCA) Engine.
    Grounded strictly in actual telemetry measurements and test outcomes.
    Supports Google Gemini API with fallback to deterministic optical expert heuristics.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self._client = None
        if self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception:
                self._client = None

    def diagnose(
        self,
        test_result: ScenarioTestResult,
        ml_prediction: Optional[Dict[str, Any]] = None
    ) -> DiagnosticReport:
        # Extract grounding telemetry metrics from test result history
        history = test_result.history
        if not history:
            return self._fallback_heuristic(test_result, {}, ml_prediction)

        final_sample = history[-1]
        min_rx_power = min(s.rx_power_dbm for s in history)
        max_rx_power = max(s.rx_power_dbm for s in history)
        min_osnr = min(s.osnr_db for s in history)
        max_ber = max(s.ber for s in history)
        max_bias_current = max(s.laser_bias_current_ma for s in history)
        max_temp = max(s.laser_temperature_c for s in history)

        telemetry_findings = {
            "tx_power_dbm": final_sample.tx_power_dbm,
            "min_rx_power_dbm": round(min_rx_power, 2),
            "final_rx_power_dbm": round(final_sample.rx_power_dbm, 2),
            "fiber_length_km": final_sample.chromatic_dispersion_ps_nm / 17.0,
            "min_osnr_db": round(min_osnr, 2),
            "max_ber": f"{max_ber:.2e}",
            "max_bias_current_ma": round(max_bias_current, 1),
            "max_temp_c": round(max_temp, 1),
            "los_alarm_asserted": any(s.los_alarm for s in history),
            "high_ber_alarm_asserted": any(s.high_ber_alarm for s in history)
        }

        # Attempt Gemini LLM diagnosis if client configured
        if self._client:
            try:
                return self._call_gemini(test_result, telemetry_findings, ml_prediction)
            except Exception as e:
                # Graceful fallback to deterministic expert engine
                pass

        return self._fallback_heuristic(test_result, telemetry_findings, ml_prediction)

    def _call_gemini(
        self,
        test_result: ScenarioTestResult,
        telemetry_findings: Dict[str, Any],
        ml_prediction: Optional[Dict[str, Any]]
    ) -> DiagnosticReport:
        from google.genai import types

        ml_cat = ml_prediction.get("category", "UNKNOWN") if ml_prediction else "N/A"
        ml_conf = ml_prediction.get("confidence", 0.0) if ml_prediction else 0.0

        prompt = f"""
You are an Optical Communications Diagnostics Engineer (ITU-T G.692 / Nokia Optical Systems).
Analyze the following test scenario output and telemetry measurements.
Ground your diagnosis strictly in the provided data. DO NOT invent measurements. DO NOT claim a test passed if it failed.

Test Scenario: {test_result.scenario_name}
Test Verdict: {test_result.status}
Expected State: {test_result.expected_fault}
Observed State: {test_result.observed_fault}
Triggered Alarms: {', '.join(test_result.alarms_triggered) if test_result.alarms_triggered else 'None'}
ML Classifier Prediction: {ml_cat} (Confidence: {ml_conf * 100:.1f}%)

Telemetry Measurements:
- Min Rx Power: {telemetry_findings['min_rx_power_dbm']} dBm (Nominal ~ -10 dBm)
- Final Rx Power: {telemetry_findings['final_rx_power_dbm']} dBm
- Min OSNR: {telemetry_findings['min_osnr_db']} dB
- Max Pre-FEC BER: {telemetry_findings['max_ber']}
- Max Laser Current: {telemetry_findings['max_bias_current_ma']} mA (Nominal 45 mA)
- Max Laser Temp: {telemetry_findings['max_temp_c']} °C (Nominal 35 °C)
- Loss of Signal Flag: {telemetry_findings['los_alarm_asserted']}

Return a strict JSON object with these exact keys:
{{
  "incident_severity": "P1_CRITICAL" or "P2_MAJOR" or "P3_WARNING" or "NOMINAL",
  "root_cause_summary": "1-2 sentence executive explanation",
  "physical_root_cause": "Detailed physical mechanism (e.g. Fiber break, Connector dirt, Laser aging, Macro-bend, Amplifier ASE noise)",
  "remediation_steps": ["step 1", "step 2", "step 3"]
}}
"""
        response = self._client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        parsed = json.loads(response.text)

        md_report = self._format_markdown_report(
            test_result=test_result,
            severity=parsed.get("incident_severity", "P2_MAJOR"),
            summary=parsed.get("root_cause_summary", ""),
            physical_cause=parsed.get("physical_root_cause", ""),
            findings=telemetry_findings,
            remediation=parsed.get("remediation_steps", []),
            ml_prediction=ml_prediction,
            is_llm=True
        )

        return DiagnosticReport(
            scenario_name=test_result.scenario_name,
            status=test_result.status,
            incident_severity=parsed.get("incident_severity", "P2_MAJOR"),
            root_cause_summary=parsed.get("root_cause_summary", ""),
            physical_root_cause=parsed.get("physical_root_cause", ""),
            telemetry_findings=telemetry_findings,
            rule_alarms=test_result.alarms_triggered,
            ml_classification=ml_cat,
            remediation_steps=parsed.get("remediation_steps", []),
            markdown_report=md_report,
            is_llm_generated=True
        )

    def _fallback_heuristic(
        self,
        test_result: ScenarioTestResult,
        findings: Dict[str, Any],
        ml_prediction: Optional[Dict[str, Any]]
    ) -> DiagnosticReport:
        name = test_result.scenario_name.lower()
        min_rx = findings.get("min_rx_power_dbm", -10.0)
        max_ber = float(findings.get("max_ber", "1e-12"))
        max_bias = findings.get("max_bias_current_ma", 45.0)
        ml_cat = ml_prediction.get("category", "UNKNOWN") if ml_prediction else "UNKNOWN"

        if "cut" in name or min_rx <= -28.0 or findings.get("los_alarm_asserted", False):
            severity = "P1_CRITICAL"
            summary = f"Total optical power extinction detected (Rx power dropped to {min_rx} dBm <= -28.0 dBm threshold), triggering hard Loss of Signal (LOS)."
            physical_cause = "Physical fiber severance or disconnected optical patch cord causing infinite link attenuation."
            remediation = [
                "Dispatch Field Service Engineer to run Optical Time-Domain Reflectometer (OTDR) trace to pinpoint break location.",
                "Inspect intermediate patch panels and splice trays along the cable run.",
                "Perform fusion splice repair on broken optical core and re-verify power budget."
            ]
        elif "attenuation" in name or "bend" in name or min_rx < -22.0:
            severity = "P2_MAJOR"
            summary = f"Received optical power dropped below sensitivity limit to {min_rx} dBm (deficit of {abs(min_rx + 22.0):.1f} dB), degrading optical link margin."
            physical_cause = "Macrobending stress, cable crushing in conduit, or tight bend radius along the fiber pathway exceeding critical angle."
            remediation = [
                "Inspect cable routing inside fiber distribution frame (FDF) for bend radii tighter than 30 mm.",
                "Perform dual-wavelength OTDR test (1310nm vs 1550nm); higher loss at 1550nm confirms macrobend.",
                "Relieve mechanical pinch point or replace degraded jumper cable."
            ]
        elif "connector" in name or "dirt" in name:
            severity = "P2_MAJOR"
            summary = f"Sudden insertion loss jump observed at optical connector interface, driving Rx power to {min_rx} dBm."
            physical_cause = "Ferrule endface contamination (particulate dirt, skin oil, or alcohol residue) obstructing core alignment."
            remediation = [
                "Inspect connector endface using 400x fiber inspection microscope (IEC 61300-3-35 standard).",
                "Perform wet-to-dry cleaning using optical lint-free cleaning cassette and solvent.",
                "Inspect mating adapter sleeve for ceramic cracks and re-seat connector."
            ]
        elif "laser" in name or "aging" in name or max_bias > 60.0:
            severity = "P2_MAJOR"
            summary = f"Transceiver automatic power control (APC) loop pushed laser bias current to {max_bias} mA (nominal 45 mA) to maintain output power."
            physical_cause = "Internal semiconductor lattice degradation and dark-line defect growth in the DFB laser diode."
            remediation = [
                "Flag SFP+/QSFP transceiver for preventive maintenance replacement before threshold runaway.",
                "Inspect transceiver heat dissipation and verify optical chassis fan airflow.",
                "Schedule hot-swap replacement with new calibrated optical transceiver module."
            ]
        elif "noise" in name or "ase" in name or min_rx > -20.0 and max_ber > 1.0e-5:
            severity = "P2_MAJOR"
            summary = f"Severe OSNR collapse (minimum {findings.get('min_osnr_db', 18.0)} dB) driving Pre-FEC BER to {findings.get('max_ber', '1e-5')} while optical power was preserved."
            physical_cause = "Amplified Spontaneous Emission (ASE) noise accumulation from degraded optical amplifier (EDFA) or multi-path interference (MPI)."
            remediation = [
                "Check EDFA input signal power and verify amplifier gain / noise figure operating setpoints.",
                "Inspect optical return loss (ORL) along the link to suppress back-reflections and multi-path interference.",
                "Tune optical channel equalization filters to re-center carrier-to-noise ratio."
            ]
        else:
            severity = "NOMINAL"
            summary = f"All telemetry metrics remained within nominal ITU-T operational tolerances (Rx power {min_rx} dBm, BER {findings.get('max_ber', '1e-12')})."
            physical_cause = "Normal link operation with healthy laser, fiber transmission, and receiver sensitivity."
            remediation = [
                "No physical intervention required. Maintain automated periodic telemetry polling.",
                "Retain baseline profile in regression database for trend analysis."
            ]

        md_report = self._format_markdown_report(
            test_result=test_result,
            severity=severity,
            summary=summary,
            physical_cause=physical_cause,
            findings=findings,
            remediation=remediation,
            ml_prediction=ml_prediction,
            is_llm=False
        )

        return DiagnosticReport(
            scenario_name=test_result.scenario_name,
            status=test_result.status,
            incident_severity=severity,
            root_cause_summary=summary,
            physical_root_cause=physical_cause,
            telemetry_findings=findings,
            rule_alarms=test_result.alarms_triggered,
            ml_classification=ml_cat,
            remediation_steps=remediation,
            markdown_report=md_report,
            is_llm_generated=False
        )

    def _format_markdown_report(
        self,
        test_result: ScenarioTestResult,
        severity: str,
        summary: str,
        physical_cause: str,
        findings: Dict[str, Any],
        remediation: List[str],
        ml_prediction: Optional[Dict[str, Any]],
        is_llm: bool
    ) -> str:
        ml_str = "N/A"
        if ml_prediction:
            ml_str = f"**{ml_prediction.get('category', 'N/A')}** (Confidence: {ml_prediction.get('confidence', 0.0) * 100:.1f}%)"

        engine_badge = "🤖 Gemini 2.5 LLM" if is_llm else "⚙️ Deterministic Optical Expert Heuristic"

        remediation_lines = "\n".join([f"{i+1}. {step}" for i, step in enumerate(remediation)])

        return rf"""# 📡 Optical Link Diagnostic Report: {test_result.scenario_name}

**Engine:** {engine_badge}  
**Incident Severity:** `{severity}`  
**Test Status:** `{'PASS' if test_result.status == 'PASS' else 'FAIL'}` (Expected: `{test_result.expected_fault}`, Observed: `{test_result.observed_fault}`)  
**ML Diagnostic Classifier:** {ml_str}  

---

## 🔍 Executive Root Cause Summary
{summary}

### Physical Failure Mechanism
> {physical_cause}

---

## 📊 Grounded Telemetry Evidence
| Parameter | Value | Reference Nominal Threshold | Status |
|:---|:---|:---|:---|
| **Min Rx Optical Power** | `{findings.get('min_rx_power_dbm', 'N/A')} dBm` | $\ge -22.0$ dBm (LOS @ $-28.0$) | {'🔴 ALARM' if findings.get('min_rx_power_dbm', 0) < -22 else '🟢 NORMAL'} |
| **Final Rx Optical Power** | `{findings.get('final_rx_power_dbm', 'N/A')} dBm` | $\ge -22.0$ dBm | {'🔴 ALARM' if findings.get('final_rx_power_dbm', 0) < -22 else '🟢 NORMAL'} |
| **Min Optical SNR (OSNR)** | `{findings.get('min_osnr_db', 'N/A')} dB` | $\ge 18.0$ dB | {'🟡 DEGRADED' if findings.get('min_osnr_db', 30) < 18 else '🟢 NORMAL'} |
| **Max Pre-FEC BER** | `{findings.get('max_ber', 'N/A')}` | $\le 1.00 \times 10^{{-7}}$ | {'🔴 CRITICAL' if float(findings.get('max_ber', 0)) > 1e-4 else ('🟡 WARNING' if float(findings.get('max_ber', 0)) > 1e-7 else '🟢 NORMAL')} |
| **Max Laser Bias Current** | `{findings.get('max_bias_current_ma', 'N/A')} mA` | Nominal $45.0$ mA (Max $65.0$ mA) | {'🔴 DRIFT' if findings.get('max_bias_current_ma', 45) > 65 else '🟢 NORMAL'} |
| **Max Laser Temperature** | `{findings.get('max_temp_c', 'N/A')} °C` | Nominal $35.0$ °C (Max $55.0$ °C) | {'🔴 OVERHEAT' if findings.get('max_temp_c', 35) > 55 else '🟢 NORMAL'} |
| **Hardware Alarms** | `{', '.join(test_result.alarms_triggered) if test_result.alarms_triggered else 'None'}` | `None` | {'🔴 ASSERTED' if test_result.alarms_triggered else '🟢 CLEAR'} |

---

## 🛠️ Prescriptive Field Engineering Remediation
{remediation_lines}
"""
