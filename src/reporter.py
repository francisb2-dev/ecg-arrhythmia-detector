"""
reporter.py — Clinical-style HTML report generation.

Generates a self-contained HTML report with embedded base64 plots,
metric cards, and clinical interpretation. The report requires no
external files and can be opened directly in any browser.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from jinja2 import Template

from .features import HRVFeatures
from .detector import DetectionResult
from .classifier import ClassificationResult
from .preprocessor import FilteredSignal


REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ECG Analysis Report — Record {{ record_name }}</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: #0d1117;
      color: #e6edf3;
      line-height: 1.6;
    }

    /* ── Header ── */
    .header {
      background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
      border-bottom: 2px solid #30363d;
      padding: 24px 40px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .header-left h1 {
      font-size: 1.6rem;
      font-weight: 700;
      color: #58a6ff;
      letter-spacing: -0.5px;
    }
    .header-left p {
      font-size: 0.85rem;
      color: #8b949e;
      margin-top: 4px;
    }
    .header-badge {
      background: {{ rhythm_color }};
      color: #0d1117;
      padding: 8px 20px;
      border-radius: 20px;
      font-weight: 700;
      font-size: 0.95rem;
      letter-spacing: 0.5px;
    }

    /* ── Container ── */
    .container { max-width: 1200px; margin: 0 auto; padding: 32px 40px; }

    /* ── Summary Cards ── */
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 16px;
      margin-bottom: 32px;
    }
    .card {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 10px;
      padding: 16px;
      transition: border-color 0.2s;
    }
    .card:hover { border-color: #58a6ff; }
    .card-label {
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: #8b949e;
      margin-bottom: 6px;
    }
    .card-value {
      font-size: 1.55rem;
      font-weight: 700;
      color: #e6edf3;
      font-variant-numeric: tabular-nums;
    }
    .card-unit {
      font-size: 0.78rem;
      color: #8b949e;
      margin-left: 4px;
    }
    .card-sub {
      font-size: 0.75rem;
      color: #8b949e;
      margin-top: 4px;
    }

    /* ── Sections ── */
    .section {
      margin-bottom: 40px;
    }
    .section-title {
      font-size: 1.05rem;
      font-weight: 600;
      color: #58a6ff;
      border-left: 3px solid #58a6ff;
      padding-left: 12px;
      margin-bottom: 20px;
      text-transform: uppercase;
      letter-spacing: 0.8px;
    }

    /* ── Plots ── */
    .plot-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
      gap: 20px;
    }
    .plot-card {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 10px;
      overflow: hidden;
    }
    .plot-card img {
      width: 100%;
      display: block;
    }
    .plot-caption {
      padding: 10px 14px;
      font-size: 0.78rem;
      color: #8b949e;
      border-top: 1px solid #30363d;
    }
    .plot-full { grid-column: 1 / -1; }

    /* ── Tables ── */
    .metrics-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.875rem;
    }
    .metrics-table th {
      text-align: left;
      padding: 8px 14px;
      background: #1c2333;
      color: #8b949e;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      border-bottom: 1px solid #30363d;
    }
    .metrics-table td {
      padding: 8px 14px;
      border-bottom: 1px solid #21262d;
      color: #e6edf3;
    }
    .metrics-table tr:hover td { background: #1c2333; }
    .metrics-table .value { font-variant-numeric: tabular-nums; font-weight: 500; }
    .badge {
      display: inline-block;
      padding: 2px 10px;
      border-radius: 12px;
      font-size: 0.73rem;
      font-weight: 600;
    }
    .badge-normal { background: #1a3a1a; color: #3fb950; border: 1px solid #238636; }
    .badge-warning { background: #3a2e1a; color: #d29922; border: 1px solid #9e6a03; }
    .badge-critical { background: #3a1a1a; color: #f85149; border: 1px solid #da3633; }
    .badge-info { background: #1a2a3a; color: #58a6ff; border: 1px solid #1f6feb; }

    /* ── Classification Block ── */
    .classification-block {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 10px;
      padding: 24px;
      margin-bottom: 24px;
    }
    .primary-rhythm {
      display: flex;
      align-items: center;
      gap: 16px;
      margin-bottom: 16px;
    }
    .rhythm-name {
      font-size: 1.4rem;
      font-weight: 700;
      color: {{ rhythm_color }};
    }
    .confidence-bar-wrap {
      flex: 1;
      background: #0d1117;
      border-radius: 6px;
      height: 8px;
      overflow: hidden;
    }
    .confidence-bar {
      height: 100%;
      background: {{ rhythm_color }};
      border-radius: 6px;
      width: {{ primary_confidence }}%;
    }
    .evidence-list {
      list-style: none;
      margin-top: 12px;
    }
    .evidence-list li {
      padding: 4px 0;
      padding-left: 18px;
      position: relative;
      font-size: 0.875rem;
      color: #c9d1d9;
    }
    .evidence-list li::before {
      content: "→";
      position: absolute;
      left: 0;
      color: #58a6ff;
    }

    /* ── Interpretation ── */
    .interpretation-box {
      background: #1c2333;
      border-left: 4px solid {{ rhythm_color }};
      border-radius: 0 8px 8px 0;
      padding: 16px 20px;
      font-size: 0.9rem;
      line-height: 1.7;
      color: #c9d1d9;
      margin-bottom: 16px;
    }
    .recommendations {
      list-style: none;
    }
    .recommendations li {
      padding: 6px 0 6px 24px;
      position: relative;
      font-size: 0.875rem;
      color: #c9d1d9;
      border-bottom: 1px solid #21262d;
    }
    .recommendations li::before {
      content: "◆";
      position: absolute;
      left: 0;
      color: {{ rhythm_color }};
      font-size: 0.6rem;
      top: 10px;
    }

    /* ── Footer ── */
    .footer {
      margin-top: 60px;
      padding: 24px 40px;
      border-top: 1px solid #21262d;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.78rem;
      color: #8b949e;
    }
    .disclaimer {
      background: #1c2333;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 12px 16px;
      font-size: 0.78rem;
      color: #8b949e;
      margin-top: 24px;
    }
    .disclaimer strong { color: #d29922; }

    /* ── Pipeline diagram ── */
    .pipeline {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      margin: 16px 0;
    }
    .pipeline-step {
      background: #1c2333;
      border: 1px solid #30363d;
      border-radius: 6px;
      padding: 8px 14px;
      font-size: 0.78rem;
      color: #8b949e;
    }
    .pipeline-arrow { color: #30363d; font-size: 1.1rem; }
  </style>
</head>
<body>

<!-- ═══ HEADER ═══════════════════════════════════════════════════════════════ -->
<div class="header">
  <div class="header-left">
    <h1>ECG Analysis Report</h1>
    <p>MIT-BIH Arrhythmia Database · Record {{ record_name }} · Lead {{ lead_name }}</p>
  </div>
  <div>
    <div style="text-align:right; font-size:0.78rem; color:#8b949e; margin-bottom:8px;">
      Generated {{ generated_at }} · Duration {{ duration_sec }}s · {{ n_beats }} beats
    </div>
    <div class="header-badge">{{ primary_rhythm }}</div>
  </div>
</div>

<div class="container">

<!-- ═══ SUMMARY CARDS ═══════════════════════════════════════════════════════ -->
<div class="summary-grid">
  <div class="card">
    <div class="card-label">Mean Heart Rate</div>
    <div class="card-value">{{ mean_hr }}<span class="card-unit">BPM</span></div>
    <div class="card-sub">Range: {{ min_hr }}–{{ max_hr }} BPM</div>
  </div>
  <div class="card">
    <div class="card-label">Mean RR Interval</div>
    <div class="card-value">{{ mean_rr }}<span class="card-unit">ms</span></div>
    <div class="card-sub">SDNN: {{ sdnn }}ms</div>
  </div>
  <div class="card">
    <div class="card-label">RMSSD</div>
    <div class="card-value">{{ rmssd }}<span class="card-unit">ms</span></div>
    <div class="card-sub">pNN50: {{ pnn50 }}%</div>
  </div>
  <div class="card">
    <div class="card-label">RR Coefficient of Variation</div>
    <div class="card-value">{{ cv_rr }}</div>
    <div class="card-sub">{{ cv_interpretation }}</div>
  </div>
  <div class="card">
    <div class="card-label">Beats Analyzed</div>
    <div class="card-value">{{ n_beats }}</div>
    <div class="card-sub">{{ duration_sec }}s record</div>
  </div>
  <div class="card">
    <div class="card-label">Signal Quality</div>
    <div class="card-value">{{ signal_quality_score }}<span class="card-unit">/100</span></div>
    <div class="card-sub">{{ signal_quality_label }}</div>
  </div>
  <div class="card">
    <div class="card-label">QRS Duration</div>
    <div class="card-value">{{ qrs_duration }}<span class="card-unit">ms</span></div>
    <div class="card-sub">Estimated</div>
  </div>
  <div class="card">
    <div class="card-label">LF/HF Ratio</div>
    <div class="card-value">{{ lf_hf_ratio }}</div>
    <div class="card-sub">Sympathovagal balance</div>
  </div>
</div>

<!-- ═══ SIGNAL VISUALIZATION ════════════════════════════════════════════════ -->
<div class="section">
  <div class="section-title">ECG Signal & QRS Detection</div>

  <div class="pipeline">
    <div class="pipeline-step">Raw ECG</div>
    <div class="pipeline-arrow">▶</div>
    <div class="pipeline-step">DC Removal</div>
    <div class="pipeline-arrow">▶</div>
    <div class="pipeline-step">HP 0.5Hz</div>
    <div class="pipeline-arrow">▶</div>
    <div class="pipeline-step">BP 0.5–40Hz</div>
    <div class="pipeline-arrow">▶</div>
    <div class="pipeline-step">Notch 60Hz</div>
    <div class="pipeline-arrow">▶</div>
    <div class="pipeline-step">Pan-Tompkins QRS</div>
    <div class="pipeline-arrow">▶</div>
    <div class="pipeline-step">Feature Extraction</div>
    <div class="pipeline-arrow">▶</div>
    <div class="pipeline-step">Rhythm Classification</div>
  </div>

  <div class="plot-grid">
    <div class="plot-card plot-full">
      <img src="data:image/png;base64,{{ plot_raw_vs_filtered }}" alt="Raw vs Filtered ECG">
      <div class="plot-caption">
        Top: Raw ECG signal with original noise and baseline wander visible.
        Bottom: Filtered signal (0.5–40 Hz bandpass + 60 Hz notch) with detected R-peaks (▼) overlaid.
        First 10 seconds shown.
      </div>
    </div>
  </div>
</div>

<!-- ═══ QRS MORPHOLOGY ═══════════════════════════════════════════════════════ -->
<div class="section">
  <div class="section-title">QRS Morphology Analysis</div>
  <div class="plot-grid">
    <div class="plot-card plot-full">
      <img src="data:image/png;base64,{{ plot_qrs_overlay }}" alt="QRS Overlay">
      <div class="plot-caption">
        Superimposed QRS complexes aligned to R-peak (t=0). Individual beats shown in teal;
        ensemble mean in white. Morphological consistency indicates stable conduction pathway.
        Variability suggests ectopic beats or multi-focal origin.
      </div>
    </div>
  </div>
</div>

<!-- ═══ HEART RATE ANALYSIS ══════════════════════════════════════════════════ -->
<div class="section">
  <div class="section-title">Heart Rate Analysis</div>
  <div class="plot-grid">
    <div class="plot-card plot-full">
      <img src="data:image/png;base64,{{ plot_heart_rate_trend }}" alt="Heart Rate Trend">
      <div class="plot-caption">
        Instantaneous heart rate derived from consecutive RR intervals. Green band = normal range (60–100 BPM).
        Persistent elevation or depression warrants clinical evaluation.
      </div>
    </div>
    <div class="plot-card plot-full">
      <img src="data:image/png;base64,{{ plot_rr_tachogram }}" alt="RR Tachogram">
      <div class="plot-caption">
        Beat-to-beat RR intervals. Red/yellow points deviate &gt;2.5/1.5 SD from mean.
        Dashed line = mean RR; shaded band = ±2 standard deviations.
        High scatter is the hallmark of atrial fibrillation.
      </div>
    </div>
  </div>
</div>

<!-- ═══ HRV ANALYSIS ═════════════════════════════════════════════════════════ -->
<div class="section">
  <div class="section-title">Heart Rate Variability (HRV) Analysis</div>
  <div class="plot-grid">
    <div class="plot-card">
      <img src="data:image/png;base64,{{ plot_poincare }}" alt="Poincaré Plot">
      <div class="plot-caption">
        Poincaré plot (RR_n vs RR_{n+1}). Football-shaped cloud = regular rhythm.
        Diffuse cloud = AF or frequent ectopy.
        SD1 = short-term variability; SD2 = long-term variability.
      </div>
    </div>
    <div class="plot-card">
      <img src="data:image/png;base64,{{ plot_frequency_domain }}" alt="HRV Frequency Domain">
      <div class="plot-caption">
        Lomb-Scargle periodogram integrated over VLF (0.003–0.04 Hz),
        LF (0.04–0.15 Hz), and HF (0.15–0.4 Hz) bands.
        LF/HF ratio reflects sympathovagal balance.
      </div>
    </div>
  </div>

  <div style="margin-top: 24px;">
    <table class="metrics-table">
      <thead>
        <tr>
          <th>Metric</th>
          <th>Value</th>
          <th>Normal Range</th>
          <th>Interpretation</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        {% for row in hrv_table %}
        <tr>
          <td>{{ row.metric }}</td>
          <td class="value">{{ row.value }}</td>
          <td style="color:#8b949e;">{{ row.normal }}</td>
          <td style="color:#8b949e; font-size:0.82rem;">{{ row.interpretation }}</td>
          <td><span class="badge badge-{{ row.status }}">{{ row.status_label }}</span></td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>

<!-- ═══ RHYTHM CLASSIFICATION ════════════════════════════════════════════════ -->
<div class="section">
  <div class="section-title">Rhythm Classification</div>

  <div class="classification-block">
    <div class="primary-rhythm">
      <div>
        <div style="font-size:0.72rem; color:#8b949e; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;">Primary Rhythm</div>
        <div class="rhythm-name">{{ primary_rhythm }}</div>
      </div>
      <div style="flex:1;">
        <div style="font-size:0.72rem; color:#8b949e; margin-bottom:6px;">Confidence: {{ primary_confidence }}%</div>
        <div class="confidence-bar-wrap"><div class="confidence-bar"></div></div>
      </div>
      <div>
        <span class="badge badge-{{ primary_severity_class }}">{{ primary_severity }}</span>
      </div>
    </div>
    <ul class="evidence-list">
      {% for ev in primary_evidence %}
      <li>{{ ev }}</li>
      {% endfor %}
    </ul>
  </div>

  {% if secondary_findings %}
  <div style="margin-bottom: 24px;">
    <div style="font-size:0.85rem; color:#8b949e; margin-bottom:12px;">Secondary Findings</div>
    {% for finding in secondary_findings %}
    <div style="background:#161b22; border:1px solid #30363d; border-radius:8px; padding:14px; margin-bottom:10px; display:flex; justify-content:space-between; align-items:center;">
      <div>
        <span style="font-weight:600; color:#c9d1d9;">{{ finding.name }}</span>
        <span style="font-size:0.78rem; color:#8b949e; margin-left:10px;">{{ finding.confidence }}% confidence</span>
        <div style="font-size:0.78rem; color:#8b949e; margin-top:4px;">{{ finding.evidence|join(' · ') }}</div>
      </div>
      <span class="badge badge-{{ finding.severity_class }}">{{ finding.severity }}</span>
    </div>
    {% endfor %}
  </div>
  {% endif %}

</div>

<!-- ═══ CLINICAL INTERPRETATION ══════════════════════════════════════════════ -->
<div class="section">
  <div class="section-title">Clinical Interpretation</div>

  <div class="interpretation-box">{{ clinical_interpretation }}</div>

  <div style="font-size:0.85rem; color:#8b949e; margin-bottom:10px;">Recommendations</div>
  <ul class="recommendations">
    {% for rec in recommendations %}
    <li>{{ rec }}</li>
    {% endfor %}
  </ul>

  <div class="disclaimer">
    <strong>⚠ Research / Educational Use Only:</strong>
    This automated analysis is generated by an algorithmic system for educational
    and portfolio purposes. It is NOT a medical device and must NOT be used for
    clinical decision-making. All findings require confirmation by a qualified
    cardiologist using certified diagnostic equipment.
  </div>
</div>

<!-- ═══ ALGORITHM DETAILS ════════════════════════════════════════════════════ -->
<div class="section">
  <div class="section-title">Algorithm Details</div>
  <table class="metrics-table">
    <thead>
      <tr><th>Component</th><th>Method</th><th>Parameters</th></tr>
    </thead>
    <tbody>
      <tr>
        <td>Baseline Wander Removal</td>
        <td class="value">Butterworth High-pass (order 4, zero-phase)</td>
        <td style="color:#8b949e;">Cutoff: 0.5 Hz</td>
      </tr>
      <tr>
        <td>Noise Reduction</td>
        <td class="value">Butterworth Bandpass (order 4, zero-phase)</td>
        <td style="color:#8b949e;">0.5–40 Hz passband</td>
      </tr>
      <tr>
        <td>Powerline Interference</td>
        <td class="value">IIR Notch Filter</td>
        <td style="color:#8b949e;">60 Hz, Q=30</td>
      </tr>
      <tr>
        <td>QRS Detection</td>
        <td class="value">Pan-Tompkins (1985) — differentiate → square → MWI → adaptive threshold</td>
        <td style="color:#8b949e;">Refractory: 200ms, Window: 150ms</td>
      </tr>
      <tr>
        <td>HRV Frequency Analysis</td>
        <td class="value">Lomb-Scargle Periodogram</td>
        <td style="color:#8b949e;">0.003–0.4 Hz, 500 freq points</td>
      </tr>
      <tr>
        <td>AF Detection</td>
        <td class="value">RR interval CV + RMSSD + pNN50 heuristic</td>
        <td style="color:#8b949e;">CV threshold: 0.15</td>
      </tr>
      <tr>
        <td>Data Source</td>
        <td class="value">MIT-BIH Arrhythmia Database (PhysioNet)</td>
        <td style="color:#8b949e;">Goldberger et al. 2000, Record {{ record_name }}</td>
      </tr>
    </tbody>
  </table>
</div>

</div><!-- /container -->

<!-- ═══ FOOTER ═══════════════════════════════════════════════════════════════ -->
<div class="footer">
  <div>
    <strong style="color:#58a6ff;">ECG Arrhythmia Detection System</strong> ·
    Built by Braden Francis · Wentworth Institute of Technology · Biomedical Engineering
  </div>
  <div>
    Data: MIT-BIH Arrhythmia Database · PhysioNet ·
    Goldberger et al. (2000) Circulation 101(23):e215
  </div>
</div>

</body>
</html>
"""


def _severity_class(severity: str) -> str:
    return {"normal": "normal", "benign": "info", "moderate": "warning",
            "critical": "critical"}.get(severity, "info")


def _rhythm_color(primary_code: str) -> str:
    return {
        "NSR":    "#3fb950",
        "AF":     "#d29922",
        "BRADY":  "#58a6ff",
        "TACHY":  "#f85149",
        "PVC":    "#e3b341",
        "HIGH_HRV": "#79c0ff",
        "LOW_HRV":  "#ff7b72",
        "NODATA":   "#8b949e",
    }.get(primary_code, "#58a6ff")


def _build_hrv_table(hrv: HRVFeatures) -> list:
    """Build the HRV metrics table rows."""

    def row(metric, value, normal, interp, is_normal):
        return {
            "metric": metric,
            "value": value,
            "normal": normal,
            "interpretation": interp,
            "status": "normal" if is_normal else "warning",
            "status_label": "Normal" if is_normal else "Atypical",
        }

    rows = [
        row("Mean RR Interval", f"{hrv.mean_rr_ms:.1f} ms",
            "600–1000 ms",
            "Corresponds to 60–100 BPM",
            600 <= hrv.mean_rr_ms <= 1000),
        row("SDNN", f"{hrv.std_rr_ms:.1f} ms",
            ">50 ms (short-term)",
            "Global HRV; reflects overall autonomic modulation",
            hrv.std_rr_ms >= 20),
        row("RMSSD", f"{hrv.rmssd_ms:.1f} ms",
            "20–50 ms (rest)",
            "Parasympathetic (vagal) tone indicator",
            15 <= hrv.rmssd_ms <= 200),
        row("pNN50", f"{hrv.pnn50_pct:.1f}%",
            "5–40%",
            "% beats differing >50ms from predecessor",
            hrv.pnn50_pct >= 0),
        row("CV of RR", f"{hrv.cv_rr:.4f}",
            "< 0.10 (NSR)",
            "Key AF discriminator; >0.15 suggests AF",
            hrv.cv_rr < 0.15),
        row("Poincaré SD1", f"{hrv.sd1_ms:.1f} ms",
            "10–40 ms",
            "Short-term beat-to-beat variation",
            True),
        row("Poincaré SD2", f"{hrv.sd2_ms:.1f} ms",
            ">SD1",
            "Long-term variation; SD2 > SD1 in NSR",
            hrv.sd2_ms >= hrv.sd1_ms),
        row("LF/HF Ratio", f"{hrv.lf_hf_ratio:.2f}",
            "1.5–2.0 (awake)",
            "Sympathovagal balance",
            True),
        row("QRS Duration", f"{hrv.qrs_duration_ms or 'N/A'} ms",
            "70–100 ms",
            "Wide QRS (>120ms) suggests bundle branch block",
            hrv.qrs_duration_ms is None or hrv.qrs_duration_ms < 120),
    ]
    return rows


def generate_report(
    record_name: str,
    lead_name: str,
    filtered: FilteredSignal,
    detection: DetectionResult,
    hrv: HRVFeatures,
    classification: ClassificationResult,
    plots: dict,
    signal_quality: dict,
    output_path: Path,
) -> Path:
    """
    Render and save the HTML report.

    Parameters
    ----------
    output_path : Path
        Where to save the .html file.

    Returns
    -------
    Path to the saved report.
    """
    primary = classification.primary_rhythm

    cv = hrv.cv_rr
    if cv < 0.05:
        cv_interp = "Very regular rhythm"
    elif cv < 0.10:
        cv_interp = "Mildly variable"
    elif cv < 0.20:
        cv_interp = "Moderately irregular"
    else:
        cv_interp = "Highly irregular — possible AF"

    secondary_findings_data = []
    for f in classification.secondary_findings[:5]:
        secondary_findings_data.append({
            "name": f.name,
            "confidence": f.confidence,
            "evidence": f.evidence,
            "severity": f.severity.title(),
            "severity_class": _severity_class(f.severity),
        })

    template = Template(REPORT_TEMPLATE)
    html = template.render(
        record_name=record_name,
        lead_name=lead_name,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        duration_sec=round(hrv.analysis_duration_sec, 1),
        n_beats=hrv.n_beats,
        # Summary cards
        mean_hr=hrv.mean_hr_bpm,
        min_hr=hrv.min_hr_bpm,
        max_hr=hrv.max_hr_bpm,
        mean_rr=hrv.mean_rr_ms,
        sdnn=hrv.std_rr_ms,
        rmssd=hrv.rmssd_ms,
        pnn50=hrv.pnn50_pct,
        cv_rr=hrv.cv_rr,
        cv_interpretation=cv_interp,
        qrs_duration=hrv.qrs_duration_ms or "N/A",
        lf_hf_ratio=hrv.lf_hf_ratio,
        signal_quality_score=signal_quality.get("score", "N/A"),
        signal_quality_label=signal_quality.get("quality_label", ""),
        # Rhythm
        primary_rhythm=primary.name,
        primary_confidence=primary.confidence,
        primary_evidence=primary.evidence,
        primary_severity=primary.severity.title(),
        primary_severity_class=_severity_class(primary.severity),
        rhythm_color=_rhythm_color(primary.code),
        secondary_findings=secondary_findings_data,
        clinical_interpretation=classification.clinical_interpretation,
        recommendations=classification.recommendations,
        # HRV table
        hrv_table=_build_hrv_table(hrv),
        # Plots (base64)
        plot_raw_vs_filtered=plots.get("raw_vs_filtered", ""),
        plot_rr_tachogram=plots.get("rr_tachogram", ""),
        plot_heart_rate_trend=plots.get("heart_rate_trend", ""),
        plot_poincare=plots.get("poincare", ""),
        plot_frequency_domain=plots.get("frequency_domain", ""),
        plot_qrs_overlay=plots.get("qrs_overlay", ""),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    return output_path
