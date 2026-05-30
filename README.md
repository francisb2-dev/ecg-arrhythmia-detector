# ECG Arrhythmia Detection System

> A clinical-grade ECG signal processing and rhythm classification system using real patient data from the MIT-BIH Arrhythmia Database (PhysioNet).

**Author:** Braden Francis — Biomedical Engineering, Wentworth Institute of Technology  
**Data:** MIT-BIH Arrhythmia Database · Goldberger et al. (2000) · PhysioNet  
**Language:** Python 3.9+

---

## Overview

This system implements a complete automated ECG analysis pipeline — from raw signal acquisition to clinical HTML report — using validated algorithms and real annotated ECG recordings.

The clinical motivation is straightforward: arrhythmias affect over 5 million Americans and are a leading cause of sudden cardiac death. Early, accurate rhythm identification is critical for timely intervention. This project demonstrates how classical signal processing and statistical analysis can automate that detection with high sensitivity.

### What It Does

1. **Downloads** real ECG recordings from the MIT-BIH Arrhythmia Database (PhysioNet) — no account required
2. **Filters** the signal to remove baseline wander, high-frequency noise, and 60 Hz powerline interference
3. **Detects** QRS complexes using the Pan-Tompkins algorithm implemented from scratch
4. **Extracts** a full suite of clinical HRV metrics (SDNN, RMSSD, pNN50, Poincaré metrics, frequency domain)
5. **Classifies** the rhythm: Normal Sinus Rhythm, Atrial Fibrillation, Bradycardia, Tachycardia, PVCs
6. **Generates** a self-contained clinical HTML report with six embedded visualization panels

---

## Quick Start

```bash
# Clone / navigate to project
cd ecg-arrhythmia-detector/

# Install dependencies
pip install -r requirements.txt

# Run full demo (downloads 3 records, generates 3 reports, opens browser)
python demo.py
```

The demo analyzes three records spanning different rhythm types and opens the HTML reports automatically. On a first run it downloads ~3 MB of data from PhysioNet. Subsequent runs use local cache.

### Analyze Any Record

```bash
# Analyze record 100 (normal sinus rhythm)
python analyze.py --record 100

# Analyze record 208 with 120 seconds of data, verbose output
python analyze.py --record 208 --duration 120 --verbose

# Custom output directory, skip browser open
python analyze.py --record 203 --output my_reports/ --no-open

# Full help
python analyze.py --help
```

**Interesting records to try:**

| Record | Clinical Content |
|--------|----------------|
| 100 | Clean normal sinus rhythm |
| 200 | PVCs, mixed rhythms |
| 203 | Frequent ectopy, complex arrhythmias |
| 208 | PVCs, AF-like irregularity |
| 232 | Atrial flutter |
| 105 | ST changes, bundle branch block |

---

## How It Works

### Signal Processing Pipeline

```
Raw ECG
  │
  ▼
DC Offset Removal ──── mean(signal) = 0
  │
  ▼
High-Pass Filter ────── Butterworth order-4, 0.5 Hz cutoff
  │                     Removes baseline wander (respiration, movement)
  ▼
Bandpass Filter ──────── Butterworth order-4, 0.5–40 Hz passband
  │                     Captures P, QRS, T waves; rejects EMG/HF noise
  ▼
Notch Filter ─────────── IIR notch, 60 Hz, Q=30
  │                     Removes powerline interference
  ▼
Filtered ECG
```

All filters use zero-phase forward-backward filtering (`filtfilt`) to prevent phase distortion — essential because QRS morphology and timing are clinically significant.

### Pan-Tompkins QRS Detection

Implemented from scratch following the original 1985 paper:

```
Filtered ECG
  │
  ▼
5-Point Derivative ──── h(n) = [-x(n-2) - 2x(n-1) + 2x(n+1) + x(n+2)] / 8
  │                     Amplifies steep QRS slopes, suppresses P and T waves
  ▼
Squaring ─────────────── y(n) = x(n)²
  │                     Ensures positivity, amplifies large excursions
  ▼
Moving Window Integration ── 150ms window
  │                     Smears energy over QRS duration
  ▼
Adaptive Dual-Threshold ── Updates SPKI/NPKI dynamically
  │                     Threshold₁ = NPKI + 0.25·(SPKI - NPKI)
  │                     Threshold₂ = 0.5 · Threshold₁
  ▼
Search-Back ─────────── Recovers missed beats when > 1.66× mean RR
  │
  ▼
R-Peak Refinement ────── ±50ms window search in original signal
  │
  ▼
Detected R-Peaks
```

**Why implement it from scratch?** Using a library function for QRS detection would black-box the most technically interesting part. Implementing it demonstrates understanding of the signal processing decisions: why squaring instead of abs-value (amplifies large peaks more aggressively), why 150ms integration window (covers typical QRS duration), and how adaptive thresholding handles amplitude variation within a record.

### HRV Feature Extraction

Time-domain metrics computed per the 1996 Task Force standards:

| Metric | Formula | Clinical Significance |
|--------|---------|----------------------|
| SDNN | std(RR intervals) | Overall HRV; global autonomic modulation |
| RMSSD | √mean(ΔRR²) | Short-term; parasympathetic (vagal) tone |
| pNN50 | % \|ΔRR\| > 50ms | Parasympathetic activity index |
| CV | SDNN / mean(RR) | Key AF discriminator; >0.15 suggests AF |
| SD1 | std((RR_n - RR_{n+1}) / √2) | Poincaré; beat-to-beat variation |
| SD2 | std((RR_n + RR_{n+1}) / √2) | Poincaré; long-term variation |

Frequency-domain analysis uses the **Lomb-Scargle periodogram** (not FFT) because RR intervals are unevenly sampled in time. Lomb-Scargle handles irregular sampling natively without interpolation artifacts.

| Band | Frequency | Physiology |
|------|-----------|-----------|
| VLF | 0.003–0.04 Hz | Thermoregulation, vasomotor activity |
| LF | 0.04–0.15 Hz | Baroreceptor reflex (sympathetic + parasympathetic) |
| HF | 0.15–0.4 Hz | Respiratory sinus arrhythmia (parasympathetic) |

### Rhythm Classification

Evidence-based rule system with confidence scoring:

**Atrial Fibrillation:** CV > 0.15 (primary), RMSSD > 80ms, pNN50 > 40%, high Poincaré SD1. AF produces a diffuse Poincaré cloud with no regularity along the identity line. Confidence caps at 95% because confirming AF without P-wave analysis requires clinical correlation.

**Bradycardia / Tachycardia:** Mean HR < 60 or > 100 BPM (AHA/ACLS thresholds). Confidence scales with magnitude of deviation.

**Normal Sinus Rhythm:** Regular RR (CV < 0.10) + HR 60–100 BPM. Confidence reduced proportionally if AF markers are present.

**PVC Detection:** Identifies short-long RR coupling pairs (ectopic beat followed by compensatory pause). Burden expressed as percentage of total beats.

---

## Report Output

Each analysis generates a self-contained HTML file (~3–5 MB) with six visualization panels:

1. **Raw vs. Filtered ECG** — side-by-side comparison showing noise removal and R-peak markers
2. **QRS Overlay** — all detected beats superimposed, with ensemble mean in white; reveals morphological consistency or variability
3. **Heart Rate Trend** — instantaneous HR over time with normal range shading
4. **RR Tachogram** — beat-to-beat interval plot; color-coded by deviation from mean
5. **Poincaré Plot** — RR_n vs RR_{n+1} scatter with SD1/SD2 annotation
6. **HRV Frequency Domain** — bar chart and pie chart of VLF/LF/HF power distribution

The report also contains:
- Summary metric cards (HR, SDNN, RMSSD, CV, signal quality)
- Full HRV metrics table with normal ranges
- Primary rhythm classification with confidence bar
- Secondary findings
- Clinical interpretation paragraph
- Actionable recommendations
- Algorithm details table

---

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run individual test files
python -m pytest tests/test_preprocessor.py -v
python -m pytest tests/test_detector.py -v
python -m pytest tests/test_classifier.py -v
```

Tests use synthetic signals with known properties — not random fuzzing. Each test verifies a specific clinical or algorithmic property (e.g., "filter should preserve ≥60% of QRS band energy", "bradycardia at 40 BPM has higher confidence than 55 BPM").

---

## Project Structure

```
ecg-arrhythmia-detector/
├── src/
│   ├── loader.py          # PhysioNet data download + loading (wfdb)
│   ├── preprocessor.py    # Butterworth filters, notch, signal quality
│   ├── detector.py        # Pan-Tompkins QRS detection from scratch
│   ├── features.py        # HRV time + frequency domain metrics
│   ├── classifier.py      # Rule-based rhythm classification
│   ├── visualizer.py      # matplotlib clinical plots (dark theme)
│   └── reporter.py        # Jinja2 HTML report template
├── reports/               # Generated HTML reports
├── data/                  # Downloaded MIT-BIH records (auto-populated)
├── tests/
│   ├── test_preprocessor.py
│   ├── test_detector.py
│   └── test_classifier.py
├── analyze.py             # CLI entry point (Click)
├── demo.py                # One-command demo
└── requirements.txt
```

---

## Technical Dependencies

| Library | Purpose |
|---------|---------|
| `wfdb` | PhysioNet record download and WFDB format parsing |
| `numpy` | Signal arrays, numerical operations |
| `scipy` | Filter design (`butter`, `iirnotch`, `filtfilt`), Lomb-Scargle periodogram |
| `matplotlib` | All visualizations (non-interactive Agg backend) |
| `pandas` | Tabular data handling |
| `jinja2` | HTML report templating |
| `click` | CLI argument parsing |

---

## Limitations and Future Work

**Current limitations:**
- Rule-based classifier cannot distinguish AF from other highly irregular rhythms (e.g., atrial flutter with variable block)
- QRS duration estimate is a threshold-based heuristic, not a true boundary detection algorithm
- P-wave analysis is not implemented — AF confirmation in clinical settings requires P-wave absence detection
- Analysis is per-record, not streaming real-time

**Potential extensions:**
- Machine learning classifier trained on MIT-BIH annotations (Random Forest or 1D-CNN)
- Real-time streaming mode via websocket or serial port (compatible with ADS1292 ECG AFE)
- 12-lead support (currently single-lead)
- WFDB annotation overlay for ground-truth comparison visualization
- ST-segment elevation detection for ischemia screening

---

## References

1. **Pan, J. & Tompkins, W.J. (1985).** A real-time QRS detection algorithm. *IEEE Transactions on Biomedical Engineering*, 32(3), 230–236.

2. **Goldberger, A.L. et al. (2000).** PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation*, 101(23), e215–e220. https://physionet.org

3. **Moody, G.B. & Mark, R.G. (2001).** The impact of the MIT-BIH arrhythmia database. *IEEE Engineering in Medicine and Biology*, 20(3), 45–50.

4. **Task Force of the European Society of Cardiology (1996).** Heart rate variability: standards of measurement, physiological interpretation and clinical use. *Circulation*, 93(5), 1043–1065.

5. **January, C.T. et al. (2014).** 2014 AHA/ACC/HRS Guideline for the Management of Patients with Atrial Fibrillation. *Journal of the American College of Cardiology*, 64(21), e1–e76.

---

> **Disclaimer:** This system is built for educational and portfolio purposes. It is not a medical device and must not be used for clinical diagnosis. All outputs require review by a qualified clinician.
