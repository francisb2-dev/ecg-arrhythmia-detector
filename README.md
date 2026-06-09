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

### Train the ML Classifier

```bash
# Train Random Forest on all 48 MIT-BIH records (downloads data if needed, ~5 min)
python train.py

# Evaluate saved model and print per-class F1
python eval_metrics.py
```

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
│   ├── loader.py           # PhysioNet data download + loading (wfdb)
│   ├── preprocessor.py     # Butterworth filters, notch, signal quality
│   ├── detector.py         # Pan-Tompkins QRS detection from scratch
│   ├── features.py         # HRV time + frequency domain metrics
│   ├── classifier.py       # Rule-based rhythm classification
│   ├── ml_classifier.py    # Random Forest beat classifier (18-feature + raw template)
│   ├── cnn_classifier.py   # 1D-CNN beat classifier (experimental)
│   ├── evaluator.py        # Evaluation report generator
│   ├── visualizer.py       # matplotlib clinical plots (dark theme)
│   └── reporter.py         # Jinja2 HTML report template
├── models/
│   └── rf_classifier.joblib   # Trained RF model (v1.2)
├── reports/                # Generated HTML reports
├── data/                   # Downloaded MIT-BIH records (auto-populated)
├── tests/
│   ├── test_preprocessor.py
│   ├── test_detector.py
│   └── test_classifier.py
├── analyze.py              # CLI entry point (Click)
├── train.py                # RF training pipeline (MIT-BIH, 48 records)
├── eval_metrics.py         # Per-class F1 evaluation + resume bullet output
├── demo.py                 # One-command demo
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

## ML Classifier Performance (Random Forest, v1.2)

Trained and evaluated on the full MIT-BIH Arrhythmia Database (48 records, record-level train/test split to prevent data leakage):

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Normal | 0.84 | 0.94 | **0.89** | 43,507 |
| PVC | 0.72 | 0.88 | **0.80** | 5,890 |
| Atrial Ectopic | 0.36 | 0.16 | 0.22 | 2,605 |
| Right BBB | 0.34 | 0.16 | 0.22 | 3,562 |
| Left BBB | 0.00 | 0.00 | 0.00 | 3,460 |

**Overall test accuracy: 77.1% (61,839 beats, 25 records)**

### Left BBB Root Cause Analysis

Left BBB F1 = 0.00 despite 4,615 annotated training beats. Root cause identified through feature analysis:

The training LBBB beats come from records 109 and 111 (two patients). When we compute a global mean LBBB template from these records and measure its correlation with test LBBB beats (records 207, 214 — different patients), the mean correlation is **0.127** — compared to **0.816** for Normal beats. The LBBB morphology from records 109/111 is more similar to Normal beats than to the LBBB beats in records 207/214.

This is not a feature engineering failure. It is a fundamental property of single-lead ECG BBB detection: LBBB morphology varies significantly between patients depending on the degree of block, the QRS axis, and lead placement. With only two training patients and two test patients, there is insufficient corpus diversity to learn a generalizable morphological signature.

**What this means for next steps:**
1. Multi-lead ECG (12-lead) provides better BBB discriminability — the characteristic M-shaped QRS in V5/V6 is lead-specific and more consistent across patients than MLII morphology
2. A 1D-CNN trained on a larger corpus (e.g., PTB-XL, 21,799 records) would implicitly learn patient-invariant BBB representations
3. A rule-based fallback for QRS duration (>120ms threshold) would help, but requires better QRS onset/offset detection than the current threshold-based estimate

---

## Limitations and Known Issues

- **Left BBB generalization**: single-lead morphological variability between patients prevents reliable detection with the current training corpus (see above)
- **Rule-based rhythm classifier**: cannot distinguish AF from other highly irregular rhythms (e.g., atrial flutter with variable block); P-wave analysis is not implemented
- **QRS duration estimate**: the threshold-based QRS width heuristic underestimates absolute duration when the signal is globally normalized — a dedicated QRS onset/offset detector would improve BBB detection
- **Single-lead analysis only**: MLII-equivalent lead provides one projection of the cardiac dipole; definitive BBB diagnosis requires a 12-lead view

**Potential extensions:**
- 12-lead support for improved BBB and ischemia detection (requires multi-channel data loader)
- 1D-CNN trained on PTB-XL or similar larger dataset for better per-class generalization
- Real-time streaming mode via websocket or serial port (compatible with ADS1292 ECG AFE)
- ST-segment elevation detection for ischemia screening
- WFDB annotation overlay for ground-truth comparison visualization

---

## References

1. **Pan, J. & Tompkins, W.J. (1985).** A real-time QRS detection algorithm. *IEEE Transactions on Biomedical Engineering*, 32(3), 230–236.

2. **Goldberger, A.L. et al. (2000).** PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation*, 101(23), e215–e220. https://physionet.org

3. **Moody, G.B. & Mark, R.G. (2001).** The impact of the MIT-BIH arrhythmia database. *IEEE Engineering in Medicine and Biology*, 20(3), 45–50.

4. **Task Force of the European Society of Cardiology (1996).** Heart rate variability: standards of measurement, physiological interpretation and clinical use. *Circulation*, 93(5), 1043–1065.

5. **January, C.T. et al. (2014).** 2014 AHA/ACC/HRS Guideline for the Management of Patients with Atrial Fibrillation. *Journal of the American College of Cardiology*, 64(21), e1–e76.

---

> **Disclaimer:** This system is built for educational and portfolio purposes. It is not a medical device and must not be used for clinical diagnosis. All outputs require review by a qualified clinician.
