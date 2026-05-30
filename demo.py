#!/usr/bin/env python3
"""
demo.py — One-command demonstration of the ECG arrhythmia detection system.

Downloads three MIT-BIH records spanning a range of cardiac rhythms,
runs the full analysis pipeline on each, and opens the HTML reports.

Usage:
    python demo.py

Requires internet connection on first run (downloads ~3MB of ECG data).
Subsequent runs use cached local data.
"""

import sys
import time
import webbrowser
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.loader import load_record, get_annotation_beats
from src.preprocessor import preprocess, estimate_signal_quality
from src.detector import detect_qrs, compute_detection_accuracy
from src.features import extract_features
from src.classifier import classify_rhythm
from src.visualizer import generate_all_plots
from src.reporter import generate_report


# ── Demo records ─────────────────────────────────────────────────────────────
DEMO_RECORDS = [
    {
        "record": "100",
        "description": "Normal Sinus Rhythm — clean signal, healthy adult",
        "duration": 60,
    },
    {
        "record": "208",
        "description": "Complex Arrhythmia — PVCs, mixed rhythms",
        "duration": 60,
    },
    {
        "record": "203",
        "description": "Highly Irregular — frequent ectopy, possible AF markers",
        "duration": 60,
    },
]

REPORTS_DIR = Path(__file__).parent / "reports"


def analyze_record(record_info: dict) -> dict:
    """Run the full pipeline on one record and return results."""
    record = record_info["record"]
    duration = record_info["duration"]

    print(f"\n  {'─'*54}")
    print(f"  Record {record}: {record_info['description']}")
    print(f"  {'─'*54}")

    # Load
    print("  [1/7] Loading data...", end=" ", flush=True)
    t0 = time.time()
    ecg = load_record(record_name=record, duration_sec=duration)
    print(f"done ({time.time()-t0:.1f}s) — {ecg.n_samples:,} samples @ {ecg.fs}Hz")

    # Signal quality
    print("  [2/7] Assessing quality...", end=" ", flush=True)
    quality = estimate_signal_quality(ecg.signal, ecg.fs)
    print(f"done — {quality['quality_label']} ({quality['score']}/100)")

    # Preprocess
    print("  [3/7] Filtering signal...", end=" ", flush=True)
    filtered = preprocess(ecg.signal, ecg.fs, apply_notch_filter=True)
    print("done — bandpass 0.5–40Hz + 60Hz notch")

    # Detect
    print("  [4/7] Detecting QRS complexes...", end=" ", flush=True)
    detection = detect_qrs(filtered.filtered, ecg.fs)
    print(f"done — {detection.n_beats} beats, mean HR {detection.mean_hr_bpm:.1f} BPM")

    # Accuracy vs ground truth
    if ecg.annotations:
        gt = get_annotation_beats(ecg)
        if len(gt["samples"]) > 0:
            acc = compute_detection_accuracy(
                detection.r_peaks, gt["samples"], ecg.fs
            )
            print(f"         Accuracy: Se={acc['sensitivity']:.3f}, "
                  f"PPV={acc['ppv']:.3f}, F1={acc['f1']:.3f}")

    # Features
    print("  [5/7] Extracting HRV features...", end=" ", flush=True)
    hrv = extract_features(
        r_peaks=detection.r_peaks,
        rr_intervals_sec=detection.rr_intervals_sec,
        r_peak_amplitudes=detection.r_peak_amplitudes,
        filtered_signal=filtered.filtered,
        fs=ecg.fs,
        duration_sec=ecg.duration_sec,
    )
    print(f"done — SDNN={hrv.std_rr_ms:.1f}ms, RMSSD={hrv.rmssd_ms:.1f}ms, "
          f"CV={hrv.cv_rr:.4f}")

    # Classify
    print("  [6/7] Classifying rhythm...", end=" ", flush=True)
    classification = classify_rhythm(hrv, detection.rr_intervals_sec)
    primary = classification.primary_rhythm
    print(f"done — {primary.name} ({primary.confidence:.0f}%)")

    # Report
    print("  [7/7] Generating report...", end=" ", flush=True)
    plots = generate_all_plots(
        filtered=filtered,
        detection=detection,
        hrv=hrv,
        classification=classification,
        record_name=record,
    )
    report_path = REPORTS_DIR / f"ecg_report_{record}.html"
    generate_report(
        record_name=record,
        lead_name=ecg.lead_name,
        filtered=filtered,
        detection=detection,
        hrv=hrv,
        classification=classification,
        plots=plots,
        signal_quality=quality,
        output_path=report_path,
    )
    print(f"done → {report_path}")

    return {
        "record": record,
        "description": record_info["description"],
        "rhythm": primary.name,
        "confidence": primary.confidence,
        "mean_hr": hrv.mean_hr_bpm,
        "sdnn": hrv.std_rr_ms,
        "rmssd": hrv.rmssd_ms,
        "cv": hrv.cv_rr,
        "n_beats": hrv.n_beats,
        "report_path": report_path,
        "quality": quality["quality_label"],
    }


def print_banner():
    banner = r"""
  ╔══════════════════════════════════════════════════════════╗
  ║       ECG Arrhythmia Detection System — Demo             ║
  ║       MIT-BIH Arrhythmia Database (PhysioNet)            ║
  ║       Braden Francis · Wentworth Institute of Technology ║
  ╚══════════════════════════════════════════════════════════╝

  Pipeline:
    Raw ECG → Bandpass Filter → Pan-Tompkins QRS Detection
    → HRV Feature Extraction → Rhythm Classification
    → Clinical HTML Report

  Records to analyze:
"""
    print(banner)
    for i, r in enumerate(DEMO_RECORDS, 1):
        print(f"    {i}. Record {r['record']:>3} — {r['description']}")
    print()


def print_summary(results: list):
    print(f"\n  {'═'*62}")
    print(f"  {'ANALYSIS SUMMARY':^62}")
    print(f"  {'═'*62}")
    print(f"  {'Record':<8} {'Rhythm':<26} {'HR':>6} {'SDNN':>7} {'CV':>7} {'Conf':>6}")
    print(f"  {'─'*62}")
    for r in results:
        rhythm = r['rhythm'][:25]
        print(
            f"  {r['record']:<8} {rhythm:<26} "
            f"{r['mean_hr']:>5.1f}  "
            f"{r['sdnn']:>5.1f}ms  "
            f"{r['cv']:>6.4f}  "
            f"{r['confidence']:>4.0f}%"
        )
    print(f"  {'─'*62}")
    print(f"\n  Reports saved to: {REPORTS_DIR}/")
    print(f"  Opening {len(results)} report(s) in browser...\n")


def main():
    t_total = time.time()
    print_banner()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for record_info in DEMO_RECORDS:
        try:
            result = analyze_record(record_info)
            results.append(result)
        except Exception as e:
            print(f"\n  ERROR processing record {record_info['record']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not results:
        print("\n  No records were successfully analyzed.")
        sys.exit(1)

    print_summary(results)

    # Open all reports in browser
    for r in results:
        path = r["report_path"]
        if path.exists():
            webbrowser.open(f"file://{path.absolute()}")
            time.sleep(0.5)  # Small delay between tabs

    elapsed = time.time() - t_total
    print(f"  Total time: {elapsed:.1f}s\n")
    print("  ✓ Demo complete! Check your browser for the HTML reports.\n")


if __name__ == "__main__":
    main()
