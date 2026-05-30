#!/usr/bin/env python3
"""
analyze.py — Main CLI for ECG arrhythmia detection.

Usage:
  python analyze.py --record 100
  python analyze.py --record 200 --duration 120 --verbose
  python analyze.py --record 203 --output reports/ --no-open
"""

import sys
import time
import webbrowser
from pathlib import Path

import click
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.loader import load_record, get_annotation_beats
from src.preprocessor import preprocess, estimate_signal_quality
from src.detector import detect_qrs, compute_detection_accuracy
from src.features import extract_features
from src.classifier import classify_rhythm
from src.visualizer import generate_all_plots
from src.reporter import generate_report


@click.command()
@click.option("--record", "-r", required=True,
              help="MIT-BIH record number (e.g. 100, 200, 203, 208)")
@click.option("--output", "-o", default="reports",
              help="Output directory for HTML report [default: reports/]")
@click.option("--duration", "-d", default=None, type=float,
              help="Seconds of signal to analyze [default: full record]")
@click.option("--start", "-s", default=0.0, type=float,
              help="Start offset in seconds [default: 0]")
@click.option("--channel", "-c", default=0, type=int,
              help="Signal channel to analyze [default: 0 (MLII)]")
@click.option("--fs", default=None, type=float,
              help="Override sampling frequency [default: from record header]")
@click.option("--no-notch", is_flag=True, default=False,
              help="Skip 60 Hz notch filter")
@click.option("--no-open", is_flag=True, default=False,
              help="Do not auto-open report in browser")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Print detailed processing information")
def analyze(record, output, duration, start, channel, fs, no_notch, no_open, verbose):
    """
    Analyze an ECG record from the MIT-BIH Arrhythmia Database.

    Downloads the record if not already cached locally, runs the full
    signal processing and arrhythmia detection pipeline, and generates
    a clinical HTML report.

    Record suggestions:
      100 — Normal sinus rhythm (clean signal)
      200 — Mixed rhythms, PVCs
      203 — Complex arrhythmias
      208 — AF present
      232 — Atrial flutter
    """
    t_start = time.time()
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"\n{'═'*60}")
    click.echo(f"  ECG Arrhythmia Detection System")
    click.echo(f"  MIT-BIH Record: {record}")
    click.echo(f"{'═'*60}\n")

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    click.echo("📂 Loading ECG data...")
    ecg = load_record(
        record_name=record,
        channel=channel,
        duration_sec=duration,
        start_sec=start,
    )
    if fs:
        ecg = ecg.__class__(**{**ecg.__dict__, "fs": fs})

    click.echo(f"   ✓ Record {record} loaded: {ecg.duration_sec:.1f}s at {ecg.fs} Hz "
               f"({ecg.n_samples:,} samples, lead: {ecg.lead_name})")

    if verbose:
        click.echo(f"   Signal range: [{ecg.signal.min():.3f}, {ecg.signal.max():.3f}] {ecg.units}")

    # ── Step 2: Signal Quality ────────────────────────────────────────────────
    click.echo("\n🔍 Assessing signal quality...")
    quality = estimate_signal_quality(ecg.signal, ecg.fs)
    click.echo(f"   ✓ Quality: {quality['quality_label']} ({quality['score']}/100)")
    if quality["flatline"]:
        click.echo("   ⚠ WARNING: Flatline signal detected!", err=True)
    if quality["clipping"]:
        click.echo("   ⚠ WARNING: Signal clipping detected!", err=True)

    # ── Step 3: Preprocessing ─────────────────────────────────────────────────
    click.echo("\n🔧 Preprocessing signal...")
    filtered = preprocess(ecg.signal, ecg.fs, apply_notch_filter=not no_notch)
    click.echo(f"   ✓ Bandpass (0.5–40 Hz) + {'60 Hz notch' if not no_notch else 'no notch'} applied")

    # ── Step 4: QRS Detection ─────────────────────────────────────────────────
    click.echo("\n⚡ Detecting QRS complexes (Pan-Tompkins)...")
    detection = detect_qrs(filtered.filtered, ecg.fs)
    click.echo(f"   ✓ {detection.n_beats} beats detected | "
               f"Mean HR: {detection.mean_hr_bpm:.1f} BPM | "
               f"Mean RR: {detection.mean_rr_sec * 1000:.1f}ms")

    if verbose and detection.n_beats > 0:
        click.echo(f"   RR range: [{detection.rr_intervals_sec.min()*1000:.1f}, "
                   f"{detection.rr_intervals_sec.max()*1000:.1f}]ms")

    # Compare against ground truth if annotations available
    if ecg.annotations:
        gt = get_annotation_beats(ecg)
        if len(gt["samples"]) > 0:
            acc = compute_detection_accuracy(detection.r_peaks, gt["samples"], ecg.fs)
            click.echo(f"   ✓ Accuracy vs ground truth: "
                       f"Se={acc['sensitivity']:.3f} | "
                       f"PPV={acc['ppv']:.3f} | "
                       f"F1={acc['f1']:.3f} "
                       f"(TP={acc['tp']}, FP={acc['fp']}, FN={acc['fn']})")

    # ── Step 5: Feature Extraction ────────────────────────────────────────────
    click.echo("\n📊 Extracting HRV features...")
    hrv = extract_features(
        r_peaks=detection.r_peaks,
        rr_intervals_sec=detection.rr_intervals_sec,
        r_peak_amplitudes=detection.r_peak_amplitudes,
        filtered_signal=filtered.filtered,
        fs=ecg.fs,
        duration_sec=ecg.duration_sec,
    )
    click.echo(f"   ✓ SDNN={hrv.std_rr_ms:.1f}ms | RMSSD={hrv.rmssd_ms:.1f}ms | "
               f"CV={hrv.cv_rr:.4f} | pNN50={hrv.pnn50_pct:.1f}%")

    # ── Step 6: Classification ────────────────────────────────────────────────
    click.echo("\n🫀 Classifying rhythm...")
    classification = classify_rhythm(hrv, detection.rr_intervals_sec)
    primary = classification.primary_rhythm
    click.echo(f"   ✓ Primary: {primary.name} (confidence: {primary.confidence:.0f}%)")
    if classification.secondary_findings:
        for sf in classification.secondary_findings[:3]:
            click.echo(f"   + {sf.name} ({sf.confidence:.0f}%)")

    # ── Step 7: Visualizations ────────────────────────────────────────────────
    click.echo("\n🎨 Generating visualizations...")
    plots = generate_all_plots(
        filtered=filtered,
        detection=detection,
        hrv=hrv,
        classification=classification,
        record_name=str(record),
        output_dir=None,  # Don't save PNGs separately; embed in HTML
    )
    click.echo(f"   ✓ {len(plots)} plots generated")

    # ── Step 8: Report ────────────────────────────────────────────────────────
    click.echo("\n📄 Generating HTML report...")
    report_path = output_dir / f"ecg_report_{record}.html"
    generate_report(
        record_name=str(record),
        lead_name=ecg.lead_name,
        filtered=filtered,
        detection=detection,
        hrv=hrv,
        classification=classification,
        plots=plots,
        signal_quality=quality,
        output_path=report_path,
    )
    click.echo(f"   ✓ Report saved: {report_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    click.echo(f"\n{'═'*60}")
    click.echo(f"  Analysis complete in {elapsed:.1f}s")
    click.echo(f"  Record:    {record} ({ecg.duration_sec:.1f}s, {ecg.n_samples:,} samples)")
    click.echo(f"  Rhythm:    {primary.name} ({primary.confidence:.0f}% confidence)")
    click.echo(f"  HR:        {hrv.mean_hr_bpm:.1f} BPM")
    click.echo(f"  Report:    {report_path.absolute()}")
    click.echo(f"{'═'*60}\n")

    # Auto-open in browser
    if not no_open:
        webbrowser.open(f"file://{report_path.absolute()}")

    return report_path


if __name__ == "__main__":
    analyze()
