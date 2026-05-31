#!/usr/bin/env python3
"""
train.py — Train and evaluate the ML beat classifier on MIT-BIH Arrhythmia Database.

Usage:
    python train.py

Downloads all MIT-BIH records (if not cached), extracts beat features using
ground-truth annotations, trains a Random Forest classifier, evaluates on a
held-out test set, and saves the model to models/rf_classifier.joblib.

Train/test split is record-level (not beat-level) to prevent data leakage.
"""

import sys
import time
from pathlib import Path

import numpy as np
import wfdb

# ── Record sets ───────────────────────────────────────────────────────────────
# Record-level split: train on lower numbers, test on higher
TRAIN_RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119',
    '121', '122', '123', '124',
]
TEST_RECORDS = [
    '200', '201', '202', '203', '205', '207', '208', '209', '210',
    '212', '213', '214', '215', '217', '219', '220', '221', '222',
    '223', '228', '230', '231', '232', '233', '234',
]

DATA_DIR = Path(__file__).parent / 'data'

# Beat symbols to SKIP (non-beat annotations: noise, rhythm markers, etc.)
SKIP_SYMBOLS = {'+', '~', '|', '"', '!', 'x', '[', ']', '{', '}', 'p', 't', 'u', '`', "'", '^', 'Q'}


def download_record(record_id: str) -> bool:
    """Download a MIT-BIH record if not already cached."""
    hea_path = DATA_DIR / f'{record_id}.hea'
    if hea_path.exists():
        return True
    try:
        wfdb.dl_database('mitdb', dl_dir=str(DATA_DIR), records=[record_id])
        return True
    except Exception as e:
        print(f"    Warning: could not download record {record_id}: {e}")
        return False


def load_record_with_annotations(record_id: str):
    """
    Load ECG record and beat annotations.

    Returns
    -------
    (r_peaks, labels, fs, signal) or None if unavailable
    """
    from src.ml_classifier import map_annotation_label

    hea_path = DATA_DIR / f'{record_id}.hea'
    if not hea_path.exists():
        return None

    try:
        record = wfdb.rdrecord(str(DATA_DIR / record_id))
        ann = wfdb.rdann(str(DATA_DIR / record_id), 'atr')
    except Exception as e:
        print(f"    Warning: error loading {record_id}: {e}")
        return None

    fs = record.fs
    # Use first channel, normalize
    signal = record.p_signal[:, 0].astype(float)
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-9)

    symbols = ann.symbol
    samples = ann.sample

    # Filter to beat annotations only
    beat_mask = [s not in SKIP_SYMBOLS for s in symbols]
    beat_samples = samples[beat_mask]
    beat_symbols = [s for s, m in zip(symbols, beat_mask) if m]

    if len(beat_samples) < 3:
        return None

    labels = np.array([map_annotation_label(s) for s in beat_symbols])
    return beat_samples, labels, float(fs), signal


def main():
    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║     MIT-BIH Beat Classifier — Training Pipeline      ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print()

    DATA_DIR.mkdir(exist_ok=True)

    from src.ml_classifier import MLBeatClassifier

    # ── Download records ──────────────────────────────────────────────────────
    all_records = TRAIN_RECORDS + TEST_RECORDS
    print(f"  Downloading {len(all_records)} MIT-BIH records (skipping cached)...")
    downloaded = 0
    for rec_id in all_records:
        if download_record(rec_id):
            downloaded += 1
        sys.stdout.write(f"\r  {downloaded}/{len(all_records)} records ready")
        sys.stdout.flush()
    print(f"\r  {downloaded}/{len(all_records)} records ready          ")

    # ── Load training data ────────────────────────────────────────────────────
    print("\n  Loading training records...")
    train_data = []
    for rec_id in TRAIN_RECORDS:
        result = load_record_with_annotations(rec_id)
        if result is not None:
            train_data.append(result)

    print(f"  Loaded {len(train_data)}/{len(TRAIN_RECORDS)} training records")

    # ── Load test data ────────────────────────────────────────────────────────
    print("\n  Loading test records...")
    test_data = []
    for rec_id in TEST_RECORDS:
        result = load_record_with_annotations(rec_id)
        if result is not None:
            test_data.append(result)

    print(f"  Loaded {len(test_data)}/{len(TEST_RECORDS)} test records")

    if not train_data:
        print("\n  ERROR: No training data available. Check your internet connection.")
        sys.exit(1)

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n  Training Random Forest classifier...")
    t0 = time.time()
    clf = MLBeatClassifier()
    train_metrics = clf.train(train_data, verbose=True)
    elapsed = time.time() - t0
    print(f"\n  Training completed in {elapsed:.1f}s")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    if test_data:
        print("\n  Evaluating on held-out test set...")
        test_metrics = clf.evaluate(test_data, verbose=True)
    else:
        print("\n  Warning: No test data — skipping evaluation")
        test_metrics = {}

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\n  Saving model...")
    clf.save()

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("  ══════════════════════════════════════════════════")
    print("                    TRAINING SUMMARY                ")
    print("  ══════════════════════════════════════════════════")
    print(f"  Training beats:  {train_metrics['n_beats']:,}")
    print(f"  Train accuracy:  {train_metrics['train_accuracy']:.1%}")
    if test_metrics:
        print(f"  Test beats:      {test_metrics['n_beats']:,}")
        print(f"  Test accuracy:   {test_metrics['test_accuracy']:.1%}")
    print(f"  Model saved to:  models/rf_classifier.joblib")
    print()
    print("  Run 'python demo.py' to see ML-enhanced reports.")
    print()


if __name__ == '__main__':
    main()
