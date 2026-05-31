#!/usr/bin/env python3
"""
train_cnn.py — Train the 1D CNN beat classifier on MIT-BIH Arrhythmia Database.

Usage:
    python train_cnn.py

Significantly higher accuracy than Random Forest by learning beat morphology
directly from raw waveforms using residual 1D convolutions.
"""

import sys
import time
import webbrowser
from pathlib import Path

import numpy as np
import wfdb

# Same record split as Random Forest
TRAIN_RECORDS = [
    '100','101','102','103','104','105','106','107','108','109',
    '111','112','113','114','115','116','117','118','119',
    '121','122','123','124',
]
TEST_RECORDS = [
    '200','201','202','203','205','207','208','209','210',
    '212','213','214','215','217','219','220','221','222',
    '223','228','230','231','232','233','234',
]

DATA_DIR = Path(__file__).parent / 'data'
SKIP_SYMBOLS = {'+','~','|','"','!','x','[',']','{','}','p','t','u','`',"'",'^','Q'}


def load_record(record_id: str):
    from src.ml_classifier import map_annotation_label
    hea = DATA_DIR / f'{record_id}.hea'
    if not hea.exists():
        try:
            wfdb.dl_database('mitdb', dl_dir=str(DATA_DIR), records=[record_id])
        except Exception as e:
            print(f"  Warning: {e}")
            return None
    try:
        record = wfdb.rdrecord(str(DATA_DIR / record_id))
        ann = wfdb.rdann(str(DATA_DIR / record_id), 'atr')
    except Exception as e:
        print(f"  Warning loading {record_id}: {e}")
        return None

    signal = record.p_signal[:, 0].astype(float)
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-9)
    fs = float(record.fs)
    beat_mask = [s not in SKIP_SYMBOLS for s in ann.symbol]
    r_peaks = ann.sample[beat_mask]
    labels = np.array([map_annotation_label(s) for s, m in zip(ann.symbol, beat_mask) if m])

    if len(r_peaks) < 3:
        return None
    return r_peaks, labels, fs, signal


def main():
    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║     MIT-BIH CNN Beat Classifier — Training           ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print()

    from src.cnn_classifier import CNNBeatClassifier
    from src.evaluator import generate_evaluation_report

    # Load all records
    print("  Loading training records...")
    train_data = [r for r in (load_record(i) for i in TRAIN_RECORDS) if r]
    print(f"  Loaded {len(train_data)}/{len(TRAIN_RECORDS)} training records")

    print("\n  Loading test records...")
    test_data = [r for r in (load_record(i) for i in TEST_RECORDS) if r]
    print(f"  Loaded {len(test_data)}/{len(TEST_RECORDS)} test records")

    if not train_data:
        print("  ERROR: No training data.")
        sys.exit(1)

    # Train
    print("\n  Training 1D CNN classifier...")
    t0 = time.time()
    clf = CNNBeatClassifier()
    train_metrics = clf.train(train_data, val_records=test_data[:5], epochs=40, verbose=True)
    print(f"\n  Training completed in {time.time()-t0:.1f}s")

    # Evaluate
    print("\n  Evaluating on held-out test set...")
    test_metrics = clf.evaluate(test_data, verbose=True)

    # Save
    clf.save()

    # Evaluation report
    if 'y_true' in test_metrics:
        print("\n  Generating evaluation report...")
        Path('reports').mkdir(exist_ok=True)
        eval_path = Path('reports/cnn_evaluation_report.html')
        generate_evaluation_report(
            y_true=test_metrics['y_true'],
            y_pred=test_metrics['y_pred'],
            y_proba=test_metrics.get('y_proba'),
            classes=test_metrics.get('classes', []),
            output_path=eval_path,
            train_metrics=train_metrics,
        )
        print(f"  CNN evaluation report → {eval_path}")
        webbrowser.open(f"file://{eval_path.resolve()}")

    # Summary
    print()
    print("  ══════════════════════════════════════════════════")
    print("                CNN TRAINING SUMMARY                ")
    print("  ══════════════════════════════════════════════════")
    print(f"  Architecture:    Residual 1D CNN (4 blocks)")
    print(f"  Input length:    128 samples (resampled beat)")
    print(f"  Training beats:  {train_metrics['n_beats']:,} (with augmentation)")
    print(f"  Train accuracy:  {train_metrics['train_accuracy']:.1%}")
    if test_metrics:
        print(f"  Test beats:      {test_metrics['n_beats']:,}")
        print(f"  Test accuracy:   {test_metrics['test_accuracy']:.1%}")
    print(f"  Model saved to:  models/cnn_classifier.pt")
    print()
    print("  Run 'python demo.py' — the demo will auto-use the CNN if available.")
    print()


if __name__ == '__main__':
    main()
