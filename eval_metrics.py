#!/usr/bin/env python3
"""
eval_metrics.py — Quick evaluation of pre-trained RF and CNN classifiers.
Outputs per-class precision/recall/F1 for resume bullet documentation.
"""

import sys
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from train import TEST_RECORDS, DATA_DIR, load_record_with_annotations, SKIP_SYMBOLS

def evaluate_rf():
    print("\n=== Random Forest Classifier ===")
    from src.ml_classifier import MLBeatClassifier, CLASS_NAMES

    clf = MLBeatClassifier.load()
    print(f"  Model loaded: {clf.model.__class__.__name__}")

    print(f"\n  Loading {len(TEST_RECORDS)} test records...")
    test_data = []
    for rec_id in TEST_RECORDS:
        result = load_record_with_annotations(rec_id)
        if result is not None:
            test_data.append(result)
    print(f"  Loaded {len(test_data)} records")

    metrics = clf.evaluate(test_data, verbose=True)
    return metrics


def evaluate_cnn():
    print("\n=== CNN Classifier ===")
    try:
        from src.cnn_classifier import CNNBeatClassifier
        import torch
    except ImportError as e:
        print(f"  CNN not available: {e}")
        return None

    model_path = Path(__file__).parent / "models" / "cnn_classifier.pt"
    le_path = Path(__file__).parent / "models" / "cnn_classifier.le"
    if not model_path.exists():
        print("  CNN model not found")
        return None

    try:
        clf = CNNBeatClassifier.load(model_path, le_path)
        print(f"  CNN model loaded")
    except Exception as e:
        print(f"  Error loading CNN: {e}")
        return None

    print(f"\n  Loading {len(TEST_RECORDS)} test records...")
    test_data = []
    for rec_id in TEST_RECORDS:
        result = load_record_with_annotations(rec_id)
        if result is not None:
            test_data.append(result)
    print(f"  Loaded {len(test_data)} records")

    try:
        from sklearn.metrics import classification_report, accuracy_score

        X_all, y_all = [], []
        for record in test_data:
            r_peaks, labels, fs, signal = record
            try:
                preds = clf.predict_beats(r_peaks, fs, signal)
                if preds is not None:
                    min_len = min(len(preds), len(labels))
                    X_all.extend(preds[:min_len])
                    y_all.extend(labels[:min_len])
            except Exception:
                continue

        if not y_all:
            print("  No predictions generated")
            return None

        print(f"\n  Test set: {len(y_all):,} beats")
        print(f"  Overall accuracy: {accuracy_score(y_all, X_all):.1%}\n")
        print(classification_report(y_all, X_all, zero_division=0))

    except Exception as e:
        print(f"  CNN evaluation error: {e}")

    return None


if __name__ == "__main__":
    print("\n  ECG Arrhythmia Classifier — Evaluation Report")
    print("  " + "=" * 50)

    rf_metrics = evaluate_rf()

    # Try CNN
    evaluate_cnn()

    print("\n" + "=" * 52)
    print("RESUME BULLET METRICS (copy these)")
    print("=" * 52)
    if rf_metrics:
        from sklearn.metrics import classification_report
        from src.ml_classifier import CLASS_NAMES
        y_true = rf_metrics['y_true']
        y_pred = rf_metrics['y_pred']
        acc = rf_metrics['test_accuracy']
        print(f"\nRF Overall Accuracy: {acc:.1%}")
        print(f"Test beats: {rf_metrics['n_beats']:,}")
        report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        print("\nPer-class F1 (RF):")
        key_classes = {'N': 'Normal', 'V': 'PVC', 'A': 'Atrial Ectopic', 'L': 'Left BBB', 'R': 'Right BBB'}
        for code, name in key_classes.items():
            if code in report:
                r = report[code]
                n = int(r['support'])
                print(f"  {name:20s}: P={r['precision']:.2f}  R={r['recall']:.2f}  F1={r['f1-score']:.2f}  (n={n:,})")
        if 'macro avg' in report:
            r = report['macro avg']
            print(f"\n  Macro avg          : P={r['precision']:.2f}  R={r['recall']:.2f}  F1={r['f1-score']:.2f}")
        if 'weighted avg' in report:
            r = report['weighted avg']
            print(f"  Weighted avg       : P={r['precision']:.2f}  R={r['recall']:.2f}  F1={r['f1-score']:.2f}")
