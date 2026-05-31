"""
ml_classifier.py — Machine learning beat classifier trained on MIT-BIH ground truth.

Uses a Random Forest classifier trained on per-beat RR interval features
extracted from the MIT-BIH Arrhythmia Database annotations.

Beat classes:
    N — Normal sinus beat
    V — Premature ventricular contraction (PVC)
    A — Atrial premature beat
    L — Left bundle branch block beat
    R — Right bundle branch block beat
    F — Fusion beat
    Q — Unknown / other (catch-all)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# MIT-BIH beat label mapping → our 7 classes
BEAT_LABEL_MAP = {
    'N': 'N', '.': 'N', 'n': 'N',   # Normal
    'V': 'V', 'E': 'V',              # PVC / ventricular ectopic
    'A': 'A', 'a': 'A', 'S': 'A', 'J': 'A', 'e': 'A',  # Atrial ectopic
    'L': 'L',                         # Left BBB
    'R': 'R',                         # Right BBB
    'F': 'F',                         # Fusion
}
CLASSES = ['N', 'V', 'A', 'L', 'R', 'F', 'Q']
CLASS_NAMES = {
    'N': 'Normal',
    'V': 'PVC',
    'A': 'Atrial Ectopic',
    'L': 'Left BBB',
    'R': 'Right BBB',
    'F': 'Fusion',
    'Q': 'Other',
}

MODELS_DIR = Path(__file__).parent.parent / 'models'


@dataclass
class BeatPrediction:
    """Per-beat classification result."""
    beat_index: int          # Index into R-peak array
    sample_index: int        # Sample index in ECG signal
    predicted_class: str     # N, V, A, L, R, F, Q
    class_name: str          # Human-readable
    confidence: float        # Max class probability (0–1)
    probabilities: Dict[str, float]  # All class probs


@dataclass
class MLClassificationResult:
    """Full ML classification for a record."""
    beat_predictions: List[BeatPrediction]
    class_counts: Dict[str, int]
    dominant_class: str
    pvc_burden_pct: float
    abnormal_burden_pct: float
    model_version: str


def extract_beat_template(
    r_peaks: np.ndarray,
    fs: float,
    signal: np.ndarray,
    pre_ms: float = 90.0,
    post_ms: float = 150.0,
    n_points: int = 64,
) -> np.ndarray:
    """
    Extract fixed-length normalized beat templates centered on each R-peak.
    Resamples each beat window to n_points for a uniform feature vector.

    Parameters
    ----------
    r_peaks  : R-peak sample indices
    fs       : sampling frequency
    signal   : ECG signal array
    pre_ms   : ms before R-peak to include
    post_ms  : ms after R-peak to include
    n_points : output template length (resampled)

    Returns
    -------
    np.ndarray, shape (n_beats, n_points)
    """
    from scipy.signal import resample

    pre_samp = int(pre_ms * fs / 1000)
    post_samp = int(post_ms * fs / 1000)
    templates = np.zeros((len(r_peaks), n_points))

    for i, r in enumerate(r_peaks):
        lo = r - pre_samp
        hi = r + post_samp
        if lo < 0 or hi > len(signal):
            continue
        win = signal[lo:hi].copy()
        # Normalize: zero-mean, unit std
        mu, sigma = np.mean(win), np.std(win)
        if sigma > 1e-9:
            win = (win - mu) / sigma
        # Resample to fixed length
        templates[i] = resample(win, n_points)

    return templates


def extract_beat_features(
    r_peaks: np.ndarray,
    fs: float,
    signal: np.ndarray = None,
) -> np.ndarray:
    """
    Extract per-beat features for classification.

    RR interval features (7):
        0: RR interval (ms)
        1: pre-RR (ms)
        2: RR ratio (RR / pre-RR)
        3: local mean RR (ms)
        4: norm RR (RR / local mean)
        5: pre-RR ratio (pre-RR / local mean)
        6: RR difference (RR - pre-RR)

    Morphological features from ECG waveform (8, if signal provided):
        7:  R amplitude (normalized)
        8:  QRS width estimate (samples)
        9:  pre-R slope (mean signal derivative before peak)
        10: post-R slope
        11: template correlation (similarity to mean beat)
        12: beat energy
        13: skewness of beat window
        14: kurtosis of beat window

    Parameters
    ----------
    r_peaks : np.ndarray — sample indices of R-peaks
    fs : float — sampling frequency
    signal : np.ndarray or None — ECG signal for morphological features

    Returns
    -------
    np.ndarray, shape (n_beats, 7) or (n_beats, 15)
    """
    from scipy.stats import skew, kurtosis as sp_kurtosis

    n = len(r_peaks)
    if n < 3:
        n_feat = 7 if signal is None else 15
        return np.zeros((n, n_feat))

    rr_samples = np.diff(r_peaks).astype(float)
    rr_ms = rr_samples * 1000.0 / fs
    mean_rr = np.mean(rr_ms)

    # ── RR features ───────────────────────────────────────────────────────────
    rr_features = np.zeros((n, 7))
    for i in range(n):
        rr = rr_ms[i] if i < n - 1 else mean_rr
        pre_rr = rr_ms[i - 1] if i > 0 else mean_rr
        lo = max(0, i - 2)
        hi = min(len(rr_ms), i + 3)
        local_mean = np.mean(rr_ms[lo:hi]) if hi > lo else mean_rr

        rr_features[i, 0] = rr
        rr_features[i, 1] = pre_rr
        rr_features[i, 2] = rr / pre_rr if pre_rr > 0 else 1.0
        rr_features[i, 3] = local_mean
        rr_features[i, 4] = rr / local_mean if local_mean > 0 else 1.0
        rr_features[i, 5] = pre_rr / local_mean if local_mean > 0 else 1.0
        rr_features[i, 6] = rr - pre_rr

    if signal is None:
        return rr_features

    # ── Morphological features ─────────────────────────────────────────────
    # Window: 100ms before and after R-peak
    half_win = int(0.1 * fs)
    morph_features = np.zeros((n, 8))

    # Build mean beat template for correlation
    templates = []
    for i, r in enumerate(r_peaks):
        lo = r - half_win
        hi = r + half_win
        if lo >= 0 and hi < len(signal):
            win = signal[lo:hi]
            templates.append(win)
    if templates:
        mean_template = np.mean(templates, axis=0)
    else:
        mean_template = None

    for i, r in enumerate(r_peaks):
        lo = r - half_win
        hi = r + half_win
        if lo < 0 or hi >= len(signal):
            continue
        win = signal[lo:hi]
        r_amp = signal[r]

        # Normalize window
        win_std = np.std(win)
        win_norm = (win - np.mean(win)) / (win_std + 1e-9)

        # Pre/post slopes
        pre_slope = np.mean(np.diff(signal[max(0, r - half_win):r]))
        post_slope = np.mean(np.diff(signal[r:min(len(signal), r + half_win)]))

        # QRS width: samples where signal > 50% of R amplitude
        threshold = 0.5 * abs(r_amp)
        qrs_mask = np.abs(signal[lo:hi]) > threshold
        qrs_width = float(np.sum(qrs_mask))

        # Template correlation
        if mean_template is not None and len(win) == len(mean_template):
            mt_std = np.std(mean_template)
            mt_norm = (mean_template - np.mean(mean_template)) / (mt_std + 1e-9)
            corr = float(np.corrcoef(win_norm, mt_norm)[0, 1])
        else:
            corr = 0.0

        morph_features[i, 0] = r_amp
        morph_features[i, 1] = qrs_width
        morph_features[i, 2] = pre_slope
        morph_features[i, 3] = post_slope
        morph_features[i, 4] = corr
        morph_features[i, 5] = float(np.sum(win ** 2))  # energy
        morph_features[i, 6] = float(skew(win))
        morph_features[i, 7] = float(sp_kurtosis(win))

    return np.hstack([rr_features, morph_features])


def map_annotation_label(symbol: str) -> str:
    """Map a raw MIT-BIH annotation symbol to our class set."""
    return BEAT_LABEL_MAP.get(symbol, 'Q')


class MLBeatClassifier:
    """
    Random Forest beat classifier trained on MIT-BIH annotations.

    Usage
    -----
    # Training (run train.py)
    clf = MLBeatClassifier()
    clf.train(records_train, records_test)
    clf.save()

    # Inference
    clf = MLBeatClassifier.load()
    result = clf.predict(r_peaks, fs)
    """

    MODEL_FILE = MODELS_DIR / 'rf_classifier.joblib'
    VERSION = '1.0'

    def __init__(self):
        self.model = None
        self.classes_ = CLASSES
        self._is_fitted = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        train_records: List,
        verbose: bool = True,
    ) -> Dict:
        """
        Train the Random Forest classifier.

        Parameters
        ----------
        train_records : list of (r_peaks, labels, fs)
            Each element is a tuple of R-peak sample indices, beat label array,
            and sampling frequency.
        verbose : bool
            Print progress.

        Returns
        -------
        dict with training metrics
        """
        from sklearn.ensemble import RandomForestClassifier

        X_all, y_all = [], []

        for record in train_records:
            if len(record) == 4:
                r_peaks, labels, fs, signal = record
            else:
                r_peaks, labels, fs = record
                signal = None
            if len(r_peaks) < 3:
                continue
            features = extract_beat_features(r_peaks, fs, signal)
            # Append beat templates if signal available
            if signal is not None:
                templates = extract_beat_template(r_peaks, fs, signal)
                features = np.hstack([features, templates])
            if len(features) != len(labels):
                min_len = min(len(features), len(labels))
                features = features[:min_len]
                labels = labels[:min_len]
            X_all.append(features)
            y_all.extend(labels)

        X = np.vstack(X_all)
        y = np.array(y_all)

        if verbose:
            unique, counts = np.unique(y, return_counts=True)
            print(f"\n  Training set: {len(X):,} beats across {len(train_records)} records")
            for cls, cnt in zip(unique, counts):
                print(f"    {CLASS_NAMES.get(cls, cls):20s}: {cnt:5,} beats ({100*cnt/len(y):.1f}%)")

        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X, y)
        self._is_fitted = True

        # Training accuracy
        train_acc = self.model.score(X, y)
        if verbose:
            print(f"\n  Training accuracy: {train_acc:.3f}")

        return {'train_accuracy': train_acc, 'n_beats': len(X)}

    def evaluate(
        self,
        test_records: List[Tuple[np.ndarray, np.ndarray, float]],
        verbose: bool = True,
    ) -> Dict:
        """Evaluate on held-out test records."""
        from sklearn.metrics import classification_report, accuracy_score

        X_all, y_all = [], []
        for record in test_records:
            if len(record) == 4:
                r_peaks, labels, fs, signal = record
            else:
                r_peaks, labels, fs = record
                signal = None
            if len(r_peaks) < 3:
                continue
            features = extract_beat_features(r_peaks, fs, signal)
            if signal is not None:
                templates = extract_beat_template(r_peaks, fs, signal)
                features = np.hstack([features, templates])
            min_len = min(len(features), len(labels))
            X_all.append(features[:min_len])
            y_all.extend(labels[:min_len])

        X = np.vstack(X_all)
        y = np.array(y_all)

        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)
        acc = accuracy_score(y, y_pred)

        if verbose:
            print(f"\n  Test set: {len(X):,} beats across {len(test_records)} records")
            print(f"  Overall accuracy: {acc:.3f} ({acc*100:.1f}%)\n")
            present = sorted(set(y) | set(y_pred))
            target_names = [CLASS_NAMES.get(c, c) for c in present]
            print(classification_report(y, y_pred, labels=present,
                                        target_names=target_names, zero_division=0))

        return {
            'test_accuracy': acc,
            'n_beats': len(X),
            'y_true': y,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'classes': list(self.model.classes_),
        }

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, r_peaks: np.ndarray, fs: float, signal: np.ndarray = None) -> MLClassificationResult:
        """
        Classify beats in a record.

        Parameters
        ----------
        r_peaks : np.ndarray
            Sample indices of R-peaks.
        fs : float
            Sampling frequency.

        Returns
        -------
        MLClassificationResult
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Run train() or load() first.")

        features = extract_beat_features(r_peaks, fs, signal)
        if signal is not None:
            templates = extract_beat_template(r_peaks, fs, signal)
            features = np.hstack([features, templates])
        probas = self.model.predict_proba(features)
        classes = self.model.classes_

        predictions = []
        for i, (proba_row, r_idx) in enumerate(zip(probas, r_peaks)):
            best_idx = np.argmax(proba_row)
            predicted = classes[best_idx]
            prob_dict = {c: float(p) for c, p in zip(classes, proba_row)}
            predictions.append(BeatPrediction(
                beat_index=i,
                sample_index=int(r_idx),
                predicted_class=predicted,
                class_name=CLASS_NAMES.get(predicted, predicted),
                confidence=float(proba_row[best_idx]),
                probabilities=prob_dict,
            ))

        # Summary stats
        class_counts = {c: 0 for c in CLASSES}
        for p in predictions:
            class_counts[p.predicted_class] = class_counts.get(p.predicted_class, 0) + 1

        n_total = len(predictions)
        dominant = max(class_counts, key=class_counts.get) if predictions else 'Q'
        pvc_pct = 100.0 * class_counts.get('V', 0) / n_total if n_total > 0 else 0.0
        abnormal_count = sum(v for k, v in class_counts.items() if k != 'N')
        abnormal_pct = 100.0 * abnormal_count / n_total if n_total > 0 else 0.0

        return MLClassificationResult(
            beat_predictions=predictions,
            class_counts=class_counts,
            dominant_class=dominant,
            pvc_burden_pct=pvc_pct,
            abnormal_burden_pct=abnormal_pct,
            model_version=self.VERSION,
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> None:
        """Save the trained model to disk."""
        import joblib
        save_path = path or self.MODEL_FILE
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': self.model, 'version': self.VERSION}, save_path)
        print(f"  Model saved → {save_path}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> 'MLBeatClassifier':
        """Load a trained model from disk."""
        import joblib
        load_path = path or cls.MODEL_FILE
        if not load_path.exists():
            raise FileNotFoundError(f"No model at {load_path}. Run train.py first.")
        data = joblib.load(load_path)
        instance = cls()
        instance.model = data['model']
        instance.VERSION = data.get('version', '1.0')
        instance._is_fitted = True
        return instance

    @classmethod
    def is_available(cls) -> bool:
        """Check if a trained model exists on disk."""
        return cls.MODEL_FILE.exists()
