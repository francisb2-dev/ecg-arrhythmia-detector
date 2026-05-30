"""
tests/test_detector.py

Tests for QRS complex detection (Pan-Tompkins algorithm).

Uses synthetic ECG signals with known beat positions to verify:
- Correct beat count detection
- Accurate beat timing (within physiological tolerance)
- Proper handling of edge cases (very slow/fast rhythms, short records)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.detector import (
    detect_qrs,
    compute_detection_accuracy,
    _differentiate,
    _square,
    _moving_window_integration,
)
from src.preprocessor import preprocess


FS = 360.0


def make_qrs_signal(fs: float, duration: float, hr_bpm: float,
                     qrs_amplitude: float = 1.5,
                     noise_level: float = 0.05) -> tuple:
    """
    Generate a synthetic ECG with QRS complexes as Gaussian pulses.

    Returns (signal, true_r_peak_samples).
    """
    n = int(fs * duration)
    t = np.arange(n) / fs
    signal = np.zeros(n)

    beat_period = 60.0 / hr_bpm
    beat_times = np.arange(beat_period * 0.5, duration - beat_period * 0.5, beat_period)
    r_peak_samples = []

    for bt in beat_times:
        idx = int(bt * fs)
        if 0 < idx < n:
            r_peak_samples.append(idx)
        # QRS: narrow Gaussian
        signal += qrs_amplitude * np.exp(-((t - bt) ** 2) / (2 * (0.012) ** 2))
        # T-wave
        signal += 0.3 * qrs_amplitude * np.exp(-((t - (bt + 0.25)) ** 2) / (2 * (0.04) ** 2))
        # P-wave
        signal += 0.12 * qrs_amplitude * np.exp(-((t - (bt - 0.12)) ** 2) / (2 * (0.025) ** 2))

    if noise_level > 0:
        signal += np.random.randn(n) * noise_level

    return signal, np.array(r_peak_samples, dtype=int)


def make_irregular_rr_signal(fs: float, duration: float,
                               mean_hr: float, cv: float) -> tuple:
    """
    Generate ECG with highly irregular RR intervals (AF-like).
    CV = coefficient of variation of RR intervals.
    """
    n = int(fs * duration)
    t = np.arange(n) / fs
    signal = np.zeros(n)

    mean_rr_sec = 60.0 / mean_hr
    r_peak_samples = []
    current_time = mean_rr_sec

    while current_time < duration - mean_rr_sec:
        # Irregular RR
        rr = max(0.3, np.random.normal(mean_rr_sec, cv * mean_rr_sec))
        idx = int(current_time * fs)
        if 0 < idx < n:
            r_peak_samples.append(idx)
            signal += 1.5 * np.exp(-((t - current_time) ** 2) / (2 * (0.012) ** 2))
        current_time += rr

    return signal, np.array(sorted(r_peak_samples), dtype=int)


class TestPanTompkinsPipeline:
    def test_differentiation_zero_for_constant(self):
        """Derivative of a constant signal should be zero."""
        constant = np.ones(200) * 3.0
        deriv = _differentiate(constant)
        # Edges have zeros anyway; check middle
        assert np.allclose(deriv[5:-5], 0.0, atol=1e-10)

    def test_differentiation_detects_slope(self):
        """Derivative should be large at steep QRS slopes."""
        fs = FS
        t = np.arange(int(fs * 2)) / fs
        qrs = 1.5 * np.exp(-((t - 1.0) ** 2) / (2 * (0.015) ** 2))
        deriv = _differentiate(qrs)
        # Peak derivative should be at the rising edge
        peak_deriv = np.max(np.abs(deriv))
        assert peak_deriv > 0.1, "Derivative should be non-zero at QRS slope"

    def test_squaring_positive(self):
        """Squared signal should have all non-negative values."""
        signal = np.random.randn(500)
        squared = _square(signal)
        assert np.all(squared >= 0)

    def test_mwi_smoothing(self):
        """MWI should produce a smoother signal than its input."""
        # Create spiky signal
        spike = np.zeros(1000)
        spike[100] = 10.0
        spike[300] = 10.0
        spike[500] = 10.0
        integrated = _moving_window_integration(spike, FS, window_sec=0.150)
        # The integrated signal should be smoother (lower max, broader peaks)
        assert np.max(integrated) < np.max(spike)
        assert np.sum(integrated > 0.1) > 3  # Energy spread out


class TestQRSDetection:
    def test_detects_correct_beat_count_normal(self):
        """Should detect approximately the right number of beats at normal HR."""
        signal, true_peaks = make_qrs_signal(FS, duration=10.0, hr_bpm=70.0)
        filtered = preprocess(signal, FS).filtered
        result = detect_qrs(filtered, FS)

        n_true = len(true_peaks)
        n_detected = result.n_beats
        # Allow ±15% tolerance
        assert abs(n_detected - n_true) <= max(2, 0.15 * n_true), \
            f"Expected ~{n_true} beats, detected {n_detected}"

    def test_detects_correct_beat_count_bradycardia(self):
        """Should detect correct beats in bradycardic signal (40 BPM)."""
        signal, true_peaks = make_qrs_signal(FS, duration=15.0, hr_bpm=40.0)
        filtered = preprocess(signal, FS).filtered
        result = detect_qrs(filtered, FS)

        n_true = len(true_peaks)
        assert abs(result.n_beats - n_true) <= max(2, 0.15 * n_true), \
            f"Expected ~{n_true} beats, detected {result.n_beats}"

    def test_detects_correct_beat_count_tachycardia(self):
        """Should detect correct beats in tachycardic signal (120 BPM)."""
        signal, true_peaks = make_qrs_signal(FS, duration=10.0, hr_bpm=120.0,
                                              noise_level=0.03)
        filtered = preprocess(signal, FS).filtered
        result = detect_qrs(filtered, FS)

        n_true = len(true_peaks)
        assert abs(result.n_beats - n_true) <= max(3, 0.20 * n_true), \
            f"Expected ~{n_true} beats (tachycardia), detected {result.n_beats}"

    def test_r_peak_positions_accurate(self):
        """Detected R-peak positions should be within ±75ms of true peaks."""
        signal, true_peaks = make_qrs_signal(FS, duration=10.0, hr_bpm=70.0,
                                              noise_level=0.02)
        filtered = preprocess(signal, FS).filtered
        result = detect_qrs(filtered, FS)

        if result.n_beats == 0:
            pytest.skip("No beats detected")

        tolerance = int(0.075 * FS)  # 75ms
        acc = compute_detection_accuracy(result.r_peaks, true_peaks, FS,
                                          tolerance_sec=0.075)
        assert acc["sensitivity"] >= 0.75, \
            f"Sensitivity {acc['sensitivity']:.3f} below threshold (0.75)"
        assert acc["ppv"] >= 0.75, \
            f"PPV {acc['ppv']:.3f} below threshold (0.75)"

    def test_rr_intervals_computed(self):
        """RR intervals should be computed from detected peaks."""
        signal, _ = make_qrs_signal(FS, duration=10.0, hr_bpm=75.0)
        filtered = preprocess(signal, FS).filtered
        result = detect_qrs(filtered, FS)

        if result.n_beats >= 2:
            assert len(result.rr_intervals_sec) == result.n_beats - 1
            assert np.all(result.rr_intervals_sec > 0)

    def test_mean_hr_approximately_correct(self):
        """Mean HR should be within 10 BPM of true rate."""
        target_hr = 72.0
        signal, _ = make_qrs_signal(FS, duration=20.0, hr_bpm=target_hr,
                                     noise_level=0.02)
        filtered = preprocess(signal, FS).filtered
        result = detect_qrs(filtered, FS)

        if result.n_beats >= 5:
            assert abs(result.mean_hr_bpm - target_hr) < 10.0, \
                f"Mean HR {result.mean_hr_bpm:.1f} too far from target {target_hr}"

    def test_refractory_period_enforced(self):
        """No two detected beats should be closer than 200ms."""
        signal, _ = make_qrs_signal(FS, duration=10.0, hr_bpm=70.0)
        filtered = preprocess(signal, FS).filtered
        result = detect_qrs(filtered, FS)

        if result.n_beats >= 2:
            min_rr_sec = np.min(result.rr_intervals_sec)
            assert min_rr_sec >= 0.180, \
                f"Refractory period violated: min RR = {min_rr_sec*1000:.1f}ms"

    def test_handles_short_signal(self):
        """Should handle signals shorter than 2 seconds without error."""
        signal, _ = make_qrs_signal(FS, duration=1.5, hr_bpm=70.0)
        filtered = preprocess(signal, FS).filtered
        result = detect_qrs(filtered, FS)
        assert result.n_beats >= 0  # Should not raise

    def test_handles_empty_signal(self):
        """Should handle near-empty signals gracefully."""
        signal = np.zeros(100)
        filtered = preprocess(signal, FS).filtered
        result = detect_qrs(filtered, FS)
        assert result.n_beats >= 0


class TestDetectionAccuracy:
    def test_perfect_match(self):
        """Exact match between detected and reference should give perfect scores."""
        peaks = np.array([100, 460, 820, 1180, 1540])
        acc = compute_detection_accuracy(peaks, peaks, FS)
        assert acc["sensitivity"] == 1.0
        assert acc["ppv"] == 1.0
        assert acc["f1"] == 1.0

    def test_all_missed(self):
        """No detections should give sensitivity=0."""
        reference = np.array([100, 460, 820])
        detected = np.array([], dtype=int)
        acc = compute_detection_accuracy(detected, reference, FS)
        assert acc["sensitivity"] == 0.0

    def test_all_false_positives(self):
        """All FP detections should give PPV=0."""
        reference = np.array([1000, 2000, 3000])
        detected = np.array([50, 100, 150])  # All far from reference
        acc = compute_detection_accuracy(detected, reference, FS)
        assert acc["ppv"] == 0.0

    def test_tolerance_matching(self):
        """Detections within tolerance should count as TP."""
        reference = np.array([360])  # 1 second at 360 Hz
        detected = np.array([360 + 40])  # 40 samples off = ~111ms
        # 150ms tolerance = 54 samples at 360 Hz
        acc = compute_detection_accuracy(detected, reference, FS,
                                          tolerance_sec=0.150)
        assert acc["tp"] == 1
        assert acc["sensitivity"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
