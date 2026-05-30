"""
tests/test_preprocessor.py

Tests for the ECG signal preprocessing pipeline.

Strategy: use synthetic signals with known properties to verify that
each filter stage behaves correctly and preserves QRS morphology.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from scipy import signal as sp_signal

from src.preprocessor import (
    preprocess,
    remove_baseline_wander,
    apply_bandpass,
    apply_notch,
    normalize_signal,
    estimate_signal_quality,
)


FS = 360.0  # MIT-BIH native sampling rate


def make_synthetic_ecg(fs: float = FS, duration: float = 5.0,
                        hr_bpm: float = 70.0) -> np.ndarray:
    """
    Generate a synthetic ECG-like signal with known beat positions.
    QRS complexes are modeled as narrow Gaussian pulses at 1mV amplitude.
    """
    n = int(fs * duration)
    t = np.arange(n) / fs
    signal = np.zeros(n)

    beat_period = 60.0 / hr_bpm
    beat_times = np.arange(0, duration, beat_period)

    # QRS: narrow Gaussian at 1 mV
    for bt in beat_times:
        qrs_center = bt + 0.05  # slight offset from P-wave
        signal += 1.0 * np.exp(-((t - qrs_center) ** 2) / (2 * (0.015) ** 2))

    # T-wave: broader Gaussian at 0.3 mV
    for bt in beat_times:
        t_center = bt + 0.25
        signal += 0.3 * np.exp(-((t - t_center) ** 2) / (2 * (0.04) ** 2))

    # P-wave: small Gaussian at 0.15 mV
    for bt in beat_times:
        p_center = bt - 0.1
        signal += 0.15 * np.exp(-((t - p_center) ** 2) / (2 * (0.025) ** 2))

    return signal


def make_baseline_wander(fs: float = FS, duration: float = 5.0,
                          freq: float = 0.3, amplitude: float = 0.5) -> np.ndarray:
    """Generate sinusoidal baseline wander at respiratory frequency (~0.3 Hz)."""
    n = int(fs * duration)
    t = np.arange(n) / fs
    return amplitude * np.sin(2 * np.pi * freq * t)


def make_powerline_noise(fs: float = FS, duration: float = 5.0,
                          amplitude: float = 0.1) -> np.ndarray:
    """Generate 60 Hz powerline interference."""
    n = int(fs * duration)
    t = np.arange(n) / fs
    return amplitude * np.sin(2 * np.pi * 60.0 * t)


class TestNormalization:
    def test_zero_mean(self):
        """DC offset removal should yield zero-mean signal."""
        signal = np.ones(1000) * 5.0 + np.random.randn(1000) * 0.1
        normalized = normalize_signal(signal)
        assert abs(np.mean(normalized)) < 1e-10

    def test_preserves_variance(self):
        """Normalization should not change signal variance."""
        signal = np.random.randn(1000) + 3.0
        original_std = np.std(signal)
        normalized = normalize_signal(signal)
        assert abs(np.std(normalized) - original_std) < 1e-10


class TestBaselineWanderRemoval:
    def test_removes_low_frequency(self):
        """High-pass filter should attenuate baseline wander."""
        ecg = make_synthetic_ecg()
        wander = make_baseline_wander(amplitude=0.5)
        noisy = ecg + wander

        filtered = remove_baseline_wander(noisy, FS)

        # After filtering, the slow wander component should be greatly reduced
        # Check power below 0.5 Hz is attenuated
        freqs, psd_noisy = sp_signal.welch(noisy, FS, nperseg=512)
        freqs, psd_filtered = sp_signal.welch(filtered, FS, nperseg=512)

        low_mask = freqs < 0.4
        if low_mask.sum() > 0:
            power_noisy_low = np.mean(psd_noisy[low_mask])
            power_filtered_low = np.mean(psd_filtered[low_mask])
            assert power_filtered_low < power_noisy_low * 0.5, \
                "Baseline wander not sufficiently attenuated"

    def test_preserves_qrs_band(self):
        """QRS energy (10–40 Hz) should be largely preserved."""
        ecg = make_synthetic_ecg()
        wander = make_baseline_wander(amplitude=0.3)
        noisy = ecg + wander

        filtered = remove_baseline_wander(noisy, FS)

        freqs, psd_ecg = sp_signal.welch(ecg, FS, nperseg=512)
        freqs, psd_filtered = sp_signal.welch(filtered, FS, nperseg=512)

        qrs_mask = (freqs >= 10) & (freqs <= 40)
        if qrs_mask.sum() > 0:
            power_ecg_qrs = np.mean(psd_ecg[qrs_mask])
            power_filt_qrs = np.mean(psd_filtered[qrs_mask])
            # Should preserve at least 60% of QRS band power
            assert power_filt_qrs > 0.6 * power_ecg_qrs, \
                "QRS band energy excessively attenuated by high-pass"


class TestBandpassFilter:
    def test_attenuates_high_frequency(self):
        """Bandpass should remove high-frequency noise above 40 Hz."""
        ecg = make_synthetic_ecg()
        # Add 100 Hz noise
        t = np.arange(len(ecg)) / FS
        hf_noise = 0.5 * np.sin(2 * np.pi * 100 * t)
        noisy = ecg + hf_noise

        filtered = apply_bandpass(noisy, FS, lowcut=0.5, highcut=40.0)

        freqs, psd_noisy = sp_signal.welch(noisy, FS, nperseg=512)
        freqs, psd_filtered = sp_signal.welch(filtered, FS, nperseg=512)

        hf_mask = freqs > 80
        if hf_mask.sum() > 0:
            assert np.mean(psd_filtered[hf_mask]) < np.mean(psd_noisy[hf_mask]) * 0.1

    def test_output_length_preserved(self):
        """Filter should not change signal length."""
        signal = np.random.randn(1800)  # 5 seconds at 360 Hz
        filtered = apply_bandpass(signal, FS)
        assert len(filtered) == len(signal)


class TestNotchFilter:
    def test_attenuates_60hz(self):
        """Notch filter should attenuate 60 Hz powerline interference."""
        ecg = make_synthetic_ecg()
        noise = make_powerline_noise(amplitude=0.5)
        noisy = ecg + noise

        filtered = apply_notch(noisy, FS, freq=60.0)

        freqs, psd_noisy = sp_signal.welch(noisy, FS, nperseg=1024)
        freqs, psd_filtered = sp_signal.welch(filtered, FS, nperseg=1024)

        # Find bins near 60 Hz
        idx_60 = np.argmin(np.abs(freqs - 60.0))
        if idx_60 > 0 and idx_60 < len(freqs):
            assert psd_filtered[idx_60] < psd_noisy[idx_60] * 0.3, \
                "60 Hz interference not sufficiently attenuated"

    def test_preserves_qrs_amplitude(self):
        """Notch should not significantly distort QRS amplitude."""
        ecg = make_synthetic_ecg()
        noise = make_powerline_noise(amplitude=0.1)
        noisy = ecg + noise

        filtered = apply_notch(noisy, FS, freq=60.0)

        # Peak amplitude of QRS should be largely preserved
        peak_noisy = np.max(np.abs(noisy))
        peak_filtered = np.max(np.abs(filtered))
        assert abs(peak_filtered - peak_noisy) < 0.2 * peak_noisy


class TestFullPipeline:
    def test_pipeline_runs(self):
        """Full preprocessing pipeline should run without error."""
        ecg = make_synthetic_ecg()
        result = preprocess(ecg, FS)
        assert result.filtered is not None
        assert len(result.filtered) == len(ecg)

    def test_pipeline_output_finite(self):
        """Pipeline should not produce NaN or Inf values."""
        ecg = make_synthetic_ecg()
        result = preprocess(ecg, FS)
        assert np.all(np.isfinite(result.filtered)), "Pipeline output contains NaN/Inf"

    def test_pipeline_reduces_noise(self):
        """Pipeline output should have lower total noise than input."""
        ecg = make_synthetic_ecg()
        wander = make_baseline_wander(amplitude=0.3)
        noise_60 = make_powerline_noise(amplitude=0.2)
        noisy = ecg + wander + noise_60

        result = preprocess(noisy, FS)

        # Out-of-band noise should be reduced
        freqs, psd_noisy = sp_signal.welch(noisy, FS, nperseg=512)
        freqs, psd_clean = sp_signal.welch(result.filtered, FS, nperseg=512)

        # Power above 50 Hz should be much lower after filtering
        hf_mask = freqs > 50
        if hf_mask.sum() > 0:
            assert np.mean(psd_clean[hf_mask]) < np.mean(psd_noisy[hf_mask])


class TestSignalQuality:
    def test_good_ecg_gets_high_score(self):
        """Clean synthetic ECG should receive a good quality score."""
        ecg = make_synthetic_ecg()
        quality = estimate_signal_quality(ecg, FS)
        assert quality["score"] >= 60, f"Clean signal got low score: {quality['score']}"

    def test_flatline_detected(self):
        """Flatline signal should be flagged."""
        flat = np.zeros(1800)
        quality = estimate_signal_quality(flat, FS)
        assert quality["flatline"] == True

    def test_quality_has_required_keys(self):
        """Quality dict should contain required keys."""
        ecg = make_synthetic_ecg()
        quality = estimate_signal_quality(ecg, FS)
        required_keys = {"score", "quality_label", "flatline", "clipping"}
        assert required_keys.issubset(set(quality.keys()))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
