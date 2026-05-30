"""
preprocessor.py — ECG signal filtering pipeline.

Implements a clinically motivated preprocessing chain:
  1. DC offset removal
  2. Baseline wander removal (high-pass, 0.5 Hz cutoff)
  3. High-frequency noise removal (low-pass, 40 Hz cutoff)
  4. Powerline interference notch filter (60 Hz)

All filters use zero-phase forward-backward filtering (scipy.signal.filtfilt)
to prevent phase distortion — critical for preserving QRS morphology.
"""

import numpy as np
from scipy import signal as sp_signal
from dataclasses import dataclass


@dataclass
class FilteredSignal:
    """Output of the preprocessing pipeline."""
    raw: np.ndarray
    baseline_removed: np.ndarray
    filtered: np.ndarray       # Final output for downstream processing
    fs: float
    filter_params: dict


def _butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    """Design a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Clamp to avoid numerical issues near Nyquist
    low = max(low, 1e-4)
    high = min(high, 0.9999)
    b, a = sp_signal.butter(order, [low, high], btype="band")
    return b, a


def _butter_highpass(cutoff: float, fs: float, order: int = 4):
    """Design a Butterworth high-pass filter."""
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    norm_cutoff = max(norm_cutoff, 1e-4)
    b, a = sp_signal.butter(order, norm_cutoff, btype="high")
    return b, a


def _butter_lowpass(cutoff: float, fs: float, order: int = 4):
    """Design a Butterworth low-pass filter."""
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    norm_cutoff = min(norm_cutoff, 0.9999)
    b, a = sp_signal.butter(order, norm_cutoff, btype="low")
    return b, a


def _notch_filter(freq: float, fs: float, Q: float = 30.0):
    """
    Design an IIR notch filter at `freq` Hz.
    Q factor controls bandwidth: higher Q = narrower notch.
    """
    nyq = 0.5 * fs
    if freq >= nyq:
        return None, None
    w0 = freq / nyq
    b, a = sp_signal.iirnotch(w0, Q)
    return b, a


def _safe_filtfilt(b, a, x: np.ndarray) -> np.ndarray:
    """
    Apply zero-phase filter. Requires signal length > 3 * max(len(a), len(b)).
    Falls back to lfilter if signal is too short.
    """
    min_len = 3 * max(len(a), len(b))
    if len(x) > min_len:
        return sp_signal.filtfilt(b, a, x)
    else:
        return sp_signal.lfilter(b, a, x)


def remove_baseline_wander(signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Remove baseline wander using a high-pass filter at 0.5 Hz.

    Baseline wander is caused by respiration and body movement and
    manifests as slow, low-frequency drift in the ECG baseline. Removing
    it is essential for accurate amplitude measurements.
    """
    b, a = _butter_highpass(0.5, fs, order=4)
    return _safe_filtfilt(b, a, signal)


def apply_bandpass(signal: np.ndarray, fs: float,
                   lowcut: float = 0.5, highcut: float = 40.0) -> np.ndarray:
    """
    Apply bandpass filter to remove out-of-band noise.

    The 0.5–40 Hz passband captures clinically relevant ECG frequencies:
    - P, QRS, T waves are primarily 0.5–40 Hz
    - High-frequency noise (EMG, electrical interference) is above 40 Hz
    """
    b, a = _butter_bandpass(lowcut, highcut, fs, order=4)
    return _safe_filtfilt(b, a, signal)


def apply_notch(signal: np.ndarray, fs: float, freq: float = 60.0) -> np.ndarray:
    """
    Apply notch filter to remove powerline interference (60 Hz in US).
    European recordings would use 50 Hz.
    """
    b, a = _notch_filter(freq, fs, Q=30.0)
    if b is None:
        return signal  # Frequency above Nyquist — skip
    return _safe_filtfilt(b, a, signal)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Normalize signal to zero mean. Unit variance normalization is NOT applied
    because amplitude information is clinically meaningful.
    """
    return signal - np.mean(signal)


def preprocess(signal: np.ndarray, fs: float,
               apply_notch_filter: bool = True,
               notch_freq: float = 60.0) -> FilteredSignal:
    """
    Full preprocessing pipeline.

    Steps:
      1. DC offset removal (mean subtraction)
      2. High-pass filter at 0.5 Hz (baseline wander removal)
      3. Low-pass filter at 40 Hz (noise reduction)
      4. Optional: Notch filter at 60 Hz (powerline interference)

    Parameters
    ----------
    signal : np.ndarray
        Raw ECG signal in mV.
    fs : float
        Sampling frequency in Hz.
    apply_notch_filter : bool
        Whether to apply 60 Hz notch filter.
    notch_freq : float
        Powerline frequency (60 Hz in US, 50 Hz in Europe).

    Returns
    -------
    FilteredSignal with raw, intermediate, and final filtered signals.
    """
    # Step 1: Remove DC offset
    centered = normalize_signal(signal)

    # Step 2: Remove baseline wander
    no_baseline = remove_baseline_wander(centered, fs)

    # Step 3: Apply bandpass
    bandpassed = apply_bandpass(no_baseline, fs, lowcut=0.5, highcut=40.0)

    # Step 4: Powerline notch
    if apply_notch_filter:
        final = apply_notch(bandpassed, fs, freq=notch_freq)
    else:
        final = bandpassed

    return FilteredSignal(
        raw=signal,
        baseline_removed=no_baseline,
        filtered=final,
        fs=fs,
        filter_params={
            "bandpass_low_hz": 0.5,
            "bandpass_high_hz": 40.0,
            "notch_hz": notch_freq if apply_notch_filter else None,
            "filter_type": "Butterworth order-4 zero-phase",
        },
    )


def estimate_signal_quality(signal: np.ndarray, fs: float) -> dict:
    """
    Estimate signal quality using several heuristics.

    Returns a dict with quality metrics and an overall score (0–100).
    """
    # Flatline detection: std too low means no signal
    std = np.std(signal)
    flatline = std < 0.01

    # Clipping detection: many samples at min/max
    p1, p99 = np.percentile(signal, [1, 99])
    clipped_frac = np.mean((signal <= p1) | (signal >= p99))
    clipping = clipped_frac > 0.05

    # High-frequency noise ratio: energy above 100 Hz vs total
    # (only meaningful if fs > 200 Hz)
    noise_ratio = 0.0
    if fs > 200:
        freqs, psd = sp_signal.welch(signal, fs, nperseg=min(1024, len(signal)))
        total_power = np.trapezoid(psd, freqs)
        if total_power > 0:
            noise_power = np.trapezoid(psd[freqs > 100], freqs[freqs > 100])
            noise_ratio = noise_power / total_power

    # SNR estimate: ratio of QRS-band power to out-of-band power
    b_signal, a_signal = _butter_bandpass(5.0, 40.0, fs)
    b_noise, a_noise = _butter_highpass(100.0, fs) if fs > 200 else (None, None)

    qrs_power = np.var(_safe_filtfilt(b_signal, a_signal, signal))

    # Score
    score = 100
    if flatline:
        score = 0
    elif clipping:
        score -= 30
    if noise_ratio > 0.3:
        score -= 20
    score = max(0, min(100, score))

    if score >= 80:
        quality_label = "Excellent"
    elif score >= 60:
        quality_label = "Good"
    elif score >= 40:
        quality_label = "Fair"
    else:
        quality_label = "Poor"

    return {
        "score": score,
        "quality_label": quality_label,
        "std_mv": float(std),
        "flatline": flatline,
        "clipping": clipping,
        "noise_ratio": float(noise_ratio),
    }
