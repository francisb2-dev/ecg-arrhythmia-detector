"""
detector.py — Pan-Tompkins inspired QRS complex detection.

Implements the full Pan-Tompkins (1985) algorithm from scratch:
  1. Differentiation — amplifies high-frequency QRS slopes
  2. Squaring — emphasizes large values, ensures positivity
  3. Moving window integration — smears energy over QRS width
  4. Adaptive dual-threshold detection — tracks signal level dynamically
  5. Refractory period enforcement — 200ms physiological minimum RR
  6. Search-back — recovers missed beats when threshold is exceeded

Reference:
  Pan, J. & Tompkins, W.J. (1985). A real-time QRS detection algorithm.
  IEEE Transactions on Biomedical Engineering, 32(3), 230-236.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DetectionResult:
    """Container for QRS detection output."""
    r_peaks: np.ndarray          # Sample indices of detected R-peaks
    r_peak_amplitudes: np.ndarray  # Signal amplitude at each R-peak (mV)
    rr_intervals: np.ndarray     # RR intervals in samples
    rr_intervals_sec: np.ndarray # RR intervals in seconds
    fs: float
    integrated_signal: np.ndarray  # MWI output (for visualization)
    squared_signal: np.ndarray
    n_beats: int
    mean_rr_sec: float
    mean_hr_bpm: float
    detection_params: dict = field(default_factory=dict)


def _differentiate(signal: np.ndarray) -> np.ndarray:
    """
    5-point derivative approximation from Pan-Tompkins.
    Emphasizes high-slope QRS transitions while suppressing
    lower-slope P and T waves.

    h(nT) = (1/8T)[-x(n-2T) - 2x(n-T) + 2x(n+T) + x(n+2T)]
    """
    n = len(signal)
    deriv = np.zeros(n)
    for i in range(2, n - 2):
        deriv[i] = (
            -signal[i - 2] - 2 * signal[i - 1] + 2 * signal[i + 1] + signal[i + 2]
        ) / 8.0
    return deriv


def _square(signal: np.ndarray) -> np.ndarray:
    """
    Element-wise squaring. Makes all values positive and amplifies
    large excursions (QRS) relative to smaller ones (P, T waves).
    """
    return signal ** 2


def _moving_window_integration(signal: np.ndarray, fs: float,
                                window_sec: float = 0.150) -> np.ndarray:
    """
    Moving window integrator with window size ~150ms.

    Smears the squared derivative energy over the typical QRS duration
    (60–120ms), creating a smooth envelope. Window is typically 30 samples
    at 200 Hz or 54 samples at 360 Hz (MIT-BIH native rate).

    Uses cumulative sum trick for O(n) computation.
    """
    window_size = int(window_sec * fs)
    if window_size < 1:
        window_size = 1

    # Pad and compute moving average via convolution
    kernel = np.ones(window_size) / window_size
    integrated = np.convolve(signal, kernel, mode="same")
    return integrated


def _find_peaks_adaptive(
    integrated: np.ndarray,
    squared: np.ndarray,
    original: np.ndarray,
    fs: float,
    refractory_sec: float = 0.200,
) -> np.ndarray:
    """
    Adaptive dual-threshold R-peak detection with search-back.

    The algorithm maintains running estimates of:
      - SPKI: signal peak level (in integrated signal)
      - NPKI: noise peak level (in integrated signal)
      - Threshold1: primary detection threshold
      - Threshold2: search-back threshold (lower)

    A candidate peak is classified as a QRS if it exceeds Threshold1.
    If no beat is detected for 1.66x the mean RR interval, search-back
    is performed using Threshold2.
    """
    refractory_samples = int(refractory_sec * fs)
    n = len(integrated)

    # --- Initialize thresholds from first 2 seconds ---
    init_samples = min(int(2.0 * fs), n)
    init_signal = integrated[:init_samples]

    # Find local maxima in the initialization window
    init_peaks = _local_maxima(init_signal, min_distance=refractory_samples)
    if len(init_peaks) > 0:
        SPKI = np.mean(integrated[init_peaks]) * 0.25
        NPKI = np.mean(integrated[init_peaks]) * 0.25 * 0.5
    else:
        SPKI = np.max(integrated) * 0.25
        NPKI = SPKI * 0.5

    threshold1 = NPKI + 0.25 * (SPKI - NPKI)
    threshold2 = 0.5 * threshold1

    r_peaks = []
    last_peak_sample = -refractory_samples
    rr_history = []  # Recent RR intervals for search-back window

    # Find all local maxima candidates
    candidates = _local_maxima(integrated, min_distance=refractory_samples // 2)

    for i, candidate in enumerate(candidates):
        peak_val = integrated[candidate]

        # Enforce refractory period
        if candidate - last_peak_sample < refractory_samples:
            continue

        # --- Search-back check ---
        if len(rr_history) >= 8:
            mean_rr = np.mean(rr_history[-8:])
            time_since_last = candidate - last_peak_sample
            if time_since_last > 1.66 * mean_rr:
                # Search back for a missed beat
                search_start = last_peak_sample + refractory_samples
                search_end = candidate
                if search_end > search_start:
                    search_region = integrated[search_start:search_end]
                    if len(search_region) > 0:
                        local_max_idx = np.argmax(search_region)
                        local_max_val = search_region[local_max_idx]
                        if local_max_val > threshold2:
                            sb_peak = search_start + local_max_idx
                            # Find closest R-peak in original signal
                            r_peak = _refine_r_peak(original, sb_peak, fs)
                            r_peaks.append(r_peak)
                            rr_history.append(r_peak - last_peak_sample)
                            last_peak_sample = r_peak
                            # Update thresholds with found peak
                            SPKI = 0.25 * local_max_val + 0.75 * SPKI

        if peak_val >= threshold1:
            # Classified as QRS
            r_peak = _refine_r_peak(original, candidate, fs)
            r_peaks.append(r_peak)
            SPKI = 0.125 * peak_val + 0.875 * SPKI
            if r_peaks:
                rr_history.append(r_peak - last_peak_sample)
            last_peak_sample = r_peak
        else:
            # Classified as noise
            NPKI = 0.125 * peak_val + 0.875 * NPKI

        # Update thresholds
        threshold1 = NPKI + 0.25 * (SPKI - NPKI)
        threshold2 = 0.5 * threshold1

    return np.array(sorted(set(r_peaks)), dtype=int)


def _local_maxima(signal: np.ndarray, min_distance: int = 10) -> np.ndarray:
    """
    Find local maxima with minimum separation of `min_distance` samples.
    Uses a sliding window approach rather than scipy to keep this
    implementation self-contained.
    """
    n = len(signal)
    if n == 0:
        return np.array([], dtype=int)

    peaks = []
    i = 1
    while i < n - 1:
        # Check if local maximum
        if signal[i] > signal[i - 1] and signal[i] >= signal[i + 1]:
            # Find the true peak in a neighborhood
            start = max(0, i - 2)
            end = min(n, i + 3)
            local_peak = start + np.argmax(signal[start:end])
            if not peaks or (local_peak - peaks[-1]) >= min_distance:
                peaks.append(local_peak)
            elif signal[local_peak] > signal[peaks[-1]]:
                peaks[-1] = local_peak
        i += 1

    return np.array(peaks, dtype=int)


def _refine_r_peak(signal: np.ndarray, candidate: int, fs: float,
                   search_window_ms: float = 50.0) -> int:
    """
    Refine a detected QRS candidate to the actual R-peak in the
    original filtered signal by searching in a ±50ms window.
    """
    half_window = int((search_window_ms / 1000.0) * fs)
    start = max(0, candidate - half_window)
    end = min(len(signal), candidate + half_window + 1)

    if end <= start:
        return candidate

    window = signal[start:end]
    # R-peak is the maximum absolute amplitude in the window
    local_idx = np.argmax(np.abs(window))
    return start + local_idx


def detect_qrs(
    filtered_signal: np.ndarray,
    fs: float,
    refractory_sec: float = 0.200,
) -> DetectionResult:
    """
    Full Pan-Tompkins QRS detection pipeline.

    Parameters
    ----------
    filtered_signal : np.ndarray
        Preprocessed ECG signal (bandpass filtered).
    fs : float
        Sampling frequency in Hz.
    refractory_sec : float
        Minimum physiological RR interval (200ms default).

    Returns
    -------
    DetectionResult with detected R-peak locations and derived metrics.
    """
    # Step 1: 5-point derivative
    deriv = _differentiate(filtered_signal)

    # Step 2: Squaring
    squared = _square(deriv)

    # Step 3: Moving window integration (~150ms window)
    integrated = _moving_window_integration(squared, fs, window_sec=0.150)

    # Step 4: Adaptive threshold detection + search-back
    r_peaks = _find_peaks_adaptive(
        integrated, squared, filtered_signal, fs, refractory_sec
    )

    # Filter out peaks too close to the edges
    margin = int(0.1 * fs)
    r_peaks = r_peaks[(r_peaks > margin) & (r_peaks < len(filtered_signal) - margin)]

    # Compute RR intervals
    if len(r_peaks) >= 2:
        rr_samples = np.diff(r_peaks)
        rr_sec = rr_samples / fs
        mean_rr = float(np.mean(rr_sec))
        mean_hr = 60.0 / mean_rr if mean_rr > 0 else 0.0
    else:
        rr_samples = np.array([])
        rr_sec = np.array([])
        mean_rr = 0.0
        mean_hr = 0.0

    amplitudes = filtered_signal[r_peaks] if len(r_peaks) > 0 else np.array([])

    return DetectionResult(
        r_peaks=r_peaks,
        r_peak_amplitudes=amplitudes,
        rr_intervals=rr_samples,
        rr_intervals_sec=rr_sec,
        fs=fs,
        integrated_signal=integrated,
        squared_signal=squared,
        n_beats=len(r_peaks),
        mean_rr_sec=mean_rr,
        mean_hr_bpm=mean_hr,
        detection_params={
            "algorithm": "Pan-Tompkins (1985) inspired",
            "refractory_ms": refractory_sec * 1000,
            "mwi_window_ms": 150,
            "derivative": "5-point",
        },
    )


def compute_detection_accuracy(
    detected_peaks: np.ndarray,
    reference_peaks: np.ndarray,
    fs: float,
    tolerance_sec: float = 0.150,
) -> dict:
    """
    Compare detected peaks against ground-truth annotations.

    Uses a ±150ms matching tolerance (standard in the field).

    Returns sensitivity, positive predictivity (precision), and F1.
    """
    if len(reference_peaks) == 0:
        return {"sensitivity": None, "ppv": None, "f1": None}

    tol_samples = int(tolerance_sec * fs)
    tp = 0
    matched_ref = set()

    for det in detected_peaks:
        for j, ref in enumerate(reference_peaks):
            if j not in matched_ref and abs(det - ref) <= tol_samples:
                tp += 1
                matched_ref.add(j)
                break

    fn = len(reference_peaks) - tp
    fp = len(detected_peaks) - tp

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * sensitivity * ppv / (sensitivity + ppv)
          if (sensitivity + ppv) > 0 else 0.0)

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "sensitivity": round(sensitivity, 4),
        "ppv": round(ppv, 4),
        "f1": round(f1, 4),
    }
