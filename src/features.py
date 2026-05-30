"""
features.py — HRV and morphology feature extraction.

Computes the full suite of standard heart rate variability (HRV) metrics
used in clinical practice and research. Time-domain metrics are most
relevant for arrhythmia classification; frequency-domain metrics provide
additional insight into autonomic nervous system tone.

Standards reference:
  Task Force of the European Society of Cardiology (1996).
  Heart rate variability: standards of measurement, physiological
  interpretation and clinical use. Circulation, 93(5), 1043–1065.
"""

import numpy as np
from scipy import signal as sp_signal
from dataclasses import dataclass
from typing import Optional


@dataclass
class HRVFeatures:
    """Container for all extracted HRV and morphological features."""

    # --- Basic Heart Rate ---
    mean_hr_bpm: float
    min_hr_bpm: float
    max_hr_bpm: float
    hr_range_bpm: float

    # --- RR Interval Statistics ---
    mean_rr_ms: float
    std_rr_ms: float      # SDNN — overall HRV
    rmssd_ms: float       # Root mean square of successive differences
    pnn50_pct: float      # % successive differences > 50ms
    cv_rr: float          # Coefficient of variation (std/mean) — key for AF

    # --- Geometric / Poincaré ---
    sd1_ms: float         # Short-term variability (Poincaré SD1)
    sd2_ms: float         # Long-term variability (Poincaré SD2)
    sd_ratio: float       # SD1/SD2

    # --- Frequency Domain ---
    vlf_power: float      # Very low frequency power (0.003–0.04 Hz)
    lf_power: float       # Low frequency power (0.04–0.15 Hz)
    hf_power: float       # High frequency power (0.15–0.4 Hz)
    lf_hf_ratio: float    # LF/HF ratio (sympathovagal balance)
    total_power: float

    # --- QRS Morphology Estimates ---
    mean_qrs_amplitude_mv: float
    std_qrs_amplitude_mv: float
    qrs_duration_ms: Optional[float]   # Estimated from filtered signal

    # --- Additional Metrics ---
    n_beats: int
    analysis_duration_sec: float
    irregular_beats: int  # Beats with RR deviation > 20% from mean


def extract_features(
    r_peaks: np.ndarray,
    rr_intervals_sec: np.ndarray,
    r_peak_amplitudes: np.ndarray,
    filtered_signal: np.ndarray,
    fs: float,
    duration_sec: float,
) -> HRVFeatures:
    """
    Extract all HRV and morphological features from detected beats.

    Parameters
    ----------
    r_peaks : np.ndarray
        Sample indices of detected R-peaks.
    rr_intervals_sec : np.ndarray
        RR intervals in seconds (len = n_beats - 1).
    r_peak_amplitudes : np.ndarray
        Signal amplitudes at R-peak locations.
    filtered_signal : np.ndarray
        Bandpass-filtered ECG signal.
    fs : float
        Sampling frequency.
    duration_sec : float
        Total record duration in seconds.
    """

    n_beats = len(r_peaks)
    rr_ms = rr_intervals_sec * 1000.0  # Convert to milliseconds

    # --- Heart Rate ---
    if len(rr_ms) > 0:
        mean_rr_ms = float(np.mean(rr_ms))
        std_rr_ms = float(np.std(rr_ms))
        mean_hr = 60000.0 / mean_rr_ms if mean_rr_ms > 0 else 0.0
        hr_series = 60000.0 / rr_ms  # Instantaneous HR for each beat
        min_hr = float(np.min(hr_series))
        max_hr = float(np.max(hr_series))
    else:
        mean_rr_ms = 0.0
        std_rr_ms = 0.0
        mean_hr = 0.0
        min_hr = 0.0
        max_hr = 0.0
        hr_series = np.array([])

    # --- Time Domain HRV ---
    if len(rr_ms) >= 2:
        successive_diffs = np.abs(np.diff(rr_ms))
        rmssd = float(np.sqrt(np.mean(successive_diffs ** 2)))
        pnn50 = float(100.0 * np.mean(successive_diffs > 50.0))
        cv_rr = std_rr_ms / mean_rr_ms if mean_rr_ms > 0 else 0.0
    else:
        rmssd = 0.0
        pnn50 = 0.0
        cv_rr = 0.0

    # --- Poincaré Plot Metrics ---
    if len(rr_ms) >= 3:
        rr_n = rr_ms[:-1]
        rr_n1 = rr_ms[1:]
        sd1 = float(np.std((rr_n1 - rr_n) / np.sqrt(2)))
        sd2 = float(np.std((rr_n1 + rr_n) / np.sqrt(2)))
        sd_ratio = sd1 / sd2 if sd2 > 0 else 0.0
    else:
        sd1 = 0.0
        sd2 = 0.0
        sd_ratio = 0.0

    # --- Frequency Domain HRV ---
    vlf_power, lf_power, hf_power, lf_hf_ratio, total_power = (
        _compute_frequency_domain_hrv(rr_intervals_sec, fs)
    )

    # --- QRS Amplitude Metrics ---
    if len(r_peak_amplitudes) > 0:
        mean_amp = float(np.mean(np.abs(r_peak_amplitudes)))
        std_amp = float(np.std(r_peak_amplitudes))
    else:
        mean_amp = 0.0
        std_amp = 0.0

    # --- QRS Duration Estimate ---
    qrs_dur = _estimate_qrs_duration(filtered_signal, r_peaks, fs)

    # --- Irregular Beats ---
    irregular = 0
    if len(rr_ms) >= 3:
        mean_rr = np.mean(rr_ms)
        irregular = int(np.sum(np.abs(rr_ms - mean_rr) > 0.2 * mean_rr))

    return HRVFeatures(
        mean_hr_bpm=round(mean_hr, 2),
        min_hr_bpm=round(min_hr, 2),
        max_hr_bpm=round(max_hr, 2),
        hr_range_bpm=round(max_hr - min_hr, 2),
        mean_rr_ms=round(mean_rr_ms, 2),
        std_rr_ms=round(std_rr_ms, 2),
        rmssd_ms=round(rmssd, 2),
        pnn50_pct=round(pnn50, 2),
        cv_rr=round(cv_rr, 4),
        sd1_ms=round(sd1, 2),
        sd2_ms=round(sd2, 2),
        sd_ratio=round(sd_ratio, 4),
        vlf_power=round(vlf_power, 6),
        lf_power=round(lf_power, 6),
        hf_power=round(hf_power, 6),
        lf_hf_ratio=round(lf_hf_ratio, 4),
        total_power=round(total_power, 6),
        mean_qrs_amplitude_mv=round(mean_amp, 4),
        std_qrs_amplitude_mv=round(std_amp, 4),
        qrs_duration_ms=qrs_dur,
        n_beats=n_beats,
        analysis_duration_sec=round(duration_sec, 2),
        irregular_beats=irregular,
    )


def _compute_frequency_domain_hrv(
    rr_sec: np.ndarray, fs: float
) -> tuple:
    """
    Compute frequency-domain HRV metrics via Lomb-Scargle periodogram.

    Lomb-Scargle is preferred over FFT for HRV because RR intervals
    are unevenly spaced in time. It handles irregular sampling natively.

    Returns (vlf_power, lf_power, hf_power, lf_hf_ratio, total_power)
    """
    if len(rr_sec) < 10:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    try:
        # Reconstruct timestamps from RR intervals
        times = np.cumsum(rr_sec)
        times = times - times[0]  # Start at 0

        # Frequency axis for HRV (0.003 to 0.4 Hz)
        freqs = np.linspace(0.003, 0.4, 500)
        angular_freqs = 2 * np.pi * freqs

        # Lomb-Scargle periodogram of RR series
        rr_ms = rr_sec * 1000.0
        rr_centered = rr_ms - np.mean(rr_ms)

        pgram = sp_signal.lombscargle(times, rr_centered, angular_freqs,
                                       normalize=False)

        # Band powers (integrate PSD over frequency bands)
        def band_power(f_low, f_high):
            mask = (freqs >= f_low) & (freqs < f_high)
            if mask.sum() < 2:
                return 0.0
            return float(np.trapezoid(pgram[mask], freqs[mask]))

        vlf = band_power(0.003, 0.04)
        lf = band_power(0.04, 0.15)
        hf = band_power(0.15, 0.4)
        total = band_power(0.003, 0.4)
        lf_hf = lf / hf if hf > 0 else 0.0

        return max(0, vlf), max(0, lf), max(0, hf), max(0, lf_hf), max(0, total)

    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0


def _estimate_qrs_duration(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    fs: float,
    search_window_ms: float = 80.0,
) -> Optional[float]:
    """
    Estimate mean QRS duration by finding onset and offset around each R-peak.

    Method: threshold-based approach — QRS onset is where the signal
    first exceeds 10% of R-peak amplitude coming from either side.
    """
    if len(r_peaks) == 0:
        return None

    half_win = int((search_window_ms / 1000.0) * fs)
    durations = []

    for rp in r_peaks:
        start = max(0, rp - half_win)
        end = min(len(signal), rp + half_win)
        window = signal[start:end]

        if len(window) < 4:
            continue

        peak_amp = np.abs(signal[rp])
        if peak_amp < 1e-6:
            continue

        threshold = 0.1 * peak_amp
        center_local = rp - start

        # Find onset (walk left from peak)
        onset_local = center_local
        for i in range(center_local, 0, -1):
            if np.abs(window[i]) < threshold:
                onset_local = i
                break

        # Find offset (walk right from peak)
        offset_local = center_local
        for i in range(center_local, len(window) - 1):
            if np.abs(window[i]) < threshold:
                offset_local = i
                break

        dur_samples = offset_local - onset_local
        dur_ms = (dur_samples / fs) * 1000.0
        if 40.0 < dur_ms < 200.0:  # Physiologically plausible range
            durations.append(dur_ms)

    if len(durations) == 0:
        return None

    return round(float(np.median(durations)), 1)


def get_instantaneous_hr(r_peaks: np.ndarray, fs: float,
                          signal_len: int) -> tuple:
    """
    Compute instantaneous heart rate at each detected beat.

    Returns (times_sec, hr_bpm) arrays aligned to beat timestamps.
    """
    if len(r_peaks) < 2:
        return np.array([]), np.array([])

    beat_times = r_peaks / fs
    rr_sec = np.diff(r_peaks) / fs

    # Instantaneous HR defined at midpoint between consecutive beats
    midpoint_times = (beat_times[:-1] + beat_times[1:]) / 2.0
    instant_hr = 60.0 / rr_sec

    return midpoint_times, instant_hr
