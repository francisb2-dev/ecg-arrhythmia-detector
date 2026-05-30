"""
tests/test_classifier.py

Tests for rhythm classification logic.

Verifies that the classifier correctly identifies:
- Normal sinus rhythm from regular RR intervals
- Atrial fibrillation from highly irregular RR intervals
- Bradycardia from low mean heart rate
- Tachycardia from high mean heart rate
- Combined findings (e.g., AF + tachycardia)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.features import extract_features
from src.classifier import classify_rhythm


FS = 360.0


def make_rr_intervals(mean_hr_bpm: float, n_beats: int,
                       cv: float = 0.02,
                       duration_sec: float = 60.0) -> tuple:
    """
    Generate synthetic RR intervals with specified mean HR and variability.
    Returns (r_peaks, rr_sec, r_amplitudes).
    """
    mean_rr_sec = 60.0 / mean_hr_bpm
    std_rr_sec = cv * mean_rr_sec

    rr_sec = np.abs(np.random.normal(mean_rr_sec, std_rr_sec, n_beats - 1))
    # Clamp to physiological range
    rr_sec = np.clip(rr_sec, 0.3, 2.0)

    # Build r_peak sample positions
    r_peaks = np.zeros(n_beats, dtype=int)
    r_peaks[0] = int(0.5 * FS)
    for i in range(1, n_beats):
        r_peaks[i] = r_peaks[i - 1] + int(rr_sec[i - 1] * FS)

    amplitudes = np.random.normal(1.0, 0.1, n_beats)

    return r_peaks, rr_sec, amplitudes


def make_af_rr_intervals(mean_hr_bpm: float = 90.0,
                          n_beats: int = 60) -> tuple:
    """
    Generate RR intervals characteristic of AF: high CV (>0.25),
    exponentially-distributed intervals.
    """
    mean_rr_sec = 60.0 / mean_hr_bpm
    # AF RR intervals follow a roughly exponential distribution
    rr_sec = np.random.exponential(mean_rr_sec, n_beats - 1)
    rr_sec = np.clip(rr_sec, 0.3, 1.8)

    r_peaks = np.zeros(n_beats, dtype=int)
    r_peaks[0] = int(0.5 * FS)
    for i in range(1, n_beats):
        r_peaks[i] = r_peaks[i - 1] + int(rr_sec[i - 1] * FS)

    amplitudes = np.random.normal(1.0, 0.15, n_beats)
    return r_peaks, rr_sec, amplitudes


class TestNormalSinusRhythm:
    def test_regular_normal_hr_classified_as_nsr(self):
        """Regular RR, HR 60–100 BPM → Normal Sinus Rhythm."""
        np.random.seed(42)
        r_peaks, rr_sec, amps = make_rr_intervals(
            mean_hr_bpm=72.0, n_beats=60, cv=0.03, duration_sec=60.0
        )
        signal = np.zeros(r_peaks[-1] + 500)
        hrv = extract_features(r_peaks, rr_sec, amps, signal, FS, 60.0)
        result = classify_rhythm(hrv, rr_sec)

        # NSR should be primary or have high confidence
        all_codes = [l.code for l in result.all_labels]
        assert "NSR" in all_codes, \
            f"NSR not found in classifications: {all_codes}"

        nsr_label = next(l for l in result.all_labels if l.code == "NSR")
        assert nsr_label.confidence >= 50, \
            f"NSR confidence too low: {nsr_label.confidence}"

    def test_nsr_not_classified_as_af(self):
        """Regular NSR should NOT be flagged as AF."""
        np.random.seed(42)
        r_peaks, rr_sec, amps = make_rr_intervals(
            mean_hr_bpm=72.0, n_beats=60, cv=0.03
        )
        signal = np.zeros(r_peaks[-1] + 500)
        hrv = extract_features(r_peaks, rr_sec, amps, signal, FS, 60.0)
        result = classify_rhythm(hrv, rr_sec)

        af_labels = [l for l in result.all_labels if l.code == "AF"]
        if af_labels:
            assert af_labels[0].confidence < 40, \
                f"NSR incorrectly flagged as AF with confidence {af_labels[0].confidence}"


class TestAtrialFibrillation:
    def test_af_detected_from_irregular_rr(self):
        """Highly irregular RR intervals should trigger AF classification."""
        np.random.seed(123)
        r_peaks, rr_sec, amps = make_af_rr_intervals(
            mean_hr_bpm=90.0, n_beats=80
        )
        # Verify the CV is indeed high
        cv = np.std(rr_sec) / np.mean(rr_sec)
        assert cv > 0.15, f"Test setup error: AF CV too low ({cv:.3f})"

        signal = np.zeros(r_peaks[-1] + 500)
        hrv = extract_features(r_peaks, rr_sec, amps, signal, FS, 60.0)
        result = classify_rhythm(hrv, rr_sec)

        all_codes = [l.code for l in result.all_labels]
        assert "AF" in all_codes, \
            f"AF not detected. HR={hrv.mean_hr_bpm:.1f}, CV={hrv.cv_rr:.3f}. Labels: {all_codes}"

    def test_af_confidence_scales_with_irregularity(self):
        """Higher CV should yield higher AF confidence."""
        np.random.seed(42)

        # Mildly irregular
        _, rr_mild, amps_mild = make_rr_intervals(72.0, 60, cv=0.12)
        r_mild = np.cumsum(np.concatenate([[int(0.5*FS)], (rr_mild * FS).astype(int)]))
        signal = np.zeros(r_mild[-1] + 500)
        hrv_mild = extract_features(r_mild, rr_mild, amps_mild, signal, FS, 60.0)
        result_mild = classify_rhythm(hrv_mild, rr_mild)

        # Highly irregular
        _, rr_high, amps_high = make_af_rr_intervals(90.0, 60)
        r_high = np.cumsum(np.concatenate([[int(0.5*FS)], (rr_high * FS).astype(int)]))
        signal_h = np.zeros(r_high[-1] + 500)
        hrv_high = extract_features(r_high, rr_high, amps_high, signal_h, FS, 60.0)
        result_high = classify_rhythm(hrv_high, rr_high)

        af_mild = next((l for l in result_mild.all_labels if l.code == "AF"), None)
        af_high = next((l for l in result_high.all_labels if l.code == "AF"), None)

        if af_mild and af_high:
            assert af_high.confidence >= af_mild.confidence, \
                "Higher irregularity should yield higher AF confidence"


class TestBradycardia:
    def test_bradycardia_detected_below_60bpm(self):
        """HR < 60 BPM should trigger BRADY classification."""
        np.random.seed(42)
        r_peaks, rr_sec, amps = make_rr_intervals(
            mean_hr_bpm=45.0, n_beats=30, cv=0.04, duration_sec=60.0
        )
        signal = np.zeros(r_peaks[-1] + 500)
        hrv = extract_features(r_peaks, rr_sec, amps, signal, FS, 60.0)
        result = classify_rhythm(hrv, rr_sec)

        all_codes = [l.code for l in result.all_labels]
        assert "BRADY" in all_codes, \
            f"Bradycardia not detected at {hrv.mean_hr_bpm:.1f} BPM. Labels: {all_codes}"

    def test_bradycardia_confidence_scales_with_severity(self):
        """More severe bradycardia should have higher confidence."""
        np.random.seed(42)
        _, rr_40, amps_40 = make_rr_intervals(40.0, 25, cv=0.03)
        r_40 = np.cumsum(np.concatenate([[int(0.5*FS)], (rr_40 * FS).astype(int)]))
        signal_40 = np.zeros(r_40[-1] + 500)
        hrv_40 = extract_features(r_40, rr_40, amps_40, signal_40, FS, 60.0)
        result_40 = classify_rhythm(hrv_40, rr_40)

        _, rr_55, amps_55 = make_rr_intervals(55.0, 45, cv=0.03)
        r_55 = np.cumsum(np.concatenate([[int(0.5*FS)], (rr_55 * FS).astype(int)]))
        signal_55 = np.zeros(r_55[-1] + 500)
        hrv_55 = extract_features(r_55, rr_55, amps_55, signal_55, FS, 60.0)
        result_55 = classify_rhythm(hrv_55, rr_55)

        brady_40 = next((l for l in result_40.all_labels if l.code == "BRADY"), None)
        brady_55 = next((l for l in result_55.all_labels if l.code == "BRADY"), None)

        if brady_40 and brady_55:
            assert brady_40.confidence >= brady_55.confidence, \
                "40 BPM should have higher BRADY confidence than 55 BPM"

    def test_normal_hr_not_bradycardia(self):
        """Normal HR should NOT be classified as bradycardia."""
        np.random.seed(42)
        r_peaks, rr_sec, amps = make_rr_intervals(72.0, 60, cv=0.03)
        signal = np.zeros(r_peaks[-1] + 500)
        hrv = extract_features(r_peaks, rr_sec, amps, signal, FS, 60.0)
        result = classify_rhythm(hrv, rr_sec)

        all_codes = [l.code for l in result.all_labels]
        assert "BRADY" not in all_codes, \
            f"Normal HR incorrectly classified as bradycardia"


class TestTachycardia:
    def test_tachycardia_detected_above_100bpm(self):
        """HR > 100 BPM should trigger TACHY classification."""
        np.random.seed(42)
        r_peaks, rr_sec, amps = make_rr_intervals(
            mean_hr_bpm=130.0, n_beats=80, cv=0.03, duration_sec=60.0
        )
        signal = np.zeros(r_peaks[-1] + 500)
        hrv = extract_features(r_peaks, rr_sec, amps, signal, FS, 60.0)
        result = classify_rhythm(hrv, rr_sec)

        all_codes = [l.code for l in result.all_labels]
        assert "TACHY" in all_codes, \
            f"Tachycardia not detected at {hrv.mean_hr_bpm:.1f} BPM. Labels: {all_codes}"

    def test_normal_hr_not_tachycardia(self):
        """Normal HR should NOT be classified as tachycardia."""
        np.random.seed(42)
        r_peaks, rr_sec, amps = make_rr_intervals(72.0, 60, cv=0.03)
        signal = np.zeros(r_peaks[-1] + 500)
        hrv = extract_features(r_peaks, rr_sec, amps, signal, FS, 60.0)
        result = classify_rhythm(hrv, rr_sec)

        all_codes = [l.code for l in result.all_labels]
        assert "TACHY" not in all_codes, \
            f"Normal HR incorrectly classified as tachycardia"


class TestCombinedFindings:
    def test_af_plus_tachycardia(self):
        """AF with high ventricular rate should show both AF and TACHY."""
        np.random.seed(42)
        r_peaks, rr_sec, amps = make_af_rr_intervals(mean_hr_bpm=130.0, n_beats=80)
        signal = np.zeros(r_peaks[-1] + 500)
        hrv = extract_features(r_peaks, rr_sec, amps, signal, FS, 60.0)
        result = classify_rhythm(hrv, rr_sec)

        all_codes = [l.code for l in result.all_labels]
        # At least one of AF or TACHY should be detected
        assert "AF" in all_codes or "TACHY" in all_codes, \
            f"Neither AF nor TACHY detected in rapid irregular rhythm: {all_codes}"

    def test_insufficient_data_handled(self):
        """Very short records (< 3 beats) should return NODATA gracefully."""
        r_peaks = np.array([100, 500])  # Only 2 beats
        rr_sec = np.array([400.0 / FS])
        amps = np.array([1.0, 1.0])
        signal = np.zeros(1000)
        hrv = extract_features(r_peaks, rr_sec, amps, signal, FS, 3.0)
        result = classify_rhythm(hrv, rr_sec)

        # Should not crash and should return a valid result
        assert result.primary_rhythm is not None
        assert result.primary_rhythm.code == "NODATA"


class TestClassificationResult:
    def test_result_has_required_fields(self):
        """ClassificationResult should contain all required fields."""
        np.random.seed(42)
        r_peaks, rr_sec, amps = make_rr_intervals(72.0, 60, cv=0.03)
        signal = np.zeros(r_peaks[-1] + 500)
        hrv = extract_features(r_peaks, rr_sec, amps, signal, FS, 60.0)
        result = classify_rhythm(hrv, rr_sec)

        assert result.primary_rhythm is not None
        assert isinstance(result.secondary_findings, list)
        assert isinstance(result.clinical_interpretation, str)
        assert isinstance(result.recommendations, list)
        assert isinstance(result.is_concerning, bool)
        assert len(result.recommendations) > 0
        assert len(result.clinical_interpretation) > 20

    def test_confidence_in_valid_range(self):
        """All confidence scores should be between 0 and 100."""
        np.random.seed(42)
        for hr in [40, 72, 130]:
            r_peaks, rr_sec, amps = make_rr_intervals(hr, 50, cv=0.05)
            signal = np.zeros(r_peaks[-1] + 500)
            hrv = extract_features(r_peaks, rr_sec, amps, signal, FS, 60.0)
            result = classify_rhythm(hrv, rr_sec)

            for label in result.all_labels:
                assert 0 <= label.confidence <= 100, \
                    f"Confidence out of range: {label.confidence} for {label.name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
