"""
classifier.py — Rule-based rhythm classification with confidence scoring.

Classifies the overall rhythm into one or more of:
  - Normal Sinus Rhythm (NSR)
  - Atrial Fibrillation (AF)
  - Bradycardia
  - Tachycardia
  - High HRV
  - PVC burden (frequent premature ventricular contractions)

Classification is based on evidence-based thresholds from clinical guidelines
(AHA/ACC/HRS 2014 AF Guidelines, ACLS Bradycardia/Tachycardia criteria).

Each classification includes a confidence score (0–100%) derived from
how strongly the evidence supports that diagnosis.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from .features import HRVFeatures


# ── Clinical thresholds ────────────────────────────────────────────────────────
BRADY_HR_THRESHOLD = 60.0       # BPM — bradycardia
TACHY_HR_THRESHOLD = 100.0      # BPM — tachycardia
AF_CV_THRESHOLD = 0.15          # CV of RR intervals — AF heuristic
AF_CV_DEFINITE = 0.25           # CV above this = very likely AF
NSR_CV_MAX = 0.10               # CV below this = likely regular rhythm
HIGH_HRV_RMSSD = 50.0           # ms — high HRV
LOW_HRV_RMSSD = 15.0            # ms — very low HRV (autonomic dysfunction)
PVC_RR_DEVIATION = 0.25         # 25% shorter than mean = possible PVC


@dataclass
class RhythmLabel:
    """A single rhythm classification with confidence and evidence."""
    name: str
    code: str               # Short code: NSR, AF, BRADY, TACHY, etc.
    confidence: float       # 0–100
    evidence: List[str]     # Human-readable supporting evidence
    severity: str           # "normal", "benign", "moderate", "critical"


@dataclass
class ClassificationResult:
    """Full classification output for one ECG record."""
    primary_rhythm: RhythmLabel
    secondary_findings: List[RhythmLabel]
    all_labels: List[RhythmLabel]
    hrv: HRVFeatures
    clinical_interpretation: str
    recommendations: List[str]
    is_concerning: bool


def classify_rhythm(hrv: HRVFeatures, rr_sec: np.ndarray) -> ClassificationResult:
    """
    Classify the cardiac rhythm from HRV features and RR intervals.

    Strategy:
    1. Score each possible rhythm based on multiple evidence signals
    2. Select primary rhythm as highest-confidence diagnosis
    3. Report secondary findings that co-exist
    4. Generate clinical interpretation text

    Parameters
    ----------
    hrv : HRVFeatures
        Pre-computed HRV feature object.
    rr_sec : np.ndarray
        Raw RR interval array in seconds.

    Returns
    -------
    ClassificationResult
    """
    if hrv.n_beats < 3 or len(rr_sec) < 2:
        # Not enough data
        no_data = RhythmLabel(
            name="Insufficient Data",
            code="NODATA",
            confidence=100.0,
            evidence=["Fewer than 3 beats detected — cannot classify rhythm"],
            severity="normal",
        )
        return ClassificationResult(
            primary_rhythm=no_data,
            secondary_findings=[],
            all_labels=[no_data],
            hrv=hrv,
            clinical_interpretation="Insufficient beats for reliable rhythm analysis.",
            recommendations=["Verify signal quality and increase analysis window."],
            is_concerning=False,
        )

    labels = []

    # ── Atrial Fibrillation ────────────────────────────────────────────────────
    af_label = _classify_af(hrv, rr_sec)
    if af_label:
        labels.append(af_label)

    # ── Bradycardia / Tachycardia ──────────────────────────────────────────────
    rate_labels = _classify_rate(hrv)
    labels.extend(rate_labels)

    # ── Normal Sinus Rhythm ────────────────────────────────────────────────────
    nsr_label = _classify_nsr(hrv, rr_sec, af_label)
    if nsr_label:
        labels.append(nsr_label)

    # ── HRV Commentary ────────────────────────────────────────────────────────
    hrv_label = _classify_hrv(hrv)
    if hrv_label:
        labels.append(hrv_label)

    # ── PVC Burden ────────────────────────────────────────────────────────────
    pvc_label = _classify_pvcs(hrv, rr_sec)
    if pvc_label:
        labels.append(pvc_label)

    # Sort by confidence descending
    labels.sort(key=lambda x: x.confidence, reverse=True)

    # Primary = highest confidence rhythm-defining label
    rhythm_defining = [l for l in labels
                       if l.code in ("NSR", "AF", "BRADY", "TACHY", "NODATA")]
    if rhythm_defining:
        primary = rhythm_defining[0]
        secondary = [l for l in labels if l is not primary]
    else:
        primary = labels[0] if labels else _default_label(hrv)
        secondary = labels[1:]

    interpretation = _build_interpretation(primary, secondary, hrv)
    recommendations = _build_recommendations(primary, secondary, hrv)
    is_concerning = primary.severity in ("moderate", "critical") or any(
        l.severity in ("moderate", "critical") for l in secondary
    )

    return ClassificationResult(
        primary_rhythm=primary,
        secondary_findings=secondary,
        all_labels=labels,
        hrv=hrv,
        clinical_interpretation=interpretation,
        recommendations=recommendations,
        is_concerning=is_concerning,
    )


# ── Individual classifiers ─────────────────────────────────────────────────────

def _classify_af(hrv: HRVFeatures, rr_sec: np.ndarray) -> Optional[RhythmLabel]:
    """
    Atrial fibrillation heuristic based on RR interval irregularity.

    Key indicators:
    - High coefficient of variation (CV) of RR intervals
    - High RMSSD relative to mean RR
    - Absence of consistent P-P regularity (approximated by high SD1)
    - pNN50 (many large inter-beat differences)
    """
    evidence = []
    score = 0.0

    cv = hrv.cv_rr
    if cv > AF_CV_DEFINITE:
        evidence.append(f"Very high RR variability (CV={cv:.3f} > {AF_CV_DEFINITE})")
        score += 50
    elif cv > AF_CV_THRESHOLD:
        evidence.append(f"Elevated RR variability (CV={cv:.3f} > {AF_CV_THRESHOLD})")
        score += 30

    if hrv.rmssd_ms > 80:
        evidence.append(f"High RMSSD ({hrv.rmssd_ms:.1f}ms > 80ms)")
        score += 20
    elif hrv.rmssd_ms > 50:
        score += 10

    if hrv.pnn50_pct > 40:
        evidence.append(f"High pNN50 ({hrv.pnn50_pct:.1f}% > 40%)")
        score += 20
    elif hrv.pnn50_pct > 20:
        score += 10

    if hrv.sd1_ms > 50:
        evidence.append(f"High Poincaré SD1 ({hrv.sd1_ms:.1f}ms > 50ms)")
        score += 10

    if hrv.irregular_beats > hrv.n_beats * 0.15:
        pct = 100 * hrv.irregular_beats / hrv.n_beats
        evidence.append(f"High irregular beat fraction ({pct:.1f}%)")
        score += 15

    # Confidence caps at 95 (we can't fully confirm without P-wave analysis)
    confidence = min(95.0, score)

    if confidence < 25:
        return None

    return RhythmLabel(
        name="Atrial Fibrillation",
        code="AF",
        confidence=round(confidence, 1),
        evidence=evidence,
        severity="moderate",
    )


def _classify_rate(hrv: HRVFeatures) -> List[RhythmLabel]:
    """Bradycardia and tachycardia based on mean heart rate."""
    labels = []
    hr = hrv.mean_hr_bpm

    if hr < BRADY_HR_THRESHOLD and hr > 0:
        deficit = BRADY_HR_THRESHOLD - hr
        # Confidence scales with how far below 60 BPM
        confidence = min(95.0, 60.0 + deficit * 1.5)
        severity = "moderate" if hr < 40 else "benign"
        labels.append(RhythmLabel(
            name="Bradycardia",
            code="BRADY",
            confidence=round(confidence, 1),
            evidence=[f"Mean heart rate {hr:.1f} BPM < 60 BPM threshold"],
            severity=severity,
        ))

    if hr > TACHY_HR_THRESHOLD:
        excess = hr - TACHY_HR_THRESHOLD
        confidence = min(95.0, 60.0 + excess * 0.8)
        severity = "moderate" if hr > 150 else "benign"
        labels.append(RhythmLabel(
            name="Tachycardia",
            code="TACHY",
            confidence=round(confidence, 1),
            evidence=[f"Mean heart rate {hr:.1f} BPM > 100 BPM threshold"],
            severity=severity,
        ))

    return labels


def _classify_nsr(
    hrv: HRVFeatures, rr_sec: np.ndarray, af_label: Optional[RhythmLabel]
) -> Optional[RhythmLabel]:
    """
    Normal Sinus Rhythm: regular rate (60–100 BPM) with consistent RR intervals.
    """
    hr = hrv.mean_hr_bpm
    cv = hrv.cv_rr

    evidence = []
    score = 0.0

    if 60 <= hr <= 100:
        evidence.append(f"Heart rate {hr:.1f} BPM in normal range (60–100 BPM)")
        score += 40
    else:
        score -= 20

    if cv < NSR_CV_MAX:
        evidence.append(f"Regular RR intervals (CV={cv:.3f} < {NSR_CV_MAX})")
        score += 40
    elif cv < 0.15:
        score += 20

    if hrv.n_beats >= 10:
        evidence.append(f"{hrv.n_beats} beats analyzed")
        score += 10

    # Penalize heavily if AF is already suspected
    if af_label and af_label.confidence > 50:
        score -= af_label.confidence * 0.6

    confidence = max(0.0, min(95.0, score))

    if confidence < 30:
        return None

    severity = "normal"
    return RhythmLabel(
        name="Normal Sinus Rhythm",
        code="NSR",
        confidence=round(confidence, 1),
        evidence=evidence,
        severity=severity,
    )


def _classify_hrv(hrv: HRVFeatures) -> Optional[RhythmLabel]:
    """Flag notable HRV findings."""
    if hrv.rmssd_ms > HIGH_HRV_RMSSD:
        return RhythmLabel(
            name="High HRV",
            code="HIGH_HRV",
            confidence=min(90.0, 50.0 + hrv.rmssd_ms * 0.3),
            evidence=[
                f"RMSSD {hrv.rmssd_ms:.1f}ms > {HIGH_HRV_RMSSD}ms",
                f"Indicates strong parasympathetic tone or arrhythmia",
            ],
            severity="benign",
        )
    if hrv.rmssd_ms < LOW_HRV_RMSSD and hrv.n_beats > 20:
        return RhythmLabel(
            name="Reduced HRV",
            code="LOW_HRV",
            confidence=min(85.0, 50.0 + (LOW_HRV_RMSSD - hrv.rmssd_ms) * 2),
            evidence=[
                f"RMSSD {hrv.rmssd_ms:.1f}ms < {LOW_HRV_RMSSD}ms",
                "May indicate autonomic dysfunction or cardiac pathology",
            ],
            severity="benign",
        )
    return None


def _classify_pvcs(hrv: HRVFeatures, rr_sec: np.ndarray) -> Optional[RhythmLabel]:
    """
    Estimate PVC burden based on early beat pattern.
    PVCs cause a short RR followed by a compensatory long RR.
    """
    if len(rr_sec) < 6:
        return None

    rr_ms = rr_sec * 1000.0
    mean_rr = np.mean(rr_ms)

    # Count short-long pairs (coupling pattern typical of PVCs)
    pvc_count = 0
    for i in range(len(rr_ms) - 1):
        short = rr_ms[i] < (1 - PVC_RR_DEVIATION) * mean_rr
        long = rr_ms[i + 1] > (1 + PVC_RR_DEVIATION) * mean_rr
        if short and long:
            pvc_count += 1

    if pvc_count == 0:
        return None

    burden_pct = 100.0 * pvc_count / len(rr_ms)
    if burden_pct < 3:
        return None

    confidence = min(75.0, 30.0 + burden_pct * 2)
    severity = "moderate" if burden_pct > 10 else "benign"

    return RhythmLabel(
        name="Frequent Ectopic Beats (possible PVCs)",
        code="PVC",
        confidence=round(confidence, 1),
        evidence=[
            f"{pvc_count} short-long RR coupling pairs detected",
            f"Ectopic burden ~{burden_pct:.1f}%",
        ],
        severity=severity,
    )


def _default_label(hrv: HRVFeatures) -> RhythmLabel:
    return RhythmLabel(
        name="Undetermined",
        code="UNK",
        confidence=0.0,
        evidence=["Insufficient evidence for classification"],
        severity="normal",
    )


def _build_interpretation(
    primary: RhythmLabel,
    secondary: List[RhythmLabel],
    hrv: HRVFeatures,
) -> str:
    """Generate a plain-English clinical interpretation."""
    lines = []

    lines.append(
        f"Primary rhythm: {primary.name} (confidence {primary.confidence:.0f}%). "
        + "; ".join(primary.evidence[:2])
    )

    if secondary:
        sec_names = ", ".join(
            f"{l.name} ({l.confidence:.0f}%)" for l in secondary[:3]
        )
        lines.append(f"Additional findings: {sec_names}.")

    lines.append(
        f"Heart rate {hrv.mean_hr_bpm:.1f} BPM "
        f"(range {hrv.min_hr_bpm:.1f}–{hrv.max_hr_bpm:.1f} BPM). "
        f"SDNN {hrv.std_rr_ms:.1f}ms, RMSSD {hrv.rmssd_ms:.1f}ms."
    )

    return " ".join(lines)


def _build_recommendations(
    primary: RhythmLabel,
    secondary: List[RhythmLabel],
    hrv: HRVFeatures,
) -> List[str]:
    """Generate clinical action recommendations."""
    recs = []

    if primary.code == "AF":
        recs.append("Confirm with 12-lead ECG and clinical correlation.")
        recs.append("Evaluate for thromboembolic risk (CHA₂DS₂-VASc score).")
        recs.append("Consider Holter monitor for persistent vs. paroxysmal AF.")
    elif primary.code == "BRADY":
        recs.append("Evaluate for symptomatic bradycardia (syncope, fatigue).")
        recs.append("Review medications (beta-blockers, calcium channel blockers).")
        if hrv.mean_hr_bpm < 40:
            recs.append("URGENT: HR < 40 BPM — consider pacing evaluation.")
    elif primary.code == "TACHY":
        recs.append("Determine tachycardia type (sinus vs. supraventricular vs. ventricular).")
        recs.append("Assess for underlying cause (dehydration, anxiety, thyroid disease).")
    elif primary.code == "NSR":
        recs.append("No immediate intervention indicated for rhythm.")

    any_pvc = next((l for l in secondary if l.code == "PVC"), None)
    if any_pvc:
        recs.append("PVC burden noted — correlation with symptoms recommended.")

    recs.append("This analysis is algorithmic and not a substitute for physician review.")
    return recs
