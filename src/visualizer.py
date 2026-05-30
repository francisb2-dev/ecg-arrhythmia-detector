"""
visualizer.py — Clinical-grade ECG visualization.

All plots use a consistent dark clinical aesthetic similar to hospital
ECG monitor displays. Figures are saved as high-quality PNGs and can
be encoded to base64 for embedding in HTML reports.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script use
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import io
import base64
from pathlib import Path
from typing import Optional, List

from .features import HRVFeatures, get_instantaneous_hr
from .detector import DetectionResult
from .classifier import ClassificationResult
from .preprocessor import FilteredSignal


# ── Color palette ─────────────────────────────────────────────────────────────
COLORS = {
    "background":  "#0d1117",
    "panel":       "#161b22",
    "ecg_raw":     "#4a9eff",
    "ecg_filtered": "#00d4aa",
    "r_peak":      "#ff6b6b",
    "rr_line":     "#ffd93d",
    "hr_line":     "#ff6b6b",
    "grid":        "#21262d",
    "text":        "#e6edf3",
    "text_dim":    "#8b949e",
    "accent":      "#58a6ff",
    "normal":      "#3fb950",
    "warning":     "#d29922",
    "critical":    "#f85149",
    "lf":          "#ff9500",
    "hf":          "#30d158",
    "vlf":         "#bf5af2",
}

plt.rcParams.update({
    "figure.facecolor":   COLORS["background"],
    "axes.facecolor":     COLORS["panel"],
    "axes.edgecolor":     COLORS["grid"],
    "axes.labelcolor":    COLORS["text"],
    "axes.titlecolor":    COLORS["text"],
    "xtick.color":        COLORS["text_dim"],
    "ytick.color":        COLORS["text_dim"],
    "grid.color":         COLORS["grid"],
    "grid.alpha":         0.6,
    "text.color":         COLORS["text"],
    "font.family":        "monospace",
    "font.size":          9,
})


def _fig_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _save_fig(fig: plt.Figure, path: Path) -> None:
    """Save figure to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_raw_vs_filtered(
    filtered: FilteredSignal,
    detection: DetectionResult,
    record_name: str,
    max_sec: float = 10.0,
) -> str:
    """
    Two-panel plot: raw ECG (top) and filtered ECG with R-peak markers (bottom).
    Shows up to `max_sec` seconds for clarity.
    """
    fs = filtered.fs
    max_samples = int(max_sec * fs)
    n = min(max_samples, len(filtered.raw))
    t = np.arange(n) / fs

    raw = filtered.raw[:n]
    filt = filtered.filtered[:n]

    # R-peaks within this window
    peaks_in_window = detection.r_peaks[detection.r_peaks < n]

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), facecolor=COLORS["background"])
    fig.suptitle(f"ECG Signal Analysis — Record {record_name}",
                 fontsize=12, color=COLORS["text"], fontweight="bold", y=0.98)

    # Top: raw signal
    ax1 = axes[0]
    ax1.plot(t, raw, color=COLORS["ecg_raw"], linewidth=0.8, alpha=0.9)
    ax1.set_title("Raw ECG Signal", fontsize=10, pad=4)
    ax1.set_ylabel("Amplitude (mV)", fontsize=9)
    ax1.grid(True, linewidth=0.4)
    ax1.set_xlim([0, max_sec])
    ax1.tick_params(labelbottom=False)

    # Bottom: filtered + R-peaks
    ax2 = axes[1]
    ax2.plot(t, filt, color=COLORS["ecg_filtered"], linewidth=0.9, alpha=0.95,
             label="Filtered ECG (0.5–40 Hz)")
    if len(peaks_in_window) > 0:
        ax2.scatter(
            peaks_in_window / fs,
            filt[peaks_in_window],
            color=COLORS["r_peak"],
            s=40, zorder=5, marker="v", label=f"R-peaks (n={len(peaks_in_window)})"
        )
    ax2.set_title("Filtered ECG with QRS Detection", fontsize=10, pad=4)
    ax2.set_xlabel("Time (seconds)", fontsize=9)
    ax2.set_ylabel("Amplitude (mV)", fontsize=9)
    ax2.grid(True, linewidth=0.4)
    ax2.set_xlim([0, max_sec])
    ax2.legend(loc="upper right", fontsize=8, facecolor=COLORS["panel"],
               edgecolor=COLORS["grid"])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return _fig_to_base64(fig)


def plot_rr_tachogram(
    detection: DetectionResult,
    hrv: HRVFeatures,
    record_name: str,
) -> str:
    """
    RR interval tachogram (beat-to-beat variability plot).
    Highlights beats with abnormal RR intervals.
    """
    rr_ms = detection.rr_intervals_sec * 1000.0
    beat_nums = np.arange(1, len(rr_ms) + 1)
    mean_rr = np.mean(rr_ms)
    std_rr = np.std(rr_ms)

    fig, ax = plt.subplots(figsize=(12, 4), facecolor=COLORS["background"])
    ax.set_facecolor(COLORS["panel"])

    # Color each point by deviation from mean
    colors = []
    for rr in rr_ms:
        dev = abs(rr - mean_rr) / std_rr if std_rr > 0 else 0
        if dev > 2.5:
            colors.append(COLORS["critical"])
        elif dev > 1.5:
            colors.append(COLORS["warning"])
        else:
            colors.append(COLORS["ecg_filtered"])

    ax.plot(beat_nums, rr_ms, color=COLORS["ecg_filtered"], linewidth=0.8,
            alpha=0.5, zorder=1)
    ax.scatter(beat_nums, rr_ms, c=colors, s=18, zorder=2, alpha=0.9)

    # Mean ± 2SD bands
    ax.axhline(mean_rr, color=COLORS["normal"], linewidth=1.2, linestyle="--",
               alpha=0.8, label=f"Mean RR = {mean_rr:.1f}ms")
    ax.fill_between(beat_nums, mean_rr - 2 * std_rr, mean_rr + 2 * std_rr,
                    alpha=0.12, color=COLORS["accent"], label="±2 SD band")

    ax.set_title(f"RR Interval Tachogram — Record {record_name}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Beat Number", fontsize=9)
    ax.set_ylabel("RR Interval (ms)", fontsize=9)
    ax.legend(fontsize=8, facecolor=COLORS["panel"], edgecolor=COLORS["grid"])
    ax.grid(True, linewidth=0.4)

    # Annotate CV
    ax.text(0.02, 0.95,
            f"CV = {hrv.cv_rr:.4f} | SDNN = {hrv.std_rr_ms:.1f}ms | "
            f"RMSSD = {hrv.rmssd_ms:.1f}ms",
            transform=ax.transAxes, fontsize=8.5, color=COLORS["text_dim"],
            verticalalignment="top")

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_heart_rate_trend(
    detection: DetectionResult,
    hrv: HRVFeatures,
    record_name: str,
) -> str:
    """Heart rate over time with normal range shading."""
    times, hr = get_instantaneous_hr(
        detection.r_peaks, detection.fs, len(detection.integrated_signal)
    )
    if len(times) == 0:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)
        return _fig_to_base64(fig)

    fig, ax = plt.subplots(figsize=(12, 4), facecolor=COLORS["background"])
    ax.set_facecolor(COLORS["panel"])

    ax.plot(times, hr, color=COLORS["hr_line"], linewidth=1.2, alpha=0.9)
    ax.fill_between(times, hr, alpha=0.15, color=COLORS["hr_line"])

    # Normal range band
    ax.axhspan(60, 100, alpha=0.06, color=COLORS["normal"], label="Normal range (60–100 BPM)")
    ax.axhline(60, color=COLORS["normal"], linewidth=0.7, linestyle=":", alpha=0.6)
    ax.axhline(100, color=COLORS["normal"], linewidth=0.7, linestyle=":", alpha=0.6)

    ax.set_title(f"Heart Rate Trend — Record {record_name}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=9)
    ax.set_ylabel("Heart Rate (BPM)", fontsize=9)
    ax.legend(fontsize=8, facecolor=COLORS["panel"], edgecolor=COLORS["grid"])
    ax.grid(True, linewidth=0.4)

    # Annotate mean
    ax.text(0.02, 0.95,
            f"Mean: {hrv.mean_hr_bpm:.1f} BPM | "
            f"Range: {hrv.min_hr_bpm:.1f}–{hrv.max_hr_bpm:.1f} BPM",
            transform=ax.transAxes, fontsize=8.5, color=COLORS["text_dim"],
            verticalalignment="top")

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_poincare(
    detection: DetectionResult,
    hrv: HRVFeatures,
    record_name: str,
) -> str:
    """
    Poincaré plot (RR_n vs RR_{n+1}).

    The shape of the Poincaré cloud encodes rhythm information:
    - Tight football shape → regular rhythm (NSR)
    - Elongated along identity line → high HRV
    - Diffuse cloud → AF or very irregular rhythm
    """
    rr_ms = detection.rr_intervals_sec * 1000.0
    if len(rr_ms) < 3:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)
        return _fig_to_base64(fig)

    rr_n = rr_ms[:-1]
    rr_n1 = rr_ms[1:]

    fig, ax = plt.subplots(figsize=(6, 6), facecolor=COLORS["background"])
    ax.set_facecolor(COLORS["panel"])

    # Density scatter
    ax.scatter(rr_n, rr_n1, alpha=0.4, s=15, color=COLORS["accent"],
               edgecolors="none")

    # Identity line (RR_n = RR_{n+1} → perfectly regular)
    rr_range = [min(rr_ms) - 20, max(rr_ms) + 20]
    ax.plot(rr_range, rr_range, color=COLORS["text_dim"], linewidth=0.8,
            linestyle="--", alpha=0.6, label="Identity (RR_n = RR_n+1)")

    # SD1/SD2 ellipse indication
    mean_rr = np.mean(rr_ms)
    ax.errorbar(mean_rr, mean_rr,
                xerr=hrv.sd2_ms, yerr=hrv.sd1_ms,
                fmt="o", color=COLORS["r_peak"], markersize=6, linewidth=1.5,
                capsize=4, label=f"SD1={hrv.sd1_ms:.1f}ms  SD2={hrv.sd2_ms:.1f}ms")

    ax.set_title(f"Poincaré Plot — Record {record_name}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("RR_n (ms)", fontsize=9)
    ax.set_ylabel("RR_{n+1} (ms)", fontsize=9)
    ax.legend(fontsize=8, facecolor=COLORS["panel"], edgecolor=COLORS["grid"])
    ax.grid(True, linewidth=0.4)
    ax.set_aspect("equal")

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_frequency_domain(hrv: HRVFeatures, record_name: str) -> str:
    """
    HRV frequency domain bar chart with VLF, LF, HF power bands.
    """
    bands = ["VLF\n(0.003–0.04 Hz)", "LF\n(0.04–0.15 Hz)", "HF\n(0.15–0.4 Hz)"]
    powers = [hrv.vlf_power, hrv.lf_power, hrv.hf_power]
    bar_colors = [COLORS["vlf"], COLORS["lf"], COLORS["hf"]]

    total = sum(powers)
    pcts = [100 * p / total if total > 0 else 0 for p in powers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                                    facecolor=COLORS["background"])

    # Bar chart
    ax1.set_facecolor(COLORS["panel"])
    bars = ax1.bar(bands, powers, color=bar_colors, alpha=0.85, edgecolor=COLORS["grid"])
    ax1.set_title("HRV Frequency Band Power", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Power (ms²)", fontsize=9)
    ax1.grid(True, axis="y", linewidth=0.4)
    for bar, pct in zip(bars, pcts):
        ax1.text(bar.get_x() + bar.get_width() / 2.0,
                 bar.get_height() + 0.5,
                 f"{pct:.1f}%", ha="center", va="bottom", fontsize=8.5,
                 color=COLORS["text"])

    # Pie chart
    ax2.set_facecolor(COLORS["background"])
    if total > 0:
        wedges, texts, autotexts = ax2.pie(
            [max(0.001, p) for p in powers],
            labels=["VLF", "LF", "HF"],
            colors=bar_colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"color": COLORS["text"], "fontsize": 8.5},
        )
        for at in autotexts:
            at.set_color(COLORS["background"])
            at.set_fontweight("bold")
    else:
        ax2.text(0.5, 0.5, "Insufficient data for\nfrequency analysis",
                 ha="center", va="center", transform=ax2.transAxes,
                 fontsize=9, color=COLORS["text_dim"])

    ax2.set_title(f"LF/HF Ratio: {hrv.lf_hf_ratio:.2f}", fontsize=10,
                  fontweight="bold")

    fig.suptitle(f"HRV Frequency Domain — Record {record_name}",
                 fontsize=11, fontweight="bold", y=1.01)

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_qrs_overlay(
    filtered: FilteredSignal,
    detection: DetectionResult,
    record_name: str,
    n_beats: int = 20,
) -> str:
    """
    Superimposed QRS complexes — all detected beats overlaid on a common
    time axis. Reveals morphological consistency or variability.
    """
    fs = filtered.fs
    half_win = int(0.25 * fs)  # ±250ms around R-peak
    beats = []

    for rp in detection.r_peaks[:n_beats]:
        start = rp - half_win
        end = rp + half_win
        if start >= 0 and end < len(filtered.filtered):
            beat = filtered.filtered[start:end]
            beats.append(beat)

    if not beats:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No beats to overlay", ha="center", va="center",
                transform=ax.transAxes)
        return _fig_to_base64(fig)

    t_ms = np.linspace(-250, 250, 2 * half_win)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=COLORS["background"])
    ax.set_facecolor(COLORS["panel"])

    for i, beat in enumerate(beats):
        alpha = 0.3 + 0.5 * (i == 0)
        color = COLORS["ecg_filtered"] if i > 0 else COLORS["r_peak"]
        ax.plot(t_ms[:len(beat)], beat, color=color, linewidth=0.7, alpha=alpha)

    if beats:
        mean_beat = np.mean([b for b in beats if len(b) == 2 * half_win], axis=0)
        if len(mean_beat) == 2 * half_win:
            ax.plot(t_ms, mean_beat, color="white", linewidth=1.8, alpha=0.9,
                    label="Mean beat")

    ax.axvline(0, color=COLORS["r_peak"], linewidth=1.0, linestyle="--",
               alpha=0.7, label="R-peak")
    ax.set_title(f"QRS Complex Overlay — Record {record_name} (n={len(beats)} beats)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Time relative to R-peak (ms)", fontsize=9)
    ax.set_ylabel("Amplitude (mV)", fontsize=9)
    ax.legend(fontsize=8, facecolor=COLORS["panel"], edgecolor=COLORS["grid"])
    ax.grid(True, linewidth=0.4)

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_all_plots(
    filtered: FilteredSignal,
    detection: DetectionResult,
    hrv: HRVFeatures,
    classification: ClassificationResult,
    record_name: str,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Generate all plots and return a dict of base64-encoded PNG strings.
    Optionally saves PNG files to output_dir.
    """
    print("  Generating plots...")

    plots = {}

    plots["raw_vs_filtered"] = plot_raw_vs_filtered(
        filtered, detection, record_name
    )
    plots["rr_tachogram"] = plot_rr_tachogram(detection, hrv, record_name)
    plots["heart_rate_trend"] = plot_heart_rate_trend(detection, hrv, record_name)
    plots["poincare"] = plot_poincare(detection, hrv, record_name)
    plots["frequency_domain"] = plot_frequency_domain(hrv, record_name)
    plots["qrs_overlay"] = plot_qrs_overlay(filtered, detection, record_name)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, b64 in plots.items():
            img_data = base64.b64decode(b64)
            path = output_dir / f"{record_name}_{name}.png"
            path.write_bytes(img_data)

    return plots
