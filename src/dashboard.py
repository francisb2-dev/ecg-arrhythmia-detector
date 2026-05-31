"""
dashboard.py — Interactive Plotly HTML dashboard for ECG analysis results.

Generates a single self-contained HTML file with interactive charts:
  - ECG signal with QRS markers (colored by beat type if ML available)
  - RR interval tachogram with hoverable beat info
  - Heart rate over time
  - Poincaré plot with SD1/SD2 ellipse
  - HRV frequency domain (LF/HF power bands)
  - Beat type distribution (if ML classifier used)
  - Summary metrics panel
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional

# Beat type colors
BEAT_COLORS = {
    'N': '#4ade80',   # green — normal
    'V': '#f87171',   # red — PVC
    'A': '#fb923c',   # orange — atrial ectopic
    'L': '#a78bfa',   # purple — left BBB
    'R': '#60a5fa',   # blue — right BBB
    'F': '#facc15',   # yellow — fusion
    'Q': '#94a3b8',   # gray — unknown
}

BEAT_NAMES = {
    'N': 'Normal', 'V': 'PVC', 'A': 'Atrial Ectopic',
    'L': 'Left BBB', 'R': 'Right BBB', 'F': 'Fusion', 'Q': 'Other',
}


def generate_interactive_report(
    record_id: str,
    signal: np.ndarray,
    fs: float,
    r_peaks: np.ndarray,
    rr_sec: np.ndarray,
    features,           # HRVFeatures
    classification,     # ClassificationResult
    output_dir: Path,
    ml_result=None,     # Optional[MLClassificationResult]
    duration_sec: float = 60.0,
) -> Path:
    """
    Generate an interactive HTML report using Plotly.

    Parameters
    ----------
    record_id : str
    signal : np.ndarray — raw ECG signal (first channel)
    fs : float — sampling frequency
    r_peaks : np.ndarray — R-peak sample indices
    rr_sec : np.ndarray — RR intervals in seconds
    features : HRVFeatures
    classification : ClassificationResult
    output_dir : Path
    ml_result : MLClassificationResult or None
    duration_sec : float — seconds of signal to display

    Returns
    -------
    Path to generated HTML file
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f'ecg_interactive_{record_id}.html'

    # ── Time axis for ECG display ─────────────────────────────────────────────
    n_display = min(len(signal), int(duration_sec * fs))
    t_ecg = np.arange(n_display) / fs
    sig_display = signal[:n_display]

    # R-peaks within display window
    r_in_window = r_peaks[r_peaks < n_display]
    t_r = r_in_window / fs
    sig_r = sig_display[r_in_window]

    # Beat colors for QRS markers
    marker_colors = []
    marker_labels = []
    if ml_result and ml_result.beat_predictions:
        pred_map = {p.beat_index: p for p in ml_result.beat_predictions}
        for i, r_idx in enumerate(r_in_window):
            orig_idx = np.searchsorted(r_peaks, r_idx)
            pred = pred_map.get(orig_idx)
            if pred:
                marker_colors.append(BEAT_COLORS.get(pred.predicted_class, '#94a3b8'))
                marker_labels.append(f"Beat {i+1}<br>Type: {pred.class_name}<br>Conf: {pred.confidence:.0%}")
            else:
                marker_colors.append('#b8965a')
                marker_labels.append(f"Beat {i+1}")
    else:
        marker_colors = ['#b8965a'] * len(r_in_window)
        marker_labels = [f"Beat {i+1}" for i in range(len(r_in_window))]

    # ── RR tachogram ─────────────────────────────────────────────────────────
    rr_ms = rr_sec * 1000.0
    beat_numbers = np.arange(1, len(rr_ms) + 1)

    # ── Heart rate over time ──────────────────────────────────────────────────
    hr_bpm = 60000.0 / (rr_ms + 1e-9)
    hr_bpm = np.clip(hr_bpm, 0, 300)

    # ── Poincaré plot ─────────────────────────────────────────────────────────
    rr_n = rr_ms[:-1]
    rr_n1 = rr_ms[1:]

    # SD1 / SD2 ellipse
    sd1 = features.sd1_ms
    sd2 = features.sd2_ms
    mean_rr = features.mean_rr_ms
    theta = np.linspace(0, 2 * np.pi, 100)
    ellipse_x = mean_rr + sd2 * np.cos(theta) * np.cos(np.pi / 4) - sd1 * np.sin(theta) * np.sin(np.pi / 4)
    ellipse_y = mean_rr + sd2 * np.cos(theta) * np.sin(np.pi / 4) + sd1 * np.sin(theta) * np.cos(np.pi / 4)

    # ── HRV Frequency domain ──────────────────────────────────────────────────
    has_freq = features.lf_power > 0 or features.hf_power > 0
    if has_freq and len(rr_sec) > 30:
        from scipy import signal as sp_signal
        rr_interp_fs = 4.0
        t_rr = np.cumsum(rr_sec)
        t_interp = np.arange(t_rr[0], t_rr[-1], 1.0 / rr_interp_fs)
        rr_interp = np.interp(t_interp, t_rr, rr_sec * 1000.0)
        rr_detrended = rr_interp - np.mean(rr_interp)
        freqs, psd = sp_signal.periodogram(rr_detrended, rr_interp_fs)
        freq_mask = freqs <= 0.5
        freqs = freqs[freq_mask]
        psd = psd[freq_mask]
        vlf_mask = freqs < 0.04
        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs <= 0.4)
    else:
        freqs = psd = vlf_mask = lf_mask = hf_mask = None

    # ── Figure layout ─────────────────────────────────────────────────────────
    n_rows = 3 if not has_freq else 3
    has_ml = ml_result is not None and len(ml_result.beat_predictions) > 0

    specs = [
        [{"colspan": 2}, None],
        [{"colspan": 2}, None],
        [{}, {}],
    ]
    subplot_titles = [
        f"ECG Signal — Record {record_id} (first {int(duration_sec)}s)",
        "RR Interval Tachogram",
        "Poincaré Plot",
        "HRV Frequency Domain" if freqs is not None else "Heart Rate Over Time",
    ]

    fig = make_subplots(
        rows=3, cols=2,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )

    # ── ECG signal ────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=t_ecg, y=sig_display,
        mode='lines',
        name='ECG',
        line=dict(color='#ede9e0', width=0.8),
        hoverinfo='skip',
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t_r, y=sig_r,
        mode='markers',
        name='QRS',
        marker=dict(color=marker_colors, size=7, symbol='circle'),
        text=marker_labels,
        hovertemplate='%{text}<extra></extra>',
    ), row=1, col=1)

    # ── RR tachogram ─────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=beat_numbers, y=rr_ms,
        mode='lines+markers',
        name='RR Interval',
        line=dict(color='#b8965a', width=1.5),
        marker=dict(size=3, color='#b8965a'),
        hovertemplate='Beat %{x}<br>RR: %{y:.1f} ms<extra></extra>',
    ), row=2, col=1)

    # Mean RR line
    fig.add_hline(
        y=features.mean_rr_ms,
        line_dash='dash', line_color='#524e5c',
        annotation_text=f"Mean: {features.mean_rr_ms:.0f}ms",
        annotation_position='right',
        row=2, col=1,
    )

    # ── Poincaré plot ─────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=rr_n, y=rr_n1,
        mode='markers',
        name='Poincaré',
        marker=dict(color='#b8965a', size=4, opacity=0.6),
        hovertemplate='RR(n): %{x:.1f}ms<br>RR(n+1): %{y:.1f}ms<extra></extra>',
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=ellipse_x, y=ellipse_y,
        mode='lines',
        name='SD1/SD2 Ellipse',
        line=dict(color='#f87171', width=1.5, dash='dot'),
        hoverinfo='skip',
    ), row=3, col=1)

    # ── Frequency domain or HR ────────────────────────────────────────────────
    if freqs is not None:
        if vlf_mask.any():
            fig.add_trace(go.Scatter(
                x=freqs[vlf_mask], y=psd[vlf_mask],
                fill='tozeroy', name='VLF',
                line=dict(color='#524e5c'),
                fillcolor='rgba(82,78,92,0.3)',
                hoverinfo='skip',
            ), row=3, col=2)
        if lf_mask.any():
            fig.add_trace(go.Scatter(
                x=freqs[lf_mask], y=psd[lf_mask],
                fill='tozeroy', name='LF',
                line=dict(color='#b8965a'),
                fillcolor='rgba(184,150,90,0.4)',
                hoverinfo='skip',
            ), row=3, col=2)
        if hf_mask.any():
            fig.add_trace(go.Scatter(
                x=freqs[hf_mask], y=psd[hf_mask],
                fill='tozeroy', name='HF',
                line=dict(color='#60a5fa'),
                fillcolor='rgba(96,165,250,0.4)',
                hoverinfo='skip',
            ), row=3, col=2)
    else:
        fig.add_trace(go.Scatter(
            x=beat_numbers, y=hr_bpm,
            mode='lines',
            name='Heart Rate',
            line=dict(color='#60a5fa', width=1.5),
            hovertemplate='Beat %{x}<br>HR: %{y:.1f} BPM<extra></extra>',
        ), row=3, col=2)

    # ── Layout ────────────────────────────────────────────────────────────────
    primary = classification.primary_rhythm
    hrv = features

    fig.update_layout(
        title=dict(
            text=(
                f"<b>ECG Analysis Report — Record {record_id}</b><br>"
                f"<span style='font-size:13px;color:#b8965a'>"
                f"{primary.name} &nbsp;|&nbsp; "
                f"HR {hrv.mean_hr_bpm:.1f} BPM &nbsp;|&nbsp; "
                f"SDNN {hrv.std_rr_ms:.1f}ms &nbsp;|&nbsp; "
                f"RMSSD {hrv.rmssd_ms:.1f}ms"
                f"</span>"
            ),
            font=dict(color='#ede9e0', size=16),
            x=0.5,
        ),
        paper_bgcolor='#08070a',
        plot_bgcolor='#0f0e12',
        font=dict(color='#ede9e0', family='SF Pro Display, -apple-system, sans-serif', size=11),
        legend=dict(
            bgcolor='#16141a',
            bordercolor='#1e1c24',
            borderwidth=1,
            font=dict(size=10),
        ),
        height=900,
        margin=dict(l=60, r=60, t=100, b=60),
        hovermode='closest',
    )

    # Axis styling
    axis_style = dict(
        gridcolor='#1e1c24',
        zerolinecolor='#2e2b35',
        tickcolor='#524e5c',
        linecolor='#1e1c24',
    )
    for i in range(1, 7):
        fig.update_xaxes(**axis_style)
        fig.update_yaxes(**axis_style)

    # Axis labels
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude (mV)", row=1, col=1)
    fig.update_xaxes(title_text="Beat number", row=2, col=1)
    fig.update_yaxes(title_text="RR interval (ms)", row=2, col=1)
    fig.update_xaxes(title_text="RR(n) (ms)", row=3, col=1)
    fig.update_yaxes(title_text="RR(n+1) (ms)", row=3, col=1)
    if freqs is not None:
        fig.update_xaxes(title_text="Frequency (Hz)", row=3, col=2)
        fig.update_yaxes(title_text="PSD (ms²/Hz)", row=3, col=2)
    else:
        fig.update_xaxes(title_text="Beat number", row=3, col=2)
        fig.update_yaxes(title_text="Heart rate (BPM)", row=3, col=2)

    # ── Build metrics summary as annotation ───────────────────────────────────
    ml_line = ""
    if has_ml:
        ml_line = (
            f"PVC burden: {ml_result.pvc_burden_pct:.1f}%  |  "
            f"Abnormal beats: {ml_result.abnormal_burden_pct:.1f}%  |  "
        )
    conf_pct = classification.primary_rhythm.confidence

    fig.add_annotation(
        text=(
            f"<b>PRIMARY:</b> {primary.name} ({conf_pct:.0f}% confidence)  |  "
            f"Beats analyzed: {hrv.n_beats}  |  "
            f"{ml_line}"
            f"SD1: {hrv.sd1_ms:.1f}ms  SD2: {hrv.sd2_ms:.1f}ms  |  "
            f"LF/HF: {hrv.lf_hf_ratio:.2f}"
        ),
        xref='paper', yref='paper',
        x=0.5, y=-0.04,
        showarrow=False,
        font=dict(size=10, color='#524e5c'),
        align='center',
    )

    # ── Write HTML ────────────────────────────────────────────────────────────
    fig.write_html(
        str(out_path),
        include_plotlyjs='cdn',
        full_html=True,
        config={'scrollZoom': True, 'displayModeBar': True},
    )

    return out_path
