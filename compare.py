#!/usr/bin/env python3
"""
compare.py — Batch ECG analysis and multi-record comparison tool.

Usage:
    python compare.py --records 100 208 203
    python compare.py --records 100 208 203 --output reports/ --csv
"""

import sys
import argparse
from pathlib import Path

import numpy as np


def run_analysis(record_id: str, data_dir: Path):
    """Run full pipeline on one record. Returns dict of results."""
    import wfdb
    from src.loader import load_mitbih_record
    from src.preprocessor import preprocess_signal
    from src.detector import detect_qrs
    from src.features import extract_hrv_features
    from src.classifier import classify_rhythm
    from src.ml_classifier import MLBeatClassifier

    print(f"  Analyzing record {record_id}...", end='', flush=True)

    try:
        signal, fs, record_id_out = load_mitbih_record(record_id, str(data_dir))
    except Exception as e:
        print(f" ERROR: {e}")
        return None

    filtered = preprocess_signal(signal, fs)
    detection = detect_qrs(filtered, fs)
    hrv = extract_hrv_features(detection.rr_intervals_sec, fs)
    classification = classify_rhythm(hrv, detection.rr_intervals_sec)

    ml_result = None
    if MLBeatClassifier.is_available():
        try:
            clf = MLBeatClassifier.load()
            ml_result = clf.predict(detection.r_peaks, fs)
        except Exception:
            pass

    print(f" {classification.primary_rhythm.name} ({classification.primary_rhythm.confidence:.0f}%)")

    return {
        'record_id': record_id,
        'signal': signal,
        'fs': fs,
        'detection': detection,
        'hrv': hrv,
        'classification': classification,
        'ml_result': ml_result,
    }


def build_comparison_html(results: list, output_path: Path) -> Path:
    """Generate interactive comparison HTML with Plotly."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    records = [r for r in results if r is not None]
    if not records:
        print("  No valid records to compare.")
        return None

    colors = ['#b8965a', '#60a5fa', '#4ade80', '#f87171', '#a78bfa', '#fb923c']

    # ── Subplot: RR tachogram overlay + HR bar chart ──────────────────────────
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "RR Interval Tachogram (Overlay)",
            "Heart Rate Comparison",
            "HRV Metrics Comparison",
            "Rhythm Classification",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    metric_labels = ['SDNN (ms)', 'RMSSD (ms)', 'pNN50 (%)', 'CV × 100']
    record_ids = [r['record_id'] for r in records]

    for i, res in enumerate(records):
        color = colors[i % len(colors)]
        hrv = res['hrv']
        cls = res['classification']
        rr_ms = res['detection'].rr_intervals_sec * 1000.0
        beat_nums = np.arange(1, len(rr_ms) + 1)

        # RR tachogram
        fig.add_trace(go.Scatter(
            x=beat_nums, y=rr_ms,
            mode='lines',
            name=f"Record {res['record_id']}",
            line=dict(color=color, width=1.5),
            hovertemplate=f"Record {res['record_id']}<br>Beat %{{x}}<br>RR: %{{y:.1f}}ms<extra></extra>",
        ), row=1, col=1)

        # HR bar
        fig.add_trace(go.Bar(
            x=[f"R{res['record_id']}"],
            y=[hrv.mean_hr_bpm],
            name=f"HR {res['record_id']}",
            marker_color=color,
            showlegend=False,
            hovertemplate=f"Record {res['record_id']}<br>HR: %{{y:.1f}} BPM<extra></extra>",
            error_y=dict(
                type='data',
                array=[hrv.std_rr_ms / hrv.mean_rr_ms * hrv.mean_hr_bpm],
                visible=True,
                color='#524e5c',
            ),
        ), row=1, col=2)

    # HRV grouped bar chart
    metric_values = {m: [] for m in metric_labels}
    for res in records:
        hrv = res['hrv']
        metric_values['SDNN (ms)'].append(hrv.std_rr_ms)
        metric_values['RMSSD (ms)'].append(hrv.rmssd_ms)
        metric_values['pNN50 (%)'].append(hrv.pnn50_pct)
        metric_values['CV × 100'].append(hrv.cv_rr * 100)

    for j, (metric, values) in enumerate(metric_values.items()):
        fig.add_trace(go.Bar(
            x=record_ids,
            y=values,
            name=metric,
            marker_color=colors[j % len(colors)],
            hovertemplate=f"{metric}: %{{y:.1f}}<extra></extra>",
        ), row=2, col=1)

    # Rhythm summary table
    table_records = [f"Record {r['record_id']}" for r in records]
    table_rhythms = [r['classification'].primary_rhythm.name for r in records]
    table_confs = [f"{r['classification'].primary_rhythm.confidence:.0f}%" for r in records]
    table_hrs = [f"{r['hrv'].mean_hr_bpm:.1f}" for r in records]
    table_sdn = [f"{r['hrv'].std_rr_ms:.1f}" for r in records]
    table_rmssd = [f"{r['hrv'].rmssd_ms:.1f}" for r in records]

    fig.add_trace(go.Table(
        header=dict(
            values=['<b>Record</b>', '<b>Rhythm</b>', '<b>Conf</b>', '<b>HR</b>', '<b>SDNN</b>', '<b>RMSSD</b>'],
            fill_color='#16141a',
            font=dict(color='#b8965a', size=11),
            line_color='#1e1c24',
            align='left',
        ),
        cells=dict(
            values=[table_records, table_rhythms, table_confs, table_hrs, table_sdn, table_rmssd],
            fill_color='#0f0e12',
            font=dict(color='#ede9e0', size=10),
            line_color='#1e1c24',
            align='left',
        ),
    ), row=2, col=2)

    fig.update_layout(
        title=dict(
            text=f"<b>ECG Batch Comparison — Records {', '.join(record_ids)}</b>",
            font=dict(color='#ede9e0', size=15),
            x=0.5,
        ),
        paper_bgcolor='#08070a',
        plot_bgcolor='#0f0e12',
        font=dict(color='#ede9e0', family='-apple-system, sans-serif', size=11),
        barmode='group',
        legend=dict(bgcolor='#16141a', bordercolor='#1e1c24', borderwidth=1),
        height=750,
        margin=dict(l=60, r=60, t=80, b=60),
    )

    axis_style = dict(gridcolor='#1e1c24', zerolinecolor='#2e2b35', tickcolor='#524e5c')
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    fig.update_xaxes(title_text="Beat number", row=1, col=1)
    fig.update_yaxes(title_text="RR interval (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Heart rate (BPM)", row=1, col=2)
    fig.update_yaxes(title_text="Value", row=2, col=1)

    fig.write_html(str(output_path), include_plotlyjs='cdn', full_html=True)
    return output_path


def export_csv(results: list, csv_path: Path):
    """Export comparison metrics as CSV."""
    import csv
    records = [r for r in results if r is not None]
    fields = ['record_id', 'rhythm', 'confidence', 'mean_hr_bpm', 'sdnn_ms',
              'rmssd_ms', 'pnn50_pct', 'cv_rr', 'n_beats', 'sd1_ms', 'sd2_ms', 'lf_hf_ratio']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for res in records:
            hrv = res['hrv']
            cls = res['classification']
            writer.writerow({
                'record_id': res['record_id'],
                'rhythm': cls.primary_rhythm.name,
                'confidence': round(cls.primary_rhythm.confidence, 1),
                'mean_hr_bpm': round(hrv.mean_hr_bpm, 1),
                'sdnn_ms': round(hrv.std_rr_ms, 1),
                'rmssd_ms': round(hrv.rmssd_ms, 1),
                'pnn50_pct': round(hrv.pnn50_pct, 1),
                'cv_rr': round(hrv.cv_rr, 4),
                'n_beats': hrv.n_beats,
                'sd1_ms': round(hrv.sd1_ms, 1),
                'sd2_ms': round(hrv.sd2_ms, 1),
                'lf_hf_ratio': round(hrv.lf_hf_ratio, 3),
            })
    print(f"  CSV exported → {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Batch ECG comparison tool')
    parser.add_argument('--records', nargs='+', required=True, help='MIT-BIH record IDs')
    parser.add_argument('--output', default='reports', help='Output directory')
    parser.add_argument('--csv', action='store_true', help='Also export CSV')
    args = parser.parse_args()

    data_dir = Path(__file__).parent / 'data'
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║     ECG Batch Comparison Tool                ║")
    print("  ╚══════════════════════════════════════════════╝")
    print(f"\n  Records: {', '.join(args.records)}\n")

    # Download missing records
    import wfdb
    for rec_id in args.records:
        hea = data_dir / f'{rec_id}.hea'
        if not hea.exists():
            print(f"  Downloading record {rec_id}...")
            try:
                wfdb.dl_database('mitdb', dl_dir=str(data_dir), records=[rec_id])
            except Exception as e:
                print(f"  Warning: {e}")

    # Analyze all records
    results = []
    for rec_id in args.records:
        res = run_analysis(rec_id, data_dir)
        results.append(res)

    # Generate comparison report
    comp_path = output_dir / f"comparison_{'_'.join(args.records)}.html"
    print(f"\n  Generating comparison report...")
    out = build_comparison_html(results, comp_path)
    if out:
        print(f"  Comparison saved → {out}")

    # CSV export
    if args.csv:
        csv_path = output_dir / f"comparison_{'_'.join(args.records)}.csv"
        export_csv(results, csv_path)

    # Summary table
    print()
    print(f"  {'Record':<10} {'Rhythm':<30} {'HR':>7} {'SDNN':>8} {'RMSSD':>8} {'Conf':>6}")
    print(f"  {'──────':<10} {'──────':<30} {'──':>7} {'────':>8} {'─────':>8} {'────':>6}")
    for res in results:
        if res is None:
            continue
        h = res['hrv']
        c = res['classification'].primary_rhythm
        print(f"  {res['record_id']:<10} {c.name:<30} {h.mean_hr_bpm:>6.1f} {h.std_rr_ms:>7.1f}ms {h.rmssd_ms:>7.1f}ms {c.confidence:>5.0f}%")
    print()

    import webbrowser
    if out:
        webbrowser.open(f'file://{out.resolve()}')


if __name__ == '__main__':
    main()
