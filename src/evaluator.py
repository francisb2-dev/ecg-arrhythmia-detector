"""
evaluator.py — Model evaluation: confusion matrix, ROC curves, per-class metrics.

Generates a self-contained interactive HTML evaluation report using Plotly.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import List, Dict


BEAT_COLORS = {
    'N': '#4ade80', 'V': '#f87171', 'A': '#fb923c',
    'L': '#a78bfa', 'R': '#60a5fa', 'F': '#facc15', 'Q': '#94a3b8',
}
BEAT_NAMES = {
    'N': 'Normal', 'V': 'PVC', 'A': 'Atrial Ectopic',
    'L': 'Left BBB', 'R': 'Right BBB', 'F': 'Fusion', 'Q': 'Other',
}


def generate_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: List[str],
    output_path: Path,
    train_metrics: Dict = None,
) -> Path:
    """
    Generate an interactive HTML evaluation report.

    Parameters
    ----------
    y_true  : ground truth labels
    y_pred  : predicted labels
    y_proba : predicted probabilities (n_samples, n_classes)
    classes : class label list matching y_proba columns
    output_path : where to save the HTML
    train_metrics : optional dict with training stats

    Returns
    -------
    Path to generated HTML
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.metrics import (
        confusion_matrix, classification_report,
        roc_curve, auc, accuracy_score, f1_score
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Metrics ───────────────────────────────────────────────────────────────
    present_classes = [c for c in classes if c in np.unique(y_true)]
    cm = confusion_matrix(y_true, y_pred, labels=present_classes)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    overall_acc = accuracy_score(y_true, y_pred)
    overall_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Per-class precision/recall/f1
    report = classification_report(
        y_true, y_pred, labels=present_classes,
        target_names=[BEAT_NAMES.get(c, c) for c in present_classes],
        output_dict=True, zero_division=0
    )

    # ── Figure ────────────────────────────────────────────────────────────────
    n_cls = len(present_classes)

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Confusion Matrix (Normalized)',
            'Per-Class F1 Score',
            'ROC Curves',
            'Precision vs Recall',
            'Class Distribution (True vs Predicted)',
            'Beat Count by Class',
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
        specs=[
            [{}, {}],
            [{}, {}],
            [{}, {}],
        ]
    )

    # ── 1. Confusion matrix heatmap ───────────────────────────────────────────
    class_names = [BEAT_NAMES.get(c, c) for c in present_classes]
    fig.add_trace(go.Heatmap(
        z=cm_norm,
        x=class_names,
        y=class_names,
        colorscale=[
            [0.0, '#08070a'], [0.3, '#16141a'], [0.6, '#6b3a1f'], [1.0, '#b8965a']
        ],
        text=[[f"{cm_norm[i][j]:.2f}\n({cm[i][j]:,})" for j in range(n_cls)]
              for i in range(n_cls)],
        texttemplate="%{text}",
        textfont=dict(size=9),
        showscale=True,
        hovertemplate="True: %{y}<br>Pred: %{x}<br>Rate: %{z:.3f}<extra></extra>",
    ), row=1, col=1)

    # ── 2. Per-class F1 bar chart ─────────────────────────────────────────────
    f1_vals = []
    prec_vals = []
    rec_vals = []
    for c in present_classes:
        name = BEAT_NAMES.get(c, c)
        r = report.get(name, {})
        f1_vals.append(r.get('f1-score', 0))
        prec_vals.append(r.get('precision', 0))
        rec_vals.append(r.get('recall', 0))

    fig.add_trace(go.Bar(
        x=class_names, y=f1_vals,
        name='F1',
        marker_color=[BEAT_COLORS.get(c, '#94a3b8') for c in present_classes],
        hovertemplate='%{x}<br>F1: %{y:.3f}<extra></extra>',
    ), row=1, col=2)

    # ── 3. ROC curves ─────────────────────────────────────────────────────────
    for i, cls in enumerate(present_classes):
        if cls not in np.unique(y_true):
            continue
        binary_true = (y_true == cls).astype(int)
        if y_proba is not None and i < y_proba.shape[1]:
            proba_col = y_proba[:, i]
        else:
            proba_col = (y_pred == cls).astype(float)
        fpr, tpr, _ = roc_curve(binary_true, proba_col)
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f"{BEAT_NAMES.get(cls, cls)} (AUC={roc_auc:.2f})",
            line=dict(color=BEAT_COLORS.get(cls, '#94a3b8'), width=2),
            hovertemplate=f"{BEAT_NAMES.get(cls, cls)}<br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>",
        ), row=2, col=1)

    # Diagonal reference
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        line=dict(color='#2e2b35', dash='dash'),
        showlegend=False, hoverinfo='skip',
    ), row=2, col=1)

    # ── 4. Precision vs Recall scatter ────────────────────────────────────────
    counts = {c: int(np.sum(y_true == c)) for c in present_classes}
    fig.add_trace(go.Scatter(
        x=rec_vals, y=prec_vals,
        mode='markers+text',
        text=class_names,
        textposition='top center',
        marker=dict(
            color=[BEAT_COLORS.get(c, '#94a3b8') for c in present_classes],
            size=[max(8, min(40, counts.get(c, 0) / 500)) for c in present_classes],
            opacity=0.85,
        ),
        hovertemplate='%{text}<br>Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>',
        showlegend=False,
    ), row=2, col=2)

    # ── 5. Class distribution bar (true vs predicted) ─────────────────────────
    true_counts = [int(np.sum(y_true == c)) for c in present_classes]
    pred_counts = [int(np.sum(y_pred == c)) for c in present_classes]

    fig.add_trace(go.Bar(
        x=class_names, y=true_counts, name='True',
        marker_color='#b8965a', opacity=0.8,
        hovertemplate='%{x}<br>True count: %{y:,}<extra></extra>',
    ), row=3, col=1)
    fig.add_trace(go.Bar(
        x=class_names, y=pred_counts, name='Predicted',
        marker_color='#60a5fa', opacity=0.8,
        hovertemplate='%{x}<br>Predicted count: %{y:,}<extra></extra>',
    ), row=3, col=1)

    # ── 6. Beat count horizontal bar ─────────────────────────────────────────
    sorted_idx = np.argsort(true_counts)[::-1]
    fig.add_trace(go.Bar(
        x=[true_counts[i] for i in sorted_idx],
        y=[class_names[i] for i in sorted_idx],
        orientation='h',
        marker_color=[BEAT_COLORS.get(present_classes[i], '#94a3b8') for i in sorted_idx],
        hovertemplate='%{y}: %{x:,} beats<extra></extra>',
        showlegend=False,
    ), row=3, col=2)

    # ── Layout ────────────────────────────────────────────────────────────────
    train_line = ""
    if train_metrics:
        train_line = (
            f"Train accuracy: {train_metrics.get('train_accuracy', 0):.1%}  |  "
            f"Training beats: {train_metrics.get('n_beats', 0):,}  |  "
        )

    fig.update_layout(
        title=dict(
            text=(
                f"<b>ML Beat Classifier — Evaluation Report</b><br>"
                f"<span style='font-size:13px;color:#b8965a'>"
                f"Test accuracy: {overall_acc:.1%}  |  "
                f"Weighted F1: {overall_f1:.3f}  |  "
                f"{train_line}"
                f"Test beats: {len(y_true):,}"
                f"</span>"
            ),
            font=dict(color='#ede9e0', size=15),
            x=0.5,
        ),
        paper_bgcolor='#08070a',
        plot_bgcolor='#0f0e12',
        font=dict(color='#ede9e0', family='-apple-system, sans-serif', size=11),
        barmode='group',
        legend=dict(bgcolor='#16141a', bordercolor='#1e1c24', borderwidth=1, font=dict(size=10)),
        height=1100,
        margin=dict(l=60, r=60, t=100, b=60),
    )

    axis_style = dict(gridcolor='#1e1c24', zerolinecolor='#2e2b35', tickcolor='#524e5c')
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)

    fig.update_xaxes(title_text="Predicted", row=1, col=1)
    fig.update_yaxes(title_text="True", row=1, col=1)
    fig.update_yaxes(title_text="F1 Score", row=1, col=2)
    fig.update_xaxes(title_text="False Positive Rate", row=2, col=1)
    fig.update_yaxes(title_text="True Positive Rate", row=2, col=1)
    fig.update_xaxes(title_text="Recall", row=2, col=2)
    fig.update_yaxes(title_text="Precision", row=2, col=2)
    fig.update_yaxes(title_text="Beat count", row=3, col=1)
    fig.update_xaxes(title_text="Beat count", row=3, col=2)

    fig.write_html(str(output_path), include_plotlyjs='cdn', full_html=True)
    return output_path
