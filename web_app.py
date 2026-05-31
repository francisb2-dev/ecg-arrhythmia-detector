#!/usr/bin/env python3
"""
web_app.py — Web interface for the ECG Arrhythmia Detection System.

Provides a clean browser-based UI to upload ECG files or analyze
MIT-BIH records directly. Returns interactive analysis reports.

Usage:
    python web_app.py
    Open: http://localhost:5001
"""

import os
import io
import json
import tempfile
import traceback
from pathlib import Path

from flask import Flask, request, jsonify, send_file, render_template_string

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

DATA_DIR = Path(__file__).parent / 'data'
REPORTS_DIR = Path(__file__).parent / 'reports'
REPORTS_DIR.mkdir(exist_ok=True)

# ── HTML Template ─────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>ECG Arrhythmia Detector</title>
  <style>
    :root {
      --bg: #08070a; --surface: #0f0e12; --surface-el: #16141a;
      --border: #1e1c24; --accent: #b8965a; --accent-dim: rgba(184,150,90,0.2);
      --text: #ede9e0; --muted: #524e5c; --subtle: #2e2b35;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg); color: var(--text);
      font-family: -apple-system, 'SF Pro Display', sans-serif;
      font-weight: 300; min-height: 100vh;
      display: flex; flex-direction: column; align-items: center;
    }
    header {
      width: 100%; padding: 20px 40px;
      border-bottom: 1px solid rgba(184,150,90,0.2);
      display: flex; align-items: center; gap: 12px;
      background: var(--bg);
    }
    .logo {
      width: 32px; height: 32px; border: 1px solid rgba(184,150,90,0.5);
      border-radius: 8px; display: flex; align-items: center;
      justify-content: center; color: var(--accent); font-size: 14px;
    }
    header h1 { font-size: 13px; font-weight: 300; letter-spacing: 0.18em; text-transform: uppercase; }
    header p { margin-left: auto; font-size: 11px; color: var(--muted); letter-spacing: 0.06em; text-transform: uppercase; }

    main { width: 100%; max-width: 860px; padding: 48px 24px; flex: 1; }

    h2 { font-size: 22px; font-weight: 300; color: var(--text); margin-bottom: 8px; letter-spacing: -0.02em; }
    .subtitle { font-size: 13px; color: var(--muted); margin-bottom: 40px; }

    .cards { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 40px; }
    .card {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 8px; padding: 28px;
    }
    .card h3 { font-size: 13px; font-weight: 500; color: var(--accent); letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 12px; }
    .card p { font-size: 13px; color: var(--muted); line-height: 1.6; margin-bottom: 20px; }

    label { display: block; font-size: 11px; color: var(--muted); letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 8px; }
    input[type=text], input[type=file], select {
      width: 100%; background: var(--surface-el); border: 1px solid var(--border);
      border-radius: 4px; padding: 10px 12px; color: var(--text);
      font-size: 13px; font-family: inherit; outline: none;
      transition: border-color 0.2s;
    }
    input[type=text]:focus, select:focus { border-color: rgba(184,150,90,0.5); }
    input[type=file] { cursor: pointer; }
    input[type=file]::file-selector-button {
      background: var(--surface-el); border: 1px solid var(--border);
      color: var(--muted); padding: 6px 12px; border-radius: 3px;
      cursor: pointer; font-size: 11px; margin-right: 10px;
    }

    .btn {
      display: inline-flex; align-items: center; justify-content: center;
      background: var(--accent); color: #08070a; border: none;
      border-radius: 4px; padding: 10px 20px; font-size: 12px;
      font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase;
      cursor: pointer; transition: opacity 0.2s; width: 100%; margin-top: 16px;
    }
    .btn:hover { opacity: 0.85; }
    .btn:disabled { opacity: 0.4; cursor: not-allowed; }
    .btn-secondary {
      background: transparent; color: var(--accent);
      border: 1px solid rgba(184,150,90,0.4);
    }
    .btn-secondary:hover { background: var(--accent-dim); }

    #status {
      margin-top: 32px; padding: 16px 20px;
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 6px; font-size: 13px; color: var(--muted);
      display: none;
    }
    #status.visible { display: block; }
    #status.error { border-color: rgba(248,113,113,0.4); color: #f87171; }
    #status.success { border-color: rgba(74,222,128,0.4); color: #4ade80; }

    .progress {
      height: 2px; background: var(--border); border-radius: 1px;
      margin-top: 12px; overflow: hidden; display: none;
    }
    .progress.visible { display: block; }
    .progress-bar {
      height: 100%; background: var(--accent);
      animation: indeterminate 1.5s ease-in-out infinite;
      width: 40%;
    }
    @keyframes indeterminate {
      0% { transform: translateX(-100%); }
      100% { transform: translateX(350%); }
    }

    .results { margin-top: 32px; }
    .result-item {
      display: flex; align-items: center; justify-content: space-between;
      padding: 14px 20px; background: var(--surface); border: 1px solid var(--border);
      border-radius: 6px; margin-bottom: 8px;
    }
    .result-item .rhythm { font-size: 14px; color: var(--text); }
    .result-item .meta { font-size: 11px; color: var(--muted); margin-top: 4px; }
    .result-item a {
      font-size: 11px; color: var(--accent); text-decoration: none;
      letter-spacing: 0.06em; text-transform: uppercase;
      padding: 6px 12px; border: 1px solid rgba(184,150,90,0.4);
      border-radius: 3px; transition: background 0.2s;
    }
    .result-item a:hover { background: var(--accent-dim); }

    .quick-records { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
    .rec-chip {
      font-size: 11px; padding: 5px 12px; border: 1px solid var(--border);
      border-radius: 20px; color: var(--muted); cursor: pointer;
      transition: all 0.15s; background: var(--surface);
    }
    .rec-chip:hover { border-color: rgba(184,150,90,0.4); color: var(--accent); }

    footer {
      width: 100%; padding: 20px 40px; border-top: 1px solid var(--border);
      text-align: center; font-size: 11px; color: var(--subtle);
      letter-spacing: 0.06em;
    }
  </style>
</head>
<body>
<header>
  <div class="logo">E</div>
  <h1>ECG Arrhythmia Detector</h1>
  <p>Braden Francis · Wentworth Institute of Technology</p>
</header>

<main>
  <h2>Cardiac Rhythm Analysis</h2>
  <p class="subtitle">Pan-Tompkins QRS detection · HRV analysis · ML beat classification · Interactive reports</p>

  <div class="cards">
    <!-- MIT-BIH Record Analysis -->
    <div class="card">
      <h3>MIT-BIH Record</h3>
      <p>Analyze any record from the MIT-BIH Arrhythmia Database (PhysioNet). Data downloads automatically.</p>
      <label>Record number</label>
      <input type="text" id="record-input" placeholder="e.g. 100, 208, 203" value="100"/>
      <div class="quick-records">
        <span class="rec-chip" onclick="setRecord('100')">100 — NSR</span>
        <span class="rec-chip" onclick="setRecord('208')">208 — Complex</span>
        <span class="rec-chip" onclick="setRecord('203')">203 — AF</span>
        <span class="rec-chip" onclick="setRecord('201')">201 — Mixed</span>
        <span class="rec-chip" onclick="setRecord('119')">119 — PVCs</span>
      </div>
      <label style="margin-top:16px">Duration (seconds)</label>
      <select id="duration-select">
        <option value="30">30 seconds</option>
        <option value="60" selected>60 seconds</option>
        <option value="120">2 minutes</option>
        <option value="300">5 minutes</option>
        <option value="0">Full record</option>
      </select>
      <button class="btn" onclick="analyzeRecord()">Analyze Record</button>
    </div>

    <!-- Upload -->
    <div class="card">
      <h3>Upload ECG File</h3>
      <p>Upload a CSV file with ECG signal data. First column = time (s), second column = voltage (mV). Sampling frequency auto-detected.</p>
      <label>ECG file (.csv)</label>
      <input type="file" id="file-input" accept=".csv,.txt"/>
      <label style="margin-top:16px">Sampling frequency (Hz)</label>
      <input type="text" id="fs-input" placeholder="e.g. 360, 500, 1000" value="360"/>
      <button class="btn btn-secondary" onclick="analyzeFile()">Upload & Analyze</button>
    </div>
  </div>

  <div id="progress" class="progress"><div class="progress-bar"></div></div>
  <div id="status"></div>
  <div class="results" id="results"></div>
</main>

<footer>ECG Arrhythmia Detection System · MIT-BIH Database · Pan-Tompkins QRS · Random Forest + 1D CNN</footer>

<script>
  function setRecord(r) { document.getElementById('record-input').value = r; }

  function showStatus(msg, type='info') {
    const el = document.getElementById('status');
    el.textContent = msg;
    el.className = 'visible ' + type;
  }

  function setLoading(on) {
    document.getElementById('progress').className = on ? 'progress visible' : 'progress';
    document.querySelectorAll('.btn').forEach(b => b.disabled = on);
  }

  async function analyzeRecord() {
    const record = document.getElementById('record-input').value.trim();
    const duration = document.getElementById('duration-select').value;
    if (!record) { showStatus('Enter a record number.', 'error'); return; }

    setLoading(true);
    showStatus(`Downloading and analyzing record ${record}...`);

    try {
      const res = await fetch('/analyze/record', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({record, duration: parseInt(duration)}),
      });
      const data = await res.json();
      if (!res.ok) { showStatus(data.error || 'Analysis failed.', 'error'); return; }
      showStatus(`Analysis complete — ${data.rhythm} (${data.confidence}% confidence)`, 'success');
      addResult(data);
    } catch(e) {
      showStatus('Server error: ' + e.message, 'error');
    } finally {
      setLoading(false);
    }
  }

  async function analyzeFile() {
    const file = document.getElementById('file-input').files[0];
    const fs = document.getElementById('fs-input').value.trim();
    if (!file) { showStatus('Select a file first.', 'error'); return; }

    setLoading(true);
    showStatus(`Uploading ${file.name}...`);

    const form = new FormData();
    form.append('file', file);
    form.append('fs', fs || '360');

    try {
      const res = await fetch('/analyze/upload', { method: 'POST', body: form });
      const data = await res.json();
      if (!res.ok) { showStatus(data.error || 'Upload failed.', 'error'); return; }
      showStatus(`Analysis complete — ${data.rhythm} (${data.confidence}% confidence)`, 'success');
      addResult(data);
    } catch(e) {
      showStatus('Server error: ' + e.message, 'error');
    } finally {
      setLoading(false);
    }
  }

  function addResult(data) {
    const el = document.getElementById('results');
    const div = document.createElement('div');
    div.className = 'result-item';
    div.innerHTML = `
      <div>
        <div class="rhythm">${data.rhythm} <span style="color:#b8965a">${data.confidence}%</span></div>
        <div class="meta">
          Record ${data.record} · HR ${data.hr} BPM · SDNN ${data.sdnn}ms · RMSSD ${data.rmssd}ms · ${data.n_beats} beats
          ${data.pvc_burden ? ' · PVC burden ' + data.pvc_burden + '%' : ''}
        </div>
      </div>
      <a href="/report/${data.report_id}" target="_blank">Open Report →</a>
    `;
    el.prepend(div);
  }

  document.getElementById('record-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') analyzeRecord();
  });
</script>
</body>
</html>
"""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/analyze/record', methods=['POST'])
def analyze_record():
    data = request.get_json()
    record_id = str(data.get('record', '')).strip()
    duration = int(data.get('duration', 60))

    if not record_id:
        return jsonify({'error': 'Record ID required'}), 400

    try:
        result = _run_pipeline(record_id=record_id, duration=duration)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/analyze/upload', methods=['POST'])
def analyze_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    fs = float(request.form.get('fs', 360))

    try:
        import numpy as np
        content = file.read().decode('utf-8')
        lines = [l.strip() for l in content.split('\n') if l.strip() and not l.startswith('#')]
        rows = [l.split(',') for l in lines]

        if len(rows[0]) >= 2:
            signal = np.array([float(r[1]) for r in rows if len(r) >= 2])
        else:
            signal = np.array([float(r[0]) for r in rows])

        record_id = f"upload_{file.filename.replace('.', '_')}"
        result = _run_pipeline(signal=signal, fs=fs, record_id=record_id)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/report/<report_id>')
def get_report(report_id):
    path = REPORTS_DIR / f'ecg_interactive_{report_id}.html'
    if not path.exists():
        path = REPORTS_DIR / f'ecg_report_{report_id}.html'
    if not path.exists():
        return 'Report not found', 404
    return send_file(str(path))


# ── Pipeline ──────────────────────────────────────────────────────────────────

def _run_pipeline(
    record_id: str,
    duration: int = 60,
    signal=None,
    fs: float = 360.0,
):
    """Run the full analysis pipeline and return a result dict."""
    import numpy as np
    from src.loader import load_record
    from src.preprocessor import preprocess, estimate_signal_quality
    from src.detector import detect_qrs
    from src.features import extract_features
    from src.classifier import classify_rhythm
    from src.dashboard import generate_interactive_report
    from src.ml_classifier import MLBeatClassifier
    from src.cnn_classifier import CNNBeatClassifier

    # Load data
    if signal is None:
        ecg = load_record(record_name=record_id, duration_sec=duration if duration > 0 else None)
        raw_signal = ecg.signal
        fs = ecg.fs
    else:
        raw_signal = signal

    # Pipeline
    filtered = preprocess(raw_signal, fs)
    quality = estimate_signal_quality(raw_signal, fs)
    detection = detect_qrs(filtered.filtered, fs)
    hrv = extract_features(
        r_peaks=detection.r_peaks,
        rr_intervals_sec=detection.rr_intervals_sec,
        r_peak_amplitudes=detection.r_peak_amplitudes,
        filtered_signal=filtered.filtered,
        fs=fs,
        duration_sec=len(raw_signal) / fs,
    )
    classification = classify_rhythm(hrv, detection.rr_intervals_sec)

    # ML classification — prefer CNN if available
    ml_result = None
    if CNNBeatClassifier.is_available():
        try:
            clf = CNNBeatClassifier.load()
            ml_result = clf.predict(detection.r_peaks, fs, filtered.filtered)
        except Exception:
            pass
    if ml_result is None and MLBeatClassifier.is_available():
        try:
            clf = MLBeatClassifier.load()
            ml_result = clf.predict(detection.r_peaks, fs, filtered.filtered)
        except Exception:
            pass

    # Generate interactive report
    report_path = generate_interactive_report(
        record_id=record_id,
        signal=filtered.filtered,
        fs=fs,
        r_peaks=detection.r_peaks,
        rr_sec=detection.rr_intervals_sec,
        features=hrv,
        classification=classification,
        output_dir=REPORTS_DIR,
        ml_result=ml_result,
        duration_sec=len(raw_signal) / fs,
    )

    primary = classification.primary_rhythm
    return {
        'record': record_id,
        'report_id': record_id,
        'rhythm': primary.name,
        'confidence': round(primary.confidence, 1),
        'hr': round(hrv.mean_hr_bpm, 1),
        'sdnn': round(hrv.std_rr_ms, 1),
        'rmssd': round(hrv.rmssd_ms, 1),
        'n_beats': hrv.n_beats,
        'quality': quality['quality_label'],
        'pvc_burden': round(ml_result.pvc_burden_pct, 1) if ml_result else None,
        'ml_model': ml_result.model_version if ml_result else None,
    }


if __name__ == '__main__':
    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║     ECG Arrhythmia Detector — Web Interface          ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print()
    print("  Open: http://localhost:5001")
    print()
    app.run(host='0.0.0.0', port=5001, debug=False)
