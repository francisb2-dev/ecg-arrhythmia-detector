"""
loader.py — PhysioNet / MIT-BIH data download and loading.

Uses the wfdb library to pull records directly from PhysioNet's
public servers. No authentication required for the MIT-BIH database.
"""

import os
import numpy as np
import wfdb
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


DATA_DIR = Path(__file__).parent.parent / "data"
MITDB = "mitdb"


@dataclass
class ECGRecord:
    """Container for a loaded ECG record and its metadata."""
    record_name: str
    signal: np.ndarray          # Raw signal, shape (n_samples,)
    fs: float                   # Sampling frequency (Hz)
    duration_sec: float         # Total duration in seconds
    n_samples: int
    channel: int                # Which lead was loaded
    lead_name: str
    units: str
    annotations: Optional[object] = None   # wfdb Annotation object


def download_record(record_name: str, data_dir: Path = DATA_DIR) -> Path:
    """
    Download a single MIT-BIH record to local disk if not already present.
    Returns the directory containing the downloaded files.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    record_path = data_dir / str(record_name)

    # Check if already downloaded (wfdb saves .hea + .dat files)
    hea_file = data_dir / f"{record_name}.hea"
    if hea_file.exists():
        return data_dir

    print(f"  Downloading MIT-BIH record {record_name} from PhysioNet...")
    try:
        wfdb.dl_database(
            MITDB,
            dl_dir=str(data_dir),
            records=[str(record_name)],
            annotators=["atr"],
        )
        print(f"  Record {record_name} downloaded successfully.")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download record {record_name}: {e}\n"
            "Check your internet connection and that PhysioNet is reachable."
        )
    return data_dir


def load_record(
    record_name: str,
    data_dir: Path = DATA_DIR,
    channel: int = 0,
    duration_sec: Optional[float] = None,
    start_sec: float = 0.0,
) -> ECGRecord:
    """
    Load an ECG record, downloading it first if necessary.

    Parameters
    ----------
    record_name : str or int
        MIT-BIH record number (e.g. 100, 200, 203).
    data_dir : Path
        Local directory to store/read data.
    channel : int
        Which signal channel to load (0 = MLII for most MIT-BIH records).
    duration_sec : float, optional
        How many seconds to load. None = full record.
    start_sec : float
        Where in the record to start loading.

    Returns
    -------
    ECGRecord
    """
    record_name = str(record_name)
    download_record(record_name, data_dir)

    record_path = str(data_dir / record_name)

    # Load the header to get sampling frequency and number of samples
    header = wfdb.rdheader(record_path)
    fs = header.fs
    total_samples = header.sig_len

    start_sample = int(start_sec * fs)
    if duration_sec is not None:
        end_sample = min(start_sample + int(duration_sec * fs), total_samples)
    else:
        end_sample = total_samples

    sampfrom = start_sample
    sampto = end_sample

    record = wfdb.rdrecord(
        record_path,
        channels=[channel],
        sampfrom=sampfrom,
        sampto=sampto,
        physical=True,
    )

    # Load annotations if available
    annotations = None
    try:
        ann = wfdb.rdann(record_path, "atr", sampfrom=sampfrom, sampto=sampto)
        # Shift annotation samples to be relative to our start
        ann.sample = ann.sample - sampfrom
        # Keep only annotations within range
        valid = (ann.sample >= 0) & (ann.sample < (sampto - sampfrom))
        ann.sample = ann.sample[valid]
        ann.symbol = [ann.symbol[i] for i in range(len(ann.symbol)) if valid[i]]
        annotations = ann
    except Exception:
        pass  # Annotations not available for all records

    signal = record.p_signal[:, 0]
    n_samples = len(signal)
    duration = n_samples / fs

    lead_name = record.sig_name[0] if record.sig_name else f"Channel {channel}"
    units = record.units[0] if record.units else "mV"

    return ECGRecord(
        record_name=record_name,
        signal=signal,
        fs=fs,
        duration_sec=duration,
        n_samples=n_samples,
        channel=channel,
        lead_name=lead_name,
        units=units,
        annotations=annotations,
    )


def get_annotation_beats(ecg: ECGRecord) -> dict:
    """
    Extract ground-truth beat annotations from the record.
    Returns dict with beat sample positions and symbols.
    """
    if ecg.annotations is None:
        return {"samples": np.array([]), "symbols": []}

    ann = ecg.annotations
    beat_symbols = set("NLRBAaJSVrFejnE/fQ|")  # Standard beat annotation symbols

    beat_samples = []
    beat_syms = []
    for i, sym in enumerate(ann.symbol):
        if sym in beat_symbols:
            beat_samples.append(ann.sample[i])
            beat_syms.append(sym)

    return {
        "samples": np.array(beat_samples),
        "symbols": beat_syms,
    }
