"""Convert raw TUEV and Bonn recordings into per-window ``.npz`` files read by :class:`NpzEEGDataset`.

Both pipelines follow the paper: signals are brought to 200 Hz, band-pass filtered between 1 and 45 Hz with a
Hamming-windowed FIR filter, scaled by a fixed factor and cut into 2 s windows (400 samples) with 50 % overlap.
TUEV recordings are additionally clipped and cleaned with ICA, and every channel is windowed independently.
"""

import glob
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import resample

TARGET_FS = 200.0
WINDOW_SEC = 2.0
OVERLAP = 0.5
L_FREQ, H_FREQ = 1.0, 45.0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def bandpass(x: np.ndarray, sfreq: float, l_freq: float = L_FREQ, h_freq: float = H_FREQ) -> np.ndarray:
    """Zero-phase FIR band-pass (Hamming window, ``firwin`` design) along the last axis."""
    import mne

    return mne.filter.filter_data(x.astype(np.float64), sfreq, l_freq, h_freq, method="fir",
                                  fir_window="hamming", fir_design="firwin", verbose="error")


def window_starts(n_samples: int, window: int, overlap: float = OVERLAP) -> range:
    """Start indices of the windows of length ``window`` with fractional ``overlap`` that fit in ``n_samples``."""
    step = max(1, int(round(window * (1.0 - overlap))))
    return range(0, n_samples - window + 1, step)


# ---------------------------------------------------------------------------
# TUH EEG Events (TUEV)
# ---------------------------------------------------------------------------

# Channel numbers in the .rec files index this ACNS TCP bipolar montage (see the TUEV AAREADME.txt).
TCP_MONTAGE = [
    ("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    ("A1", "T3"), ("T3", "C3"), ("C3", "CZ"), ("CZ", "C4"), ("C4", "T4"), ("T4", "A2"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
]
TCP_ELECTRODES = sorted({e for pair in TCP_MONTAGE for e in pair})
EPILEPTIFORM_LABELS = {1, 2, 3}  # SPSW, GPED, PLED; 4-6 (EYEM, ARTF, BCKG) are non-epileptiform


def parse_rec_file(rec_path: str) -> Dict[int, List[Tuple[float, float, int]]]:
    """Parse a TUEV ``.rec`` file (lines ``channel,start_sec,stop_sec,label``) into ``{channel: [(start, stop, label)]}``."""
    events = defaultdict(list)
    with open(rec_path, "r") as f:
        for line in f:
            parts = line.strip().replace(",", " ").split()
            if len(parts) < 4 or parts[0].startswith("#"):
                continue
            try:
                channel, start, stop, label = int(parts[0]), float(parts[1]), float(parts[2]), int(parts[3])
            except ValueError:
                continue
            if 1 <= label <= 6:
                events[channel].append((start, stop, label))
    return dict(events)


def standardize_channel_name(name: str) -> str:
    """``"EEG FP1-REF"`` / ``"EEG FP1-LE"`` -> ``"FP1"``."""
    return name.upper().replace("EEG ", "").replace("-REF", "").replace("-LE", "").strip()


def label_window(t0: float, t1: float, events: Sequence[Tuple[float, float, int]], tau: float = 0.25) -> Optional[int]:
    """Binary label of the window ``[t0, t1)`` from one channel's events.

    1 (epileptic) if its overlap with any SPSW/GPED/PLED event exceeds ``tau`` of the window length,
    0 (non-epileptic) if it overlaps any annotation but not enough epileptiform activity,
    None if it overlaps no annotation at all (the window is dropped).
    """
    length = t1 - t0
    annotated = False
    for start, stop, label in events:
        overlap = min(t1, stop) - max(t0, start)
        if overlap <= 0:
            continue
        annotated = True
        if label in EPILEPTIFORM_LABELS and overlap / length > tau:
            return 1
    return 0 if annotated else None


def clean_tuev_recording(edf_path: str, clip_uv: float = 100.0, ica: bool = True, random_state: int = 56):
    """Load one TUEV EDF and return ``(tcp_signals (22, T) in µV, channel names)`` at 200 Hz.

    Steps: standardise channel names, clip to ±``clip_uv`` µV, resample to 200 Hz, 1-45 Hz FIR band-pass,
    ICA removal of muscle and eye-movement components, TCP bipolar montage, scaling from V to µV.
    """
    import mne

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="error")
    raw.rename_channels({ch: standardize_channel_name(ch) for ch in raw.ch_names})
    missing = [e for e in TCP_ELECTRODES if e not in raw.ch_names]
    if missing:
        raise ValueError(f"missing electrodes {missing}")
    raw.pick(TCP_ELECTRODES)

    raw.apply_function(lambda x: np.clip(x, -clip_uv * 1e-6, clip_uv * 1e-6))
    raw.resample(TARGET_FS, verbose="error")
    raw.filter(L_FREQ, H_FREQ, method="fir", fir_window="hamming", fir_design="firwin", verbose="error")

    if ica:
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), match_case=False, verbose="error")
        model = mne.preprocessing.ICA(method="fastica", max_iter="auto", random_state=random_state, verbose="error")
        model.fit(raw, verbose="error")
        muscle, _ = model.find_bads_muscle(raw, verbose="error")
        eye, _ = model.find_bads_eog(raw, ch_name=["FP1", "FP2"], verbose="error")
        model.exclude = sorted(set(muscle) | set(eye))
        model.apply(raw, verbose="error")

    data = raw.get_data()
    index = {ch: i for i, ch in enumerate(raw.ch_names)}
    tcp = np.stack([data[index[a]] - data[index[b]] for a, b in TCP_MONTAGE]) * 1e6  # V -> µV
    return tcp, [f"{a}-{b}" for a, b in TCP_MONTAGE]


def preprocess_tuev(
    root_dir: str,
    out_dir: str,
    splits=("train", "eval"),
    tau: float = 0.25,
    clip_uv: float = 100.0,
    ica: bool = True,
    limit: Optional[int] = None,
) -> Dict[str, Dict[int, int]]:
    """Window the TUEV corpus.

    Expects ``root_dir/edf/{train,eval}/**/*.edf`` with a matching ``.rec`` next to each EDF (the layout of the
    official v2.0.x release). Every TCP channel is windowed independently and labelled from its own annotations
    (see :func:`label_window`). Writes ``out_dir/<split>/<recording>_<channel>_<window>.npz`` with keys ``x``
    (float32, ``(1, 400)``, µV), ``y`` (0 = non-epileptic, 1 = epileptic), ``channel``, ``beg`` and ``end``
    (sample indices at 200 Hz). ``limit`` processes only the first ``limit`` recordings per split.

    Returns the number of windows written per split and class.
    """
    window = int(WINDOW_SEC * TARGET_FS)
    counts = {}
    for split in splits:
        edf_files = sorted(glob.glob(os.path.join(root_dir, "edf", split, "**", "*.edf"), recursive=True))[:limit]
        split_out = os.path.join(out_dir, split)
        os.makedirs(split_out, exist_ok=True)
        print(f"[{split}] {len(edf_files)} EDF files")

        split_counts = {0: 0, 1: 0}
        for i, edf_path in enumerate(edf_files, 1):
            rec_path = os.path.splitext(edf_path)[0] + ".rec"
            if not os.path.exists(rec_path):
                print(f"[WARN] missing .rec for {edf_path}, skipping")
                continue
            events = parse_rec_file(rec_path)
            try:
                signals, names = clean_tuev_recording(edf_path, clip_uv, ica)
            except Exception as e:  # unreadable EDF or missing electrodes
                print(f"[WARN] {edf_path}: {e}, skipping")
                continue

            rec_id = os.path.splitext(os.path.basename(edf_path))[0]
            for ch, name in enumerate(names):
                ch_events = events.get(ch, [])
                if not ch_events:
                    continue
                for start in window_starts(signals.shape[1], window):
                    label = label_window(start / TARGET_FS, (start + window) / TARGET_FS, ch_events, tau)
                    if label is None:
                        continue
                    np.savez(os.path.join(split_out, f"{rec_id}_{name}_{start // (window // 2):05d}.npz"),
                             x=signals[ch, start:start + window].astype(np.float32)[None, :], y=np.int64(label),
                             channel=name, beg=np.int64(start), end=np.int64(start + window))
                    split_counts[label] += 1
            print(f"[{split}] {i}/{len(edf_files)} {rec_id}: {split_counts}")
        counts[split] = split_counts
        print(f"[{split}] non-epileptic={split_counts[0]} epileptic={split_counts[1]}")
    return counts


# ---------------------------------------------------------------------------
# Bonn University EEG
# ---------------------------------------------------------------------------

BONN_FS = 173.61
# Set letter (first character of each file name) -> raw ab_label.
# Z, O: healthy volunteers (eyes open / closed); N, F: interictal; S: ictal.
BONN_SET_LABELS = {"Z": 0, "O": 0, "N": 1, "F": 1, "S": 2}


def _load_bonn_record(path: str) -> np.ndarray:
    if path.lower().endswith(".npz"):
        return np.load(path)["data"].ravel().astype(np.float64)
    return np.loadtxt(path).ravel()


def find_bonn_records(raw_dir: str) -> Dict[str, str]:
    """Map record id (e.g. ``"Z001"``) to file path for every Bonn ``.txt`` (or ``.npz``) file under ``raw_dir``."""
    records = {}
    for path in glob.glob(os.path.join(raw_dir, "**", "*"), recursive=True):
        stem, ext = os.path.splitext(os.path.basename(path))
        if ext.lower() in (".txt", ".npz") and stem[:1].upper() in BONN_SET_LABELS:
            records[stem.upper()] = path
    return records


def make_bonn_split(record_ids, eval_frac=0.15, test_frac=0.15, seed=42) -> Dict[str, List[str]]:
    """Record-level split, stratified by Bonn set, so no recording contributes windows to two splits."""
    by_set = defaultdict(list)
    for rid in sorted(record_ids):
        by_set[rid[0]].append(rid)
    rng = random.Random(seed)
    split = {"train": [], "eval": [], "test": []}
    for rids in by_set.values():
        rng.shuffle(rids)
        n_eval, n_test = round(len(rids) * eval_frac), round(len(rids) * test_frac)
        split["eval"] += rids[:n_eval]
        split["test"] += rids[n_eval:n_eval + n_test]
        split["train"] += rids[n_eval + n_test:]
    return {k: sorted(v) for k, v in split.items()}


def bonn_windows(signal: np.ndarray, original_fs=BONN_FS, scale: float = 1.0):
    """Resample a Bonn record to 200 Hz, band-pass 1-45 Hz, scale, and cut it into ``(1, 400)`` windows (50 % overlap)."""
    x = resample(signal, int(signal.shape[0] * TARGET_FS / original_fs))
    x = bandpass(x, TARGET_FS) * scale
    window = int(WINDOW_SEC * TARGET_FS)
    for start in window_starts(x.shape[0], window):
        yield x[start:start + window].reshape(1, -1)


def preprocess_bonn(
    raw_dir: str,
    out_dir: str,
    split_file: Optional[str] = None,
    seed: int = 42,
    scale: float = 1.0,
) -> Dict[str, int]:
    """Window the Bonn dataset into ``out_dir/{train,eval,test}/<record>_segXXXX.npz``.

    Recordings are assigned to splits *before* windowing. If ``split_file`` (JSON with ``train``/``eval``/
    ``test`` lists of record ids) is given it is used as-is, otherwise a stratified 70/15/15 split is drawn
    with ``seed`` and written to ``out_dir/split.json``. Bonn signals are already in µV, so ``scale`` defaults
    to 1. Each window file has keys ``data`` (``(1, 400)`` at 200 Hz) and ``ab_label``.
    """
    records = find_bonn_records(raw_dir)
    if not records:
        raise FileNotFoundError(f"No Bonn recordings found under {raw_dir}")

    if split_file:
        with open(split_file) as f:
            split = json.load(f)
        missing = [rid for ids in split.values() for rid in ids if rid not in records]
        if missing:
            raise FileNotFoundError(f"{len(missing)} records from the split file are missing, e.g. {missing[:5]}")
    else:
        split = make_bonn_split(records, seed=seed)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "split.json"), "w") as f:
            json.dump(split, f, indent=1)

    counts = {}
    for split_name, rids in split.items():
        split_out = os.path.join(out_dir, split_name)
        os.makedirs(split_out, exist_ok=True)
        n = 0
        for rid in rids:
            label = np.int64(BONN_SET_LABELS[rid[0]])
            for i, window in enumerate(bonn_windows(_load_bonn_record(records[rid]), BONN_FS, scale)):
                np.savez(os.path.join(split_out, f"{rid}_seg{i:04d}.npz"), data=window, ab_label=label)
                n += 1
        counts[split_name] = n
        print(f"[{split_name}] {len(rids)} records -> {n} windows")
    return counts
