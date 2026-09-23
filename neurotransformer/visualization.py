"""Plots for inspecting Neurotransformer predictions.

The recording-level plots (topomaps, EEG timelines) need windows that carry recording metadata, i.e. the
NMT-events ``.npz`` files: name ``<recording>_<...>.npz`` and keys ``data``, ``ab_label``, ``channel``,
``beg``, ``end`` (sample indices at 200 Hz). ``matplotlib`` and ``mne`` are required.
"""

import glob
import os
from collections import defaultdict

import numpy as np
import torch

# 10-20 electrode positions on a unit head (x: left/right, y: back/front).
ELECTRODE_POS = {
    "Fz": (0.0, 0.45), "Cz": (0.0, 0.0), "Pz": (0.0, -0.45),
    "F3": (-0.35, 0.45), "C3": (-0.45, 0.0), "P3": (-0.35, -0.45),
    "F4": (0.35, 0.45), "C4": (0.45, 0.0), "P4": (0.35, -0.45),
    "Fp1": (-0.30, 0.85), "Fp2": (0.30, 0.85),
    "F7": (-0.75, 0.55), "T3": (-0.85, 0.0), "T5": (-0.75, -0.55),
    "F8": (0.75, 0.55), "T4": (0.85, 0.0), "T6": (0.75, -0.55),
    "O1": (-0.30, -0.85), "O2": (0.30, -0.85),
}
# Upper-case channel name (old and new 10-20 nomenclature) -> key in ELECTRODE_POS.
CHANNEL_ALIASES = {name.upper(): name for name in ELECTRODE_POS}
CHANNEL_ALIASES.update({"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"})

CLINICAL_ORDER = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
                  "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "PZ", "CZ"]

# Colours for ground-truth boxes and prediction highlights, by class index.
GT_COLORS = {1: "red", 2: "blue"}
PRED_COLORS = {1: "yellow", 2: "cyan"}


def plot_confusion_matrix(cm, class_names, ax=None, title=None, cmap="Blues"):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap=cmap)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    if title:
        ax.set_title(title)
    return ax


def plot_roc_curves(labels, probs, class_names, ax=None, title="ROC"):
    """One-vs-rest ROC curve per class."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))
    classes = range(len(class_names)) if len(class_names) > 2 else [1]
    for c in classes:
        fpr, tpr, _ = roc_curve(labels == c, probs[:, c])
        ax.plot(fpr, tpr, label=f"{class_names[c]} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return ax


def _load_recording_windows(npz_root, recording_id):
    files = sorted(glob.glob(os.path.join(npz_root, f"{recording_id}_*.npz")))
    if not files:
        raise FileNotFoundError(f"No windows for recording {recording_id} under {npz_root}")
    windows = []
    for path in files:
        item = np.load(path, allow_pickle=True)
        windows.append({
            "data": np.asarray(item["data"], dtype=np.float32).reshape(1, -1),
            "raw_label": int(np.asarray(item["ab_label"]).item()),
            "channel": str(np.asarray(item["channel"]).item()).upper(),
            "beg": int(item["beg"]),
            "end": int(item["end"]),
        })
    return windows


@torch.no_grad()
def predict_recording(model, npz_root, recording_id, label_map, device="cpu", batch_size=128, sfreq=200):
    """Predict every window of one recording.

    Returns ``{channel: [{"t_start", "t_end", "true", "pred"}, ...]}`` with class indices.
    """
    model.eval()
    windows = _load_recording_windows(npz_root, recording_id)
    timeline = defaultdict(list)
    for i in range(0, len(windows), batch_size):
        batch = windows[i:i + batch_size]
        x = torch.from_numpy(np.stack([w["data"] for w in batch])).to(device)
        preds = model(x).argmax(dim=1).cpu().numpy()
        for w, pred in zip(batch, preds):
            timeline[w["channel"]].append({
                "t_start": w["beg"] / sfreq,
                "t_end": w["end"] / sfreq,
                "true": label_map[w["raw_label"]],
                "pred": int(pred),
            })
    return dict(timeline)


def abnormal_counts(timeline, key):
    """Per-channel ``{"total", "abnormal"}`` window counts, where abnormal means class != 0 in ``key``."""
    return {ch: {"total": len(ws), "abnormal": sum(w[key] != 0 for w in ws)} for ch, ws in timeline.items()}


def plot_channel_topomap(channel_stats, ax=None, title="", vmax=None, use_ratio=False):
    """Scalp heat map of the number (or ratio) of abnormal windows per electrode."""
    import matplotlib.pyplot as plt
    import mne
    from matplotlib.patches import Circle, Polygon, Wedge

    stats_by_pos = {CHANNEL_ALIASES.get(ch.upper()): s for ch, s in channel_stats.items()}
    names = list(ELECTRODE_POS)
    pos = np.array([ELECTRODE_POS[n] for n in names])
    values = []
    for n in names:
        s = stats_by_pos.get(n, {"total": 0, "abnormal": 0})
        if use_ratio:
            values.append(s["abnormal"] / s["total"] if s["total"] else 0.0)
        else:
            values.append(float(s["abnormal"]))

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    im, _ = mne.viz.plot_topomap(
        np.array(values), pos, axes=ax, show=False, cmap="RdBu_r", contours=0, extrapolate="head",
        sphere=(0, 0, 0, 1.0), border="mean", outlines=None, sensors=False, vlim=(0, vmax),
    )
    # Mask everything outside the head, then draw the head outline, nose and electrodes.
    ax.add_patch(Wedge((0, 0), r=2.0, theta1=0, theta2=360, width=1.0, color="white", zorder=2))
    ax.add_patch(Circle((0, 0), radius=1.0, color="black", linewidth=2, fill=False, zorder=3))
    ax.add_patch(Polygon([(-0.1, 0.99), (0, 1.1), (0.1, 0.99)], color="black", fill=False, linewidth=2, zorder=3))
    ax.scatter(pos[:, 0], pos[:, 1], s=150, edgecolors="black", facecolors="none", linewidth=1, zorder=4)
    for (x, y), n in zip(pos, names):
        ax.text(x + 0.08, y, n, fontsize=10, ha="left", va="center", zorder=5)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, fontweight="bold", pad=20)
    return im


def plot_topomap_comparison(timeline, recording_id=""):
    """Side-by-side ground-truth vs. predicted abnormal-window topomaps on a shared colour scale."""
    import matplotlib.pyplot as plt

    true_stats, pred_stats = abnormal_counts(timeline, "true"), abnormal_counts(timeline, "pred")
    vmax = max([s["abnormal"] for s in list(true_stats.values()) + list(pred_stats.values())] + [1])
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    plot_channel_topomap(true_stats, ax=axes[0], title=f"Ground truth\n({recording_id})", vmax=vmax)
    im = plot_channel_topomap(pred_stats, ax=axes[1], title=f"Model prediction\n({recording_id})", vmax=vmax)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.04).set_label("# abnormal windows")
    fig.suptitle(f"Abnormal windows per electrode: recording {recording_id}", fontsize=16)
    return fig


def _overlapping(timeline, t0, t1):
    for windows in timeline.values():
        for w in windows:
            if w["t_end"] > t0 and w["t_start"] < t1:
                yield w


def find_segment_perfect_recall(timeline, duration=10):
    """Start second of the ``duration``-s segment with the most abnormal windows and no missed ones."""
    starts = [w["t_start"] for ws in timeline.values() for w in ws]
    best, best_count = None, 0
    for t in range(int(min(starts)), int(max(starts)) - duration):
        ws = [w for w in _overlapping(timeline, t, t + duration) if w["true"] != 0]
        if ws and all(w["pred"] != 0 for w in ws) and len(ws) > best_count:
            best, best_count = t, len(ws)
    return best


def find_segment_with_misses(timeline, duration=10):
    """Start second of the ``duration``-s segment with the most missed abnormal windows (false negatives)."""
    starts = [w["t_start"] for ws in timeline.values() for w in ws]
    best, best_count = None, 0
    for t in range(int(min(starts)), int(max(starts)) - duration):
        misses = sum(w["true"] != 0 and w["pred"] == 0 for w in _overlapping(timeline, t, t + duration))
        if misses > best_count:
            best, best_count = t, misses
    return best


def plot_eeg_with_predictions(edf_path, timeline, start_times, titles, duration=10, y_offset=200, scale=50):
    """Plot the multichannel EEG around each start time with ground truth (boxes) and predictions (fills)."""
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import mne

    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    mne.rename_channels(raw.info, lambda x: x.replace("EEG ", "").replace("-REF", "").replace("-LE", "").upper())
    channels = [ch for ch in CLINICAL_ORDER if ch in raw.ch_names]
    channels += [ch for ch in timeline if ch in raw.ch_names and ch not in channels]
    raw.pick(channels)
    raw.load_data()
    raw.filter(1.0, 40.0, fir_design="firwin", verbose=False)
    sf = raw.info["sfreq"]

    fig, axes = plt.subplots(1, len(start_times), figsize=(10 * len(start_times), 12), squeeze=False)
    for ax, t0, title in zip(axes[0], start_times, titles):
        if t0 is None:
            ax.text(0.5, 0.5, "No such segment", ha="center", fontsize=14)
            ax.set_axis_off()
            continue
        data = raw.get_data(start=int(t0 * sf), stop=int((t0 + duration) * sf)) * 1e6 * scale
        times = np.arange(data.shape[1]) / sf
        for row, ch in enumerate(channels[::-1]):
            base = row * y_offset
            ax.plot(times, data[channels.index(ch)] + base, color="k", linewidth=0.6, zorder=10)
            for w in timeline.get(ch, []):
                rel0, rel1 = max(0, w["t_start"] - t0), min(duration, w["t_end"] - t0)
                if rel1 <= rel0:
                    continue
                if w["pred"] != 0:
                    ax.add_patch(patches.Rectangle((rel0, base - y_offset / 2 + 15), rel1 - rel0, y_offset - 30,
                                                   linewidth=0, facecolor=PRED_COLORS.get(w["pred"], "orange"),
                                                   alpha=0.5, zorder=2))
                if w["true"] != 0:
                    ax.add_patch(patches.Rectangle((rel0, base - y_offset / 2 + 5), rel1 - rel0, y_offset - 10,
                                                   linewidth=2.5, edgecolor=GT_COLORS.get(w["true"], "purple"),
                                                   facecolor="none", zorder=5))
        ax.set_yticks([i * y_offset for i in range(len(channels))], channels[::-1], fontweight="bold")
        ax.set_xlim(0, duration)
        ax.set_ylim(-y_offset / 2, len(channels) * y_offset - y_offset / 2)
        ax.set_xlabel("Time (s)")
        ax.set_title(f"{title}\nstart: {t0} s", fontsize=12, fontweight="bold")

    legend = [
        patches.Patch(facecolor="none", edgecolor=GT_COLORS[1], linewidth=2, label="Ground truth class 1"),
        patches.Patch(facecolor=PRED_COLORS[1], alpha=0.5, label="Predicted class 1"),
        patches.Patch(facecolor="none", edgecolor=GT_COLORS[2], linewidth=2, label="Ground truth class 2"),
        patches.Patch(facecolor=PRED_COLORS[2], alpha=0.5, label="Predicted class 2"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=4, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig
