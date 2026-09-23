"""PyTorch dataset for pre-windowed EEG stored as one ``.npz`` file per window."""

import os
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class DatasetSpec:
    """How to read windows of one dataset and map its raw labels to class indices."""

    x_key: str
    y_key: str
    label_map: Dict[int, int]
    class_names: List[str] = field(default_factory=list)

    @property
    def n_classes(self):
        return len(self.class_names)


DATASETS = {
    # NMT scalp-EEG events. Raw `ab_label`: 0 = normal, 1 and 3 are merged, 2 is kept on its own.
    "nmt": DatasetSpec(
        x_key="data",
        y_key="ab_label",
        label_map={0: 0, 1: 1, 3: 1, 2: 2},
        class_names=["Normal", "SW", "SSW"],
    ),
    # Bonn University EEG. `ab_label`: sets Z/O -> 0, sets N/F -> 1, set S -> 2 (see preprocessing.py).
    "bonn": DatasetSpec(
        x_key="data",
        y_key="ab_label",
        label_map={0: 0, 1: 1, 3: 1, 2: 2},
        class_names=["Normal", "Interictal", "Ictal"],
    ),
    # TUH EEG Events. `y` is already binary (see preprocessing.label_window): 1 = window overlaps
    # SPSW/GPED/PLED by more than tau, 0 = otherwise annotated (EYEM/ARTF/BCKG).
    "tuev": DatasetSpec(
        x_key="x",
        y_key="y",
        label_map={0: 0, 1: 1},
        class_names=["Non-epileptic", "Epileptic"],
    ),
}


def get_spec(dataset):
    try:
        return DATASETS[dataset]
    except KeyError:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {', '.join(DATASETS)}") from None


class NpzEEGDataset(Dataset):
    """Reads every ``*.npz`` window in ``folder``.

    Each file holds one window of shape ``(1, L)`` under ``spec.x_key`` and a scalar raw label under
    ``spec.y_key``. Items are returned as ``(float32 tensor (1, L), int64 class index)``.
    """

    def __init__(self, folder, dataset="nmt", transform=None):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Dataset folder not found: {folder}")
        self.folder = folder
        self.spec = get_spec(dataset)
        self.transform = transform
        self.files = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {folder}")

    def __len__(self):
        return len(self.files)

    def _map_label(self, raw):
        raw = int(np.asarray(raw).item())
        try:
            return self.spec.label_map[raw]
        except KeyError:
            raise ValueError(f"Unexpected label value {raw}") from None

    def __getitem__(self, idx):
        sample = np.load(self.files[idx], allow_pickle=True)
        data = torch.from_numpy(np.asarray(sample[self.spec.x_key], dtype=np.float32).reshape(1, -1))
        target = torch.tensor(self._map_label(sample[self.spec.y_key])).long()
        if self.transform:
            data = self.transform(data)
        return data, target

    def targets(self):
        """Class index of every window. Reads every file, so it can be slow on large datasets."""
        return np.array([self._map_label(np.load(f, allow_pickle=True)[self.spec.y_key]) for f in self.files])


def available_splits(data_root):
    """Subset of ``train``/``eval``/``test`` that exist as folders under ``data_root``."""
    return [s for s in ("train", "eval", "test") if os.path.isdir(os.path.join(data_root, s))]


def build_dataloaders(data_root, dataset, batch_size, num_workers=4, seed=None, splits=None):
    """DataLoaders for each split folder under ``data_root``; only ``train`` is shuffled."""
    from ..utils import seed_worker

    loaders = {}
    for split in splits or available_splits(data_root):
        ds = NpzEEGDataset(os.path.join(data_root, split), dataset)
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=generator,
        )
    return loaders
