"""Resample the Bonn EEG dataset to 200 Hz, band-pass filter it (1-45 Hz FIR) and cut it into 2 s windows
with 50 % overlap.

<raw-dir> holds the five Bonn sets (Z, O, N, F, S), 100 .txt files each, in any folder structure.
Recordings are split into train/eval/test before windowing. By default the split is the one used in the
experiments (configs/splits/bonn_split.json).

Example:
    python scripts/preprocess_bonn.py --raw-dir /path/to/bonn --out data/bonn
    python scripts/preprocess_bonn.py --raw-dir /path/to/bonn --out data/bonn --split-file none --seed 0
"""

import argparse
import os

from neurotransformer.data.preprocessing import preprocess_bonn

DEFAULT_SPLIT = os.path.join(os.path.dirname(__file__), "..", "configs", "splits", "bonn_split.json")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--out", default="data/bonn")
    p.add_argument("--split-file", default=DEFAULT_SPLIT,
                   help="JSON record split; pass 'none' to draw a new stratified 70/15/15 split")
    p.add_argument("--seed", type=int, default=42, help="Seed for a new split")
    p.add_argument("--scale", type=float, default=1.0, help="Fixed scaling factor (Bonn data is already in µV)")
    args = p.parse_args()
    split_file = None if args.split_file.lower() == "none" else args.split_file
    preprocess_bonn(args.raw_dir, args.out, split_file=split_file, seed=args.seed, scale=args.scale)


if __name__ == "__main__":
    main()
