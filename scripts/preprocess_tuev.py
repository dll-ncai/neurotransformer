"""Preprocess the TUH EEG Events (TUEV) corpus into per-channel 2 s windows.

Each recording is clipped, resampled to 200 Hz, band-pass filtered (1-45 Hz FIR), cleaned with ICA and converted
to the 22-channel TCP montage. Every channel is windowed independently and a window is labelled epileptic when
its overlap with an SPSW/GPED/PLED annotation on that channel exceeds tau.

Expects the official layout <root>/edf/{train,eval}/**/*.edf with a .rec file next to each EDF.

Example:
    python scripts/preprocess_tuev.py --root /path/to/tuev/v2.0.1 --out data/tuev
"""

import argparse

from neurotransformer.data.preprocessing import preprocess_tuev


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="TUEV release folder (contains edf/)")
    p.add_argument("--out", default="data/tuev")
    p.add_argument("--tau", type=float, default=0.25, help="Minimum epileptiform overlap (fraction of the window)")
    p.add_argument("--clip-uv", type=float, default=100.0, help="Amplitude clipping threshold in µV")
    p.add_argument("--no-ica", action="store_true", help="Skip ICA artifact removal (faster, for quick tests)")
    p.add_argument("--limit", type=int, help="Only process the first N recordings per split")
    args = p.parse_args()
    preprocess_tuev(args.root, args.out, tau=args.tau, clip_uv=args.clip_uv, ica=not args.no_ica, limit=args.limit)


if __name__ == "__main__":
    main()
