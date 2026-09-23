"""Report parameters, FLOPs and single-window inference latency for Neurotransformer and its ablations.

Example:
    python scripts/benchmark.py                       # all models, 3 classes
    python scripts/benchmark.py --models neurotransformer --output results/model_costs.csv
"""

import argparse
import os

import pandas as pd

from neurotransformer.benchmark import model_costs
from neurotransformer.models import MODEL_REGISTRY, build_model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY), choices=list(MODEL_REGISTRY))
    p.add_argument("--n-classes", type=int, default=3)
    p.add_argument("--window-samples", type=int, default=400)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--output", help="Optional CSV path")
    args = p.parse_args()

    rows = []
    for name in args.models:
        costs = model_costs(build_model(name, args.n_classes), args.window_samples, iters=args.iters)
        rows.append({"model_name": name, **costs})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if df["mflops_per_segment"].isna().any():
        print("\nInstall `thop` (pip install thop) to report FLOPs.")
    if args.output:
        if os.path.dirname(args.output):
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
        df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
