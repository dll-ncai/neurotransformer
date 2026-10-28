"""Evaluate a trained checkpoint on one split and append the metrics to a CSV.

Pass the config.yaml saved next to the checkpoint by train.py: it already holds the class counts, so the
weighted loss is reported without recounting the training labels.

With --save-preds, the per-window labels, predictions, probabilities and window file names are also saved
to an .npz, which scripts/mcnemar.py reads.

Example:
    python scripts/evaluate.py --config runs/bonn/neurotransformer/config.yaml \
        --checkpoint runs/bonn/neurotransformer/best.pth --split test
"""

import argparse
import os

import numpy as np
import torch

from neurotransformer.data import NpzEEGDataset, build_dataloaders, get_spec
from neurotransformer.engine import append_results_csv, compute_metrics, format_metrics, predict
from neurotransformer.models import MODEL_NAMES, build_model
from neurotransformer.utils import build_criterion, load_config, resolve_class_counts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["train", "eval", "test"])
    p.add_argument("--data-root", help="Overrides the config")
    p.add_argument("--model", choices=MODEL_NAMES, help="Overrides the config")
    p.add_argument("--output", default="results/results.csv", help="CSV to append the metrics to")
    p.add_argument("--save-preds", help="Also save per-window predictions to this .npz (input for mcnemar.py)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.save_preds and args.split == "train":
        raise SystemExit("--save-preds needs an unshuffled split: use --split eval or --split test")

    cfg = load_config(args.config)
    data_root = args.data_root or cfg["data_root"]
    spec = get_spec(cfg["dataset"])
    device = torch.device(args.device)

    if not os.path.isdir(os.path.join(data_root, args.split)):
        raise SystemExit(f"No '{args.split}' folder under {data_root}")
    loader = build_dataloaders(data_root, cfg["dataset"], args.batch_size, args.num_workers, splits=[args.split])[args.split]

    model_name = args.model or cfg["model"]
    model = build_model(model_name, spec.n_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))

    loss_cfg = cfg.get("loss")
    if (loss_cfg or {}).get("class_weights") == "balanced" and not loss_cfg.get("class_counts"):
        train_dir = os.path.join(data_root, "train")
        if os.path.isdir(train_dir):
            loss_cfg = resolve_class_counts(loss_cfg, NpzEEGDataset(train_dir, cfg["dataset"]))
        else:
            print("No train/ folder to count class weights from; reporting unweighted loss.")

    labels, preds, probs, avg_loss = predict(model, loader, device, build_criterion(loss_cfg, device))
    metrics, cm = compute_metrics(labels, preds, probs, spec.n_classes)
    metrics = {"total_samples": len(labels), "avg_loss": avg_loss, **metrics}
    append_results_csv(args.output, args.checkpoint, metrics, cm)
    print(f"[{args.split}] {len(loader.dataset)} windows  {format_metrics(metrics)}")
    print(cm)
    print(f"Appended to {args.output}")

    if args.save_preds:
        if os.path.dirname(args.save_preds):
            os.makedirs(os.path.dirname(args.save_preds), exist_ok=True)
        # eval/test loaders are not shuffled, so predictions follow the dataset's sorted file list
        files = np.array([os.path.basename(f) for f in loader.dataset.files])
        np.savez(args.save_preds, files=files, y=labels, pred=preds, prob=probs,
                 dataset=cfg["dataset"], split=args.split, model=model_name)
        print(f"Saved per-window predictions to {args.save_preds}")


if __name__ == "__main__":
    main()
