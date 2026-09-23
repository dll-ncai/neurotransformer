"""Train Neurotransformer (or an ablation variant) on one dataset.

Example:
    python scripts/train.py --config configs/bonn.yaml --data-root data/bonn
    python scripts/train.py --config configs/bonn.yaml --model wo_decoder
"""

import argparse
import os

import torch
import yaml

from neurotransformer.data import build_dataloaders, get_spec
from neurotransformer.engine import append_results_csv, evaluate, fit, format_metrics
from neurotransformer.models import MODEL_REGISTRY, build_model
from neurotransformer.utils import (build_criterion, build_optimizer, build_scheduler, count_params, load_config,
                                    resolve_class_counts, set_seed)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="YAML config, e.g. configs/bonn.yaml")
    p.add_argument("--data-root", help="Folder with train/ eval/ [test/] (overrides the config)")
    p.add_argument("--model", choices=list(MODEL_REGISTRY), help="Model name (overrides the config)")
    p.add_argument("--epochs", type=int, help="Overrides the config")
    p.add_argument("--patience", type=int, help="Early-stopping patience in epochs (overrides the config)")
    p.add_argument("--batch-size", type=int, help="Overrides the config")
    p.add_argument("--num-workers", type=int, help="Overrides the config")
    p.add_argument("--seed", type=int, help="Seed everything for a reproducible run (default: unseeded)")
    p.add_argument("--output-dir", help="Default: runs/<dataset>/<model>")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-progress", action="store_true", help="Hide per-batch progress bars")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    for key in ("data_root", "model", "epochs", "patience", "batch_size", "num_workers", "seed"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value

    spec = get_spec(cfg["dataset"])
    out_dir = args.output_dir or os.path.join("runs", cfg["dataset"], cfg["model"])
    os.makedirs(out_dir, exist_ok=True)

    seed = cfg.get("seed")
    if seed is not None:
        set_seed(seed)
    device = torch.device(args.device)
    loaders = build_dataloaders(cfg["data_root"], cfg["dataset"], cfg["batch_size"], cfg["num_workers"], seed)
    if "train" not in loaders or "eval" not in loaders:
        raise SystemExit(f"{cfg['data_root']} must contain train/ and eval/ folders")
    print({split: len(loader.dataset) for split, loader in loaders.items()})

    # Save the config with the class counts filled in, so evaluate.py can reuse it without recounting.
    cfg["loss"] = resolve_class_counts(cfg.get("loss"), loaders["train"].dataset)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    model = build_model(cfg["model"], spec.n_classes).to(device)
    print(f"{cfg['model']}: {count_params(model):,} parameters, device={device}")

    criterion = build_criterion(cfg.get("loss"), device)
    optimizer = build_optimizer(model.parameters(), cfg["optimizer"])
    scheduler = build_scheduler(optimizer, cfg.get("scheduler"))
    checkpoint = os.path.join(out_dir, "best.pth")

    history = fit(model, loaders["train"], loaders["eval"], optimizer, criterion, device, cfg["epochs"],
                  spec.n_classes, checkpoint, scheduler, cfg.get("monitor", "eval_loss"), cfg.get("patience"),
                  not args.no_progress)
    history.to_csv(os.path.join(out_dir, "history.csv"), index=False)

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    for split in ("eval", "test"):
        if split not in loaders:
            continue
        metrics, cm = evaluate(model, loaders[split], device, spec.n_classes, criterion, not args.no_progress)
        append_results_csv(os.path.join(out_dir, f"{split}_results.csv"), checkpoint, metrics, cm)
        print(f"[{split}] {format_metrics(metrics)}\n{cm}")
    print(f"Best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
