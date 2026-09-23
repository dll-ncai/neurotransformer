"""Seeding, configuration and training-object helpers."""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import yaml


def set_seed(seed, deterministic=True):
    """Seed Python, NumPy and PyTorch. With ``deterministic``, also force deterministic CUDA kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Required by torch.use_deterministic_algorithms for cuBLAS; must be set before CUDA is used.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    """DataLoader ``worker_init_fn`` that derives each worker's NumPy/random seed from the torch seed."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_class_counts(loss_cfg, train_dataset):
    """Fill in ``loss_cfg["class_counts"]`` from the training labels when ``class_weights`` is ``"balanced"``.

    Returns an updated copy of ``loss_cfg``. Counting reads every training file once.
    """
    loss_cfg = dict(loss_cfg or {})
    if loss_cfg.get("class_weights") == "balanced" and not loss_cfg.get("class_counts"):
        print("Counting training labels for class weights...")
        counts = np.bincount(train_dataset.targets(), minlength=train_dataset.spec.n_classes)
        loss_cfg["class_counts"] = counts.tolist()
        print(f"class counts: {loss_cfg['class_counts']}")
    return loss_cfg


def build_criterion(loss_cfg, device):
    """Cross-entropy with optional inverse-frequency class weights ``N / (K * count_k)``.

    Weights are used when ``class_counts`` is set (see :func:`resolve_class_counts`); otherwise the loss is
    unweighted.
    """
    loss_cfg = loss_cfg or {}
    weight = None
    counts = loss_cfg.get("class_counts")
    if counts:
        counts = np.asarray(counts, dtype=np.float64)
        weight = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=loss_cfg.get("label_smoothing", 0.0))


def build_optimizer(params, opt_cfg):
    opt_cfg = dict(opt_cfg)
    name = opt_cfg.pop("name", "adam").lower()
    if "betas" in opt_cfg:
        opt_cfg["betas"] = tuple(opt_cfg["betas"])
    if name == "adam":
        return torch.optim.Adam(params, **opt_cfg)
    if name == "adamw":
        return torch.optim.AdamW(params, **opt_cfg)
    raise ValueError(f"Unknown optimizer '{name}'")


def build_scheduler(optimizer, sched_cfg):
    if not sched_cfg:
        return None
    sched_cfg = dict(sched_cfg)
    name = sched_cfg.pop("name").lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **sched_cfg)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **sched_cfg)
    raise ValueError(f"Unknown scheduler '{name}'")
