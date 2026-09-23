"""Training and evaluation loops."""

import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from tqdm.auto import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None, progress=True):
    """One pass over ``loader``. Steps ``scheduler`` once at the end. Returns ``(mean loss, accuracy)``."""
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for x, y in tqdm(loader, desc="Training", leave=False, disable=not progress):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.float(), y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_samples += y.size(0)
        total_correct += (output.argmax(dim=1) == y).sum().item()
    if scheduler is not None:
        scheduler.step()
    return total_loss / len(loader), total_correct / total_samples


@torch.no_grad()
def predict(model, loader, device, criterion=None, progress=True):
    """Return ``(labels, predictions, probabilities, mean loss or None)`` over ``loader``."""
    model.eval()
    labels, probs, total_loss = [], [], 0.0
    for x, y in tqdm(loader, desc="Evaluating", leave=False, disable=not progress):
        x, y = x.to(device), y.to(device)
        output = model(x)
        if criterion is not None:
            total_loss += criterion(output.float(), y).item()
        labels.append(y.cpu().numpy())
        probs.append(torch.softmax(output.float(), dim=1).cpu().numpy())
    labels, probs = np.concatenate(labels), np.concatenate(probs)
    avg_loss = total_loss / len(loader) if criterion is not None else None
    return labels, probs.argmax(axis=1), probs, avg_loss


def compute_metrics(labels, preds, probs, n_classes):
    """Accuracy, precision, recall, F1, specificity, AUROC and confusion matrix.

    For two classes, precision/recall/F1/specificity refer to the positive class (1). For more classes
    they are macro-averaged and AUROC is one-vs-rest.
    """
    classes = list(range(n_classes))
    cm = confusion_matrix(labels, preds, labels=classes)
    average = "binary" if n_classes == 2 else "macro"

    total = cm.sum()
    specificity = []
    for i in classes:
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        specificity.append(tn / (tn + fp) if tn + fp > 0 else 0.0)

    try:
        if n_classes == 2:
            auroc = roc_auc_score(labels, probs[:, 1])
        else:
            auroc = roc_auc_score(labels, probs, multi_class="ovr", labels=classes)
    except ValueError:  # e.g. a class is absent from `labels`
        auroc = float("nan")

    metrics = {
        "accuracy": float((labels == preds).mean()),
        "precision": precision_score(labels, preds, average=average, labels=classes, zero_division=0),
        "recall": recall_score(labels, preds, average=average, labels=classes, zero_division=0),
        "f1_score": f1_score(labels, preds, average=average, labels=classes, zero_division=0),
        "specificity": specificity[1] if n_classes == 2 else float(np.mean(specificity)),
        "auroc": auroc,
    }
    return metrics, cm


def evaluate(model, loader, device, n_classes, criterion=None, progress=True):
    """Run ``model`` over ``loader`` and return ``(metrics dict, confusion matrix)``."""
    labels, preds, probs, avg_loss = predict(model, loader, device, criterion, progress)
    metrics, cm = compute_metrics(labels, preds, probs, n_classes)
    metrics = {"total_samples": len(labels), "avg_loss": avg_loss, **metrics}
    return metrics, cm


def fit(model, train_loader, eval_loader, optimizer, criterion, device, epochs, n_classes,
        checkpoint_path, scheduler=None, monitor="eval_loss", patience=None, progress=True):
    """Train for up to ``epochs`` epochs, evaluating after each and saving the best weights to ``checkpoint_path``.

    ``monitor`` is ``"eval_loss"`` (lower is better) or ``"eval_acc"`` (higher is better). With ``patience``,
    training stops early once ``monitor`` has not improved for that many consecutive epochs.
    Returns the per-epoch history as a DataFrame.
    """
    if monitor not in ("eval_loss", "eval_acc"):
        raise ValueError("monitor must be 'eval_loss' or 'eval_acc'")
    if os.path.dirname(checkpoint_path):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    best = float("inf") if monitor == "eval_loss" else -float("inf")
    epochs_since_best = 0
    history = []
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler, progress)
        metrics, _ = evaluate(model, eval_loader, device, n_classes, criterion, progress)
        eval_loss, eval_acc = metrics["avg_loss"], metrics["accuracy"]

        improved = eval_loss <= best if monitor == "eval_loss" else eval_acc > best
        if improved:
            best = eval_loss if monitor == "eval_loss" else eval_acc
            torch.save(model.state_dict(), checkpoint_path)
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "eval_loss": eval_loss,
                        "eval_acc": eval_acc, "eval_f1": metrics["f1_score"], "saved": improved})
        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"eval_loss={eval_loss:.4f} eval_acc={eval_acc:.4f} eval_f1={metrics['f1_score']:.4f}"
              + ("  *saved*" if improved else ""))
        if patience is not None and epochs_since_best >= patience:
            print(f"Early stopping: no improvement in {monitor} for {patience} epochs")
            break
    return pd.DataFrame(history)


def append_results_csv(path, model_name, metrics, cm):
    """Append one row (metrics and flattened confusion matrix) to a results CSV, creating it if needed."""
    row = {"model_name": model_name, **metrics}
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            row[f"conf_mat_{i}_{j}"] = int(cm[i, j])
    df = pd.DataFrame([row])
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def format_metrics(metrics):
    keys = ["accuracy", "precision", "recall", "f1_score", "specificity", "auroc"]
    return "  ".join(f"{k}={metrics[k]:.4f}" for k in keys)
