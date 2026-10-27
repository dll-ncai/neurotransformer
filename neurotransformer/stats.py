"""Paired significance tests for two classifiers evaluated on the same windows."""

import numpy as np
from scipy.stats import binomtest, chi2


def mcnemar(correct_a, correct_b, exact_below=25):
    """McNemar's test on per-window correctness of model A and model B.

    ``b`` counts the windows only A classifies correctly and ``c`` those only B does. The statistic is
    continuity-corrected, ``(|b - c| - 1)^2 / (b + c)`` with one degree of freedom. When ``b + c < exact_below``
    the p-value comes from the exact binomial test instead. Returns ``{b, c, chi2, p, exact}``.
    """
    a, other = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    b, c = int((a & ~other).sum()), int((~a & other).sum())
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p": 1.0, "exact": True}
    stat = (abs(b - c) - 1) ** 2 / n
    exact = n < exact_below
    p = binomtest(min(b, c), n, 0.5).pvalue if exact else chi2.sf(stat, 1)
    return {"b": b, "c": c, "chi2": float(stat), "p": float(p), "exact": exact}


def clustered_mcnemar(correct_a, correct_b, groups):
    """McNemar's test for clustered data (Durkalski et al., Stat Med 2003).

    Overlapping windows from one recording are not independent, so the recording (or patient) is the unit:
    ``chi2 = (sum_k d_k)^2 / sum_k d_k^2`` with ``d_k = b_k - c_k`` per group ``k``, one degree of freedom.
    Returns ``{chi2, p, n_groups}``.
    """
    a, other, groups = np.asarray(correct_a, bool), np.asarray(correct_b, bool), np.asarray(groups)
    uniq, idx = np.unique(groups, return_inverse=True)
    d = np.bincount(idx, weights=(a & ~other).astype(float)) - np.bincount(idx, weights=(~a & other).astype(float))
    denom = float((d ** 2).sum())
    stat = d.sum() ** 2 / denom if denom > 0 else 0.0
    return {"chi2": float(stat), "p": float(chi2.sf(stat, 1)) if denom > 0 else 1.0, "n_groups": len(uniq)}


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, in the input order."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def compare(labels, reference_pred, others, groups=None):
    """Test the reference model against each model in ``others`` (``{name: predictions}``).

    Holm's correction is applied across the comparisons, separately for the window-level and the clustered
    p-values. Returns one dict per model.
    """
    labels = np.asarray(labels)
    ok_ref = np.asarray(reference_pred) == labels
    rows = []
    for name, pred in others.items():
        ok = np.asarray(pred) == labels
        row = {"model": name, "acc_reference": float(ok_ref.mean()), "acc_model": float(ok.mean()), **mcnemar(ok_ref, ok)}
        if groups is not None:
            cl = clustered_mcnemar(ok_ref, ok, groups)
            row.update(chi2_clustered=cl["chi2"], p_clustered=cl["p"], n_groups=cl["n_groups"])
        rows.append(row)
    keys = ["p", "p_clustered"] if groups is not None else ["p"]
    for key in keys:
        for row, adj in zip(rows, holm([r[key] for r in rows])):
            row[key + "_holm"] = float(adj)
    return rows
