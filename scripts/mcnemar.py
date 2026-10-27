"""McNemar tests of one model's per-window predictions against other models on the same windows.

Inputs are .npz files from scripts/evaluate.py --save-preds. Predictions from other code work too if the file
has `files` (window file names), `y` (class indices) and `pred`. Windows are matched by file name.

b = windows only the reference model classifies correctly; c = windows only the compared model does, so
c > b means the compared model is more accurate. Window-level McNemar is continuity-corrected (exact binomial
when b + c < 25). Windows overlap by 50 % and many come from one recording, so they are not independent: the
clustered test (Durkalski et al., 2003) treats each recording (or TUEV patient) as one unit and is the one to
report. p-values are Holm-adjusted over the compared models.

Example:
    python scripts/evaluate.py --config runs/tuev/neurotransformer/config.yaml \
        --checkpoint runs/tuev/neurotransformer/best.pth --split eval \
        --save-preds runs/tuev/neurotransformer/preds_eval.npz
    python scripts/mcnemar.py --reference runs/tuev/neurotransformer/preds_eval.npz \
        --compare wo_trans=runs/tuev/wo_trans/preds_eval.npz wo_conv=runs/tuev/wo_conv/preds_eval.npz \
        --group patient --latex results/mcnemar_tuev.tex
"""

import argparse
import os
import re

import numpy as np

from neurotransformer.stats import compare

# TUEV eval recordings are named <event>_<patient>_..., train recordings <patient>_<session>
TUEV_EVENT_PREFIXES = {"spsw", "gped", "pled", "eyem", "artf", "bckg"}


def window_group(name, dataset, level):
    """Recording or patient id of a window, from the file names written by the preprocessing scripts."""
    stem = os.path.splitext(os.path.basename(name))[0]
    if dataset == "tuev":
        recording = stem.rsplit("_", 2)[0]  # <recording>_<channel>_<index>
        if level == "recording":
            return recording
        parts = recording.split("_")
        return parts[1] if parts[0] in TUEV_EVENT_PREFIXES else parts[0]
    if level == "patient":
        raise SystemExit("--group patient is only defined for TUEV; use --group recording or --group-regex")
    return stem.split("_")[0]  # NMT: 0000011_..., Bonn: F001_seg0003


def load_preds(path):
    d = np.load(path, allow_pickle=False)
    for key in ("files", "y", "pred"):
        if key not in d:
            raise SystemExit(f"{path} has no '{key}' array")
    return {k: d[k] for k in d.files}


def aligned(ref, other, name):
    """Reorder ``other``'s predictions to the reference's window order and check the labels agree."""
    index = {f: i for i, f in enumerate(other["files"])}
    missing = [f for f in ref["files"] if f not in index]
    if missing or len(index) != len(ref["files"]):
        raise SystemExit(f"{name}: windows differ from the reference ({len(missing)} reference windows missing, "
                         f"{len(index)} vs {len(ref['files'])} windows)")
    order = np.array([index[f] for f in ref["files"]])
    if not (other["y"][order] == ref["y"]).all():
        raise SystemExit(f"{name}: labels differ from the reference for the same windows")
    return other["pred"][order]


def fmt_p(p):
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"


def latex_p(p):
    return "$<$0.001" if p < 1e-3 else f"{p:.3f}"


def latex_escape(text):
    return re.sub(r"([_&%#])", r"\\\1", text)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", required=True, help="Predictions of the model tested against all others")
    p.add_argument("--reference-name", help="Defaults to the model name stored in the file")
    p.add_argument("--compare", nargs="+", required=True, metavar="[NAME=]PATH")
    p.add_argument("--dataset", choices=["nmt", "bonn", "tuev"], help="Defaults to the dataset stored in the reference file")
    p.add_argument("--group", default="recording", choices=["recording", "patient", "none"],
                   help="Unit of the clustered test (patient: TUEV only); 'none' runs only the window-level test")
    p.add_argument("--group-regex", help="Take the group id from each window file name with this regex instead "
                                         "(first capture group, or the whole match)")
    p.add_argument("--latex", help="Also write the results as a LaTeX table to this path")
    args = p.parse_args()

    ref = load_preds(args.reference)
    ref_name = args.reference_name or str(ref.get("model", "reference"))
    others = {}
    for item in args.compare:
        name, path = item.split("=", 1) if "=" in item else (None, item)
        d = load_preds(path)
        name = name or str(d.get("model", os.path.splitext(os.path.basename(path))[0]))
        if name in others:
            raise SystemExit(f"Two compared models are named '{name}'; use NAME=PATH")
        others[name] = aligned(ref, d, name)

    groups = None
    if args.group_regex:
        rx = re.compile(args.group_regex)
        matches = [rx.search(f) for f in ref["files"]]
        if not all(matches):
            raise SystemExit("--group-regex does not match every window file name")
        groups = np.array([m.group(1) if m.groups() else m.group(0) for m in matches])
    elif args.group != "none":
        dataset = args.dataset or (str(ref["dataset"]) if "dataset" in ref else None)
        if dataset is None:
            raise SystemExit("The reference file stores no dataset name; pass --dataset or --group-regex")
        groups = np.array([window_group(f, dataset, args.group) for f in ref["files"]])

    rows = compare(ref["y"], ref["pred"], others, groups)
    unit = "groups" if args.group_regex else f"{args.group}s"
    print(f"{ref_name} vs others on {len(ref['y'])} windows"
          + (f", {rows[0]['n_groups']} {unit}" if groups is not None else ""))
    header = f"{'model':24s} {'acc ref':>7s} {'acc':>7s} {'b':>6s} {'c':>6s} {'chi2':>8s} {'p':>8s} {'p_holm':>8s}"
    if groups is not None:
        header += f" | {'chi2_cl':>7s} {'p_cl':>8s} {'p_cl_holm':>9s}"
    print(header)
    for r in rows:
        line = (f"{r['model']:24s} {100 * r['acc_reference']:7.2f} {100 * r['acc_model']:7.2f} {r['b']:6d} {r['c']:6d} "
                f"{r['chi2']:8.2f} {fmt_p(r['p']):>8s} {fmt_p(r['p_holm']):>8s}")
        if groups is not None:
            line += f" | {r['chi2_clustered']:7.2f} {fmt_p(r['p_clustered']):>8s} {fmt_p(r['p_clustered_holm']):>9s}"
        print(line + ("  (exact)" if r["exact"] else ""))

    if args.latex:
        cols = "lrrrr" + ("r" if groups is not None else "")
        ref_tex = latex_escape(ref_name)
        head = rf"\textbf{{{ref_tex} vs.}} & $b$ & $c$ & $\chi^2$ & $p$" + (r" & $p$ (clustered)" if groups is not None else "")
        lines = [r"\begin{table}[ht]", r"\centering",
                 rf"\caption{{McNemar tests of {ref_tex} against each model. $b$: windows classified correctly by "
                 rf"{ref_tex} only; $c$: windows classified correctly by the other model only. $\chi^2$ is "
                 r"continuity-corrected and $p$-values are Holm-adjusted"
                 + (rf"; the clustered test (Durkalski et al., 2003) treats {unit} as the independent units." if groups is not None else ".")
                 + "}",
                 r"\begin{tabular}{@{}" + cols + "@{}}", r"\toprule", head + r" \\", r"\midrule"]
        for r in rows:
            cells = [latex_escape(r["model"]), str(r["b"]), str(r["c"]), f"{r['chi2']:.2f}", latex_p(r["p_holm"])]
            if groups is not None:
                cells.append(latex_p(r["p_clustered_holm"]))
            lines.append(" & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        if os.path.dirname(args.latex):
            os.makedirs(os.path.dirname(args.latex), exist_ok=True)
        with open(args.latex, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Wrote {args.latex}")


if __name__ == "__main__":
    main()
