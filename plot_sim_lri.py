#!/usr/bin/env python3
"""Plot test AUC versus LRI for one fixed beta/CLS/TC/CMI configuration."""

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BRANCHES = ("x", "y", "s", "r")

SERIES = {
    f"downstream_z{branch}_{target}": (task, branch)
    for branch in BRANCHES
    for target, task in (
        ("y", "Y"),
        ("s", "population"),
    )
}

SERIES.update({
    "downstream_zyr_y": ("Y", "yr"),
    "downstream_zsr_s": ("population", "sr"),
})

PANELS = tuple(
    (
        f"downstream_z{branch}_{target}",
        rf"Downstream: $z_{branch} \rightarrow {symbol}$ ({name})",
    )
    for branch in BRANCHES
    for target, symbol, name in (
        ("y", "Y", "phenotype_binary"),
        ("s", "S", "population"),
    )
) 

PANELS += (
    (
        "downstream_zyr_y",
        r"Downstream: $[z_y,z_r] \rightarrow Y$ (phenotype_binary)",
    ),
    (
        "downstream_zsr_s",
        r"Downstream: $[z_s,z_r] \rightarrow S$ (population)",
    ),
)

PANELS +=  (
    (
        "vae_cls_y",
        r"VAE head: $[z_y,z_r] \rightarrow Y$ (phenotype_binary)",
    ),
    (
        "vae_cls_s",
        r"VAE head: $[z_s,z_r] \rightarrow S$ (population)",
    ),
)

PANELS += (
    (
        "vae_opp_y",
        r"Opponent: $z_s \rightarrow Y$ (phenotype_binary)",
    ),
    (
        "vae_opp_s",
        r"Opponent: $z_y \rightarrow S$ (population)",
    ),
)



def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--cls", type=float, default=500)
    parser.add_argument("--tc", type=float, default=500)
    parser.add_argument("--cmi", type=float, default=500)
    parser.add_argument("--lri_values", type=float, nargs="+",
                        default=[0, 1, 50, 100, 500, 1000])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1024)
    return parser.parse_args()


def run_directory(args, lri):
    return (
        args.root
        / f"epochs_{args.epochs}_warmup_{args.warmup}"
        / f"beta_{args.beta:g}_cls_{args.cls:g}_tc_{args.tc:g}_cmi_{args.cmi:g}"
        / f"lri_{lri:g}" / f"seed_{args.seed}"
    )


def add_configuration(frame, args, lri):
    return frame.assign(
        repetition=args.root.name,
        beta=args.beta, cls=args.cls, tc=args.tc, cmi=args.cmi, lri=lri,
        seed=args.seed, epochs=args.epochs, warmup=args.warmup,
    )


def collect_auc(args):
    frames = []
    vae_series = (
        ("vae_cls_y", "auc_y"),
        ("vae_cls_s", "auc_s"),
        ("vae_opp_y", "auc_opponent_y"),
        ("vae_opp_s", "auc_opponent_s"),
    )
    for lri in args.lri_values:
        directory = run_directory(args, lri)
        downstream_path = directory / "downstream_metrics.csv"
        vae_path = directory / "vae_test_metrics.csv"
        if not downstream_path.is_file() or not vae_path.is_file():
            warnings.warn(f"Missing result files; leaving a gap: {directory}")
            continue

        downstream = pd.read_csv(downstream_path)
        vae = pd.read_csv(vae_path)
        if len(vae) != 1:
            raise ValueError(f"Expected one test-result row: {vae_path}")

        # Keep the existing prediction-task and branch definitions.
        for series, (task, branch) in SERIES.items():
            selected = downstream.loc[
                (downstream["task"] == task)
                & (downstream["branch"] == branch), ["auc"]
            ].copy()
            if selected.empty:
                warnings.warn(f"Missing {series}: {downstream_path}")
            selected["series"] = series
            frames.append(add_configuration(selected, args, lri))

        for series, column in vae_series:
            if column not in vae.columns:
                warnings.warn(f"Missing {column}: {vae_path}")
                continue
            selected = vae[[column]].rename(columns={column: "auc"}).copy()
            selected["series"] = series
            frames.append(add_configuration(selected, args, lri))

    if not frames:
        raise FileNotFoundError(f"No matching results under {args.root}")
    values = pd.concat(frames, ignore_index=True)
    if values.empty:
        raise FileNotFoundError(f"No AUC values under {args.root}")
    keys = ["lri", "series"]
    if values.duplicated(keys).any():
        raise ValueError("Duplicate AUC rows for the same LRI setting and series.")
    return values.sort_values(keys).reset_index(drop=True)





def plot_lri(values, args, output_dir):
    lri_values = sorted(set(args.lri_values))
    # Keep the parameter settings equally spaced, as in the CMI plot.
    positions = np.arange(len(lri_values))
    fig, axes = plt.subplots(7, 2, figsize=(13, 24), sharex=True, sharey=True)

    for ax, (series, title) in zip(axes.flat, PANELS):
        points = values.loc[
            values["series"] == series
        ].set_index("lri").reindex(lri_values)
        ax.plot(
            positions, points["auc"].to_numpy(dtype=float),
            color="#0072B2", marker="o", linewidth=1.8, markersize=5,
        )
        ax.axhline(0.5, color="0.5", linestyle=":", linewidth=1)
        ax.set_title(title)
        ax.set_xticks(positions, [f"{value:g}" for value in lri_values])
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.1)

    for ax in axes[:, 0]:
        ax.set_ylabel("AUC")
    for ax in axes[-1, :]:
        ax.set_xlabel("LRI weight")
    fig.suptitle(
        f"CAD-VAE | {args.root.name} | KL beta = {args.beta:g} | "
        f"CLS = {args.cls:g} | TC = {args.tc:g} | CMI = {args.cmi:g}",
        fontsize=15, y=0.985,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    output_path = output_dir / f"beta_{args.beta:g}_cls_{args.cls:g}_auc_vs_lri.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")

def plot_training_losses(root):
    """Save a loss figure beside each training log."""
    root = Path(root)

    metrics = (
        ("loss", "Total objective loss"),
        ("recon", "Reconstruction loss"),
        ("kl", "KL loss"),
        ("cls_y", "Y classification loss"),
        ("cls_s", "Population classification loss"),
        ("cmi_y", "CMI: Y from z_s"),
        ("cmi_s", "CMI: Population from z_y"),
        ("relevance_y", "LRI: Y relevance"),
        ("relevance_s", "LRI: Population relevance"),
        ("tc", "TC loss"),
        ("opponent_y", "Y opponent classification from z_s"),
        ("opponent_s", "Population opponent classification from z_y"),
    )

    count = 0

    for log_path in sorted(root.rglob("training_log.csv")):
        history = pd.read_csv(log_path)

        fig, axes = plt.subplots(
            4,
            3,
            figsize=(16, 12),
            sharex=True,
        )

        for ax, (metric,metric_title) in zip(axes.flat, metrics):
            for prefix, label, color in (
                ("train", "Train", "#0072B2"),
                ("val", "Validation", "#D55E00"),
            ):
                column = f"{prefix}_{metric}"

                # Some objectives are recorded only during training.
                if column in history.columns:
                    ax.plot(
                        history["epoch"],
                        history[column],
                        label=label,
                        color=color,
                        linewidth=1.2,
                    )

            ax.set_title(metric_title, fontsize=10)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)

        # Remove the two unused panels.
        for ax in list(axes.flat)[len(metrics):]:
            fig.delaxes(ax)

        fig.suptitle(
            f"CAD-VAE training losses\n{log_path.parent.relative_to(root)}",
            fontsize=11,
        )

        fig.supxlabel("Epoch")

        fig.tight_layout(
            rect=(0, 0.03, 1, 0.94)
        )

        output_path = log_path.with_name(
            "training_losses.png"
        )

        fig.savefig(
            output_path,
            dpi=150,
        )

        plt.close(fig)
        count += 1

    print(f"Saved {count} loss figures.")
    
    
def main():
    args = parse_args()
    output_dir = args.output_dir or (
        args.root / "auc_plots" / (
            f"epochs_{args.epochs}_warmup_{args.warmup}"
            f"_beta_{args.beta:g}_cls_{args.cls:g}_tc_{args.tc:g}"
            f"_cmi_{args.cmi:g}_seed_{args.seed}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    values = collect_auc(args)
    values.to_csv(output_dir / "auc_values.csv", index=False)
    plot_lri(values, args, output_dir)
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
