import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = "/mnt/vast-nhr/home/nimanwai/medico-sam/experiments/evaluation/res/piccolo"

LABELS = {
    ("sam", True): "SAM",  # with masks result used as reference.
    ("sam", False): "SAM",
    ("medicosam-old", True): r"MedicoSAM*$_{\mathrm{Mask}}$",
    ("medicosam-old", False): "MedicoSAM*",
    ("medicosam-neu", True): r"$\mathbf{MedicoSAM*}_{\mathrm{Mask}}$",
    ("medicosam-neu", False): r"$\bf{MedicoSAM*}$",
}

GROUP_ORDER = [
    ("sam", False),
    ("medicosam-old", False),
    ("medicosam-old", True),
    ("medicosam-neu", False),
    ("medicosam-neu", True),
]

BAR_KEYS = ["p0", "b0", "p7", "b7"]
BAR_LABELS = ["Point", "Box", r"I$_P$", r"I$_B$"]
BAR_COLORS = ["#7CCBA2", "#FCDE9C", "#045275", "#90477F"]


def _read_series(csv_path):
    df = pd.read_csv(csv_path)
    it_col = "Unnamed: 0" if "Unnamed: 0" in df.columns else df.columns[0]
    ser = pd.Series(df["dice"].values, index=df[it_col].astype(int)).sort_index()
    return ser.reindex(range(32))


def _load_point_box_series(exp_dir, use_mask):
    mask_dir = "with_mask" if use_mask else "without_mask"
    base = os.path.join(exp_dir, mask_dir, "results", "polyp")
    p_csv = os.path.join(base, "iterative_prompts_start_point.csv")
    b_csv = os.path.join(base, "iterative_prompts_start_box.csv")
    return _read_series(p_csv), _read_series(b_csv)


def _plot_iterative_prompting_use_mask():
    models = ["sam", "medicosam-old", "medicosam-neu"]
    cols = [False, True]  # True = with mask, False = without
    col_titles = ["Without Mask Inputs", "With Mask Inputs"]
    model_titles = {
        "sam": "SAM",
        "medicosam-old": r"MedicoSAM*",
        "medicosam-neu": r"$\mathbf{MedicoSAM*}$",
    }

    data = {}
    global_max_box = 0.0
    for m in models:
        exp_dir = os.path.join(ROOT, m)
        if not os.path.isdir(exp_dir):
            continue
        for use_mask in cols:
            try:
                p_ser, b_ser = _load_point_box_series(exp_dir, use_mask)
            except FileNotFoundError:
                continue
            p8, b8 = p_ser.iloc[:8], b_ser.iloc[:8]
            data[(m, use_mask)] = (p8, b8)
            max_here = np.nanmax(b8.values)
            if np.isfinite(max_here):
                global_max_box = max(global_max_box, max_here)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 10), sharey=True)
    x = np.arange(8)

    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=24, fontweight="bold")

    for r, m in enumerate(models):
        for c, use_mask in enumerate(cols):
            ax = axes[r, c]
            if (m, use_mask) not in data:
                ax.axis("off")
                continue

            p8, b8 = data[(m, use_mask)]
            rel = b8 - p8

            ax.bar(x, p8, width=0.8, color="#7CCBA2", edgecolor="grey", label="Point")
            ax.bar(x, rel, width=0.8, bottom=p8, color="#FCDE9C", edgecolor="grey", label="Box")

            ax.set_xlim(-0.6, 7.6)
            ax.set_xticks(np.arange(0, 8, 1))

            if c == 0:
                ax.set_ylabel(model_titles[m], rotation=90, labelpad=10, fontsize=20)

            ax.grid(axis="y", linestyle=":", alpha=0.4)
            ax.tick_params(axis="y", labelsize=18)
            ax.tick_params(axis="x", labelsize=18)

    y_top = min(1.02, global_max_box + max(0.02, 0.05 * global_max_box))
    for ax in axes.ravel():
        if ax.has_data():
            ax.set_ylim(0, y_top)

    fig.text(-0.01, 0.5, "Dice Similarity Coefficient", va="center", rotation=90, fontsize=24, fontweight="bold")
    legend_handles = [
        Patch(facecolor="#7CCBA2", edgecolor="grey", label="Point"),
        Patch(facecolor="#FCDE9C", edgecolor="grey", label="Box")
    ]
    fig.legend(
        legend_handles, ["Point", "Box"], loc="lower center",
        ncols=2, bbox_to_anchor=(0.5, -0.055), fontsize=20,
    )

    plt.tight_layout()
    plt.savefig("./fig_iterative_prompting_mask_ablation.png", dpi=600, bbox_inches="tight")
    plt.savefig("./fig_iterative_prompting_mask_ablation.svg", dpi=600, bbox_inches="tight")
    plt.close()


def _plot_iterative_prompting_n_iterations():
    present = []
    for name in ["sam", "medsam-self", "simplesam", "medicosam-neu"]:
        exp_dir = os.path.join(ROOT, name)
        if os.path.isdir(exp_dir):
            present.append((name, exp_dir))

    n = len(present)
    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(6*n, 6), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (ename, exp_dir) in zip(axes, present):
        use_mask = ename in ["sam", "medicosam-neu"]
        point, box = _load_point_box_series(exp_dir, use_mask)
        rel_box = box - point

        x = np.arange(32)
        ax.bar(x, point, width=0.8, color="#7CCBA2", edgecolor="grey", label="Point")
        ax.bar(x, rel_box, width=0.8, bottom=point, color="#FCDE9C", edgecolor="grey", label="Box")

        model_names = {
            "sam": "SAM",
            "medsam-self": "MedSAM*",
            "simplesam": "Simple-FT*",
            "medicosam-neu": r"$\bf{MedicoSAM*}$",
        }

        ax.set_title(model_names.get(ename, ename), fontsize=14)
        ax.set_xlim(-0.6, 31.6)
        ax.set_xticks(np.arange(0, 32, 2))
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_ylim(0, 1)

    axes[0].set_ylabel("Dice Similarity Coefficient")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.04))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("./test.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    _plot_iterative_prompting_use_mask()
    # _plot_iterative_prompting_n_iterations()


if __name__ == "__main__":
    main()
