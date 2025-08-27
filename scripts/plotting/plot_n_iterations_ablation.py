import os
from glob import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


def _score_at_iteration(df: pd.DataFrame, it: int) -> float:
    if "dice" not in df.columns:
        raise ValueError("Expected 'dice' column in CSV.")

    if "Unnamed: 0" in df.columns:
        if (df["Unnamed: 0"].astype(int) == it).any():
            return float(df.loc[df["Unnamed: 0"].astype(int) == it, "dice"].iloc[0])

    if len(df) <= it:
        raise ValueError(f"Dataframe has only {len(df)} rows; cannot fetch iteration {it}.")

    return float(df["dice"].iloc[it])


def _load_point_box_iter(exp_dir: str, use_mask: bool, it: int) -> tuple[float, float]:
    mask_dir = "with_mask" if use_mask else "without_mask"
    base = os.path.join(exp_dir, mask_dir, "results", "polyp")
    p_csv = os.path.join(base, "iterative_prompts_start_point.csv")
    b_csv = os.path.join(base, "iterative_prompts_start_box.csv")

    p_df = pd.read_csv(p_csv)
    b_df = pd.read_csv(b_csv)
    return _score_at_iteration(p_df, it), _score_at_iteration(b_df, it)


def _plot_iterative_prompting_use_mask():
    experiments = []
    for exp_dir in glob(os.path.join(ROOT, "*")):
        ename = os.path.basename(exp_dir)
        if ename in ["sam", "medicosam-old", "medicosam-neu"]:
            experiments.append((ename, exp_dir))

    table = {}
    for ename, exp_dir in experiments:
        for use_mask in (True, False):
            p0, b0 = _load_point_box_iter(exp_dir, use_mask, it=0)
            p7, b7 = _load_point_box_iter(exp_dir, use_mask, it=7)
            table[(ename, use_mask)] = {"p0": p0, "b0": b0, "p7": p7, "b7": b7}

    ref_key = ("sam", True)

    ref = table[ref_key]
    groups, rel_vals = [], {k: [] for k in BAR_KEYS}

    for key in GROUP_ORDER:
        if key not in table:
            continue

        groups.append(LABELS.get(key, f"{key[0]} {'w/' if key[1] else 'w/o'} mask"))
        vals = table[key]
        rel_vals["p0"].append(vals["p0"] - ref["p0"])
        rel_vals["b0"].append(vals["b0"] - ref["b0"])
        rel_vals["p7"].append(vals["p7"] - ref["p7"])
        rel_vals["b7"].append(vals["b7"] - ref["b7"])

    fig, ax = plt.subplots(figsize=(10, 9))
    n_groups = len(groups)
    n_bars = len(BAR_KEYS)
    bar_width = 0.18

    group_x = np.arange(n_groups)
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * bar_width

    for i, (k, label, color) in enumerate(zip(BAR_KEYS, BAR_LABELS, BAR_COLORS)):
        x = group_x + offsets[i]
        ax.bar(x, rel_vals[k], width=bar_width, edgecolor="grey", label=label, color=color)

    all_vals = np.concatenate([np.array(rel_vals[k]) for k in BAR_KEYS]) if n_groups else np.array([0.0])
    max_val = float(np.nanmax(all_vals)) if all_vals.size else 0.0
    min_val = float(np.nanmin(all_vals)) if all_vals.size else 0.0
    ax.axhspan(0, max_val, facecolor="lightgreen", alpha=0.2)
    ax.axhspan(min_val, 0, facecolor="lightcoral", alpha=0.2)

    ax.set_xticks(group_x)
    ax.set_xticklabels(groups, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylabel(
        "Relative Dice Similarity Score (compared to " + r"SAM$_{\mathrm{Mask}}$" + ")", fontweight="bold", fontsize=16
    )
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.legend(ncols=4, loc="upper center")
    plt.tight_layout()

    plt.savefig("./fig_iterative_prompting_mask_ablation.png", dpi=600, bbox_inches="tight")
    plt.savefig("./fig_iterative_prompting_mask_ablation.svg", dpi=600, bbox_inches="tight")
    plt.close()


def _read_series(csv_path):
    df = pd.read_csv(csv_path)
    it_col = "Unnamed: 0" if "Unnamed: 0" in df.columns else df.columns[0]
    it = df[it_col].astype(int)
    ser = pd.Series(df["dice"].values, index=it).sort_index()
    ser = ser.reindex(range(32))
    return ser


def _load_point_box_series(exp_dir, use_mask):
    base = os.path.join(exp_dir, ("with_mask" if use_mask else "without_mask"), "results", "polyp")
    p_csv = os.path.join(base, "iterative_prompts_start_point.csv")
    b_csv = os.path.join(base, "iterative_prompts_start_box.csv")

    if not os.path.isfile(p_csv):
        raise FileNotFoundError(f"Missing file: {p_csv}")

    if not os.path.isfile(b_csv):
        raise FileNotFoundError(f"Missing file: {b_csv}")

    p_ser = _read_series(p_csv)
    b_ser = _read_series(b_csv)

    return p_ser, b_ser


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
    # _plot_iterative_prompting_use_mask()
    _plot_iterative_prompting_n_iterations()


if __name__ == "__main__":
    main()
