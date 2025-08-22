import os
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


MODEL_MAPS = {
    "vanilla": "SAM",
    "generalistv2-full": r"$\bf{MedicoSAM*}$",
    "medsam": "MedSAM",
}


def _concat_df(all_res):
    df_concat = pd.concat([df["dice"] for df in all_res], axis=1)
    df_stats = pd.DataFrame({"mean": df_concat.mean(axis=1), "std": df_concat.std(axis=1)})
    df_stats.insert(0, "Unnamed: 0", all_res[0]["Unnamed: 0"])
    return df_stats


def _interactive_seg_stats():
    base_dir = "/mnt/vast-nhr/projects/cidas/cca/experiments/v4"

    results = {}
    for exp_dir in sorted(glob(os.path.join(base_dir, "*"))):
        curr_exp = os.path.basename(exp_dir)

        all_box, all_point = [], []
        for i in range(5):
            res_dir = os.path.join(exp_dir, f"abus_{i}", "vit_b", "results", "tumor")
            all_box.append(pd.read_csv(os.path.join(res_dir, "iterative_prompts_start_box.csv")))
            all_point.append(pd.read_csv(os.path.join(res_dir, "iterative_prompts_start_point.csv")))

        box_stats = _concat_df(all_box)
        point_stats = _concat_df(all_point)

        results[curr_exp] = {
            "steps":  box_stats["Unnamed: 0"].tolist(),
            "p_mean": point_stats["mean"].to_numpy(),
            "p_std":  point_stats["std"].to_numpy(),
            "b_mean": box_stats["mean"].to_numpy(),
            "b_std":  box_stats["std"].to_numpy(),
        }

    desired_order = ["vanilla", "medsam", "generalistv2-full"]
    exps = [e for e in desired_order if e in results] + [e for e in results.keys() if e not in desired_order]
    if not exps:
        print("No experiments found.")
        return

    n_exps = len(exps)
    steps = results[exps[0]]["steps"]
    n_steps = len(steps)

    bar_width = 0.08
    inner_gap = 0.02
    block_width = n_steps * bar_width + (n_steps - 1) * inner_gap
    group_gap = 0.25
    exp_centers = np.arange(n_exps) * (block_width + group_gap)

    fig, ax = plt.subplots(figsize=(max(10, 3 + 3*n_exps), 6))

    point_base = "#7CCBA2"
    box_base = "#FCDE9C"
    point_last = "#045275"
    box_last = "#90477F"
    err_color = "black"

    y_top = 0.0
    all_iter_positions, all_iter_labels = [], []

    for ei, exp in enumerate(exps):
        data = results[exp]
        p_mean, p_std = data["p_mean"], data["p_std"]
        b_mean, b_std = data["b_mean"], data["b_std"]
        delta_mean = b_mean - p_mean

        left = exp_centers[ei] - block_width / 2.0
        offsets = left + np.arange(n_steps) * (bar_width + inner_gap)

        point_colors = [point_base] * n_steps
        box_colors = [box_base] * n_steps
        point_colors[-1] = point_last
        box_colors[-1] = box_last

        ax.bar(offsets, p_mean, width=bar_width, color=point_colors, zorder=1)
        ax.bar(offsets, delta_mean, width=bar_width, bottom=p_mean, color=box_colors, zorder=1)

        ax.errorbar(offsets, p_mean, yerr=p_std, fmt="none", capsize=4, elinewidth=1.2, zorder=3, ecolor=err_color)
        ax.errorbar(offsets, b_mean, yerr=b_std, fmt="none", capsize=4, elinewidth=1.2, zorder=3, ecolor=err_color)

        all_iter_positions.extend(offsets)
        all_iter_labels.extend([str(s) for s in steps])

        y_top = max(y_top, np.max(b_mean + b_std))

    ax.set_xticks(all_iter_positions)
    ax.set_xticklabels(all_iter_labels, fontsize=8)

    secax = ax.secondary_xaxis(-0.04, functions=(lambda x: x, lambda x: x))
    secax.set_xticks(exp_centers)
    secax.set_xticklabels([MODEL_MAPS.get(exp, exp) for exp in exps], fontsize=11)
    secax.set_xlabel("")
    secax.spines["bottom"].set_visible(False)
    secax.tick_params(axis="x", length=0)

    ax.set_ylabel("Dice Score Coefficient", fontweight="bold")
    ax.set_title("Interactive Segmentation (Breast Tumor in Ultrasound)", fontweight="bold")

    upper = max(1.0, y_top * 1.05)
    ax.set_ylim(0, min(1.05, upper))

    legend_handles = [
        Patch(facecolor=point_base, label="Point"),
        Patch(facecolor=box_base, label="Box"),
        Patch(facecolor=point_last, label=r"I$_{P}$"),
        Patch(facecolor=box_last, label=r"I$_{B}$"),
    ]
    fig.legend(
        legend_handles, [h.get_label() for h in legend_handles], loc="lower center",
        ncols=4, bbox_to_anchor=(0.525, -0.05), fontsize=11,
    )
    fig.tight_layout()
    fig.savefig("./test.png", dpi=600, bbox_inches="tight")
    plt.show()


def main():
    _interactive_seg_stats()


if __name__ == "__main__":
    main()
