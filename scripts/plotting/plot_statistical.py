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

    fig, ax = plt.subplots(figsize=(10, 6))

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
    fig.savefig("./fig_stats_interactive_segmentation.png", dpi=600, bbox_inches="tight")
    fig.savefig("./fig_stats_interactive_segmentation.svg", dpi=600, bbox_inches="tight")


def _semantic_seg_stats():
    base_dir = "/mnt/vast-nhr/projects/cidas/cca/models/semantic_sam/v3"

    def _get_res(set_dir, method, decoder=False):
        res_path = os.path.join(
            set_dir, "full_finetuning", method, "inference", "abus",
            ("w_decoder" if decoder else "wo_decoder"),
            "results", "tumor", "semantic_segmentation.csv"
        )
        df = pd.read_csv(res_path)
        return float(df["dice"].iloc[0])

    sam_vals, medsam_vals = [], []
    medico_wo_vals, medico_w_vals = [], []

    for set_dir in sorted(glob(os.path.join(base_dir, "*"))):
        sam_vals.append(_get_res(set_dir, "sam"))
        medsam_vals.append(_get_res(set_dir, "medsam"))
        medico_wo_vals.append(_get_res(set_dir, "medico-samv2-full", decoder=False))
        medico_w_vals.append(_get_res(set_dir, "medico-samv2-full", decoder=True))

    labels = ["SAM", "MedSAM", r"$\mathbf{MedicoSAM*}$", r"$\mathbf{MedicoSAM*}_{\mathrm{Dec}}$"]
    stacks = [
        np.asarray(sam_vals),
        np.asarray(medsam_vals),
        np.asarray(medico_wo_vals),
        np.asarray(medico_w_vals),
    ]
    means = np.array([arr.mean() for arr in stacks])
    stds = np.array([arr.std(ddof=1) if len(arr) > 1 else 0.0 for arr in stacks])
    ns = np.array([len(arr) for arr in stacks])

    base_hex = "#045275"

    def hex_to_rgb01(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))

    def rgb01_to_hex(rgb):
        return "#{:02X}{:02X}{:02X}".format(*(int(round(c*255)) for c in rgb))

    def mix_with_white(hex_color, factor):
        r, g, b = hex_to_rgb01(hex_color)
        rw, gw, bw = 1.0, 1.0, 1.0
        return rgb01_to_hex(
            (r*(1-factor)+rw*factor, g*(1-factor)+gw*factor, b*(1-factor)+bw*factor)
        )

    order_by_perf = np.argsort(-means)
    ranks = np.empty_like(order_by_perf)
    ranks[order_by_perf] = np.arange(len(means))

    light_factors = {0: 0.00, 1: 0.20, 2: 0.40, 3: 0.60}
    bar_colors = [mix_with_white(base_hex, light_factors[int(r)]) for r in ranks]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=6, ecolor="black", color=bar_colors, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Dice Score Coefficient", fontweight="bold")
    ax.set_title("Semantic Segmentation (Breast Tumor in Ultrasound)", fontweight="bold")

    summary_df = pd.DataFrame({"method": labels, "mean": means, "std": stds, "n": ns})
    print(summary_df.to_string(index=False))

    fig.tight_layout()
    fig.savefig("./fig_stats_semantic_segmentation.png", dpi=600, bbox_inches="tight")
    fig.savefig("./fig_stats_semantic_segmentation.svg", dpi=600, bbox_inches="tight")


def main():
    _interactive_seg_stats()
    _semantic_seg_stats()


if __name__ == "__main__":
    main()
