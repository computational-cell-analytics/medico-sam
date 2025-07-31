import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/medico_sam/3d"

DATASETS = {
    "lgg_mri": "LGG MRI (Low-Grade Glioma in MRI)",
    "microusp": "MicroUSP (Prostate in Micro-Ultrasound)",
    "kits": "KiTS (Kidney Tumor in CT)",
    "duke_liver": "DLDS (Liver in MRI)",
    "segthy": "SegThy (Thyroid, Jugular Vein & Cartoid Artery and Vein in US)",
    "osic_pulmofib": "OSIC PulmoFib (Thoracic Organs in CT)",
}

MODEL_MAPS = {
    "sam": "SAM",
    "sam2.1": "SAM2 (2.1)",
    "medsam": "MedSAM",
    "simplesam": "SimpleFT",
    "medico-samv2-full": r"$\bf{MedicoSAM*}$",
}


def _get_sam2_results_per_dataset(dataset_name):
    res_list = []
    for bpath in glob(os.path.join(ROOT, "*")):
        backbone = os.path.basename(bpath)
        for res_path in glob(os.path.join(bpath, "hvit_b", dataset_name, "results", "**", "*.csv")):
            res = pd.read_csv(res_path)

            score_columns = [
                col for col in res.columns if col not in ["Unnamed: 0", "lung_1", "lung_2"]
            ]
            mean_score = res[score_columns].iloc[0].mean()

            res_dict = {
                "backbone": backbone,
                "prompt_choice": Path(res_path).stem.split("_")[-1],
                "type": "mean_score",
                "score": mean_score
            }
            res_list.append(pd.DataFrame.from_dict([res_dict]))

    res_df = pd.concat(res_list, ignore_index=True)
    return res_df


def _get_sam_results_per_dataset(dataset_name):
    res_list = []
    for res_path in glob(os.path.join(ROOT, "sam", "*", dataset_name, "results", "*.csv")):
        res = pd.read_csv(res_path)
        roi = res.columns[-1]

        pchoice = Path(res_path).stem.split("_")[-1]
        if pchoice == "points":
            pchoice = pchoice[:-1]

        res_dict = {
            "backbone": res_path.rsplit("/")[-4][6:],
            "prompt_choice": pchoice,
            "type": roi,
            "score": res.iloc[0][roi]
        }
        res_list.append(pd.DataFrame.from_dict([res_dict]))

    res_df = pd.concat(res_list, ignore_index=True)
    return res_df


def _get_plots():
    fig, ax = plt.subplots(2, 3, figsize=(32, 23))
    ax = ax.flatten()

    bar_width = 0.2
    scale = 1.2
    fontsize_base = int(16 * scale)
    fontsize_legend = int(16 * scale)
    fontsize_axis_label = int(20 * scale)

    for i, (dname, dmap) in enumerate(DATASETS.items()):
        df1 = _get_sam_results_per_dataset(dname)
        df2 = _get_sam2_results_per_dataset(dname)
        df = pd.concat([df1, df2], ignore_index=True)

        methods = [m for m in MODEL_MAPS.keys() if m != "sam"]
        labels = [MODEL_MAPS[m] for m in methods]

        scores = []
        for _method in methods + ["sam"]:
            base_res = df[df['backbone'] == _method]
            res = {"method": _method}

            res["box"] = base_res[base_res['prompt_choice'] == 'box']['score'].iloc[0] \
                if not base_res[base_res['prompt_choice'] == 'box'].empty else 0

            res["point"] = base_res[base_res['prompt_choice'] == 'point']['score'].iloc[0] \
                if not base_res[base_res['prompt_choice'] == 'point'].empty else 0

            scores.append(res)

        sam_scores = next((s for s in scores if s["method"] == "sam"), None)
        if sam_scores is None:
            raise ValueError("No SAM results found for comparison.")

        relative_scores = [
            {
                "method": s["method"],
                "box": s["box"] - sam_scores["box"],
                "point": s["point"] - sam_scores["point"]
            }
            for s in scores if s["method"] != "sam"
        ]

        x = [pos * 0.6 for pos in range(len(methods))]

        point_scores = [s.get("point") for s in relative_scores]
        box_scores = [s.get("box") for s in relative_scores]
        all_scores = point_scores + box_scores

        max_height = max(all_scores) if all_scores else 0
        min_height = min(all_scores) if all_scores else 0

        ax[i].bar(
            [pos - bar_width / 2 for pos in x], point_scores,
            width=bar_width, label='Point', color="#7CCBA2", edgecolor="grey",
        )

        ax[i].bar(
            [pos + bar_width / 2 for pos in x], box_scores,
            width=bar_width, label='Box', color="#FCDE9C", edgecolor="grey",
        )

        ax[i].axhspan(min_height, 0, color='lightcoral', alpha=0.2)
        ax[i].axhspan(0, max_height, color='lightgreen', alpha=0.2)

        ax[i].set_xticks(x)
        ax[i].set_xticklabels(labels, fontsize=fontsize_base)
        ax[i].tick_params(axis='y', labelsize=fontsize_base)
        ax[i].set_title(dmap, fontweight="bold", fontsize=fontsize_base)

        ax[i].axhline(0, color='black', linewidth=0.8, linestyle="--")

    for idx in range(len(DATASETS), len(ax)):
        ax[idx].axis("off")

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', fontsize=fontsize_legend, ncol=2)

    plt.text(
        x=-6, y=-0.175, s="Relative Dice Similarity Coefficient (compared to SAM)",
        rotation=90, fontweight="bold", fontsize=fontsize_axis_label
    )

    plt.subplots_adjust(top=0.95, bottom=0.055, right=0.95, left=0.05, hspace=0.125, wspace=0.1)
    plt.savefig("./fig_5_interactive_segmentation_3d_per_dataset.png", bbox_inches="tight")
    plt.savefig("./fig_5_interactive_segmentation_3d_per_dataset.svg", bbox_inches="tight")
    plt.close()


def _get_average_plots():
    method_scores = {method: {'box': [], 'point': []} for method in MODEL_MAPS.keys() if method != "simplesam"}

    for dname in DATASETS.keys():
        df1 = _get_sam_results_per_dataset(dname)
        df2 = _get_sam2_results_per_dataset(dname)
        df = pd.concat([df1, df2], ignore_index=True)

        for method in method_scores.keys():
            base_res = df[df['backbone'] == method]

            box_score = base_res[base_res['prompt_choice'] == 'box']['score'].iloc[0] \
                if not base_res[base_res['prompt_choice'] == 'box'].empty else 0
            method_scores[method]['box'].append(box_score)

            point_score = base_res[base_res['prompt_choice'] == 'point']['score'].iloc[0] \
                if not base_res[base_res['prompt_choice'] == 'point'].empty else 0
            method_scores[method]['point'].append(point_score)

    averaged_scores = []
    for method, scores in method_scores.items():
        avg_box = np.mean(scores['box'])
        avg_point = np.mean(scores['point'])
        averaged_scores.append({'method': method, 'box': avg_box, 'point': avg_point})

    fig, ax = plt.subplots(figsize=(20, 15))

    methods = [MODEL_MAPS[m] for m in MODEL_MAPS.keys() if m != "simplesam"]
    x = np.arange(len(methods))
    bar_width = 0.2

    avg_box_scores = [s['box'] for s in averaged_scores]
    avg_point_scores = [s['point'] for s in averaged_scores]

    ax.bar(
        x - bar_width / 2, avg_point_scores, width=bar_width, label='Point', color="#7CCBA2", edgecolor="grey"
    )
    ax.bar(
        x + bar_width / 2, avg_box_scores, width=bar_width, label='Box', color="#FCDE9C", edgecolor="grey"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=18)
    ax.set_ylabel("Dice Similarity Coefficient", fontsize=20, fontweight="bold")
    ax.set_title("Interactive Segmentation (3D)", fontsize=24, fontweight="bold")
    ax.tick_params(axis='y', labelsize=18)

    all_lines, all_labels = [], []
    for ax in fig.axes:
        lines, labels = ax.get_legend_handles_labels()
        for line, label in zip(lines, labels):
            if label not in all_labels:
                all_lines.append(line)
                all_labels.append(label)
        ax.legend().remove()

    fig.legend(all_lines, all_labels, loc="upper center", ncols=4, bbox_to_anchor=(0.22, 0.875), fontsize=18)

    plt.savefig("./fig_1b_interactive_segmentation_3d_average.png", bbox_inches="tight")
    plt.savefig("./fig_1b_interactive_segmentation_3d_average.svg", bbox_inches="tight")
    plt.close()


def main():
    _get_plots()
    _get_average_plots()


if __name__ == "__main__":
    main()
