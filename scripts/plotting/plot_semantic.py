import os
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


NNUNET_RESULTS = {
    # 2d
    "oimhs": [0.9899, 0.8763,  0.8537, 0.9966],
    "isic": [0.8404],
    "dca1": [0.8003],
    "cbis_ddsm": [0.4329],
    "piccolo": [0.6749],
    "hil_toothseg": [0.8921],

    # 3d
    "osic_pulmofib": [0.4984, 0.8858, 0.7850],
    "leg_3d_us": [0.8943, 0.9059, 0.8865],
    "oasis": [0.9519, 0.9689, 0.9773, 0.9656],
    "micro_usp": [0.8402],
    "lgg_mri": [0.8875],
    "duke_liver": [0.9117],
}

DATASET_MAPS = {
    # 2d
    "oimhs": "OIMHS (Macular Hole and Retinal Structures in OCT)",
    "isic": "ISIC (Skin Lesion in Dermoscopy)",
    "dca1": "DCA1 (Vessels in X-Ray Coronary Angiograms)",
    "cbis_ddsm": "CBIS DDSM (Lesion Mass in Mammography)",
    "piccolo": "PICCOLO (Polyps in Narrow Band Imaging)",
    "hil_toothseg": "HIL ToothSeg (Teeth in Panoramic Dental Radiographs)",
    # 3d
    "osic_pulmofib": "OSIC PulmoFib (Thoracic Organs in CT)",
    "leg_3d_us": "LEG 3D US (Leg Muscles in Ultrasound)",
    "oasis": "OASIS (Brain Tissue in MRI)",
    "micro_usp": "MicroUSP (Prostate in Micro-Ultrasound)",
    "lgg_mri": "LGG MRI (Low-Grade Glioma in Brain MRI)",
    "duke_liver": "Duke Liver (Liver in MRI)",
}

DATASETS_2D = ["oimhs", "isic", "dca1", "cbis_ddsm", "piccolo", "hil_toothseg"]

DATASETS_3D = ["osic_pulmofib", "leg_3d_us", "oasis", "micro_usp", "lgg_mri", "duke_liver"]

MODEL_MAPS = {
    "nnunet": "nnUNet",
    "full/sam": "SAM",
    "lora/sam": "SAM\n(LoRA)",
    "full/medico-samv2-full": "MedicoSAM",
    "lora/medico-samv2-full": "MedicoSAM\n(LoRA)",
    "full/medsam": "MedSAM",
    "lora/medsam": "MedSAM\n(LoRA)",
    "full/simplesam": "Simple FT",
    "lora/simplesam": "Simple FT\n(LoRA)",
}

ROOT = "/mnt/vast-nhr/projects/cidas/cca/models/semantic_sam/v2"


def get_results(dataset_name):
    all_res, all_comb_names = [], []
    for rpath in sorted(glob(os.path.join(ROOT, "*", "*", "inference", dataset_name, "results", "**", "*.csv"))):
        psplits = rpath[len(ROOT) + 1:].rsplit("/")
        ft_name, mname = psplits[0], psplits[1]
        ft_name = ft_name.split("_")[0]

        if mname == "medico-sam-1g":  # HACK: we do not get results for medico-sam trained on 1 GPU.
            continue

        res = pd.read_csv(rpath)
        combination_name = f"{ft_name}/{mname}"
        score = res.iloc[0]["dice"]
        if f"{ft_name}_{mname}" in all_comb_names:
            idx = all_comb_names.index(f"{ft_name}_{mname}")
            all_res[idx].at[0, "dice"].append(score)
        else:
            all_res.append(pd.DataFrame.from_dict([{"name": combination_name, "dice": [score]}]))
            all_comb_names.append(f"{ft_name}_{mname}")

    all_res = pd.concat(all_res, ignore_index=True)
    return all_res


def _make_per_dataset_plot():
    results = {}
    for dataset, nnunet_scores in NNUNET_RESULTS.items():
        scores = get_results(dataset)
        results[dataset] = {"nnunet": np.mean(nnunet_scores)}
        for df_val in scores.iloc:
            name = df_val["name"]
            dice = df_val["dice"]
            results[dataset][name] = np.mean(dice)

    fig, axes = plt.subplots(4, 3, figsize=(35, 30))
    axes = axes.flatten()

    top_colors = ["#045275", "#2B6C8F", "#5093A9"]
    default_color = "#D3D3D3"

    for ax, (dataset, methods) in zip(axes, results.items()):
        methods_list = [
            "full/sam", "lora/sam",
            "full/medsam", "lora/medsam",
            "full/simplesam", "lora/simplesam",
            "full/medico-samv2-full", "lora/medico-samv2-full",
        ]
        scores, neu_methods_list = [], []
        for _method in methods_list:
            if _method in methods:
                scores.append(methods[_method])
                neu_methods_list.append(_method)

        sorted_indices = np.argsort(scores)[::-1]
        bar_colors = [default_color] * len(scores)
        edge_colors = ["none"] * len(scores)
        edge_styles = ["solid"] * len(scores)

        for rank, idx in enumerate(sorted_indices[:3]):
            bar_colors[idx] = top_colors[rank]
            edge_colors[idx] = "none"

        if "full/medico-samv2-full" in neu_methods_list:
            index = neu_methods_list.index("full/medico-samv2-full")
            if index not in sorted_indices[:3]:
                edge_colors[index] = "black"
                edge_styles[index] = "dashed"
            else:
                edge_colors[index] = "none"

        bars = ax.bar(
            neu_methods_list, scores, color=bar_colors, edgecolor=edge_colors, linewidth=1.5
        )

        for bar, style in zip(bars, edge_styles):
            if style == "dashed":
                bar.set_linestyle("--")
                bar.set_linewidth(3)

        ax.axhline(methods.get("nnunet"), color="#DC3977", linewidth=4)

        ax.set_ylim([0, 1])
        _xticklabels = [MODEL_MAPS[_exp] for _exp in neu_methods_list]
        ax.set_xticks(np.arange(len(neu_methods_list)))
        ax.set_xticklabels(_xticklabels, rotation=45, fontsize=18)
        ax.tick_params(axis='y', labelsize=14)

        for label, method in zip(ax.get_xticklabels(), neu_methods_list):
            if method == "full/medico-sam-8g":
                label.set_fontweight("bold")

        fontdict = {"fontsize": 18}
        if dataset in ["oimhs", "isic", "dca1", "cbis_ddsm", "piccolo", "hil_toothseg"]:
            fontdict["fontstyle"] = "italic"
        else:
            fontdict["fontweight"] = "bold"

        ax.set_title(f'{DATASET_MAPS[dataset]}', fontdict=fontdict)
        ax.title.set_color("#212427")

    plt.text(
        x=-20.5, y=2.1, s="Dice Similarity Coefficient", rotation=90, fontweight="bold", fontsize=20,
    )

    plt.subplots_adjust(hspace=0.45, wspace=0.1)
    plt.savefig("./fig_4_semantic_segmentation_per_dataset.png", bbox_inches="tight")
    plt.savefig("./fig_4_semantic_segmentation_per_dataset.svg", bbox_inches="tight")
    plt.close()


def _plot_absolute_mean_per_experiment(dim):
    methods = [
        "nnunet",
        "lora/sam", "full/sam",
        "lora/medsam", "full/medsam",
        "lora/simplesam", "full/simplesam",
        "lora/medico-samv2-full", "full/medico-samv2-full",
    ]

    results = {}
    for dataset, nnunet_scores in NNUNET_RESULTS.items():
        if dim == "3d" and dataset not in DATASETS_3D:
            continue

        if dim == "2d" and dataset not in DATASETS_2D:
            continue

        scores = get_results(dataset)
        for method in methods:
            if method == "nnunet":
                res = np.mean(nnunet_scores)
            else:
                res = scores.loc[scores["name"] == method].iloc[0]["dice"]
                res = np.mean(res)

            if method in results:
                results[method] = np.mean([results[method], res])
            else:
                results[method] = res

    fig, ax = plt.subplots(figsize=(22, 15))

    top_colors = ["#045275", "#2B6C8F", "#5093A9"]
    sorted_methods = sorted(results, key=results.get, reverse=True)
    top_methods = sorted_methods[:3]  # get the top 3 methods.

    means = [results[_method] for _method in methods]

    edgecolors = ["None" if method in top_methods else "grey" for method in methods]

    bars = ax.bar(
        methods, means,
        edgecolor=edgecolors,
        linewidth=1.5,
        color=[top_colors[top_methods.index(_method)] if _method in top_methods else "#D3D3D3" for _method in methods],
    )

    for bar, method in zip(bars, methods):
        if method == "full/medico-sam-8g" and method not in top_methods:
            bar.set_edgecolor("black")
            bar.set_linestyle("--")
            bar.set_linewidth(3)

    ax.set_ylim([0, 1])
    ax.set_xticks(np.arange(len(methods)))
    _xticklabels = [MODEL_MAPS[_exp] for _exp in methods]
    ax.set_xticklabels(_xticklabels, fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_ylabel('Dice Similarity Coefficient', fontsize=20, fontweight="bold")

    # NOTE: adds values on top of each bar
    for bar, mean, method in zip(bars, means, methods):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, round(mean, 4),
            ha='center', va='bottom', fontsize=18,
            color="black" if method in top_methods else "#696969",
            fontweight="bold" if method in top_methods else "normal",
        )

    for label, method in zip(ax.get_xticklabels(), methods):
        if method == "full/medico-sam-8g":
            label.set_fontweight("bold")

    plt.title(f"Semantic Segmentation {dim.upper()}", fontsize=24, fontweight="bold")
    plt.savefig(f"./fig_1b_semantic_segmentation_{dim}_average.png", bbox_inches="tight")
    plt.savefig(f"./fig_1b_semantic_segmentation_{dim}_average.svg", bbox_inches="tight")
    plt.close()


def main():
    # For figure 4
    _make_per_dataset_plot()

    # For figure 1
    # _plot_absolute_mean_per_experiment(dim="2d")
    # _plot_absolute_mean_per_experiment(dim="3d")


if __name__ == "__main__":
    main()
