import os
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


NNUNET_RESULTS = {
    # 2d
    "oimhs": [0.9897, 0.8752,  0.8419, 0.9966],  # {'choroid': 0.9899565409670795, 'retina': 0.9965755988640617, 'intraretinal_cysts': 0.8763200710233164, 'macular_hole': 0.8537020264548896}
    "isic": [0.8443],  # {'skin_lesion': 0.8404070538263777}
    "dca1": [0.7984],  # {'vessel': 0.8003136169445781}
    "cbis_ddsm": [0.4281],  # {'mass': 0.4328963546173533}
    "drive": [0.8143],  # {'veins': 0.8141129617943484}
    "piccolo": [0.6507],  # {'polyps': 0.6748682908069755}
    "siim_acr": [0.5621],  # {'pneumothorax': 0.5743679009079821}
    "hil_toothseg": [0.8911],  # {'teeth': 0.8921318959526281}
    "covid_qu_ex": [0.9799],  # {'lung': 0.9799401018869183}
    # 3d
    "curvas": [],  # {'kidney': 0.8839940579359213, 'liver': 0.9643119099900705, 'pancreas': 0.9604672064032412}
    "osic_pulmofib": [0.5356, 0.8832, 0.7914],  # {'heart': 0.4984178627020397, 'lung': 0.8858154314548179, 'trachea': 0.7850454301483626}
    "sega": [0.7872],  # {'aorta': 0.7675813642187753}
    "duke_liver": [0.911],  # {'liver': 0.9116960073750964}
    "oasis": [0.9519, 0.9689, 0.9773, 0.9655],  # {'gray matter': 0.9518826368795583, 'thalamus': 0.9689392350715701, 'white matter': 0.9773897972016621, 'csf': 0.9656624999845734}
    "lgg_mri": [0.8855],  # {'glioma': 0.887488830298482}
    "leg_3d_us": [0.8947, 0.9059, 0.8887],  # {'SOL': 0.8943315824577803, 'GM': 0.9059994338568392, 'GL': 0.8865953259705519}
    "micro_usp": [0.8605],  # {'prostate': 0.8401653361752514}
}


DATASET_MAPS = {
    "oimhs": "OIMHS (Macular Hole and Retinal Structures in OCT)",
    "isic": "ISIC (Skin Lesion in Dermoscopy)",
    "dca1": "DCA1 (Veins in X-Ray Coronary Angiograms)",
    "cbis_ddsm": "CBIS DDSM (Lesion Mass in Mammography)",
    "drive": "DRIVE (Vessel Segmentation in Fundus)",
    "piccolo": "PICCOLO (Polyps in Narrow Band Imaging)",
    "siim_acr": "SIIM ACR (Pneumothorax in Chest X-Ray)",
    "hil_toothseg": "HIL ToothSeg (Teeth in Panoramic Dental Radiographs)",
    "covid_qu_ex": "COVID QU Ex (Lungs in Infected Chest X-Ray)",
    "osic_pulmofib": "OSIC PulmoFib",
    "duke_liver": "Duke Liver",
    "toothfairy": "Toothfairy",
    "lgg_mri": "LGG MRI",
    "micro_usp": "MicroUSP",
}


MODEL_MAPS = {
    "full/sam": "SAM",
    "lora/sam": "SAM\n(LoRA)",
    "full/medico-sam-8g": "MedicoSAM",
    "lora/medico-sam-8g": "MedicoSAM\n(LoRA)",
    "full/medsam": "MedSAM",
    "lora/medsam": "MedSAM\n(LoRA)",
    "full/simplesam": "Simple FT",
    "lora/simplesam": "Simple FT\n(LoRA)",
}

MULTICLASS_DATASETS = ["oimhs"]

ROOT = "/mnt/vast-nhr/projects/cidas/cca/models/semantic_sam"


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
    percentage_results = {}
    for dataset, nnunet_scores in NNUNET_RESULTS.items():
        scores = get_results(dataset)
        percentage_results[dataset] = {}
        for df_val in scores.iloc:
            name = df_val["name"]
            dice = df_val["dice"]
            percentage_scores = [(s - nnunet_scores[i]) / nnunet_scores[i] * 100 for i, s in enumerate(dice)]
            percentage_results[dataset][name] = np.mean(percentage_scores)

    fig, axes = plt.subplots(3, 4, figsize=(35, 20))
    axes = axes.flatten()

    for ax, (dataset, methods) in zip(axes, percentage_results.items()):
        methods_list = [
            "full/sam", "lora/sam", "full/medsam", "lora/medsam", "full/simplesam",
            "lora/simplesam", "full/medico-sam-8g", "lora/medico-sam-8g",
        ]
        percentage_scores = [methods[_method] for _method in methods_list]

        ax.bar(methods_list, percentage_scores, color="#F0746E", edgecolor="grey")
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

        min_val, max_val = min(percentage_scores), max(percentage_scores)

        if min_val < 0:
            ax.axhspan(min_val, 0, facecolor='lightcoral', alpha=0.2)
        if max_val > 0:
            ax.axhspan(0, max_val, facecolor='lightgreen', alpha=0.2)

        ax.set_ylim([min_val - 10, max_val + 10])
        _xticklabels = [MODEL_MAPS[_exp] for _exp in methods_list]
        ax.set_xticks(np.arange(len(methods_list)))
        ax.set_xticklabels(_xticklabels, rotation=45, fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_title(f'{DATASET_MAPS[dataset]}', fontsize=14)

    plt.text(
        x=-30.1, y=5, s="(%) Relative Dice Similarity Coefficient (compared to nnU-Net)",
        rotation=90, fontweight="bold", fontsize=16
    )

    plt.subplots_adjust(hspace=0.4, wspace=0.1)
    plt.savefig("./fig_4_semantic_segmentation_per_dataset.png")
    plt.savefig("./fig_4_semantic_segmentation_per_dataset.svg")
    plt.close()


def _plot_absolute_mean_per_experiment():
    method_sums = {}
    method_counts = {}
    for methods in NNUNET_RESULTS.values():
        for method, scores in methods.items():
            if isinstance(scores, list):
                mean_score = np.mean(scores)
            else:
                mean_score = scores
            if method not in method_sums:
                method_sums[method] = 0
                method_counts[method] = 0
            method_sums[method] += mean_score
            method_counts[method] += 1

    absolute_means = {method: method_sums[method] / method_counts[method] for method in method_sums}

    methods = [
        "nnunet", "sam-fft", "sam-lora", "medsam-fft", "medsam-lora",
        "simplesam-fft", "simplesam-lora", "medicosam-fft", "medicosam-lora"
    ]
    means = [absolute_means[_method] for _method in methods]

    fig, ax = plt.subplots(figsize=(20, 10))

    bars = ax.bar(methods, means, color="#F0746E", edgecolor="grey")

    ax.set_xticks(np.arange(len(methods)))
    _xticklabels = [MODEL_MAPS[_exp] for _exp in methods]
    ax.set_xticklabels(_xticklabels, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_ylabel('Dice Similarity Coefficient', fontsize=16, fontweight="bold")

    # NOTE: adds values on top of each bar
    for bar, mean in zip(bars, means):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(mean, 3), ha='center', va='bottom', fontsize=14)

    plt.savefig("./fig_1_semantic_segmentation_average.png")
    plt.savefig("./fig_1_semantic_segmentation_average.svg")
    plt.close()


def main():
    _make_per_dataset_plot()
    # _plot_absolute_mean_per_experiment()


if __name__ == "__main__":
    main()
