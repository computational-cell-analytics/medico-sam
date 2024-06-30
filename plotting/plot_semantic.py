import numpy as np
import matplotlib.pyplot as plt

RESULTS = {
    "oimhs": {
        "sam-fft": [0.972, 0.987, 0.83, 0.87],
        "medicosam-fft": [0.977, 0.988, 0.843, 0.871],
        "medsam-fft": [0.976, 0.988, 0.84, 0.875],
        "simplesam-fft": [0.974, 0.988, 0.847, 0.873],
        "sam-lora": [0.947, 0.984, 0.802, 0.837],
        "medicosam-lora": [0.941, 0.979, 0.745, 0.83],
        "nnunet": [0.998, 0.993, 0.912, 0.876]
    },
    "dca1": {
        "sam-fft": [0.739],
        "medicosam-fft": [0.775],
        "medsam-fft": [0.793],
        "simplesam-fft": [0.782],
        "sam-lora": [0.747],
        "medicosam-lora": [0.725],
        "nnunet": [0.802],
    },
    "isic": {
        "sam-fft": [0.855],
        "medicosam-fft": [0.87],
        "medsam-fft": [0.879],
        "simplesam-fft": [0.885],
        "sam-lora": [0.885],
        "medicosam-lora": [0.876],
        "nnunet": [0.833],
    },
    "piccolo": {
        "sam-fft": [0.461],
        "medicosam-fft": [0.59],
        "medsam-fft": [0.764],
        "simplesam-fft": [0.755],
        "sam-lora": [0.601],
        "medicosam-lora": [0.466],
        "nnunet": [0.5],  # TODO
    },
    "drive": {
        "sam-fft": [0.739],
        "medicosam-fft": [0.787],
        "medsam-fft": [0.758],
        "simplesam-fft": [0.781],
        "sam-lora": [0.733],
        "medicosam-lora": [0.669],
        "nnunet": [0.5],  # TODO
    },
    "cbis_ddsm": {
        "sam-fft": [0.316],
        "medicosam-fft": [0.448],
        "medsam-fft": [0.477],
        "simplesam-fft": [0.423],
        "sam-lora": [0.429],
        "medicosam-lora": [0.429],
        "nnunet": [0.425],
    },
}


DATASET_MAPS = {
    "oimhs": "OIMHS (Macular Hole and Retinal Structures in OCT)",
    "isic": "ISIC (Skin Lesion in Dermoscopy)",
    "drive": "DRIVE (Vessel Segmentation in Fundus)",
    "piccolo": "PICCOLO (Polyps in Narrow Band Imaging)",
    "cbis_ddsm": "CBIS DDSM (Lesion Mass in Mammography)",
    "dca1": "DCA1 (Vessels in X-Ray Coronary Angiograms)",
}


MODEL_MAPS = {
    "sam-fft": "SAM",
    "medicosam-fft": "MedicoSAM",
    "simplesam-fft": "Simple FT*",
    "medsam-fft": "MedSAM",
    "sam-lora": "SAM\n(LoRA)",
    "medicosam-lora": "MedicoSAM\n(LoRA)",
    "nnunet": "nnU-Net",
}


def _make_per_dataset_plot():
    percentage_results = {}
    for dataset, methods in RESULTS.items():
        nnunet_scores = methods.pop("nnunet")
        percentage_results[dataset] = {}
        for method, scores in methods.items():
            if isinstance(scores, list):
                percentage_scores = [(s - nnunet_scores[i]) / nnunet_scores[i] * 100 for i, s in enumerate(scores)]
                percentage_results[dataset][method] = np.mean(percentage_scores)
            else:
                percentage_results[dataset][method] = (scores - nnunet_scores) / nnunet_scores * 100

    fig, axes = plt.subplots(2, 3, figsize=(30, 15))
    axes = axes.flatten()

    for ax, (dataset, methods) in zip(axes, percentage_results.items()):
        methods_list = ["sam-fft", "sam-lora", "medsam-fft", "simplesam-fft", "medicosam-fft", "medicosam-lora"]
        percentage_scores = [methods[_method] for _method in methods_list]

        ax.bar(methods_list, percentage_scores, color="#F0746E", edgecolor="grey")
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

        min_val = min(percentage_scores)
        max_val = max(percentage_scores)

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
        x=-15.5, y=-13, s="(%) Relative Dice Similarity Coefficient (compared to nnU-Net)",
        rotation=90, fontweight="bold", fontsize=16
    )

    plt.subplots_adjust(hspace=0.4, wspace=0.1)
    plt.savefig("./figure_4_semantic_segmentation_per_dataset.png")
    plt.savefig("./figure_4_semantic_segmentation_per_dataset.svg")
    plt.close()


def _plot_absolute_mean_per_experimet():
    method_sums = {}
    method_counts = {}
    for methods in RESULTS.values():
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

    methods = ["nnunet", "sam-fft", "sam-lora", "medsam-fft", "simplesam-fft", "medicosam-fft", "medicosam-lora"]
    means = [absolute_means[_method] for _method in methods]

    fig, ax = plt.subplots(figsize=(15, 10))

    ax.bar(methods, means, color="#F0746E")

    ax.set_xticks(np.arange(len(methods)))
    _xticklabels = [MODEL_MAPS[_exp] for _exp in methods]
    ax.set_xticklabels(_xticklabels, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_ylabel('Dice Similarity Coefficient', fontsize=16, fontweight="bold")

    plt.savefig("./fig_1_semantic_segmentation_average.png")
    plt.savefig("./fig_1_semantic_segmentation_average.svg")
    plt.close()


def main():
    # _make_per_dataset_plot()
    _plot_absolute_mean_per_experimet()


main()
