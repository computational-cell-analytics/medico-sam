import numpy as np
import matplotlib.pyplot as plt

RESULTS = {
    "oimhs": {
        "sam-fft": [0.9803, 0.9898, 0.862, 0.8767],
        "sam-lora": [0.9684, 0.9862, 0.8148, 0.8552],
        "medicosam-fft": [0.9793, 0.9879, 0.8395, 0.8725],
        "medicosam-lora": [0.9607, 0.9837, 0.79, 0.8372],
        "medsam-fft": [0.9742, 0.9871, 0.837, 0.8661],
        "medsam-lora": [0.9319, 0.9785, 0.7309, 0.8039],
        "simplesam-fft": [0.9797, 0.9882, 0.8401, 0.8762],
        "simplesam-lora": [0.9536, 0.9812, 0.7752, 0.8315],
        "nnunet": [0.998, 0.993, 0.912, 0.876]
    },
    "dca1": {
        "sam-fft": [0.7982],
        "sam-lora": [0.7884],
        "medicosam-fft": [0.7933],
        "medicosam-lora": [0.7809],
        "medsam-fft": [0.7762],
        "medsam-lora": [0.7652],
        "simplesam-fft": [0.7932],
        "simplesam-lora": [0.7753],
        "nnunet": [0.802],
    },
    "isic": {
        "sam-fft": [0.8817],
        "sam-lora": [0.8906],
        "medicosam-fft": [0.9024],
        "medicosam-lora": [0.8937],
        "medsam-fft": [0.8985],
        "medsam-lora": [0.8812],
        "simplesam-fft": [0.8908],
        "simplesam-lora": [0.8863],
        "nnunet": [0.833],
    },
    "drive": {
        "sam-fft": [0.7936],
        "sam-lora": [0.7903],
        "medicosam-fft": [0.7553],
        "medicosam-lora": [0.7136],
        "medsam-fft": [0.6413],
        "medsam-lora": [0.6799],
        "simplesam-fft": [0.7315],
        "simplesam-lora": [0.7146],
        "nnunet": [0.814],
    },
    "piccolo": {
        "sam-fft": [0.835],
        "sam-lora": [0.7185],
        "medicosam-fft": [0.7505],
        "medicosam-lora": [0.7561],
        "medsam-fft": [0.7631],
        "medsam-lora": [0.6499],
        "simplesam-fft": [0.7535],
        "simplesam-lora": [0.7264],
        "nnunet": [0.6868],
    },
    "cbis_ddsm": {
        "sam-fft": [0.5201],
        "sam-lora": [0.57],
        "medicosam-fft": [0.5197],
        "medicosam-lora": [0.5154],
        "medsam-fft": [0.5338],
        "medsam-lora": [0.4755],
        "simplesam-fft": [0.5431],
        "simplesam-lora": [0.4745],
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
    "sam-lora": "SAM\n(LoRA)",
    "medicosam-fft": "MedicoSAM",
    "medicosam-lora": "MedicoSAM\n(LoRA)",
    "medsam-fft": "MedSAM",
    "medsam-lora": "MedSAM\n(LoRA)",
    "simplesam-fft": "Simple FT",
    "simplesam-lora": "Simple FT\n(LoRA)",
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
        methods_list = [
            "sam-fft", "sam-lora", "medsam-fft", "medsam-lora", "simplesam-fft",
            "simplesam-lora", "medicosam-fft", "medicosam-lora"
        ]
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
    # _make_per_dataset_plot()
    _plot_absolute_mean_per_experimet()


main()
