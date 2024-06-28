import os
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = "/scratch/share/cidas/cca/experiments/v1/"

EXPERIMENTS = [
    "vanilla", "generalist_8", "simplesam_8", "medsam-self_8", "medsam", "sam-med2d", "sam-med2d-adapter"
    # "generalist_1", "simplesam_1", "medsam-self_1",
]

MODEL = "vit_b"

DATASET_MAPS = {
    "idrid": "IDRiD (Optic Disc in Fundus)",
    "camus": "CAMUS (Cardiac Structures in Echocardipgraphy)",
    "uwaterloo_skin": "UWaterloo Skin (Skin Lesion in Dermoscopy)",
    "montgomery": "Montgomery (Lungs in Chest X-Ray)",
    "sega": "SegA (Aorta in CT)",
    "piccolo": "PICCOLO (Polyps in Narrow Band Imaging)",
    "cbis_ddsm": "CBIS DDSM (Lesion Mass in Mammography)",
    "dca1": "DCA1 (Vessels in X-Ray Coronary Angiograms)",
    "papila": "Papila (Optic Disc & Optic Cup in Fundus)",
    "jnu-ifm": "JNU IFM (Pubic Symphysis & Fetal Head in US)",
    "siim_acr": "SIIM ACR (Pneumothorax in Chest X-Ray)",
    "isic": "ISIC (Skin Lesion in Dermoscopy)",
    "m2caiseg": "m2caiseg (Tools and Organs in Laparoscopy)",
    "btcv": "BTCV (Abdominal Organs in CT)",
}

MODEL_MAPS = {
    "vanilla": "Default",
    "generalist_8": r"$\bf{MedicoSAM}$",
    "simplesam_8": "Simple FT*",
    "medsam-self_8": "MedSAM*",
    "medsam": "MedSAM",
    "sam-med2d": "FT-SAM",
    "sam-med2d-adapter": "SAM-Med2D"
    # "generalist_1": "Generalist (Single GPU)",
    # "simplesam_1": "Simple Generalist* (Single GPU)",
    # "medsam-self_1": "MedSAM* (Single GPU)",
}


def _get_results_per_dataset_per_class(dataset_name, experiment_name):
    res_per_class = []
    for res_dir in glob(os.path.join(ROOT, experiment_name, dataset_name, MODEL, "results", "*")):
        semantic_class = os.path.split(res_dir)[-1]

        ib_results = pd.read_csv(os.path.join(res_dir, "iterative_prompts_start_box.csv"))
        ip_results = pd.read_csv(os.path.join(res_dir, "iterative_prompts_start_point.csv"))

        res = {
            "semantic_class": semantic_class,
            "experiment": experiment_name,
            "dataset": dataset_name,
            "point": ip_results["dice"][0],
            "box": ib_results["dice"][0],
            "ip": ip_results["dice"][7],
            "ib": ib_results["dice"][7],
        }

        res_per_class.append(pd.DataFrame.from_dict([res]))

    _multiclass = True if len(res_per_class) > 1 else False

    if _multiclass:
        res_per_class = pd.concat(res_per_class, ignore_index=True)
        mean_values = res_per_class.mean(numeric_only=True).to_dict()

        mean_df = pd.DataFrame([{
            'semantic_class': "all",
            'experiment':  res_per_class['experiment'].iloc[0],
            'dataset': res_per_class['dataset'].iloc[0],
            **mean_values
        }])

    else:
        mean_df = pd.concat(res_per_class, ignore_index=True)

    return mean_df


def _get_results_per_dataset(dataset_name):
    res_per_dataset = []
    for experiment_name in EXPERIMENTS:
        res_per_dataset.append(_get_results_per_dataset_per_class(dataset_name, experiment_name))

    res_per_dataset = pd.concat(res_per_dataset, ignore_index=True)
    return res_per_dataset


def _make_plots(dataframes, datasets):
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(30, 30))
    axes = axes.flatten()

    bar_width = 0.2
    for i, df in enumerate(dataframes):
        ref = df[df['experiment'] == 'vanilla'].iloc[0]

        df = df[df['experiment'] != 'vanilla']

        df_diff = df.copy()
        df_diff['point_diff'] = df['point'] - ref['point']
        df_diff['box_diff'] = df['box'] - ref['box']
        df_diff['ip_diff'] = df['ip'] - ref['ip']
        df_diff['ib_diff'] = df['ib'] - ref['ib']

        r1 = np.arange(len(df))
        r2 = [x + bar_width for x in r1]
        r3 = [x + 2 * bar_width for x in r1]
        r4 = [x + 3 * bar_width for x in r1]

        axes[i].bar(r1, df_diff['point_diff'], color='#045275', width=bar_width, edgecolor='grey', label='Point')
        axes[i].bar(r2, df_diff['box_diff'], color='#FCDE9C', width=bar_width, edgecolor='grey', label='Box')
        axes[i].bar(r3, df_diff['ip_diff'], color='#7CCBA2', width=bar_width, edgecolor='grey', label=r"I$_{P}$")
        axes[i].bar(r4, df_diff['ib_diff'], color='#90477F', width=bar_width, edgecolor='grey', label=r"I$_{B}$")

        max_val = max(df_diff[['point_diff', 'box_diff', 'ip_diff', 'ib_diff']].values.flatten())
        min_val = min(df_diff[['point_diff', 'box_diff', 'ip_diff', 'ib_diff']].values.flatten())

        axes[i].axhspan(0, max_val, facecolor='lightgreen', alpha=0.2)  # Positive region
        axes[i].axhspan(min_val, 0, facecolor='lightcoral', alpha=0.2)  # Negative region

        _xticklabels = [MODEL_MAPS[_exp] for _exp in df["experiment"]]
        tick_positions = [r + 1.5 * bar_width for r in range(len(df))]
        axes[i].set_xticks(tick_positions)
        axes[i].set_xticklabels(_xticklabels, rotation=45, ha='right', fontsize=16)
        axes[i].tick_params(axis='y', labelsize=14)

        axes[i].set_title(f'{DATASET_MAPS[datasets[i]]}', fontsize=16)
        axes[i].legend()

    all_lines, all_labels = [], []
    for ax in fig.axes:
        lines, labels = ax.get_legend_handles_labels()
        for line, label in zip(lines, labels):
            if label not in all_labels:
                all_lines.append(line)
                all_labels.append(label)
        ax.legend().remove()

    fig.legend(all_lines, all_labels, loc="lower center", ncols=4, bbox_to_anchor=(0.5, 0), fontsize=16)

    plt.text(x=-5.1, y=1.675, s="Dice Score", rotation=90, fontweight="bold", fontsize=16)

    plt.subplots_adjust(top=0.95, bottom=0.075, right=0.95, left=0.05, hspace=0.225, wspace=0.225)
    plt.savefig("./test.png")
    plt.savefig("./test.svg")


def main():
    results = []
    for dataset_name in list(DATASET_MAPS.keys()):
        res = _get_results_per_dataset(dataset_name=dataset_name)
        results.append(res)

    _make_plots(results, list(DATASET_MAPS.keys()))


if __name__ == "__main__":
    main()
