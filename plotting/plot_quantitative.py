import os
from glob import glob

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


ROOT = "/scratch/share/cidas/cca/experiments/v1/"
DATASETS = [
    "idrid", "camus", "uwaterloo_skin", "montgomery", "sega"
]
EXPERIMENTS = [
    "vanilla", "generalist_8", "simplesam_8", "medsam-self_8", "medsam", "sam-med2d", "sam-med2d-adapter"
    # "generalist_1", "simplesam_1", "medsam-self_1",
]
MODEL = "vit_b"


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

    res_per_class = pd.concat(res_per_class, ignore_index=True)
    return res_per_class


def _get_results_per_dataset(dataset_name):
    res_per_dataset = []
    for experiment_name in EXPERIMENTS:
        res_per_dataset.append(_get_results_per_dataset_per_class(dataset_name, experiment_name))

    res_per_dataset = pd.concat(res_per_dataset, ignore_index=True)
    return res_per_dataset


def _make_plots2(dataframes, datasets):
    # Create subplots in a 3x3 grid
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
    axes = axes.flatten()

    # Iterate over each dataframe and plot
    bar_width = 0.2  # Width of the bars
    for i, df in enumerate(dataframes):
        r1 = np.arange(len(df))  # Positions for point bars
        r2 = [x + bar_width for x in r1]  # Positions for box bars
        r3 = [x + 2*bar_width for x in r1]  # Positions for ip bars
        r4 = [x + 3*bar_width for x in r1]  # Positions for ib bars

        # Plot bars for point, box, ip, and ib
        axes[i].bar(r1, df['point'], color='b', width=bar_width, edgecolor='grey', label='point')
        axes[i].bar(r2, df['box'], color='lightblue', width=bar_width, edgecolor='grey', label='box')
        axes[i].bar(r3, df['ip'], color='g', width=bar_width, edgecolor='grey', label='ip')
        axes[i].bar(r4, df['ib'], color='lightgreen', width=bar_width, edgecolor='grey', label='ib')

        # Adjust x-axis ticks and labels
        axes[i].set_xticks(r2)
        axes[i].set_xticklabels(df['experiment'], rotation=45, ha='right')
        axes[i].set_title(f'{datasets[i]}')

        # Add legend
        axes[i].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()
    plt.savefig("./test.png")


def main():
    results = []
    for dataset_name in DATASETS:
        res = _get_results_per_dataset(dataset_name=dataset_name)
        results.append(res)

    _make_plots2(results, DATASETS)
    breakpoint()


main()
